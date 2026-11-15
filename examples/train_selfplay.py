"""Self-play PPO: the agent plays past snapshots of itself (plus a few envs against the built-in AI,
which keep an absolute yardstick in the logs as rollout/win_rate_builtin).

    python examples/train_selfplay.py --init runs/night/r2_fs2.zip --frame-skip 2 --out runs/sp/sp1

Snapshots of the policy are written to <out>_pool/ every --snapshot-every steps; each self-play env
picks a new opponent from that pool at every episode: the newest with --latest-prob, otherwise by
prioritized fictitious self-play (PFSP) weights, so snapshots the agent still loses to come up more.
Self-play envs alternate sides, so the agent learns both P1 and P2 (observations are egocentric).
--shaping-anneal fades the ball-progress shaping linearly to 0, leaving only the goal reward.
"""

import argparse
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from soccer_rl import SoccerPhysicsEnv
from soccer_rl.noise import CorrelatedPolicy
from soccer_rl.selfplay import SnapshotOpponent, export_policy, pfsp_weights, set_press_prior, write_weights


def opponent_kind(i, builtin_envs, idle_envs):
    return "builtin" if i < builtin_envs else "idle" if i < builtin_envs + idle_envs else "self"


def make_env(i, builtin_envs, idle_envs, pool, latest_prob, noise_beta, env_kwargs, seed):
    kind = opponent_kind(i, builtin_envs, idle_envs)
    if kind == "builtin":
        env = SoccerPhysicsEnv(side="p1", opponent="builtin", **env_kwargs)
    elif kind == "idle":  # a standing opponent is a wall: forces the agent to learn to actually score
        env = SoccerPhysicsEnv(side=("p1", "p2")[i % 2], opponent="idle", **env_kwargs)
    else:
        opp = SnapshotOpponent(pool, latest_prob=latest_prob, seed=None if seed is None else seed + i,
                               noise_beta=noise_beta)
        env = SoccerPhysicsEnv(side=("p1", "p2")[i % 2], opponent=opp, **env_kwargs)
    return Monitor(env)


class SelfPlay(BaseCallback):
    """Writes policy snapshots to the pool, publishes PFSP weights, anneals the shaping reward and logs
    win rates per opponent kind."""

    def __init__(self, pool: Path, every: int, builtin_envs: int, idle_envs: int, shaping: float, anneal: int,
                 switch_cost: float = 0.0, switch_anneal: int = 0,
                 window: int = 500, per_snapshot_window: int = 200):
        super().__init__()
        self.pool, self.every, self.builtin_envs, self.idle_envs = pool, every, builtin_envs, idle_envs
        self.shaping, self.anneal = shaping, anneal
        self.switch_cost, self.switch_anneal = switch_cost, switch_anneal
        self.run_len: np.ndarray | None = None  # current press run per env
        self.holds = deque(maxlen=5000)  # finished press lengths (decisions)
        self.results = {k: deque(maxlen=window) for k in ("builtin", "idle", "self")}
        # Recent results only: the agent keeps improving, old games say little about it now.
        self.per_snapshot: dict[str, deque] = {}
        self.per_snapshot_window = per_snapshot_window
        self.last = 0
        self.start = 0
        self.presses = deque(maxlen=20_000)

    def _on_training_start(self) -> None:
        self.last = self.start = self.num_timesteps

    def _on_rollout_start(self) -> None:
        if self.anneal > 0:
            frac = min((self.num_timesteps - self.start) / self.anneal, 1.0)
            shaping = self.shaping * (1.0 - frac)
            self.training_env.env_method("set_shaping", shaping)  # set_attr would stop at the Monitor wrapper
            self.logger.record("train/shaping", shaping)
        if self.switch_cost:
            frac = min((self.num_timesteps - self.start) / self.switch_anneal, 1.0) if self.switch_anneal else 0.0
            cost = self.switch_cost * (1.0 - frac)
            self.training_env.env_method("set_switch_cost", cost)
            self.logger.record("train/switch_cost", cost)
        stats = {stem: [int(sum(r)), len(r)] for stem, r in self.per_snapshot.items()}
        known = {p.stem for p in self.pool.glob("*.npz")}
        stats.update({stem: [0, 0] for stem in known - stats.keys()})
        weights = pfsp_weights(stats)
        write_weights(self.pool, weights)
        if weights:
            self.logger.record("selfplay/snapshots", len(weights))
            hardest = max(weights, key=weights.get)
            w, n = stats[hardest]
            self.logger.record("selfplay/hardest", f"{hardest} ({w}/{n} won)")

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        self.presses.extend(actions.tolist())
        self.logger.record("rollout/press_rate", float(np.mean(self.presses)))
        # Hold lengths: 1-step taps can't trigger the second jump boost (needs 3+ ticks held).
        if self.run_len is None:
            self.run_len = np.zeros(len(actions), dtype=int)
        ended = (actions == 0) & (self.run_len > 0) | np.asarray(self.locals["dones"], bool) & (self.run_len > 0)
        self.holds.extend(self.run_len[ended].tolist())
        self.run_len = np.where(ended, 0, self.run_len) + (actions > 0)
        if self.holds:
            h = np.asarray(self.holds)
            self.logger.record("rollout/tap_share", float((h == 1).mean()))
            self.logger.record("rollout/hold_mean", float(h.mean()))
        if self.num_timesteps - self.last >= self.every:
            export_policy(self.model.policy, self.pool / f"snap_{self.num_timesteps:010d}.npz")
            self.last = self.num_timesteps
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if done:
                mine, theirs = info["score"]
                kind = opponent_kind(i, self.builtin_envs, self.idle_envs)
                self.results[kind].append(np.sign(mine - theirs))
                if kind == "self" and info.get("opponent_id"):
                    self.per_snapshot.setdefault(info["opponent_id"], deque(maxlen=self.per_snapshot_window)) \
                        .append(int(mine > theirs))
        for k, r in self.results.items():
            if r:
                r = np.array(r)
                self.logger.record(f"rollout/win_rate_{k}", float((r > 0).mean()))
                self.logger.record(f"rollout/loss_rate_{k}", float((r < 0).mean()))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="start from this PPO model .zip (recommended: one trained vs the built-in AI)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--builtin-envs", type=int, default=4, help="envs that play the built-in AI instead")
    ap.add_argument("--idle-envs", type=int, default=0, help="envs that play a never-pressing opponent")
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--snapshot-every", type=int, default=500_000)
    ap.add_argument("--latest-prob", type=float, default=0.5)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--shaping", type=float, default=0.05)
    ap.add_argument("--shaping-anneal", type=int, default=10_000_000, help="steps to fade shaping to 0 (0 = keep)")
    ap.add_argument("--obs", default="v1", choices=["v1", "v2"], help="must match --init")
    ap.add_argument("--noise-beta", type=float, default=0.0,
                    help="temporally correlated action sampling (0 = white/plain, 0.5-1 = pinkish, 2 = red)")
    ap.add_argument("--switch-cost", type=float, default=0.0, help="reward cost per new press (anti-mashing)")
    ap.add_argument("--switch-anneal", type=int, default=0, help="steps to fade the switch cost to 0 (0 = keep)")
    ap.add_argument("--draw-penalty", type=float, default=0.0, help="reward cost of a goalless episode")
    ap.add_argument("--time-feature", action="store_true", help="add the episode clock to the observation")
    ap.add_argument("--history", type=int, default=1, help="frames per observation (must match --init)")
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--target-kl", type=float, default=None,
                    help="stop each PPO update early once the policy moved this far (e.g. 0.02)")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--press-prior", type=float, default=None,
                    help="new models: start pressing with this probability (e.g. 0.03) instead of 0.5")
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--net", default="256,256")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="runs/sp/selfplay")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    pool = Path(str(out) + "_pool")
    pool.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(frame_skip=args.frame_skip, episode="goal", max_steps=args.max_steps, shaping=args.shaping,
                      history=args.history, obs=args.obs, draw_penalty=args.draw_penalty, switch_cost=args.switch_cost,
                      time_feature=args.time_feature)
    venv = SubprocVecEnv([partial(make_env, i, args.builtin_envs, args.idle_envs, pool, args.latest_prob,
                                  args.noise_beta, env_kwargs, args.seed)
                          for i in range(args.envs)])

    policy_kwargs = dict(net_arch=[int(n) for n in args.net.split(",")])
    if args.noise_beta:
        policy_kwargs["noise_beta"] = args.noise_beta
    model = PPO(CorrelatedPolicy if args.noise_beta else "MlpPolicy", venv, n_steps=args.n_steps,
                batch_size=args.batch, n_epochs=args.epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
                target_kl=args.target_kl,
                ent_coef=args.ent, learning_rate=args.lr, policy_kwargs=policy_kwargs, seed=args.seed, device="cpu")
    if args.init:
        # Copy the weights into the freshly configured model (so hyperparameters and the sampling scheme
        # can change); the network shapes (--net, --obs, --history, --time-feature) must match.
        init = PPO.load(args.init, device="cpu")
        model.policy.load_state_dict(init.policy.state_dict())
        model.num_timesteps = init.num_timesteps
    elif args.press_prior:
        set_press_prior(model.policy, args.press_prior)
    # The first env reset happens inside learn(), before any callback runs: seed the pool now.
    export_policy(model.policy, pool / f"snap_{model.num_timesteps:010d}.npz")
    model.set_logger(configure_logger(str(out) + "_logs", ["stdout", "csv"]))
    ckpt = CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1), save_path=str(out) + "_ckpt")
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, SelfPlay(pool, args.snapshot_every, args.builtin_envs, args.idle_envs,
                                                                         args.shaping, args.shaping_anneal,
                                                                         args.switch_cost, args.switch_anneal)],
                    reset_num_timesteps=not args.init)  # keep counting from the initial model
    finally:
        model.save(out)
        venv.close()
    print("saved", str(out) + ".zip")


if __name__ == "__main__":
    main()
