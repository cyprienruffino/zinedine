"""Train PPO (stable-baselines3) against the game's built-in AI.

    uv pip install -e '.[train]'        # stable-baselines3 + torch
    python examples/train_ppo.py --envs 16 --steps 5_000_000
    python examples/watch.py --model runs/ppo_soccer.zip

Each env is its own headless game process. Self-play: once you have a model, pass
opponent=<callable> to SoccerPhysicsEnv (observations are egocentric, so the same network
plays either side).
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import SubprocVecEnv

from soccer_rl import SoccerPhysicsEnv
from soccer_rl.selfplay import set_press_prior


class Anneal(BaseCallback):
    """Linearly fades env reward terms to 0 over `steps` (calls env.set_<name> before each rollout)."""

    def __init__(self, start: dict[str, float], steps: int):
        super().__init__()
        self.start, self.steps, self.t0 = start, steps, 0

    def _on_training_start(self) -> None:
        self.t0 = self.num_timesteps

    def _on_rollout_start(self) -> None:
        frac = 1.0 - min((self.num_timesteps - self.t0) / self.steps, 1.0)
        for name, v in self.start.items():
            self.training_env.env_method(f"set_{name}", v * frac)
            self.logger.record(f"train/{name}", v * frac)

    def _on_step(self) -> bool:
        return True


class WinRate(BaseCallback):
    """Logs the share of finished episodes the agent won / lost (by goals, ignoring shaping) and how
    often it presses (sampled actions)."""

    def __init__(self, window: int = 500):
        super().__init__()
        self.results = deque(maxlen=window)
        self.presses = deque(maxlen=20_000)

    def _on_step(self) -> bool:
        self.presses.extend(np.asarray(self.locals["actions"]).reshape(-1).tolist())
        self.logger.record("rollout/press_rate", float(np.mean(self.presses)))
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                mine, theirs = info["score"]
                self.results.append(np.sign(mine - theirs))
        if self.results:
            r = np.array(self.results)
            self.logger.record("rollout/win_rate", float((r > 0).mean()))
            self.logger.record("rollout/loss_rate", float((r < 0).mean()))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--opponent", default="builtin", choices=["builtin", "idle", "random"])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--shaping", type=float, default=0.05, help="dense reward for pushing the ball forward")
    ap.add_argument("--obs", default="v1", choices=["v1", "v2"], help="v2 = + jump timers, normalized")
    ap.add_argument("--history", type=int, default=1, help="frames per observation (short-term memory)")
    ap.add_argument("--draw-penalty", type=float, default=0.0, help="reward cost of a goalless episode")
    ap.add_argument("--press-cost", type=float, default=0.0, help="reward cost per step holding the button")
    ap.add_argument("--anneal", type=int, default=0, help="steps to fade shaping and press cost to 0 (0 = keep)")
    ap.add_argument("--modifiers", default="none", choices=["none", "random"])
    ap.add_argument("--max-steps", type=int, default=1000, help="episode truncation, in env steps")
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--target-kl", type=float, default=None,
                    help="stop each PPO update early once the policy moved this far (e.g. 0.02)")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--press-prior", type=float, default=None,
                    help="new models: start pressing with this probability (e.g. 0.03) instead of 0.5")
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--net", default="256,256")
    ap.add_argument("--threads", type=int, default=2, help="torch threads (the games need the other cores)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", help="continue from a saved model .zip")
    ap.add_argument("--out", default="runs/ppo_soccer")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    env_kwargs = dict(
        opponent=args.opponent,
        frame_skip=args.frame_skip,
        episode="goal",
        max_steps=args.max_steps,
        shaping=args.shaping,
        press_cost=args.press_cost,
        draw_penalty=args.draw_penalty,
        modifiers=args.modifiers,
        history=args.history,
        obs=args.obs,
    )
    venv = make_vec_env(SoccerPhysicsEnv, n_envs=args.envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv,
                        seed=args.seed)

    out = Path(args.out)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=args.n_steps,
            batch_size=args.batch,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda, target_kl=args.target_kl,
            ent_coef=args.ent,
            learning_rate=args.lr,
            policy_kwargs=dict(net_arch=[int(n) for n in args.net.split(",")]),
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
        if args.press_prior:
            set_press_prior(model.policy, args.press_prior)
    model.set_logger(configure_logger(str(out) + "_logs", ["stdout", "csv"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointCallback(save_freq=max(250_000 // args.envs, 1), save_path=str(out) + "_ckpt")
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, WinRate()] + ([Anneal(dict(shaping=args.shaping, press_cost=args.press_cost), args.anneal)]
                                                  if args.anneal else []), reset_num_timesteps=not args.resume)
    finally:
        model.save(out)
        venv.close()
    print("saved", str(out) + ".zip")


if __name__ == "__main__":
    main()
