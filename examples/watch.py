"""Watch a policy play in a real-time window, or play against it yourself.

    python examples/watch.py                         # scripted "pulse" policy vs the built-in AI
    python examples/watch.py --policy random
    python examples/watch.py --model runs/ppo.zip    # a trained stable-baselines3 model
    python examples/watch.py --model runs/ppo.zip --human   # you play P1 (Up arrow) against it
    python examples/watch.py --snapshot runs/final2/es_pool/es4_bigpop.npz --argmax --human   # a self-play
        snapshot (obs v2, 4-frame history, frame_skip 2 by default, as trained); --argmax for ES snapshots
    python examples/watch.py --snapshot runs/sf/latest/p0.npz --opponent-snapshot runs/sf/latest/p2.npz  # bot vs bot
    python examples/watch.py --speed 4               # 4x real time
"""

import argparse

import numpy as np

from soccer_rl import SoccerPhysicsEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["pulse", "random", "idle"], default="pulse")
    ap.add_argument("--model", help="stable-baselines3 PPO .zip to load instead of --policy")
    ap.add_argument("--snapshot", help="self-play snapshot .npz (PPO pool or ES) to load instead of --policy")
    ap.add_argument("--argmax", action="store_true", help="snapshot acts by argmax (ES) instead of sampling (PPO)")
    ap.add_argument("--obs", default=None, choices=["v1", "v2"], help="default: v2 with --snapshot, else v1")
    ap.add_argument("--history", type=int, default=None, help="default: 4 with --snapshot, else 1")
    ap.add_argument("--human", action="store_true", help="you control P1 with the Up arrow; the agent plays P2")
    ap.add_argument("--opponent-snapshot", help="another snapshot .npz plays P2 against --snapshot (bot vs bot)")
    ap.add_argument("--opponent-argmax", action="store_true", help="the opponent snapshot acts by argmax")
    ap.add_argument("--episodes", type=int, default=0, help="0 = forever")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--frame-skip", type=int, default=None, help="default: 2 with --snapshot, else 4")
    args = ap.parse_args()
    snap = bool(args.snapshot)
    obs_kind = args.obs or ("v2" if snap else "v1")
    history = args.history or (4 if snap else 1)
    frame_skip = args.frame_skip or (2 if snap else 4)

    opponent = "keyboard" if args.human else "builtin"
    if args.opponent_snapshot:
        from soccer_rl.selfplay import NumpyPolicy

        opp_policy = NumpyPolicy(args.opponent_snapshot)

        def opponent(obs):  # the env passes the opponent its own (mirrored) view
            z = opp_policy.logits(obs)
            if args.opponent_argmax:
                return int(z[1] > z[0])
            p = np.exp(z - z.max())
            return int(np.random.rand() < p[1] / p.sum())

        opponent.reset = opp_policy.reset_state

    env = SoccerPhysicsEnv(
        side="p2" if args.human else "p1",
        opponent=opponent,
        frame_skip=frame_skip,
        obs=obs_kind,
        history=history,
        episode="match",
        max_steps=None,
        render_mode="human",
        game_kwargs=dict(fps=int(50 * args.speed), width=1280, height=720),
    )

    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        act = lambda obs, t: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    elif args.snapshot:
        from soccer_rl.selfplay import NumpyPolicy

        policy = NumpyPolicy(args.snapshot)

        def act(obs, t):
            z = policy.logits(obs)
            if args.argmax:
                return int(z[1] > z[0])
            p = np.exp(z - z.max())
            return int(np.random.rand() < p[1] / p.sum())
    elif args.policy == "pulse":
        act = lambda obs, t: int((t // 6) % 2 == 0)  # noqa: E731  kick/jump rhythmically
    elif args.policy == "random":
        act = lambda obs, t: int(np.random.rand() < 0.3)  # noqa: E731
    else:
        act = lambda obs, t: 0  # noqa: E731

    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            obs, info = env.reset()
            if args.snapshot:
                policy.reset_state()  # recurrent snapshots start every match with a fresh memory
            t, done = 0, False
            while not done:
                obs, r, term, trunc, info = env.step(act(obs, t))
                t += 1
                if r:
                    print(f"{'GOAL!' if r > 0 else 'conceded'}  score (agent-opponent) {info['score']}")
                done = term or trunc
            ep += 1
            print(f"match {ep} over: {info['score']}")
    except ConnectionError:
        print("game closed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
