"""Score a policy against the built-in AI (or --opponent idle/random) over many single-goal episodes.

    python examples/evaluate.py --model runs/ppo_soccer.zip --episodes 1000 --procs 8
    python examples/evaluate.py --policy random
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from soccer_rl import SoccerPhysicsEnv


def make_policy(args, worker):
    if args.model:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        model = PPO.load(args.model, device="cpu")
        return lambda o, t: int(model.predict(o, deterministic=not args.stochastic)[0])
    rng = np.random.default_rng(args.seed + 7919 * worker)
    return {
        "random": lambda o, t: int(rng.random() < 0.3),
        "pulse": lambda o, t: int((t // 6) % 2 == 0),
        "idle": lambda o, t: 0,
    }[args.policy]


def run(args, worker, episodes):
    """Plays `episodes` (a list of episode indices); returns the goal difference of each."""
    act = make_policy(args, worker)
    env = SoccerPhysicsEnv(opponent=args.opponent, frame_skip=args.frame_skip, episode="goal", max_steps=args.max_steps,
                           obs=args.obs, history=args.history)
    results = []
    try:
        for ep in episodes:
            o, _ = env.reset(seed=args.seed + ep)
            t, done = 0, False
            while not done:
                o, _, term, trunc, info = env.step(act(o, t))
                t += 1
                done = term or trunc
            mine, theirs = info["score"]
            results.append(mine - theirs)
    finally:
        env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--policy", choices=["random", "pulse", "idle"], default="random")
    ap.add_argument("--stochastic", action="store_true", help="sample the model's actions instead of argmax")
    ap.add_argument("--opponent", default="builtin", choices=["builtin", "idle", "random"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--obs", default="v1", choices=["v1", "v2"], help="must match the model")
    ap.add_argument("--history", type=int, default=1, help="must match the model")
    args = ap.parse_args()

    chunks = [list(range(args.episodes))[i::args.procs] for i in range(args.procs)]
    with ProcessPoolExecutor(args.procs) as ex:
        r = np.concatenate([np.array(x) for x in ex.map(run, [args] * args.procs, range(args.procs), chunks)])
    n = len(r)
    wins, losses = int((r > 0).sum()), int((r < 0).sum())
    name = (args.model + (" (stochastic)" if args.stochastic else "")) if args.model else args.policy
    print(f"{name} vs {args.opponent}: won {wins}/{n} ({100 * wins / n:.1f}% ± {100 * np.sqrt(wins / n * (1 - wins / n) / n):.1f}), "
          f"lost {losses}, no goal {n - wins - losses}; goal difference per episode {r.mean():+.3f}")


if __name__ == "__main__":
    main()
