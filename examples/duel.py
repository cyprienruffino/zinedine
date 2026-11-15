"""Head-to-head between two PPO models over many single-goal episodes, alternating sides.

    python examples/duel.py runs/night/sp1.zip runs/night/sp2_v2h4.zip:v2:4 --episodes 1000 --procs 8

A model is given as path[:obs[:history]] (defaults v1, 1), matching how it was trained. Each model
builds its own observation from the shared game state, so models with different inputs can play.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from soccer_rl import SoccerPhysicsEnv


def parse(spec):
    parts = spec.split(":")
    return parts[0], parts[1] if len(parts) > 1 else "v1", int(parts[2]) if len(parts) > 2 else 1


class ModelOpponent:
    """Plays model B for the env that runs model A: rebuilds B's own view from env.state each step."""

    def __init__(self, model, view: SoccerPhysicsEnv, deterministic: bool):
        self.model, self.view, self.det = model, view, deterministic
        self.env = None  # set once the env exists

    def reset(self):
        self.view._frames.clear()

    def __call__(self, _obs):
        o = self.view._view(self.env.state, not self.env.mirror)
        return int(self.model.predict(o, deterministic=self.det)[0])


def run(args, episodes):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    (pa, oa, ha), (pb, ob, hb) = parse(args.a), parse(args.b)
    a, b = PPO.load(pa, device="cpu"), PPO.load(pb, device="cpu")
    results = []
    envs = {}
    try:
        for ep in episodes:
            side = ("p1", "p2")[ep % 2]
            if side not in envs:
                opp = ModelOpponent(b, SoccerPhysicsEnv(obs=ob, history=hb, frame_skip=args.frame_skip),
                                    args.deterministic)
                env = SoccerPhysicsEnv(side=side, opponent=opp, obs=oa, history=ha, frame_skip=args.frame_skip,
                                       episode="goal", max_steps=args.max_steps)
                opp.env = env
                envs[side] = env
            env = envs[side]
            o, _ = env.reset(seed=args.seed + ep)
            done = False
            while not done:
                o, _, term, trunc, info = env.step(int(a.predict(o, deterministic=args.deterministic)[0]))
                done = term or trunc
            mine, theirs = info["score"]
            results.append(mine - theirs)
    finally:
        for env in envs.values():
            env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    chunks = [list(range(args.episodes))[i::args.procs] for i in range(args.procs)]
    with ProcessPoolExecutor(args.procs) as ex:
        r = np.concatenate([np.array(x) for x in ex.map(run, [args] * args.procs, chunks)])
    n = len(r)
    wa, wb = int((r > 0).sum()), int((r < 0).sum())
    print(f"A={args.a} vs B={args.b}: A won {wa}, B won {wb}, no goal {n - wa - wb} "
          f"(A share of decided games {100 * wa / max(wa + wb, 1):.1f}%)")


if __name__ == "__main__":
    main()
