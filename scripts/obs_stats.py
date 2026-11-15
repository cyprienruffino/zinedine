"""Measures per-feature mean/std of the raw v2 observation frame on real games and writes
soccer_rl/obs_stats_v2.npz, which SoccerPhysicsEnv(obs="v2") uses to normalize its inputs.

    python scripts/obs_stats.py --steps 40000 --procs 4

Frames come from a random policy (presses ~30% of steps) against the built-in AI, from both teams'
egocentric views. Binary flags keep mean 0 / std 1 so they stay exactly 0/1.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from soccer_rl import SoccerPhysicsEnv
from soccer_rl.env import BODY_FEATURES, OBS_STATS_PATH, PARTS, PLAYER_FLAGS, PLAYER_TIMERS


def collect(steps, seed):
    env = SoccerPhysicsEnv(obs="v2", frame_skip=2, max_steps=2000)
    env._norm = None  # raw features
    rng = np.random.default_rng(seed)
    frames = []
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(int(rng.random() < 0.3))
            frames += [env._obs(env.state, False), env._obs(env.state, True)]
            if term or trunc:
                env.reset()
    finally:
        env.close()
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40_000, help="env steps in total")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.procs) as ex:
        x = np.concatenate(list(ex.map(collect, [args.steps // args.procs] * args.procs, range(args.procs))))
    mean, std = x.mean(0), x.std(0)
    std[std < 0.02] = 1.0  # constant features (the ball never rotates): don't amplify float noise
    n_body = (4 * len(PARTS) + 1) * BODY_FEATURES
    per_player = PLAYER_FLAGS + PLAYER_TIMERS
    for i in range(4):
        flags = slice(n_body + i * per_player, n_body + i * per_player + PLAYER_FLAGS)
        mean[flags], std[flags] = 0.0, 1.0
    np.savez(OBS_STATS_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"{len(x)} frames → {OBS_STATS_PATH}")
    print("std range of body features:", std[:n_body].min(), std[:n_body].max())


if __name__ == "__main__":
    main()
