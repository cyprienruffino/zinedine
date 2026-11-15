"""Elo ratings for self-play snapshots: round-robin games between snapshots of a pool (plus fixed
reference players), then a Bradley-Terry fit of all results (draws count as half a win each).

    python examples/elo.py runs/night/sp3_from_r10_pool --obs v2 --history 4 --max-players 12 \\
        --games 20 --procs 8

Ratings are on the usual Elo scale (400 points = 10:1 odds), anchored so that `idle` = 1000. The env
settings (--obs, --history, --time-feature, --frame-skip) must match how the snapshots were trained.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from soccer_rl import SoccerPhysicsEnv
from soccer_rl.selfplay import NumpyPolicy

REFERENCE_PLAYERS = ("idle", "random", "pulse")  # pulse: hold 6 steps, release 6 (full jumps, no mashing)


def make_player(name, snapshots, rng, argmax=(), clock_frames=0, frame_dim=0):
    """name → callable obs -> action: sampled like PPO's training, or argmax for players from pools in
    `argmax` (e.g. ES snapshots, which train deterministic)."""
    if name == "idle":
        return lambda o: 0
    if name == "random":
        return lambda o: int(rng.random() < 0.3)
    if name == "pulse":
        t = [0]

        def pulse(o):
            t[0] += 1
            return int((t[0] // 6) % 2 == 0)

        return pulse
    policy = NumpyPolicy(snapshots[name])
    logits = policy.logits
    n_in = policy.layers[0][0].shape[1]
    if clock_frames and n_in < clock_frames * frame_dim:
        # A player trained on shorter frames (no clock, no modifiers): keep the first features of each frame.
        f_old = n_in // clock_frames
        keep = np.array([i for i in range(clock_frames * frame_dim) if i % frame_dim < f_old])
        logits = lambda o: policy.logits(o[keep])  # noqa: E731
    if name.split("/")[0] in argmax:
        def act(o):
            return int(np.argmax(logits(o)))
    else:
        def act(o):
            z = logits(o)
            p = np.exp(z - z.max())
            return int(rng.random() < p[1] / p.sum())
    act.reset = policy.reset_state  # recurrent policies: fresh hidden state every episode (the env calls
    return act                      # opponent.reset(); the agent's is called in play())


def play(args, snapshots, matches):
    """matches: list of (a, b, side, seed); returns (a, b, score of a in {0, 0.5, 1})."""
    rng = np.random.default_rng(matches[0][3] if matches else 0)
    players = {}
    frame_dim = 195 + int(args.time_feature) + 7 * int(args.modifier_feature)
    clock_frames = args.history if (args.time_feature or args.modifier_feature) else 0
    get = lambda n: players.setdefault(  # noqa: E731
        n, make_player(n, snapshots, rng, args.argmax_pool_names, clock_frames, frame_dim))
    envs, out = {}, []
    try:
        for a, b, side, seed in matches:
            if side not in envs:
                envs[side] = SoccerPhysicsEnv(side=side, opponent=lambda o: 0, obs=args.obs, history=args.history,
                                              time_feature=args.time_feature, frame_skip=args.frame_skip,
                                              modifiers=args.modifiers, modifier_feature=args.modifier_feature,
                                              episode="goal", max_steps=args.max_steps)
            env = envs[side]
            # The game's built-in AI is a game setting, not a callable: it only drives P2 (matches put it there).
            env.opponent = "builtin" if b == "builtin" else get(b)  # the env gives b its own egocentric view
            o, _ = env.reset(seed=seed)
            done = False
            act = get(a)
            getattr(act, "reset", lambda: None)()
            while not done:
                o, _, term, trunc, info = env.step(act(o))
                done = term or trunc
            mine, theirs = info["score"]
            out.append((a, b, 1.0 if mine > theirs else 0.0 if mine < theirs else 0.5))
    finally:
        for env in envs.values():
            env.close()
    return out


def bradley_terry(names, results, iters=2000):
    """Maximum-likelihood ratings from (a, b, score) results (MM algorithm, draws as half wins)."""
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k))
    for a, b, s in results:
        wins[idx[a], idx[b]] += s
        wins[idx[b], idx[a]] += 1 - s
    games = wins + wins.T
    gamma = np.ones(k)
    w = wins.sum(1) + 0.5  # small prior keeps unbeaten/winless players finite
    for _ in range(iters):
        denom = (games / (gamma[:, None] + gamma[None, :])).sum(1) + 1.0 / (gamma + 1.0)
        gamma = w / denom
        gamma /= np.exp(np.log(gamma).mean())
    return 400 * np.log10(gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+", help="snapshot pool directories (*.npz)")
    ap.add_argument("--max-players", type=int, default=12, help="snapshots per pool, evenly spaced, newest included")
    ap.add_argument("--games", type=int, default=20, help="games per pair (half on each side)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--obs", default="v2", choices=["v1", "v2"])
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--time-feature", action="store_true")
    ap.add_argument("--modifier-feature", action="store_true", help="observation includes the match modifiers")
    ap.add_argument("--modifiers", default="none", choices=["none", "random"], help="classic games, or random variants")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--argmax-pools", nargs="*", default=[], help="pools whose players act by argmax (ES)")
    ap.add_argument("--builtin", action="store_true", help="also rate the game's built-in AI (it always plays P2)")
    args = ap.parse_args()
    args.argmax_pool_names = {Path(p).name.removesuffix("_pool") for p in args.argmax_pools}

    snapshots = {}
    for pool in args.pools:
        snaps = sorted(Path(pool).glob("*.npz"), key=lambda p: p.stat().st_mtime)
        keep = sorted({round(i) for i in np.linspace(0, len(snaps) - 1, min(args.max_players, len(snaps)))})
        for i in keep:
            snapshots[f"{Path(pool).name.removesuffix('_pool')}/{snaps[i].stem}"] = snaps[i]
    names = list(REFERENCE_PLAYERS) + (["builtin"] if args.builtin else []) + list(snapshots)

    matches = []
    seed = args.seed
    for a, b in itertools.combinations(names, 2):
        for g in range(args.games):
            if "builtin" in (a, b):  # the built-in AI only plays P2: the other player always takes P1
                matches.append((b if a == "builtin" else a, "builtin", "p1", seed))
            else:
                matches.append((a, b, ("p1", "p2")[g % 2], seed))
            seed += 1
    rng = np.random.default_rng(args.seed)
    rng.shuffle(matches)
    chunks = [matches[i::args.procs] for i in range(args.procs)]
    with ProcessPoolExecutor(args.procs) as ex:
        results = [r for part in ex.map(play, [args] * args.procs, [snapshots] * args.procs, chunks) for r in part]

    elo = bradley_terry(names, results)
    elo += 1000 - elo[names.index("idle")]
    draws = np.mean([s == 0.5 for _, _, s in results])
    print(f"{len(results)} games, {len(names)} players, {100 * draws:.0f}% draws")
    for n, e in sorted(zip(names, elo), key=lambda t: -t[1]):
        print(f"{e:7.0f}  {n}")


if __name__ == "__main__":
    main()
