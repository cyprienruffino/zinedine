"""Evolution strategies (OpenAI-ES, Salimans et al. 2017) self-play for the one-button policy.

    python examples/train_es.py --init runs/n3/E_pink_pool/snap_0027000000.npz --out runs/es/es1 --workers 24

Why ES here: in this chaotic game a press pays off (or backfires) seconds later, and PPO's per-step credit
assignment plateaued at the same level whatever we changed. ES scores whole episodes, so the delay between
action and goal doesn't matter, and exploring in weight space gives each candidate a consistent strategy
for a whole episode (actions are argmax: no per-step noise).

Each generation: `--pairs` perturbations eps_i of the weights are tried as theta ± sigma * eps_i
(antithetic). Both signs of a pair play the *same* opponents with the *same* episode seeds (common random
numbers), which cancels much of the game's chaos. Fitness = mean episode score (win 1, loss -1, no goal
-draw_penalty); the update uses centred ranks, Adam and weight decay. Opponents: idle, plus snapshots from
the hall of fame `<out>_pool/` (newest-biased), which gets a new snapshot every --snapshot-every gens.
Snapshots use the SnapshotOpponent/NumpyPolicy format, so examples/elo.py can rate ES and PPO players
together.
"""

import argparse
import csv
import os
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from soccer_rl import SoccerPhysicsEnv
from soccer_rl.selfplay import NumpyPolicy

# --- flat parameter vector <-> NumpyPolicy-style layers ---------------------------------------------


def shapes_of(policy: NumpyPolicy):
    return [w.shape for w, _ in policy.layers] + [policy.out[0].shape]


def flatten(policy: NumpyPolicy) -> np.ndarray:
    parts = []
    for w, b in policy.layers + [policy.out]:
        parts += [w.ravel(), b.ravel()]
    return np.concatenate(parts).astype(np.float32)


def unflatten(theta: np.ndarray, shapes) -> list[tuple[np.ndarray, np.ndarray]]:
    out, i = [], 0
    for rows, cols in shapes:
        w = theta[i:i + rows * cols].reshape(rows, cols)
        i += rows * cols
        b = theta[i:i + rows]
        i += rows
        out.append((w, b))
    return out


def save_snapshot(theta, shapes, path: Path) -> None:
    layers = unflatten(theta, shapes)
    arrays = {}
    for i, (w, b) in enumerate(layers[:-1]):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    arrays["w_out"], arrays["b_out"] = layers[-1]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def init_theta(in_dim, hidden, rng) -> tuple[np.ndarray, list]:
    """Fresh tanh MLP (scaled normal init) with a tiny output layer."""
    dims = [in_dim] + hidden
    parts, shapes = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        parts += [rng.standard_normal((b, a)).ravel() / np.sqrt(a), np.zeros(b)]
        shapes.append((b, a))
    parts += [0.01 * rng.standard_normal((2, dims[-1])).ravel() / np.sqrt(dims[-1]), np.zeros(2)]
    shapes.append((2, dims[-1]))
    return np.concatenate(parts).astype(np.float32), shapes


def act(layers, obs) -> int:
    h = obs
    for w, b in layers[:-1]:
        h = np.tanh(w @ h + b)
    z = layers[-1][0] @ h + layers[-1][1]
    return int(z[1] > z[0])


# --- workers -------------------------------------------------------------------------------------------

_W = {}


def _worker_init(env_kwargs):
    _W["env_kwargs"] = env_kwargs
    _W["env"] = SoccerPhysicsEnv(opponent=lambda o: 0, **env_kwargs)
    _W["opponents"] = {}
    _W["theta_gen"] = None


def _opponent(spec):
    if spec == "idle":
        return lambda o: 0
    if spec not in _W["opponents"]:
        if len(_W["opponents"]) > 64:
            _W["opponents"].clear()
        pol = NumpyPolicy(spec)
        layers = pol.layers + [pol.out]
        _W["opponents"][spec] = lambda o, L=layers: act(L, o)
    return _W["opponents"][spec]


def _evaluate(task):
    """Plays one candidate (theta + sign * sigma * eps(seed)) against its games; returns its fitness.
    A game that fails to start or dies (e.g. many launching at once) is relaunched and the task replayed."""
    for attempt in range(3):
        try:
            return _play(task)
        except (ConnectionError, OSError, EOFError):
            if attempt == 2:
                raise
            try:
                _W["env"].close()
            except Exception:
                pass
            time.sleep(2 + 3 * attempt)
            _W["env"] = SoccerPhysicsEnv(opponent=lambda o: 0, **_W["env_kwargs"])


def _play(task):
    gen, theta_path, shapes, seed, sign, sigma, games, draw_penalty = task
    if _W["theta_gen"] != gen:
        _W["theta"] = np.load(theta_path)
        _W["theta_gen"] = gen
    theta = _W["theta"]
    if seed is not None:
        eps = np.random.default_rng(seed).standard_normal(theta.shape, dtype=np.float32)
        theta = theta + sign * sigma * eps
    layers = unflatten(theta, shapes)
    env = _W["env"]
    total = 0.0
    for side, opp, ep_seed in games:
        env.set_side(side)
        env.opponent = _opponent(opp)
        o, _ = env.reset(seed=ep_seed)
        done = False
        while not done:
            o, _, term, trunc, info = env.step(act(layers, o))
            done = term or trunc
        mine, theirs = info["score"]
        total += 1.0 if mine > theirs else -1.0 if mine < theirs else -draw_penalty
    return total / len(games)


# --- main ----------------------------------------------------------------------------------------------


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="start from this snapshot .npz (e.g. a PPO self-play snapshot)")
    ap.add_argument("--net", default="256,256", help="hidden sizes when starting from scratch")
    ap.add_argument("--out", default="runs/es/es")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=48, help="antithetic pairs per generation")
    ap.add_argument("--games", type=int, default=6, help="episodes per candidate")
    ap.add_argument("--idle-games", type=int, default=2, help="of those, how many against idle")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--draw-penalty", type=float, default=0.3)
    ap.add_argument("--latest-prob", type=float, default=0.5, help="share of snapshot games vs the newest one")
    ap.add_argument("--snapshot-every", type=int, default=10, help="generations between hall-of-fame snapshots")
    ap.add_argument("--eval-every", type=int, default=10, help="generations between 48-game evals of theta vs idle")
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--obs", default="v2", choices=["v1", "v2"])
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    pool = Path(str(out) + "_pool")
    pool.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)
    env_kwargs = dict(obs=args.obs, history=args.history, frame_skip=args.frame_skip, episode="goal",
                      max_steps=args.max_steps)
    probe = SoccerPhysicsEnv(**env_kwargs)  # no game launched: only the observation size
    if args.init:
        policy = NumpyPolicy(args.init)
        theta, shapes = flatten(policy), shapes_of(policy)
        assert shapes[0][1] == probe.obs_dim, f"--init expects {shapes[0][1]} inputs, env gives {probe.obs_dim}"
    else:
        theta, shapes = init_theta(probe.obs_dim, [int(n) for n in args.net.split(",")], rng)
    print(f"{theta.size} parameters, {args.pairs * 2} candidates x {args.games} games per generation")
    save_snapshot(theta, shapes, pool / "gen_000000.npz")
    theta_path = str(out) + "_theta.npy"
    opt = Adam(theta.size, args.lr)

    log_path = Path(str(out) + "_log.csv")
    fields = ["gen", "time", "fit_mean", "fit_max", "fit_spread", "theta_vs_idle", "snapshots", "episodes"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    ctx = get_context("forkserver")
    episodes = 0
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(env_kwargs,)) as workers:
        for gen in range(1, args.generations + 1):
            tmp = theta_path + ".tmp.npy"
            np.save(tmp, theta)
            os.replace(tmp, theta_path)
            snaps = sorted(pool.glob("*.npz"), key=lambda p: p.stat().st_mtime)
            tasks, seeds = [], []
            for _ in range(args.pairs):
                s = int(rng.integers(2 ** 31))
                games = []
                for g in range(args.games):
                    side = ("p1", "p2")[g % 2]
                    if g < args.idle_games:
                        opp = "idle"
                    else:
                        opp = str(snaps[-1] if pyrng.random() < args.latest_prob else pyrng.choice(snaps))
                    games.append((side, opp, int(rng.integers(2 ** 31))))
                seeds.append(s)
                for sign in (1, -1):  # same games for both signs: common random numbers
                    tasks.append((gen, theta_path, shapes, s, sign, args.sigma, games, args.draw_penalty))
            fit = np.array(workers.map(_evaluate, tasks, chunksize=1))
            episodes += len(tasks) * args.games
            ranks = centred_ranks(fit)
            weights = ranks[0::2] - ranks[1::2]  # antithetic: f(+eps) - f(-eps)
            grad = np.zeros_like(theta)
            for w, s in zip(weights, seeds):
                grad += w * np.random.default_rng(s).standard_normal(theta.shape, dtype=np.float32)
            grad = grad / (len(seeds) * args.sigma) - args.weight_decay * theta
            theta = (theta + opt.step(grad)).astype(np.float32)  # gradient ascent on fitness

            vs_idle = ""
            if gen % args.eval_every == 0:
                np.save(tmp, theta)
                os.replace(tmp, theta_path)
                # Same 48 episodes every time (theta is deterministic), so the curve isn't seed noise.
                evals = [(gen + 0.5, theta_path, shapes, None, 0, 0.0,
                          [(("p1", "p2")[g % 2], "idle", 10_000 + 4 * t + g) for g in range(4)], args.draw_penalty)
                         for t in range(12)]
                res = workers.map(_evaluate, evals, chunksize=1)
                vs_idle = f"{np.mean(res):+.3f}"
                episodes += 48
            if gen % args.snapshot_every == 0:
                save_snapshot(theta, shapes, pool / f"gen_{gen:06d}.npz")
            row = [gen, f"{time.time() - t0:.0f}", f"{fit.mean():+.3f}", f"{fit.max():+.3f}",
                   f"{fit.std():.3f}", vs_idle, len(snaps), episodes]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(dict(zip(fields, row)), flush=True)


if __name__ == "__main__":
    main()
