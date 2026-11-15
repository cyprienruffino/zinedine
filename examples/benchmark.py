"""Measure simulation throughput, single process and in parallel.

    python examples/benchmark.py --procs 1 2 4
"""

import argparse
import threading
import time

from soccer_rl import GameProcess

REAL_TIME_TICKS_PER_S = 50  # the game runs 50 physics ticks per real second


def run(game, n_steps, repeat, out, i):
    c = game.client
    for k in range(n_steps):
        c.step(buttons=3 * ((k * repeat // 20) % 2), mask=3, repeat=repeat)
    out[i] = n_steps * repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ticks", type=int, default=6000, help="ticks per process")
    ap.add_argument("--repeat", type=int, default=4, help="ticks per step (frame skip)")
    args = ap.parse_args()

    for n in args.procs:
        games = [GameProcess() for _ in range(n)]
        for g in games:
            g.client.reset(seed=0, endless=1)
        out = [0] * n
        threads = [threading.Thread(target=run, args=(g, args.ticks // args.repeat, args.repeat, out, i))
                   for i, g in enumerate(games)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.perf_counter() - t0
        tps = sum(out) / dt
        print(f"{n} process(es): {tps:8.0f} ticks/s  ({tps / REAL_TIME_TICKS_PER_S:5.0f}x real time, "
              f"{tps / args.repeat:6.0f} steps/s at repeat={args.repeat})")
        for g in games:
            g.close()


if __name__ == "__main__":
    main()
