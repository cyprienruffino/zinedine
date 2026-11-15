"""Prints Sample Factory episode statistics per policy (from its tensorboard summaries): win/draw rate,
press rate, tap share (one-decision presses), mean hold length, switch cost, over the run.

    .venv-sf/bin/python scripts/sf_stats.py runs/sf/sf1 [--points 6]
"""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = ("win", "draw", "press_rate", "tap_share", "hold_mean", "switch_cost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--points", type=int, default=6)
    args = ap.parse_args()
    for pdir in sorted((Path(args.experiment) / ".summary").glob("*")):
        acc = EventAccumulator(str(pdir), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        print(f"policy {pdir.name}")
        for key in KEYS:
            tag = next((t for t in tags if t.endswith(f"/{key}") and "mean" in t.lower()), None) or \
                  next((t for t in tags if t.endswith(key)), None)
            if not tag:
                continue
            ev = sorted(acc.Scalars(tag), key=lambda e: e.wall_time)  # several sessions: order by time
            idx = [round(i * (len(ev) - 1) / max(args.points - 1, 1)) for i in range(args.points)] if ev else []
            t0 = ev[0].wall_time if ev else 0
            print(f"  {key:<12}" + "  ".join(f"{(ev[i].wall_time - t0) / 3600:.0f}h:{ev[i].value:.2f}"
                                              for i in sorted(set(idx))))


if __name__ == "__main__":
    main()
