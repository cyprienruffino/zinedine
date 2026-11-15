"""Exports Sample Factory checkpoints as numpy snapshots (the SnapshotOpponent / NumpyPolicy format), so
examples/elo.py, examples/watch.py and SB3 self-play can use Sample Factory players.

    .venv-sf/bin/python scripts/export_sf.py runs/sf/sf1                # latest checkpoint of each policy
    .venv-sf/bin/python scripts/export_sf.py runs/sf/sf1 --every 1800  # keep exporting during a long run

Writes runs/sf/<experiment>/pool_p<k>/steps_<env_steps>.npz. Supports the models used by
examples/train_sf.py (no input normalization): encoder Linear+tanh layers, an optional one-layer GRU core
(--use_rnn True --rnn_type gru), then the action head.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch


def export(ckpt: Path, out: Path) -> None:
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in d["model"].items()}
    assert not any(k.startswith("core.") and "lstm" in k.lower() for k in sd), "LSTM cores are not supported (use GRU)"
    assert not any("obs_normalizer" in k for k in sd), "input normalization is not supported"
    prefix = "encoder.encoders.obs.mlp_head."
    layers = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})
    arrays = {}
    for i, idx in enumerate(layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = sd[f"{prefix}{idx}.weight"], sd[f"{prefix}{idx}.bias"]
    if "core.core.weight_ih_l0" in sd:  # GRU core (--use_rnn True --rnn_type gru, one layer)
        assert "core.core.weight_ih_l1" not in sd, "multi-layer GRUs are not supported"
        for ours, theirs in (("gru_wih", "weight_ih_l0"), ("gru_whh", "weight_hh_l0"),
                             ("gru_bih", "bias_ih_l0"), ("gru_bhh", "bias_hh_l0")):
            arrays[ours] = sd[f"core.core.{theirs}"]
    arrays["w_out"] = sd["action_parameterization.distribution_linear.weight"]
    arrays["b_out"] = sd["action_parameterization.distribution_linear.bias"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)


def export_latest(exp: Path) -> list[Path]:
    written = []
    for pdir in sorted(exp.glob("checkpoint_p*")):
        ckpts = sorted((c for c in pdir.glob("checkpoint_*.pth")), key=lambda c: c.stat().st_mtime)
        if not ckpts:
            continue
        ck = ckpts[-1]
        env_steps = int(ck.stem.split("_")[-1])
        out = exp / f"pool_{pdir.name.removeprefix('checkpoint_')}" / f"steps_{env_steps:012d}.npz"
        if not out.exists():
            export(ck, out)
            written.append(out)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="Sample Factory experiment directory, e.g. runs/sf/sf1")
    ap.add_argument("--every", type=float, default=0, help="seconds between exports (0 = once)")
    args = ap.parse_args()
    exp = Path(args.experiment)
    while True:
        for p in export_latest(exp):
            print("exported", p, flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
