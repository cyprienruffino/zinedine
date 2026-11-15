"""Warm-starts a Sample Factory experiment whose observation gained new features (e.g. the point clock,
obs 4 x 195 -> 4 x 196): copies the latest checkpoint of each policy and inserts zero-weighted input
columns for the new features, so the network behaves exactly as before until it learns to use them.

    .venv-sf/bin/python scripts/expand_sf_obs.py runs/sf/sf2 runs/sf/sf3 --frame-in 195 --frame-out 196 --history 4

New features are appended at the end of each stacked frame (as SoccerPhysicsEnv does). The first encoder
layer and its Adam moments (exp_avg, exp_avg_sq) are expanded; everything else is copied unchanged, so
Sample Factory resumes training (counters included) instead of failing to load and starting from scratch.
"""

import argparse
from pathlib import Path

import torch

LAYER = "encoder.encoders.obs.mlp_head.0.weight"


def expand(w: torch.Tensor, frame_in: int, frame_out: int, history: int) -> torch.Tensor:
    assert w.shape[1] == frame_in * history, f"expected {frame_in * history} inputs, got {w.shape[1]}"
    out = torch.zeros(w.shape[0], frame_out * history, dtype=w.dtype)
    for k in range(history):
        out[:, k * frame_out:k * frame_out + frame_in] = w[:, k * frame_in:(k + 1) * frame_in]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="source experiment directory")
    ap.add_argument("dst", help="new experiment directory (must not contain checkpoints yet)")
    ap.add_argument("--frame-in", type=int, default=195)
    ap.add_argument("--frame-out", type=int, default=196)
    ap.add_argument("--history", type=int, default=4)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    n_in = args.frame_in * args.history
    for pdir in sorted(src.glob("checkpoint_p*")):
        ckpt = sorted(pdir.glob("checkpoint_*.pth"))[-1]
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        d["model"][LAYER] = expand(d["model"][LAYER], args.frame_in, args.frame_out, args.history)
        n_expanded = 0
        for state in d["optimizer"]["state"].values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n_in:
                    state[k] = expand(v, args.frame_in, args.frame_out, args.history)
                    n_expanded += 1
        assert n_expanded == 2, f"expected Adam exp_avg and exp_avg_sq for the input layer, expanded {n_expanded}"
        out_dir = dst / pdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        assert not list(out_dir.glob("checkpoint_*.pth")), f"{out_dir} already has checkpoints"
        torch.save(d, out_dir / ckpt.name)
        print(f"{ckpt} -> {out_dir / ckpt.name} ({tuple(d['model'][LAYER].shape)})")


if __name__ == "__main__":
    main()
