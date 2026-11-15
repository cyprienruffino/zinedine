"""Self-play helpers: export an SB3 MlpPolicy as a numpy snapshot, and an opponent that plays
snapshots drawn from a pool directory.

The opponent runs in plain numpy so each SubprocVecEnv worker can evaluate it cheaply without
torch threads fighting the game processes for cores.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from .noise import CorrelatedUniform


def export_policy(policy, path: str | os.PathLike) -> None:
    """Saves the actor half of an SB3 ActorCriticPolicy (Linear+Tanh MLP, then action_net) atomically."""
    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    layers = sorted({int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.policy_net.")})
    arrays = {}
    for i, idx in enumerate(layers):
        arrays[f"w{i}"] = sd[f"mlp_extractor.policy_net.{idx}.weight"]
        arrays[f"b{i}"] = sd[f"mlp_extractor.policy_net.{idx}.bias"]
    arrays["w_out"] = sd["action_net.weight"]
    arrays["b_out"] = sd["action_net.bias"]
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)  # workers glob the pool; never let them see a half-written file


class NumpyPolicy:
    """Actor of a snapshot: tanh MLP layers (w0, b0, w1, b1, ...), optionally a GRU (gru_wih, gru_whh,
    gru_bih, gru_bhh; PyTorch gate order r, z, n), then the action head (w_out, b_out). A recurrent policy
    keeps a hidden state between calls: call reset_state() at the start of every episode."""

    def __init__(self, path: str | os.PathLike):
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith("w") and k[1:].isdigit())
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
        self.out = (z["w_out"], z["b_out"])
        self.gru = (z["gru_wih"], z["gru_whh"], z["gru_bih"], z["gru_bhh"]) if "gru_wih" in z.files else None
        self.h = None
        self.reset_state()

    @property
    def recurrent(self) -> bool:
        return self.gru is not None

    def reset_state(self) -> None:
        if self.gru is not None:
            self.h = np.zeros(self.gru[1].shape[1], dtype=np.float32)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        if self.gru is not None:
            wih, whh, bih, bhh = self.gru
            gi, gh = wih @ x + bih, whh @ self.h + bhh
            k = len(self.h)
            r = 1 / (1 + np.exp(-(gi[:k] + gh[:k])))
            u = 1 / (1 + np.exp(-(gi[k:2 * k] + gh[k:2 * k])))
            n = np.tanh(gi[2 * k:] + r * gh[2 * k:])
            self.h = ((1 - u) * n + u * self.h).astype(np.float32)
            x = self.h
        return self.out[0] @ x + self.out[1]


def set_press_prior(policy, p: float) -> None:
    """Biases a freshly initialized Discrete(2) policy so it presses with probability ~p.

    Standing still is a strong default in this game (an idle player stays upright and blocks, and a
    single press topples it for seconds), so exploration should start near "rarely press" rather than
    at the uniform 50%. SB3 initializes action_net with gain 0.01, so the bias dominates the logits."""
    import torch

    with torch.no_grad():
        policy.action_net.bias.copy_(torch.tensor([0.0, float(np.log(p / (1 - p)))]))


WEIGHTS_FILE = "weights.json"


def write_weights(pool_dir: str | os.PathLike, weights: dict[str, float]) -> None:
    """Publishes sampling weights (snapshot stem → weight) for SnapshotOpponent, atomically."""
    path = Path(pool_dir) / WEIGHTS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(weights))
    os.replace(tmp, path)


def pfsp_weights(stats: dict[str, list[int]], power: float = 2.0, floor: float = 0.05,
                 prior: tuple[int, int] = (1, 1)) -> dict[str, float]:
    """Prioritized fictitious self-play: weight each snapshot by (1 - agent win rate against it)^power,
    so opponents the agent still struggles with get played more. stats: stem → [wins, games];
    `prior` smooths win rates of rarely-played snapshots, `floor` keeps every snapshot in rotation."""
    out = {}
    for stem, (wins, games) in stats.items():
        p = (wins + prior[0]) / (games + prior[0] + prior[1])
        out[stem] = max((1.0 - p) ** power, floor)
    return out


class SnapshotOpponent:
    """Opponent for SoccerPhysicsEnv(opponent=...): on every episode reset it picks a snapshot from
    `pool_dir` (the newest with probability `latest_prob`, otherwise one drawn by the weights in
    `pool_dir/weights.json` if present, else uniformly) and samples its actions from the policy
    distribution (temporally correlated when noise_beta > 0, see soccer_rl.noise). With history > 1 the env passes it stacked frames, like the agent."""

    def __init__(self, pool_dir: str | os.PathLike | list, latest_prob: float = 0.5, seed: int | None = None,
                 noise_beta: float = 0.0, recent_frac: float = 1.0, history: int = 4):
        """pool_dir: one directory or a list of them (snapshots are pooled, ordered by mtime).
        recent_frac: draw only from the newest share of snapshots (0.5 = newest half, Bansal et al.'s
        delta = 0.5); latest_prob still picks the very newest first."""
        dirs = pool_dir if isinstance(pool_dir, (list, tuple)) else [pool_dir]
        self.pool_dirs = [Path(d) for d in dirs]
        self.pool_dir = self.pool_dirs[0]  # holds weights.json (PFSP) if any
        self.recent_frac = recent_frac
        self.history = history  # to trim per-frame features an older snapshot doesn't know (e.g. modifiers)
        self.latest_prob = latest_prob
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.noise = CorrelatedUniform(noise_beta, seed=seed) if noise_beta else None  # like the agent's sampling
        self._cache: dict[str, NumpyPolicy] = {}
        self.policy: NumpyPolicy | None = None
        self.current: str | None = None
        self._weights_cache: dict[str, float] | None = None
        self._weights_mtime: float | None = None

    def reset(self) -> None:
        snaps = sorted((p for d in self.pool_dirs for p in d.glob("*.npz")), key=lambda p: p.stat().st_mtime)
        if self.recent_frac < 1.0 and len(snaps) > 1:
            snaps = snaps[-max(1, round(len(snaps) * self.recent_frac)):]
        if not snaps:
            raise FileNotFoundError(f"no snapshots in {self.pool_dir}")
        if self.rng.random() < self.latest_prob:
            pick = snaps[-1]
        else:
            weights = self._weights()
            w = [weights.get(p.stem, 1.0) for p in snaps] if weights else None
            pick = self.rng.choices(snaps, weights=w)[0]
        key = str(pick)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = NumpyPolicy(pick)
        self.policy, self.current = self._cache[key], pick.stem
        self.policy.reset_state()

    def _weights(self) -> dict[str, float] | None:
        path = self.pool_dir / WEIGHTS_FILE
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        if mtime != self._weights_mtime:
            try:
                self._weights_cache = json.loads(path.read_text())
                self._weights_mtime = mtime
            except (OSError, ValueError):
                pass  # keep the previous weights
        return self._weights_cache

    def __call__(self, obs: np.ndarray) -> int:
        if self.policy is None:
            self.reset()
        n_in = self.policy.layers[0][0].shape[1]
        if len(obs) != n_in:  # snapshot trained on shorter frames: keep the first features of each frame
            f, f_old = len(obs) // self.history, n_in // self.history
            obs = obs.reshape(self.history, f)[:, :f_old].reshape(-1)
        z = self.policy.logits(obs)
        p = np.exp(z - z.max())
        u = self.noise.next(1)[0] if self.noise else self.np_rng.random()
        return int(u < p[1] / p.sum())
