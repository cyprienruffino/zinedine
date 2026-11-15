"""Temporally correlated exploration for the Discrete(2) press/release policy.

Independent per-step sampling turns exploration into scattered taps, the worst behaviour in this game
(each press topples the players for seconds). Instead, actions are drawn by inverse CDF from a
uniform variable u_t that is *temporally correlated* (u_t = Phi(z_t), z_t colored Gaussian noise
with power spectrum ∝ 1/f^beta):

    press  ⇔  u_t < P(press | s_t)

Each u_t is still Uniform(0, 1) on its own, so every action keeps exactly the policy's probability and
PPO's likelihood ratios stay valid (the argument of "Colored Noise in PPO", Hollenstein et al. 2024,
for Gaussian policies). Consecutive draws are correlated, so presses and pauses come in runs.
beta = 0 is white noise (plain sampling), beta = 1 pink noise; ~0.5 worked best for on-policy PPO.
"""

from __future__ import annotations

import numpy as np
import torch


def colored_gaussian(beta: float, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """n independent sequences of `length` samples with PSD ∝ 1/f^beta, each scaled to unit variance."""
    white = rng.standard_normal((n, length))
    if beta == 0:
        return white
    f = np.fft.rfftfreq(length)
    f[0] = f[1]  # no infinite DC power
    spectrum = np.fft.rfft(white, axis=1) * f ** (-beta / 2)
    x = np.fft.irfft(spectrum, n=length, axis=1)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


class CorrelatedUniform:
    """Streams of correlated Uniform(0, 1) numbers, one stream per env, refilled in chunks."""

    def __init__(self, beta: float, chunk: int = 4096, seed: int | None = None):
        self.beta, self.chunk = beta, chunk
        self.rng = np.random.default_rng(seed)
        self.buf: np.ndarray | None = None
        self.pos = 0

    def next(self, n: int) -> np.ndarray:
        if self.buf is None or self.buf.shape[0] != n or self.pos >= self.chunk:
            z = colored_gaussian(self.beta, n, self.chunk, self.rng)
            self.buf = torch.special.ndtr(torch.from_numpy(z)).numpy()  # Phi(z): Uniform(0, 1) marginals
            self.pos = 0
        u = self.buf[:, self.pos]
        self.pos += 1
        return u


def __getattr__(name):
    # CorrelatedPolicy needs stable-baselines3, which other trainers (Sample Factory venv) don't install:
    # build it on first access only.
    if name == "CorrelatedPolicy":
        cls = _make_correlated_policy()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


def _make_correlated_policy():
    from stable_baselines3.common.policies import ActorCriticPolicy

    class CorrelatedPolicy(ActorCriticPolicy):
        """ActorCriticPolicy whose rollout sampling uses correlated noise (policy_kwargs noise_beta=...).

        Only `forward` changes, which SB3's PPO uses to collect rollouts; training (evaluate_actions) and
        predict() are untouched, so saved models load as a plain MlpPolicy too."""

        def __init__(self, *args, noise_beta: float = 0.5, **kwargs):
            self.noise_beta = noise_beta
            super().__init__(*args, **kwargs)
            self._noise = CorrelatedUniform(noise_beta)

        def forward(self, obs: torch.Tensor, deterministic: bool = False):
            if deterministic or self.noise_beta == 0:
                return super().forward(obs, deterministic)
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            values = self.value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            p_press = distribution.distribution.probs[:, 1]
            u = torch.as_tensor(self._noise.next(len(p_press)), dtype=p_press.dtype, device=p_press.device)
            actions = (u < p_press).long()
            return actions, values, distribution.log_prob(actions)

    CorrelatedPolicy.__qualname__ = "CorrelatedPolicy"  # pickled by reference as soccer_rl.noise.CorrelatedPolicy
    return CorrelatedPolicy
