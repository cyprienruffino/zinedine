"""Two-player Soccer Physics for trainers that drive both teams (e.g. Sample Factory's multi-agent mode,
where the learning policies play each other: self-play without snapshot files).

Agent 0 plays P1 (blue), agent 1 plays P2 (red); each gets its own egocentric view (P2 mirrored), with its
own frame history, so one network plays both sides. Rewards per agent: goals (zero-sum), the draw penalty
(both), the switch cost on each new press (per agent), and the ball-progress shaping (antisymmetric). The
env resets itself when an episode ends (multi-agent trainers expect that) and reports episode statistics.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import SoccerPhysicsEnv


class SoccerDuelEnv(gym.Env):
    num_agents = 2
    is_multiagent = True

    def __init__(self, draw_penalty: float = 0.3, switch_cost: float = 0.0, shaping: float = 0.0,
                 anneal_steps: int = 0, anneal_start: int = 0, **env_kwargs):
        """anneal_steps: fade switch_cost and shaping linearly to 0 over this many of *this env's* steps
        (trainers without callbacks, like Sample Factory, can't change env settings mid-run).
        anneal_start: steps this env slot already did before a restart, so the fade resumes, not restarts."""
        super().__init__()
        self._action = 0
        # The inner env runs the game for P1 (agent 0); agent 1's buttons come from self._action.
        self.inner = SoccerPhysicsEnv(side="p1", opponent=lambda _obs: self._action, draw_penalty=draw_penalty,
                                      shaping=shaping, **env_kwargs)
        self.view = SoccerPhysicsEnv(side="p1", **env_kwargs)  # no game: builds agent 1's own (mirrored) history
        self.draw_penalty, self.switch_cost = draw_penalty, switch_cost
        self.switch_cost0, self.shaping0, self.anneal_steps = switch_cost, shaping, anneal_steps
        self.total_steps = anneal_start
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self._stats = None

    def _p2_obs(self):
        self.view._point_steps = self.inner._point_steps  # the view env never steps: share the point clock
        self.view._mods_vec = self.inner._mods_vec  # and the match modifiers
        return self.view._view(self.inner.state, True)

    def reset(self, seed=None, options=None):
        o1, info = self.inner.reset(seed=seed, options=options)
        self.view._frames.clear()
        self._last = np.zeros(2, bool)
        self._run = np.zeros(2, int)
        self._stats = dict(steps=0, presses=np.zeros(2), onsets=np.zeros(2), taps=np.zeros(2))
        return [o1, self._p2_obs()], [info, info]

    def step(self, actions):
        self.total_steps += 1
        if self.anneal_steps and self.total_steps % 1000 == 1:
            frac = max(0.0, 1.0 - self.total_steps / self.anneal_steps)
            self.switch_cost = self.switch_cost0 * frac
            self.inner.set_shaping(self.shaping0 * frac)
        a = np.asarray(actions, dtype=int).reshape(2)
        self._action = int(a[1])
        o1, r1, term, trunc, info = self.inner.step(int(a[0]))
        st = self.inner.state
        draw = (term or trunc) and not (st.goal_p1 or st.goal_p2)
        # r1 = goals + shaping - draw penalty (from the inner env); agent 1 gets the zero-sum mirror image.
        r2 = -(r1 + self.draw_penalty * draw) - self.draw_penalty * draw
        pressed = a > 0
        onset = pressed & ~self._last
        rewards = np.array([r1, r2]) - self.switch_cost * onset
        s = self._stats
        s["steps"] += 1
        s["presses"] += pressed
        s["onsets"] += onset
        s["taps"] += (~pressed) & self._last & (self._run == 1)  # a press that lasted one decision
        self._run = np.where(pressed, self._run + 1, 0)
        self._last = pressed
        obs = [o1, self._p2_obs()]
        done = term or trunc
        infos = [dict(info), dict(info)]
        if done:
            mine = (st.goal_p1 - st.goal_p2, st.goal_p2 - st.goal_p1)
            onsets = np.maximum(s["onsets"], 1)
            for k in range(2):
                infos[k]["episode_extra_stats"] = dict(
                    win=float(mine[k] > 0), loss=float(mine[k] < 0), draw=float(draw),
                    press_rate=float(s["presses"][k] / s["steps"]), tap_share=float(s["taps"][k] / onsets[k]),
                    hold_mean=float(s["presses"][k] / onsets[k]), episode_len=float(s["steps"]),
                    switch_cost=float(self.switch_cost))
            obs, _ = self.reset()
        return obs, list(rewards), [term] * 2, [trunc] * 2, infos

    def set_switch_cost(self, cost: float) -> None:
        self.switch_cost = cost

    def close(self):
        self.inner.close()


class SoccerVsPoolEnv(gym.Env):
    """One learning agent against frozen snapshots (curriculum phase 3: playing past selves instead of only
    the current partner, against forgetting and strategy cycling). Single-agent, so Sample Factory
    records only the learner's moves; the opponent is drawn per episode by SnapshotOpponent and acts by
    sampling, like in training. Reports the same episode statistics as SoccerDuelEnv."""

    def __init__(self, pool_dirs, side: str = "p1", latest_prob: float = 0.1, recent_frac: float = 0.5,
                 seed: int | None = None, **env_kwargs):
        super().__init__()
        from .selfplay import SnapshotOpponent

        self.opponent = SnapshotOpponent(pool_dirs, latest_prob=latest_prob, recent_frac=recent_frac, seed=seed)
        self.env = SoccerPhysicsEnv(side=side, opponent=self.opponent, **env_kwargs)
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)  # also draws a new opponent
        self._stats = dict(steps=0, presses=0, onsets=0, taps=0, run=0, last=False)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        s, pressed = self._stats, bool(int(np.asarray(action).reshape(-1)[0]))
        s["steps"] += 1
        s["presses"] += pressed
        s["onsets"] += pressed and not s["last"]
        s["taps"] += (not pressed) and s["last"] and s["run"] == 1
        s["run"] = s["run"] + 1 if pressed else 0
        s["last"] = pressed
        if term or trunc:
            mine, theirs = info["score"]
            onsets = max(s["onsets"], 1)
            info = dict(info, episode_extra_stats=dict(
                win=float(mine > theirs), loss=float(mine < theirs), draw=float(mine == theirs),
                press_rate=s["presses"] / s["steps"], tap_share=s["taps"] / onsets,
                hold_mean=s["presses"] / onsets, episode_len=float(s["steps"])))
        return obs, reward, term, trunc, info

    def close(self):
        self.env.close()
