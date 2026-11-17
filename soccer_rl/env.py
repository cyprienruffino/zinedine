"""Gymnasium environment for Soccer Physics."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import P1, P1G, P2, P2G, State
from .launcher import GameProcess

# Bodies per player in the order the plugin reports them (stable across reloads).
PARTS = ("Torso", "Head", "ArmB", "Leg", "LegKick", "ArmF")
BODY_FEATURES = 7  # x, y, sin(angle), cos(angle), vx, vy, angular_velocity
PLAYER_FLAGS = 3  # grounded, jumped, button held
PLAYER_TIMERS = 2  # jump_limiter, jump2_delay (obs="v2" only)
# Per-feature mean/std of the v2 frame measured on real games (scripts/obs_stats.py); flags are left as-is.
OBS_STATS_PATH = Path(__file__).with_name("obs_stats_v2.npz")

# The game's own modifier probabilities (Game.Start), used for modifiers="random".
_GAME_MODIFIER_ROLLS = {
    "ground": lambda r: 2 if r.random() >= 0.8 else 1,
    "splits": lambda r: int(r.random() >= 0.87),
    "mini_goals": lambda r: int(r.random() > 0.9),
    "big_goals": lambda r: int(r.random() > 0.82),
    "ball": lambda r: (lambda v: 2 if v > 0.88 else 1 if v < 0.15 else 0)(r.random()),
    "headless": lambda r: int(r.random() > 0.92),
}
NO_MODIFIERS = {"ground": 1, "splits": 0, "mini_goals": 0, "big_goals": 0, "ball": 0, "headless": 0}

Opponent = Callable[[np.ndarray], Any]


class SoccerPhysicsEnv(gym.Env):
    """Single-agent Soccer Physics.

    The agent controls one team (P1 = blue, right side by default; P2 = red, left side).
    Observations are egocentric: when playing P2 everything is mirrored (x → -x) so that the agent
    always defends the goal at +x and attacks towards -x. A policy trained on one side therefore
    plays the other side as-is, which is what makes `opponent=<policy>` self-play work.

    Args:
        side: "p1" or "p2" — which team the agent controls.
        opponent: "builtin" (the game's AI, only drives P2), "idle", "random", "keyboard" (a human
            plays P1 with the Up arrow / Left+Right in two-button mode; needs side="p2" and a
            window), or a callable obs -> action receiving the opponent's own egocentric observation.
        two_buttons: False → Discrete(2) (one button moves both team members, as in the original
            one-button mode); True → MultiBinary(2) = [outfield player, goalie].
        frame_skip: physics ticks per env step (game runs 50 ticks / second of real time).
        episode: "goal" ends the episode at the first goal; "match" plays first-to-5.
        end_on_out_of_bounds: in "goal" episodes, truncate when the ball leaves the arena.
        max_steps: truncate after this many env steps (None = never).
        modifiers: "none" (classic match), "random" (game's own probabilities, re-rolled per
            episode), or a dict like {"ground": 2, "ball": 1} (missing keys = none).
        shaping: weight of a dense reward for moving the ball towards the opponent's goal
            (per unit of x per step). 0 disables it. Can be changed between episodes (annealing).
        draw_penalty: reward subtracted when an episode ends without a goal (timeout, or ball out of
            bounds in goal mode): the goal is to win, not to draw. Pair it with time_feature so the agent can see the deadline coming.
        time_feature: append a point clock to every frame: env steps since the last kickoff (episode start
            or goal) / clock_horizon, capped at 1. In single-goal episodes that's the fraction of max_steps
            used; in matches it restarts every point. Lets the agent see the draw penalty coming.
        clock_horizon: steps for the clock to reach 1 (default: max_steps, or 2000 without a cap).
        modifier_feature: append the episode's match modifiers to every frame (after the clock), so one
            policy can play every variant: slippery ground, splits, mini goals, big goals, American
            football, beach ball, headless (7 values in {0, 1}).
        switch_cost: reward subtracted at every new press (button going down). An action-rate
            penalty: committing to long holds becomes cheaper than mashing, without restricting actions.
            Anneal it to 0.
        press_cost: reward subtracted for every step the agent holds its button. Keeps exploration near
            "stand still" (a strong default in this game) while the value function learns; anneal it to 0.
        obs: "v1" (body states + 3 flags per player, hand-picked scales; what the first models used) or
            "v2" (adds each player's hidden jump timers, and normalizes every feature with the mean/std
            measured on real games, stored in obs_stats_v2.npz; raw v2 if that file doesn't exist yet).
        history: number of consecutive frames in each observation (oldest first). Gives the policy
            short-term memory, e.g. of hidden jump cooldowns. Policy opponents get the same history.
        render_mode: None, "rgb_array" (needs headless=False) or "human" (a real-time window).
        headless: run with -batchmode -nographics (no window, no pixels).
        game_kwargs: extra GameProcess arguments (game_dir, ticks_per_frame, log_path...).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        side: str = "p1",
        opponent: str | Opponent = "builtin",
        two_buttons: bool = False,
        frame_skip: int = 4,
        episode: str = "goal",
        end_on_out_of_bounds: bool = True,
        max_steps: int | None = 1500,
        modifiers: str | dict = "none",
        shaping: float = 0.0,
        press_cost: float = 0.0,
        switch_cost: float = 0.0,
        draw_penalty: float = 0.0,
        time_feature: bool = False,
        clock_horizon: int | None = None,
        modifier_feature: bool = False,
        obs: str = "v1",
        history: int = 1,
        render_mode: str | None = None,
        headless: bool | None = None,
        render_size: tuple[int, int] = (480, 270),
        game_kwargs: dict | None = None,
    ):
        super().__init__()
        assert side in ("p1", "p2")
        assert episode in ("goal", "match")
        assert render_mode in (None, "human", "rgb_array")
        if opponent == "builtin" and side != "p1":
            raise ValueError("the game's built-in AI can only drive P2; use side='p1' or another opponent")
        self.side = side
        self.opponent = opponent
        self.two_buttons = two_buttons
        self.frame_skip = frame_skip
        self.episode = episode
        self.end_on_oob = end_on_out_of_bounds
        self.max_steps = max_steps
        self.modifiers = modifiers
        self.shaping = shaping
        self.history = history
        self.press_cost = press_cost
        self.switch_cost = switch_cost
        self._last_press = False
        self.draw_penalty = draw_penalty
        self.time_feature = time_feature
        self.clock_horizon = clock_horizon or max_steps or 2000
        self._point_steps = 0
        assert obs in ("v1", "v2")
        self.obs_version = obs
        self._norm = None
        if obs == "v2" and OBS_STATS_PATH.exists():
            z = np.load(OBS_STATS_PATH)
            self._norm = (z["mean"].astype(np.float32), z["std"].astype(np.float32))
        self.render_mode = render_mode
        self.render_size = render_size
        if headless is None:
            headless = render_mode is None
        if headless and render_mode is not None:
            raise ValueError("rendering needs headless=False")
        if opponent == "keyboard" and (side != "p2" or headless):
            raise ValueError("opponent='keyboard' needs side='p2' (you play P1 with the Up arrow) and a window")
        kw = dict(game_kwargs or {})
        if render_mode == "human":
            kw.setdefault("fps", 50)  # one tick per frame at 50 fps = real time
        elif render_mode == "rgb_array":
            # Frames are captured off-screen; only draw the window once per env step.
            kw.setdefault("ticks_per_frame", frame_skip)
            kw.setdefault("width", render_size[0])
            kw.setdefault("height", render_size[1])
        self._game_kwargs = dict(headless=headless, **kw)
        self.game: GameProcess | None = None

        self.mirror = side == "p2"
        self.me = (P2, P2G) if self.mirror else (P1, P1G)
        self.them = (P1, P1G) if self.mirror else (P2, P2G)

        n_bodies = 4 * len(PARTS) + 1
        self.frame_dim = (n_bodies * BODY_FEATURES + 4 * (PLAYER_FLAGS + (PLAYER_TIMERS if obs == "v2" else 0))
                          + int(time_feature) + 7 * int(modifier_feature))
        self.modifier_feature = modifier_feature
        self._mods_vec = np.zeros(7, np.float32)
        self.obs_dim = self.frame_dim * history
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self._frames: dict[bool, deque] = {}  # mirror flag → last `history` frames of that view
        self._last_obs: dict[bool, np.ndarray] = {}  # mirror flag → last complete frame (for reload ticks)
        self.action_space = spaces.MultiBinary(2) if two_buttons else spaces.Discrete(2)

        self._state: State | None = None
        self._steps = 0
        self._names: list[str] | None = None
        self._index: dict[bool, tuple[np.ndarray, np.ndarray]] = {}

    # --- lifecycle -------------------------------------------------------------------------------
    @property
    def state(self) -> State | None:
        """Raw game state from the last reset/step (absolute coordinates, not mirrored)."""
        return self._state

    @property
    def client(self):
        if self.game is None:
            self.game = GameProcess(**self._game_kwargs)
        return self.game.client

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        game_seed = int(self.np_random.integers(0, 2**31 - 1))
        mods = options.pop("modifiers", self.modifiers)
        if mods == "none":
            mods = dict(NO_MODIFIERS)
        elif mods == "random":
            mods = {k: f(self.np_random) for k, f in _GAME_MODIFIER_ROLLS.items()}
        else:
            mods = {**NO_MODIFIERS, **mods}
        reset_opts = dict(
            seed=game_seed,
            ai=int(self.opponent == "builtin"),
            two_buttons=int(self.two_buttons),
            endless=0,
            # Single-goal episodes end on every reload trigger and reset() reloads the pitch itself,
            # so the game's own reload would be wasted.
            auto_reload=int(not (self.episode == "goal" and self.end_on_oob)),
        )
        reset_opts.update(mods)
        self._mods_vec = np.array([mods["ground"] == 2, mods["splits"], mods["mini_goals"], mods["big_goals"],
                                   mods["ball"] == 1, mods["ball"] == 2, mods["headless"]], dtype=np.float32)
        reset_opts.update(options)  # raw plugin reset options win (see Bridge.BeginReset)
        self._state = self.client.reset(**reset_opts)
        self._steps = 0
        self._point_steps = 0
        self._last_press = False
        self._frames.clear()
        if hasattr(self.opponent, "reset"):  # e.g. a self-play opponent picking a new snapshot
            self.opponent.reset()
        return self._view(self._state, self.mirror), self._info(self._state)

    def step(self, action):
        st = self._state
        my_btn = self._buttons(action)
        buttons = 0
        mask = 0
        for i, b in zip(self.me, my_btn):
            mask |= 1 << i
            buttons |= int(b) << i
        if self.opponent not in ("builtin", "keyboard"):
            opp = self._opponent_action(st)
            for i, b in zip(self.them, opp):
                mask |= 1 << i
                buttons |= int(b) << i

        prev_ball_x = self._ball_x(st)
        st = self.client.step(buttons, mask, self.frame_skip)
        self._state = st
        self._steps += 1
        self._point_steps += 1

        scored_p1, scored_p2 = st.goal_p1, st.goal_p2
        if scored_p1 or scored_p2:  # new point: the clock restarts (matches)
            self._point_steps = 0
        i_scored = scored_p2 if self.mirror else scored_p1
        they_scored = scored_p1 if self.mirror else scored_p2
        reward = float(i_scored) - float(they_scored) - self.press_cost * float(any(my_btn))
        pressed = bool(any(my_btn))
        if pressed and not self._last_press:
            reward -= self.switch_cost
        self._last_press = pressed
        if self.shaping and not st.events:  # skip ticks where the ball teleported (goal/reload)
            # Egocentric: we attack towards -x, so ball moving to -x is good.
            dx = self._ball_x(st) - prev_ball_x
            reward += self.shaping * (-dx)

        terminated = False
        truncated = False
        if self.episode == "goal":
            scored = scored_p1 or scored_p2
            # Ball out ends the round for good (a draw), so it's terminal: SB3 must not bootstrap past it.
            out = bool(self.end_on_oob and st.out_of_bounds)
            terminated = scored or out
        else:
            scored = out = False
            terminated = st.game_over
        timeout = self.max_steps is not None and self._steps >= self.max_steps and not terminated
        if timeout:
            # With the clock in the observation the deadline is part of the game: a real end. Without it,
            # it's a plain time limit and SB3 bootstraps the value as usual.
            terminated, truncated = (True, False) if self.time_feature else (False, True)
        if (out or timeout) and not scored:  # no winner: penalized alike, so kicking the ball out
            reward -= self.draw_penalty       # can't dodge the penalty

        obs = self._view(st, self.mirror)
        return obs, reward, terminated, truncated, self._info(st)

    def set_side(self, side: str) -> None:
        """Switches which team the agent controls (takes effect at the next reset), reusing the game."""
        assert side in ("p1", "p2")
        if self.opponent == "builtin" and side != "p1":
            raise ValueError("the game's built-in AI can only drive P2")
        self.side = side
        self.mirror = side == "p2"
        self.me = (P2, P2G) if self.mirror else (P1, P1G)
        self.them = (P1, P1G) if self.mirror else (P2, P2G)
        self._frames.clear()

    def set_shaping(self, weight: float) -> None:
        """Changes the shaping weight (e.g. annealing it from a vec env via env_method)."""
        self.shaping = weight

    def set_switch_cost(self, cost: float) -> None:
        """Changes the per-new-press cost (e.g. annealing it from a vec env via env_method)."""
        self.switch_cost = cost

    def set_press_cost(self, cost: float) -> None:
        """Changes the per-step press cost (e.g. annealing it from a vec env via env_method)."""
        self.press_cost = cost

    def render(self):
        if self.render_mode == "rgb_array":
            return self.client.pixels(*self.render_size)
        return None

    # --- helpers ---------------------------------------------------------------------------------
    def _buttons(self, action) -> tuple[int, int]:
        if self.two_buttons:
            a = np.asarray(action).reshape(-1)
            return int(a[0]), int(a[1])
        a = int(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else int(action)
        return a, a

    def _opponent_action(self, st: State) -> tuple[int, int]:
        if self.opponent == "idle":
            return 0, 0
        if self.opponent == "random":
            r = self.np_random.random(2) < 0.3
            return (int(r[0]), int(r[1])) if self.two_buttons else (int(r[0]), int(r[0]))
        # Policy opponent: give it its own egocentric observation, unless it says it doesn't need one
        # (the duel env drives the opponent's buttons itself and builds that view on its own).
        if getattr(self.opponent, "needs_obs", True) is False:
            return self._buttons(self.opponent(None))
        obs = self._view(st, not self.mirror)
        return self._buttons(self.opponent(obs))

    def _indices(self, st: State, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
        """Body and player index arrays for the egocentric view of the team that `mirror` designates."""
        if st.body_names is not self._names:  # schema (re)announced by the plugin
            self._names = st.body_names
            self._index.clear()
        if mirror not in self._index:
            team = [P2, P2G, P1, P1G] if mirror else [P1, P1G, P2, P2G]
            names = st.body_names
            bodies = [names.index(f"{('P1', 'P1G', 'P2', 'P2G')[p]}/{part}") for p in team for part in PARTS]
            bodies.append(names.index("Ball"))
            self._index[mirror] = (np.array(bodies), np.array(team))
        return self._index[mirror]

    def _ball_x(self, st: State) -> float:
        x = float(st.ball[0])
        return -x if self.mirror else x

    def _view(self, st: State, mirror: bool) -> np.ndarray:
        """Observation with history for one team; call once per state and team."""
        try:
            frame = self._obs(st, mirror)
            self._last_obs[mirror] = frame
        except ValueError:
            # Match mode: right after a goal the game reloads the pitch and, for a tick or so, reports an
            # incomplete body list. Repeat the last complete frame for that tick.
            frame = self._last_obs.get(mirror)
            if frame is None:
                raise
        if self.time_feature or self.modifier_feature:
            full = np.empty(self.frame_dim, np.float32)  # one allocation instead of append + concatenate
            n = len(frame)
            full[:n] = frame
            if self.time_feature:
                full[n] = min(self._point_steps / self.clock_horizon, 1.0)
                n += 1
            if self.modifier_feature:
                full[n:n + 7] = self._mods_vec
            frame = full
        if self.history == 1:
            return frame
        frames = self._frames.get(mirror)
        if frames is None:  # episode start: pad with copies of the first frame
            frames = self._frames[mirror] = deque([frame] * self.history, maxlen=self.history)
        else:
            frames.append(frame)
        return np.concatenate(frames)

    def _obs(self, st: State, mirror: bool) -> np.ndarray:
        """Egocentric observation: own player, own goalie, their player, their goalie, ball.
        mirror=True flips the x axis so that the observing team always attacks towards -x."""
        body_idx, player_idx = self._indices(st, mirror)
        if self.obs_version == "v2":
            return self._obs_v2(st, mirror, body_idx, player_idx)
        b = st.bodies[body_idx].astype(np.float32)  # (25, 6)
        x, y, ang, vx, vy, w = b.T
        if mirror:
            x, vx, ang, w = -x, -vx, -ang, -w
        rad = np.deg2rad(ang)
        feats = np.stack([x / 5.0, y / 5.0, np.sin(rad), np.cos(rad), vx / 10.0, vy / 10.0, w / 500.0], axis=1)
        if self.obs_version == "v1":
            flags = st.players[player_idx][:, [0, 1, 3]]  # grounded, jumped, button
            return np.concatenate([feats.reshape(-1), flags.reshape(-1)]).astype(np.float32)
        flags = st.players[player_idx][:, [0, 1, 3, 4, 5]]  # + jump_limiter, jump2_delay
        frame = np.concatenate([feats.reshape(-1), flags.reshape(-1)]).astype(np.float32)
        if self._norm is not None:
            frame = np.clip((frame - self._norm[0]) / self._norm[1], -10.0, 10.0)
        return frame

    # x, y, angle, vx, vy, angular velocity: mirroring flips x, angle, vx and angular velocity.
    _MIRROR = np.array([-1, 1, -1, -1, 1, -1], np.float32)
    _SCALE = np.array([1 / 5.0, 1 / 5.0, 1 / 10.0, 1 / 10.0, 1 / 500.0], np.float32)  # x, y, vx, vy, w

    def _obs_v2(self, st: State, mirror: bool, body_idx: np.ndarray, player_idx: np.ndarray) -> np.ndarray:
        """Same frame as the original v2 code (within float32 rounding), with far fewer temporary arrays:
        it runs twice per duel step in every env, so it was a large share of the rollout workers' CPU."""
        b = st.bodies[body_idx]  # (25, 6) float32 (fancy indexing already copies)
        if mirror:
            b *= self._MIRROR
        n = len(body_idx)
        out = np.empty(n * BODY_FEATURES + 4 * (PLAYER_FLAGS + PLAYER_TIMERS), np.float32)
        f = out[:n * BODY_FEATURES].reshape(n, BODY_FEATURES)
        rad = np.deg2rad(b[:, 2])
        f[:, 0:2] = b[:, 0:2] * self._SCALE[0:2]
        np.sin(rad, out=f[:, 2])
        np.cos(rad, out=f[:, 3])
        f[:, 4:7] = b[:, 3:6] * self._SCALE[2:5]
        out[n * BODY_FEATURES:] = st.players[player_idx][:, [0, 1, 3, 4, 5]].ravel()
        if self._norm is not None:
            if not hasattr(self, "_inv_std"):
                self._inv_std = (1.0 / self._norm[1]).astype(np.float32)
            np.subtract(out, self._norm[0], out=out)
            np.multiply(out, self._inv_std, out=out)
            np.clip(out, -10.0, 10.0, out=out)
        return out

    def _info(self, st: State) -> dict:
        mine, theirs = (st.p2_score, st.p1_score) if self.mirror else (st.p1_score, st.p2_score)
        return {
            "score": (mine, theirs),
            "events": st.events,
            "episode_tick": st.episode_tick,
            "ball_live": st.ball_live,
            "out_of_bounds": st.out_of_bounds,
            "opponent_id": getattr(self.opponent, "current", None),  # e.g. which self-play snapshot
        }


def make_vec_env(n: int, asynchronous: bool = True, **env_kwargs) -> gym.vector.VectorEnv:
    """n environments, each backed by its own game process."""
    fns = [lambda: SoccerPhysicsEnv(**env_kwargs) for _ in range(n)]
    if asynchronous:
        return gym.vector.AsyncVectorEnv(fns, shared_memory=False)
    return gym.vector.SyncVectorEnv(fns)
