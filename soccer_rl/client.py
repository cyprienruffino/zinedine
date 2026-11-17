"""Low-level client for the SoccerRL BepInEx bridge (see plugin/SoccerRL/Bridge.cs for the protocol)."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np

# Player indices / bit positions in the button & mask bytes.
P1, P1G, P2, P2G = 0, 1, 2, 3
PLAYER_NAMES = ("P1", "P1G", "P2", "P2G")
BODY_FIELDS = ("x", "y", "angle", "vx", "vy", "angular_velocity")
PLAYER_FIELDS = ("grounded", "jumped", "ai_triggered", "button", "jump_limiter", "jump2_delay")

# Event bits (accumulated since the previous state).
EV_GOAL_P1 = 1 << 0  # P1 scored
EV_GOAL_P2 = 1 << 1  # P2 scored
EV_OUT_OF_BOUNDS = 1 << 2
EV_SCENE_LOADED = 1 << 3
EV_GAME_OVER = 1 << 4

# Flag bits (instantaneous).
FL_IN_MATCH = 1 << 0
FL_BALL_LIVE = 1 << 1
FL_GAME_OVER = 1 << 2
FL_BALL_GROUNDED = 1 << 3
FL_BALL_TOUCHING_P1 = 1 << 4
FL_BALL_TOUCHING_P2 = 1 << 5

_HEADER = struct.Struct("<qiiiiiffi")
_U32 = struct.Struct("<I")


@dataclass
class State:
    global_tick: int
    episode_tick: int
    p1_score: int
    p2_score: int
    events: int
    flags: int
    time_scale: float
    fixed_dt: float
    bodies: np.ndarray  # (n_bodies, 6): x, y, angle_deg, vx, vy, angular_velocity
    players: np.ndarray  # (4, 6): grounded, jumped, ai_triggered, button, jump_limiter, jump2_delay (timers in [0, 1])
    body_names: list[str] = field(default_factory=list)

    # --- convenience -------------------------------------------------------------------------
    def body(self, name: str) -> np.ndarray:
        return self.bodies[self.body_names.index(name)]

    @property
    def ball(self) -> np.ndarray:
        return self.body("Ball")

    @property
    def goal_p1(self) -> bool:
        return bool(self.events & EV_GOAL_P1)

    @property
    def goal_p2(self) -> bool:
        return bool(self.events & EV_GOAL_P2)

    @property
    def out_of_bounds(self) -> bool:
        return bool(self.events & EV_OUT_OF_BOUNDS)

    @property
    def game_over(self) -> bool:
        return bool(self.flags & FL_GAME_OVER)

    @property
    def ball_live(self) -> bool:
        return bool(self.flags & FL_BALL_LIVE)

    @property
    def in_match(self) -> bool:
        return bool(self.flags & FL_IN_MATCH)


def _kv(opts: dict | str | None) -> str:
    if opts is None:
        return ""
    if isinstance(opts, str):
        return opts
    out = []
    for k, v in opts.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out.append(f"{k}={v}")
    return ";".join(out)


class SoccerClient:
    """Synchronous connection to one game instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, connect_timeout: float = 60.0):
        deadline = time.monotonic() + connect_timeout
        last_err: Exception | None = None
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError as e:  # game still booting
                last_err = e
                if time.monotonic() > deadline:
                    raise ConnectionError(f"could not connect to game on {host}:{port}: {last_err}") from e
                time.sleep(0.1)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rbuf = bytearray(1 << 16)  # reused receive buffer (see _fill / _take)
        self._rview = memoryview(self._rbuf)
        self._rlen = 0
        self.body_names: list[str] = []
        self.info_cache: dict[str, str] = {}

    # --- framing -----------------------------------------------------------------------------
    def _fill(self, n: int) -> None:
        """Read from the socket until at least n bytes are buffered. Reads as much as is available, so a
        whole message (4-byte length + payload) usually arrives in a single recv_into call."""
        while self._rlen < n:
            if n > len(self._rbuf):
                self._rbuf.extend(bytes(n - len(self._rbuf)))
                self._rview = memoryview(self._rbuf)
            k = self.sock.recv_into(self._rview[self._rlen:])
            if k == 0:
                raise ConnectionError("game closed the connection")
            self._rlen += k

    def _take(self, n: int) -> bytes:
        """Consumes n buffered bytes (a copy, so returned data never aliases the reused buffer)."""
        self._fill(n)
        out = bytes(self._rview[:n])
        rest = self._rlen - n
        if rest:
            self._rview[:rest] = self._rview[n:self._rlen]
        self._rlen = rest
        return out

    def _recv_exact(self, n: int) -> bytes:
        return self._take(n)

    def _recv_message(self) -> bytes:
        """One length-prefixed message (u32 length, then payload), usually with one system call."""
        self._fill(4)
        (n,) = _U32.unpack_from(self._rbuf, 0)
        return self._take(4 + n)[4:]

    def _send(self, data: bytes) -> None:
        self.sock.sendall(data)

    @staticmethod
    def _str(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<I", len(b)) + b

    def _recv_str(self) -> str:
        return self._recv_message().decode()

    def _recv_state(self) -> State:
        buf = self._recv_message()
        gt, et, s1, s2, ev, fl, ts, fdt, nb = _HEADER.unpack_from(buf, 0)
        off = _HEADER.size
        bodies = np.frombuffer(buf, dtype="<f4", count=nb * 6, offset=off).reshape(nb, 6)
        off += nb * 6 * 4
        players = np.frombuffer(buf, dtype="<f4", count=4 * len(PLAYER_FIELDS), offset=off).reshape(4, len(PLAYER_FIELDS))
        if nb != len(self.body_names):
            self.info()  # schema changed (first reset / different modifiers)
        return State(gt, et, s1, s2, ev, fl, ts, fdt, bodies, players, self.body_names)

    # --- commands ----------------------------------------------------------------------------
    def step(self, buttons: int = 0, mask: int = 0b1111, repeat: int = 1) -> State:
        """Hold `buttons` (bit i = player i pressed) for `repeat` physics ticks on players in `mask`."""
        self._send(struct.pack("<cBBH", b"S", buttons & 0xF, mask & 0xF, max(1, min(repeat, 65535))))
        return self._recv_state()

    def reset(self, **options) -> State:
        """Start a fresh kickoff. See Bridge.BeginReset for options (seed, ai, two_buttons, modifiers...)."""
        self._send(b"R" + self._str(_kv(options)))
        state = self._recv_state()
        self.info()
        state.body_names = self.body_names
        return state

    def observe(self) -> State:
        self._send(b"O")
        return self._recv_state()

    def info(self) -> dict[str, str]:
        self._send(b"I")
        text = self._recv_str()
        d = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        self.body_names = [b for b in d.get("bodies", "").split(",") if b]
        self.info_cache = d
        return d

    def get(self, *queries: str) -> list[str]:
        """Read values by path, e.g. get('pref:P1Score', 'obj:P1:Player.grounded', 'static:Time.timeScale')."""
        self._send(b"G" + self._str("\n".join(queries)))
        return self._recv_str().splitlines()

    def set_values(self, assignments: dict[str, object]) -> list[str]:
        """Write values by path, e.g. set_values({'static:UnityEngine.Physics2D.gravity': '0,-10'})."""
        lines = []
        for k, v in assignments.items():
            if isinstance(v, (tuple, list, np.ndarray)):
                v = ",".join(str(float(x)) for x in v)
            elif isinstance(v, bool):
                v = int(v)
            lines.append(f"{k}={v}")
        self._send(b"X" + self._str("\n".join(lines)))
        return self._recv_str().splitlines()

    def configure(self, **settings) -> str:
        """Runtime settings: fast_restart, fast_spawn, mute, ticks_per_frame, fps."""
        self._send(b"C" + self._str(_kv(settings)))
        return self._recv_str()

    def pixels(self, width: int = 320, height: int = 180) -> np.ndarray | None:
        """Render the cameras into an (H, W, 3) uint8 image. Needs a graphics device (no -nographics)."""
        self._send(struct.pack("<cHH", b"P", width, height))
        n, w, h = struct.unpack("<IHH", self._recv_exact(8))
        if n == 0:
            return None
        img = np.frombuffer(self._recv_exact(n), dtype=np.uint8).reshape(h, w, 3)
        return img[::-1].copy()  # Unity rows are bottom-up

    def tree(self) -> str:
        """Dump the scene hierarchy (names, tags, positions, components)."""
        self._send(b"T")
        return self._recv_str()

    def quit(self) -> None:
        try:
            self._send(b"Q")
        except OSError:
            pass
        self.close()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
