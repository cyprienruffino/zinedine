"""Launch Soccer Physics game instances with the SoccerRL bridge enabled."""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import tempfile
from pathlib import Path

from .client import SoccerClient

DEFAULT_GAME_DIR = Path(__file__).resolve().parent.parent / "SoccerPhysics"


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def default_wrapper() -> list[str]:
    """On NixOS the Unity binary needs an FHS environment; steam-run provides one."""
    env = os.environ.get("SOCCER_RL_WRAPPER")
    if env is not None:
        return env.split()
    if Path("/etc/NIXOS").exists() and shutil.which("steam-run"):
        return ["steam-run"]
    return []


class GameProcess:
    """One game process + a connected SoccerClient.

    headless=True  → -batchmode -nographics: no window, no GPU, fastest. Pixel observations unavailable.
    headless=False → a normal window (use fps=50 to watch at real-time speed).
    """

    def __init__(
        self,
        game_dir: str | os.PathLike | None = None,
        headless: bool = True,
        port: int | None = None,
        width: int = 960,
        height: int = 540,
        ticks_per_frame: int = 1,
        fps: int = -1,
        log_path: str | os.PathLike | None = None,
        isolate_prefs: bool = True,
        sound: bool = False,
        extra_args: list[str] | None = None,
        connect_timeout: float = 60.0,
    ):
        self.game_dir = Path(game_dir or os.environ.get("SOCCER_PHYSICS_DIR") or DEFAULT_GAME_DIR).resolve()
        launcher = self.game_dir / "run_bepinex.sh"
        if not launcher.exists():
            raise FileNotFoundError(f"{launcher} not found: is BepInEx installed in {self.game_dir}?")
        self.port = port or find_free_port()
        self._tmp = tempfile.mkdtemp(prefix="soccer_rl_")
        self.log_path = Path(log_path) if log_path else Path(self._tmp) / "player.log"

        cmd = [*default_wrapper(), str(launcher)]
        if headless:
            cmd += ["-batchmode", "-nographics"]
        else:
            cmd += ["-screen-fullscreen", "0", "-screen-width", str(width), "-screen-height", str(height)]
        cmd += [
            # Log to stdout and let us write the file: wrappers like steam-run give the game a
            # private /tmp, so a -logFile path there would silently go nowhere.
            "-logFile", "-",
            "-rl-port", str(self.port),
            "-rl-exit-on-disconnect",
            "-rl-ticks-per-frame", str(ticks_per_frame),
            "-rl-fps", str(fps),
            # The game is single-threaded; Unity's job workers only burn CPU spinning, which
            # hurts badly when running many instances side by side.
            "-job-worker-count", "0",
            *(extra_args or []),
        ]
        env = dict(os.environ)
        if isolate_prefs:
            # Unity stores PlayerPrefs under $XDG_CONFIG_HOME/unity3d; keep each instance (and your real
            # save) separate since the bridge rewrites lots of prefs.
            env["XDG_CONFIG_HOME"] = str(Path(self._tmp) / "config")
        if not sound:
            # The in-game mute only zeroes the volume; make the audio backends unreachable so the
            # game never opens an output stream (no sound even before the plugin connects).
            env["PULSE_SERVER"] = "unix:/nonexistent"
            env["ALSA_CONFIG_PATH"] = "/dev/null"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.game_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so we can kill the whole wrapper process group
        )
        atexit.register(self.close)
        try:
            self.client = SoccerClient(port=self.port, connect_timeout=connect_timeout)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.quit()
            self.client = None
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        if getattr(self, "_log_file", None) is not None:
            self._log_file.close()
            self._log_file = None
        # Temp dir holds the isolated PlayerPrefs and (unless log_path was given) the player log.
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> "GameProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
