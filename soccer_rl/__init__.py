"""Reinforcement-learning tooling for Soccer Physics (via the SoccerRL BepInEx plugin)."""

from .client import (
    P1,
    P1G,
    P2,
    P2G,
    PLAYER_NAMES,
    SoccerClient,
    State,
)
from .launcher import GameProcess

__all__ = ["P1", "P1G", "P2", "P2G", "PLAYER_NAMES", "SoccerClient", "State", "GameProcess"]

try:  # gymnasium is optional
    from .env import SoccerPhysicsEnv, make_vec_env  # noqa: F401

    __all__ += ["SoccerPhysicsEnv", "make_vec_env"]
except ImportError:  # pragma: no cover
    pass
