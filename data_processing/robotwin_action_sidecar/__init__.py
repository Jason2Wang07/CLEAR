"""RoboTwin dual-arm ActionAE sidecar contract and conversion utilities."""

from .geometry import canonical_dual_arm_waypoints
from .schema import (
    EFFECT_DIM,
    HORIZON,
    NATIVE_ACTION_DIM,
    SCHEMA_VERSION,
    STATE_DIM,
    TRAJECTORY_DIM,
)

__all__ = [
    "EFFECT_DIM",
    "HORIZON",
    "NATIVE_ACTION_DIM",
    "SCHEMA_VERSION",
    "STATE_DIM",
    "TRAJECTORY_DIM",
    "canonical_dual_arm_waypoints",
]
