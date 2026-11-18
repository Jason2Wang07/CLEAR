"""Canonical dual-arm action trajectory geometry for RoboTwin."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _validated_arm(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [T,8] xyz+qwxyz+gripper, got {array.shape}")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    result = array.copy()
    result[:, 3:7] /= norms[:, None]
    return result


def _canonical_arm(start: np.ndarray, post_states: np.ndarray, *, name: str) -> np.ndarray:
    first = _validated_arm(np.asarray(start)[None], name=f"{name}_start")[0]
    states = _validated_arm(post_states, name=f"{name}_post_states")
    start_rotation = Rotation.from_quat(first[[4, 5, 6, 3]])
    rotations = Rotation.from_quat(states[:, [4, 5, 6, 3]])
    local_position = start_rotation.inv().apply(states[:, :3] - first[:3])
    relative_rotation = (start_rotation.inv() * rotations).as_rotvec()
    return np.concatenate((local_position, relative_rotation, states[:, 7:8]), axis=1)


def canonical_dual_arm_waypoints(start_state: np.ndarray, post_action_states: np.ndarray) -> np.ndarray:
    """Return left/right local xyz, relative rotvec, and physical gripper."""

    start = np.asarray(start_state, dtype=np.float64)
    states = np.asarray(post_action_states, dtype=np.float64)
    if start.shape != (16,) or states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"expected start [16] and post-action states [T,16], got {start.shape}/{states.shape}")
    if not np.isfinite(start).all() or not np.isfinite(states).all():
        raise ValueError("dual-arm states contain NaN/Inf")
    left = _canonical_arm(start[:8], states[:, :8], name="left")
    right = _canonical_arm(start[8:], states[:, 8:], name="right")
    return np.concatenate((left, right), axis=1).astype(np.float32)
