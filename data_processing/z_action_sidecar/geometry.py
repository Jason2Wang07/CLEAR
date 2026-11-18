"""Canonical trajectory geometry for the z-action sidecar."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _validated_poses(poses: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [T,7] xyz+qwxyz poses, got {value.shape}")
    norms = np.linalg.norm(value[:, 3:], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    value = value.copy()
    value[:, 3:] /= norms[:, None]
    return value


def canonical_waypoints(start_pose: np.ndarray, post_action_poses: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
    """Return local position, relative rotvec, and physical gripper width."""

    start = _validated_poses(np.asarray(start_pose)[None], name="start_pose")[0]
    poses = _validated_poses(post_action_poses, name="post_action_poses")
    gripper = np.asarray(gripper_width, dtype=np.float64)
    if gripper.shape != (len(poses),) or not np.isfinite(gripper).all():
        raise ValueError(f"gripper_width must be finite [{len(poses)}], got {gripper.shape}")

    start_rotation = Rotation.from_quat([start[4], start[5], start[6], start[3]])
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    local_position = start_rotation.inv().apply(poses[:, :3] - start[:3])
    relative_rotation = (start_rotation.inv() * rotations).as_rotvec()
    return np.concatenate((local_position, relative_rotation, gripper[:, None]), axis=1).astype(np.float32)


def pose_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return position and SO(3) geodesic errors for xyz+qwxyz poses."""

    a = _validated_poses(np.asarray(left)[None], name="left")[0]
    b = _validated_poses(np.asarray(right)[None], name="right")[0]
    ra = Rotation.from_quat(a[[4, 5, 6, 3]])
    rb = Rotation.from_quat(b[[4, 5, 6, 3]])
    return float(np.linalg.norm(a[:3] - b[:3])), float((ra.inv() * rb).magnitude())
