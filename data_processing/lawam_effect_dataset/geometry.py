"""Pose operations for ManiSkill's ``[xyz, qw, qx, qy, qz]`` convention."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_pose(pose: np.ndarray, *, name: str = "pose") -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite shape-(7,) pose")
    norm = float(np.linalg.norm(value[3:]))
    if norm < 1e-8:
        raise ValueError(f"{name} has a zero quaternion")
    value = value.copy()
    value[3:] /= norm
    return value


def rotation(pose: np.ndarray) -> Rotation:
    p = validate_pose(pose)
    return Rotation.from_quat([p[4], p[5], p[6], p[3]])


def relative_pose(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Return end relative to start as local translation plus rotation vector."""
    a, b = validate_pose(start, name="start"), validate_pose(end, name="end")
    r0, r1 = rotation(a), rotation(b)
    return np.concatenate((r0.inv().apply(b[:3] - a[:3]), (r0.inv() * r1).as_rotvec()))


def pose_in_frame(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Represent target pose in reference coordinates as ``[xyz, rotvec]``."""
    t, r = validate_pose(target, name="target"), validate_pose(reference, name="reference")
    rr = rotation(r)
    return np.concatenate((rr.inv().apply(t[:3] - r[:3]), (rr.inv() * rotation(t)).as_rotvec()))


def state14(eef_pose: np.ndarray, gripper: float) -> tuple[np.ndarray, np.ndarray]:
    pose = validate_pose(eef_pose, name="eef_pose")
    result = np.zeros(14, dtype=np.float32)
    result[:3] = pose[:3]
    result[3:6] = rotation(pose).as_rotvec()
    result[6] = float(gripper)
    mask = np.zeros(14, dtype=np.uint8)
    mask[:7] = 1
    return result, mask


def delta_state14(start_pose: np.ndarray, end_pose: np.ndarray, start_gripper: float, end_gripper: float) -> np.ndarray:
    result = np.zeros(14, dtype=np.float32)
    # World-frame positional delta is the contract for robot state; orientation is relative.
    a, b = validate_pose(start_pose), validate_pose(end_pose)
    result[:3] = b[:3] - a[:3]
    result[3:6] = (rotation(a).inv() * rotation(b)).as_rotvec()
    result[6] = float(end_gripper) - float(start_gripper)
    return result
