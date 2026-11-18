"""Dual-arm RoboTwin state and image geometry."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.spatial.transform import Rotation


def normalize_pose_wxyz(raw: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [xyz,qw,qx,qy,qz,gripper]")
    quaternion = value[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError(f"{name} has a zero quaternion")
    result = value.copy()
    result[3:7] = quaternion / norm
    return result


def _rotation(pose: np.ndarray) -> Rotation:
    return Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])


def arm_state7(raw: np.ndarray, *, name: str) -> np.ndarray:
    pose = normalize_pose_wxyz(raw, name=name)
    result = np.empty(7, dtype=np.float32)
    result[:3] = pose[:3]
    result[3:6] = _rotation(pose).as_rotvec()
    result[6] = pose[7]
    return result


def arm_delta7(start: np.ndarray, end: np.ndarray, *, name: str) -> np.ndarray:
    first = normalize_pose_wxyz(start, name=f"{name}_start")
    last = normalize_pose_wxyz(end, name=f"{name}_end")
    result = np.empty(7, dtype=np.float32)
    result[:3] = last[:3] - first[:3]
    result[3:6] = (_rotation(first).inv() * _rotation(last)).as_rotvec()
    result[6] = last[7] - first[7]
    return result


def arm_effect6(start: np.ndarray, end: np.ndarray, *, name: str) -> np.ndarray:
    first = normalize_pose_wxyz(start, name=f"{name}_start")
    last = normalize_pose_wxyz(end, name=f"{name}_end")
    rotation = _rotation(first)
    return np.concatenate(
        (rotation.inv().apply(last[:3] - first[:3]), (rotation.inv() * _rotation(last)).as_rotvec())
    ).astype(np.float32)


def dual_arm_contract(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, ...]:
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    if first.shape != (16,) or last.shape != (16,):
        raise ValueError(f"dual-arm state must be shape (16,), got {first.shape}/{last.shape}")
    state_start = np.concatenate(
        (arm_state7(first[:8], name="left_start"), arm_state7(first[8:], name="right_start"))
    ).astype(np.float32)
    state_end = np.concatenate(
        (arm_state7(last[:8], name="left_end"), arm_state7(last[8:], name="right_end"))
    ).astype(np.float32)
    delta = np.concatenate(
        (arm_delta7(first[:8], last[:8], name="left"), arm_delta7(first[8:], last[8:], name="right"))
    ).astype(np.float32)
    effect = np.concatenate(
        (arm_effect6(first[:8], last[:8], name="left"), arm_effect6(first[8:], last[8:], name="right"))
    ).astype(np.float32)
    mask = np.ones(14, dtype=np.uint8)
    return state_start, state_end, delta, mask, effect


def effect_direction(delta_state14: np.ndarray) -> str:
    delta = np.asarray(delta_state14, dtype=np.float64)
    if delta.shape != (14,):
        raise ValueError("delta_state14 must have shape (14,)")
    left = float(np.linalg.norm(delta[:6]) + 0.1 * abs(delta[6]))
    right = float(np.linalg.norm(delta[7:13]) + 0.1 * abs(delta[13]))
    if max(left, right) < 1.0e-5:
        return "static"
    if left > 1.5 * max(right, 1.0e-8):
        return "left"
    if right > 1.5 * max(left, 1.0e-8):
        return "right"
    return "both"


def resize_cam_high_batch(frames: list[np.ndarray]) -> np.ndarray:
    """Match LaWAM RoboTwin max-4:3 crop and antialiased 256x256 resize."""

    if not frames:
        return np.empty((0, 3, 256, 256), dtype=np.uint8)
    array = np.stack(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"cam_high frames must be uint8 [N,H,W,3], got {array.dtype} {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).contiguous()
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    maximum_aspect = 4.0 / 3.0
    if width / height > maximum_aspect:
        crop_width = max(1, int(np.floor(height * maximum_aspect)))
        left = (width - crop_width) // 2
        tensor = tensor[..., left : left + crop_width]
    resized = functional.interpolate(
        tensor.float(), size=(256, 256), mode="bilinear", align_corners=False, antialias=True
    )
    return resized.round().clamp_(0, 255).to(torch.uint8).cpu().numpy()
