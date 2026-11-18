"""LIBERO pose and image conversion helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def poses_from_rotvec(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    xyz = np.asarray(position, dtype=np.float64)
    rotvec = np.asarray(orientation, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rotvec.shape != xyz.shape:
        raise ValueError(f"position/orientation must both be [T,3], got {xyz.shape}/{rotvec.shape}")
    if not np.isfinite(xyz).all() or not np.isfinite(rotvec).all():
        raise ValueError("LIBERO EEF pose contains NaN/Inf")
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.concatenate((xyz, xyzw[:, 3:4], xyzw[:, :3]), axis=1).astype(np.float32)


def gripper_width(gripper_states: np.ndarray) -> np.ndarray:
    states = np.asarray(gripper_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2 or not np.isfinite(states).all():
        raise ValueError(f"gripper_states must be finite [T,2], got {states.shape}")
    return np.abs(states[:, 0] - states[:, 1]).astype(np.float32)


def resize_agentview(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"agentview_rgb frame must be uint8 [128,128,3], got {rgb.dtype} {rgb.shape}")
    resized = Image.fromarray(rgb).resize((256, 256), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1))
