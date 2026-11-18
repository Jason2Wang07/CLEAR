"""Train-only robust statistics for z-action targets and conditions."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def robust_group(values: np.ndarray, mask: np.ndarray, *, floor: float = 1e-6) -> dict:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if array.shape != valid.shape or array.ndim < 2:
        raise ValueError(f"values and mask must have equal rank>=2 shapes, got {array.shape} and {valid.shape}")
    width = array.shape[-1]
    flat_values = array.reshape(-1, width)
    flat_mask = valid.reshape(-1, width)
    median = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    count = flat_mask.sum(axis=0)
    for index in range(width):
        active = flat_values[flat_mask[:, index], index]
        if not active.size:
            continue
        median[index] = np.median(active)
        scale[index] = max(1.4826 * np.median(np.abs(active - median[index])), floor)
    return {"count": count.tolist(), "median": median.tolist(), "scale_mad": scale.tolist()}


def sidecar_normalization(
    rows: Iterable[Mapping],
    *,
    schema_version: str = "maniskill-z-action-normalization-v1",
) -> dict:
    train = [row for row in rows if str(row["split"]) == "train"]
    if not train:
        raise ValueError("cannot fit normalization without train rows")
    waypoint = np.stack([np.asarray(row["waypoints"], np.float32) for row in train])
    time = np.stack([np.asarray(row["time_mask"], bool) for row in train])
    pose = np.stack([np.asarray(row["pose_mask"], bool) for row in train])
    gripper = np.stack([np.asarray(row["gripper_mask"], bool) for row in train])
    state = np.stack([np.asarray(row["current_state"], np.float32) for row in train])
    state_mask = np.stack([np.asarray(row["current_state_mask"], bool) for row in train])
    pose_time = time & pose
    return {
        "schema_version": schema_version,
        "fit_split": "train",
        "estimator": "median_and_1.4826_mad",
        "sample_count": len(train),
        "position": robust_group(waypoint[..., :3], np.repeat(pose_time[..., None], 3, axis=-1)),
        "rotation_rotvec": robust_group(waypoint[..., 3:6], np.repeat(pose_time[..., None], 3, axis=-1)),
        "gripper": robust_group(waypoint[..., 6:7], gripper[..., None] & time[..., None]),
        "current_state": robust_group(state, state_mask),
    }
