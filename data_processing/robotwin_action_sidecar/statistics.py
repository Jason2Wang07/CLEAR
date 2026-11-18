"""Train-only robust normalization for RoboTwin dual-arm trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .schema import HORIZON, NORMALIZATION_SCHEMA_VERSION, STATE_DIM, TRAJECTORY_DIM


def robust_group(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    floor: float = 1.0e-8,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.ones(array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if array.shape != valid.shape or array.ndim < 2:
        raise ValueError(f"values/mask must have matching rank>=2 shapes, got {array.shape}/{valid.shape}")
    width = array.shape[-1]
    flat_values = array.reshape(-1, width)
    flat_mask = valid.reshape(-1, width)
    count = flat_mask.sum(axis=0)
    median = np.zeros(width, dtype=np.float64)
    mad_scale = np.zeros(width, dtype=np.float64)
    rms = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    for index in range(width):
        active = flat_values[flat_mask[:, index], index]
        if active.size == 0:
            continue
        if not np.isfinite(active).all():
            raise ValueError(f"normalization input contains NaN/Inf in dimension {index}")
        median[index] = np.median(active)
        centered = active - median[index]
        mad_scale[index] = 1.4826 * np.median(np.abs(centered))
        rms[index] = np.sqrt(np.mean(np.square(centered)))
        scale[index] = max(mad_scale[index], rms[index], float(floor))
    return {
        "count": count.tolist(),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "scale_mad": mad_scale.tolist(),
        "rms_about_median": rms.tolist(),
    }


def normalization_from_train_shards(
    root: str | Path,
    *,
    temporary_dir: str | Path,
    horizon: int = HORIZON,
    schema_version: str = NORMALIZATION_SCHEMA_VERSION,
) -> dict[str, Any]:
    release = Path(root)
    shards = sorted((release / "shards/train").glob("*.h5"))
    if not shards:
        raise ValueError("cannot fit normalization without train shards")
    sample_count = 0
    for path in shards:
        with h5py.File(path, "r") as handle:
            sample_count += int(handle["sample_id"].shape[0])
    workspace = Path(temporary_dir)
    workspace.mkdir(parents=True, exist_ok=False)
    waypoint_path = workspace / "waypoints.float32"
    state_path = workspace / "states.float32"
    waypoint = np.memmap(
        waypoint_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, int(horizon), TRAJECTORY_DIM),
    )
    state = np.memmap(state_path, mode="w+", dtype=np.float32, shape=(sample_count, STATE_DIM))
    cursor = 0
    try:
        for path in shards:
            with h5py.File(path, "r") as handle:
                count = int(handle["sample_id"].shape[0])
                waypoint[cursor : cursor + count] = handle["waypoints"][...]
                state[cursor : cursor + count] = handle["current_state"][...]
                if not np.all(handle["time_mask"][...] == 1):
                    raise ValueError(f"RoboTwin v1 train shard has a padded time mask: {path}")
                if not np.all(handle["arm_pose_mask"][...] == 1):
                    raise ValueError(f"RoboTwin v1 train shard has an invalid pose mask: {path}")
                if not np.all(handle["arm_gripper_mask"][...] == 1):
                    raise ValueError(f"RoboTwin v1 train shard has an invalid gripper mask: {path}")
            cursor += count
        waypoint.flush()
        state.flush()
        groups = {
            "left_position": robust_group(waypoint[..., 0:3]),
            "left_rotation_rotvec": robust_group(waypoint[..., 3:6]),
            "left_gripper": robust_group(waypoint[..., 6:7]),
            "right_position": robust_group(waypoint[..., 7:10]),
            "right_rotation_rotvec": robust_group(waypoint[..., 10:13]),
            "right_gripper": robust_group(waypoint[..., 13:14]),
            "current_state": robust_group(state),
        }
        return {
            "schema_version": schema_version,
            "fit_split": "train",
            "estimator": "median_and_max_1.4826_mad_rms_about_median",
            "scale_floor": 1.0e-8,
            "sample_count": sample_count,
            **groups,
        }
    finally:
        del waypoint
        del state
        waypoint_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        workspace.rmdir()


def write_normalization(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
