"""Bridge WidowX pose, gripper, and 24-slot trajectory geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lawam_effect_dataset.geometry import delta_state14, relative_pose, state14
from z_action_sidecar.geometry import canonical_waypoints

from .schema import PHYSICAL_ACTION_STEPS, REPRESENTATION_STEPS


DEFAULT_ROTATION = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
PHASES_PER_SOURCE_STEP = REPRESENTATION_STEPS // PHYSICAL_ACTION_STEPS


@dataclass(frozen=True)
class BridgeWindowGeometry:
    poses: np.ndarray
    widths: np.ndarray
    waypoints: np.ndarray
    state14_start: np.ndarray
    state14_end: np.ndarray
    delta_state14: np.ndarray
    state_mask: np.ndarray
    effect_raw: np.ndarray
    effect_mask: np.ndarray
    effect_family: str
    teacher_compatibility: str
    effect_direction: str
    effect_magnitude: float


def validate_raw_states(raw_states: np.ndarray) -> np.ndarray:
    states = np.asarray(raw_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(f"Bridge states must be [T,8], got {states.shape}")
    if len(states) != PHYSICAL_ACTION_STEPS + 1:
        raise ValueError(
            f"Bridge window requires {PHYSICAL_ACTION_STEPS + 1} states, got {len(states)}"
        )
    if not np.isfinite(states).all():
        raise ValueError("Bridge states contain NaN/Inf")
    if not np.allclose(states[:, 6], 0.0, atol=0.0, rtol=0.0):
        raise ValueError("Bridge state pad dimension must be exactly zero")
    return states


def bridge_poses(raw_states: np.ndarray) -> np.ndarray:
    states = np.asarray(raw_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 8 or not np.isfinite(states).all():
        raise ValueError(f"Bridge states must be finite [T,8], got {states.shape}")
    source_rotations = Rotation.from_euler("xyz", states[:, 3:6])
    world_rotations = Rotation.from_matrix(source_rotations.as_matrix() @ DEFAULT_ROTATION)
    xyzw = world_rotations.as_quat()
    result = np.empty((len(states), 7), dtype=np.float64)
    result[:, :3] = states[:, :3]
    result[:, 3] = xyzw[:, 3]
    result[:, 4:] = xyzw[:, :3]
    return result


def bridge_aperture(raw_gripper: np.ndarray) -> np.ndarray:
    gripper = np.asarray(raw_gripper, dtype=np.float64)
    if not np.isfinite(gripper).all():
        raise ValueError("Bridge gripper contains NaN/Inf")
    return 0.020 + 0.050 * gripper


def _resampled_world_trajectory(poses: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    output_poses = np.empty((REPRESENTATION_STEPS, 7), dtype=np.float64)
    output_widths = np.empty(REPRESENTATION_STEPS, dtype=np.float64)
    output_index = 0
    for source_step in range(PHYSICAL_ACTION_STEPS):
        rotations = Rotation.from_quat(
            np.stack(
                (source_rotations[source_step].as_quat(), source_rotations[source_step + 1].as_quat())
            )
        )
        slerp = Slerp([0.0, 1.0], rotations)
        for phase in range(1, PHASES_PER_SOURCE_STEP + 1):
            fraction = phase / float(PHASES_PER_SOURCE_STEP)
            output_poses[output_index, :3] = poses[source_step, :3] + fraction * (
                poses[source_step + 1, :3] - poses[source_step, :3]
            )
            xyzw = slerp([fraction])[0].as_quat()
            output_poses[output_index, 3] = xyzw[3]
            output_poses[output_index, 4:] = xyzw[:3]
            output_widths[output_index] = (
                widths[source_step + 1]
                if phase == PHASES_PER_SOURCE_STEP
                else widths[source_step]
            )
            output_index += 1
    if output_index != REPRESENTATION_STEPS:
        raise AssertionError("Bridge trajectory resampler emitted the wrong slot count")
    return output_poses, output_widths


def _physical_labels(effect: np.ndarray, gripper_delta: float) -> tuple[str, str, str]:
    translation = np.asarray(effect[:3], dtype=np.float64)
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm < 0.005:
        direction = "still"
    else:
        axis = int(np.argmax(np.abs(translation)))
        direction = f"{'xyz'[axis]}{'p' if translation[axis] >= 0.0 else 'n'}"
    rotation_magnitude = float(np.linalg.norm(effect[3:6]))
    rotation_bin = "r0" if rotation_magnitude < 0.05 else ("r1" if rotation_magnitude < 0.25 else "r2")
    if abs(float(gripper_delta)) <= 0.005:
        gripper_mode = "hold"
    else:
        gripper_mode = "open" if gripper_delta > 0.0 else "close"
    family = f"{direction}-{rotation_bin}-{gripper_mode}"
    teacher_compatibility = f"{direction}-{gripper_mode}"
    return family, teacher_compatibility, direction


def bridge_window_geometry(raw_states: np.ndarray) -> BridgeWindowGeometry:
    states = validate_raw_states(raw_states)
    poses = bridge_poses(states)
    widths = bridge_aperture(states[:, 7])
    resampled_poses, resampled_widths = _resampled_world_trajectory(poses, widths)
    waypoints = canonical_waypoints(poses[0], resampled_poses, resampled_widths)
    start_state, state_mask = state14(poses[0], float(widths[0]))
    end_state, _ = state14(poses[-1], float(widths[-1]))
    state_delta = delta_state14(poses[0], poses[-1], float(widths[0]), float(widths[-1]))
    effect = np.zeros(12, dtype=np.float32)
    effect[:6] = relative_pose(poses[0], poses[-1])
    effect[6] = float(widths[-1] - widths[0])
    effect_mask = np.zeros(12, dtype=np.uint8)
    effect_mask[:7] = 1
    family, compatibility, direction = _physical_labels(effect, float(effect[6]))
    return BridgeWindowGeometry(
        poses=poses.astype(np.float32),
        widths=widths.astype(np.float32),
        waypoints=waypoints,
        state14_start=start_state,
        state14_end=end_state,
        delta_state14=state_delta,
        state_mask=state_mask,
        effect_raw=effect,
        effect_mask=effect_mask,
        effect_family=family,
        teacher_compatibility=compatibility,
        effect_direction=direction,
        effect_magnitude=float(np.linalg.norm(effect[:7])),
    )
