"""Adapt one 0.4-second LIBERO state window to the frozen 1.2s ActionVAE shape."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .state_transform import (
    build_libero_actionvae_raw_state,
    libero_xyzw_quaternion_to_principal_rotvec,
)


LIBERO_RAW_STATE_DIM = 8
LIBERO_04S_CONTROL_STEPS = 8
ACTIONVAE_12S_REPRESENTATION_STEPS = 24
PHASES_PER_CONTROL_STEP = (
    ACTIONVAE_12S_REPRESENTATION_STEPS // LIBERO_04S_CONTROL_STEPS
)
CANONICAL_WAYPOINT_DIM = 7


@dataclass(frozen=True)
class TemporalAdapter04sOutput:
    """One shape-adapted trajectory and its distinct representation/physical contracts."""

    waypoints: np.ndarray
    representation_mask: np.ndarray
    physical_valid_count: int


def _validate_start_index(start_index: Any, *, state_count: int) -> int:
    if isinstance(start_index, (bool, np.bool_)):
        raise TypeError("start_index must be an integer, not bool")
    try:
        index = operator.index(start_index)
    except TypeError as exc:
        raise TypeError("start_index must be an integer") from exc
    if not 0 <= index < state_count:
        raise IndexError(
            f"start_index must be in [0,{state_count}), got {index}"
        )
    return index


def _canonicalize_raw_states(raw_states: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        states = np.asarray(raw_states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_states must be a numeric array") from exc
    if states.ndim != 2 or states.shape[1:] != (LIBERO_RAW_STATE_DIM,):
        raise ValueError(
            "raw LIBERO states must have shape [T,8] with "
            "xyz+rotvec+two-finger-qpos layout"
        )
    if states.shape[0] < 1:
        raise ValueError("raw_states must contain at least one state")
    if not np.isfinite(states).all():
        raise ValueError("raw_states must contain only finite values")

    poses = np.empty((states.shape[0], 6), dtype=np.float64)
    widths = np.empty(states.shape[0], dtype=np.float64)
    for row_index, state in enumerate(states):
        try:
            principal_rotvec = libero_xyzw_quaternion_to_principal_rotvec(
                Rotation.from_rotvec(state[3:6]).as_quat()
            )
            raw14, _ = build_libero_actionvae_raw_state(
                state[:3],
                principal_rotvec,
                state[6:8],
            )
        except ValueError as exc:
            raise ValueError(f"invalid raw LIBERO state at row {row_index}: {exc}") from exc
        poses[row_index] = raw14[:6]
        widths[row_index] = raw14[6]
    return poses, widths


def adapt_libero_04s_states(
    raw_states: Any,
    start_index: int,
) -> TemporalAdapter04sOutput:
    """Build 24 post-phase canonical waypoints from eight 20Hz transitions.

    ``raw_states`` must contain exactly one episode in
    ``xyz + world rotvec + two finger qpos`` layout. LIBERO rotvecs are first
    mapped through their quaternion rotation equivalence to the canonical
    principal representation. Future indices are clamped to the episode tail.
    Each native control interval is represented by three phases: translation is
    piecewise linear, orientation follows the shortest quaternion SLERP arc,
    and gripper width uses zero-order hold before updating at the native
    endpoint.

    The returned 24 slots are all valid representation targets even when the
    source reaches the episode tail. ``physical_valid_count`` records the eight
    native 20Hz control intervals rather than the number of interpolated slots.
    """

    poses, widths = _canonicalize_raw_states(raw_states)
    start = _validate_start_index(start_index, state_count=int(poses.shape[0]))
    source_indices = np.minimum(
        start + np.arange(LIBERO_04S_CONTROL_STEPS + 1, dtype=np.int64),
        poses.shape[0] - 1,
    )
    source_poses = poses[source_indices]
    source_widths = widths[source_indices]

    source_rotations = Rotation.from_rotvec(source_poses[:, 3:6])
    start_position = source_poses[0, :3]
    start_rotation_inverse = source_rotations[0].inv()
    waypoints = np.empty(
        (ACTIONVAE_12S_REPRESENTATION_STEPS, CANONICAL_WAYPOINT_DIM),
        dtype=np.float32,
    )

    output_index = 0
    phase_denominator = float(PHASES_PER_CONTROL_STEP)
    for control_step in range(LIBERO_04S_CONTROL_STEPS):
        position_start = source_poses[control_step, :3]
        position_end = source_poses[control_step + 1, :3]
        rotation_pair = Rotation.from_quat(
            np.stack(
                (
                    source_rotations[control_step].as_quat(),
                    source_rotations[control_step + 1].as_quat(),
                )
            )
        )
        rotation_slerp = Slerp([0.0, 1.0], rotation_pair)

        for phase in range(1, PHASES_PER_CONTROL_STEP + 1):
            fraction = phase / phase_denominator
            world_position = position_start + fraction * (
                position_end - position_start
            )
            world_rotation = rotation_slerp([fraction])[0]
            relative_position = start_rotation_inverse.apply(
                world_position - start_position
            )
            relative_rotation = start_rotation_inverse * world_rotation
            relative_rotvec = libero_xyzw_quaternion_to_principal_rotvec(
                relative_rotation.as_quat()
            )
            gripper_width = (
                source_widths[control_step + 1]
                if phase == PHASES_PER_CONTROL_STEP
                else source_widths[control_step]
            )
            waypoints[output_index, :3] = relative_position
            waypoints[output_index, 3:6] = relative_rotvec
            waypoints[output_index, 6] = gripper_width
            output_index += 1

    if output_index != ACTIONVAE_12S_REPRESENTATION_STEPS:
        raise AssertionError("temporal adapter emitted the wrong waypoint count")
    if not np.isfinite(waypoints).all():
        raise ValueError("temporal adapter produced a non-finite waypoint")
    return TemporalAdapter04sOutput(
        waypoints=waypoints,
        representation_mask=np.ones(
            ACTIONVAE_12S_REPRESENTATION_STEPS, dtype=np.bool_
        ),
        physical_valid_count=LIBERO_04S_CONTROL_STEPS,
    )


class FrozenActionVAETemporalAdapter04s:
    """Stateless callable wrapper for cache and dataset integration."""

    def __call__(
        self,
        raw_states: Any,
        start_index: int,
    ) -> TemporalAdapter04sOutput:
        return adapt_libero_04s_states(raw_states, start_index)
