from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .temporal_adapter_04s import (
    ACTIONVAE_12S_REPRESENTATION_STEPS,
    LIBERO_04S_CONTROL_STEPS,
    FrozenActionVAETemporalAdapter04s,
    adapt_libero_04s_states,
)


def _state(
    position: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    width: float,
) -> np.ndarray:
    return np.asarray(
        [*position, *rotvec, 0.5 * width, -0.5 * width],
        dtype=np.float64,
    )


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float((Rotation.from_rotvec(left).inv() * Rotation.from_rotvec(right)).magnitude())


def test_endpoint_and_start_frame_canonical_translation() -> None:
    start_rotation = (0.0, 0.0, np.pi / 2.0)
    states = np.stack(
        [
            _state((float(step), 0.0, 0.0), start_rotation, 0.08)
            for step in range(9)
        ]
    )

    result = FrozenActionVAETemporalAdapter04s()(states, 0)

    assert result.waypoints.shape == (ACTIONVAE_12S_REPRESENTATION_STEPS, 7)
    assert result.waypoints.dtype == np.float32
    assert result.representation_mask.dtype == np.bool_
    assert result.representation_mask.all()
    assert result.physical_valid_count == LIBERO_04S_CONTROL_STEPS
    np.testing.assert_allclose(result.waypoints[0, :3], [0.0, -1.0 / 3.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.waypoints[2, :3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.waypoints[-1, :3], [0.0, -8.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.waypoints[:, 3:6], 0.0, atol=1e-6)


def test_orientation_uses_shortest_path_quaternion_slerp() -> None:
    start_angle = np.deg2rad(170.0)
    end_angle = np.deg2rad(-170.0)
    states = np.stack(
        [
            _state((0.0, 0.0, 0.0), (0.0, 0.0, start_angle), 0.08),
            *[
                _state((0.0, 0.0, 0.0), (0.0, 0.0, end_angle), 0.08)
                for _ in range(8)
            ],
        ]
    )

    waypoints = adapt_libero_04s_states(states, 0).waypoints

    expected_angles = np.deg2rad([20.0 / 3.0, 40.0 / 3.0, 20.0])
    for output, expected in zip(waypoints[:3, 3:6], expected_angles):
        assert _rotation_distance(output, np.asarray([0.0, 0.0, expected])) < 1e-6
    assert np.all(waypoints[:3, 5] > 0.0)


def test_gripper_uses_zero_order_hold_and_updates_at_native_endpoint() -> None:
    widths = [0.08, 0.06, 0.04, 0.02, 0.01, 0.03, 0.05, 0.07, 0.09]
    states = np.stack(
        [
            _state((float(step), 0.0, 0.0), (0.0, 0.0, 0.0), width)
            for step, width in enumerate(widths)
        ]
    )

    gripper = adapt_libero_04s_states(states, 0).waypoints[:, 6]

    expected = []
    for width_start, width_end in zip(widths[:-1], widths[1:]):
        expected.extend((width_start, width_start, width_end))
    np.testing.assert_allclose(gripper, expected, atol=1e-7)


def test_nonprincipal_raw_libero_rotvec_is_canonicalized() -> None:
    # Published baseline states can lie slightly above pi due to their stored
    # axis-angle representation while still denoting a valid rotation.
    raw_rotation = np.asarray([3.1407702, 0.0017221, -0.0896675])
    assert np.linalg.norm(raw_rotation) > np.pi
    states = np.stack(
        [
            _state((float(step), 0.0, 0.0), tuple(raw_rotation), 0.08)
            for step in range(9)
        ]
    )

    waypoints = adapt_libero_04s_states(states, 0).waypoints

    np.testing.assert_allclose(waypoints[:, 3:6], 0.0, atol=1e-6)


def test_tail_clamp_holds_terminal_pose_and_width() -> None:
    states = np.stack(
        [
            _state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.08),
            _state((1.0, 0.0, 0.0), (0.0, 0.0, 0.2), 0.06),
            _state((2.0, 0.0, 0.0), (0.0, 0.0, 0.4), 0.04),
        ]
    )

    result = adapt_libero_04s_states(states, 1)

    terminal = result.waypoints[2]
    np.testing.assert_allclose(
        result.waypoints[3:],
        np.broadcast_to(terminal, result.waypoints[3:].shape),
        atol=1e-7,
    )
    expected_position = Rotation.from_rotvec([0.0, 0.0, 0.2]).inv().apply([1.0, 0.0, 0.0])
    np.testing.assert_allclose(terminal[:3], expected_position, atol=1e-7)
    assert _rotation_distance(terminal[3:6], np.asarray([0.0, 0.0, 0.2])) < 1e-6
    assert terminal[6] == pytest.approx(0.04)
    assert result.representation_mask.all()
    assert result.physical_valid_count == 8


@pytest.mark.parametrize(
    ("states", "start_index", "error_type", "message"),
    [
        (np.zeros((9, 7)), 0, ValueError, r"shape \[T,8\]"),
        (np.zeros((0, 8)), 0, ValueError, "at least one"),
        (np.full((9, 8), np.nan), 0, ValueError, "finite"),
        (np.zeros((9, 8)), -1, IndexError, "start_index"),
        (np.zeros((9, 8)), 9, IndexError, "start_index"),
        (np.zeros((9, 8)), 0.5, TypeError, "integer"),
        (np.zeros((9, 8)), True, TypeError, "not bool"),
    ],
)
def test_input_validation(
    states: np.ndarray,
    start_index: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        adapt_libero_04s_states(states, start_index)  # type: ignore[arg-type]
