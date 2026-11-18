"""Explicit LIBERO command, controller-scale, and flow normalization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


NATIVE_ACTION_DIM = 7
FLOW_ACTION_DIM = 32
FLOW_FIELD_ORDER = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
RELEASE_STATISTICS_FIELD_ORDER = ("gripper", "x", "y", "z", "roll", "pitch", "yaw")
STATISTICS_TO_FLOW_PERMUTATION = (1, 2, 3, 4, 5, 6, 0)
NORMALIZATION_SCHEMA_VERSION = "libero-stage2-action-normalization-v1"
RELEASE_TRANSFORM_SCHEMA_VERSION = "libero-stage2-release-action-transform-v1"
LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256 = (
    "938b27fa9dd145dcf649821539e9a7b7dd0af23a0d1e46031452da2dde33d39a"
)


def _finite_vector(value: Any, *, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite shape-({width},) vector")
    return array


def release_statistics_to_flow(value: Any) -> np.ndarray:
    """Reorder a release statistics vector from gripper-first to flow order."""

    array = _finite_vector(value, name="release statistics", width=NATIVE_ACTION_DIM)
    return array[np.asarray(STATISTICS_TO_FLOW_PERMUTATION, dtype=np.int64)]


@dataclass(frozen=True)
class ReleaseLiberoActionTransform:
    """Release-pretrain min-max/binary transform in canonical flow field order."""

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = _finite_vector(self.minimum, name="minimum", width=NATIVE_ACTION_DIM)
        upper = _finite_vector(self.maximum, name="maximum", width=NATIVE_ACTION_DIM)
        if np.any(upper <= lower):
            raise ValueError("release action maximum must be greater than minimum")

    @classmethod
    def from_statistics(cls, action_statistics: Mapping[str, Any]) -> "ReleaseLiberoActionTransform":
        return cls(
            minimum=tuple(release_statistics_to_flow(action_statistics["min"]).tolist()),
            maximum=tuple(release_statistics_to_flow(action_statistics["max"]).tolist()),
        )

    def transform(self, actions: Any, *, clip: bool) -> np.ndarray:
        value = np.asarray(actions, dtype=np.float64)
        if value.shape[-1] != NATIVE_ACTION_DIM or not np.isfinite(value).all():
            raise ValueError("release actions must be finite with final dimension 7")
        lower = np.asarray(self.minimum)
        upper = np.asarray(self.maximum)
        continuous = 2.0 * (value[..., :6] - lower[:6]) / (upper[:6] - lower[:6]) - 1.0
        if clip:
            continuous = np.clip(continuous, -1.0, 1.0)
        gripper = (value[..., 6:7] > 0.5).astype(np.float64)
        return np.concatenate((continuous, gripper), axis=-1).astype(np.float32)

    def inverse(self, normalized: Any) -> np.ndarray:
        value = np.asarray(normalized, dtype=np.float64)
        if value.shape[-1] != NATIVE_ACTION_DIM or not np.isfinite(value).all():
            raise ValueError("normalized release actions must be finite with final dimension 7")
        lower = np.asarray(self.minimum)
        upper = np.asarray(self.maximum)
        continuous = 0.5 * (value[..., :6] + 1.0) * (upper[:6] - lower[:6]) + lower[:6]
        gripper = (value[..., 6:7] > 0.5).astype(np.float64)
        return np.concatenate((continuous, gripper), axis=-1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_TRANSFORM_SCHEMA_VERSION,
            "source_statistics_field_order": list(RELEASE_STATISTICS_FIELD_ORDER),
            "flow_field_order": list(FLOW_FIELD_ORDER),
            "statistics_to_flow_permutation": list(STATISTICS_TO_FLOW_PERMUTATION),
            "continuous_transform": "min_max_to_minus1_plus1",
            "gripper_transform": "value_gt_0.5_to_binary_0_1",
            "minimum_flow_order": list(self.minimum),
            "maximum_flow_order": list(self.maximum),
        }


@dataclass(frozen=True)
class ControllerScale:
    input_min: tuple[float, ...]
    input_max: tuple[float, ...]
    output_min: tuple[float, ...]
    output_max: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("input_min", "input_max", "output_min", "output_max"):
            _finite_vector(getattr(self, name), name=name, width=6)
        if np.any(np.asarray(self.input_max) <= np.asarray(self.input_min)):
            raise ValueError("controller input_max must be greater than input_min")
        if np.any(np.asarray(self.output_max) <= np.asarray(self.output_min)):
            raise ValueError("controller output_max must be greater than output_min")

    def command_to_physical(self, command: Any) -> np.ndarray:
        value = np.asarray(command, dtype=np.float64)
        if value.shape[-1] != 6 or not np.isfinite(value).all():
            raise ValueError("controller command must be finite with final dimension 6")
        input_min = np.asarray(self.input_min)
        input_span = np.asarray(self.input_max) - input_min
        output_min = np.asarray(self.output_min)
        output_span = np.asarray(self.output_max) - output_min
        return output_min + (value - input_min) * output_span / input_span

    def physical_to_command(self, physical: Any) -> np.ndarray:
        value = np.asarray(physical, dtype=np.float64)
        if value.shape[-1] != 6 or not np.isfinite(value).all():
            raise ValueError("controller physical delta must be finite with final dimension 6")
        input_min = np.asarray(self.input_min)
        input_span = np.asarray(self.input_max) - input_min
        output_min = np.asarray(self.output_min)
        output_span = np.asarray(self.output_max) - output_min
        return input_min + (value - output_min) * input_span / output_span


@dataclass(frozen=True)
class LiberoActionNormalizer:
    q01: tuple[float, ...]
    q99: tuple[float, ...]
    fit_split: str = "train"
    source_semantics: str = "official_normalized_osc_pose_command"

    def __post_init__(self) -> None:
        lower = _finite_vector(self.q01, name="q01", width=6)
        upper = _finite_vector(self.q99, name="q99", width=6)
        if self.fit_split != "train":
            raise ValueError("LIBERO Stage 2 action normalization must be fit on train")
        if np.any(upper <= lower):
            raise ValueError("each q99 value must be greater than q01")

    @classmethod
    def fit(
        cls,
        actions: Any,
        mask: Any,
        *,
        split: str,
    ) -> "LiberoActionNormalizer":
        if split != "train":
            raise ValueError("normalizer fitting is restricted to train")
        value = np.asarray(actions, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if value.shape != valid.shape or value.shape[-1] != NATIVE_ACTION_DIM:
            raise ValueError("actions and mask must have identical [...,7] shape")
        if not np.isfinite(value[valid]).all():
            raise ValueError("valid actions contain NaN/Inf")
        lowers = []
        uppers = []
        for dimension in range(6):
            selected = value[..., dimension][valid[..., dimension]]
            if selected.size == 0:
                raise ValueError(f"action dimension {dimension} has no train values")
            lower, upper = np.quantile(selected, (0.01, 0.99))
            if not np.isfinite((lower, upper)).all() or upper <= lower:
                raise ValueError(f"action dimension {dimension} has degenerate train range")
            lowers.append(float(lower))
            uppers.append(float(upper))
        return cls(q01=tuple(lowers), q99=tuple(uppers))

    def transform(
        self,
        actions: Any,
        mask: Any,
        *,
        clip: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        value = np.asarray(actions, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if value.shape != valid.shape or value.shape[-1] != NATIVE_ACTION_DIM:
            raise ValueError("actions and mask must have identical [...,7] shape")
        if not np.isfinite(value[valid]).all():
            raise ValueError("valid actions contain NaN/Inf")
        lower = np.asarray(self.q01)
        upper = np.asarray(self.q99)
        continuous = 2.0 * (value[..., :6] - lower) / (upper - lower) - 1.0
        below = continuous < -1.0
        above = continuous > 1.0
        if clip:
            continuous = np.clip(continuous, -1.0, 1.0)
        gripper = (value[..., 6:7] > 0.0).astype(np.float64)
        native = np.concatenate((continuous, gripper), axis=-1)
        native = np.where(valid, native, 0.0)
        padded_shape = (*native.shape[:-1], FLOW_ACTION_DIM)
        padded = np.zeros(padded_shape, dtype=np.float32)
        padded_mask = np.zeros(padded_shape, dtype=bool)
        padded[..., :NATIVE_ACTION_DIM] = native.astype(np.float32)
        padded_mask[..., :NATIVE_ACTION_DIM] = valid
        continuous_valid = valid[..., :6]
        report = {
            "clip": bool(clip),
            "below_q01": int(np.count_nonzero(below & continuous_valid)),
            "above_q99": int(np.count_nonzero(above & continuous_valid)),
            "valid_continuous_values": int(np.count_nonzero(continuous_valid)),
            "gripper_open": int(np.count_nonzero((value[..., 6] > 0.0) & valid[..., 6])),
            "gripper_closed": int(np.count_nonzero((value[..., 6] <= 0.0) & valid[..., 6])),
        }
        return padded, padded_mask, report

    def inverse(self, padded: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(padded, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if value.shape != valid.shape or value.shape[-1] != FLOW_ACTION_DIM:
            raise ValueError("padded actions and mask must have identical [...,32] shape")
        if np.any(valid[..., NATIVE_ACTION_DIM:]):
            raise ValueError("reserved flow dimensions must be masked")
        lower = np.asarray(self.q01)
        upper = np.asarray(self.q99)
        continuous = 0.5 * (value[..., :6] + 1.0) * (upper - lower) + lower
        gripper = np.where(value[..., 6:7] > 0.5, 1.0, -1.0)
        native = np.concatenate((continuous, gripper), axis=-1)
        native_mask = valid[..., :NATIVE_ACTION_DIM].copy()
        native = np.where(native_mask, native, 0.0)
        return native.astype(np.float32), native_mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "fit_split": self.fit_split,
            "source_semantics": self.source_semantics,
            "source_field_order": list(FLOW_FIELD_ORDER),
            "model_field_order": list(FLOW_FIELD_ORDER) + [f"reserved_{index}" for index in range(25)],
            "continuous_transform": "q01_q99_to_minus1_plus1",
            "gripper_transform": "command_gt_zero_to_binary_0_1",
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LiberoActionNormalizer":
        if value.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
            raise ValueError("LIBERO Stage 2 action normalization schema mismatch")
        return cls(
            q01=tuple(float(item) for item in value["q01"]),
            q99=tuple(float(item) for item in value["q99"]),
            fit_split=str(value["fit_split"]),
            source_semantics=str(value["source_semantics"]),
        )


def assert_field_order(fields: Sequence[str]) -> None:
    if tuple(fields) != FLOW_FIELD_ORDER:
        raise ValueError(f"action field order must be {FLOW_FIELD_ORDER}, got {tuple(fields)}")
