"""Bridge ActionVAE current-state contract shared by training and SIMPLER."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # SIMPLER's renderer environment only builds raw NumPy state.
    torch = None  # type: ignore[assignment]


BRIDGE_ACTIONVAE_RAW_STATE_DIM = 14
BRIDGE_ACTIONVAE_VALID_STATE_DIM = 7
ACTION_PREDICTOR_STATE_DIM = 32
BRIDGE_ACTIONVAE_STATE_SOURCE = "actionvae14_padded32"
BRIDGE_ACTIONVAE_NORMALIZATION_SCHEMA = (
    "bridge-z-action-normalization-v1-12s-resampled24"
)
BRIDGE_ACTIONVAE_NORMALIZATION_SHA256 = (
    "8957439e33177a720c7a0bdeaf9de84726be4f08931b0b234fa7c3d8d6f38f4c"
)
BRIDGE_GRIPPER_MIN_WIDTH_METERS = 0.02
BRIDGE_GRIPPER_TRAVEL_METERS = 0.05
_QUATERNION_PI_TIE_TOLERANCE = 1.0e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bridge_wxyz_quaternion_to_principal_rotvec(quaternion: Any) -> np.ndarray:
    """Convert a SAPIEN wxyz quaternion to a deterministic principal rotvec."""

    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Bridge TCP quaternion must be finite shape [4] in wxyz order")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Bridge TCP quaternion must have non-zero norm")
    value = value / norm
    if abs(float(value[0])) <= _QUATERNION_PI_TIE_TOLERANCE:
        axis_index = int(np.argmax(np.abs(value[1:]))) + 1
        if value[axis_index] < 0.0:
            value = -value
        value[0] = 0.0
    elif value[0] < 0.0:
        value = -value
    vector = value[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1.0e-12:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(value[0]))
    return (vector * (angle / vector_norm)).astype(np.float32)


def build_bridge_actionvae_raw_state(
    robot_base_tcp_position: Any,
    robot_base_tcp_quaternion_wxyz: Any,
    gripper_closedness: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw14=[base-relative TCP xyz, rotvec, aperture, zeros7]."""

    position = np.asarray(robot_base_tcp_position, dtype=np.float32).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Bridge robot-base-relative TCP position must be finite shape [3]")
    rotation = bridge_wxyz_quaternion_to_principal_rotvec(
        robot_base_tcp_quaternion_wxyz
    )
    closedness_values = np.asarray(gripper_closedness, dtype=np.float64).reshape(-1)
    if closedness_values.shape != (1,) or not np.isfinite(closedness_values).all():
        raise ValueError("Bridge gripper closedness must be one finite scalar")
    closedness = float(closedness_values[0])
    if closedness < 0.0 or closedness > 1.0:
        raise ValueError("Bridge gripper closedness must be in [0,1]")

    raw = np.zeros(BRIDGE_ACTIONVAE_RAW_STATE_DIM, dtype=np.float32)
    raw[:3] = position
    raw[3:6] = rotation
    raw[6] = BRIDGE_GRIPPER_MIN_WIDTH_METERS + BRIDGE_GRIPPER_TRAVEL_METERS * (
        1.0 - closedness
    )
    mask = np.zeros(BRIDGE_ACTIONVAE_RAW_STATE_DIM, dtype=np.bool_)
    mask[:BRIDGE_ACTIONVAE_VALID_STATE_DIM] = True
    return raw, mask


class BridgeActionVAE14Padded32StateAdapter:
    """Apply the selected Bridge train-only normalizer and zero-pad to 32D."""

    _GROUP_WIDTHS = {
        "position": 3,
        "rotation_rotvec": 3,
        "gripper": 1,
        "current_state": BRIDGE_ACTIONVAE_RAW_STATE_DIM,
    }

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        expected_sha256: str = BRIDGE_ACTIONVAE_NORMALIZATION_SHA256,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required to instantiate the Bridge ActionVAE state adapter"
            )
        self.normalization_path = Path(normalization_path).expanduser().resolve()
        actual_sha256 = _sha256_file(self.normalization_path)
        if actual_sha256 != str(expected_sha256).lower():
            raise ValueError(
                "Bridge actionvae14_padded32 normalization SHA256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        self.normalization_sha256 = actual_sha256
        metadata = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError("Bridge ActionVAE normalization root must be an object")
        self._validate_metadata(metadata)
        current = metadata["current_state"]
        self._median = torch.as_tensor(current["median"], dtype=torch.float32)
        self._scale = torch.as_tensor(current["scale_mad"], dtype=torch.float32)

    @classmethod
    def _validate_metadata(cls, metadata: Mapping[str, Any]) -> None:
        if metadata.get("schema_version") != BRIDGE_ACTIONVAE_NORMALIZATION_SCHEMA:
            raise ValueError("Bridge ActionVAE normalization schema mismatch")
        if metadata.get("fit_split") != "train":
            raise ValueError("Bridge ActionVAE normalization must be fit on train")
        if metadata.get("estimator") != "median_and_1.4826_mad":
            raise ValueError("Bridge ActionVAE normalization estimator mismatch")
        sample_count = int(metadata.get("sample_count", 0))
        if sample_count <= 0:
            raise ValueError("Bridge ActionVAE normalization sample_count must be positive")

        for name, width in cls._GROUP_WIDTHS.items():
            group = metadata.get(name)
            if not isinstance(group, Mapping):
                raise ValueError(f"Bridge ActionVAE normalization is missing group {name!r}")
            median = np.asarray(group.get("median"), dtype=np.float64)
            scale = np.asarray(group.get("scale_mad"), dtype=np.float64)
            count = np.asarray(group.get("count"), dtype=np.int64)
            if median.shape != (width,) or scale.shape != (width,) or count.shape != (width,):
                raise ValueError(
                    f"Bridge ActionVAE normalization group {name!r} must have width {width}"
                )
            if not np.isfinite(median).all() or not np.isfinite(scale).all():
                raise ValueError(
                    f"Bridge ActionVAE normalization group {name!r} contains NaN/Inf"
                )
            if np.any(scale <= 0) or np.any(count < 0):
                raise ValueError(
                    f"Bridge ActionVAE normalization group {name!r} has invalid scale/count"
                )

        current = metadata["current_state"]
        current_median = np.asarray(current["median"], dtype=np.float64)
        current_scale = np.asarray(current["scale_mad"], dtype=np.float64)
        current_count = np.asarray(current["count"], dtype=np.int64)
        if not np.all(current_count[:BRIDGE_ACTIONVAE_VALID_STATE_DIM] == sample_count):
            raise ValueError("Bridge ActionVAE valid current-state counts must equal sample_count")
        if not np.all(current_count[BRIDGE_ACTIONVAE_VALID_STATE_DIM:] == 0):
            raise ValueError("Bridge ActionVAE reserved current-state counts must be zero")
        if not np.all(current_median[BRIDGE_ACTIONVAE_VALID_STATE_DIM:] == 0.0):
            raise ValueError("Bridge ActionVAE reserved current-state medians must be zero")
        if not np.all(current_scale[BRIDGE_ACTIONVAE_VALID_STATE_DIM:] == 1.0):
            raise ValueError("Bridge ActionVAE reserved current-state scales must be one")

    def __call__(self, raw_state14: Any, current_state_mask14: Any) -> torch.Tensor:
        raw = torch.as_tensor(raw_state14, dtype=torch.float32).reshape(-1)
        mask_value = torch.as_tensor(current_state_mask14).reshape(-1)
        expected_shape = (BRIDGE_ACTIONVAE_RAW_STATE_DIM,)
        if tuple(raw.shape) != expected_shape or tuple(mask_value.shape) != expected_shape:
            raise ValueError("Bridge ActionVAE raw state and mask must both be shape [14]")
        if mask_value.dtype != torch.bool and not torch.all(
            (mask_value == 0) | (mask_value == 1)
        ):
            raise ValueError("Bridge ActionVAE current-state mask must be binary")
        mask = mask_value.to(dtype=torch.bool)
        canonical_mask = torch.zeros_like(mask)
        canonical_mask[:BRIDGE_ACTIONVAE_VALID_STATE_DIM] = True
        if not torch.equal(mask, canonical_mask):
            raise ValueError("Bridge ActionVAE current-state mask must be true7/false7")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Bridge ActionVAE raw state contains NaN/Inf")
        if not torch.equal(
            raw[BRIDGE_ACTIONVAE_VALID_STATE_DIM:],
            torch.zeros_like(raw[BRIDGE_ACTIONVAE_VALID_STATE_DIM:]),
        ):
            raise ValueError("Bridge ActionVAE reserved raw-state dimensions must be zero")
        if bool(raw[6] < BRIDGE_GRIPPER_MIN_WIDTH_METERS) or bool(
            raw[6] > BRIDGE_GRIPPER_MIN_WIDTH_METERS + BRIDGE_GRIPPER_TRAVEL_METERS
        ):
            raise ValueError("Bridge ActionVAE gripper aperture must be in [0.02,0.07] meters")

        median = self._median.to(device=raw.device)
        scale = self._scale.to(device=raw.device)
        normalized = torch.where(mask, (raw - median) / scale, torch.zeros_like(raw))
        padded = torch.zeros(
            ACTION_PREDICTOR_STATE_DIM, dtype=torch.float32, device=raw.device
        )
        padded[:BRIDGE_ACTIONVAE_RAW_STATE_DIM] = normalized
        return padded


__all__ = [
    "ACTION_PREDICTOR_STATE_DIM",
    "BRIDGE_ACTIONVAE_NORMALIZATION_SCHEMA",
    "BRIDGE_ACTIONVAE_NORMALIZATION_SHA256",
    "BRIDGE_ACTIONVAE_RAW_STATE_DIM",
    "BRIDGE_ACTIONVAE_STATE_SOURCE",
    "BRIDGE_ACTIONVAE_VALID_STATE_DIM",
    "BRIDGE_GRIPPER_MIN_WIDTH_METERS",
    "BRIDGE_GRIPPER_TRAVEL_METERS",
    "BridgeActionVAE14Padded32StateAdapter",
    "bridge_wxyz_quaternion_to_principal_rotvec",
    "build_bridge_actionvae_raw_state",
]
