"""Shared LIBERO ActionVAE current-state contract for training and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from latent_action_model.libero_stage2.teacher_bundle import (
    LIBERO_12S_ACTION_NORMALIZATION_SHA256,
)


ACTIONVAE_RAW_STATE_DIM = 14
ACTION_PREDICTOR_STATE_DIM = 32
ACTIONVAE_VALID_STATE_DIM = 7
ACTIONVAE_STATE_SOURCE = "actionvae14_padded32"
ACTIONVAE_NORMALIZATION_SCHEMA = "libero-z-action-normalization-v1-12s"
PRINCIPAL_ROTVEC_NORM_TOLERANCE = 1e-5
_QUATERNION_PI_TIE_TOLERANCE = 1e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def libero_xyzw_quaternion_to_principal_rotvec(quaternion: Any) -> np.ndarray:
    """Convert a LIBERO xyzw quaternion to the principal rotation vector."""

    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("LIBERO end-effector quaternion must be finite shape [4] in xyzw order")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("LIBERO end-effector quaternion must have non-zero norm")
    value = value / norm
    if abs(float(value[3])) <= _QUATERNION_PI_TIE_TOLERANCE:
        # At exactly pi, q and -q have the same zero scalar component. Resolve
        # the remaining axis-sign ambiguity by making its largest component positive.
        axis_index = int(np.argmax(np.abs(value[:3])))
        if value[axis_index] < 0.0:
            value = -value
        value[3] = 0.0
    elif value[3] < 0.0:
        value = -value
    vector = value[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(value[3]))
    return (vector * (angle / vector_norm)).astype(np.float32)


def build_libero_actionvae_raw_state(
    world_position: Any,
    world_rotation_rotvec: Any,
    gripper_qpos: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw14=[world xyz, principal rotvec, physical width, zeros7]."""

    position = np.asarray(world_position, dtype=np.float32).reshape(-1)
    rotation = np.asarray(world_rotation_rotvec, dtype=np.float32).reshape(-1)
    fingers = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    if position.shape != (3,) or rotation.shape != (3,):
        raise ValueError(
            "LIBERO ActionVAE state requires world position/rotvec with shape [3]"
        )
    if fingers.shape != (2,):
        raise ValueError(
            "LIBERO ActionVAE state requires exactly two finger qpos values, "
            f"got shape {fingers.shape}"
        )
    if not all(np.isfinite(value).all() for value in (position, rotation, fingers)):
        raise ValueError("LIBERO ActionVAE raw state inputs must be finite")
    rotation_norm = float(np.linalg.norm(rotation.astype(np.float64)))
    if rotation_norm > np.pi + PRINCIPAL_ROTVEC_NORM_TOLERANCE:
        raise ValueError(
            "LIBERO ActionVAE rotation must be a principal rotvec with norm <= pi"
        )
    if rotation_norm > np.pi:
        # Float32 conversion can put an exact-pi vector a few ulps above pi.
        # Treat only that tolerated boundary as the pi sign tie; larger inputs fail above.
        axis_index = int(np.argmax(np.abs(rotation)))
        if rotation[axis_index] < 0.0:
            rotation = -rotation

    raw = np.zeros(ACTIONVAE_RAW_STATE_DIM, dtype=np.float32)
    raw[:3] = position
    raw[3:6] = rotation
    raw[6] = np.abs(fingers[0] - fingers[1])
    mask = np.zeros(ACTIONVAE_RAW_STATE_DIM, dtype=np.bool_)
    mask[:ACTIONVAE_VALID_STATE_DIM] = True
    return raw, mask


class ActionVAE14Padded32StateAdapter:
    """Apply the SHA-bound train-only robust normalization, then zero-pad to 32D."""

    _GROUP_WIDTHS = {
        "position": 3,
        "rotation_rotvec": 3,
        "gripper": 1,
        "current_state": ACTIONVAE_RAW_STATE_DIM,
    }

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        expected_sha256: str = LIBERO_12S_ACTION_NORMALIZATION_SHA256,
    ) -> None:
        self.normalization_path = Path(normalization_path).expanduser().resolve()
        self.normalization_sha256 = sha256_file(self.normalization_path)
        if self.normalization_sha256 != str(expected_sha256):
            raise ValueError(
                "actionvae14_padded32 requires ActionVAE normalization SHA256 "
                f"{expected_sha256}, got {self.normalization_sha256}"
            )
        metadata = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError("ActionVAE normalization root must be a JSON object")
        self._validate_metadata(metadata)
        current = metadata["current_state"]
        self._median = torch.as_tensor(current["median"], dtype=torch.float32)
        self._scale = torch.as_tensor(current["scale_mad"], dtype=torch.float32)

    @classmethod
    def _validate_metadata(cls, metadata: Mapping[str, Any]) -> None:
        if metadata.get("schema_version") != ACTIONVAE_NORMALIZATION_SCHEMA:
            raise ValueError(
                "ActionVAE state normalization must use schema "
                f"{ACTIONVAE_NORMALIZATION_SCHEMA!r}"
            )
        if metadata.get("fit_split") != "train":
            raise ValueError("ActionVAE state normalization must be fit on train")
        if metadata.get("estimator") != "median_and_1.4826_mad":
            raise ValueError("ActionVAE state normalization must use median/MAD")
        if int(metadata.get("sample_count", 0)) <= 0:
            raise ValueError("ActionVAE state normalization sample_count must be positive")
        for name, width in cls._GROUP_WIDTHS.items():
            group = metadata.get(name)
            if not isinstance(group, Mapping):
                raise ValueError(f"ActionVAE state normalization is missing group {name!r}")
            median = np.asarray(group.get("median"), dtype=np.float64)
            scale = np.asarray(group.get("scale_mad"), dtype=np.float64)
            count = np.asarray(group.get("count"), dtype=np.int64)
            if median.shape != (width,) or scale.shape != (width,) or count.shape != (width,):
                raise ValueError(f"ActionVAE normalization group {name!r} must have width {width}")
            if not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(
                    f"ActionVAE normalization group {name!r} must contain finite median/scale"
                )
            if np.any(count < 0):
                raise ValueError(f"ActionVAE normalization group {name!r} has negative count")

    @staticmethod
    def pad(normalized_state14: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(normalized_state14, dtype=torch.float32).reshape(-1)
        if state.shape != (ACTIONVAE_RAW_STATE_DIM,) or not torch.isfinite(state).all():
            raise ValueError("normalized ActionVAE state must be finite shape [14]")
        result = torch.zeros(
            ACTION_PREDICTOR_STATE_DIM,
            dtype=torch.float32,
            device=state.device,
        )
        result[:ACTIONVAE_RAW_STATE_DIM] = state
        return result

    def __call__(self, raw_state14: Any, current_state_mask14: Any) -> torch.Tensor:
        raw = torch.as_tensor(raw_state14, dtype=torch.float32).reshape(-1)
        mask_value = torch.as_tensor(current_state_mask14).reshape(-1)
        if raw.shape != (ACTIONVAE_RAW_STATE_DIM,) or mask_value.shape != (
            ACTIONVAE_RAW_STATE_DIM,
        ):
            raise ValueError("canonical ActionVAE raw state and mask must both have shape [14]")
        if mask_value.dtype != torch.bool and not torch.all(
            (mask_value == 0) | (mask_value == 1)
        ):
            raise ValueError("canonical ActionVAE current-state mask must be binary")
        mask = mask_value.to(dtype=torch.bool)
        canonical_mask = torch.zeros_like(mask)
        canonical_mask[:ACTIONVAE_VALID_STATE_DIM] = True
        if not torch.equal(mask, canonical_mask):
            raise ValueError("canonical ActionVAE current-state mask must be true7/false7")
        if torch.any(mask & ~torch.isfinite(raw)):
            raise ValueError("canonical ActionVAE raw state contains non-finite valid values")
        if not torch.equal(
            raw[ACTIONVAE_VALID_STATE_DIM:],
            torch.zeros_like(raw[ACTIONVAE_VALID_STATE_DIM:]),
        ):
            raise ValueError("canonical ActionVAE reserved raw-state dimensions must be zero")
        if bool(raw[6] < 0):
            raise ValueError("canonical ActionVAE physical gripper width must be non-negative")
        median = self._median.to(device=raw.device)
        scale = self._scale.to(device=raw.device)
        normalized = torch.where(mask, (raw - median) / scale, torch.zeros_like(raw))
        return self.pad(normalized)
