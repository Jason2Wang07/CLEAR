"""RoboTwin ActionVAE current-state contract shared by training and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROBOTWIN_ACTIONVAE_STATE_SOURCE = "robotwin_actionvae14_padded32"
ROBOTWIN_RAW_EEF_STATE_DIM = 16
ROBOTWIN_ACTIONVAE_STATE_DIM = 14
ACTION_PREDICTOR_STATE_DIM = 32
ROBOTWIN_ACTIONVAE_NORMALIZATION_SCHEMA = "robotwin-z-action-normalization-v1-12s"
ROBOTWIN_ACTIONVAE_NORMALIZATION_SHA256 = (
    "255a793877156dc668fdb69b1e74f1bf2415a109ae287821f8f4b5f50e605896"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def robotwin_wxyz_quaternion_to_principal_rotvec(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("RoboTwin quaternion must be finite shape [4] in wxyz order")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("RoboTwin quaternion must have non-zero norm")
    value = value / norm
    if value[0] < 0.0:
        value = -value
    vector = value[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1.0e-12:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(value[0]))
    return (vector * (angle / vector_norm)).astype(np.float32)


def build_robotwin_actionvae_raw_state(raw_eef_state16: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert dual-arm raw16 wxyz EEF state to ActionVAE raw14 rotvec state."""

    raw = np.asarray(raw_eef_state16, dtype=np.float32).reshape(-1)
    if raw.shape != (ROBOTWIN_RAW_EEF_STATE_DIM,) or not np.isfinite(raw).all():
        raise ValueError("RoboTwin ActionVAE input must be finite raw16 EEF state")
    state = np.empty(ROBOTWIN_ACTIONVAE_STATE_DIM, dtype=np.float32)
    state[0:3] = raw[0:3]
    state[3:6] = robotwin_wxyz_quaternion_to_principal_rotvec(raw[3:7])
    state[6] = raw[7]
    state[7:10] = raw[8:11]
    state[10:13] = robotwin_wxyz_quaternion_to_principal_rotvec(raw[11:15])
    state[13] = raw[15]
    return state, np.ones(ROBOTWIN_ACTIONVAE_STATE_DIM, dtype=np.bool_)


class RoboTwinActionVAE14Padded32StateAdapter:
    """Apply the selected train-only robust normalizer and zero-pad to 32D."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        expected_sha256: str = ROBOTWIN_ACTIONVAE_NORMALIZATION_SHA256,
    ) -> None:
        self.normalization_path = Path(normalization_path).expanduser().resolve()
        actual_sha256 = _sha256_file(self.normalization_path)
        if actual_sha256 != str(expected_sha256).lower():
            raise ValueError(
                "robotwin_actionvae14_padded32 normalization SHA256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        metadata = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError("RoboTwin ActionVAE normalization root must be an object")
        if metadata.get("schema_version") != ROBOTWIN_ACTIONVAE_NORMALIZATION_SCHEMA:
            raise ValueError("RoboTwin ActionVAE normalization schema mismatch")
        if metadata.get("fit_split") != "train":
            raise ValueError("RoboTwin ActionVAE normalization must be fit on train")
        if metadata.get("estimator") != "median_and_max_1.4826_mad_rms_about_median":
            raise ValueError("RoboTwin ActionVAE normalization estimator mismatch")
        group = metadata.get("current_state")
        if not isinstance(group, Mapping):
            raise ValueError("RoboTwin ActionVAE normalization is missing current_state")
        self._median = torch.as_tensor(group.get("median"), dtype=torch.float32).reshape(-1)
        self._scale = torch.as_tensor(group.get("scale"), dtype=torch.float32).reshape(-1)
        count = torch.as_tensor(group.get("count"), dtype=torch.int64).reshape(-1)
        expected_shape = (ROBOTWIN_ACTIONVAE_STATE_DIM,)
        if (
            tuple(self._median.shape) != expected_shape
            or tuple(self._scale.shape) != expected_shape
            or tuple(count.shape) != expected_shape
        ):
            raise ValueError("RoboTwin current-state normalization must have width 14")
        if (
            not torch.isfinite(self._median).all()
            or not torch.isfinite(self._scale).all()
            or bool((self._scale <= 0).any())
            or bool((count <= 0).any())
        ):
            raise ValueError("RoboTwin current-state normalization contains invalid values")

    def __call__(self, raw_state14: Any, current_state_mask14: Any) -> torch.Tensor:
        raw = torch.as_tensor(raw_state14, dtype=torch.float32).reshape(-1)
        mask_value = torch.as_tensor(current_state_mask14).reshape(-1)
        expected_shape = (ROBOTWIN_ACTIONVAE_STATE_DIM,)
        if tuple(raw.shape) != expected_shape or tuple(mask_value.shape) != expected_shape:
            raise ValueError("RoboTwin ActionVAE raw state and mask must both be shape [14]")
        if mask_value.dtype != torch.bool and not torch.all(
            (mask_value == 0) | (mask_value == 1)
        ):
            raise ValueError("RoboTwin ActionVAE current-state mask must be binary")
        mask = mask_value.to(dtype=torch.bool)
        if not bool(mask.all()):
            raise ValueError("RoboTwin dual-arm ActionVAE current-state mask must be all valid")
        if not torch.isfinite(raw).all():
            raise ValueError("RoboTwin ActionVAE raw state contains NaN/Inf")
        normalized = (raw - self._median.to(raw.device)) / self._scale.to(raw.device)
        padded = torch.zeros(ACTION_PREDICTOR_STATE_DIM, dtype=torch.float32, device=raw.device)
        padded[:ROBOTWIN_ACTIONVAE_STATE_DIM] = normalized
        return padded
