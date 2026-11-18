"""Strict runtime assets for phased LIBERO Stage-2 training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from latent_action_model.action_ae.beta_vae import StandardNormalActionVAE
from latent_action_model.action_ae.evaluate import load_action_ae_checkpoint

from .interventions import LATENT_NORMALIZATION_SCHEMA_VERSION
from . import robotwin_action_transfer as action_transfer
from .teacher_bundle import (
    LIBERO_12S_ACTION_CHECKPOINT_SHA256,
    LIBERO_12S_ACTION_SCHEMA_VERSION,
    LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
    LIBERO_12S_HORIZON_ACTIONS,
    LIBERO_12S_ROW_COUNT,
)


LATENT_CACHE_SCHEMA_VERSION = "libero-stage2-latent-cache-v2"
LATENT_CACHE_SCHEMA_VERSION_04S = "libero-stage2-latent-cache-v3"
LATENT_CACHE_SCHEMA_VERSION_ORIGINAL_PLUS_04S = "libero-stage2-latent-cache-v4"
LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION = (
    "libero-stage2-latent-normalization-bundle-v2"
)
LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_04S = (
    "libero-stage2-latent-normalization-bundle-v3"
)
LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_ORIGINAL_PLUS_04S = (
    "libero-stage2-latent-normalization-bundle-v4"
)
LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT = "libero_official_12s_v1"
LIBERO_BASELINE_04S_ADAPTER_TEMPORAL_CONTRACT = (
    "libero_baseline_04s_adapter_v1"
)
LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT = (
    "libero_original_plus_04s_adapter_v1"
)
LIBERO_ORIGINAL_PLUS_04S_ROW_COUNT = 273_465 + 2_238_036
SELECTED_ACTION_RAW_LOGVAR_LOWER = -6.0
SELECTED_ACTION_RAW_LOGVAR_UPPER = 2.0


@dataclass(frozen=True)
class LiberoStage2TemporalContract:
    identifier: str
    cache_schema_version: str
    normalization_schema_version: str
    row_count: int
    control_hz: float
    duration_seconds: float
    effect_teacher_delta_steps: int
    dino_future_delta_steps: int
    representation_steps: int
    physical_valid_count: int
    require_explicit_cache_marker: bool
    normalization_fit_membership: str | None = None
    trajectory_dim: int = 7
    state_dim: int = 14
    arm_count: int = 1


LIBERO_STAGE2_TEMPORAL_CONTRACTS: Mapping[str, LiberoStage2TemporalContract] = {
    LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT: LiberoStage2TemporalContract(
        identifier=LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT,
        cache_schema_version=LATENT_CACHE_SCHEMA_VERSION,
        normalization_schema_version=LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION,
        row_count=LIBERO_12S_ROW_COUNT,
        control_hz=20.0,
        duration_seconds=1.2,
        effect_teacher_delta_steps=24,
        dino_future_delta_steps=23,
        representation_steps=24,
        physical_valid_count=24,
        require_explicit_cache_marker=False,
    ),
    LIBERO_BASELINE_04S_ADAPTER_TEMPORAL_CONTRACT: LiberoStage2TemporalContract(
        identifier=LIBERO_BASELINE_04S_ADAPTER_TEMPORAL_CONTRACT,
        cache_schema_version=LATENT_CACHE_SCHEMA_VERSION_04S,
        normalization_schema_version=LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_04S,
        row_count=273_465,
        control_hz=20.0,
        duration_seconds=0.4,
        effect_teacher_delta_steps=8,
        dino_future_delta_steps=7,
        representation_steps=24,
        physical_valid_count=8,
        require_explicit_cache_marker=True,
        normalization_fit_membership="all_rows",
    ),
    LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT: LiberoStage2TemporalContract(
        identifier=LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT,
        cache_schema_version=LATENT_CACHE_SCHEMA_VERSION_ORIGINAL_PLUS_04S,
        normalization_schema_version=(
            LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_ORIGINAL_PLUS_04S
        ),
        row_count=LIBERO_ORIGINAL_PLUS_04S_ROW_COUNT,
        control_hz=20.0,
        duration_seconds=0.4,
        effect_teacher_delta_steps=8,
        dino_future_delta_steps=7,
        representation_steps=24,
        physical_valid_count=8,
        require_explicit_cache_marker=True,
        normalization_fit_membership="all_rows",
    ),
}


def is_libero_04s_adapter_temporal_contract(
    value: str | LiberoStage2TemporalContract | None,
) -> bool:
    identifier = getattr(value, "identifier", value)
    return str(identifier or "").strip() in {
        LIBERO_BASELINE_04S_ADAPTER_TEMPORAL_CONTRACT,
        LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT,
    }


def resolve_libero_stage2_temporal_contract(
    value: str | LiberoStage2TemporalContract | None,
) -> LiberoStage2TemporalContract:
    """Resolve the legacy default or one explicit Stage-2 temporal contract."""

    if isinstance(value, LiberoStage2TemporalContract):
        return value
    identifier = str(value or "").strip() or LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT
    try:
        return LIBERO_STAGE2_TEMPORAL_CONTRACTS[identifier]
    except KeyError as exc:
        raise ValueError(
            "unsupported stage2_temporal_contract "
            f"{identifier!r}; expected one of {sorted(LIBERO_STAGE2_TEMPORAL_CONTRACTS)}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _latent_affine(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Tensor, Tensor]:
    if payload.get("schema_version") != LATENT_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(f"{name} uses the wrong latent-standardization schema")
    if payload.get("fit_split") != "train" or int(payload.get("dimension", -1)) != 32:
        raise ValueError(f"{name} must be a train-fitted 32D standardizer")
    center = torch.as_tensor(payload.get("mean"), dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(payload.get("std"), dtype=torch.float32).reshape(-1)
    epsilon = float(payload.get("epsilon", 0.0))
    if center.shape != (32,) or std.shape != center.shape:
        raise ValueError(f"{name} mean/std must both have shape [32]")
    if not bool(torch.isfinite(center).all() and torch.isfinite(std).all()):
        raise ValueError(f"{name} mean/std contain NaN/Inf")
    if not math.isfinite(epsilon) or epsilon <= 0.0 or bool((std < 0).any()):
        raise ValueError(f"{name} requires non-negative std and positive epsilon")
    return center, std.clamp_min(epsilon)


@dataclass(frozen=True)
class LiberoStage2LatentContract:
    cache_root: Path
    effect_center: Tensor
    effect_scale: Tensor
    action_center: Tensor
    action_scale: Tensor
    action_logvar_lower: Tensor
    action_logvar_upper: Tensor
    action_raw_logvar_lower: float
    action_raw_logvar_upper: float
    normalization_sha256: str
    temporal_contract: str = LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT
    row_count: int = LIBERO_12S_ROW_COUNT
    cache_schema_version: str = LATENT_CACHE_SCHEMA_VERSION


def _validate_temporal_cache_metadata(
    payload: Mapping[str, Any],
    *,
    name: str,
    contract: LiberoStage2TemporalContract,
) -> None:
    marker = payload.get("stage2_temporal_contract")
    if contract.require_explicit_cache_marker:
        if marker != contract.identifier:
            raise ValueError(
                f"{name} stage2_temporal_contract must be {contract.identifier!r}, "
                f"got {marker!r}"
            )
    elif marker not in (None, "", contract.identifier):
        raise ValueError(
            f"{name} temporal marker {marker!r} is incompatible with {contract.identifier!r}"
        )

    if not contract.require_explicit_cache_marker:
        return
    if name != "provenance":
        return
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{name} has no temporal source contract")
    temporal = source.get("temporal_contract")
    if not isinstance(temporal, Mapping):
        raise ValueError(f"{name} has no structured temporal_contract")
    expected = {
        "control_hz": contract.control_hz,
        "duration_seconds": contract.duration_seconds,
        "source_action_steps": contract.physical_valid_count,
        "effect_teacher_delta_steps": contract.effect_teacher_delta_steps,
        "dino_future_delta_steps": contract.dino_future_delta_steps,
        "representation_steps": contract.representation_steps,
        "physical_valid_count": contract.physical_valid_count,
    }
    mismatches = {
        key: (temporal.get(key), expected_value)
        for key, expected_value in expected.items()
        if temporal.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{name} temporal contract mismatch: {mismatches}")


def _validate_latent_hdf5(
    root: Path,
    *,
    contract: LiberoStage2TemporalContract,
    expected_rows: int,
    expected_schema_version: str,
    action_teacher_transfer: str | None = None,
) -> None:
    path = root / "latents.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        action_transfer.check_marker(handle.attrs, action_teacher_transfer)
        schema = handle.attrs.get("schema_version")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        marker = handle.attrs.get("stage2_temporal_contract")
        if isinstance(marker, bytes):
            marker = marker.decode("utf-8")
        release_level = handle.attrs.get("release_level")
        if isinstance(release_level, bytes):
            release_level = release_level.decode("utf-8")
        if (
            schema != expected_schema_version
            or release_level != "full"
            or not bool(handle.attrs.get("complete", False))
        ):
            raise ValueError("latent-cache HDF5 is incomplete or uses the wrong schema")
        if contract.require_explicit_cache_marker and marker != contract.identifier:
            raise ValueError("latent-cache HDF5 temporal marker mismatch")
        if not contract.require_explicit_cache_marker and marker not in (
            None,
            "",
            contract.identifier,
        ):
            raise ValueError("legacy latent-cache HDF5 has an incompatible temporal marker")

        latent_arrays = (
            "z_effect",
            "z_action",
            "z_action_mean",
            "z_action_logvar",
        )
        bad_latents = {
            name: tuple(handle[name].shape) if name in handle else None
            for name in latent_arrays
            if name not in handle
            or tuple(handle[name].shape) != (expected_rows, 32)
            or handle[name].dtype != np.dtype(np.float32)
        }
        if bad_latents:
            raise ValueError(f"latent-cache Gaussian array mismatch: {bad_latents}")

        if not contract.require_explicit_cache_marker:
            return
        expected_shapes = {
            "teacher_trajectory": (expected_rows, contract.representation_steps, 7),
            "teacher_current_state": (expected_rows, 14),
            "teacher_time_mask": (expected_rows, contract.representation_steps),
            "teacher_pose_mask": (expected_rows, contract.representation_steps),
            "teacher_gripper_mask": (expected_rows, contract.representation_steps),
            "teacher_current_state_mask": (expected_rows, 14),
            "native_actions": (expected_rows, contract.physical_valid_count, 7),
            "native_action_mask": (expected_rows, contract.physical_valid_count, 7),
            "physical_valid_count": (expected_rows,),
            "effect_future_step": (expected_rows,),
            "dino_future_step": (expected_rows,),
        }
        bad_shapes = {
            name: tuple(handle[name].shape) if name in handle else None
            for name, expected_shape in expected_shapes.items()
            if name not in handle or tuple(handle[name].shape) != expected_shape
        }
        if bad_shapes:
            raise ValueError(f"0.4s adapter cache array mismatch: {bad_shapes}")
        physical_count = np.asarray(handle["physical_valid_count"][:])
        if not np.all(physical_count == contract.physical_valid_count):
            raise ValueError(
                "0.4s adapter cache physical_valid_count must be exactly "
                f"{contract.physical_valid_count} for every row"
            )


def standardized_action_logvar_bounds(
    action_scale: Tensor,
    *,
    raw_lower: float,
    raw_upper: float,
) -> tuple[Tensor, Tensor]:
    """Map raw diagonal-Gaussian support through z=(raw-center)/scale."""

    scale = torch.as_tensor(action_scale, dtype=torch.float32).reshape(-1)
    if scale.shape != (32,) or not bool(torch.isfinite(scale).all()):
        raise ValueError("action scale must be a finite [32] tensor")
    if bool((scale <= 0).any()):
        raise ValueError("action scale must be strictly positive")
    if not math.isfinite(raw_lower) or not math.isfinite(raw_upper) or raw_lower >= raw_upper:
        raise ValueError("raw action logvar bounds must be finite and ordered")
    shift = -2.0 * torch.log(scale)
    lower = torch.full_like(scale, float(raw_lower)) + shift
    upper = torch.full_like(scale, float(raw_upper)) + shift
    return lower, upper


def load_libero_stage2_latent_contract(
    cache_root: str | Path,
    *,
    temporal_contract: str | LiberoStage2TemporalContract | None = None,
    expected_rows: int | None = None,
    expected_schema_version: str | None = None,
    action_teacher_transfer: str | None = None,
) -> LiberoStage2LatentContract:
    """Load affine parameters and Gaussian bounds from one immutable full cache."""

    temporal = resolve_libero_stage2_temporal_contract(temporal_contract)
    transferred = action_transfer.checked_transfer(action_teacher_transfer)
    action_sha = action_transfer.CHECKPOINT_SHA256 if transferred else LIBERO_12S_ACTION_CHECKPOINT_SHA256
    condition_sha = action_transfer.CONDITION_SHA256 if transferred else LIBERO_12S_EFFECT_CHECKPOINT_SHA256
    action_schema = action_transfer.SCHEMA if transferred else LIBERO_12S_ACTION_SCHEMA_VERSION
    source_horizon = action_transfer.SOURCE_STEPS if transferred else LIBERO_12S_HORIZON_ACTIONS
    row_count = temporal.row_count if expected_rows is None else int(expected_rows)
    cache_schema = (
        temporal.cache_schema_version
        if expected_schema_version is None
        else str(expected_schema_version)
    )
    if row_count <= 0:
        raise ValueError("expected_rows must be positive")
    if cache_schema != temporal.cache_schema_version:
        raise ValueError(
            f"cache schema {cache_schema!r} is incompatible with temporal contract "
            f"{temporal.identifier!r}"
        )
    if row_count != temporal.row_count:
        raise ValueError(
            f"row count {row_count} is incompatible with temporal contract "
            f"{temporal.identifier!r} ({temporal.row_count})"
        )
    root = Path(cache_root).expanduser().resolve()
    success = _read_json(root / "_SUCCESS")
    provenance = _read_json(root / "provenance.json")
    normalization_path = root / "normalization.json"
    normalization = _read_json(normalization_path)
    expected_teachers = {
        "effect": condition_sha,
        "action": action_sha,
    }
    for name, payload in (("_SUCCESS", success), ("provenance", provenance)):
        action_transfer.check_marker(payload, action_teacher_transfer)
        if payload.get("schema_version") != cache_schema:
            raise ValueError(f"{name} does not use {cache_schema}")
        if payload.get("release_level") != "full":
            raise ValueError(f"{name} must describe a full latent-cache release")
        row_value = payload.get("row_count", payload.get("rows", -1))
        if int(row_value) != row_count:
            raise ValueError(
                f"{name} row count must be {row_count}, got {row_value}"
            )
        if payload.get("teacher_checkpoint_sha256") != expected_teachers:
            raise ValueError(f"{name} selected the wrong Effect/Action teachers")
        _validate_temporal_cache_metadata(
            payload,
            name=name,
            contract=temporal,
        )
    if success.get("all_gates_passed") is not True:
        raise ValueError("latent-cache _SUCCESS did not pass all gates")
    if provenance.get("ready") is not True or int(provenance.get("issue_count", -1)) != 0:
        raise ValueError("latent-cache provenance is not ready")

    if normalization.get("schema_version") != temporal.normalization_schema_version:
        raise ValueError("latent cache uses the wrong normalization-bundle schema")
    if normalization.get("fit_split") != "train":
        raise ValueError("latent normalization must be fit on train")
    if temporal.normalization_fit_membership is not None:
        if normalization.get("fit_membership") != temporal.normalization_fit_membership:
            raise ValueError(
                "latent normalization fit_membership mismatch: "
                f"expected={temporal.normalization_fit_membership!r}, "
                f"got={normalization.get('fit_membership')!r}"
            )
        if int(normalization.get("fit_row_count", -1)) != row_count:
            raise ValueError("latent normalization fit_row_count mismatch")
    effect_center, effect_scale = _latent_affine(
        normalization.get("z_effect", {}), name="z_effect"
    )
    action_center, action_scale = _latent_affine(
        normalization.get("z_action", {}), name="z_action"
    )

    actual_normalization_sha = _sha256_file(normalization_path)
    artifact = provenance.get("artifacts", {}).get("normalization", {})
    if artifact.get("path") != "normalization.json" or artifact.get(
        "sha256"
    ) != actual_normalization_sha:
        raise ValueError("latent normalization artifact SHA256 mismatch")

    _validate_latent_hdf5(
        root,
        contract=temporal,
        expected_rows=row_count,
        expected_schema_version=cache_schema,
        action_teacher_transfer=action_teacher_transfer,
    )

    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("latent-cache provenance has no source contract")
    if (
        source.get("action_checkpoint_sha256") != action_sha
        or source.get("action_checkpoint_schema_version")
        != action_schema
        or source.get("action_prior") != "standard_normal"
    ):
        raise ValueError("latent-cache provenance has the wrong ActionVAE contract")
    checkpoint_path = Path(str(source.get("action_checkpoint", ""))).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if transferred and checkpoint_path != action_transfer.CHECKPOINT.resolve():
        raise ValueError("transfer cache selected the wrong source checkpoint path")
    if _sha256_file(checkpoint_path) != action_sha:
        raise ValueError("latent-cache ActionVAE checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    metadata = checkpoint.get("action_ae_metadata") if isinstance(checkpoint, Mapping) else None
    model_config = metadata.get("model_config") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(model_config, Mapping)
        or metadata.get("schema_version") != action_schema
        or model_config.get("variant") != "a1"
        or int(model_config.get("latent_dim", -1)) != 32
        or int(model_config.get("horizon", -1)) != source_horizon
    ):
        raise ValueError("selected ActionVAE checkpoint metadata is incompatible")
    raw_lower = float(model_config.get("logvar_min", math.nan))
    raw_upper = float(model_config.get("logvar_max", math.nan))
    if (
        raw_lower != SELECTED_ACTION_RAW_LOGVAR_LOWER
        or raw_upper != SELECTED_ACTION_RAW_LOGVAR_UPPER
    ):
        raise ValueError("selected ActionVAE raw logvar bounds changed")
    logvar_lower, logvar_upper = standardized_action_logvar_bounds(
        action_scale,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
    )

    statistics = provenance.get("action_distribution", {}).get("teacher_statistics", {})
    declared_scale_min = float(statistics.get("action_scale_min", math.nan))
    declared_scale_max = float(statistics.get("action_scale_max", math.nan))
    if not math.isclose(declared_scale_min, float(action_scale.min()), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("latent-cache action_scale_min provenance mismatch")
    if not math.isclose(declared_scale_max, float(action_scale.max()), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("latent-cache action_scale_max provenance mismatch")
    observed_min = float(statistics.get("standardized_logvar_min", math.nan))
    observed_max = float(statistics.get("standardized_logvar_max", math.nan))
    if (
        not math.isfinite(observed_min)
        or not math.isfinite(observed_max)
        or observed_min < float(logvar_lower.min())
        or observed_max > float(logvar_upper.max())
    ):
        raise ValueError("observed teacher logvar lies outside the derived support")

    return LiberoStage2LatentContract(
        cache_root=root,
        effect_center=effect_center,
        effect_scale=effect_scale,
        action_center=action_center,
        action_scale=action_scale,
        action_logvar_lower=logvar_lower,
        action_logvar_upper=logvar_upper,
        action_raw_logvar_lower=raw_lower,
        action_raw_logvar_upper=raw_upper,
        normalization_sha256=actual_normalization_sha,
        temporal_contract=temporal.identifier,
        row_count=row_count,
        cache_schema_version=cache_schema,
    )


class Stage2LatentAffine(nn.Module):
    """Convert train-standardized condition/Action latents to and from teacher space."""

    def __init__(self, contract: LiberoStage2LatentContract) -> None:
        super().__init__()
        self.register_buffer("effect_center", contract.effect_center.clone())
        self.register_buffer("effect_scale", contract.effect_scale.clone())
        self.register_buffer("action_center", contract.action_center.clone())
        self.register_buffer("action_scale", contract.action_scale.clone())

    @staticmethod
    def _validate(value: Tensor, *, name: str) -> None:
        if value.ndim != 2 or tuple(value.shape[1:]) != (32,):
            raise ValueError(f"{name} must have shape [B,32]")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite floating-point values")

    def effect_to_raw(self, standardized: Tensor) -> Tensor:
        self._validate(standardized, name="standardized z_effect")
        return standardized * self.effect_scale + self.effect_center

    def condition_to_raw(self, standardized: Tensor) -> Tensor:
        self._validate(standardized, name="standardized condition latent")
        return standardized * self.effect_scale + self.effect_center

    def condition_to_standardized(self, raw: Tensor) -> Tensor:
        self._validate(raw, name="raw condition latent")
        return (raw - self.effect_center) / self.effect_scale

    def action_to_raw(self, standardized: Tensor) -> Tensor:
        self._validate(standardized, name="standardized z_action")
        return standardized * self.action_scale + self.action_center


class FrozenActionVAEDecoder(nn.Module):
    """The selected A1 beta-VAE condition encoder and trajectory decoder only."""

    def __init__(self, model: StandardNormalActionVAE) -> None:
        super().__init__()
        if model.config.variant != "a1" or (
            model.config.horizon,
            model.config.trajectory_dim,
            model.config.state_dim,
            model.config.latent_dim,
        ) != (LIBERO_12S_HORIZON_ACTIONS, 7, 14, 32):
            raise ValueError("frozen ActionVAE decoder architecture is not LIBERO A1 1.2s")
        self.condition_mlp = model.condition_mlp
        self.decoder = model.decoder
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenActionVAEDecoder":
        del mode
        super().train(False)
        return self

    def forward(
        self,
        z_action_raw: Tensor,
        current_state: Tensor,
        z_effect_raw: Tensor,
    ) -> Tensor:
        if z_action_raw.ndim != 2 or tuple(z_action_raw.shape[1:]) != (32,):
            raise ValueError("raw z_action must be [B,32]")
        condition = self.condition_mlp(current_state, z_effect_raw.detach())
        return self.decoder(z_action_raw.unsqueeze(1), condition)


def load_frozen_libero_action_decoder(
    *,
    checkpoint_path: str | Path,
    action_data_root: str | Path,
    action_teacher_transfer: str | None = None,
) -> FrozenActionVAEDecoder:
    if action_transfer.checked_transfer(action_teacher_transfer):
        return action_transfer.load_decoder(checkpoint_path, action_data_root)
    loaded = load_action_ae_checkpoint(
        checkpoint_path,
        action_data_root,
        device=torch.device("cpu"),
    )
    if loaded.checkpoint_sha256 != LIBERO_12S_ACTION_CHECKPOINT_SHA256:
        raise ValueError("frozen trajectory decoder selected the wrong ActionVAE checkpoint")
    if loaded.metadata.get("schema_version") != LIBERO_12S_ACTION_SCHEMA_VERSION:
        raise ValueError("frozen trajectory decoder requires action-tcn-beta-vae-v1")
    if not isinstance(loaded.model, StandardNormalActionVAE):
        raise TypeError("frozen trajectory decoder requires StandardNormalActionVAE")
    return FrozenActionVAEDecoder(loaded.model)
