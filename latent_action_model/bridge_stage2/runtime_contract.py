"""Strict runtime assets for phased Bridge Stage-2 training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from latent_action_model.action_ae.beta_vae import StandardNormalActionVAE
from latent_action_model.action_ae.evaluate import load_action_ae_checkpoint
from latent_action_model.libero_stage2.runtime_contract import (
    Stage2LatentAffine,
    standardized_action_logvar_bounds,
)

from .data import sha256_file
from .export_teacher_cache import ACTION_SCHEMA_VERSION, BUNDLE_SCHEMA_VERSION
from .temporal import (
    BRIDGE_STAGE2_SPLIT_ROWS,
    DEFAULT_TEMPORAL_CONTRACT,
    BridgeStage2TemporalContract,
    resolve_bridge_stage2_temporal_contract,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _latent_affine(payload: Mapping[str, Any], *, name: str) -> tuple[Tensor, Tensor]:
    center = torch.as_tensor(payload.get("mean"), dtype=torch.float32).reshape(-1)
    scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32).reshape(-1)
    if center.shape != (32,) or scale.shape != center.shape:
        raise ValueError(f"Bridge {name} mean/scale must both have shape [32]")
    if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
        raise ValueError(f"Bridge {name} mean/scale contain NaN/Inf")
    if bool((scale <= 0).any()):
        raise ValueError(f"Bridge {name} scale must be positive")
    return center, scale


@dataclass(frozen=True)
class BridgeStage2LatentContract:
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
    temporal_contract: str
    row_count: int
    cache_schema_version: str


def load_bridge_stage2_latent_contract(
    cache_root: str | Path,
    *,
    temporal_contract: str | BridgeStage2TemporalContract | None = None,
    expected_action_checkpoint_sha256: str | None = None,
    expected_effect_checkpoint_sha256: str | None = None,
) -> BridgeStage2LatentContract:
    temporal = resolve_bridge_stage2_temporal_contract(temporal_contract)
    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required_files = (
        "_SUCCESS",
        "checksums.sha256",
        "teacher_bundle.json",
        "latent_normalization.json",
        "paired_latents.npz",
    )
    for name in required_files:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    checksums_path = root / "checksums.sha256"
    if (root / "_SUCCESS").read_text(encoding="ascii").strip().lower() != sha256_file(
        checksums_path
    ):
        raise ValueError("Bridge cache _SUCCESS does not bind checksums.sha256")
    checksum_records: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        cells = line.split()
        if len(cells) != 2 or cells[1] in checksum_records:
            raise ValueError("Bridge cache checksums.sha256 is invalid")
        checksum_records[cells[1]] = cells[0].lower()
    expected_names = {
        "paired_latents.npz",
        "latent_normalization.json",
        "teacher_bundle.json",
    }
    if set(checksum_records) != expected_names:
        raise ValueError("Bridge cache checksum file set mismatch")
    actual_checksums = {name: sha256_file(root / name) for name in expected_names}
    if checksum_records != actual_checksums:
        raise ValueError("Bridge cache checksum content mismatch")

    bundle = _read_json(root / "teacher_bundle.json")
    normalization_path = root / "latent_normalization.json"
    normalization = _read_json(normalization_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Bridge teacher bundle schema mismatch")
    if bundle.get("temporal_contract") != temporal.to_dict():
        raise ValueError("Bridge teacher bundle temporal contract mismatch")
    cache = bundle.get("cache")
    if not isinstance(cache, Mapping) or int(cache.get("rows", -1)) != temporal.row_count:
        raise ValueError(f"Bridge cache must contain exactly {temporal.row_count} rows")
    if cache.get("split_rows") != BRIDGE_STAGE2_SPLIT_ROWS:
        raise ValueError("Bridge teacher bundle split counts mismatch")
    if cache.get("sha256") != actual_checksums["paired_latents.npz"]:
        raise ValueError("Bridge teacher bundle paired cache SHA256 mismatch")
    effect = bundle.get("effect")
    action = bundle.get("action")
    if not isinstance(effect, Mapping) or not isinstance(action, Mapping):
        raise ValueError("Bridge teacher bundle is missing Effect/Action provenance")
    if (
        expected_effect_checkpoint_sha256 is not None
        and effect.get("checkpoint_sha256") != expected_effect_checkpoint_sha256
    ):
        raise ValueError("Bridge cache selected the wrong Effect checkpoint")
    if (
        expected_action_checkpoint_sha256 is not None
        and action.get("checkpoint_sha256") != expected_action_checkpoint_sha256
    ):
        raise ValueError("Bridge cache selected the wrong ActionVAE checkpoint")
    if action.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("Bridge cache action schema mismatch")
    if action.get("distribution", {}).get("prior") != "standard_normal":
        raise ValueError("Bridge cache must use a standard-normal ActionVAE")

    if normalization.get("schema_version") != temporal.normalization_schema_version:
        raise ValueError("Bridge latent normalization schema mismatch")
    if (
        normalization.get("fit_split") != "train"
        or int(normalization.get("sample_count", -1)) != BRIDGE_STAGE2_SPLIT_ROWS["train"]
    ):
        raise ValueError("Bridge latent normalization must use the exact train split")
    effect_center, effect_scale = _latent_affine(normalization.get("effect", {}), name="effect")
    action_center, action_scale = _latent_affine(normalization.get("action", {}), name="action")

    cache_path = root / "paired_latents.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        expected_shapes = {
            "sample_id": (temporal.row_count,),
            "split": (temporal.row_count,),
            "z_effect_raw": (temporal.row_count, 32),
            "z_effect": (temporal.row_count, 32),
            "z_action_raw": (temporal.row_count, 32),
            "z_action": (temporal.row_count, 32),
            "z_action_logvar_raw": (temporal.row_count, 32),
            "z_action_logvar": (temporal.row_count, 32),
            "teacher_trajectory": (temporal.row_count, 24, 7),
            "teacher_current_state": (temporal.row_count, 14),
            "teacher_time_mask": (temporal.row_count, 24),
            "teacher_pose_mask": (temporal.row_count, 24),
            "teacher_gripper_mask": (temporal.row_count, 24),
            "teacher_current_state_mask": (temporal.row_count, 14),
            "physical_valid_count": (temporal.row_count,),
        }
        mismatches = {
            name: tuple(payload[name].shape) if name in payload.files else None
            for name, shape in expected_shapes.items()
            if name not in payload.files or tuple(payload[name].shape) != shape
        }
        if mismatches:
            raise ValueError(f"Bridge paired cache shape mismatch: {mismatches}")
        splits = np.asarray(payload["split"]).astype(str)
        observed_split_rows = {
            split: int(np.count_nonzero(splits == split))
            for split in BRIDGE_STAGE2_SPLIT_ROWS
        }
        if observed_split_rows != BRIDGE_STAGE2_SPLIT_ROWS:
            raise ValueError("Bridge paired cache split row counts mismatch")
        sample_ids = np.asarray(payload["sample_id"]).astype(str)
        if len(set(sample_ids.tolist())) != temporal.row_count:
            raise ValueError("Bridge paired cache sample IDs must be globally unique")
        if not np.asarray(payload["teacher_time_mask"]).astype(bool).all():
            raise ValueError("Bridge paired cache must keep all 24 representation slots")
        if not np.all(
            np.asarray(payload["physical_valid_count"]) == temporal.physical_valid_count
        ):
            raise ValueError("Bridge paired cache physical_valid_count must be six")
        for name in (
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        ):
            if not np.isin(payload[name], (0, 1)).all():
                raise ValueError(f"Bridge paired cache {name} must be binary")
        expected_state_mask = np.zeros((temporal.row_count, 14), dtype=bool)
        expected_state_mask[:, :7] = True
        if not np.array_equal(
            np.asarray(payload["teacher_current_state_mask"], dtype=bool),
            expected_state_mask,
        ):
            raise ValueError("Bridge paired cache current-state mask must be true7/false7")
        for name in (
            "z_effect_raw",
            "z_effect",
            "z_action_raw",
            "z_action",
            "z_action_logvar_raw",
            "z_action_logvar",
            "teacher_trajectory",
            "teacher_current_state",
        ):
            if not np.isfinite(payload[name]).all():
                raise ValueError(f"Bridge paired cache {name} contains NaN/Inf")
        observed_logvar_min = np.min(np.asarray(payload["z_action_logvar"]), axis=0)
        observed_logvar_max = np.max(np.asarray(payload["z_action_logvar"]), axis=0)

    checkpoint_path = Path(str(action.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != action.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Bridge ActionVAE checkpoint is missing or changed")
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
        or metadata.get("schema_version") != ACTION_SCHEMA_VERSION
        or model_config.get("variant") != "a1"
        or int(model_config.get("horizon", -1)) != 24
        or int(model_config.get("trajectory_dim", -1)) != 7
        or int(model_config.get("state_dim", -1)) != 14
        or int(model_config.get("latent_dim", -1)) != 32
        or metadata.get("latent_distribution", {}).get("prior") != "standard_normal"
    ):
        raise ValueError("selected Bridge ActionVAE checkpoint metadata is incompatible")
    raw_lower = float(model_config.get("logvar_min", math.nan))
    raw_upper = float(model_config.get("logvar_max", math.nan))
    logvar_lower, logvar_upper = standardized_action_logvar_bounds(
        action_scale,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
    )
    if np.any(observed_logvar_min < logvar_lower.numpy() - 1.0e-5) or np.any(
        observed_logvar_max > logvar_upper.numpy() + 1.0e-5
    ):
        raise ValueError("Bridge teacher logvar lies outside checkpoint-derived support")
    return BridgeStage2LatentContract(
        cache_root=root,
        effect_center=effect_center,
        effect_scale=effect_scale,
        action_center=action_center,
        action_scale=action_scale,
        action_logvar_lower=logvar_lower,
        action_logvar_upper=logvar_upper,
        action_raw_logvar_lower=raw_lower,
        action_raw_logvar_upper=raw_upper,
        normalization_sha256=sha256_file(normalization_path),
        temporal_contract=temporal.identifier,
        row_count=temporal.row_count,
        cache_schema_version=temporal.cache_schema_version,
    )


class FrozenBridgeActionVAEDecoder(nn.Module):
    """Selected Bridge A1 beta-VAE condition encoder and decoder."""

    def __init__(self, model: StandardNormalActionVAE) -> None:
        super().__init__()
        if model.config.variant != "a1" or (
            model.config.horizon,
            model.config.trajectory_dim,
            model.config.state_dim,
            model.config.latent_dim,
        ) != (24, 7, 14, 32):
            raise ValueError("frozen ActionVAE decoder is not Bridge A1 1.2s")
        self.condition_mlp = model.condition_mlp
        self.decoder = model.decoder
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenBridgeActionVAEDecoder":
        del mode
        super().train(False)
        return self

    def forward(self, z_action_raw: Tensor, current_state: Tensor, z_effect_raw: Tensor) -> Tensor:
        if z_action_raw.ndim != 2 or tuple(z_action_raw.shape[1:]) != (32,):
            raise ValueError("raw z_action must be [B,32]")
        condition = self.condition_mlp(current_state, z_effect_raw.detach())
        return self.decoder(z_action_raw.unsqueeze(1), condition)


def load_frozen_bridge_action_decoder(
    *,
    checkpoint_path: str | Path,
    action_data_root: str | Path,
    expected_checkpoint_sha256: str,
) -> FrozenBridgeActionVAEDecoder:
    loaded = load_action_ae_checkpoint(
        checkpoint_path,
        action_data_root,
        device=torch.device("cpu"),
    )
    if loaded.checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("frozen Bridge ActionVAE checkpoint SHA256 mismatch")
    if loaded.metadata.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("frozen Bridge decoder requires action-tcn-beta-vae-v1")
    if not isinstance(loaded.model, StandardNormalActionVAE):
        raise TypeError("frozen Bridge decoder requires StandardNormalActionVAE")
    return FrozenBridgeActionVAEDecoder(loaded.model)


__all__ = [
    "BridgeStage2LatentContract",
    "FrozenBridgeActionVAEDecoder",
    "Stage2LatentAffine",
    "load_bridge_stage2_latent_contract",
    "load_frozen_bridge_action_decoder",
]
