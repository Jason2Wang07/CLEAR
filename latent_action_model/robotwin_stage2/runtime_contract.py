"""Strict runtime assets for phased RoboTwin Stage-2 training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from latent_action_model.action_ae.beta_vae import StandardNormalActionVAE
from latent_action_model.libero_stage2.runtime_contract import (
    Stage2LatentAffine,
    standardized_action_logvar_bounds,
)
from latent_action_model.robotwin_action_ae.beta_vae_lightning import (
    RoboTwinActionBetaVAELightningModule,
)

from .export_teacher_cache import ACTION_SCHEMA_VERSION, BUNDLE_SCHEMA_VERSION
from .temporal import (
    DEFAULT_TEMPORAL_CONTRACT,
    RoboTwinStage2TemporalContract,
    resolve_robotwin_stage2_temporal_contract,
)


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return digest


def _latent_affine(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Tensor, Tensor]:
    center = torch.as_tensor(payload.get("mean"), dtype=torch.float32).reshape(-1)
    scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32).reshape(-1)
    if center.shape != (32,) or scale.shape != center.shape:
        raise ValueError(f"{name} mean/scale must both have shape [32]")
    if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
        raise ValueError(f"{name} mean/scale contain NaN/Inf")
    if bool((scale <= 0).any()):
        raise ValueError(f"{name} scale must be positive")
    return center, scale


@dataclass(frozen=True)
class RoboTwinStage2LatentContract:
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
    condition_source: str = "e39"
    condition_cache_name: str = "effect"


def load_robotwin_stage2_latent_contract(
    cache_root: str | Path,
    *,
    temporal_contract: str | RoboTwinStage2TemporalContract | None = None,
    expected_action_checkpoint_sha256: str | None = None,
    expected_effect_checkpoint_sha256: str | None = None,
    expected_condition_checkpoint_sha256: str | None = None,
) -> RoboTwinStage2LatentContract:
    temporal = resolve_robotwin_stage2_temporal_contract(temporal_contract)
    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in (
        "_SUCCESS",
        "checksums.sha256",
        "teacher_bundle.json",
        "latent_normalization.json",
        "paired_latents.npz",
    ):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)

    checksums_path = root / "checksums.sha256"
    success_digest = (root / "_SUCCESS").read_text(encoding="ascii").strip().lower()
    if success_digest != _sha256_file(checksums_path):
        raise ValueError("RoboTwin cache _SUCCESS does not bind checksums.sha256")
    checksum_records: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        cells = line.split()
        if len(cells) != 2:
            raise ValueError("RoboTwin cache checksums.sha256 has an invalid record")
        if cells[1] in checksum_records:
            raise ValueError("RoboTwin cache checksums.sha256 contains duplicate files")
        checksum_records[cells[1]] = _require_sha256(
            cells[0], name=f"cache checksum {cells[1]}"
        )
    expected_checksum_files = {
        "paired_latents.npz",
        "latent_normalization.json",
        "teacher_bundle.json",
    }
    if set(checksum_records) != expected_checksum_files:
        raise ValueError("RoboTwin cache checksum file set mismatch")

    bundle = _read_json(root / "teacher_bundle.json")
    normalization_path = root / "latent_normalization.json"
    normalization = _read_json(normalization_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("RoboTwin teacher bundle schema mismatch")
    if bundle.get("temporal_contract") != temporal.to_dict():
        raise ValueError("RoboTwin teacher bundle temporal contract mismatch")
    cache = bundle.get("cache")
    if not isinstance(cache, Mapping) or int(cache.get("rows", -1)) != temporal.row_count:
        raise ValueError(f"RoboTwin cache must contain exactly {temporal.row_count} rows")
    cache_path = root / "paired_latents.npz"
    if Path(str(cache.get("path", ""))).name != cache_path.name:
        raise ValueError("RoboTwin teacher bundle cache path mismatch")
    cache_sha256 = _sha256_file(cache_path)
    if cache.get("sha256") != cache_sha256:
        raise ValueError("RoboTwin paired cache SHA256 mismatch")
    checksum_actual = {
        "paired_latents.npz": cache_sha256,
        "latent_normalization.json": _sha256_file(normalization_path),
        "teacher_bundle.json": _sha256_file(root / "teacher_bundle.json"),
    }
    if checksum_records != checksum_actual:
        raise ValueError("RoboTwin cache checksums.sha256 content mismatch")

    action = bundle.get("action")
    condition = bundle.get("condition", bundle.get("effect"))
    if not isinstance(action, Mapping) or not isinstance(condition, Mapping):
        raise ValueError("RoboTwin teacher bundle is missing Action/condition provenance")
    action_sha = _require_sha256(action.get("checkpoint_sha256", ""), name="action checkpoint")
    condition_sha = _require_sha256(
        condition.get("checkpoint_sha256", ""), name="condition checkpoint"
    )
    condition_source = str(condition.get("source", "e39")).strip().lower()
    if condition_source in {"legacy_cache", "sidecar_e39_mean"}:
        condition_source = "e39"
    condition_cache_name = str(
        condition.get(
            "cache_name", "lawam" if condition_source == "lawam_idm" else "effect"
        )
    )
    if condition_source not in {"e39", "lawam_idm"}:
        raise ValueError(f"unsupported RoboTwin cache condition source: {condition_source!r}")
    if condition_cache_name not in {"effect", "lawam"}:
        raise ValueError("RoboTwin cache condition name must be effect or lawam")
    if expected_action_checkpoint_sha256 is not None and action_sha != _require_sha256(
        expected_action_checkpoint_sha256, name="expected action checkpoint"
    ):
        raise ValueError("RoboTwin cache selected the wrong ActionVAE checkpoint")
    if expected_effect_checkpoint_sha256 is not None and condition_sha != _require_sha256(
        expected_effect_checkpoint_sha256, name="expected effect checkpoint"
    ):
        raise ValueError("RoboTwin cache selected the wrong Effect checkpoint")
    if expected_condition_checkpoint_sha256 is not None and condition_sha != _require_sha256(
        expected_condition_checkpoint_sha256, name="expected condition checkpoint"
    ):
        raise ValueError("RoboTwin cache selected the wrong condition checkpoint")
    if action.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("RoboTwin cache action schema mismatch")
    if action.get("distribution", {}).get("prior") != "standard_normal":
        raise ValueError("RoboTwin cache must use a standard-normal ActionVAE")

    if normalization.get("schema_version") != temporal.normalization_schema_version:
        raise ValueError("RoboTwin latent normalization schema mismatch")
    if normalization.get("fit_split") != "train" or int(
        normalization.get("sample_count", -1)
    ) != int(cache.get("split_rows", {}).get("train", -2)):
        raise ValueError("RoboTwin latent normalization must be fit on the exact train split")
    normalization_source = str(normalization.get("condition_source", condition_source))
    normalization_name = str(
        normalization.get("condition_cache_name", condition_cache_name)
    )
    if normalization_source != condition_source or normalization_name != condition_cache_name:
        raise ValueError("RoboTwin latent normalization condition metadata mismatch")
    effect_center, effect_scale = _latent_affine(
        normalization.get(condition_cache_name, {}), name=condition_cache_name
    )
    action_center, action_scale = _latent_affine(normalization.get("action", {}), name="action")

    with np.load(cache_path, allow_pickle=False) as payload:
        expected_shapes = {
            "sample_id": (temporal.row_count,),
            "split": (temporal.row_count,),
            f"z_{condition_cache_name}": (temporal.row_count, 32),
            f"z_{condition_cache_name}_raw": (temporal.row_count, 32),
            "z_action": (temporal.row_count, 32),
            "z_action_logvar": (temporal.row_count, 32),
            "teacher_trajectory": (
                temporal.row_count,
                temporal.representation_steps,
                temporal.trajectory_dim,
            ),
            "teacher_current_state": (temporal.row_count, temporal.state_dim),
            "teacher_time_mask": (temporal.row_count, temporal.representation_steps),
            "teacher_pose_mask": (
                temporal.row_count,
                temporal.representation_steps,
                temporal.arm_count,
            ),
            "teacher_gripper_mask": (
                temporal.row_count,
                temporal.representation_steps,
                temporal.arm_count,
            ),
            "teacher_current_state_mask": (temporal.row_count, temporal.state_dim),
            "physical_valid_count": (temporal.row_count,),
        }
        mismatches = {
            name: tuple(payload[name].shape) if name in payload.files else None
            for name, shape in expected_shapes.items()
            if name not in payload.files or tuple(payload[name].shape) != shape
        }
        if mismatches:
            raise ValueError(f"RoboTwin paired cache shape mismatch: {mismatches}")
        action_logvar = np.asarray(payload["z_action_logvar"], dtype=np.float32)
        observed_logvar_min = np.min(action_logvar, axis=0)
        observed_logvar_max = np.max(action_logvar, axis=0)
        physical_count = np.asarray(payload["physical_valid_count"])
        if not np.all(physical_count == temporal.physical_valid_count):
            raise ValueError("RoboTwin paired cache physical_valid_count mismatch")
        if not np.asarray(payload["teacher_time_mask"]).astype(bool).all():
            raise ValueError("RoboTwin paired cache must keep all 36 time slots valid")
        for name in (
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        ):
            if not np.isin(payload[name], (0, 1)).all():
                raise ValueError(f"RoboTwin paired cache {name} must be binary")
        for name in (
            f"z_{condition_cache_name}",
            f"z_{condition_cache_name}_raw",
            "z_action",
            "z_action_logvar",
            "teacher_trajectory",
            "teacher_current_state",
        ):
            if not np.isfinite(payload[name]).all():
                raise ValueError(f"RoboTwin paired cache {name} contains NaN/Inf")
        sample_ids = np.asarray(payload["sample_id"]).astype(str)
        if len(set(sample_ids.tolist())) != temporal.row_count:
            raise ValueError("RoboTwin paired cache sample IDs must be globally unique")
        splits = np.asarray(payload["split"]).astype(str)
        observed_split_rows = {
            split: int(np.count_nonzero(splits == split))
            for split in ("train", "validation", "test")
        }
        if observed_split_rows != {
            str(name): int(value) for name, value in cache.get("split_rows", {}).items()
        } or int(sum(observed_split_rows.values())) != temporal.row_count:
            raise ValueError("RoboTwin paired cache split counts mismatch teacher bundle")

    checkpoint_path = Path(str(action.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != action_sha:
        raise ValueError("RoboTwin ActionVAE checkpoint is missing or changed")
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
        or int(model_config.get("horizon", -1)) != temporal.representation_steps
        or int(model_config.get("trajectory_dim", -1)) != temporal.trajectory_dim
        or int(model_config.get("state_dim", -1)) != temporal.state_dim
        or int(model_config.get("latent_dim", -1)) != 32
        or metadata.get("latent_distribution", {}).get("prior") != "standard_normal"
        or str(metadata.get("condition_source", "e39")) != condition_source
        or (
            condition_source == "lawam_idm"
            and metadata.get("condition_checkpoint_sha256") != condition_sha
        )
    ):
        raise ValueError("selected RoboTwin ActionVAE checkpoint metadata is incompatible")
    raw_lower = float(model_config.get("logvar_min", math.nan))
    raw_upper = float(model_config.get("logvar_max", math.nan))
    logvar_lower, logvar_upper = standardized_action_logvar_bounds(
        action_scale,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
    )
    lower_numpy = logvar_lower.cpu().numpy()
    upper_numpy = logvar_upper.cpu().numpy()
    if np.any(observed_logvar_min < lower_numpy - 1.0e-5) or np.any(
        observed_logvar_max > upper_numpy + 1.0e-5
    ):
        raise ValueError("RoboTwin teacher logvar lies outside checkpoint-derived support")

    return RoboTwinStage2LatentContract(
        cache_root=root,
        effect_center=effect_center,
        effect_scale=effect_scale,
        action_center=action_center,
        action_scale=action_scale,
        action_logvar_lower=logvar_lower,
        action_logvar_upper=logvar_upper,
        action_raw_logvar_lower=raw_lower,
        action_raw_logvar_upper=raw_upper,
        normalization_sha256=_sha256_file(normalization_path),
        temporal_contract=temporal.identifier,
        row_count=temporal.row_count,
        cache_schema_version=temporal.cache_schema_version,
        condition_source=condition_source,
        condition_cache_name=condition_cache_name,
    )


class FrozenRoboTwinActionVAEDecoder(nn.Module):
    """Selected RoboTwin A1 beta-VAE condition encoder and decoder."""

    def __init__(self, model: StandardNormalActionVAE) -> None:
        super().__init__()
        temporal = DEFAULT_TEMPORAL_CONTRACT
        if model.config.variant != "a1" or (
            model.config.horizon,
            model.config.trajectory_dim,
            model.config.state_dim,
            model.config.latent_dim,
        ) != (temporal.representation_steps, temporal.trajectory_dim, temporal.state_dim, 32):
            raise ValueError("frozen ActionVAE decoder is not RoboTwin A1 1.2s")
        self.condition_mlp = model.condition_mlp
        self.decoder = model.decoder
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenRoboTwinActionVAEDecoder":
        del mode
        super().train(False)
        return self

    def forward(
        self,
        z_action_raw: Tensor,
        current_state: Tensor,
        condition_raw: Tensor,
    ) -> Tensor:
        if z_action_raw.ndim != 2 or tuple(z_action_raw.shape[1:]) != (32,):
            raise ValueError("raw z_action must be [B,32]")
        condition = self.condition_mlp(current_state, condition_raw.detach())
        return self.decoder(z_action_raw.unsqueeze(1), condition)


def load_frozen_robotwin_action_decoder(
    *,
    checkpoint_path: str | Path,
    action_data_root: str | Path,
    expected_checkpoint_sha256: str,
    expected_condition_source: str | None = None,
) -> FrozenRoboTwinActionVAEDecoder:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = _require_sha256(
        expected_checkpoint_sha256, name="expected ActionVAE checkpoint"
    )
    if _sha256_file(path) != expected_sha:
        raise ValueError("frozen RoboTwin ActionVAE checkpoint SHA256 mismatch")
    normalization_path = (
        Path(action_data_root).expanduser().resolve() / "stats" / "normalization.json"
    )
    module = RoboTwinActionBetaVAELightningModule.load_from_checkpoint(
        path,
        map_location="cpu",
        normalization_path=normalization_path,
        condition_source=expected_condition_source,
    )
    if not isinstance(module.model, StandardNormalActionVAE):
        raise TypeError("frozen RoboTwin decoder requires StandardNormalActionVAE")
    if (
        expected_condition_source is not None
        and module.condition_source != str(expected_condition_source)
    ):
        raise ValueError("frozen RoboTwin decoder condition source mismatch")
    return FrozenRoboTwinActionVAEDecoder(module.model)


__all__ = [
    "FrozenRoboTwinActionVAEDecoder",
    "RoboTwinStage2LatentContract",
    "Stage2LatentAffine",
    "load_frozen_robotwin_action_decoder",
    "load_robotwin_stage2_latent_contract",
]
