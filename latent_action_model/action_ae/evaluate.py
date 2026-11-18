"""Held-out physical and representation evaluation for ActionAE v0.

The evaluator keeps reconstruction interventions explicit: z-action
interventions are decoder-only, while A1 effect interventions are reported
both decoder-only and end-to-end. The terminal-state probe is fit on train,
selects Ridge regularization on validation, and reports test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import Tensor
from torch.utils.data import DataLoader

from latent_action_model.action_ae.data import (
    ActionSidecarDataset,
    ShardAwareBatchSampler,
    action_sidecar_collate,
)
from latent_action_model.action_ae.beta_vae import (
    ActionBetaVAEConfig,
    StandardNormalActionVAE,
)
from latent_action_model.action_ae.losses import so3_geodesic_error
from latent_action_model.action_ae.model import (
    ActionAEConfig,
    ActionAutoencoder,
    ActionVAEConfig,
    ActionVariationalAutoencoder,
)
from latent_action_model.stage1.evaluation import (
    covariance_and_effective_rank,
    pairwise_median_distance,
)


EVALUATION_SCHEMA_VERSION = "action-tcn-ae-v0-held-out-evaluation-v1"
DEFAULT_RIDGE_ALPHAS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)
_SIDECAR_HASH_FILES = {
    "normalization_sha256": Path("stats/normalization.json"),
    "sidecar_checksums_sha256": Path("checksums.sha256"),
    "sidecar_schema_sha256": Path("schema.json"),
    "sidecar_provenance_sha256": Path("provenance.json"),
}


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_string(value.item())
    return str(value)


@dataclass(frozen=True)
class EmbeddedTrajectoryNormalization:
    position_median: Tensor
    position_scale: Tensor
    rotation_median: Tensor
    rotation_scale: Tensor
    gripper_median: Tensor
    gripper_scale: Tensor

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> "EmbeddedTrajectoryNormalization":
        def tensor(group: str, field: str, width: int) -> Tensor:
            try:
                result = torch.as_tensor(payload[group][field], dtype=torch.float32)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"checkpoint normalization is missing {group}.{field}") from exc
            if result.shape != (width,) or not torch.isfinite(result).all():
                raise ValueError(f"checkpoint normalization {group}.{field} must be finite shape ({width},)")
            if field == "scale_mad" and torch.any(result <= 0):
                raise ValueError(f"checkpoint normalization {group}.{field} must be positive")
            return result

        return cls(
            position_median=tensor("position", "median", 3),
            position_scale=tensor("position", "scale_mad", 3),
            rotation_median=tensor("rotation_rotvec", "median", 3),
            rotation_scale=tensor("rotation_rotvec", "scale_mad", 3),
            gripper_median=tensor("gripper", "median", 1),
            gripper_scale=tensor("gripper", "scale_mad", 1),
        )

    def denormalize(self, trajectory: Tensor) -> Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 7:
            raise ValueError(f"trajectory must be [B,T,7], got {tuple(trajectory.shape)}")

        def like(value: Tensor) -> Tensor:
            return value.to(device=trajectory.device, dtype=trajectory.dtype)

        return torch.cat(
            (
                trajectory[..., :3] * like(self.position_scale) + like(self.position_median),
                trajectory[..., 3:6] * like(self.rotation_scale) + like(self.rotation_median),
                trajectory[..., 6:7] * like(self.gripper_scale) + like(self.gripper_median),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class LoadedActionAE:
    model: ActionAutoencoder | ActionVariationalAutoencoder | StandardNormalActionVAE
    normalization: EmbeddedTrajectoryNormalization
    checkpoint_path: Path
    checkpoint_sha256: str
    epoch: int | None
    global_step: int | None
    metadata: dict[str, Any]


def load_action_ae_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    *,
    device: torch.device,
    sidecar_binding: str = "exact",
    sidecar_reference_root: str | Path | None = None,
) -> LoadedActionAE:
    """Load model weights and reject checkpoint/sidecar provenance mismatches."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("action_ae_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing ActionAE metadata")
    schema_version = str(metadata.get("schema_version", ""))
    if schema_version not in {
        "action-tcn-ae-v0",
        "action-tcn-beta-vae-v1",
        "action-vae-v2",
    }:
        raise ValueError(f"unsupported ActionAE checkpoint schema: {schema_version!r}")

    if sidecar_binding not in {"exact", "semantic"}:
        raise ValueError(f"unsupported sidecar binding: {sidecar_binding!r}")
    if sidecar_binding == "semantic":
        if sidecar_reference_root is None:
            raise ValueError("semantic sidecar binding requires sidecar_reference_root")
        from latent_action_model.robotwin_stage2.sidecar_binding import (
            validate_semantic_sidecar_binding,
        )

        validate_semantic_sidecar_binding(
            data_root,
            sidecar_reference_root,
            action_metadata=metadata,
            verify_content_checksums=True,
        )
    else:
        for metadata_key, relative_path in _SIDECAR_HASH_FILES.items():
            expected = metadata.get(metadata_key)
            path = data_root / relative_path
            if expected is None:
                if metadata_key == "normalization_sha256":
                    raise ValueError("checkpoint is missing normalization_sha256")
                continue
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint provenance file does not exist: {path}")
            actual = _sha256_file(path)
            if str(expected).lower() != actual:
                raise ValueError(
                    f"checkpoint/sidecar provenance mismatch for {metadata_key}: "
                    f"checkpoint={expected}, sidecar={actual}"
                )

    raw_config = metadata.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint metadata is missing model_config")
    config_values = dict(raw_config)
    config_values["dilations"] = tuple(int(value) for value in config_values["dilations"])
    if schema_version == "action-vae-v2":
        config = ActionVAEConfig(**config_values)
        model: ActionAutoencoder | ActionVariationalAutoencoder | StandardNormalActionVAE = (
            ActionVariationalAutoencoder(config)
        )
    elif schema_version == "action-tcn-beta-vae-v1":
        config = ActionBetaVAEConfig(**config_values)
        model = StandardNormalActionVAE(config)
    else:
        config = ActionAEConfig(**config_values)
        model = ActionAutoencoder(config)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing state_dict")
    model_state = {
        name[len("model.") :]: value
        for name, value in state_dict.items()
        if str(name).startswith("model.")
    }
    if not model_state:
        raise ValueError("checkpoint state_dict does not contain model.* weights")
    model.load_state_dict(model_state, strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    normalization_payload = metadata.get("normalization")
    if not isinstance(normalization_payload, dict):
        raise ValueError("checkpoint metadata is missing embedded normalization")
    return LoadedActionAE(
        model=model,
        normalization=EmbeddedTrajectoryNormalization.from_metadata(normalization_payload),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        epoch=int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None,
        global_step=int(checkpoint["global_step"]) if checkpoint.get("global_step") is not None else None,
        metadata=dict(metadata),
    )


def _last_valid_mask(valid: Tensor) -> Tensor:
    if valid.ndim != 2:
        raise ValueError(f"valid mask must be [B,T], got {tuple(valid.shape)}")
    valid = valid.bool()
    result = torch.zeros_like(valid)
    has_valid = valid.any(dim=1)
    if torch.any(has_valid):
        reverse_index = valid[has_valid].flip(1).to(torch.int64).argmax(dim=1)
        index = valid.shape[1] - 1 - reverse_index
        rows = torch.nonzero(has_valid, as_tuple=False).flatten()
        result[rows, index] = True
    return result


def _masked_error_summary(errors: Tensor, valid: Tensor, *, mae_key: str, rmse_key: str) -> dict[str, Any]:
    valid = valid.to(device=errors.device, dtype=torch.bool)
    count = int(valid.sum().item())
    if count == 0:
        return {"valid_steps": 0, mae_key: None, rmse_key: None}
    selected = errors[valid].double()
    return {
        "valid_steps": count,
        mae_key: float(selected.mean()),
        rmse_key: float(selected.square().mean().sqrt()),
    }


def _region_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    pose_valid: Tensor,
    gripper_valid: Tensor,
) -> dict[str, Any]:
    safe_pose = pose_valid.unsqueeze(-1)
    predicted_position = torch.where(safe_pose, prediction[..., :3], torch.zeros_like(prediction[..., :3]))
    target_position = torch.where(safe_pose, target[..., :3], torch.zeros_like(target[..., :3]))
    position_l2_cm = torch.linalg.vector_norm(predicted_position - target_position, dim=-1) * 100.0

    predicted_rotation = torch.where(safe_pose, prediction[..., 3:6], torch.zeros_like(prediction[..., 3:6]))
    target_rotation = torch.where(safe_pose, target[..., 3:6], torch.zeros_like(target[..., 3:6]))
    rotation_deg = so3_geodesic_error(predicted_rotation, target_rotation, degrees=True)

    safe_gripper = gripper_valid
    gripper_error = torch.where(
        safe_gripper,
        (prediction[..., 6] - target[..., 6]).abs(),
        torch.zeros_like(prediction[..., 6]),
    )
    return {
        "position": _masked_error_summary(
            position_l2_cm,
            pose_valid,
            mae_key="l2_mae_cm",
            rmse_key="l2_rmse_cm",
        ),
        "rotation": _masked_error_summary(
            rotation_deg,
            pose_valid,
            mae_key="geodesic_mae_deg",
            rmse_key="geodesic_rmse_deg",
        ),
        "gripper": _masked_error_summary(
            gripper_error,
            gripper_valid,
            mae_key="mae_m",
            rmse_key="rmse_m",
        ),
    }


def _gripper_transition_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    gripper_valid: Tensor,
    initial_gripper: Tensor,
    initial_gripper_valid: Tensor,
    control_hz: Tensor,
    threshold_m: float,
) -> dict[str, Any]:
    if not math.isfinite(float(threshold_m)) or threshold_m < 0:
        raise ValueError("gripper transition threshold must be finite and non-negative")
    batch = prediction.shape[0]
    expected = (batch,)
    for name, value in (
        ("initial_gripper", initial_gripper),
        ("initial_gripper_valid", initial_gripper_valid),
        ("control_hz", control_hz),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if torch.any(~torch.isfinite(control_hz)) or torch.any(control_hz <= 0):
        raise ValueError("control_hz must be finite and positive")

    target_width = torch.cat((initial_gripper[:, None], target[..., 6]), dim=1)
    predicted_width = torch.cat((initial_gripper[:, None], prediction[..., 6]), dim=1)
    sequence_valid = torch.cat((initial_gripper_valid[:, None].bool(), gripper_valid.bool()), dim=1)
    pair_valid = sequence_valid[:, :-1] & sequence_valid[:, 1:]
    safe_target = torch.where(sequence_valid, target_width, torch.zeros_like(target_width))
    safe_prediction = torch.where(sequence_valid, predicted_width, torch.zeros_like(predicted_width))
    target_delta = (safe_target[:, 1:] - safe_target[:, :-1]).abs()
    predicted_delta = (safe_prediction[:, 1:] - safe_prediction[:, :-1]).abs()
    negative = torch.full_like(target_delta, -1.0)
    target_masked = torch.where(pair_valid, target_delta, negative)
    predicted_masked = torch.where(pair_valid, predicted_delta, negative)
    target_peak, target_step = target_masked.max(dim=1)
    _, predicted_step = predicted_masked.max(dim=1)
    has_pair = pair_valid.any(dim=1)
    eligible = has_pair & (target_peak >= float(threshold_m))
    eligible_count = int(eligible.sum().item())
    gripper_samples = int(gripper_valid.any(dim=1).sum().item())
    result: dict[str, Any] = {
        "definition": "argmax absolute physical-width change after prepending the true window-start width",
        "step_index": "0-based action index; waypoint[t] is post-action t",
        "target_peak_threshold_m": float(threshold_m),
        "gripper_samples": gripper_samples,
        "eligible_samples": eligible_count,
        "eligible_fraction": eligible_count / gripper_samples if gripper_samples else None,
        "timing_mae_steps": None,
        "timing_rmse_steps": None,
        "timing_mae_seconds": None,
        "timing_rmse_seconds": None,
        "target_peak_delta_mean_m": None,
    }
    if not eligible_count:
        return result
    step_error = (predicted_step[eligible] - target_step[eligible]).abs().double()
    seconds_error = step_error / control_hz[eligible].double()
    result.update(
        {
            "timing_mae_steps": float(step_error.mean()),
            "timing_rmse_steps": float(step_error.square().mean().sqrt()),
            "timing_mae_seconds": float(seconds_error.mean()),
            "timing_rmse_seconds": float(seconds_error.square().mean().sqrt()),
            "target_peak_delta_mean_m": float(target_peak[eligible].double().mean()),
        }
    )
    return result


def trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    time_mask: Tensor,
    pose_mask: Tensor,
    gripper_mask: Tensor,
    initial_gripper: Tensor,
    initial_gripper_valid: Tensor,
    control_hz: Tensor,
    gripper_transition_threshold_m: float = 1.0e-4,
) -> dict[str, Any]:
    """Compute physical metrics with disjoint middle and endpoint reductions."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 7:
        raise ValueError(
            f"prediction and target must have matching [B,T,7] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    prediction = prediction.double()
    target = target.double()
    batch, horizon = prediction.shape[:2]
    for name, mask in (("time_mask", time_mask), ("pose_mask", pose_mask), ("gripper_mask", gripper_mask)):
        if mask.shape != (batch, horizon):
            raise ValueError(f"{name} must be {(batch, horizon)}, got {tuple(mask.shape)}")
    pose_valid = time_mask.bool() & pose_mask.bool()
    gripper_valid = time_mask.bool() & gripper_mask.bool()
    pose_endpoint = _last_valid_mask(pose_valid)
    gripper_endpoint = _last_valid_mask(gripper_valid)
    regions = {
        "all": (pose_valid, gripper_valid),
        "middle": (pose_valid & ~pose_endpoint, gripper_valid & ~gripper_endpoint),
        "endpoint": (pose_endpoint, gripper_endpoint),
    }
    report = {
        name: _region_metrics(
            prediction,
            target,
            pose_valid=region_pose,
            gripper_valid=region_gripper,
        )
        for name, (region_pose, region_gripper) in regions.items()
    }
    report["gripper_transition"] = _gripper_transition_metrics(
        prediction,
        target,
        gripper_valid=gripper_valid,
        initial_gripper=initial_gripper.double(),
        initial_gripper_valid=initial_gripper_valid.bool(),
        control_hz=control_hz.double(),
        threshold_m=gripper_transition_threshold_m,
    )
    return report


class TerminalDeltaReader:
    """Read exact Stage-1 delta_state14 targets referenced by sidecar rows."""

    def __init__(self, *, fallback_root: str | Path | None = None, max_open_shards: int = 8) -> None:
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.fallback_root = Path(fallback_root).expanduser().resolve() if fallback_root else None
        self.max_open_shards = int(max_open_shards)
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def _resolve_path(self, metadata: Mapping[str, Any]) -> Path:
        if "effect_shard_path" not in metadata:
            raise ValueError("sidecar manifest row is missing effect_shard_path")
        raw = Path(str(metadata["effect_shard_path"])).expanduser()
        candidates = [raw]
        if self.fallback_root is not None:
            if not raw.is_absolute():
                candidates.append(self.fallback_root / raw)
            shard_positions = [index for index, part in enumerate(raw.parts) if part == "shards"]
            if shard_positions:
                candidates.append(self.fallback_root.joinpath(*raw.parts[shard_positions[-1] :]))
            candidates.append(
                self.fallback_root / "shards" / str(metadata.get("split", "")) / raw.name
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"could not resolve effect shard {raw}; tried {', '.join(str(path) for path in candidates)}"
        )

    def _handle(self, path: Path) -> h5py.File:
        key = str(path)
        cached = self._handles.pop(key, None)
        if cached is not None:
            self._handles[key] = cached
            return cached
        handle = h5py.File(path, "r")
        missing = [name for name in ("sample_id", "delta_state14", "state_mask") if name not in handle]
        if missing:
            handle.close()
            raise ValueError(f"effect shard {path} is missing fields: {missing}")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_shards:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def read_batch(self, metadata_rows: Sequence[Mapping[str, Any]]) -> tuple[Tensor, Tensor]:
        deltas: list[Tensor] = []
        masks: list[Tensor] = []
        for metadata in metadata_rows:
            path = self._resolve_path(metadata)
            handle = self._handle(path)
            row = int(metadata.get("effect_shard_row", -1))
            if row < 0 or row >= int(handle["sample_id"].shape[0]):
                raise IndexError(f"invalid effect_shard_row={row} for {path}")
            expected_id = str(metadata["sample_id"])
            stored_id = _decode_h5_string(handle["sample_id"][row])
            if expected_id != stored_id:
                raise ValueError(
                    f"sidecar/effect sample mismatch at {path}:{row}: {expected_id!r} != {stored_id!r}"
                )
            delta = torch.from_numpy(np.asarray(handle["delta_state14"][row], dtype=np.float32).copy())
            raw_mask = np.asarray(handle["state_mask"][row])
            if delta.shape != (14,) or raw_mask.shape != (14,):
                raise ValueError(f"delta_state14/state_mask must be shape (14,) in {path}:{row}")
            if not np.isin(raw_mask, (0, 1)).all():
                raise ValueError(f"state_mask must be binary in {path}:{row}")
            mask = torch.from_numpy(raw_mask.astype(np.bool_, copy=True))
            if torch.any(mask & ~torch.isfinite(delta)):
                raise ValueError(f"delta_state14 contains invalid values in valid dimensions at {path}:{row}")
            deltas.append(delta)
            masks.append(mask)
        return torch.stack(deltas), torch.stack(masks)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "TerminalDeltaReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def resolve_terminal_delta_fallback_root(
    data_root: str | Path,
    provenance: Mapping[str, Any],
) -> Path:
    """Resolve the base used by manifest effect-shard pointers.

    ManiSkill sidecars record an explicit effect root. LIBERO A1 releases are
    immutable copies whose ``../effect`` pointers remain relative to their
    parent A0 action release, recorded as ``parent_action_release``.
    """

    root = Path(data_root).expanduser().resolve()
    raw = provenance.get("effect_root") or provenance.get("parent_action_release")
    if raw is None:
        return root
    fallback = Path(str(raw)).expanduser()
    if not fallback.is_absolute():
        fallback = root / fallback
    return fallback.resolve()


@dataclass(frozen=True)
class SplitTensors:
    split: str
    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    suites: tuple[str, ...]
    trajectory: Tensor
    trajectory_raw: Tensor
    current_state: Tensor
    current_state_raw: Tensor
    current_state_mask: Tensor
    e39_mean: Tensor
    time_mask: Tensor
    pose_mask: Tensor
    gripper_mask: Tensor
    control_hz: Tensor
    latent: Tensor
    terminal_delta_state14: Tensor
    terminal_state_mask: Tensor
    normal_prediction_raw: Tensor | None

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


def _evaluation_loader(
    dataset: ActionSidecarDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    sampler = ShardAwareBatchSampler(
        dataset,
        batch_size=batch_size,
        block_size=4 * batch_size,
        shuffle=False,
        drop_last=False,
        order="shard",
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": action_sidecar_collate,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def collect_split_tensors(
    loaded: LoadedActionAE,
    dataset: ActionSidecarDataset,
    terminal_reader: TerminalDeltaReader,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    include_reconstruction: bool,
    log_every_batches: int = 25,
) -> SplitTensors:
    """Encode one official split and optionally retain its normal reconstruction."""

    loader = _evaluation_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    values: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "trajectory",
            "trajectory_raw",
            "current_state",
            "current_state_raw",
            "current_state_mask",
            "e39_mean",
            "time_mask",
            "pose_mask",
            "gripper_mask",
            "control_hz",
            "latent",
            "terminal_delta_state14",
            "terminal_state_mask",
            "normal_prediction_raw",
        )
    }
    sample_ids: list[str] = []
    task_ids: list[str] = []
    suites: list[str] = []
    model = loaded.model
    for batch_index, batch in enumerate(loader):
        trajectory = batch["trajectory"].to(device=device, non_blocking=True)
        current_state = batch["current_state"].to(device=device, non_blocking=True)
        time_mask = batch["time_mask"].to(device=device, non_blocking=True)
        effect: Tensor | None = None
        if model.config.variant == "a1":
            if not bool(batch["e39_valid"].all()):
                raise ValueError("A1 evaluation encountered an invalid E39 row")
            effect = batch["e39_mean"].to(device=device, non_blocking=True)
        with torch.inference_mode():
            latent = model.encode(
                trajectory,
                current_state,
                effect,
                time_mask=time_mask,
            )
            if include_reconstruction:
                prediction = model.decode(latent, current_state, effect)
                prediction_raw = loaded.normalization.denormalize(prediction).float().cpu()
                if not torch.isfinite(prediction_raw).all():
                    raise ValueError(f"non-finite normal reconstruction on split {dataset.split}")
                values["normal_prediction_raw"].append(prediction_raw)
        terminal_delta, terminal_mask = terminal_reader.read_batch(batch["metadata"])
        # Timing metrics operate on representation slots. Bridge rows retain the
        # physical 5 Hz source rate in ``control_hz`` while their 24-slot ActionVAE
        # trajectory is sampled at 20 Hz.
        control_hz = torch.as_tensor(
            [
                float(item.get("representation_hz", item["control_hz"]))
                for item in batch["metadata"]
            ],
            dtype=torch.float32,
        )
        sample_ids.extend(str(value) for value in batch["sample_ids"])
        task_ids.extend(str(item["task_id"]) for item in batch["metadata"])
        suites.extend(str(item.get("suite", "unknown")) for item in batch["metadata"])
        for name in (
            "trajectory",
            "trajectory_raw",
            "current_state",
            "current_state_raw",
            "current_state_mask",
            "e39_mean",
            "time_mask",
            "pose_mask",
            "gripper_mask",
        ):
            values[name].append(batch[name].cpu())
        values["control_hz"].append(control_hz)
        values["latent"].append(latent[:, 0].float().cpu())
        values["terminal_delta_state14"].append(terminal_delta)
        values["terminal_state_mask"].append(terminal_mask)
        if log_every_batches > 0 and (batch_index + 1) % log_every_batches == 0:
            print(
                f"[{dataset.split}] encoded {len(sample_ids)}/{len(dataset)} samples",
                file=sys.stderr,
                flush=True,
            )
    if (
        len(sample_ids) != len(dataset)
        or len(task_ids) != len(dataset)
        or len(suites) != len(dataset)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise RuntimeError(f"split {dataset.split} was not evaluated exactly once")

    def concatenate(name: str) -> Tensor:
        if not values[name]:
            raise RuntimeError(f"no collected values for {name}")
        return torch.cat(values[name], dim=0)

    normal = concatenate("normal_prediction_raw") if include_reconstruction else None
    return SplitTensors(
        split=dataset.split,
        sample_ids=tuple(sample_ids),
        task_ids=tuple(task_ids),
        suites=tuple(suites),
        trajectory=concatenate("trajectory"),
        trajectory_raw=concatenate("trajectory_raw"),
        current_state=concatenate("current_state"),
        current_state_raw=concatenate("current_state_raw"),
        current_state_mask=concatenate("current_state_mask"),
        e39_mean=concatenate("e39_mean"),
        time_mask=concatenate("time_mask"),
        pose_mask=concatenate("pose_mask"),
        gripper_mask=concatenate("gripper_mask"),
        control_hz=concatenate("control_hz"),
        latent=concatenate("latent"),
        terminal_delta_state14=concatenate("terminal_delta_state14"),
        terminal_state_mask=concatenate("terminal_state_mask"),
        normal_prediction_raw=normal,
    )


def deterministic_derangement(length: int, seed: int) -> tuple[Tensor, int]:
    """Return a deterministic cyclic donor mapping with no fixed points."""

    if length < 2:
        raise ValueError("a shuffled intervention requires at least two samples")
    generator = np.random.default_rng(int(seed))
    offset = int(generator.integers(1, length))
    mapping = (torch.arange(length, dtype=torch.long) + offset) % length
    if torch.any(mapping == torch.arange(length)):
        raise AssertionError("cyclic shuffle unexpectedly contains a fixed point")
    return mapping, offset


def grouped_derangement(
    groups: Sequence[str],
    seed: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Return a deterministic no-fixed-point mapping within each group."""

    normalized = tuple(str(group) for group in groups)
    if not normalized:
        raise ValueError("a grouped shuffle requires at least one sample")
    mapping = torch.empty(len(normalized), dtype=torch.long)
    offsets: dict[str, int] = {}
    for group in sorted(set(normalized)):
        indices = [index for index, value in enumerate(normalized) if value == group]
        if len(indices) < 2:
            raise ValueError(f"grouped shuffle requires at least two samples in {group!r}")
        digest = hashlib.sha256(f"{int(seed)}\0{group}".encode("utf-8")).digest()
        group_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        offset = int(np.random.default_rng(group_seed).integers(1, len(indices)))
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        mapping[index_tensor] = index_tensor[(torch.arange(len(indices)) + offset) % len(indices)]
        offsets[group] = offset
    if torch.any(mapping == torch.arange(len(normalized))):
        raise AssertionError("grouped shuffle unexpectedly contains a fixed point")
    return mapping, {
        "seed": int(seed),
        "group_count": len(offsets),
        "group_sizes": {
            group: normalized.count(group) for group in sorted(set(normalized))
        },
        "groups_sha256": _sha256_strings(normalized),
        "offsets": offsets,
        "mapping_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
    }


def intervention_reconstructions(
    loaded: LoadedActionAE,
    split: SplitTensors,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    shuffle_groups: Sequence[str] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Decode normal and controlled z-action/E39 interventions."""

    if split.normal_prediction_raw is None:
        raise ValueError(f"split {split.split} does not contain normal reconstructions")
    if shuffle_groups is None:
        z_mapping, z_offset = deterministic_derangement(split.sample_count, seed)
        effect_mapping, effect_offset = deterministic_derangement(split.sample_count, seed + 1)
        shuffle_metadata = {
            "strategy": "global cyclic cross-sample donor mapping with no fixed points",
            "z_action": {
                "seed": int(seed),
                "offset": z_offset,
                "mapping_sha256": hashlib.sha256(z_mapping.numpy().tobytes()).hexdigest(),
            },
            "z_effect": {
                "seed": int(seed + 1),
                "offset": effect_offset,
                "mapping_sha256": hashlib.sha256(effect_mapping.numpy().tobytes()).hexdigest(),
            },
        }
    else:
        if len(shuffle_groups) != split.sample_count:
            raise ValueError("shuffle_groups must have one value per split sample")
        z_mapping, z_metadata = grouped_derangement(shuffle_groups, seed)
        effect_mapping, effect_metadata = grouped_derangement(shuffle_groups, seed + 1)
        shuffle_metadata = {
            "strategy": "group-local cyclic donor mapping with no fixed points",
            "z_action": z_metadata,
            "z_effect": effect_metadata,
        }
    outputs: dict[str, list[Tensor]] = {
        "z_zero_decoder": [],
        "z_shuffled_decoder": [],
    }
    if loaded.model.config.variant == "a1":
        outputs.update(
            {
                "effect_zero_decoder": [],
                "effect_shuffled_decoder": [],
                "effect_zero_end_to_end": [],
                "effect_shuffled_end_to_end": [],
            }
        )
    model = loaded.model
    for start in range(0, split.sample_count, batch_size):
        stop = min(start + batch_size, split.sample_count)
        item = slice(start, stop)
        trajectory = split.trajectory[item].to(device)
        state = split.current_state[item].to(device)
        time_mask = split.time_mask[item].to(device)
        latent = split.latent[item].unsqueeze(1).to(device)
        effect = split.e39_mean[item].to(device) if model.config.variant == "a1" else None
        shuffled_z = split.latent[z_mapping[item]].unsqueeze(1).to(device)
        with torch.inference_mode():
            normalized: dict[str, Tensor] = {
                "z_zero_decoder": model.decode(torch.zeros_like(latent), state, effect),
                "z_shuffled_decoder": model.decode(shuffled_z, state, effect),
            }
            if model.config.variant == "a1":
                assert effect is not None
                zero_effect = torch.zeros_like(effect)
                shuffled_effect = split.e39_mean[effect_mapping[item]].to(device)
                normalized["effect_zero_decoder"] = model.decode(latent, state, zero_effect)
                normalized["effect_shuffled_decoder"] = model.decode(latent, state, shuffled_effect)
                zero_effect_z = model.encode(
                    trajectory,
                    state,
                    zero_effect,
                    time_mask=time_mask,
                )
                shuffled_effect_z = model.encode(
                    trajectory,
                    state,
                    shuffled_effect,
                    time_mask=time_mask,
                )
                normalized["effect_zero_end_to_end"] = model.decode(zero_effect_z, state, zero_effect)
                normalized["effect_shuffled_end_to_end"] = model.decode(
                    shuffled_effect_z, state, shuffled_effect
                )
            for name, prediction in normalized.items():
                raw = loaded.normalization.denormalize(prediction).float().cpu()
                if not torch.isfinite(raw).all():
                    raise ValueError(f"non-finite {name} reconstruction on split {split.split}")
                outputs[name].append(raw)
    combined = {"normal": split.normal_prediction_raw}
    combined.update({name: torch.cat(parts, dim=0) for name, parts in outputs.items()})
    return combined, shuffle_metadata


def latent_diagnostics(
    latent: Tensor,
    *,
    seed: int,
    pairwise_max_samples: int = 4096,
) -> dict[str, Any]:
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent diagnostics require [N,D] with N >= 2")
    values = latent.double().numpy()
    if not np.isfinite(values).all():
        raise ValueError("latent contains NaN/Inf")
    _, trace, effective_rank = covariance_and_effective_rank(values)
    std = np.std(values, axis=0, ddof=1)
    pairwise_values = values
    if len(values) > pairwise_max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(values), size=pairwise_max_samples, replace=False))
        pairwise_values = values[indices]
    return {
        "sample_count": int(len(values)),
        "latent_dim": int(values.shape[1]),
        "per_dimension_std": std.tolist(),
        "mean_dimension_std": float(np.mean(std)),
        "min_dimension_std": float(np.min(std)),
        "max_dimension_std": float(np.max(std)),
        "active_dimensions_std_gt_1e-6": int(np.sum(std > 1.0e-6)),
        "covariance_trace": trace,
        "effective_rank": effective_rank,
        "pairwise_median_distance": pairwise_median_distance(pairwise_values),
        "pairwise_sample_count": int(len(pairwise_values)),
        "pairwise_is_exact": len(pairwise_values) == len(values),
    }


_ERROR_PATHS = (
    "all.position.l2_mae_cm",
    "all.position.l2_rmse_cm",
    "all.rotation.geodesic_mae_deg",
    "all.gripper.mae_m",
    "middle.position.l2_mae_cm",
    "middle.rotation.geodesic_mae_deg",
    "middle.gripper.mae_m",
    "endpoint.position.l2_mae_cm",
    "endpoint.rotation.geodesic_mae_deg",
    "endpoint.gripper.mae_m",
    "gripper_transition.timing_mae_steps",
)


def _nested_value(payload: Mapping[str, Any], path: str) -> float | None:
    value: Any = payload
    for key in path.split("."):
        value = value[key]
    return float(value) if value is not None else None


def intervention_error_increase(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Report intervention minus normal error; positive values mean degradation."""

    normal = metrics["normal"]
    report: dict[str, Any] = {}
    for mode, values in metrics.items():
        if mode == "normal":
            continue
        differences: dict[str, float | None] = {}
        for path in _ERROR_PATHS:
            baseline = _nested_value(normal, path)
            intervention = _nested_value(values, path)
            differences[path] = (
                intervention - baseline
                if baseline is not None and intervention is not None
                else None
            )
        report[mode] = differences
    return report


def terminal_delta_state_probe(
    splits: Mapping[str, SplitTensors],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    minimum_samples_per_split: int = 8,
) -> dict[str, Any]:
    """Fit train z->delta_state14 Ridge probes, tune on val, and report test."""

    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"terminal probe requires exactly {sorted(required)}, got {sorted(splits)}")
    if minimum_samples_per_split < 2:
        raise ValueError("minimum_samples_per_split must be at least 2")
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(not math.isfinite(value) or value < 0 for value in alpha_values):
        raise ValueError("Ridge alphas must be finite and non-negative")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split in splits.items():
        x = split.latent.double().numpy()
        y = split.terminal_delta_state14.double().numpy()
        mask = split.terminal_state_mask.bool().numpy()
        if x.ndim != 2 or y.shape != (len(x), 14) or mask.shape != y.shape:
            raise ValueError(f"invalid terminal probe arrays for split {split_name}")
        arrays[split_name] = {"x": x, "y": y, "mask": mask}

    x_mean = np.mean(arrays["train"]["x"], axis=0)
    x_scale = np.std(arrays["train"]["x"], axis=0, ddof=1)
    x_scale = np.maximum(x_scale, 1.0e-6)
    standardized_x = {
        split: (values["x"] - x_mean) / x_scale for split, values in arrays.items()
    }
    per_dimension: dict[str, Any] = {}
    probe_standardized_squared: list[np.ndarray] = []
    baseline_standardized_squared: list[np.ndarray] = []
    probe_r2_values: list[float] = []
    for dimension in range(14):
        valid = {split: values["mask"][:, dimension] for split, values in arrays.items()}
        counts = {split: int(np.sum(mask)) for split, mask in valid.items()}
        if min(counts.values()) < minimum_samples_per_split:
            continue
        train_target = arrays["train"]["y"][valid["train"], dimension]
        target_mean = float(np.mean(train_target))
        target_scale = max(float(np.std(train_target, ddof=1)), 1.0e-8)
        targets = {
            split: (values["y"][:, dimension] - target_mean) / target_scale
            for split, values in arrays.items()
        }
        best_alpha: float | None = None
        best_validation_mse = float("inf")
        for alpha in alpha_values:
            model = Ridge(alpha=alpha, solver="svd")
            model.fit(standardized_x["train"][valid["train"]], targets["train"][valid["train"]])
            prediction = model.predict(standardized_x["validation"][valid["validation"]])
            mse = float(np.mean(np.square(prediction - targets["validation"][valid["validation"]])))
            if mse < best_validation_mse:
                best_alpha = alpha
                best_validation_mse = mse
        assert best_alpha is not None
        model = Ridge(alpha=best_alpha, solver="svd")
        model.fit(standardized_x["train"][valid["train"]], targets["train"][valid["train"]])
        test_prediction = model.predict(standardized_x["test"][valid["test"]])
        test_target = targets["test"][valid["test"]]
        squared = np.square(test_prediction - test_target)
        baseline_squared = np.square(test_target)
        raw_rmse = float(np.sqrt(np.mean(squared)) * target_scale)
        denominator = float(np.sum(np.square(test_target - np.mean(test_target))))
        r2 = 1.0 - float(np.sum(squared)) / denominator if denominator > 0 else float("nan")
        baseline_denominator = denominator
        baseline_r2 = (
            1.0 - float(np.sum(baseline_squared)) / baseline_denominator
            if baseline_denominator > 0
            else float("nan")
        )
        per_dimension[str(dimension)] = {
            "train_samples": counts["train"],
            "validation_samples": counts["validation"],
            "test_samples": counts["test"],
            "alpha": best_alpha,
            "train_target_mean": target_mean,
            "train_target_std": target_scale,
            "validation_standardized_mse": best_validation_mse,
            "test_standardized_rmse": float(np.sqrt(np.mean(squared))),
            "test_raw_rmse": raw_rmse,
            "test_r2": _finite_or_none(r2),
            "mean_baseline_test_standardized_rmse": float(np.sqrt(np.mean(baseline_squared))),
            "mean_baseline_test_r2": _finite_or_none(baseline_r2),
        }
        probe_standardized_squared.append(squared)
        baseline_standardized_squared.append(baseline_squared)
        if math.isfinite(r2):
            probe_r2_values.append(r2)
    if not per_dimension:
        raise RuntimeError("no delta_state14 dimension has enough valid train/validation/test samples")
    probe_squared = np.concatenate(probe_standardized_squared)
    baseline_squared = np.concatenate(baseline_standardized_squared)
    probe_rmse = float(np.sqrt(np.mean(probe_squared)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_squared)))
    return {
        "protocol": {
            "predictor": "frozen z-action",
            "target": "raw Stage-1 effect-window delta_state14 joined by sample_id",
            "fit_split": "train",
            "alpha_selection_split": "validation",
            "report_split": "test",
            "feature_standardization": "train mean/std with 1e-6 floor",
            "target_standardization": "per-dimension train mean/std with 1e-8 floor",
            "baseline": "per-dimension train target mean",
            "alphas": list(alpha_values),
            "minimum_samples_per_split": int(minimum_samples_per_split),
        },
        "valid_dimensions": len(per_dimension),
        "pooled_test_standardized_rmse": probe_rmse,
        "mean_baseline_pooled_test_standardized_rmse": baseline_rmse,
        "relative_rmse_improvement_over_mean": (
            (baseline_rmse - probe_rmse) / baseline_rmse if baseline_rmse > 0 else None
        ),
        "macro_test_r2": float(np.mean(probe_r2_values)) if probe_r2_values else None,
        "per_dimension": per_dimension,
    }


def _split_subset_size(args: argparse.Namespace, split: str) -> int | None:
    return getattr(args, f"{split}_subset_size")


def _check_split_disjointness(splits: Mapping[str, SplitTensors]) -> dict[str, int]:
    names = sorted(splits)
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = len(set(splits[left].sample_ids).intersection(splits[right].sample_ids))
            overlaps[f"{left}__{right}"] = overlap
            if overlap:
                raise ValueError(f"sample leakage between {left} and {right}: {overlap}")
    return overlaps


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    data_root = Path(args.data_root).expanduser().resolve()
    loaded = load_action_ae_checkpoint(
        args.checkpoint,
        data_root,
        device=device,
        sidecar_binding=args.sidecar_binding,
        sidecar_reference_root=args.sidecar_reference_root,
    )
    variant = loaded.model.config.variant
    provenance_path = data_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    terminal_delta_fallback_root = resolve_terminal_delta_fallback_root(data_root, provenance)
    reconstruction_splits = tuple(dict.fromkeys(args.splits))
    needed_splits = set(reconstruction_splits)
    if args.terminal_probe:
        needed_splits.update(("train", "validation", "test"))

    collected: dict[str, SplitTensors] = {}
    with TerminalDeltaReader(
        fallback_root=terminal_delta_fallback_root,
        max_open_shards=args.max_open_shards,
    ) as terminal_reader:
        for split in ("train", "validation", "test"):
            if split not in needed_splits:
                continue
            dataset = ActionSidecarDataset(
                data_root,
                split,
                max_open_shards=args.max_open_shards,
                verify_row_identity=args.verify_row_identity,
                require_e39=variant == "a1",
                subset_size=_split_subset_size(args, split),
                subset_seed=args.seed,
            )
            try:
                collected[split] = collect_split_tensors(
                    loaded,
                    dataset,
                    terminal_reader,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    include_reconstruction=split in reconstruction_splits,
                    log_every_batches=args.log_every_batches,
                )
            finally:
                dataset.close()
    overlap_audit = _check_split_disjointness(collected)

    split_reports: dict[str, Any] = {}
    shuffle_reports: dict[str, Any] = {}
    for split_name in reconstruction_splits:
        split = collected[split_name]
        predictions, shuffle = intervention_reconstructions(
            loaded,
            split,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        metrics = {
            name: trajectory_metrics(
                prediction,
                split.trajectory_raw,
                time_mask=split.time_mask,
                pose_mask=split.pose_mask,
                gripper_mask=split.gripper_mask,
                initial_gripper=split.current_state_raw[:, 6],
                initial_gripper_valid=split.current_state_mask[:, 6],
                control_hz=split.control_hz,
                gripper_transition_threshold_m=args.gripper_transition_threshold_m,
            )
            for name, prediction in predictions.items()
        }
        split_reports[split_name] = {
            "sample_count": split.sample_count,
            "sample_ids_sha256": _sha256_strings(split.sample_ids),
            "latent": latent_diagnostics(
                split.latent,
                seed=args.seed,
                pairwise_max_samples=args.pairwise_max_samples,
            ),
            "reconstruction": metrics,
            "intervention_error_increase_vs_normal": intervention_error_increase(metrics),
        }
        shuffle_reports[split_name] = shuffle

    probe = None
    if args.terminal_probe:
        probe = terminal_delta_state_probe(
            {name: collected[name] for name in ("train", "validation", "test")},
            alphas=args.ridge_alphas,
            minimum_samples_per_split=args.probe_minimum_samples,
        )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": loaded.checkpoint_sha256,
            "epoch": loaded.epoch,
            "global_step": loaded.global_step,
            "variant": variant,
            "model_config": loaded.metadata["model_config"],
            "normalization_sha256": loaded.metadata["normalization_sha256"],
            "e39_cache_sha256": loaded.metadata.get("e39_cache_sha256"),
            "e39_checkpoint_sha256": loaded.metadata.get("e39_checkpoint_sha256"),
            "effect_manifest_sha256": loaded.metadata.get("effect_manifest_sha256"),
        },
        "data": {
            "root": str(data_root),
            "official_splits_only": True,
            "sample_overlap": overlap_audit,
            "collected": {
                name: {
                    "sample_count": value.sample_count,
                    "sample_ids_sha256": _sha256_strings(value.sample_ids),
                }
                for name, value in collected.items()
            },
        },
        "metric_semantics": {
            "position": "Euclidean waypoint error; MAE/RMSE over valid waypoints in cm",
            "rotation": "shortest-path SO(3) geodesic error in degrees",
            "middle": "all valid modality-specific waypoints except each sample's last valid waypoint",
            "endpoint": "each sample's last valid modality-specific waypoint",
            "gripper": "physical width error in meters",
            "gripper_transition": (
                "argmax absolute width change after prepending true start width; target peak must be "
                f">= {args.gripper_transition_threshold_m} m"
            ),
            "effect_decoder_only": "encode with true E39, hold z-action fixed, replace E39 only in decoder",
            "effect_end_to_end": "replace E39 consistently in encoder and decoder",
        },
        "shuffle": shuffle_reports,
        "splits": split_reports,
        "terminal_delta_state_probe": probe,
        "arguments": {
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "reconstruction_splits": list(reconstruction_splits),
            "terminal_probe": bool(args.terminal_probe),
            "sidecar_binding": args.sidecar_binding,
            "sidecar_reference_root": (
                str(args.sidecar_reference_root)
                if args.sidecar_reference_root is not None
                else None
            ),
            "subset_sizes": {
                split: _split_subset_size(args, split) for split in ("train", "validation", "test")
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--sidecar-binding",
        choices=("exact", "semantic"),
        default="exact",
    )
    parser.add_argument("--sidecar-reference-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-open-shards", default=8, type=int)
    parser.add_argument("--seed", default=20260807, type=int)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--terminal-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-row-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-subset-size", type=int)
    parser.add_argument("--validation-subset-size", type=int)
    parser.add_argument("--test-subset-size", type=int)
    parser.add_argument("--gripper-transition-threshold-m", default=1.0e-4, type=float)
    parser.add_argument("--pairwise-max-samples", default=4096, type=int)
    parser.add_argument("--probe-minimum-samples", default=8, type=int)
    parser.add_argument("--ridge-alphas", nargs="+", default=DEFAULT_RIDGE_ALPHAS, type=float)
    parser.add_argument("--log-every-batches", default=25, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.max_open_shards < 1:
        parser.error("batch-size/max-open-shards must be positive and num-workers non-negative")
    if args.pairwise_max_samples < 2:
        parser.error("pairwise-max-samples must be at least 2")
    if args.sidecar_binding == "semantic" and args.sidecar_reference_root is None:
        parser.error("--sidecar-reference-root is required with --sidecar-binding=semantic")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")
    report = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "variant": report["checkpoint"]["variant"],
                "splits": {name: value["sample_count"] for name, value in report["splits"].items()},
                "terminal_probe": report["terminal_delta_state_probe"] is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
