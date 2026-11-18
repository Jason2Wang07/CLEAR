"""Export the immutable RoboTwin Stage-2 paired teacher-token cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from latent_action_model.robotwin_action_ae.data import (
    RoboTwinActionDataset,
    robotwin_action_collate,
)
from latent_action_model.robotwin_action_ae.beta_vae_lightning import (
    RoboTwinActionBetaVAELightningModule,
)

from .temporal import (
    DEFAULT_TEMPORAL_CONTRACT,
    RoboTwinStage2TemporalContract,
    resolve_robotwin_stage2_temporal_contract,
)
from .sidecar_binding import (
    validate_lawam_idm_sidecar_contract,
    validate_semantic_sidecar_binding,
)


CACHE_SCHEMA_VERSION = "robotwin-stage2-paired-teacher-cache-v3"
LEGACY_CACHE_SCHEMA_VERSION = "robotwin-stage2-paired-teacher-cache-v1"
BUNDLE_SCHEMA_VERSION = "robotwin-stage2-teacher-bundle-v3"
EXPECTED_SELECTION_SCHEMA = "lawam-stage1-checkpoint-selection-v1"
EXPECTED_DOWNSTREAM_SELECTION_SCHEMA = "lawam-stage1-downstream-selection-v1"
ACTION_SCHEMA_VERSION = "robotwin-action-tcn-beta-vae-v1"


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_dense_sidecar_effect_source(
    *,
    paths: dict[str, Path | None],
    metadata: dict[str, Any],
    effect_sha: str,
    temporal: RoboTwinStage2TemporalContract,
    binding: str,
) -> dict[str, Any]:
    sidecar_root = paths["sidecar_root"]
    effect_selection = paths["effect_selection"]
    assert sidecar_root is not None and effect_selection is not None
    required = {
        name: sidecar_root / name
        for name in (
            "_SUCCESS",
            "checksums.sha256",
            "coverage_report.json",
            "provenance.json",
            "recovery_audit.json",
        )
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    coverage = _read_json(required["coverage_report.json"])
    provenance = _read_json(required["provenance.json"])
    audit = _read_json(required["recovery_audit.json"])
    expected_splits = {"train": 4_065_338, "validation": 510_933, "test": 510_976}
    if (
        not coverage.get("valid")
        or int(coverage.get("rows", -1)) != temporal.row_count
        or int(coverage.get("e39_valid_rows", -1)) != temporal.row_count
        or int(coverage.get("window_stride", -1)) != 1
        or coverage.get("split_counts") != expected_splits
    ):
        raise ValueError("dense sidecar coverage does not match the locked stride-1 release")
    if (
        not audit.get("valid")
        or int(audit.get("issue_count", -1)) != 0
        or int(audit.get("rows", -1)) != temporal.row_count
        or int(audit.get("e39_valid_rows", -1)) != temporal.row_count
        or audit.get("split_counts") != expected_splits
    ):
        raise ValueError("dense sidecar recovery audit is missing or invalid")
    e39 = provenance.get("e39")
    if not isinstance(e39, dict):
        raise ValueError("dense sidecar provenance is missing its E39 binding")
    configured_effect = paths.get("effect_config")
    effect_config = (
        configured_effect
        if configured_effect is not None
        else Path(str(e39.get("model_config", ""))).expanduser().resolve()
    )
    if not effect_config.is_file():
        raise FileNotFoundError(effect_config)
    expected_e39 = {
        "checkpoint_sha256": effect_sha,
        "model_config_sha256": sha256_file(effect_config),
        "selection_report_sha256": sha256_file(effect_selection),
        "posterior_statistic": "mean",
        "sampling": False,
    }
    mismatches = {
        key: (e39.get(key), value)
        for key, value in expected_e39.items()
        if e39.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dense sidecar E39 provenance mismatch: {mismatches}")
    if metadata.get("e39_checkpoint_sha256") != effect_sha:
        raise ValueError("ActionVAE checkpoint is not bound to the dense sidecar E39 teacher")
    binding_metadata: dict[str, Any]
    if binding == "exact":
        action_sidecar_hashes = {
            "sidecar_checksums_sha256": sha256_file(required["checksums.sha256"]),
            "sidecar_provenance_sha256": sha256_file(required["provenance.json"]),
        }
        action_mismatches = {
            key: (metadata.get(key), value)
            for key, value in action_sidecar_hashes.items()
            if metadata.get(key) != value
        }
        if action_mismatches:
            raise ValueError(
                "ActionVAE checkpoint was not trained against this exact dense sidecar: "
                f"{action_mismatches}"
            )
        binding_metadata = {
            "schema": "robotwin-dense-sidecar-exact-binding-v1",
            **action_sidecar_hashes,
        }
    else:
        reference_root = paths.get("sidecar_reference_root")
        if reference_root is None:
            raise ValueError("semantic sidecar binding requires --sidecar-reference-root")
        binding_metadata = validate_semantic_sidecar_binding(
            sidecar_root,
            reference_root,
            action_metadata=metadata,
            effect_checkpoint=paths["effect_checkpoint"],
            effect_config=effect_config,
            effect_selection=effect_selection,
            verify_content_checksums=True,
        )
    return {
        "effect_source": "sidecar_e39_mean",
        "effect_config": effect_config,
        "effect_config_sha256": sha256_file(effect_config),
        "sidecar_provenance_sha256": sha256_file(required["provenance.json"]),
        "sidecar_binding": binding,
        "sidecar_binding_metadata": binding_metadata,
        "split_rows": expected_splits,
    }


def _validate_inputs(
    args: argparse.Namespace,
    temporal: RoboTwinStage2TemporalContract,
) -> dict[str, Any]:
    paths = {
        "sidecar_root": args.sidecar_root.expanduser().resolve(),
        "effect_checkpoint": (
            args.effect_checkpoint.expanduser().resolve()
            if args.effect_checkpoint is not None
            else None
        ),
        "effect_selection": (
            args.effect_selection.expanduser().resolve()
            if args.effect_selection is not None
            else None
        ),
        "effect_config": (
            args.effect_config.expanduser().resolve()
            if args.effect_config is not None
            else None
        ),
        "sidecar_reference_root": (
            args.sidecar_reference_root.expanduser().resolve()
            if args.sidecar_reference_root is not None
            else None
        ),
        "effect_cache": (
            args.effect_cache.expanduser().resolve()
            if args.effect_cache is not None
            else None
        ),
        "action_checkpoint": args.action_checkpoint.expanduser().resolve(),
        "condition_checkpoint": (
            args.condition_checkpoint.expanduser().resolve()
            if args.condition_checkpoint is not None
            else None
        ),
        "condition_config": (
            args.condition_config.expanduser().resolve()
            if args.condition_config is not None
            else None
        ),
    }
    if not paths["sidecar_root"].is_dir():
        raise FileNotFoundError(paths["sidecar_root"])
    for name in ("action_checkpoint",):
        path = paths[name]
        if path is None or not path.is_file():
            raise FileNotFoundError(path)

    action_checkpoint = paths["action_checkpoint"]
    assert action_checkpoint is not None
    action_payload = torch.load(action_checkpoint, map_location="cpu", weights_only=False)
    metadata = action_payload.get("action_ae_metadata", {})
    model_config = metadata.get("model_config", {})
    schema_version = str(metadata.get("schema_version", ""))
    latent_distribution = metadata.get("latent_distribution", {})
    if schema_version != ACTION_SCHEMA_VERSION or model_config.get("variant") != "a1":
        raise ValueError(
            f"Stage-2 cache requires an {ACTION_SCHEMA_VERSION} checkpoint"
        )
    if (
        model_config.get("horizon") != temporal.teacher_action_steps
        or model_config.get("trajectory_dim") != 14
        or model_config.get("latent_dim") != 32
    ):
        raise ValueError("ActionVAE checkpoint does not implement the locked H36x14/32D contract")
    if latent_distribution.get("prior") != "standard_normal":
        raise ValueError("RoboTwin Stage-2 requires a standard-normal ActionVAE prior")

    requested_source = str(args.condition_source).strip().lower()
    metadata_source = str(metadata.get("condition_source", "e39")).strip().lower()
    condition_source = metadata_source if requested_source == "auto" else requested_source
    if condition_source != metadata_source:
        raise ValueError(
            "requested condition source differs from ActionVAE metadata: "
            f"requested={condition_source!r}, checkpoint={metadata_source!r}"
        )
    if condition_source == "lawam_idm":
        if args.effect_source != "sidecar_condition_mean":
            raise ValueError(
                "lawam_idm cache export requires --effect-source=sidecar_condition_mean"
            )
        if args.sidecar_binding != "contract":
            raise ValueError("lawam_idm cache export requires --sidecar-binding=contract")
        binding_metadata = validate_lawam_idm_sidecar_contract(
            paths["sidecar_root"],
            action_metadata=metadata,
        )
        condition_sha = str(binding_metadata["condition_checkpoint_sha256"])
        condition_checkpoint = paths["condition_checkpoint"]
        if condition_checkpoint is not None:
            if not condition_checkpoint.is_file():
                raise FileNotFoundError(condition_checkpoint)
            if sha256_file(condition_checkpoint) != condition_sha:
                raise ValueError(
                    "configured LaWAM condition checkpoint differs from sidecar/ActionVAE provenance"
                )
        condition_config = paths["condition_config"]
        provenance = _read_json(paths["sidecar_root"] / "provenance.json")
        condition_provenance = provenance.get("condition", provenance.get("lawam_idm", {}))
        if condition_config is not None:
            if not condition_config.is_file():
                raise FileNotFoundError(condition_config)
            if sha256_file(condition_config) != condition_provenance.get("model_config_sha256"):
                raise ValueError("configured LaWAM condition config differs from sidecar provenance")
        return {
            **paths,
            "condition_source": condition_source,
            "condition_cache_name": "lawam",
            "condition_checkpoint_sha256": condition_sha,
            "condition_config_sha256": condition_provenance.get("model_config_sha256"),
            "sidecar_binding": "contract",
            "sidecar_binding_metadata": binding_metadata,
            "split_rows": dict(binding_metadata["split_rows"]),
            "selection": None,
            "selection_assessment": None,
            "action_payload": action_payload,
            "action_metadata": metadata,
            "effect_source": "sidecar_condition_mean",
        }

    if condition_source != "e39":
        raise ValueError(f"unsupported RoboTwin Stage-2 condition source: {condition_source!r}")
    for name in ("effect_checkpoint", "effect_selection"):
        path = paths[name]
        if path is None or not path.is_file():
            raise FileNotFoundError(path)
    if args.effect_source == "legacy_cache":
        effect_cache_path = paths["effect_cache"]
        if effect_cache_path is None or not effect_cache_path.is_file():
            raise FileNotFoundError(effect_cache_path)

    effect_selection = paths["effect_selection"]
    effect_checkpoint = paths["effect_checkpoint"]
    assert effect_selection is not None and effect_checkpoint is not None
    selection = _read_json(effect_selection)
    if selection.get("schema") not in {
        EXPECTED_SELECTION_SCHEMA,
        EXPECTED_DOWNSTREAM_SELECTION_SCHEMA,
    }:
        raise ValueError("Effect checkpoint-selection schema mismatch")
    if (
        args.sidecar_binding == "exact"
        and selection.get("selected_checkpoint_id") != effect_checkpoint.name
    ):
        raise ValueError("provided Effect checkpoint is not the validation-selected checkpoint")
    if selection.get("schema") == EXPECTED_SELECTION_SCHEMA:
        selected = next(
            (
                item
                for item in selection.get("assessments", [])
                if item.get("checkpoint_id") == effect_checkpoint.name
            ),
            None,
        )
        selected_is_eligible = isinstance(selected, dict) and selected.get("eligible") is True
    else:
        selected = selection
        selected_is_eligible = selection.get("eligible_under_locked_gates") is True
    if not selected_is_eligible:
        raise ValueError("selected Effect checkpoint is not eligible in the selection report")

    effect_sha = sha256_file(effect_checkpoint)
    expected_effect_sha = metadata.get("e39_checkpoint_sha256")
    if expected_effect_sha != effect_sha:
        raise ValueError("ActionAE provenance does not bind the selected Effect checkpoint")
    source_metadata: dict[str, Any]
    if args.effect_source == "legacy_cache":
        effect_cache = paths["effect_cache"]
        assert effect_cache is not None
        cache_sha = sha256_file(effect_cache)
        if metadata.get("e39_cache_sha256") != cache_sha:
            raise ValueError("ActionAE provenance does not bind the selected Effect cache")
        source_metadata = {
            "effect_source": "legacy_cache",
            "effect_cache_sha256": cache_sha,
        }
    else:
        source_metadata = _validate_dense_sidecar_effect_source(
            paths=paths,
            metadata=metadata,
            effect_sha=effect_sha,
            temporal=temporal,
            binding=args.sidecar_binding,
        )
    return {
        **paths,
        **source_metadata,
        "condition_source": "e39",
        "condition_cache_name": "effect",
        "condition_checkpoint_sha256": effect_sha,
        "selection": selection,
        "selection_assessment": selected,
        "action_payload": action_payload,
        "action_metadata": metadata,
        "effect_checkpoint_sha256": effect_sha,
    }


def _effect_cache(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_ids", "splits", "posterior_mean"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Effect posterior cache is missing fields: {missing}")
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
        splits = np.asarray(payload["splits"]).astype(str)
        means = np.asarray(payload["posterior_mean"], dtype=np.float32)
    if means.shape != (len(sample_ids), 32) or len(splits) != len(sample_ids):
        raise ValueError("Effect posterior cache shape mismatch")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Effect posterior cache contains duplicate sample IDs")
    return {value: index for index, value in enumerate(sample_ids.tolist())}, splits, means


def _loader(dataset: RoboTwinActionDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=robotwin_action_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )


def _collect(
    *,
    module: RoboTwinActionBetaVAELightningModule,
    sidecar_root: Path,
    split: str,
    effect_index: dict[str, int] | None,
    effect_splits: np.ndarray | None,
    effect_means: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
    condition_source: str,
    condition_cache_name: str,
) -> dict[str, np.ndarray]:
    dataset = RoboTwinActionDataset(
        sidecar_root,
        split,
        condition_source=condition_source,
        require_condition=True,
    )
    sample_ids: list[str] = []
    task_ids: list[str] = []
    episode_ids: list[int] = []
    start_steps: list[int] = []
    condition_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    action_logvar_values: list[np.ndarray] = []
    teacher_trajectories: list[np.ndarray] = []
    teacher_current_states: list[np.ndarray] = []
    teacher_time_masks: list[np.ndarray] = []
    teacher_pose_masks: list[np.ndarray] = []
    teacher_gripper_masks: list[np.ndarray] = []
    teacher_current_state_masks: list[np.ndarray] = []
    maximum_effect_error = 0.0
    try:
        for batch_index, batch in enumerate(_loader(dataset, args)):
            ids = [str(value) for value in batch["sample_ids"]]
            sidecar_condition = batch["condition_mean"].numpy().astype(
                np.float32, copy=False
            )
            if effect_index is not None:
                assert effect_splits is not None and effect_means is not None
                missing = [value for value in ids if value not in effect_index]
                if missing:
                    raise ValueError(f"sidecar rows missing from Effect cache: {missing[:4]}")
                cache_rows = np.asarray([effect_index[value] for value in ids], dtype=np.int64)
                if set(effect_splits[cache_rows].tolist()) != {split}:
                    raise ValueError("Effect cache split differs from sidecar split")
                cached_effect = effect_means[cache_rows]
                maximum_effect_error = max(
                    maximum_effect_error,
                    float(np.max(np.abs(cached_effect - sidecar_condition))),
                )
                if maximum_effect_error > 1e-6:
                    raise ValueError(
                        "sidecar Effect values differ from the selected posterior cache"
                    )

            trajectory = batch["trajectory"].to(device, non_blocking=True)
            state = batch["current_state"].to(device, non_blocking=True)
            condition = batch["condition_mean"].to(device, non_blocking=True)
            time_mask = batch["time_mask"].to(device, non_blocking=True)
            if not bool(time_mask.all()):
                raise ValueError(
                    "RoboTwin H36 Stage-2 cache cannot contain padded time slots"
                )
            if not bool(batch["current_state_mask"].all()):
                raise ValueError(
                    "RoboTwin Stage-2 cache requires all 14 current-state values"
                )
            with torch.inference_mode():
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with context:
                    posterior = module.model.posterior(
                        trajectory,
                        state,
                        condition,
                        time_mask=time_mask,
                    )
            action_values.append(posterior.mean.float().cpu().numpy())
            action_logvar_values.append(posterior.logvar.float().cpu().numpy())
            condition_values.append(sidecar_condition.copy())
            teacher_trajectories.append(batch["trajectory"].numpy().astype(np.float32, copy=False))
            teacher_current_states.append(
                batch["current_state"].numpy().astype(np.float32, copy=False)
            )
            teacher_time_masks.append(batch["time_mask"].numpy().astype(np.uint8, copy=False))
            teacher_pose_masks.append(
                batch["arm_pose_mask"].numpy().astype(np.uint8, copy=False)
            )
            teacher_gripper_masks.append(
                batch["arm_gripper_mask"].numpy().astype(np.uint8, copy=False)
            )
            teacher_current_state_masks.append(
                batch["current_state_mask"].numpy().astype(np.uint8, copy=False)
            )
            sample_ids.extend(ids)
            task_ids.extend(str(item["task_id"]) for item in batch["metadata"])
            episode_ids.extend(int(item["episode_id"]) for item in batch["metadata"])
            start_steps.extend(int(item["start_step"]) for item in batch["metadata"])
            if batch_index % 50 == 0:
                print(
                    f"ROBOTWIN_STAGE2_CACHE split={split} rows={len(sample_ids)}/{len(dataset)}",
                    flush=True,
                )
    finally:
        dataset.close()
    if len(sample_ids) != len(dataset) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{split} export did not cover unique sidecar rows exactly once")
    return {
        "sample_id": np.asarray(sample_ids, dtype="U32"),
        "split": np.full(len(sample_ids), split, dtype="U10"),
        "task_id": np.asarray(task_ids, dtype="U128"),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "start_step": np.asarray(start_steps, dtype=np.int32),
        "teacher_end_step": np.asarray(start_steps, dtype=np.int32)
        + DEFAULT_TEMPORAL_CONTRACT.teacher_end_delta,
        "release_future_step": np.asarray(start_steps, dtype=np.int32)
        + DEFAULT_TEMPORAL_CONTRACT.release_future_delta,
        f"z_{condition_cache_name}_raw": np.concatenate(
            condition_values, axis=0
        ).astype(np.float32),
        "z_action_raw": np.concatenate(action_values, axis=0).astype(np.float32),
        "z_action_logvar_raw": np.concatenate(action_logvar_values, axis=0).astype(np.float32),
        "teacher_trajectory": np.concatenate(teacher_trajectories, axis=0).astype(np.float32),
        "teacher_current_state": np.concatenate(teacher_current_states, axis=0).astype(np.float32),
        "teacher_time_mask": np.concatenate(teacher_time_masks, axis=0).astype(np.uint8),
        "teacher_pose_mask": np.concatenate(teacher_pose_masks, axis=0).astype(np.uint8),
        "teacher_gripper_mask": np.concatenate(teacher_gripper_masks, axis=0).astype(np.uint8),
        "teacher_current_state_mask": np.concatenate(
            teacher_current_state_masks, axis=0
        ).astype(np.uint8),
        "physical_valid_count": np.full(
            len(sample_ids),
            DEFAULT_TEMPORAL_CONTRACT.expert_valid_action_steps,
            dtype=np.int16,
        ),
        "effect_cache_max_abs_error": np.asarray(maximum_effect_error, dtype=np.float64),
    }


def _normalization(
    train: dict[str, np.ndarray],
    *,
    condition_source: str,
    condition_cache_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "robotwin-stage2-latent-standardization-v2",
        "fit_split": "train",
        "sample_count": int(len(train["sample_id"])),
        "scale_floor": 1e-6,
        "condition_source": condition_source,
        "condition_cache_name": condition_cache_name,
    }
    for name in (condition_cache_name, "action"):
        value = train[f"z_{name}_raw"].astype(np.float64)
        mean = value.mean(axis=0)
        scale = np.maximum(value.std(axis=0), 1e-6)
        result[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
    return result


def _standardize(
    rows: dict[str, np.ndarray],
    normalization: dict[str, Any],
    *,
    condition_cache_name: str,
) -> None:
    for name in (condition_cache_name, "action"):
        mean = np.asarray(normalization[name]["mean"], dtype=np.float32)
        scale = np.asarray(normalization[name]["scale"], dtype=np.float32)
        rows[f"z_{name}"] = ((rows[f"z_{name}_raw"] - mean) / scale).astype(np.float32)
    action_scale = np.asarray(normalization["action"]["scale"], dtype=np.float32)
    rows["z_action_logvar"] = (
        rows["z_action_logvar_raw"] - 2.0 * np.log(action_scale)
    ).astype(np.float32)


def export(args: argparse.Namespace) -> dict[str, Any]:
    temporal = resolve_robotwin_stage2_temporal_contract(args.temporal_contract)
    inputs = _validate_inputs(args, temporal)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable cache: {output_root}")
    temporary = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    module = RoboTwinActionBetaVAELightningModule.load_from_checkpoint(
        inputs["action_checkpoint"],
        map_location="cpu",
        normalization_path=inputs["sidecar_root"] / "stats" / "normalization.json",
        condition_source=inputs["condition_source"],
    )
    if module.variant != "a1":
        raise ValueError("paired cache requires ActionAE A1")
    module.requires_grad_(False).eval().to(device)
    if inputs["effect_source"] == "legacy_cache":
        effect_index, effect_splits, effect_means = _effect_cache(inputs["effect_cache"])
    else:
        effect_index, effect_splits, effect_means = None, None, None
    try:
        split_rows = {
            split: _collect(
                module=module,
                sidecar_root=inputs["sidecar_root"],
                split=split,
                effect_index=effect_index,
                effect_splits=effect_splits,
                effect_means=effect_means,
                args=args,
                device=device,
                condition_source=inputs["condition_source"],
                condition_cache_name=inputs["condition_cache_name"],
            )
            for split in ("train", "validation", "test")
        }
        all_ids = np.concatenate([split_rows[split]["sample_id"] for split in split_rows])
        if len(all_ids) != temporal.row_count:
            raise ValueError(
                f"paired cache contains {len(all_ids)} rows, expected {temporal.row_count}"
            )
        if effect_index is not None and (
            len(all_ids) != len(effect_index)
            or set(all_ids.tolist()) != set(effect_index)
        ):
            raise ValueError("paired cache and Effect cache do not have an exact sample-ID set join")
        normalization = _normalization(
            split_rows["train"],
            condition_source=inputs["condition_source"],
            condition_cache_name=inputs["condition_cache_name"],
        )
        for rows in split_rows.values():
            _standardize(
                rows,
                normalization,
                condition_cache_name=inputs["condition_cache_name"],
            )

        cache_fields = {}
        for split in ("train", "validation", "test"):
            for key, value in split_rows[split].items():
                if key == "effect_cache_max_abs_error":
                    continue
                cache_fields.setdefault(key, []).append(value)
        packed = {key: np.concatenate(value, axis=0) for key, value in cache_fields.items()}
        if (
            not np.isfinite(
                packed[f"z_{inputs['condition_cache_name']}"]
            ).all()
            or not np.isfinite(packed["z_action"]).all()
            or not np.isfinite(packed["z_action_logvar"]).all()
        ):
            raise ValueError("standardized paired latent cache contains NaN/Inf")

        cache_path = temporary / "paired_latents.npz"
        np.savez_compressed(cache_path, **packed)
        normalization_path = temporary / "latent_normalization.json"
        normalization_path.write_text(
            json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        condition_bundle = {
            "checkpoint": str(
                inputs.get("condition_checkpoint")
                or inputs.get("effect_checkpoint")
                or ""
            ),
            "checkpoint_sha256": inputs["condition_checkpoint_sha256"],
            "config": str(inputs.get("condition_config") or inputs.get("effect_config") or ""),
            "config_sha256": inputs.get("condition_config_sha256")
            or inputs.get("effect_config_sha256"),
            "source": inputs["condition_source"],
            "posterior_statistic": "mean",
            "sampling": False,
            "cache_name": inputs["condition_cache_name"],
        }
        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "temporal_contract": temporal.to_dict(),
            "condition": condition_bundle,
            "action": {
                "checkpoint": str(inputs["action_checkpoint"]),
                "checkpoint_sha256": sha256_file(inputs["action_checkpoint"]),
                "checkpoint_epoch": int(inputs["action_payload"].get("epoch", -1)),
                "checkpoint_global_step": int(inputs["action_payload"]["global_step"]),
                "variant": "a1",
                "sampling": False,
                "teacher_value": "posterior_mean",
                "distribution": {
                    "prior": "standard_normal",
                    "latent_dim": 32,
                    "parameter_dim": 64,
                    "layout": "mean_then_logvar",
                    "standardized_logvar": "logvar-2*log(scale)",
                },
                "schema_version": ACTION_SCHEMA_VERSION,
            },
            "sidecar": {
                "root": str(inputs["sidecar_root"]),
                "checksums_sha256": sha256_file(inputs["sidecar_root"] / "checksums.sha256"),
                "provenance_sha256": sha256_file(inputs["sidecar_root"] / "provenance.json"),
                "binding": inputs.get("sidecar_binding", "exact"),
                "binding_metadata": inputs.get("sidecar_binding_metadata"),
            },
            "cache": {
                "path": str(output_root / cache_path.name),
                "sha256": sha256_file(cache_path),
                "rows": int(len(packed["sample_id"])),
                "split_rows": {
                    split: int(len(split_rows[split]["sample_id"])) for split in split_rows
                },
                "exact_condition_join": True,
                "condition_cache_max_abs_error": max(
                    float(split_rows[split]["effect_cache_max_abs_error"])
                    for split in split_rows
                ),
            },
        }
        if inputs["condition_source"] == "e39":
            effect_selection = inputs["effect_selection"]
            assert effect_selection is not None
            bundle["effect"] = {
                **condition_bundle,
                "selection": str(effect_selection),
                "selection_sha256": sha256_file(effect_selection),
                "selected_checkpoint_id": inputs["selection"]["selected_checkpoint_id"],
                "source": inputs["effect_source"],
                "posterior_cache": (
                    str(inputs["effect_cache"])
                    if inputs["effect_cache"] is not None
                    else None
                ),
                "posterior_cache_sha256": inputs.get("effect_cache_sha256"),
                "sidecar_provenance_sha256": inputs.get("sidecar_provenance_sha256"),
            }
        bundle_path = temporary / "teacher_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksum_lines = [
            f"{sha256_file(temporary / name)}  {name}"
            for name in ("paired_latents.npz", "latent_normalization.json", "teacher_bundle.json")
        ]
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        (temporary / "_SUCCESS").write_text(sha256_file(checksums_path) + "\n", encoding="ascii")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument(
        "--condition-source",
        choices=("auto", "e39", "lawam_idm"),
        default="auto",
    )
    parser.add_argument("--condition-checkpoint", type=Path)
    parser.add_argument("--condition-config", type=Path)
    parser.add_argument("--effect-checkpoint", type=Path)
    parser.add_argument("--effect-selection", type=Path)
    parser.add_argument("--effect-config", type=Path)
    parser.add_argument(
        "--effect-source",
        choices=("legacy_cache", "sidecar_e39_mean", "sidecar_condition_mean"),
        default="legacy_cache",
    )
    parser.add_argument("--effect-cache", type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--sidecar-binding",
        choices=("exact", "semantic", "contract"),
        default="exact",
        help="Use exact legacy aggregate hashes or the portable semantic release gate.",
    )
    parser.add_argument("--sidecar-reference-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temporal-contract", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    if (
        args.condition_source == "e39"
        and args.effect_source == "legacy_cache"
        and args.effect_cache is None
    ):
        parser.error("--effect-cache is required with --effect-source=legacy_cache")
    if args.condition_source == "e39" and (
        args.effect_checkpoint is None or args.effect_selection is None
    ):
        parser.error("e39 condition export requires --effect-checkpoint and --effect-selection")
    if args.condition_source == "lawam_idm" and args.effect_source != "sidecar_condition_mean":
        parser.error("lawam_idm requires --effect-source=sidecar_condition_mean")
    if args.sidecar_binding == "contract" and args.effect_source != "sidecar_condition_mean":
        parser.error("contract sidecar binding requires --effect-source=sidecar_condition_mean")
    if args.sidecar_binding == "semantic":
        if args.effect_source != "sidecar_e39_mean":
            parser.error("semantic sidecar binding requires --effect-source=sidecar_e39_mean")
        if args.sidecar_reference_root is None:
            parser.error("--sidecar-reference-root is required with --sidecar-binding=semantic")
        if args.effect_config is None:
            parser.error("--effect-config is required with --sidecar-binding=semantic")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    bundle = export(parse_args(argv))
    print(json.dumps(bundle["cache"], sort_keys=True))


if __name__ == "__main__":
    main()
