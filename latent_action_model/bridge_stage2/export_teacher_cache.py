"""Export the immutable exact-row Bridge Stage-2 paired-teacher cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from latent_action_model.action_ae.data import ActionSidecarDataset, action_sidecar_collate
from latent_action_model.action_ae.evaluate import load_action_ae_checkpoint

from .data import sha256_file
from .temporal import (
    BRIDGE_STAGE2_SPLIT_ROWS,
    DEFAULT_TEMPORAL_CONTRACT,
    BridgeStage2TemporalContract,
    resolve_bridge_stage2_temporal_contract,
)


CACHE_SCHEMA_VERSION = "bridge-stage2-paired-teacher-cache-v1"
BUNDLE_SCHEMA_VERSION = "bridge-stage2-teacher-bundle-v1"
ACTION_SCHEMA_VERSION = "action-tcn-beta-vae-v1"
EXPECTED_EFFECT_CHECKPOINT_SHA256 = (
    "bbb2aae91353c2d21009adc9027281b5d42c7c1c539d5acddabce3ebca0867b0"
)
EXPECTED_ACTION_CHECKPOINT_SHA256 = (
    "5edf2556414cfcc1f645b58c4d3767b7e746dd2e665ad451c84e700f6f7c1c29"
)


class _SelectedRows(Dataset[dict[str, Any]]):
    def __init__(self, source: ActionSidecarDataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = tuple(int(value) for value in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[self.indices[int(index)]]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _manifest_path(root: Path, split: str) -> Path:
    candidate = root / f"{split}.parquet"
    if not candidate.is_file() and (root / "manifests").is_dir():
        candidate = root / "manifests" / f"{split}.parquet"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _validate_inputs(
    args: argparse.Namespace,
    temporal: BridgeStage2TemporalContract,
) -> dict[str, Any]:
    sidecar_root = args.action_sidecar_root.expanduser().resolve()
    manifest_root = args.stage2_manifest_root.expanduser().resolve()
    effect_checkpoint = args.effect_checkpoint.expanduser().resolve()
    effect_config = args.effect_config.expanduser().resolve()
    action_checkpoint = args.action_checkpoint.expanduser().resolve()
    for path in (sidecar_root, manifest_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (effect_checkpoint, effect_config, action_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(effect_checkpoint) != EXPECTED_EFFECT_CHECKPOINT_SHA256:
        raise ValueError("Bridge cache must use the validation-selected epoch-00 Effect teacher")
    if sha256_file(action_checkpoint) != EXPECTED_ACTION_CHECKPOINT_SHA256:
        raise ValueError("Bridge cache must use the selected step-0007503 Gaussian ActionVAE")
    sidecar_provenance = _read_json(sidecar_root / "provenance.json")
    e39 = sidecar_provenance.get("e39")
    if not isinstance(e39, dict) or e39.get("checkpoint_sha256") != EXPECTED_EFFECT_CHECKPOINT_SHA256:
        raise ValueError("Bridge Action sidecar is not bound to the selected Effect teacher")
    sidecar_schema = _read_json(sidecar_root / "schema.json")
    if (
        int(sidecar_schema.get("horizon_actions", -1)) != temporal.representation_steps
        or int(sidecar_schema.get("physical_action_steps", -1)) != temporal.physical_valid_count
        or float(sidecar_schema.get("source_control_hz", -1.0)) != temporal.control_hz
        or float(sidecar_schema.get("representation_hz", -1.0)) != temporal.representation_hz
    ):
        raise ValueError("Bridge Action sidecar temporal representation is incompatible")
    stage2_schema = _read_json(manifest_root.parent / "schema.json")
    if (
        int(stage2_schema.get("physical_action_steps", -1)) != temporal.physical_valid_count
        or int(stage2_schema.get("dino_future_delta_steps", -1))
        != temporal.dino_future_delta_steps
        or int(stage2_schema.get("effect_teacher_delta_steps", -1))
        != temporal.effect_teacher_delta_steps
        or float(stage2_schema.get("action_hz", -1.0)) != temporal.control_hz
    ):
        raise ValueError("Bridge Stage-2 manifest schema is incompatible")
    return {
        "sidecar_root": sidecar_root,
        "manifest_root": manifest_root,
        "effect_checkpoint": effect_checkpoint,
        "effect_config": effect_config,
        "action_checkpoint": action_checkpoint,
        "sidecar_provenance": sidecar_provenance,
    }


def _selected_dataset(
    *,
    sidecar_root: Path,
    manifest_root: Path,
    split: str,
    temporal: BridgeStage2TemporalContract,
) -> tuple[ActionSidecarDataset, _SelectedRows, pd.DataFrame]:
    manifest = pd.read_parquet(_manifest_path(manifest_root, split))
    if len(manifest) != BRIDGE_STAGE2_SPLIT_ROWS[split]:
        raise ValueError(f"Bridge {split} Stage-2 manifest row count mismatch")
    if set(manifest["split"].astype(str)) != {split}:
        raise ValueError(f"Bridge manifest must contain only split={split}")
    for row in manifest.to_dict("records"):
        temporal.validate_manifest_row(row)
    source = ActionSidecarDataset(
        sidecar_root,
        split,
        require_e39=True,
        verify_row_identity=True,
    )
    source_ids = source.sample_ids
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    if len(lookup) != len(source_ids):
        raise ValueError("Bridge Action sidecar contains duplicate sample IDs")
    manifest_ids = manifest["sample_id"].astype(str).tolist()
    missing = [sample_id for sample_id in manifest_ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"Bridge Stage-2 rows missing from Action sidecar: {missing[:4]}")
    selected = _SelectedRows(source, [lookup[sample_id] for sample_id in manifest_ids])
    return source, selected, manifest.reset_index(drop=True)


def _collect_split(
    *,
    model: torch.nn.Module,
    sidecar_root: Path,
    manifest_root: Path,
    split: str,
    temporal: BridgeStage2TemporalContract,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    source, selected, manifest = _selected_dataset(
        sidecar_root=sidecar_root,
        manifest_root=manifest_root,
        split=split,
        temporal=temporal,
    )
    loader = DataLoader(
        selected,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=action_sidecar_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "z_effect_raw",
            "z_action_raw",
            "z_action_logvar_raw",
            "teacher_trajectory",
            "teacher_current_state",
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
            "physical_valid_count",
        )
    }
    observed_ids: list[str] = []
    try:
        for batch_index, batch in enumerate(loader):
            ids = [str(value) for value in batch["sample_ids"]]
            if not bool(batch["e39_valid"].all()):
                raise ValueError("Bridge Stage-2 selected one row without z-effect")
            if not bool(batch["time_mask"].all()):
                raise ValueError("Bridge 24-slot interpolated trajectory must be fully valid")
            expected_state_mask = torch.zeros_like(
                batch["current_state_mask"], dtype=torch.bool
            )
            expected_state_mask[:, :7] = True
            if not torch.equal(
                batch["current_state_mask"].to(dtype=torch.bool), expected_state_mask
            ):
                raise ValueError("Bridge teacher current-state mask must be true7/false7")
            if not bool((batch["physical_valid_count"] == temporal.physical_valid_count).all()):
                raise ValueError("Bridge physical_valid_count must be six")
            trajectory = batch["trajectory"].to(device, non_blocking=True)
            current_state = batch["current_state"].to(device, non_blocking=True)
            effect = batch["e39_mean"].to(device, non_blocking=True)
            time_mask = batch["time_mask"].to(device, non_blocking=True)
            with torch.inference_mode():
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with context:
                    posterior = model.posterior(
                        trajectory,
                        current_state,
                        effect,
                        time_mask=time_mask,
                    )
            values["z_effect_raw"].append(batch["e39_mean"].numpy().astype(np.float32))
            values["z_action_raw"].append(posterior.mean.float().cpu().numpy())
            values["z_action_logvar_raw"].append(posterior.logvar.float().cpu().numpy())
            values["teacher_trajectory"].append(batch["trajectory"].numpy().astype(np.float32))
            values["teacher_current_state"].append(
                batch["current_state"].numpy().astype(np.float32)
            )
            for source_name, target_name in (
                ("time_mask", "teacher_time_mask"),
                ("pose_mask", "teacher_pose_mask"),
                ("gripper_mask", "teacher_gripper_mask"),
                ("current_state_mask", "teacher_current_state_mask"),
            ):
                values[target_name].append(batch[source_name].numpy().astype(np.uint8))
            values["physical_valid_count"].append(
                batch["physical_valid_count"].numpy().astype(np.int16)
            )
            observed_ids.extend(ids)
            if batch_index % 50 == 0:
                print(
                    f"BRIDGE_STAGE2_CACHE split={split} rows={len(observed_ids)}/{len(selected)}",
                    flush=True,
                )
    finally:
        source.close()
    expected_ids = manifest["sample_id"].astype(str).tolist()
    if observed_ids != expected_ids:
        raise ValueError("Bridge cache export changed exact manifest row order")
    result = {
        "sample_id": np.asarray(observed_ids, dtype="U32"),
        "split": np.full(len(observed_ids), split, dtype="U10"),
        "task_id": manifest["task_id"].astype(str).to_numpy(dtype="U64"),
        "episode_id": manifest["episode_id"].to_numpy(dtype=np.int64),
        "start_step": manifest["start_step"].to_numpy(dtype=np.int32),
        "dino_future_step": manifest["dino_future_step"].to_numpy(dtype=np.int32),
        "effect_teacher_step": manifest["effect_teacher_step"].to_numpy(dtype=np.int32),
    }
    result.update({name: np.concatenate(chunks, axis=0) for name, chunks in values.items()})
    return result


def _fit_normalization(train: dict[str, np.ndarray], temporal: BridgeStage2TemporalContract) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": temporal.normalization_schema_version,
        "fit_split": "train",
        "sample_count": int(len(train["sample_id"])),
        "scale_floor": 1.0e-6,
    }
    for name in ("effect", "action"):
        values = train[f"z_{name}_raw"].astype(np.float64)
        result[name] = {
            "mean": values.mean(axis=0).tolist(),
            "scale": np.maximum(values.std(axis=0), 1.0e-6).tolist(),
        }
    return result


def _standardize(rows: dict[str, np.ndarray], normalization: dict[str, Any]) -> None:
    for name in ("effect", "action"):
        mean = np.asarray(normalization[name]["mean"], dtype=np.float32)
        scale = np.asarray(normalization[name]["scale"], dtype=np.float32)
        rows[f"z_{name}"] = ((rows[f"z_{name}_raw"] - mean) / scale).astype(np.float32)
    action_scale = np.asarray(normalization["action"]["scale"], dtype=np.float32)
    rows["z_action_logvar"] = (
        rows["z_action_logvar_raw"] - 2.0 * np.log(action_scale)
    ).astype(np.float32)


def export(args: argparse.Namespace) -> dict[str, Any]:
    temporal = resolve_bridge_stage2_temporal_contract(args.temporal_contract)
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
    loaded = load_action_ae_checkpoint(
        inputs["action_checkpoint"],
        inputs["sidecar_root"],
        device=device,
    )
    if loaded.metadata.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("Bridge paired cache requires a standard-normal Action beta-VAE")
    model_config = loaded.metadata.get("model_config", {})
    expected_model = {
        "variant": "a1",
        "horizon": temporal.representation_steps,
        "trajectory_dim": temporal.trajectory_dim,
        "state_dim": temporal.state_dim,
        "latent_dim": 32,
    }
    mismatches = {
        name: (model_config.get(name), expected)
        for name, expected in expected_model.items()
        if model_config.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Bridge ActionVAE architecture mismatch: {mismatches}")
    try:
        split_rows = {
            split: _collect_split(
                model=loaded.model,
                sidecar_root=inputs["sidecar_root"],
                manifest_root=inputs["manifest_root"],
                split=split,
                temporal=temporal,
                args=args,
                device=device,
            )
            for split in ("train", "validation", "test")
        }
        normalization = _fit_normalization(split_rows["train"], temporal)
        for rows in split_rows.values():
            _standardize(rows, normalization)
        packed: dict[str, np.ndarray] = {}
        for split in ("train", "validation", "test"):
            for name, value in split_rows[split].items():
                packed.setdefault(name, []).append(value)  # type: ignore[arg-type]
        packed = {
            name: np.concatenate(chunks, axis=0)  # type: ignore[arg-type]
            for name, chunks in packed.items()
        }
        if len(packed["sample_id"]) != temporal.row_count:
            raise ValueError("Bridge paired cache row count mismatch")
        cache_path = temporary / "paired_latents.npz"
        np.savez_compressed(cache_path, **packed)
        normalization_path = temporary / "latent_normalization.json"
        normalization_path.write_text(
            json.dumps(normalization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "temporal_contract": temporal.to_dict(),
            "effect": {
                "checkpoint": str(inputs["effect_checkpoint"]),
                "checkpoint_sha256": sha256_file(inputs["effect_checkpoint"]),
                "config": str(inputs["effect_config"]),
                "config_sha256": sha256_file(inputs["effect_config"]),
                "teacher_value": "deterministic_posterior_mean",
                "sampling": False,
            },
            "action": {
                "checkpoint": str(inputs["action_checkpoint"]),
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "schema_version": ACTION_SCHEMA_VERSION,
                "variant": "a1",
                "teacher_value": "posterior_mean",
                "sampling": False,
                "distribution": {
                    "prior": "standard_normal",
                    "latent_dim": 32,
                    "layout": "mean_then_logvar",
                    "standardized_logvar": "logvar-2*log(scale)",
                },
            },
            "sidecar": {
                "root": str(inputs["sidecar_root"]),
                "checksums_sha256": sha256_file(inputs["sidecar_root"] / "checksums.sha256"),
                "provenance_sha256": sha256_file(inputs["sidecar_root"] / "provenance.json"),
            },
            "manifest": {
                "root": str(inputs["manifest_root"]),
                "split_sha256": {
                    split: sha256_file(_manifest_path(inputs["manifest_root"], split))
                    for split in ("train", "validation", "test")
                },
            },
            "cache": {
                "path": str(output_root / cache_path.name),
                "sha256": sha256_file(cache_path),
                "rows": temporal.row_count,
                "split_rows": dict(BRIDGE_STAGE2_SPLIT_ROWS),
                "exact_manifest_action_join": True,
            },
        }
        bundle_path = temporary / "teacher_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        names = ("paired_latents.npz", "latent_normalization.json", "teacher_bundle.json")
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text(
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in names),
            encoding="ascii",
        )
        (temporary / "_SUCCESS").write_text(
            sha256_file(checksums_path) + "\n", encoding="ascii"
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-sidecar-root", required=True, type=Path)
    parser.add_argument("--stage2-manifest-root", required=True, type=Path)
    parser.add_argument("--effect-checkpoint", required=True, type=Path)
    parser.add_argument("--effect-config", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temporal-contract", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    bundle = export(parse_args(argv))
    print(json.dumps(bundle["cache"], sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION",
    "CACHE_SCHEMA_VERSION",
    "export",
    "main",
    "parse_args",
]
