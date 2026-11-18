"""Paired held-out evaluation for RoboTwin ActionAE A0/A1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.nn.functional as functional
from torch import Tensor

from latent_action_model.action_ae.losses import so3_geodesic_error

from .data import RoboTwinActionDataModule
from .lightning import RoboTwinActionAELightningModule


SCHEMA_VERSION = "robotwin-action-ae-held-out-evaluation-v1"
METRIC_NAMES = (
    "loss",
    "position_l2_cm",
    "rotation_geodesic_deg",
    "gripper_mae",
    "left_position_l2_cm",
    "left_rotation_geodesic_deg",
    "left_gripper_mae",
    "right_position_l2_cm",
    "right_rotation_geodesic_deg",
    "right_gripper_mae",
    "endpoint_position_l2_cm",
    "endpoint_rotation_geodesic_deg",
    "endpoint_gripper_mae",
)


@dataclass(frozen=True)
class TestSplit:
    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    trajectory: Tensor
    trajectory_raw: Tensor
    current_state: Tensor
    e39_mean: Tensor
    time_mask: Tensor
    arm_pose_mask: Tensor
    arm_gripper_mask: Tensor

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True)
class LoadedModel:
    module: RoboTwinActionAELightningModule
    path: Path
    sha256: str
    epoch: int
    global_step: int


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


def deterministic_derangement(size: int, seed: int) -> tuple[Tensor, int]:
    if size < 2:
        raise ValueError("derangement requires at least two samples")
    rng = np.random.default_rng(int(seed))
    offset = int(rng.integers(1, size))
    permutation = torch.roll(torch.arange(size, dtype=torch.long), shifts=offset)
    if torch.any(permutation == torch.arange(size)):
        raise AssertionError("circular derangement contains a fixed point")
    return permutation, offset


def grouped_derangement(groups: Sequence[str], seed: int) -> tuple[Tensor, dict[str, Any]]:
    if len(groups) < 2:
        raise ValueError("grouped derangement requires at least two samples")
    permutation = torch.empty(len(groups), dtype=torch.long)
    offsets: dict[str, int] = {}
    for group in sorted(set(str(value) for value in groups)):
        indices = torch.tensor(
            [index for index, value in enumerate(groups) if str(value) == group], dtype=torch.long
        )
        if len(indices) < 2:
            raise ValueError(f"shuffle group {group!r} has fewer than two samples")
        group_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()[:8], "little"
        )
        _, offset = deterministic_derangement(len(indices), group_seed)
        permutation[indices] = torch.roll(indices, shifts=offset)
        offsets[group] = offset
    identity = torch.arange(len(groups), dtype=torch.long)
    if torch.any(permutation == identity):
        raise AssertionError("grouped derangement contains a fixed point")
    if any(str(groups[index]) != str(groups[donor]) for index, donor in enumerate(permutation.tolist())):
        raise AssertionError("grouped derangement crossed group boundaries")
    return permutation, {"group_count": len(offsets), "offsets": offsets}


def _manifest_identity(root: Path) -> tuple[tuple[str, ...], tuple[str, ...], Path]:
    path = root / "manifests/test.parquet"
    table = parquet.read_table(path, columns=["sample_id", "task_id"])
    sample_ids = tuple(str(value) for value in table.column("sample_id").to_pylist())
    task_ids = tuple(str(value) for value in table.column("task_id").to_pylist())
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"test manifest contains duplicate sample IDs: {path}")
    return sample_ids, task_ids, path


def _verify_paired_releases(a0_root: Path, a1_root: Path) -> dict[str, Any]:
    a0_ids, a0_tasks, a0_manifest = _manifest_identity(a0_root)
    a1_ids, a1_tasks, a1_manifest = _manifest_identity(a1_root)
    if a0_ids != a1_ids or a0_tasks != a1_tasks:
        raise ValueError("A0/A1 test manifests are not row-identical")
    a0_norm = a0_root / "stats/normalization.json"
    a1_norm = a1_root / "stats/normalization.json"
    a0_norm_sha = _sha256_file(a0_norm)
    a1_norm_sha = _sha256_file(a1_norm)
    if a0_norm_sha != a1_norm_sha:
        raise ValueError("A0/A1 trajectory normalization differs")
    return {
        "sample_count": len(a0_ids),
        "task_count": len(set(a0_tasks)),
        "sample_ids_sha256": _sha256_strings(a0_ids),
        "task_ids_sha256": _sha256_strings(a0_tasks),
        "normalization_sha256": a0_norm_sha,
        "a0_test_manifest": str(a0_manifest),
        "a0_test_manifest_sha256": _sha256_file(a0_manifest),
        "a1_test_manifest": str(a1_manifest),
        "a1_test_manifest_sha256": _sha256_file(a1_manifest),
    }


def collect_test_split(
    data_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    log_every_batches: int,
) -> TestSplit:
    data = RoboTwinActionDataModule(
        data_root,
        variant="a1",
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        verify_row_identity=True,
    )
    data.setup("test")
    assert data.test_dataset is not None
    count = len(data.test_dataset)
    tensors = {
        "trajectory": torch.empty(count, 48, 14),
        "trajectory_raw": torch.empty(count, 48, 14),
        "current_state": torch.empty(count, 14),
        "e39_mean": torch.empty(count, 32),
        "time_mask": torch.empty(count, 48, dtype=torch.bool),
        "arm_pose_mask": torch.empty(count, 48, 2, dtype=torch.bool),
        "arm_gripper_mask": torch.empty(count, 48, 2, dtype=torch.bool),
    }
    sample_ids: list[str | None] = [None] * count
    task_ids: list[str | None] = [None] * count
    seen = torch.zeros(count, dtype=torch.bool)
    try:
        for batch_index, batch in enumerate(data.test_dataloader(), start=1):
            indices = batch["dataset_indices"].long()
            if torch.any(seen[indices]):
                raise ValueError("test loader emitted a dataset row more than once")
            seen[indices] = True
            for name in tensors:
                tensors[name][indices] = batch[name]
            for dataset_index, sample_id, metadata in zip(
                indices.tolist(), batch["sample_ids"], batch["metadata"]
            ):
                sample_ids[dataset_index] = str(sample_id)
                task_ids[dataset_index] = str(metadata["task_id"])
            if batch_index % log_every_batches == 0:
                print(f"ROBOTWIN_EVAL_COLLECT batches={batch_index} rows={int(seen.sum())}/{count}", flush=True)
    finally:
        data.teardown(None)
    if not torch.all(seen) or any(value is None for value in sample_ids + task_ids):
        raise ValueError("test loader did not cover the full held-out split")
    return TestSplit(
        sample_ids=tuple(str(value) for value in sample_ids),
        task_ids=tuple(str(value) for value in task_ids),
        **tensors,
    )


def load_model(path: Path, *, expected_variant: str, device: torch.device) -> LoadedModel:
    resolved = path.expanduser().resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    module = RoboTwinActionAELightningModule.load_from_checkpoint(resolved, map_location="cpu")
    if module.variant != expected_variant:
        raise ValueError(f"expected checkpoint variant={expected_variant}, got {module.variant}")
    module.requires_grad_(False)
    module.eval().to(device)
    return LoadedModel(
        module=module,
        path=resolved,
        sha256=_sha256_file(resolved),
        epoch=int(checkpoint["epoch"]),
        global_step=int(checkpoint["global_step"]),
    )


def _masked_mean_per_sample(values: Tensor, valid: Tensor) -> Tensor:
    values = values.float()
    valid = valid.bool()
    flat_values = torch.where(valid, values, torch.zeros_like(values)).flatten(1)
    flat_valid = valid.flatten(1)
    count = flat_valid.sum(dim=1)
    if torch.any(count == 0):
        raise ValueError("metric mask contains a sample without valid elements")
    return flat_values.sum(dim=1) / count.to(flat_values.dtype)


def _last_valid_per_sample(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid.bool()
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("endpoint values and mask must have matching [B,T] shapes")
    if torch.any(~valid.any(dim=1)):
        raise ValueError("endpoint mask contains a sample without valid steps")
    reverse = valid.flip(1).to(torch.int64).argmax(dim=1)
    indices = valid.shape[1] - 1 - reverse
    return values.gather(1, indices[:, None]).squeeze(1)


def per_sample_metrics(
    module: RoboTwinActionAELightningModule,
    prediction: Tensor,
    target_normalized: Tensor,
    target_raw: Tensor,
    *,
    time_mask: Tensor,
    arm_pose_mask: Tensor,
    arm_gripper_mask: Tensor,
) -> dict[str, Tensor]:
    time = time_mask.bool()
    pose = arm_pose_mask.bool() & time.unsqueeze(-1)
    gripper = arm_gripper_mask.bool() & time.unsqueeze(-1)
    position_prediction = torch.cat((prediction[..., 0:3], prediction[..., 7:10]), dim=-1)
    position_target = torch.cat((target_normalized[..., 0:3], target_normalized[..., 7:10]), dim=-1)
    position_valid = torch.cat(
        (pose[..., 0:1].expand(-1, -1, 3), pose[..., 1:2].expand(-1, -1, 3)), dim=-1
    )
    rotation_prediction = torch.cat((prediction[..., 3:6], prediction[..., 10:13]), dim=-1)
    rotation_target = torch.cat((target_normalized[..., 3:6], target_normalized[..., 10:13]), dim=-1)
    gripper_prediction = torch.stack((prediction[..., 6], prediction[..., 13]), dim=-1)
    gripper_target = torch.stack((target_normalized[..., 6], target_normalized[..., 13]), dim=-1)
    position_loss = _masked_mean_per_sample(
        functional.smooth_l1_loss(position_prediction, position_target, beta=module.beta, reduction="none"),
        position_valid,
    )
    rotation_loss = _masked_mean_per_sample(
        functional.smooth_l1_loss(rotation_prediction, rotation_target, beta=module.beta, reduction="none"),
        position_valid,
    )
    gripper_loss = _masked_mean_per_sample(
        functional.smooth_l1_loss(gripper_prediction, gripper_target, beta=module.beta, reduction="none"),
        gripper,
    )
    prediction_raw = module._denormalize(prediction).float()
    arm_specs = {
        "left": (slice(0, 3), slice(3, 6), 6, 0),
        "right": (slice(7, 10), slice(10, 13), 13, 1),
    }
    arm_position: list[Tensor] = []
    arm_rotation: list[Tensor] = []
    arm_gripper: list[Tensor] = []
    endpoint_position: list[Tensor] = []
    endpoint_rotation: list[Tensor] = []
    endpoint_gripper: list[Tensor] = []
    result: dict[str, Tensor] = {
        "loss": (
            module.position_weight * position_loss
            + module.rotation_weight * rotation_loss
            + module.gripper_weight * gripper_loss
        )
    }
    for arm, (position_slice, rotation_slice, gripper_index, mask_index) in arm_specs.items():
        pose_valid = pose[..., mask_index]
        gripper_valid = gripper[..., mask_index]
        position = torch.linalg.vector_norm(
            prediction_raw[..., position_slice] - target_raw[..., position_slice], dim=-1
        ) * 100.0
        rotation = so3_geodesic_error(
            prediction_raw[..., rotation_slice], target_raw[..., rotation_slice], degrees=True
        )
        grip = (prediction_raw[..., gripper_index] - target_raw[..., gripper_index]).abs()
        result[f"{arm}_position_l2_cm"] = _masked_mean_per_sample(position, pose_valid)
        result[f"{arm}_rotation_geodesic_deg"] = _masked_mean_per_sample(rotation, pose_valid)
        result[f"{arm}_gripper_mae"] = _masked_mean_per_sample(grip, gripper_valid)
        arm_position.append(position)
        arm_rotation.append(rotation)
        arm_gripper.append(grip)
        endpoint_position.append(_last_valid_per_sample(position, pose_valid))
        endpoint_rotation.append(_last_valid_per_sample(rotation, pose_valid))
        endpoint_gripper.append(_last_valid_per_sample(grip, gripper_valid))
    result.update(
        position_l2_cm=_masked_mean_per_sample(torch.stack(arm_position, dim=-1), pose),
        rotation_geodesic_deg=_masked_mean_per_sample(torch.stack(arm_rotation, dim=-1), pose),
        gripper_mae=_masked_mean_per_sample(torch.stack(arm_gripper, dim=-1), gripper),
        endpoint_position_l2_cm=torch.stack(endpoint_position, dim=1).mean(dim=1),
        endpoint_rotation_geodesic_deg=torch.stack(endpoint_rotation, dim=1).mean(dim=1),
        endpoint_gripper_mae=torch.stack(endpoint_gripper, dim=1).mean(dim=1),
    )
    if set(result) != set(METRIC_NAMES):
        raise AssertionError(f"metric set mismatch: {sorted(result)}")
    return result


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate_condition(
    loaded: LoadedModel,
    split: TestSplit,
    *,
    device: torch.device,
    batch_size: int,
    effect: Tensor | None,
    z_action: Tensor | None = None,
    return_latent: bool = False,
    label: str,
) -> tuple[dict[str, Tensor], Tensor | None]:
    count = split.sample_count
    metrics = {name: torch.empty(count) for name in METRIC_NAMES}
    latent = torch.empty(count, 1, 32) if return_latent else None
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        trajectory = split.trajectory[start:stop].to(device, non_blocking=True)
        state = split.current_state[start:stop].to(device, non_blocking=True)
        time = split.time_mask[start:stop].to(device, non_blocking=True)
        selected_effect = effect[start:stop].to(device, non_blocking=True) if effect is not None else None
        with torch.inference_mode(), _autocast(device):
            if z_action is None:
                output = loaded.module.model(trajectory, state, selected_effect, time_mask=time)
                prediction = output.reconstruction
                batch_latent = output.z_action
            else:
                batch_latent = z_action[start:stop].to(device, non_blocking=True)
                prediction = loaded.module.model.decode(batch_latent, state, selected_effect)
        batch_metrics = per_sample_metrics(
            loaded.module,
            prediction,
            trajectory,
            split.trajectory_raw[start:stop].to(device, non_blocking=True),
            time_mask=time,
            arm_pose_mask=split.arm_pose_mask[start:stop].to(device, non_blocking=True),
            arm_gripper_mask=split.arm_gripper_mask[start:stop].to(device, non_blocking=True),
        )
        for name, values in batch_metrics.items():
            metrics[name][start:stop] = values.detach().cpu()
        if latent is not None:
            latent[start:stop] = batch_latent.detach().float().cpu()
    print(f"ROBOTWIN_EVAL_CONDITION label={label} rows={count}", flush=True)
    return metrics, latent


def _metric_summary(values: Tensor) -> dict[str, float]:
    array = values.double().numpy()
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _summarize_conditions(conditions: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Any]:
    return {
        condition: {metric: _metric_summary(values) for metric, values in metrics.items()}
        for condition, metrics in conditions.items()
    }


def _task_balanced_delta(
    baseline: Tensor,
    candidate: Tensor,
    task_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    delta = (baseline.double() - candidate.double()).numpy()
    tasks = sorted(set(task_ids))
    task_delta = np.asarray(
        [np.mean(delta[[index for index, value in enumerate(task_ids) if value == task]]) for task in tasks],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(tasks), size=(replicates, len(tasks)))
    bootstrap = task_delta[draws].mean(axis=1)
    return {
        "direction": "baseline_minus_candidate; positive means candidate has lower error",
        "window_micro_mean": float(delta.mean()),
        "window_nonnegative_fraction": float(np.mean(delta >= 0.0)),
        "task_balanced_mean": float(task_delta.mean()),
        "task_balanced_median": float(np.median(task_delta)),
        "nonnegative_task_count": int(np.sum(task_delta >= 0.0)),
        "task_count": len(tasks),
        "paired_bootstrap_replicates": replicates,
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "by_task": {task: float(value) for task, value in zip(tasks, task_delta)},
    }


def _comparison_report(
    baseline: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    task_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result = {}
    for metric_index, metric in enumerate(METRIC_NAMES):
        paired = _task_balanced_delta(
            baseline[metric], candidate[metric], task_ids,
            seed=seed + metric_index,
            replicates=replicates,
        )
        baseline_mean = float(baseline[metric].double().mean())
        candidate_mean = float(candidate[metric].double().mean())
        paired.update(
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            micro_relative_improvement=(baseline_mean - candidate_mean) / baseline_mean,
        )
        result[metric] = paired
    return result


def _intervention_report(
    normal: Mapping[str, Tensor],
    intervention: Mapping[str, Tensor],
    task_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result = {}
    for metric_index, metric in enumerate(METRIC_NAMES):
        paired = _task_balanced_delta(
            intervention[metric],
            normal[metric],
            task_ids,
            seed=seed + metric_index,
            replicates=replicates,
        )
        normal_mean = float(normal[metric].double().mean())
        intervention_mean = float(intervention[metric].double().mean())
        paired.update(
            direction="intervention_minus_normal; positive means the intervention degrades reconstruction",
            normal_mean=normal_mean,
            intervention_mean=intervention_mean,
            micro_relative_error_increase=(intervention_mean - normal_mean) / normal_mean,
        )
        result[metric] = paired
    return result


def _checkpoint_report(loaded: LoadedModel) -> dict[str, Any]:
    return {
        "path": str(loaded.path),
        "sha256": loaded.sha256,
        "epoch": loaded.epoch,
        "global_step": loaded.global_step,
        "variant": loaded.module.variant,
        "normalization_sha256": loaded.module.normalization_sha256,
        "e39_cache_sha256": loaded.module.e39_cache_sha256,
        "e39_checkpoint_sha256": loaded.module.e39_checkpoint_sha256,
        "effect_manifest_sha256": loaded.module.effect_manifest_sha256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    a0_root = args.a0_data_root.expanduser().resolve()
    a1_root = args.a1_data_root.expanduser().resolve()
    paired_data = _verify_paired_releases(a0_root, a1_root)
    split = collect_test_split(
        a1_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_every_batches=args.log_every_batches,
    )
    if split.sample_ids != _manifest_identity(a0_root)[0]:
        raise ValueError("loaded test order differs from paired manifest order")
    a0 = load_model(args.a0_checkpoint, expected_variant="a0", device=device)
    a1 = load_model(args.a1_checkpoint, expected_variant="a1", device=device)
    if a0.module.normalization_sha256 != paired_data["normalization_sha256"]:
        raise ValueError("A0 checkpoint normalization provenance mismatch")
    if a1.module.normalization_sha256 != paired_data["normalization_sha256"]:
        raise ValueError("A1 checkpoint normalization provenance mismatch")

    global_permutation, global_offset = deterministic_derangement(split.sample_count, args.seed)
    task_permutation, task_shuffle = grouped_derangement(split.task_ids, args.seed)
    true_effect = split.e39_mean
    zero_effect = torch.zeros_like(true_effect)
    global_effect = true_effect[global_permutation]
    task_effect = true_effect[task_permutation]
    conditions: dict[str, dict[str, Tensor]] = {}

    conditions["a0_normal"], a0_latent = evaluate_condition(
        a0, split, device=device, batch_size=args.batch_size, effect=None, return_latent=True, label="a0_normal"
    )
    assert a0_latent is not None
    conditions["a0_z_zero_decoder"], _ = evaluate_condition(
        a0,
        split,
        device=device,
        batch_size=args.batch_size,
        effect=None,
        z_action=torch.zeros_like(a0_latent),
        label="a0_z_zero_decoder",
    )
    conditions["a0_z_shuffled_decoder"], _ = evaluate_condition(
        a0,
        split,
        device=device,
        batch_size=args.batch_size,
        effect=None,
        z_action=a0_latent[global_permutation],
        label="a0_z_shuffled_decoder",
    )

    conditions["a1_normal"], a1_latent = evaluate_condition(
        a1,
        split,
        device=device,
        batch_size=args.batch_size,
        effect=true_effect,
        return_latent=True,
        label="a1_normal",
    )
    assert a1_latent is not None
    conditions["a1_z_zero_decoder"], _ = evaluate_condition(
        a1,
        split,
        device=device,
        batch_size=args.batch_size,
        effect=true_effect,
        z_action=torch.zeros_like(a1_latent),
        label="a1_z_zero_decoder",
    )
    conditions["a1_z_shuffled_decoder"], _ = evaluate_condition(
        a1,
        split,
        device=device,
        batch_size=args.batch_size,
        effect=true_effect,
        z_action=a1_latent[global_permutation],
        label="a1_z_shuffled_decoder",
    )
    for effect_label, effect in (
        ("zero", zero_effect),
        ("global_shuffled", global_effect),
        ("task_shuffled", task_effect),
    ):
        conditions[f"a1_effect_{effect_label}_decoder"], _ = evaluate_condition(
            a1,
            split,
            device=device,
            batch_size=args.batch_size,
            effect=effect,
            z_action=a1_latent,
            label=f"a1_effect_{effect_label}_decoder",
        )
        conditions[f"a1_effect_{effect_label}_end_to_end"], _ = evaluate_condition(
            a1,
            split,
            device=device,
            batch_size=args.batch_size,
            effect=effect,
            label=f"a1_effect_{effect_label}_end_to_end",
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "test",
            "checkpoint_selection_split": "validation",
            "test_used_for_selection": False,
            "precision": "bf16 autocast inference",
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "effect_intervention_semantics": {
                "decoder": "encode with the true effect, alter effect only at decode",
                "end_to_end": "alter effect consistently at both encode and decode",
                "task_shuffled": "fixed-point-free shuffle within each RoboTwin task family",
                "global_shuffled": "fixed-point-free shuffle across the full test split",
            },
        },
        "data": paired_data,
        "checkpoints": {"a0": _checkpoint_report(a0), "a1": _checkpoint_report(a1)},
        "shuffle": {
            "global_offset": global_offset,
            "global_fixed_points": int((global_permutation == torch.arange(split.sample_count)).sum()),
            "task": task_shuffle,
            "task_fixed_points": int((task_permutation == torch.arange(split.sample_count)).sum()),
        },
        "condition_summaries": _summarize_conditions(conditions),
        "a0_vs_a1": _comparison_report(
            conditions["a0_normal"],
            conditions["a1_normal"],
            split.task_ids,
            seed=args.seed + 1000,
            replicates=args.bootstrap_replicates,
        ),
        "interventions": {
            condition: _intervention_report(
                conditions["a0_normal" if condition.startswith("a0_") else "a1_normal"],
                metrics,
                split.task_ids,
                seed=args.seed + 2000 + index * 100,
                replicates=args.bootstrap_replicates,
            )
            for index, (condition, metrics) in enumerate(conditions.items())
            if condition not in {"a0_normal", "a1_normal"}
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "held_out_test.json"
    arrays_path = args.output_dir / "paired_window_metrics.npz"
    if not args.overwrite and (report_path.exists() or arrays_path.exists()):
        raise FileExistsError(f"refusing to overwrite held-out evaluation in {args.output_dir}")
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays = {
        "sample_ids": np.asarray(split.sample_ids),
        "task_ids": np.asarray(split.task_ids),
    }
    for condition, metrics in conditions.items():
        for metric, values in metrics.items():
            arrays[f"{condition}__{metric}"] = values.numpy()
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_report.replace(report_path)
    temporary_arrays.replace(arrays_path)
    print(json.dumps({"report": str(report_path), "arrays": str(arrays_path)}, sort_keys=True), flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-checkpoint", required=True, type=Path)
    parser.add_argument("--a1-checkpoint", required=True, type=Path)
    parser.add_argument("--a0-data-root", required=True, type=Path)
    parser.add_argument("--a1-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every_batches < 1:
        parser.error("batch size and log interval must be positive; workers must be non-negative")
    if args.bootstrap_replicates < 1000:
        parser.error("bootstrap replicates must be at least 1000")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
