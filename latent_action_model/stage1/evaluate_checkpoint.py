"""Evaluate release or Stage-1 checkpoints with the locked validation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader, Subset

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.effect_window_datamodule import EffectWindowDataModule
from latent_action_model.data_loader.effect_window_dataset import effect_window_collate
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from latent_action_model.stage1.checkpoint_selection import geometry_score
from latent_action_model.stage1.evaluation import (
    evaluate_balanced_geometry,
    evaluate_latent_diagnostics,
    evaluate_shuffle_gate,
    select_val_balanced,
)
from latent_action_model.stage1.teacher import mean_huber_distance


EVALUATION_SCHEMA = "lawam-stage1-checkpoint-evaluation-v2"
LIBERO_GEOMETRY_K = 8
LIBERO_LOCAL_TEACHER_K = 16


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_model_config(config: Path, vision_model: Path) -> Path:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("model config must contain a model mapping")
    payload["model"]["vision_model_id"] = str(vision_model.resolve())
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False)
    yaml.safe_dump(payload, handle, sort_keys=False)
    handle.close()
    return Path(handle.name)


def _split_manifest(data_root: Path, split: str) -> dict[str, list[Any]]:
    return parquet.read_table(data_root / "manifests" / f"{split}.parquet").to_pydict()


def _load_descriptors(
    path: Path,
    expected_ids: Sequence[str],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        sample_ids = archive["sample_ids"].astype(np.str_)
        if metadata.get("schema") != "lawam-stage1-split-descriptors-v1":
            raise ValueError(f"{split} descriptor schema mismatch")
        if metadata.get("split") != split:
            raise ValueError(f"descriptor split must be {split!r}")
        if not np.array_equal(sample_ids, np.asarray(expected_ids, dtype=np.str_)):
            raise ValueError(f"{split} descriptor sample_id order does not match the manifest")
        return (
            archive["descriptors"].astype(np.float32),
            archive["start_context"].astype(np.float32),
            metadata,
        )


def _selection_magnitudes(
    *,
    descriptor_path: Path,
    descriptor_metadata: dict[str, Any],
    train_manifest: dict[str, list[Any]],
    split_manifest: dict[str, list[Any]],
    split_descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if descriptor_metadata.get("teacher_kind") != "libero_object_state_effect_v2":
        return (
            np.asarray(train_manifest["effect_magnitude"], dtype=np.float64),
            np.asarray(split_manifest["effect_magnitude"], dtype=np.float64),
            "manifest_effect_magnitude",
        )

    train_ids = [str(value) for value in train_manifest["sample_id"]]
    train_path = descriptor_path.with_name("train_descriptors.npz")
    train_descriptors, _, train_metadata = _load_descriptors(
        train_path,
        train_ids,
        split="train",
    )
    if train_metadata.get("teacher_kind") != "libero_object_state_effect_v2":
        raise ValueError("train descriptors do not use the LIBERO object-state v2 teacher")
    contract = descriptor_metadata.get("object_state_contract_sha256")
    if not contract or train_metadata.get("object_state_contract_sha256") != contract:
        raise ValueError("train and evaluation descriptors use different object-state contracts")
    return (
        np.linalg.norm(train_descriptors, axis=1).astype(np.float64),
        np.linalg.norm(split_descriptors, axis=1).astype(np.float64),
        "object_descriptor_l2_norm",
    )


def _prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    videos, dec_videos = gpu_two_view_video_aug(
        batch["videos"].to(device, non_blocking=True),
        training=False,
        dual_view_aug=False,
    )
    return {
        **batch,
        "videos": videos,
        "dec_videos": dec_videos,
        "states": batch["states"].to(device, non_blocking=True),
        "state_mask": batch["state_mask"].to(device, non_blocking=True),
        "embodiment_ids": batch["embodiment_ids"].to(device, non_blocking=True),
        "delta_proprio": batch["delta_proprio"].to(device, non_blocking=True),
    }


def _select_far_context_shuffles(
    *,
    anchors: np.ndarray,
    descriptors: np.ndarray,
    contexts: np.ndarray,
    manifest: dict[str, list[Any]],
) -> tuple[np.ndarray, np.ndarray]:
    families = np.asarray(manifest["effect_family"], dtype=np.str_)
    episodes = np.asarray(
        [f"{source}\0{episode}" for source, episode in zip(manifest["source_id"], manifest["episode_id"])],
        dtype=np.str_,
    )
    starts = np.asarray(manifest["start_step"], dtype=np.int64)
    ends = np.asarray(manifest["end_step"], dtype=np.int64)
    context_norm = np.linalg.norm(contexts, axis=1, keepdims=True)
    if np.any(context_norm <= 0):
        raise ValueError("validation start-context descriptor has a zero row")
    context_unit = contexts / context_norm
    selected = np.empty((len(anchors), 5), dtype=np.int64)
    context_percentile = np.empty((len(anchors), 5), dtype=np.float32)

    members_by_family = {
        family: np.flatnonzero(families == family).astype(np.int64)
        for family in sorted(set(families.tolist()))
    }
    for row, anchor in enumerate(anchors.tolist()):
        candidates = members_by_family[str(families[anchor])]
        same_episode = episodes[candidates] == episodes[anchor]
        overlap = np.maximum(starts[candidates], starts[anchor]) <= np.minimum(ends[candidates], ends[anchor])
        candidates = candidates[~(same_episode & overlap)]
        candidates = candidates[candidates != anchor]
        if len(candidates) < 20:
            raise ValueError(f"validation anchor {anchor} has too few compatible shuffle candidates")

        teacher_distance = mean_huber_distance(descriptors[anchor], descriptors[candidates])
        teacher_order = candidates[np.lexsort((candidates, teacher_distance))]
        band_start = int(np.ceil(0.70 * len(teacher_order)))
        band_stop = int(np.ceil(0.95 * len(teacher_order)))
        far = teacher_order[band_start:band_stop]
        context_distance = np.maximum(1.0 - context_unit[candidates] @ context_unit[anchor], 0.0)
        context_order = candidates[np.lexsort((candidates, context_distance))]
        ranks = {int(index): rank + 1 for rank, index in enumerate(context_order.tolist())}
        far_context_distance = np.maximum(1.0 - context_unit[far] @ context_unit[anchor], 0.0)
        chosen = far[np.lexsort((far, far_context_distance))[:5]]
        selected[row] = chosen
        context_percentile[row] = np.asarray([ranks[int(index)] / len(candidates) for index in chosen])
    return selected, context_percentile


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    train_manifest = _split_manifest(data_root, "train")
    split = str(args.split)
    val_manifest = _split_manifest(data_root, split)
    val_ids = [str(value) for value in val_manifest["sample_id"]]
    descriptor_path = Path(args.descriptors)
    descriptors, contexts, descriptor_metadata = _load_descriptors(
        descriptor_path,
        val_ids,
        split=split,
    )
    train_magnitudes, split_magnitudes, magnitude_source = _selection_magnitudes(
        descriptor_path=descriptor_path,
        descriptor_metadata=descriptor_metadata,
        train_manifest=train_manifest,
        split_manifest=val_manifest,
        split_descriptors=descriptors,
    )
    selection = select_val_balanced(
        train_magnitudes=train_magnitudes,
        train_families=train_manifest["effect_family"],
        val_sample_ids=val_ids,
        val_magnitudes=split_magnitudes,
        val_families=val_manifest["effect_family"],
        val_tasks=val_manifest["task_id"],
        samples_per_family=int(args.samples_per_family),
        samples_per_bin=int(args.samples_per_bin),
        min_family_size=int(args.min_family_size),
    )

    data = EffectWindowDataModule(
        data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
        rank=0,
        world_size=1,
        state_normalization_path=args.state_normalization,
    )
    data.setup(None)
    selected_dataset = data.val_dataset if split == "validation" else data.test_dataset
    if selected_dataset is None:
        raise RuntimeError(f"{split} dataset was not initialized")
    if tuple(selected_dataset.sample_ids) != tuple(val_ids):
        raise ValueError(f"{split} dataset and manifest sample_id order differ")

    resolved = _resolved_model_config(args.model_config, args.vision_model)
    try:
        model = load_latent_action_model(args.checkpoint, resolved).to(args.device).eval()
    finally:
        resolved.unlink(missing_ok=True)
    device = torch.device(args.device)
    count = len(val_ids)
    mu = np.empty((count, 32), dtype=np.float32)
    logvar = np.empty((count, 32), dtype=np.float32)
    true_lwm_per_sample = np.empty(count, dtype=np.float32)
    seen = np.zeros(count, dtype=bool)
    lwm_sum = 0.0
    lwm_count = 0
    state_sum = 0.0
    state_count = 0

    split_loader = data.val_dataloader if split == "validation" else data.test_dataloader
    for raw_batch in split_loader():
        indices = raw_batch["dataset_indices"].numpy().astype(np.int64, copy=False)
        batch = _prepare_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model.get_latent_action(
                videos=batch["videos"],
                states=batch["states"],
                dec_videos=batch["dec_videos"],
                state_mask=batch["state_mask"],
                embodiment_ids=batch["embodiment_ids"],
                predict_future_frame=True,
            )
        latent_mu = output["latent_mu"]
        latent_logvar = output["latent_logvar"]
        if latent_mu is None or latent_logvar is None:
            raise RuntimeError("checkpoint does not expose VAE posterior statistics")
        if np.any(seen[indices]):
            raise RuntimeError(f"{split} loader returned duplicate indices")
        mu[indices] = latent_mu[:, 0].float().cpu().numpy()
        logvar[indices] = latent_logvar[:, 0].float().cpu().numpy()
        seen[indices] = True

        recon = output["recon"].float()
        target = output["tgt"].float()
        element_lwm = functional.smooth_l1_loss(recon, target, beta=0.1, reduction="none")
        per_sample = element_lwm.flatten(1).mean(dim=1)
        true_lwm_per_sample[indices] = per_sample.cpu().numpy()
        lwm_sum += float(element_lwm.sum().item())
        lwm_count += int(element_lwm.numel())

        state_prediction = output["s_pred"]
        if state_prediction is None:
            raise RuntimeError("checkpoint has no state-decoder output")
        state_prediction = state_prediction.float().squeeze(1)
        state_target = batch["delta_proprio"].float()
        state_mask = batch["state_mask"].any(dim=1).to(dtype=torch.float32)
        state_error = (state_prediction - state_target).square() * state_mask
        state_sum += float(state_error.sum().item())
        state_count += int(state_mask.sum().item())
    if not np.all(seen):
        raise RuntimeError(f"{split} extraction missed {int(np.sum(~seen))} rows")

    diagnostics = evaluate_latent_diagnostics(mu, logvar)
    selected_indices = selection.indices
    geometry = evaluate_balanced_geometry(
        teacher_descriptors=descriptors[selected_indices],
        latent_means=mu[selected_indices],
        families=np.asarray(val_manifest["effect_family"], dtype=np.str_)[selected_indices],
        expected_family_size=int(args.samples_per_family),
        k=LIBERO_GEOMETRY_K,
        teacher_local_k=LIBERO_LOCAL_TEACHER_K,
    )
    shuffle_indices, context_percentile = _select_far_context_shuffles(
        anchors=selected_indices,
        descriptors=descriptors,
        contexts=contexts,
        manifest=val_manifest,
    )

    subset = Subset(selected_dataset, selected_indices.tolist())
    shuffle_loader = DataLoader(
        subset,
        batch_size=int(args.shuffle_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=effect_window_collate,
    )
    row_by_index = {int(index): row for row, index in enumerate(selected_indices.tolist())}
    shuffled_losses = np.empty((len(selected_indices), 5), dtype=np.float32)
    for raw_batch in shuffle_loader:
        indices = raw_batch["dataset_indices"].numpy().astype(np.int64, copy=False)
        rows = np.asarray([row_by_index[int(index)] for index in indices], dtype=np.int64)
        batch = _prepare_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model.get_latent_action(
                videos=batch["videos"],
                states=batch["states"],
                dec_videos=batch["dec_videos"],
                state_mask=batch["state_mask"],
                embodiment_ids=batch["embodiment_ids"],
                predict_future_frame=True,
            )
            features = output["dec_in"].repeat_interleave(5, dim=0)
            candidate_mu = torch.from_numpy(mu[shuffle_indices[rows]].reshape(-1, 1, 32)).to(device)
            shuffled_recon = model.decoder(features=features, actions=candidate_mu)
        targets = output["tgt"].repeat_interleave(5, dim=0).float()
        loss = functional.smooth_l1_loss(
            shuffled_recon.float(), targets, beta=0.1, reduction="none"
        ).flatten(1).mean(dim=1)
        shuffled_losses[rows] = loss.reshape(len(rows), 5).cpu().numpy()
    shuffle = evaluate_shuffle_gate(true_lwm_per_sample[selected_indices], shuffled_losses)

    per_family = {
        str(family): {
            "sample_count": metrics.sample_count,
            "overlap_at_k": metrics.overlap_at_16,
            "trustworthiness_at_k": metrics.trustworthiness_at_16,
            "rho_local": metrics.rho_local,
        }
        for family, metrics in geometry.per_family.items()
    }
    natural = {
        "sample_count": count,
        "lwm_loss": lwm_sum / lwm_count,
        "state_loss": state_sum / max(state_count, 1),
    }
    latent = {
        "effective_rank": diagnostics.effective_rank,
        "covariance_trace": diagnostics.covariance_trace,
        "pairwise_median_distance": diagnostics.pairwise_median_distance,
        "active_dimensions": diagnostics.active_dimensions,
        "mean_kl_per_dim": diagnostics.mean_kl_per_dim,
        "kl_per_dim": diagnostics.kl_per_dim.tolist(),
        "mean_sigma_per_dim": diagnostics.mean_sigma_per_dim.tolist(),
    }
    geometry_report = {
        "balanced_sample_count": len(selected_indices),
        "families": len(per_family),
        "geometry_k": geometry.geometry_k,
        "teacher_local_k": geometry.teacher_local_k,
        "overlap_at_k": geometry.overlap_at_16,
        "trustworthiness_at_k": geometry.trustworthiness_at_16,
        "rho_local": geometry.rho_local,
        "score": geometry_score(
            overlap_at_16=geometry.overlap_at_16,
            trustworthiness_at_16=geometry.trustworthiness_at_16,
            rho_local=geometry.rho_local,
        ),
        "per_family": per_family,
        "anchor_sample_ids": selection.sample_ids.tolist(),
        "excluded_families": [str(value) for value in selection.excluded_families],
        "samples_per_family": int(args.samples_per_family),
        "samples_per_bin": int(args.samples_per_bin),
        "magnitude_source": magnitude_source,
    }
    shuffle_report = {
        "anchor_count": len(selected_indices),
        "shuffles_per_anchor": 5,
        "selection": "compatible_teacher_far_70_95_then_start_context_top5",
        "true_loss": shuffle.true_loss,
        "shuffle_loss": shuffle.shuffle_loss,
        "gain": shuffle.gain,
        "bootstrap_ci_low": shuffle.bootstrap_ci_low,
        "bootstrap_ci_high": shuffle.bootstrap_ci_high,
        "passes_gain_gate": shuffle.passes_gain_gate,
        "ci_excludes_zero": shuffle.ci_excludes_zero,
        "context_rank_percentile_p10_p50_p90": np.quantile(
            context_percentile, [0.1, 0.5, 0.9]
        ).tolist(),
    }
    return {
        "schema": EVALUATION_SCHEMA,
        "split": split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "descriptors": str(Path(args.descriptors).resolve()),
        "input_sha256": {
            "checkpoint": _sha256_file(Path(args.checkpoint)),
            "model_config": _sha256_file(Path(args.model_config)),
            "manifest": _sha256_file(data_root / "manifests" / f"{split}.parquet"),
            "descriptors": _sha256_file(Path(args.descriptors)),
            "state_normalization": _sha256_file(selected_dataset.state_normalization_path),
        },
        "state_normalization": str(selected_dataset.state_normalization_path),
        "natural": natural,
        "latent": latent,
        "geometry": geometry_report,
        "shuffle": shuffle_report,
        "release_baseline": {
            "lwm_loss": natural["lwm_loss"],
            "state_loss": natural["state_loss"],
            "effective_rank": latent["effective_rank"],
            "covariance_trace": latent["covariance_trace"],
            "pairwise_median_distance": latent["pairwise_median_distance"],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--state-normalization", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--vision-model", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--descriptors", required=True, type=Path)
    parser.add_argument("--samples-per-family", default=64, type=int)
    parser.add_argument("--samples-per-bin", default=16, type=int)
    parser.add_argument("--min-family-size", default=64, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shuffle-batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **report["natural"], **report["geometry"]}, default=str))


if __name__ == "__main__":
    main()
