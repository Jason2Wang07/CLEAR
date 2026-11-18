"""Export deterministic posterior means from a frozen Stage-1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.effect_window_datamodule import EffectWindowDataModule
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug


SCHEMA_VERSION = "lawam-effect-posterior-means-v1"
LATENT_DIM = 32


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolved_model_config(config: Path, vision_model: Path | None) -> Path:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("model config must contain a model mapping")
    if vision_model is not None:
        payload["model"]["vision_model_id"] = str(vision_model.resolve())
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False)
    yaml.safe_dump(payload, handle, sort_keys=False)
    handle.close()
    return Path(handle.name)


@torch.inference_mode()
def _encode_split(
    *,
    split: str,
    dataset: Any,
    loader: Any,
    model: torch.nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    progress_every: int,
) -> tuple[np.ndarray, list[str]]:
    count = len(dataset)
    means = np.empty((count, LATENT_DIM), dtype=np.float32)
    seen = np.zeros(count, dtype=bool)
    for batch_index, raw_batch in enumerate(loader, 1):
        indices = raw_batch["dataset_indices"].numpy().astype(np.int64, copy=False)
        videos, dec_videos = gpu_two_view_video_aug(
            raw_batch["videos"].to(device, non_blocking=True),
            training=False,
            dual_view_aug=False,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            output = model.get_latent_action(
                videos=videos,
                states=raw_batch["states"].to(device, non_blocking=True),
                dec_videos=dec_videos,
                state_mask=raw_batch["state_mask"].to(device, non_blocking=True),
                embodiment_ids=raw_batch["embodiment_ids"].to(device, non_blocking=True),
                predict_future_frame=False,
            )
        mean = output.get("latent_mu")
        if mean is None or tuple(mean.shape[1:]) != (1, LATENT_DIM):
            raise RuntimeError(f"{split}: checkpoint returned invalid latent_mu {getattr(mean, 'shape', None)}")
        if np.any(seen[indices]):
            raise RuntimeError(f"{split}: loader returned duplicate dataset indices")
        means[indices] = mean[:, 0].float().cpu().numpy()
        seen[indices] = True
        if progress_every > 0 and batch_index % progress_every == 0:
            print(f"E39_EXPORT split={split} batches={batch_index} rows={int(seen.sum())}/{count}", flush=True)
    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise RuntimeError(f"{split}: extraction missed {len(missing)} rows; first={missing[:8].tolist()}")
    if not np.isfinite(means).all():
        raise RuntimeError(f"{split}: posterior means contain non-finite values")
    return means, list(dataset.sample_ids)


def export(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    data = EffectWindowDataModule(
        args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        eval_block_size=int(args.eval_block_size),
        max_open_shards=int(args.max_open_shards),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        rank=0,
        world_size=1,
        verify_row_identity=True,
        train_sampling="natural",
    )
    data.setup(None)
    datasets = {
        "train": data.train_dataset,
        "validation": data.val_dataset,
        "test": data.test_dataset,
    }
    loaders = {
        "train": data.full_train_dataloader,
        "validation": data.val_dataloader,
        "test": data.test_dataloader,
    }

    resolved = _resolved_model_config(args.model_config, args.vision_model)
    try:
        model = load_latent_action_model(args.checkpoint, resolved).to(device).eval()
    finally:
        resolved.unlink(missing_ok=True)

    split_outputs: list[tuple[str, str, np.ndarray]] = []
    split_counts: dict[str, int] = {}
    try:
        for split in args.splits:
            dataset = datasets[split]
            if dataset is None:
                raise RuntimeError(f"{split} dataset was not initialized")
            means, sample_ids = _encode_split(
                split=split,
                dataset=dataset,
                loader=loaders[split](),
                model=model,
                device=device,
                amp_dtype=amp_dtype,
                progress_every=args.progress_every,
            )
            split_counts[split] = len(sample_ids)
            split_outputs.extend((sample_id, split, means[index]) for index, sample_id in enumerate(sample_ids))
    finally:
        data.teardown(None)

    split_outputs.sort(key=lambda item: item[0])
    sample_ids = [item[0] for item in split_outputs]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("selected split manifests contain duplicate sample_id values")
    splits = [item[1] for item in split_outputs]
    posterior_mean = np.stack([item[2] for item in split_outputs]).astype(np.float32)
    selection_report = getattr(args, "selection_report", None)
    selection = {
        "status": str(getattr(args, "selection_status", "unspecified")),
        "rationale": getattr(args, "selection_rationale", None),
        "report": str(selection_report.resolve()) if selection_report is not None else None,
        "report_sha256": _sha256_file(selection_report) if selection_report is not None else None,
    }
    metadata = {
        "schema": SCHEMA_VERSION,
        "version": 1,
        "deterministic": True,
        "posterior_statistic": "mean",
        "latent_shape": [1, LATENT_DIM],
        "sample_count": len(sample_ids),
        "split_counts": split_counts,
        "sample_id_sha256": _sample_id_digest(sample_ids),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "model_config": str(args.model_config.resolve()),
        "model_config_sha256": _sha256_file(args.model_config),
        "teacher": str(getattr(args, "teacher", "unspecified")),
        "checkpoint_selection": selection,
        "data_root": str(args.data_root.resolve()),
        "effect_manifest_sha256": _sha256_file(args.data_root / "manifests/all.parquet"),
        "precision": args.precision,
        "image_augmentation": False,
        "sampling": False,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable posterior cache already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            splits=np.asarray(splits, dtype=np.str_),
            posterior_mean=posterior_mean,
        )
        with np.load(temporary, allow_pickle=False) as check:
            if check["posterior_mean"].shape != (len(sample_ids), LATENT_DIM):
                raise RuntimeError("published posterior cache failed shape verification")
            if not np.array_equal(check["sample_ids"], np.asarray(sample_ids, dtype=np.str_)):
                raise RuntimeError("published posterior cache failed identity verification")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {**metadata, "output": str(output), "output_sha256": _sha256_file(output)}
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--teacher", default="unspecified")
    parser.add_argument(
        "--selection-status",
        choices=("validation_selected", "explicit_ablation", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--selection-rationale")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-block-size", type=int, default=512)
    parser.add_argument("--max-open-shards", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)
    for name in ("data_root", "checkpoint", "model_config"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.vision_model is not None and not args.vision_model.exists():
        parser.error(f"--vision-model does not exist: {args.vision_model}")
    if args.selection_report is not None and not args.selection_report.is_file():
        parser.error(f"--selection-report does not exist: {args.selection_report}")
    if args.selection_status == "explicit_ablation" and not args.selection_rationale:
        parser.error("--selection-rationale is required for an explicit ablation")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and num workers non-negative")
    args.splits = tuple(dict.fromkeys(args.splits))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = export(parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
