"""Encode release posterior means and build the epoch-zero latent bank."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.effect_window_datamodule import EffectWindowDataModule
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from latent_action_model.stage1.calibration import calibrate_student_gamma
from latent_action_model.stage1.memory_bank import LatentMemoryBank
from latent_action_model.stage1.teacher import TeacherGraph


def _resolved_model_config(config: Path, vision_model: Path) -> Path:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("model config must contain a model mapping")
    payload["model"]["vision_model_id"] = str(vision_model.resolve())
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False)
    yaml.safe_dump(payload, handle, sort_keys=False)
    handle.close()
    return Path(handle.name)


@torch.inference_mode()
def encode_release_latents(args: argparse.Namespace) -> None:
    data = EffectWindowDataModule(
        args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
        train_sampling="natural",
        rank=0,
        world_size=1,
    )
    data.setup("fit")
    assert data.train_dataset is not None
    sample_ids = data.train_dataset.sample_ids
    graph = TeacherGraph.load(args.teacher_graph, expected_sample_ids=sample_ids)

    resolved = _resolved_model_config(args.model_config, args.vision_model)
    try:
        model = load_latent_action_model(args.checkpoint, resolved).to(args.device).eval()
    finally:
        resolved.unlink(missing_ok=True)

    latents = np.empty((len(sample_ids), 32), dtype=np.float32)
    seen = np.zeros(len(sample_ids), dtype=bool)
    for batch in data.full_train_dataloader():
        indices = batch["dataset_indices"].numpy().astype(np.int64, copy=False)
        videos_uint8 = batch["videos"].to(args.device, non_blocking=True)
        videos, dec_videos = gpu_two_view_video_aug(videos_uint8, training=False, dual_view_aug=False)
        output = model.get_latent_action(
            videos=videos,
            states=batch["states"].to(args.device, non_blocking=True),
            dec_videos=dec_videos,
            state_mask=batch["state_mask"].to(args.device, non_blocking=True),
            embodiment_ids=batch["embodiment_ids"].to(args.device, non_blocking=True),
            predict_future_frame=False,
        )
        mu = output.get("latent_mu")
        if mu is None or tuple(mu.shape[1:]) != (1, 32):
            raise RuntimeError(f"release VAE returned invalid latent_mu shape: {getattr(mu, 'shape', None)}")
        if np.any(seen[indices]):
            raise RuntimeError("full-train loader returned a duplicate dataset index")
        latents[indices] = mu[:, 0].float().cpu().numpy()
        seen[indices] = True
    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise RuntimeError(f"release latent extraction missed {len(missing)} rows")

    bank = LatentMemoryBank.initialize_release(
        sample_ids,
        latents,
        graph.pool_ids,
        graph.episode_keys,
        graph.start_steps,
        graph.end_steps,
        backend=args.index_backend,
    )
    calibration = calibrate_student_gamma(bank, neighbor_rank=8)
    destination = Path(args.output)
    bank.save(destination)
    report_path = Path(args.calibration_output) if args.calibration_output else destination.with_suffix(".gamma.json")
    report = {
        **calibration.as_dict(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "latent_bank": str(destination.resolve()),
        "teacher_graph": str(Path(args.teacher_graph).resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--vision-model", required=True, type=Path)
    parser.add_argument("--teacher-graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-backend", choices=("faiss",), default="faiss")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    encode_release_latents(parse_args(argv))


if __name__ == "__main__":
    main()
