"""Decode predicted Stage-2 action latents on the validation split."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import PartialState
from omegaconf import OmegaConf
from torch import Tensor

from latent_action_model.action_ae.distributions import (
    DiagonalGaussian,
    diagonal_gaussian_kl,
)
from latent_action_model.action_ae.evaluate import load_action_ae_checkpoint
from latent_action_model.action_ae.losses import so3_geodesic_error
from latent_action_model.bridge_stage2.action_decode_metrics import (
    PHYSICAL_ENDPOINT_INDICES as BRIDGE_PHYSICAL_ENDPOINT_INDICES,
    summarize_decoded_trajectories as summarize_bridge_decoded_trajectories,
)
from latent_action_model.bridge_stage2.temporal import (
    is_bridge_stage2_temporal_contract,
    resolve_bridge_stage2_temporal_contract,
)
from latent_action_model.libero_stage2.action_decode_metrics import (
    DECODE_MODES,
    PHYSICAL_ENDPOINT_INDICES as LIBERO_PHYSICAL_ENDPOINT_INDICES,
    denormalize_current_state,
    summarize_decoded_trajectories as summarize_libero_decoded_trajectories,
)
from latent_action_model.libero_stage2.runtime_contract import (
    is_libero_04s_adapter_temporal_contract,
)
from starVLA.dataloader import build_dataloaders
from starVLA.model.framework import build_framework
from starVLA.training.evaluate_stage2_phase_p import (
    _move_batch,
    _selected_checkpoint,
)
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    apply_training_freeze_policy,
)


SCHEMA_VERSION = "lawam-libero-stage2-action-decode-validation-v1"
BRIDGE_SCHEMA_VERSION = "lawam-bridge-stage2-action-decode-validation-v1"


def _mode_definitions(temporal_description: str) -> dict[str, dict[str, str]]:
    return {
        "teacher_teacher": {
            "z_action": "teacher posterior mean",
            "decoder_z_effect": "teacher",
            "purpose": f"frozen decoder ceiling under {temporal_description}",
        },
        "predicted_teacher": {
            "z_action": "predicted Gaussian mean from the normal Stage-2 path",
            "decoder_z_effect": "teacher",
            "purpose": "action-predictor error with decoder effect held at teacher",
        },
        "teacher_predicted": {
            "z_action": "teacher posterior mean",
            "decoder_z_effect": "predicted",
            "purpose": "decoder sensitivity to predicted effect",
        },
        "predicted_predicted": {
            "z_action": "predicted Gaussian mean from the normal Stage-2 path",
            "decoder_z_effect": "predicted",
            "purpose": "deployed latent-to-trajectory path before the action expert",
        },
    }


def _summarize(values: Tensor) -> dict[str, float | int]:
    flat = values.detach().double().reshape(-1).cpu()
    if flat.numel() < 1 or not bool(torch.isfinite(flat).all()):
        raise ValueError("diagnostic metric is empty or non-finite")
    mean = float(flat.mean())
    std = float(flat.std(unbiased=False))
    sem = std / math.sqrt(flat.numel())
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "count": int(flat.numel()),
    }


def _validation_identity(dataset: Any, index: int) -> dict[str, Any]:
    result: dict[str, Any] = {"validation_loader_index": int(index)}
    row_index = getattr(dataset, "_stage2_row_index", None)
    if row_index is not None and callable(getattr(row_index, "row", None)):
        row = row_index.row(int(index))
        for key in ("sample_id", "task_id", "episode_id", "start_step"):
            if key in row:
                value = row[key]
                result[key] = (
                    int(value) if key in {"episode_id", "start_step"} else str(value)
                )
        return result
    sample_step = getattr(dataset, "sample_step", None)
    if not callable(sample_step):
        return result
    try:
        _, episode_id, start_step = sample_step(int(index))
    except Exception:
        return result
    result.update(
        {
            "episode_id": int(episode_id),
            "start_step": int(start_step),
            "sample_id": (
                f"libero_merged_no_noops_20hz:{int(episode_id)}:{int(start_step)}"
            ),
        }
    )
    return result


def _plot_example(
    example: Mapping[str, Any],
    output: Path,
    *,
    physical_endpoint_indices: Sequence[int],
    physical_endpoint_label: str,
) -> None:
    target = example["target_raw"].numpy()
    predictions = {
        mode: example["predictions_raw"][mode].numpy() for mode in DECODE_MODES
    }
    pose_valid = (example["time_mask"] & example["pose_mask"]).numpy()
    gripper_valid = (example["time_mask"] & example["gripper_mask"]).numpy()
    steps = np.arange(1, target.shape[0] + 1)
    physical_steps = np.asarray(physical_endpoint_indices, dtype=np.int64)
    colors = {
        "teacher_teacher": "tab:green",
        "predicted_teacher": "tab:orange",
        "teacher_predicted": "tab:purple",
        "predicted_predicted": "tab:blue",
    }
    labels = {
        "teacher_teacher": "teacher action / teacher effect",
        "predicted_teacher": "predicted action / teacher effect",
        "teacher_predicted": "teacher action / predicted effect",
        "predicted_predicted": "predicted action / predicted effect",
    }

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    path_axis = fig.add_subplot(2, 2, 1, projection="3d")
    position_axis = fig.add_subplot(2, 2, 2)
    rotation_axis = fig.add_subplot(2, 2, 3)
    gripper_axis = fig.add_subplot(2, 2, 4)

    path_axis.plot(*target[pose_valid, :3].T, color="black", linewidth=2.5, label="target")
    for mode in DECODE_MODES:
        prediction = predictions[mode]
        path_axis.plot(
            *prediction[pose_valid, :3].T,
            color=colors[mode],
            linewidth=1.4,
            label=labels[mode],
        )
        position_error = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1) * 100.0
        position_axis.plot(
            steps[pose_valid], position_error[pose_valid], color=colors[mode], label=labels[mode]
        )
        rotation_error = so3_geodesic_error(
            torch.from_numpy(prediction[:, 3:6]),
            torch.from_numpy(target[:, 3:6]),
            degrees=True,
        ).numpy()
        rotation_axis.plot(
            steps[pose_valid], rotation_error[pose_valid], color=colors[mode], label=labels[mode]
        )
        if np.any(gripper_valid):
            gripper_axis.plot(
                steps[gripper_valid],
                prediction[gripper_valid, 6] * 100.0,
                color=colors[mode],
                label=labels[mode],
            )

    position_axis.scatter(
        steps[physical_steps],
        np.zeros_like(physical_steps, dtype=np.float64),
        marker="|",
        color="black",
        label=physical_endpoint_label,
    )
    if np.any(gripper_valid):
        gripper_axis.plot(
            steps[gripper_valid],
            target[gripper_valid, 6] * 100.0,
            color="black",
            linewidth=2.5,
            label="target",
        )
    path_axis.set(xlabel="local x (m)", ylabel="local y (m)", zlabel="local z (m)")
    path_axis.set_title("Decoded local TCP path")
    position_axis.set(xlabel="representation slot", ylabel="position L2 error (cm)")
    position_axis.set_title("Position error")
    rotation_axis.set(xlabel="representation slot", ylabel="SO(3) error (degree)")
    rotation_axis.set_title("Rotation error")
    gripper_axis.set(xlabel="representation slot", ylabel="gripper width (cm)")
    gripper_axis.set_title("Gripper trajectory")
    for axis in (path_axis, position_axis, rotation_axis, gripper_axis):
        axis.legend(fontsize=7)
    identity = example["identity"]
    fig.suptitle(str(identity.get("sample_id", identity["validation_loader_index"])), fontsize=10)
    fig.savefig(output, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def _evaluate(
    *,
    framework: Any,
    loader: Any,
    device: torch.device,
    normalization: Any,
    normalization_metadata: Mapping[str, Any],
    plot_samples: int,
    plot_dir: Path,
    log_every: int,
    summarize_trajectories: Any,
    physical_endpoint_indices: Sequence[int],
    physical_endpoint_label: str,
) -> dict[str, Any]:
    framework.eval()
    backend = framework.policy_backend
    if backend.stage2_phase != "E" or backend.paired_latent_predictor is None:
        raise ValueError("action decode evaluation requires a Phase E paired predictor")
    if backend.frozen_action_decoder is None or backend.stage2_latent_affine is None:
        raise RuntimeError("Phase E did not load the frozen ActionVAE decoder contract")
    if backend.model_cfg.stage2_action_inference != "mean":
        raise ValueError("formal action decode evaluation requires Gaussian-mean inference")

    row_count = len(loader.dataset)
    selected_indices = set(
        np.linspace(0, row_count - 1, num=min(plot_samples, row_count), dtype=np.int64).tolist()
    )
    normalized_chunks: defaultdict[str, list[Tensor]] = defaultdict(list)
    raw_chunks: defaultdict[str, list[Tensor]] = defaultdict(list)
    common_chunks: defaultdict[str, list[Tensor]] = defaultdict(list)
    diagnostic_chunks: defaultdict[str, list[Tensor]] = defaultdict(list)
    examples: dict[int, dict[str, Any]] = {}
    rows = 0

    for batch_index, raw_batch in enumerate(loader, start=1):
        prepared = _move_batch(raw_batch, device)
        shared = backend._run_shared_encoding_train(
            prepared_batch=prepared,
            source="evaluate_stage2_action_decode",
            lam_features_with_no_grad=True,
        )
        if shared.predicted_effect is None or shared.paired_tokens is None:
            raise RuntimeError("Phase E encoding omitted predicted effect/action tokens")
        if shared.paired_tokens.action_logvar is None:
            raise RuntimeError("Phase E action predictor omitted Gaussian logvar")

        predicted_action = shared.paired_tokens.action.float()
        predicted_effect = shared.predicted_effect.float()
        teacher_action = prepared["z_action"].float()
        teacher_effect = prepared["z_effect"].float()
        state_mask = prepared["teacher_current_state_mask"].bool()
        normalized_state = torch.where(
            state_mask,
            prepared["teacher_current_state"].float(),
            torch.zeros_like(prepared["teacher_current_state"].float()),
        )
        current_state_raw = denormalize_current_state(
            normalized_state, state_mask, normalization_metadata
        )
        action_by_mode = {
            "teacher_teacher": teacher_action,
            "predicted_teacher": predicted_action,
            "teacher_predicted": teacher_action,
            "predicted_predicted": predicted_action,
        }
        effect_by_mode = {
            "teacher_teacher": teacher_effect,
            "predicted_teacher": teacher_effect,
            "teacher_predicted": predicted_effect,
            "predicted_predicted": predicted_effect,
        }
        batch_normalized: dict[str, Tensor] = {}
        batch_raw: dict[str, Tensor] = {}
        for mode in DECODE_MODES:
            decoded = backend.frozen_action_decoder(
                backend.stage2_latent_affine.action_to_raw(action_by_mode[mode]),
                normalized_state,
                backend.stage2_latent_affine.effect_to_raw(effect_by_mode[mode]),
            ).float()
            batch_normalized[mode] = decoded
            batch_raw[mode] = normalization.denormalize(decoded).float()
            normalized_chunks[mode].append(decoded.cpu())
            raw_chunks[mode].append(batch_raw[mode].cpu())

        target_normalized = prepared["teacher_trajectory"].float()
        target_raw = normalization.denormalize(target_normalized).float()
        common = {
            "target_normalized": target_normalized,
            "target_raw": target_raw,
            "time_mask": prepared["teacher_time_mask"].bool(),
            "pose_mask": prepared["teacher_pose_mask"].bool(),
            "gripper_mask": prepared["teacher_gripper_mask"].bool(),
            "current_state_raw": current_state_raw,
            "current_state_mask": state_mask,
            "physical_valid_count": prepared["physical_valid_count"].long(),
            "action_hz": prepared["action_hz"].float(),
        }
        for name, value in common.items():
            common_chunks[name].append(value.cpu())

        teacher_distribution = DiagonalGaussian(
            mean=teacher_action,
            logvar=prepared["z_action_logvar"].float(),
        )
        predicted_distribution = DiagonalGaussian(
            mean=predicted_action,
            logvar=shared.paired_tokens.action_logvar.float(),
        )
        action_delta = predicted_action - teacher_action
        effect_delta = predicted_effect - teacher_effect
        diagnostics = {
            "action_kl_teacher_to_predicted": diagonal_gaussian_kl(
                teacher_distribution, predicted_distribution, reduction="none"
            ),
            "action_mean_smooth_l1": F.smooth_l1_loss(
                predicted_action, teacher_action, beta=0.1, reduction="none"
            ).mean(dim=-1),
            "action_mean_rmse": action_delta.square().mean(dim=-1).sqrt(),
            "action_mean_cosine": F.cosine_similarity(
                predicted_action, teacher_action, dim=-1
            ),
            "effect_smooth_l1": F.smooth_l1_loss(
                predicted_effect, teacher_effect, beta=0.1, reduction="none"
            ).mean(dim=-1),
            "effect_rmse": effect_delta.square().mean(dim=-1).sqrt(),
            "effect_cosine": F.cosine_similarity(
                predicted_effect, teacher_effect, dim=-1
            ),
            "dino_mse": (shared.h_t1_pred.float() - shared.h_t1_gt.float())
            .square()
            .flatten(1)
            .mean(dim=-1),
            "dino_cosine": F.cosine_similarity(
                shared.h_t1_pred.float().flatten(1),
                shared.h_t1_gt.float().flatten(1),
                dim=-1,
            ),
            "predicted_action_logvar_mean": predicted_distribution.logvar.mean(dim=-1),
        }
        for name, value in diagnostics.items():
            diagnostic_chunks[name].append(value.cpu())

        batch_size = int(target_raw.shape[0])
        for local_index in range(batch_size):
            global_index = rows + local_index
            if global_index not in selected_indices:
                continue
            examples[global_index] = {
                "identity": _validation_identity(loader.dataset, global_index),
                "target_raw": target_raw[local_index].cpu(),
                "predictions_raw": {
                    mode: batch_raw[mode][local_index].cpu() for mode in DECODE_MODES
                },
                "time_mask": common["time_mask"][local_index].cpu(),
                "pose_mask": common["pose_mask"][local_index].cpu(),
                "gripper_mask": common["gripper_mask"][local_index].cpu(),
            }
        rows += batch_size
        if batch_index == 1 or batch_index % log_every == 0:
            print(
                f"ACTION_DECODE_PROGRESS batches={batch_index} rows={rows}/{row_count}",
                flush=True,
            )

    if rows != row_count:
        raise ValueError(f"action decode covered {rows} rows, expected {row_count}")
    if set(examples) != selected_indices:
        raise AssertionError("qualitative example selection did not cover its fixed indices")
    predictions_normalized = {
        mode: torch.cat(normalized_chunks[mode], dim=0) for mode in DECODE_MODES
    }
    predictions_raw = {mode: torch.cat(raw_chunks[mode], dim=0) for mode in DECODE_MODES}
    common = {name: torch.cat(values, dim=0) for name, values in common_chunks.items()}
    trajectories = summarize_trajectories(
        predictions_normalized=predictions_normalized,
        predictions_raw=predictions_raw,
        **common,
    )

    plot_dir.mkdir(parents=True, exist_ok=False)
    qualitative = []
    for plot_index, sample_index in enumerate(sorted(examples)):
        filename = f"sample-{plot_index:02d}.png"
        _plot_example(
            examples[sample_index],
            plot_dir / filename,
            physical_endpoint_indices=physical_endpoint_indices,
            physical_endpoint_label=physical_endpoint_label,
        )
        qualitative.append(
            {
                **examples[sample_index]["identity"],
                "plot": f"plots/{filename}",
            }
        )
    return {
        "rows": rows,
        "batches": len(loader),
        "trajectory_metrics": trajectories,
        "latent_and_future_diagnostics": {
            name: _summarize(torch.cat(values, dim=0))
            for name, values in sorted(diagnostic_chunks.items())
        },
        "qualitative_examples": qualitative,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--plot-samples", default=8, type=int)
    parser.add_argument("--log-every", default=20, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if min(args.batch_size, args.plot_samples, args.log_every) < 1 or args.num_workers < 0:
        raise ValueError("batch-size/plot-samples/log-every must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite action decode output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full Stage-2 action decode evaluation requires CUDA")

    checkpoint, checkpoint_metadata = _selected_checkpoint(
        args.selection.expanduser().resolve(), expected_phase="E"
    )
    cfg = OmegaConf.load(args.config)
    action_cfg = cfg.framework.action_model
    if str(action_cfg.stage2_phase).upper() != "E":
        raise ValueError("action decode config must instantiate stage2_phase=E")
    temporal_identifier = str(action_cfg.stage2_temporal_contract)
    if is_libero_04s_adapter_temporal_contract(temporal_identifier):
        schema_version = SCHEMA_VERSION
        temporal_description = "the LIBERO 0.4s compatibility bridge"
        temporal_report = {
            "identifier": temporal_identifier,
            "duration_seconds": 0.4,
            "representation_slots": 24,
            "representation_hz": 60.0,
            "physical_control_steps": 8,
            "physical_endpoint_indices_zero_based": list(
                LIBERO_PHYSICAL_ENDPOINT_INDICES
            ),
            "control_hz": 20.0,
        }
        summarize_trajectories = summarize_libero_decoded_trajectories
        physical_endpoint_indices = LIBERO_PHYSICAL_ENDPOINT_INDICES
        physical_endpoint_label = "20 Hz endpoints"
    elif is_bridge_stage2_temporal_contract(temporal_identifier):
        temporal = resolve_bridge_stage2_temporal_contract(temporal_identifier)
        schema_version = BRIDGE_SCHEMA_VERSION
        temporal_description = "the Bridge 1.2s 6x5Hz-to-24x20Hz representation"
        temporal_report = {
            "identifier": temporal.identifier,
            "duration_seconds": temporal.duration_seconds,
            "representation_slots": temporal.representation_steps,
            "representation_hz": temporal.representation_hz,
            "physical_control_steps": temporal.physical_valid_count,
            "physical_endpoint_indices_zero_based": list(
                BRIDGE_PHYSICAL_ENDPOINT_INDICES
            ),
            "control_hz": temporal.control_hz,
            "dino_future_delta_steps": temporal.dino_future_delta_steps,
            "effect_teacher_delta_steps": temporal.effect_teacher_delta_steps,
            "replan_steps": temporal.replan_steps,
        }
        summarize_trajectories = summarize_bridge_decoded_trajectories
        physical_endpoint_indices = BRIDGE_PHYSICAL_ENDPOINT_INDICES
        physical_endpoint_label = "native 5 Hz endpoints"
    else:
        raise ValueError(
            f"unsupported Stage-2 action-decode temporal contract: {temporal_identifier!r}"
        )
    cfg.output_dir = str(temporary / "dataset-runtime")
    cfg.trackers = []
    cfg.datasets.vla_data.per_device_batch_size = int(args.batch_size)
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.val_num_workers = int(args.num_workers)
    cfg.datasets.vla_data.drop_last = False
    cfg.datasets.vla_data.in_order = True

    PartialState()
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    framework = build_framework(cfg)
    framework = apply_training_freeze_policy(framework, cfg)
    _, validation_loader = build_dataloaders(cfg)
    if validation_loader is None:
        raise RuntimeError("action decode evaluator could not build the validation loader")
    TrainerUtils.load_finetune_init_weights(
        framework, str(checkpoint), strict_structure=True
    )
    framework.to(device)

    loaded_action_ae = load_action_ae_checkpoint(
        action_cfg.stage2_action_decoder_checkpoint_path,
        action_cfg.stage2_action_decoder_data_root,
        device=torch.device("cpu"),
    )
    expected_decoder_sha = str(action_cfg.stage2_action_decoder_checkpoint_sha256)
    if loaded_action_ae.checkpoint_sha256 != expected_decoder_sha:
        raise ValueError("ActionVAE decoder SHA differs from the Stage-2 config")
    normalization_metadata = loaded_action_ae.metadata.get("normalization")
    if not isinstance(normalization_metadata, Mapping):
        raise ValueError("ActionVAE checkpoint has no embedded normalization mapping")

    evaluation = _evaluate(
        framework=framework,
        loader=validation_loader,
        device=device,
        normalization=loaded_action_ae.normalization,
        normalization_metadata=normalization_metadata,
        plot_samples=args.plot_samples,
        plot_dir=temporary / "plots",
        log_every=args.log_every,
        summarize_trajectories=summarize_trajectories,
        physical_endpoint_indices=physical_endpoint_indices,
        physical_endpoint_label=physical_endpoint_label,
    )
    report = {
        "schema_version": schema_version,
        "split": "validation",
        "validation_loader_contract": "the same deterministic validation loader used during Phase E",
        "checkpoint_selected_on_this_split": True,
        "test_metrics_read": False,
        "checkpoint": checkpoint_metadata,
        "action_decoder": {
            "path": str(loaded_action_ae.checkpoint_path),
            "sha256": loaded_action_ae.checkpoint_sha256,
            "schema_version": loaded_action_ae.metadata.get("schema_version"),
            "normalization_sha256": loaded_action_ae.metadata.get("normalization_sha256"),
            "frozen": True,
            "decoder_inputs": ["z_action_raw", "normalized_current_state14", "z_effect_raw"],
        },
        "temporal_contract": temporal_report,
        "prediction_contract": {
            "action_distribution": str(action_cfg.stage2_action_distribution),
            "action_value": "Gaussian mean; no sampling",
            "expert_residual_included": False,
            "expert_residual_note": (
                "the expert residual projection is not an ActionVAE z_action and cannot be decoded"
            ),
            "modes": _mode_definitions(temporal_description),
        },
        "evaluation": evaluation,
    }
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_dir)
    print(f"ACTION_DECODE_COMPLETE output={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
