"""Full-validation gate for the phased LIBERO Stage-2 action predictor."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from accelerate import PartialState
from omegaconf import OmegaConf
from torch import Tensor

from latent_action_model.action_ae.distributions import (
    DiagonalGaussian,
    diagonal_gaussian_kl,
)
from starVLA.dataloader import build_dataloaders
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    apply_training_freeze_policy,
)


SCHEMA_VERSION = "lawam-libero-stage2-phase-p-gate-v1"


@dataclass
class _RunningStats:
    total: float = 0.0
    total_squared: float = 0.0
    count: int = 0

    def update(self, values: Tensor) -> None:
        flat = values.detach().float().reshape(-1).cpu()
        if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
            raise ValueError("Phase P gate received an empty or non-finite metric")
        flat64 = flat.double()
        self.total += float(flat64.sum().item())
        self.total_squared += float(flat64.square().sum().item())
        self.count += int(flat64.numel())

    def summarize(self) -> dict[str, float | int]:
        if self.count < 1:
            raise ValueError("Phase P gate metric has no observations")
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        std = math.sqrt(variance)
        sem = std / math.sqrt(self.count)
        return {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem,
            "count": self.count,
        }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _selected_checkpoint(
    selection_path: Path,
    *,
    expected_phase: str,
) -> tuple[Path, dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("stage2_phase") != expected_phase:
        raise ValueError(
            f"selection phase mismatch: expected {expected_phase}, "
            f"got {selection.get('stage2_phase')!r}"
        )
    if selection.get("test_metrics_read") is not False:
        raise ValueError("Phase P gate requires validation-only checkpoint selection")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or not selected.get("checkpoint"):
        raise ValueError("selection has no selected checkpoint")
    checkpoint = Path(str(selected["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
    sidecar_path = Path(str(selected.get("phase_contract", f"{checkpoint}.stage2.json")))
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"selected checkpoint sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("stage2_phase") != expected_phase:
        raise ValueError("checkpoint sidecar phase differs from selection")
    expected_bytes = int(selected.get("checkpoint_bytes", -1))
    if checkpoint.stat().st_size != expected_bytes:
        raise ValueError("checkpoint byte size differs from validation selection")
    if sidecar.get("checkpoint_sha256") != selected.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA differs between sidecar and validation selection")
    return checkpoint, {
        "path": str(checkpoint),
        "bytes": expected_bytes,
        "sha256": str(selected["checkpoint_sha256"]),
        "step": int(selected["step"]),
        "val_loss_total": float(selected["val_loss_total"]),
        "selection": str(selection_path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
    }


def _masked_smooth_l1_per_sample(
    prediction: Tensor,
    target: Tensor,
    valid_steps: Tensor,
    *,
    start: int,
    end: int,
) -> Tensor:
    prediction = prediction[..., start:end].float()
    target = target[..., start:end].float()
    valid = valid_steps.bool().unsqueeze(-1).expand_as(prediction)
    pointwise = F.smooth_l1_loss(prediction, target, beta=0.1, reduction="none")
    numerator = torch.where(valid, pointwise, torch.zeros_like(pointwise)).sum((1, 2))
    denominator = valid.sum((1, 2))
    if bool((denominator <= 0).any()):
        raise ValueError("trajectory metric encountered a sample with no valid elements")
    return numerator / denominator


def _prior_metrics(
    *,
    backend: Any,
    prepared_batch: Mapping[str, Tensor],
    prior: DiagonalGaussian,
    decoder_effect: Tensor,
) -> dict[str, Tensor]:
    teacher = DiagonalGaussian(
        mean=prepared_batch["z_action"].float(),
        logvar=prepared_batch["z_action_logvar"].float(),
    )
    mean_pointwise = F.smooth_l1_loss(
        prior.mean.float(), teacher.mean, beta=0.1, reduction="none"
    )
    mean_delta = prior.mean.float() - teacher.mean

    state = prepared_batch["teacher_current_state"].float()
    state_mask = prepared_batch["teacher_current_state_mask"].bool()
    state = torch.where(state_mask, state, torch.zeros_like(state))
    predicted_trajectory = backend.frozen_action_decoder(
        backend.stage2_latent_affine.action_to_raw(prior.mean.float()),
        state,
        backend.stage2_latent_affine.effect_to_raw(decoder_effect.float()),
    ).float()
    target_trajectory = prepared_batch["teacher_trajectory"].float()
    time = prepared_batch["teacher_time_mask"].bool()
    pose = time & prepared_batch["teacher_pose_mask"].bool()
    gripper = time & prepared_batch["teacher_gripper_mask"].bool()
    position = _masked_smooth_l1_per_sample(
        predicted_trajectory, target_trajectory, pose, start=0, end=3
    )
    rotation = _masked_smooth_l1_per_sample(
        predicted_trajectory, target_trajectory, pose, start=3, end=6
    )
    gripper_loss = _masked_smooth_l1_per_sample(
        predicted_trajectory, target_trajectory, gripper, start=6, end=7
    )
    trajectory = position + rotation + 0.5 * gripper_loss
    kl = diagonal_gaussian_kl(teacher, prior, reduction="none")
    mean_loss = mean_pointwise.mean(dim=-1)
    plan = kl + 0.5 * mean_loss + 0.2 * trajectory
    return {
        "action_kl": kl,
        "action_mean_smooth_l1": mean_loss,
        "action_mean_rmse": mean_delta.square().mean(dim=-1).sqrt(),
        "action_mean_cosine": F.cosine_similarity(
            prior.mean.float(), teacher.mean, dim=-1
        ),
        "trajectory_position_smooth_l1": position,
        "trajectory_rotation_smooth_l1": rotation,
        "trajectory_gripper_smooth_l1": gripper_loss,
        "trajectory_total": trajectory,
        "plan_loss": plan,
    }


def _paired_delta(candidate: Tensor, reference: Tensor) -> dict[str, float | int]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired gate comparison requires matching one-dimensional rows")
    delta = candidate.double() - reference.double()
    stats = _RunningStats()
    stats.update(delta)
    summary = stats.summarize()
    reference_mean = float(reference.double().mean().item())
    summary["fraction_of_reference"] = (
        float(summary["mean"]) / reference_mean if reference_mean != 0.0 else float("inf")
    )
    return summary


@torch.inference_mode()
def _evaluate_checkpoint(
    *,
    framework: Any,
    loader: Any,
    device: torch.device,
    label: str,
    include_ablations: bool,
    log_every: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    framework.eval()
    backend = framework.policy_backend
    predictor = backend.paired_latent_predictor
    if predictor is None or backend.stage2_phase != "P":
        raise ValueError("Phase P gate requires a P-configured paired latent predictor")

    stats: defaultdict[str, _RunningStats] = defaultdict(_RunningStats)
    sample_values: defaultdict[str, list[Tensor]] = defaultdict(list)
    logvars: list[Tensor] = []
    rows = 0
    batches = 0

    for batches, raw_batch in enumerate(loader, start=1):
        prepared = _move_batch(raw_batch, device)
        shared = backend._run_shared_encoding_train(
            prepared_batch=prepared,
            source="evaluate_stage2_phase_p",
            lam_features_with_no_grad=True,
        )
        if shared.predicted_effect is None or shared.paired_tokens is None:
            raise RuntimeError("Phase P shared encoding omitted predicted paired tokens")
        if shared.paired_tokens.action_logvar is None:
            raise RuntimeError("Phase P action predictor omitted Gaussian logvar")

        batch_size = int(shared.predicted_effect.shape[0])
        rows += batch_size
        predicted_effect = shared.predicted_effect.float()
        teacher_effect = prepared["z_effect"].float()
        effect_delta = predicted_effect - teacher_effect
        shared_metrics = {
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
            "predicted_future_delta_rms": (
                shared.h_t1_pred.float() - shared.h_t.float()
            )
            .square()
            .flatten(1)
            .mean(dim=-1)
            .sqrt(),
        }
        for name, values in shared_metrics.items():
            stats[name].update(values)
            sample_values[name].append(values.detach().cpu())

        normal = DiagonalGaussian(
            mean=shared.paired_tokens.action.float(),
            logvar=shared.paired_tokens.action_logvar.float(),
        )
        priors: dict[str, tuple[DiagonalGaussian, Tensor]] = {
            "normal": (normal, predicted_effect),
        }
        if include_ablations:
            state_key = (
                "action_predictor_state"
                if backend.model_cfg.stage2_action_state_source
                == "actionvae14_padded32"
                else "state"
            )
            common = {
                "state": prepared[state_key],
                "embodiment_id": prepared["embodiment_id"],
                "horizon_condition": backend._stage2_horizon_condition(prepared),
            }
            shift = max(1, batch_size // 2)
            priors.update(
                {
                    "current_only": (
                        predictor.predict_action(
                            shared.h_plan,
                            current_visual=shared.h_t,
                            future_visual=shared.h_t,
                            effect=predicted_effect,
                            **common,
                        ),
                        predicted_effect,
                    ),
                    "shuffled_future": (
                        predictor.predict_action(
                            shared.h_plan,
                            current_visual=shared.h_t,
                            future_visual=shared.h_t1_pred.roll(shift, dims=0),
                            effect=predicted_effect,
                            **common,
                        ),
                        predicted_effect,
                    ),
                    "shuffled_plan": (
                        predictor.predict_action(
                            shared.h_plan.roll(shift, dims=0),
                            current_visual=shared.h_t,
                            future_visual=shared.h_t1_pred,
                            effect=predicted_effect,
                            **common,
                        ),
                        predicted_effect,
                    ),
                    "shuffled_effect": (
                        predictor.predict_action(
                            shared.h_plan,
                            current_visual=shared.h_t,
                            future_visual=shared.h_t1_pred,
                            effect=predicted_effect.roll(shift, dims=0),
                            **common,
                        ),
                        predicted_effect.roll(shift, dims=0),
                    ),
                    "teacher_effect": (
                        predictor.predict_action(
                            shared.h_plan,
                            current_visual=shared.h_t,
                            future_visual=shared.h_t1_pred,
                            effect=teacher_effect,
                            **common,
                        ),
                        teacher_effect,
                    ),
                    "oracle_future": (
                        predictor.predict_action(
                            shared.h_plan,
                            current_visual=shared.h_t,
                            future_visual=shared.h_t1_gt,
                            effect=predicted_effect,
                            **common,
                        ),
                        predicted_effect,
                    ),
                }
            )

        for variant, (prior, decoder_effect) in priors.items():
            variant_metrics = _prior_metrics(
                backend=backend,
                prepared_batch=prepared,
                prior=prior,
                decoder_effect=decoder_effect,
            )
            for metric, values in variant_metrics.items():
                name = f"{variant}/{metric}"
                stats[name].update(values)
                sample_values[name].append(values.detach().cpu())
        logvars.append(normal.logvar.detach().float().cpu())

        if batches == 1 or batches % log_every == 0:
            print(
                f"PHASE_P_GATE_PROGRESS checkpoint={label} batches={batches} rows={rows}",
                flush=True,
            )

    if rows != len(loader.dataset):
        raise ValueError(
            f"Phase P gate did not cover the full validation dataset: {rows} != "
            f"{len(loader.dataset)}"
        )
    concatenated = {
        name: torch.cat(values, dim=0) for name, values in sample_values.items()
    }
    if any(int(values.shape[0]) != rows for values in concatenated.values()):
        raise AssertionError("Phase P gate metric row counts diverged")

    logvar = torch.cat(logvars, dim=0)
    lower = predictor.logvar_min.detach().float().cpu().reshape(1, -1)
    upper = predictor.logvar_max.detach().float().cpu().reshape(1, -1)
    tolerance = 1.0e-6
    logvar_report = {
        "minimum": float(logvar.min().item()),
        "maximum": float(logvar.max().item()),
        "mean": float(logvar.mean().item()),
        "std": float(logvar.std(unbiased=False).item()),
        "at_lower_bound_fraction": float(
            (logvar <= lower + tolerance).float().mean().item()
        ),
        "at_upper_bound_fraction": float(
            (logvar >= upper - tolerance).float().mean().item()
        ),
    }
    report = {
        "rows": rows,
        "batches": batches,
        "metrics": {name: value.summarize() for name, value in sorted(stats.items())},
        "normal_logvar": logvar_report,
    }
    return report, concatenated


def _gate_decision(
    *,
    baseline_report: Mapping[str, Any],
    baseline_samples: Mapping[str, Tensor],
    candidate_report: Mapping[str, Any],
    candidate_samples: Mapping[str, Tensor],
) -> dict[str, Any]:
    plan_improvement = _paired_delta(
        baseline_samples["normal/plan_loss"],
        candidate_samples["normal/plan_loss"],
    )
    trajectory_improvement = _paired_delta(
        baseline_samples["normal/trajectory_total"],
        candidate_samples["normal/trajectory_total"],
    )
    future_shuffle_degradation = _paired_delta(
        candidate_samples["shuffled_future/plan_loss"],
        candidate_samples["normal/plan_loss"],
    )
    current_only_degradation = _paired_delta(
        candidate_samples["current_only/plan_loss"],
        candidate_samples["normal/plan_loss"],
    )
    plan_shuffle_degradation = _paired_delta(
        candidate_samples["shuffled_plan/plan_loss"],
        candidate_samples["normal/plan_loss"],
    )
    effect_change = _paired_delta(
        candidate_samples["effect_smooth_l1"],
        baseline_samples["effect_smooth_l1"],
    )
    dino_change = _paired_delta(
        candidate_samples["dino_mse"], baseline_samples["dino_mse"]
    )
    logvar = candidate_report["normal_logvar"]
    checks = {
        "plan_beats_w_random_head": float(plan_improvement["ci95_low"]) > 0.0,
        "trajectory_beats_w_random_head": float(trajectory_improvement["ci95_low"])
        > 0.0,
        "future_condition_has_positive_signal": max(
            float(future_shuffle_degradation["ci95_low"]),
            float(current_only_degradation["ci95_low"]),
        )
        > 0.0,
        "plan_query_has_positive_signal": float(plan_shuffle_degradation["ci95_low"])
        > 0.0,
        "effect_regression_below_10_percent": float(
            effect_change["fraction_of_reference"]
        )
        <= 0.10,
        "dino_regression_below_10_percent": float(
            dino_change["fraction_of_reference"]
        )
        <= 0.10,
        "logvar_boundary_fraction_below_5_percent": (
            float(logvar["at_lower_bound_fraction"])
            + float(logvar["at_upper_bound_fraction"])
        )
        <= 0.05,
    }
    return {
        "checks": checks,
        "recommended_enter_phase_e": all(checks.values()),
        "paired_comparisons": {
            "w_minus_p_plan_loss": plan_improvement,
            "w_minus_p_trajectory_total": trajectory_improvement,
            "shuffled_future_minus_normal_plan_loss": future_shuffle_degradation,
            "current_only_minus_normal_plan_loss": current_only_degradation,
            "shuffled_plan_minus_normal_plan_loss": plan_shuffle_degradation,
            "p_minus_w_effect_smooth_l1": effect_change,
            "p_minus_w_dino_mse": dino_change,
        },
        "threshold_notes": {
            "paired_signal": "95% paired CI lower bound must be positive",
            "effect_dino_regression": "candidate may regress by at most 10% vs W",
            "logvar_saturation": "combined exact lower/upper clamp occupancy <=5%",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--w-selection", required=True, type=Path)
    parser.add_argument("--p-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--log-every", default=20, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1:
        raise ValueError("batch-size/log-every must be positive and num-workers non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase P gate: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase P gate requires a CUDA device")

    w_checkpoint, w_metadata = _selected_checkpoint(
        args.w_selection, expected_phase="W"
    )
    p_checkpoint, p_metadata = _selected_checkpoint(
        args.p_selection, expected_phase="P"
    )
    cfg = OmegaConf.load(args.config)
    if str(cfg.framework.action_model.stage2_phase).upper() != "P":
        raise ValueError("Phase P gate config must instantiate stage2_phase=P")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_output = args.output.parent / "dataset-runtime"
    runtime_output.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(runtime_output)
    cfg.trackers = []
    cfg.trainer.pretrained_checkpoint = str(w_checkpoint)
    cfg.trainer.stage2_parent_selection = str(args.w_selection.resolve())
    cfg.datasets.vla_data.per_device_batch_size = int(args.batch_size)
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.val_num_workers = int(args.num_workers)
    cfg.datasets.vla_data.drop_last = False

    PartialState()
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    framework = build_framework(cfg)
    framework = apply_training_freeze_policy(framework, cfg)
    _, validation_loader = build_dataloaders(cfg)
    if validation_loader is None:
        raise RuntimeError("Phase P gate could not build the validation dataloader")

    TrainerUtils.load_finetune_init_weights(
        framework,
        str(w_checkpoint),
        strict_structure=True,
    )
    framework.to(device)
    baseline_report, baseline_samples = _evaluate_checkpoint(
        framework=framework,
        loader=validation_loader,
        device=device,
        label="W",
        include_ablations=False,
        log_every=args.log_every,
    )

    TrainerUtils.load_finetune_init_weights(
        framework,
        str(p_checkpoint),
        strict_structure=True,
    )
    candidate_report, candidate_samples = _evaluate_checkpoint(
        framework=framework,
        loader=validation_loader,
        device=device,
        label="P",
        include_ablations=True,
        log_every=args.log_every,
    )
    if baseline_report["rows"] != candidate_report["rows"]:
        raise AssertionError("W and P evaluations covered different validation rows")

    output = {
        "schema_version": SCHEMA_VERSION,
        "split": "validation",
        "full_validation_split": True,
        "temporal_contract": str(
            cfg.framework.action_model.stage2_temporal_contract
        ),
        "trajectory_representation_slots": 24,
        "physical_valid_count": 8,
        "w_checkpoint": w_metadata,
        "p_checkpoint": p_metadata,
        "w_baseline": baseline_report,
        "p_candidate": candidate_report,
        "gate": _gate_decision(
            baseline_report=baseline_report,
            baseline_samples=baseline_samples,
            candidate_report=candidate_report,
            candidate_samples=candidate_samples,
        ),
        "test_metrics_read": False,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PHASE_P_GATE_COMPLETE output={args.output}", flush=True)
    print(
        "PHASE_P_GATE_DECISION recommended_enter_phase_e="
        f"{output['gate']['recommended_enter_phase_e']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
