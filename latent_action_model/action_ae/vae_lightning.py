"""Lightning training module for future-aware ActionVAE v2."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import Tensor
from torch.nn import functional

from .distributions import (
    diagonal_gaussian_kl,
    diagonal_gaussian_nll,
    scheduled_kl_weight,
)
from .lightning import ActionAELightningModule
from .losses import mean_so3_geodesic_error, trajectory_reconstruction_loss
from .model import ActionVAEConfig, ActionVAEOutput, ActionVariationalAutoencoder
from .phase1_gate import validate_phase1_gate


_DETERMINISTIC_BACKBONE_PREFIXES = ("condition_mlp.", "encoder.", "decoder.")
_PHASE2_PREFIXES = ("posterior_logvar.", "prior.")


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_deterministic_backbone(
    model: ActionVariationalAutoencoder,
    checkpoint_path: str | Path,
    *,
    expected_normalization_sha256: str,
    expected_schema_versions: tuple[str, ...] = ("action-tcn-ae-v0",),
    expected_condition_source: str | None = None,
    expected_condition_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly transplant a deterministic ActionAE representation into Phase 2."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"deterministic ActionAE checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("deterministic ActionAE checkpoint must be a mapping")
    metadata = checkpoint.get("action_ae_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") not in set(
        expected_schema_versions
    ):
        raise ValueError(
            "deterministic initialization checkpoint schema mismatch: "
            f"got={metadata.get('schema_version') if isinstance(metadata, Mapping) else None!r}, "
            f"expected={sorted(expected_schema_versions)}"
        )
    if str(metadata.get("normalization_sha256", "")).lower() != expected_normalization_sha256.lower():
        raise ValueError("deterministic checkpoint normalization does not match ActionVAE data")
    checkpoint_condition_source = metadata.get("condition_source", "e39")
    if (
        expected_condition_source is not None
        and checkpoint_condition_source != expected_condition_source
    ):
        raise ValueError(
            "deterministic checkpoint condition source mismatch: "
            f"got={checkpoint_condition_source!r}, expected={expected_condition_source!r}"
        )
    checkpoint_condition_sha256 = metadata.get(
        "condition_checkpoint_sha256", metadata.get("e39_checkpoint_sha256")
    )
    if (
        expected_condition_checkpoint_sha256 is not None
        and checkpoint_condition_sha256 != expected_condition_checkpoint_sha256
    ):
        raise ValueError(
            "deterministic checkpoint condition teacher mismatch: "
            f"got={checkpoint_condition_sha256!r}, "
            f"expected={expected_condition_checkpoint_sha256!r}"
        )

    source_config = metadata.get("model_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("deterministic checkpoint is missing model_config metadata")
    target_config = model.config.to_dict()
    shared_fields = (
        "state_dim",
        "variant",
        "horizon",
        "trajectory_dim",
        "latent_dim",
        "effect_dim",
        "hidden_dim",
        "condition_dim",
        "kernel_size",
        "dilations",
    )
    def comparable_value(values: Mapping[str, Any], name: str) -> Any:
        value = values.get(name)
        return tuple(value) if name == "dilations" and value is not None else value

    mismatched = {
        name: (source_config.get(name), target_config.get(name))
        for name in shared_fields
        if comparable_value(source_config, name) != comparable_value(target_config, name)
    }
    if mismatched:
        raise ValueError(f"deterministic/variational architecture mismatch: {mismatched}")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("deterministic checkpoint is missing state_dict")
    source = {
        str(name)[len("model.") :]: value
        for name, value in state_dict.items()
        if str(name).startswith("model.")
    }
    expected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(_DETERMINISTIC_BACKBONE_PREFIXES)
    }
    missing = sorted(set(expected).difference(source))
    unexpected = sorted(set(source).difference(expected))
    shape_mismatch = sorted(
        name
        for name in set(source).intersection(expected)
        if tuple(source[name].shape) != tuple(expected[name].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "deterministic backbone state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape={shape_mismatch[:8]}"
        )
    incompatible = model.load_state_dict(source, strict=False)
    allowed_missing = sorted(
        name
        for name in model.state_dict()
        if name.startswith(("posterior_logvar.", "prior."))
    )
    if sorted(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected deterministic transplant result: {incompatible}")
    for name, value in source.items():
        if not torch.equal(model.state_dict()[name].cpu(), value.cpu()):
            raise AssertionError(f"deterministic backbone tensor changed during transplant: {name}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": metadata["schema_version"],
        "copied_tensors": len(source),
        "frozen_prefixes": list(_DETERMINISTIC_BACKBONE_PREFIXES),
    }


def load_phase2_initialization(
    model: ActionVariationalAutoencoder,
    checkpoint_path: str | Path,
    *,
    expected_normalization_sha256: str,
) -> dict[str, Any]:
    """Load trained Phase-2 heads without restoring optimizer or loop state."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Phase-2 initialization checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Phase-2 initialization checkpoint must be a mapping")
    metadata = checkpoint.get("action_ae_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != "action-vae-v2":
        raise ValueError("Phase-2 initialization requires an action-vae-v2 checkpoint")
    if str(metadata.get("normalization_sha256", "")).lower() != expected_normalization_sha256.lower():
        raise ValueError("Phase-2 initialization normalization does not match the target model")

    source_config = metadata.get("model_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("Phase-2 initialization is missing model_config metadata")
    target_config = model.config.to_dict()
    # Normalize tuple/list serialization before enforcing an exact architecture match.
    mismatched = {}
    for name, target in target_config.items():
        source = source_config.get(name)
        if name == "dilations" and source is not None:
            source = tuple(source)
            target = tuple(target)
        if source != target:
            mismatched[name] = (source_config.get(name), target_config.get(name))
    if mismatched:
        raise ValueError(f"Phase-2 initialization architecture mismatch: {mismatched}")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Phase-2 initialization checkpoint is missing state_dict")
    source = {
        str(name)[len("model.") :]: value
        for name, value in state_dict.items()
        if str(name).startswith("model.")
    }
    expected = model.state_dict()
    missing = sorted(set(expected).difference(source))
    unexpected = sorted(set(source).difference(expected))
    shape_mismatch = sorted(
        name
        for name in set(source).intersection(expected)
        if tuple(source[name].shape) != tuple(expected[name].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "Phase-2 initialization state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape={shape_mismatch[:8]}"
        )
    backbone_mismatch = [
        name
        for name in expected
        if name.startswith(_DETERMINISTIC_BACKBONE_PREFIXES)
        and not torch.equal(source[name].cpu(), expected[name].cpu())
    ]
    if backbone_mismatch:
        raise ValueError(
            "Phase-2 initialization was trained from a different deterministic backbone: "
            f"{backbone_mismatch[:8]}"
        )
    phase2_state = {name: value for name, value in source.items() if name.startswith(_PHASE2_PREFIXES)}
    expected_phase2 = {name for name in expected if name.startswith(_PHASE2_PREFIXES)}
    if set(phase2_state) != expected_phase2:
        raise AssertionError("Phase-2 initialization did not resolve every Phase-2 tensor")
    incompatible = model.load_state_dict(phase2_state, strict=False)
    allowed_missing = sorted(name for name in expected if not name.startswith(_PHASE2_PREFIXES))
    if sorted(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected Phase-2 initialization result: {incompatible}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "source_global_step": int(checkpoint.get("global_step", 0)),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "loaded_prefixes": list(_PHASE2_PREFIXES),
        "optimizer_restored": False,
    }


class ActionVAELightningModule(ActionAELightningModule):
    """Train a 32D ActionVAE with a 64D mean/logvar distribution interface."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        state_dim: int = 14,
        horizon: int = 24,
        hidden_dim: int = 128,
        condition_dim: int = 128,
        context_dim: int = 2048,
        visual_feature_dim: int = 768,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        reconstruction_beta: float = 0.1,
        kl_beta: float = 1.0e-2,
        kl_mean_warmup_fraction: float = 0.05,
        kl_ramp_fraction: float = 0.20,
        total_training_steps: int | None = None,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        posterior_logvar_init: float = -4.0,
        deterministic_checkpoint_path: str | Path | None = None,
        phase1_gate_path: str | Path | None = None,
        phase2_init_checkpoint_path: str | Path | None = None,
        freeze_deterministic_backbone: bool = False,
        training_objective: Literal["gaussian_alignment", "teacher_regression"] = "gaussian_alignment",
        teacher_latent_weight: float = 1.0,
        teacher_action_weight: float = 1.0,
        teacher_nll_weight: float = 1.0e-2,
        teacher_latent_beta: float = 0.1,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        gripper_weight: float = 0.5,
    ) -> None:
        super().__init__(
            normalization_path,
            variant="a1",
            state_dim=state_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta=reconstruction_beta,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
            gripper_weight=gripper_weight,
        )
        if kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if training_objective not in {"gaussian_alignment", "teacher_regression"}:
            raise ValueError(f"unsupported ActionVAE training objective: {training_objective!r}")
        teacher_weights = (teacher_latent_weight, teacher_action_weight, teacher_nll_weight)
        if any(not math.isfinite(float(value)) or value < 0 for value in teacher_weights):
            raise ValueError("teacher-regression loss weights must be finite and non-negative")
        if not math.isfinite(float(teacher_latent_beta)) or teacher_latent_beta <= 0:
            raise ValueError("teacher_latent_beta must be finite and positive")
        if total_training_steps is not None and total_training_steps < 1:
            raise ValueError("total_training_steps must be positive when provided")
        config = ActionVAEConfig(
            state_dim=state_dim,
            horizon=int(horizon),
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            context_dim=context_dim,
            visual_feature_dim=visual_feature_dim,
            kernel_size=kernel_size,
            dilations=tuple(int(value) for value in dilations),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            posterior_logvar_init=float(posterior_logvar_init),
        )
        self.model = ActionVariationalAutoencoder(config)
        self.deterministic_backbone: dict[str, Any] | None = None
        self.phase1_gate: dict[str, Any] | None = None
        self.phase2_initialization: dict[str, Any] | None = None
        if deterministic_checkpoint_path is not None:
            self.deterministic_backbone = load_deterministic_backbone(
                self.model,
                deterministic_checkpoint_path,
                expected_normalization_sha256=self.normalization_sha256,
            )
        if freeze_deterministic_backbone:
            if self.deterministic_backbone is None:
                raise ValueError("frozen Phase 2 requires deterministic_checkpoint_path")
            if phase1_gate_path is None:
                raise ValueError("frozen Phase 2 requires phase1_gate_path")
            self.phase1_gate = validate_phase1_gate(
                phase1_gate_path,
                checkpoint_path=deterministic_checkpoint_path,
            )
            self.model.freeze_deterministic_backbone()
        if phase2_init_checkpoint_path is not None:
            self.phase2_initialization = load_phase2_initialization(
                self.model,
                phase2_init_checkpoint_path,
                expected_normalization_sha256=self.normalization_sha256,
            )
        self.freeze_deterministic_backbone = bool(freeze_deterministic_backbone)
        self.training_objective = str(training_objective)
        self.teacher_latent_weight = float(teacher_latent_weight)
        self.teacher_action_weight = float(teacher_action_weight)
        self.teacher_nll_weight = float(teacher_nll_weight)
        self.teacher_latent_beta = float(teacher_latent_beta)
        if self.training_objective == "teacher_regression":
            if not self.freeze_deterministic_backbone:
                raise ValueError("teacher regression requires a frozen deterministic backbone")
            self.model.posterior_logvar.requires_grad_(False)
        if self.freeze_deterministic_backbone:
            self.model.assert_phase2_trainable_contract(
                train_posterior_logvar=self.training_objective == "gaussian_alignment"
            )
        self.kl_beta = float(kl_beta)
        self.kl_mean_warmup_fraction = float(kl_mean_warmup_fraction)
        self.kl_ramp_fraction = float(kl_ramp_fraction)
        self.total_training_steps = int(total_training_steps) if total_training_steps is not None else None
        self.save_hyperparameters(
            {
                "architecture": "action-vae-v2",
                "context_dim": context_dim,
                "visual_feature_dim": visual_feature_dim,
                "reconstruction_beta": reconstruction_beta,
                "kl_beta": self.kl_beta,
                "kl_mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "kl_ramp_fraction": self.kl_ramp_fraction,
                "total_training_steps": self.total_training_steps,
                "logvar_min": logvar_min,
                "logvar_max": logvar_max,
                "posterior_logvar_init": posterior_logvar_init,
                "training_objective": self.training_objective,
                "teacher_latent_weight": self.teacher_latent_weight,
                "teacher_action_weight": self.teacher_action_weight,
                "teacher_nll_weight": self.teacher_nll_weight,
                "teacher_latent_beta": self.teacher_latent_beta,
                "deterministic_checkpoint_path": (
                    str(Path(deterministic_checkpoint_path).expanduser().resolve())
                    if deterministic_checkpoint_path is not None
                    else None
                ),
                "phase1_gate_path": (
                    str(Path(phase1_gate_path).expanduser().resolve())
                    if phase1_gate_path is not None
                    else None
                ),
                "phase2_init_checkpoint_path": (
                    str(Path(phase2_init_checkpoint_path).expanduser().resolve())
                    if phase2_init_checkpoint_path is not None
                    else None
                ),
                "freeze_deterministic_backbone": self.freeze_deterministic_backbone,
            }
        )

    def _schedule_total_steps(self) -> int:
        if self.total_training_steps is not None:
            return self.total_training_steps
        trainer = getattr(self, "_trainer", None)
        estimate = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if estimate < 1:
            raise RuntimeError(
                "ActionVAE needs total_training_steps or an attached Trainer to compute the KL schedule"
            )
        return estimate

    def current_kl_weight(self) -> float:
        return scheduled_kl_weight(
            int(self.global_step),
            total_steps=self._schedule_total_steps(),
            target=self.kl_beta,
            mean_warmup_fraction=self.kl_mean_warmup_fraction,
            ramp_fraction=self.kl_ramp_fraction,
        )

    def forward(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        e39_mean: Tensor,
        *,
        current_context: Tensor,
        current_visual: Tensor,
        future_visual: Tensor,
        time_mask: Tensor | None = None,
        sample_posterior: bool | None = None,
    ) -> ActionVAEOutput:
        return self.model(
            trajectory,
            current_state,
            e39_mean,
            current_context=current_context,
            current_visual=current_visual,
            future_visual=future_visual,
            time_mask=time_mask,
            sample_posterior=sample_posterior,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        required = {"current_context", "current_visual", "future_visual"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"ActionVAE batch is missing prior context fields: {missing}")
        if not torch.all(batch["e39_valid"]):
            raise ValueError("ActionVAE batch contains rows without a frozen z-effect mean")
        if self.training_objective == "teacher_regression":
            return self._teacher_regression_step(batch, stage)
        return self._gaussian_alignment_step(batch, stage)

    def _gaussian_alignment_step(self, batch: dict[str, Any], stage: str) -> Tensor:
        total_steps = self._schedule_total_steps()
        warmup_steps = int(math.ceil(total_steps * self.kl_mean_warmup_fraction))
        sample_posterior = stage == "train" and int(self.global_step) >= warmup_steps
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["e39_mean"],
            current_context=batch["current_context"],
            current_visual=batch["current_visual"],
            future_visual=batch["future_visual"],
            time_mask=batch["time_mask"],
            sample_posterior=sample_posterior,
        )
        reconstruction = trajectory_reconstruction_loss(
            output.reconstruction,
            batch["trajectory"],
            time_mask=batch["time_mask"],
            pose_mask=batch["pose_mask"],
            gripper_mask=batch["gripper_mask"],
            beta=self.beta,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
            gripper_weight=self.gripper_weight,
        )
        kl = diagonal_gaussian_kl(output.posterior, output.prior)
        kl_weight = self.current_kl_weight() if stage == "train" else self.kl_beta
        total = reconstruction.total + kl_weight * kl
        batch_size = int(batch["trajectory"].shape[0])
        self.log(
            f"{stage}/loss",
            total,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/reconstruction_loss",
            reconstruction.total,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/kl",
            kl,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/kl_weight",
            total.new_tensor(kl_weight),
            on_step=stage == "train",
            on_epoch=stage != "train",
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_modality_loss(
            f"{stage}/position_loss",
            reconstruction.position,
            reconstruction.position_valid_elements,
            stage=stage,
        )
        self._log_modality_loss(
            f"{stage}/rotation_loss",
            reconstruction.rotation,
            reconstruction.rotation_valid_elements,
            stage=stage,
        )
        self._log_modality_loss(
            f"{stage}/gripper_loss",
            reconstruction.gripper,
            reconstruction.gripper_valid_elements,
            stage=stage,
        )
        if stage != "train":
            prediction_raw = self._denormalize(output.reconstruction)
            target_raw = batch["trajectory_raw"]
            pose_valid = batch["time_mask"].bool() & batch["pose_mask"].bool()
            gripper_valid = batch["time_mask"].bool() & batch["gripper_mask"].bool()
            pose_count = int(pose_valid.sum().item())
            if pose_count:
                position_l2_cm = torch.linalg.vector_norm(
                    prediction_raw[..., :3] - target_raw[..., :3], dim=-1
                ) * 100.0
                rotation_deg = mean_so3_geodesic_error(
                    prediction_raw[..., 3:6],
                    target_raw[..., 3:6],
                    mask=pose_valid,
                    degrees=True,
                )
                self.log(
                    f"{stage}/position_l2_cm",
                    self._masked_mean(position_l2_cm, pose_valid),
                    on_step=False,
                    on_epoch=True,
                    batch_size=pose_count,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/rotation_geodesic_deg",
                    rotation_deg,
                    on_step=False,
                    on_epoch=True,
                    batch_size=pose_count,
                    sync_dist=True,
                )
            gripper_count = int(gripper_valid.sum().item())
            if gripper_count:
                self.log(
                    f"{stage}/gripper_mae_m",
                    self._masked_mean(
                        (prediction_raw[..., 6] - target_raw[..., 6]).abs(), gripper_valid
                    ),
                    on_step=False,
                    on_epoch=True,
                    batch_size=gripper_count,
                    sync_dist=True,
                )
        return total

    def _teacher_regression_step(self, batch: dict[str, Any], stage: str) -> Tensor:
        posterior = self.model.posterior(
            batch["trajectory"],
            batch["current_state"],
            batch["e39_mean"],
            time_mask=batch["time_mask"],
        )
        teacher = posterior.mean.detach()
        prior = self.model.prior(
            batch["current_context"],
            batch["current_visual"],
            batch["future_visual"],
            batch["current_state"],
            batch["e39_mean"],
        )
        prior_reconstruction = self.model.decode(
            prior.mean.unsqueeze(1),
            batch["current_state"],
            batch["e39_mean"],
        )
        latent = functional.smooth_l1_loss(
            prior.mean,
            teacher,
            beta=self.teacher_latent_beta,
        )
        action = trajectory_reconstruction_loss(
            prior_reconstruction,
            batch["trajectory"],
            time_mask=batch["time_mask"],
            pose_mask=batch["pose_mask"],
            gripper_mask=batch["gripper_mask"],
            beta=self.beta,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
            gripper_weight=self.gripper_weight,
        )
        nll = diagonal_gaussian_nll(teacher, prior)
        kl = diagonal_gaussian_kl(posterior.detached(), prior)
        total = (
            self.teacher_latent_weight * latent
            + self.teacher_action_weight * action.total
            + self.teacher_nll_weight * nll
        )
        batch_size = int(batch["trajectory"].shape[0])
        for name, value in (
            ("loss", total),
            ("teacher_latent_loss", latent),
            ("prior_reconstruction_loss", action.total),
            ("teacher_nll", nll),
            ("kl", kl),
        ):
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        for name, value, count in (
            ("prior_position_loss", action.position, action.position_valid_elements),
            ("prior_rotation_loss", action.rotation, action.rotation_valid_elements),
            ("prior_gripper_loss", action.gripper, action.gripper_valid_elements),
        ):
            self._log_modality_loss(f"{stage}/{name}", value, count, stage=stage)

        if stage != "train":
            prediction_raw = self._denormalize(prior_reconstruction)
            target_raw = batch["trajectory_raw"]
            pose_valid = batch["time_mask"].bool() & batch["pose_mask"].bool()
            gripper_valid = batch["time_mask"].bool() & batch["gripper_mask"].bool()
            pose_count = int(pose_valid.sum().item())
            if pose_count:
                position_l2_cm = torch.linalg.vector_norm(
                    prediction_raw[..., :3] - target_raw[..., :3], dim=-1
                ) * 100.0
                rotation_deg = mean_so3_geodesic_error(
                    prediction_raw[..., 3:6],
                    target_raw[..., 3:6],
                    mask=pose_valid,
                    degrees=True,
                )
                self.log(
                    f"{stage}/prior_position_l2_cm",
                    self._masked_mean(position_l2_cm, pose_valid),
                    on_step=False,
                    on_epoch=True,
                    batch_size=pose_count,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/prior_rotation_geodesic_deg",
                    rotation_deg,
                    on_step=False,
                    on_epoch=True,
                    batch_size=pose_count,
                    sync_dist=True,
                )
            gripper_count = int(gripper_valid.sum().item())
            if gripper_count:
                self.log(
                    f"{stage}/prior_gripper_mae_m",
                    self._masked_mean(
                        (prediction_raw[..., 6] - target_raw[..., 6]).abs(), gripper_valid
                    ),
                    on_step=False,
                    on_epoch=True,
                    batch_size=gripper_count,
                    sync_dist=True,
                )
        return total

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("ActionVAE has no trainable Phase-2 parameters")
        if self.freeze_deterministic_backbone:
            self.model.assert_phase2_trainable_contract(
                train_posterior_logvar=self.training_objective == "gaussian_alignment"
            )
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        metadata = checkpoint["action_ae_metadata"]
        metadata.update(
            schema_version="action-vae-v2",
            latent_distribution={
                "latent_dim": self.model.config.latent_dim,
                "parameter_dim": 2 * self.model.config.latent_dim,
                "parameter_layout": "mean_then_logvar",
                "logvar_bounds": [self.model.config.logvar_min, self.model.config.logvar_max],
                "teacher_value": "posterior_mean",
            },
            prior_conditioning=[
                "current_vlm_context",
                "current_dino",
                "future_dino",
                "current_state",
                "stopgrad(z_effect)",
            ],
            kl={
                "direction": "posterior_to_prior",
                "reduction": "mean_over_latent_then_batch",
                "target_beta": self.kl_beta,
                "mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "ramp_fraction": self.kl_ramp_fraction,
                "total_training_steps": self.total_training_steps,
            },
            objective={
                "name": self.training_objective,
                "teacher_regression": {
                    "latent_loss": "smooth_l1",
                    "latent_beta": self.teacher_latent_beta,
                    "latent_weight": self.teacher_latent_weight,
                    "prior_action_weight": self.teacher_action_weight,
                    "gaussian_nll_weight": self.teacher_nll_weight,
                    "teacher": "stopgrad(frozen_phase1_posterior_mean)",
                    "action_path": "frozen_decoder(prior_mean)",
                },
            },
            training_phases={
                "phase": self.training_objective if self.freeze_deterministic_backbone else "joint",
                "deterministic_backbone": self.deterministic_backbone,
                "phase1_gate": self.phase1_gate,
                "phase2_initialization": self.phase2_initialization,
                "frozen": self.freeze_deterministic_backbone,
                "trainable_prefixes": (
                    ["posterior_logvar.", "prior."]
                    if self.training_objective == "gaussian_alignment"
                    else ["prior."]
                ),
            },
        )
