"""Standard-normal Gaussian ActionVAE for RoboTwin 1.2-second trajectories."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from latent_action_model.action_ae.beta_vae import (
    ActionBetaVAEConfig,
    ActionBetaVAEOutput,
    StandardNormalActionVAE,
    standard_normal_kl_per_dimension,
)
from latent_action_model.action_ae.distributions import scheduled_kl_weight
from latent_action_model.action_ae.vae_lightning import load_deterministic_backbone

from .lightning import RoboTwinActionAELightningModule
from .losses import dual_arm_trajectory_reconstruction_loss


class RoboTwinActionBetaVAELightningModule(RoboTwinActionAELightningModule):
    """Jointly fine-tune the selected dual-arm ActionAE as a beta-VAE."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        state_dim: int = 14,
        horizon: int = 36,
        hidden_dim: int = 128,
        condition_dim: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        reconstruction_beta: float = 0.1,
        kl_beta: float = 1.0e-3,
        kl_mean_warmup_fraction: float = 0.05,
        kl_ramp_fraction: float = 0.25,
        free_bits_nats: float = 0.02,
        total_training_steps: int | None = None,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        posterior_logvar_init: float = -4.0,
        deterministic_checkpoint_path: str | Path | None = None,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        gripper_weight: float = 0.5,
        condition_source: str | None = None,
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
            condition_source=condition_source,
        )
        schedule_values = (
            kl_beta,
            kl_mean_warmup_fraction,
            kl_ramp_fraction,
            free_bits_nats,
        )
        if any(not math.isfinite(float(value)) for value in schedule_values):
            raise ValueError("RoboTwin beta-VAE KL settings must be finite")
        if kl_beta < 0 or free_bits_nats < 0:
            raise ValueError("kl_beta and free_bits_nats must be non-negative")
        if total_training_steps is not None and total_training_steps < 1:
            raise ValueError("total_training_steps must be positive when provided")

        config = ActionBetaVAEConfig(
            state_dim=int(state_dim),
            variant="a1",
            horizon=int(horizon),
            trajectory_dim=14,
            hidden_dim=int(hidden_dim),
            condition_dim=int(condition_dim),
            kernel_size=int(kernel_size),
            dilations=tuple(int(value) for value in dilations),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            posterior_logvar_init=float(posterior_logvar_init),
        )
        self.model = StandardNormalActionVAE(config)
        self.deterministic_initialization: dict[str, Any] | None = None
        if deterministic_checkpoint_path is not None:
            self.deterministic_initialization = load_deterministic_backbone(
                self.model,
                deterministic_checkpoint_path,
                expected_normalization_sha256=self.normalization_sha256,
                expected_schema_versions=("robotwin-action-tcn-ae-v1",),
                expected_condition_source=self.condition_source,
                expected_condition_checkpoint_sha256=self.condition_checkpoint_sha256,
            )

        self.kl_beta = float(kl_beta)
        self.kl_mean_warmup_fraction = float(kl_mean_warmup_fraction)
        self.kl_ramp_fraction = float(kl_ramp_fraction)
        self.free_bits_nats = float(free_bits_nats)
        self.total_training_steps = (
            int(total_training_steps) if total_training_steps is not None else None
        )
        self.save_hyperparameters(
            {
                "architecture": "robotwin-action-tcn-beta-vae-v1",
                "reconstruction_beta": reconstruction_beta,
                "kl_beta": self.kl_beta,
                "kl_mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "kl_ramp_fraction": self.kl_ramp_fraction,
                "free_bits_nats": self.free_bits_nats,
                "total_training_steps": self.total_training_steps,
                "logvar_min": logvar_min,
                "logvar_max": logvar_max,
                "posterior_logvar_init": posterior_logvar_init,
                "deterministic_checkpoint_path": (
                    str(Path(deterministic_checkpoint_path).expanduser().resolve())
                    if deterministic_checkpoint_path is not None
                    else None
                ),
            }
        )

    def _schedule_total_steps(self) -> int:
        if self.total_training_steps is not None:
            return self.total_training_steps
        trainer = getattr(self, "_trainer", None)
        estimate = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if estimate < 1:
            raise RuntimeError(
                "RoboTwin Action beta-VAE needs total_training_steps or an attached Trainer"
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
        condition_mean: Tensor,
        *,
        time_mask: Tensor | None = None,
        sample_posterior: bool | None = None,
    ) -> ActionBetaVAEOutput:
        return self.model(
            trajectory,
            current_state,
            condition_mean,
            time_mask=time_mask,
            sample_posterior=sample_posterior,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        if not torch.all(batch["condition_valid"]):
            raise ValueError("RoboTwin A1 beta-VAE requires valid frozen condition means")
        warmup_steps = int(
            math.ceil(self._schedule_total_steps() * self.kl_mean_warmup_fraction)
        )
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["condition_mean"],
            time_mask=batch["time_mask"],
            sample_posterior=stage == "train" and int(self.global_step) >= warmup_steps,
        )
        reconstruction = dual_arm_trajectory_reconstruction_loss(
            output.reconstruction,
            batch["trajectory"],
            time_mask=batch["time_mask"],
            arm_pose_mask=batch["arm_pose_mask"],
            arm_gripper_mask=batch["arm_gripper_mask"],
            beta=self.beta,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
            gripper_weight=self.gripper_weight,
        )
        kl_per_dimension = standard_normal_kl_per_dimension(output.posterior)
        kl_raw = kl_per_dimension.mean()
        kl_regularized = kl_per_dimension.clamp_min(self.free_bits_nats).mean()
        kl_weight = self.current_kl_weight() if stage == "train" else self.kl_beta
        total = reconstruction.total + kl_weight * kl_regularized
        batch_size = int(batch["trajectory"].shape[0])
        scalars = (
            ("loss", total),
            ("reconstruction_loss", reconstruction.total),
            ("kl", kl_raw),
            ("kl_free_bits", kl_regularized),
            ("posterior_mean_abs", output.posterior.mean.float().abs().mean()),
            ("posterior_std", torch.exp(0.5 * output.posterior.logvar.float()).mean()),
            (
                "posterior_mu_batch_var",
                output.posterior.mean.float().var(dim=0, unbiased=False).mean(),
            ),
            (
                "active_units",
                (
                    output.posterior.mean.float().var(dim=0, unbiased=False) > 1.0e-2
                )
                .float()
                .sum(),
            ),
        )
        for name, value in scalars:
            self.log(
                f"{stage}/{name}",
                value,
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
        for name, value, count in (
            ("position_loss", reconstruction.position, reconstruction.position_valid_elements),
            ("rotation_loss", reconstruction.rotation, reconstruction.rotation_valid_elements),
            ("gripper_loss", reconstruction.gripper, reconstruction.gripper_valid_elements),
        ):
            self._log_loss(f"{stage}/{name}", value, count)
        if stage != "train":
            self._physical_metrics(
                self._denormalize(output.reconstruction),
                batch["trajectory_raw"],
                batch,
                stage,
            )
        return total

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["action_ae_metadata"].update(
            schema_version="robotwin-action-tcn-beta-vae-v1",
            model_config=self.model.config.to_dict(),
            latent_distribution={
                "family": "diagonal_gaussian",
                "prior": "standard_normal",
                "latent_dim": self.model.config.latent_dim,
                "parameter_dim": 2 * self.model.config.latent_dim,
                "parameter_layout": "mean_then_logvar",
                "logvar_parameterization": "tanh_bounded",
                "logvar_bounds": [
                    self.model.config.logvar_min,
                    self.model.config.logvar_max,
                ],
                "teacher_value": "posterior_mean",
            },
            kl={
                "direction": "trajectory_posterior_to_standard_normal",
                "reduction": "mean_over_latent_then_batch",
                "target_beta": self.kl_beta,
                "free_bits_nats_per_dimension": self.free_bits_nats,
                "mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "ramp_fraction": self.kl_ramp_fraction,
                "total_training_steps": self.total_training_steps,
            },
            training_initialization={
                "deterministic_backbone": self.deterministic_initialization,
                "jointly_trainable": [
                    "condition_mlp.",
                    "encoder.",
                    "posterior_logvar.",
                    "decoder.",
                ],
            },
        )
