"""Lightning module for future-aware RoboTwin ActionVAE v2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from latent_action_model.action_ae.distributions import diagonal_gaussian_kl, scheduled_kl_weight
from latent_action_model.action_ae.model import (
    ActionVAEConfig,
    ActionVAEOutput,
    ActionVariationalAutoencoder,
)

from .lightning import RoboTwinActionAELightningModule
from .losses import dual_arm_trajectory_reconstruction_loss


class RoboTwinActionVAELightningModule(RoboTwinActionAELightningModule):
    """Train a 32D dual-arm ActionVAE with a future-aware Gaussian prior."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        state_dim: int = 14,
        horizon: int = 48,
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
        if total_training_steps is not None and total_training_steps < 1:
            raise ValueError("total_training_steps must be positive when provided")
        self.model = ActionVariationalAutoencoder(
            ActionVAEConfig(
                state_dim=state_dim,
                horizon=int(horizon),
                trajectory_dim=14,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                context_dim=context_dim,
                visual_feature_dim=visual_feature_dim,
                kernel_size=kernel_size,
                dilations=tuple(int(value) for value in dilations),
                logvar_min=float(logvar_min),
                logvar_max=float(logvar_max),
            )
        )
        self.kl_beta = float(kl_beta)
        self.kl_mean_warmup_fraction = float(kl_mean_warmup_fraction)
        self.kl_ramp_fraction = float(kl_ramp_fraction)
        self.total_training_steps = int(total_training_steps) if total_training_steps is not None else None
        self.save_hyperparameters(
            {
                "architecture": "robotwin-action-vae-v2",
                "context_dim": context_dim,
                "visual_feature_dim": visual_feature_dim,
                "reconstruction_beta": reconstruction_beta,
                "kl_beta": self.kl_beta,
                "kl_mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "kl_ramp_fraction": self.kl_ramp_fraction,
                "total_training_steps": self.total_training_steps,
                "logvar_min": logvar_min,
                "logvar_max": logvar_max,
            }
        )

    def _schedule_total_steps(self) -> int:
        if self.total_training_steps is not None:
            return self.total_training_steps
        estimate = int(getattr(getattr(self, "_trainer", None), "estimated_stepping_batches", 0) or 0)
        if estimate < 1:
            raise RuntimeError("ActionVAE needs total_training_steps or an attached Trainer")
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
            raise KeyError(f"RoboTwin ActionVAE batch is missing prior context fields: {missing}")
        if not torch.all(batch["e39_valid"]):
            raise ValueError("RoboTwin ActionVAE requires valid frozen z-effect means")
        warmup_steps = int(math.ceil(self._schedule_total_steps() * self.kl_mean_warmup_fraction))
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["e39_mean"],
            current_context=batch["current_context"],
            current_visual=batch["current_visual"],
            future_visual=batch["future_visual"],
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
        kl = diagonal_gaussian_kl(output.posterior, output.prior)
        kl_weight = self.current_kl_weight() if stage == "train" else self.kl_beta
        total = reconstruction.total + kl_weight * kl
        batch_size = int(batch["trajectory"].shape[0])
        self.log(f"{stage}/loss", total, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/reconstruction_loss", reconstruction.total, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/kl", kl, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/kl_weight", total.new_tensor(kl_weight), on_step=stage == "train", on_epoch=stage != "train", batch_size=batch_size)
        self._log_loss(f"{stage}/position_loss", reconstruction.position, reconstruction.position_valid_elements)
        self._log_loss(f"{stage}/rotation_loss", reconstruction.rotation, reconstruction.rotation_valid_elements)
        self._log_loss(f"{stage}/gripper_loss", reconstruction.gripper, reconstruction.gripper_valid_elements)
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
        metadata = checkpoint["action_ae_metadata"]
        metadata.update(
            schema_version="robotwin-action-vae-v2",
            model_config=self.model.config.to_dict(),
            latent_distribution={
                "latent_dim": 32,
                "parameter_dim": 64,
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
                "target_beta": self.kl_beta,
                "mean_warmup_fraction": self.kl_mean_warmup_fraction,
                "ramp_fraction": self.kl_ramp_fraction,
                "total_training_steps": self.total_training_steps,
            },
        )
