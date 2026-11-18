"""Lightning training module for the standard-normal TCN Action beta-VAE."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .beta_vae import (
    ActionBetaVAEConfig,
    ActionBetaVAEOutput,
    StandardNormalActionVAE,
    standard_normal_kl_per_dimension,
)
from .distributions import scheduled_kl_weight
from .lightning import ActionAELightningModule
from .losses import mean_so3_geodesic_error, trajectory_reconstruction_loss
from .vae_lightning import load_deterministic_backbone


class ActionBetaVAELightningModule(ActionAELightningModule):
    """Jointly fine-tune a reconstruction-trained TCN AE as a beta-VAE."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        variant: str = "a1",
        state_dim: int = 14,
        horizon: int = 24,
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
    ) -> None:
        super().__init__(
            normalization_path,
            variant=variant,
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
        schedule_values = (kl_beta, kl_mean_warmup_fraction, kl_ramp_fraction, free_bits_nats)
        if any(not math.isfinite(float(value)) for value in schedule_values):
            raise ValueError("beta-VAE KL settings must be finite")
        if kl_beta < 0 or free_bits_nats < 0:
            raise ValueError("kl_beta and free_bits_nats must be non-negative")
        if total_training_steps is not None and total_training_steps < 1:
            raise ValueError("total_training_steps must be positive when provided")

        config = ActionBetaVAEConfig(
            state_dim=state_dim,
            variant=variant,
            horizon=int(horizon),
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            kernel_size=kernel_size,
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
            )

        self.kl_beta = float(kl_beta)
        self.kl_mean_warmup_fraction = float(kl_mean_warmup_fraction)
        self.kl_ramp_fraction = float(kl_ramp_fraction)
        self.free_bits_nats = float(free_bits_nats)
        self.total_training_steps = int(total_training_steps) if total_training_steps is not None else None
        self.save_hyperparameters(
            {
                "architecture": "action-tcn-beta-vae-v1",
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
            raise RuntimeError("Action beta-VAE needs total_training_steps or an attached Trainer")
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
        e39_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        sample_posterior: bool | None = None,
    ) -> ActionBetaVAEOutput:
        return self.model(
            trajectory,
            current_state,
            e39_mean if self.variant == "a1" else None,
            time_mask=time_mask,
            sample_posterior=sample_posterior,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        if self.variant == "a1" and not torch.all(batch["e39_valid"]):
            raise ValueError("A1 beta-VAE batch contains rows without a frozen z-effect mean")
        total_steps = self._schedule_total_steps()
        warmup_steps = int(math.ceil(total_steps * self.kl_mean_warmup_fraction))
        sample_posterior = stage == "train" and int(self.global_step) >= warmup_steps
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["e39_mean"] if self.variant == "a1" else None,
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
            ("posterior_mu_batch_var", output.posterior.mean.float().var(dim=0, unbiased=False).mean()),
            ("active_units", (output.posterior.mean.float().var(dim=0, unbiased=False) > 1.0e-2).float().sum()),
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
            self._log_modality_loss(f"{stage}/{name}", value, count, stage=stage)

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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["action_ae_metadata"].update(
            schema_version="action-tcn-beta-vae-v1",
            latent_distribution={
                "family": "diagonal_gaussian",
                "prior": "standard_normal",
                "latent_dim": self.model.config.latent_dim,
                "parameter_dim": 2 * self.model.config.latent_dim,
                "parameter_layout": "mean_then_logvar",
                "logvar_parameterization": "tanh_bounded",
                "logvar_bounds": [self.model.config.logvar_min, self.model.config.logvar_max],
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
                "jointly_trainable": ["condition_mlp.", "encoder.", "posterior_logvar.", "decoder."],
            },
        )
