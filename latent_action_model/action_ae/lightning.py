"""Lightning training module for deterministic z-action TCN autoencoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor

from .losses import mean_so3_geodesic_error, trajectory_reconstruction_loss
from .model import ActionAEConfig, ActionAutoencoder


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_tensor(payload: dict[str, Any], group: str, field: str, width: int) -> Tensor:
    value = torch.as_tensor(payload[group][field], dtype=torch.float32)
    if value.shape != (width,) or not torch.isfinite(value).all():
        raise ValueError(f"normalization {group}.{field} must be finite shape-({width},)")
    if field == "scale_mad" and torch.any(value <= 0):
        raise ValueError(f"normalization {group}.{field} must be positive")
    return value


def _optional_sha256(payload: dict[str, Any], *keys: str) -> str | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"provenance {'.'.join(keys)} must be a SHA256 hex digest")
    return digest


class ActionAELightningModule(LightningModule):
    """Train A0/A1 with the locked v0 reconstruction contract."""

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        variant: str = "a0",
        state_dim: int = 14,
        horizon: int = 32,
        hidden_dim: int = 128,
        condition_dim: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        beta: float = 0.1,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        gripper_weight: float = 0.5,
    ) -> None:
        super().__init__()
        normalization_file = Path(normalization_path).expanduser().resolve()
        if not normalization_file.is_file():
            raise FileNotFoundError(f"normalization file does not exist: {normalization_file}")
        normalization = json.loads(normalization_file.read_text(encoding="utf-8"))
        if normalization.get("fit_split") != "train":
            raise ValueError("ActionAE normalization must be fit on the official train split")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")

        config = ActionAEConfig(
            state_dim=state_dim,
            variant=variant,
            horizon=int(horizon),
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            kernel_size=kernel_size,
            dilations=tuple(int(value) for value in dilations),
        )
        self.model = ActionAutoencoder(config)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.beta = float(beta)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(rotation_weight)
        self.gripper_weight = float(gripper_weight)
        self.normalization_path = str(normalization_file)
        self.normalization_sha256 = _sha256_file(normalization_file)
        self.normalization = normalization
        sidecar_root = normalization_file.parents[1]
        checksums_path = sidecar_root / "checksums.sha256"
        schema_path = sidecar_root / "schema.json"
        provenance_path = sidecar_root / "provenance.json"
        self.sidecar_checksums_sha256 = _sha256_file(checksums_path) if checksums_path.is_file() else None
        self.sidecar_schema_sha256 = _sha256_file(schema_path) if schema_path.is_file() else None
        self.sidecar_provenance_sha256 = _sha256_file(provenance_path) if provenance_path.is_file() else None
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
        self.e39_cache_sha256 = _optional_sha256(provenance, "e39", "sha256")
        self.e39_checkpoint_sha256 = _optional_sha256(provenance, "e39", "checkpoint_sha256")
        self.effect_manifest_sha256 = _optional_sha256(provenance, "effect_manifest_sha256")

        for name, group, width in (
            ("position", "position", 3),
            ("rotation", "rotation_rotvec", 3),
            ("gripper", "gripper", 1),
        ):
            self.register_buffer(
                f"{name}_median",
                _normalization_tensor(normalization, group, "median", width),
                persistent=True,
            )
            self.register_buffer(
                f"{name}_scale",
                _normalization_tensor(normalization, group, "scale_mad", width),
                persistent=True,
            )
        self.save_hyperparameters(
            {
                "normalization_path": self.normalization_path,
                "normalization_sha256": self.normalization_sha256,
                "sidecar_checksums_sha256": self.sidecar_checksums_sha256,
                "sidecar_schema_sha256": self.sidecar_schema_sha256,
                "sidecar_provenance_sha256": self.sidecar_provenance_sha256,
                "e39_cache_sha256": self.e39_cache_sha256,
                "e39_checkpoint_sha256": self.e39_checkpoint_sha256,
                "effect_manifest_sha256": self.effect_manifest_sha256,
                "variant": variant,
                "state_dim": state_dim,
                "horizon": int(horizon),
                "hidden_dim": hidden_dim,
                "condition_dim": condition_dim,
                "kernel_size": kernel_size,
                "dilations": tuple(int(value) for value in dilations),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "beta": self.beta,
                "position_weight": self.position_weight,
                "rotation_weight": self.rotation_weight,
                "gripper_weight": self.gripper_weight,
            }
        )

    @property
    def variant(self) -> str:
        return self.model.config.variant

    def forward(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        e39_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ):
        return self.model(
            trajectory,
            current_state,
            e39_mean if self.variant == "a1" else None,
            time_mask=time_mask,
        )

    def _denormalize(self, trajectory: Tensor) -> Tensor:
        return torch.cat(
            (
                trajectory[..., :3] * self.position_scale + self.position_median,
                trajectory[..., 3:6] * self.rotation_scale + self.rotation_median,
                trajectory[..., 6:7] * self.gripper_scale + self.gripper_median,
            ),
            dim=-1,
        )

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        valid = mask.to(device=value.device, dtype=torch.bool)
        if not torch.any(valid):
            return value.new_tensor(float("nan"))
        return value[valid].mean()

    def _log_modality_loss(
        self,
        name: str,
        value: Tensor,
        valid_elements: Tensor,
        *,
        stage: str,
    ) -> None:
        count = int(valid_elements.detach().item())
        if count == 0:
            return
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            batch_size=count,
            sync_dist=True,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        if self.variant == "a1" and not torch.all(batch["e39_valid"]):
            raise ValueError("A1 batch contains rows without a valid frozen E39 posterior mean")
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["e39_mean"] if self.variant == "a1" else None,
            time_mask=batch["time_mask"],
        )
        losses = trajectory_reconstruction_loss(
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
        batch_size = int(batch["trajectory"].shape[0])
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_modality_loss(
            f"{stage}/position_loss", losses.position, losses.position_valid_elements, stage=stage
        )
        self._log_modality_loss(
            f"{stage}/rotation_loss", losses.rotation, losses.rotation_valid_elements, stage=stage
        )
        self._log_modality_loss(
            f"{stage}/gripper_loss", losses.gripper, losses.gripper_valid_elements, stage=stage
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
        return losses.total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "train")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["action_ae_metadata"] = {
            "schema_version": "action-tcn-ae-v0",
            "model_config": self.model.config.to_dict(),
            "normalization": self.normalization,
            "normalization_path": self.normalization_path,
            "normalization_sha256": self.normalization_sha256,
            "sidecar_checksums_sha256": self.sidecar_checksums_sha256,
            "sidecar_schema_sha256": self.sidecar_schema_sha256,
            "sidecar_provenance_sha256": self.sidecar_provenance_sha256,
            "e39_cache_sha256": self.e39_cache_sha256,
            "e39_checkpoint_sha256": self.e39_checkpoint_sha256,
            "effect_manifest_sha256": self.effect_manifest_sha256,
            "loss": {
                "type": "masked_smooth_l1",
                "beta": self.beta,
                "weights": {
                    "position": self.position_weight,
                    "rotation_rotvec": self.rotation_weight,
                    "gripper": self.gripper_weight,
                },
            },
        }
