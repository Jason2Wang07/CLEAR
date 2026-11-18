"""Lightning module for paired RoboTwin ActionAE A0/A1 experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor

from latent_action_model.action_ae.losses import mean_so3_geodesic_error, so3_geodesic_error
from latent_action_model.action_ae.model import ActionAEConfig, ActionAutoencoder

from .losses import dual_arm_trajectory_reconstruction_loss


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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


class RoboTwinActionAELightningModule(LightningModule):
    GROUP_WIDTHS = {
        "left_position": 3,
        "left_rotation_rotvec": 3,
        "left_gripper": 1,
        "right_position": 3,
        "right_rotation_rotvec": 3,
        "right_gripper": 1,
    }

    def __init__(
        self,
        normalization_path: str | Path,
        *,
        variant: str = "a0",
        state_dim: int = 14,
        horizon: int = 48,
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
        condition_source: str | None = None,
    ) -> None:
        super().__init__()
        normalization_file = Path(normalization_path).expanduser().resolve()
        if not normalization_file.is_file():
            raise FileNotFoundError(normalization_file)
        normalization = json.loads(normalization_file.read_text(encoding="utf-8"))
        if normalization.get("schema_version") not in {
            "robotwin-z-action-normalization-v1",
            "robotwin-z-action-normalization-v1-12s",
        }:
            raise ValueError("RoboTwin ActionAE received an unsupported normalization schema")
        if normalization.get("fit_split") != "train":
            raise ValueError("normalization must be fit on train")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        config = ActionAEConfig(
            state_dim=state_dim,
            variant=variant,
            horizon=int(horizon),
            trajectory_dim=14,
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
        self.normalization = normalization
        self.normalization_path = str(normalization_file)
        self.normalization_sha256 = _sha256_file(normalization_file)

        sidecar_root = normalization_file.parents[1]
        checksums_path = sidecar_root / "checksums.sha256"
        schema_path = sidecar_root / "schema.json"
        provenance_path = sidecar_root / "provenance.json"
        self.sidecar_checksums_sha256 = _sha256_file(checksums_path) if checksums_path.is_file() else None
        self.sidecar_schema_sha256 = _sha256_file(schema_path) if schema_path.is_file() else None
        self.sidecar_provenance_sha256 = _sha256_file(provenance_path) if provenance_path.is_file() else None
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
        schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {}
        detected_condition_source = str(
            schema.get("condition", {}).get(
                "source", "e39" if variant == "a1" else "none"
            )
        )
        if condition_source is not None and str(condition_source) != detected_condition_source:
            raise ValueError(
                f"requested condition_source={condition_source!r}, sidecar provides "
                f"{detected_condition_source!r}"
            )
        self.condition_source = detected_condition_source
        self.e39_cache_sha256 = _optional_sha256(provenance, "e39", "sha256")
        self.e39_checkpoint_sha256 = _optional_sha256(provenance, "e39", "checkpoint_sha256")
        self.effect_manifest_sha256 = _optional_sha256(provenance, "effect_manifest_sha256")
        condition_provenance = provenance.get("condition", provenance.get(self.condition_source, {}))
        self.condition_checkpoint_sha256 = _optional_sha256(
            {"condition": condition_provenance}, "condition", "checkpoint_sha256"
        )

        for group, width in self.GROUP_WIDTHS.items():
            payload = normalization.get(group, {})
            median = torch.as_tensor(payload.get("median"), dtype=torch.float32)
            scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32)
            if median.shape != (width,) or scale.shape != (width,):
                raise ValueError(f"normalization group {group!r} must have width {width}")
            if not torch.isfinite(median).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError(f"normalization group {group!r} is invalid")
            self.register_buffer(f"{group}_median", median, persistent=True)
            self.register_buffer(f"{group}_scale", scale, persistent=True)
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
                "condition_source": self.condition_source,
                "condition_checkpoint_sha256": self.condition_checkpoint_sha256,
                "variant": variant,
                "state_dim": state_dim,
                "horizon": int(horizon),
                "trajectory_dim": 14,
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
        condition_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ):
        return self.model(
            trajectory,
            current_state,
            condition_mean if self.variant == "a1" else None,
            time_mask=time_mask,
        )

    def _denormalize(self, trajectory: Tensor) -> Tensor:
        parts = []
        for name, value in (
            ("left_position", trajectory[..., 0:3]),
            ("left_rotation_rotvec", trajectory[..., 3:6]),
            ("left_gripper", trajectory[..., 6:7]),
            ("right_position", trajectory[..., 7:10]),
            ("right_rotation_rotvec", trajectory[..., 10:13]),
            ("right_gripper", trajectory[..., 13:14]),
        ):
            parts.append(value * getattr(self, f"{name}_scale") + getattr(self, f"{name}_median"))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        valid = mask.to(device=value.device, dtype=torch.bool)
        return value[valid].mean() if torch.any(valid) else value.new_tensor(float("nan"))

    def _log_loss(self, name: str, value: Tensor, count: Tensor) -> None:
        scalar_count = int(count.detach().item())
        if scalar_count:
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                batch_size=scalar_count,
                sync_dist=True,
            )

    def _physical_metrics(self, prediction: Tensor, target: Tensor, batch: dict[str, Any], stage: str) -> None:
        time = batch["time_mask"].bool()
        pose = batch["arm_pose_mask"].bool() & time.unsqueeze(-1)
        gripper = batch["arm_gripper_mask"].bool() & time.unsqueeze(-1)
        arm_specs = {
            "left": (slice(0, 3), slice(3, 6), 6, 0),
            "right": (slice(7, 10), slice(10, 13), 13, 1),
        }
        position_values = []
        position_masks = []
        rotation_values = []
        rotation_masks = []
        gripper_values = []
        gripper_masks = []
        for arm, (position_slice, rotation_slice, gripper_index, mask_index) in arm_specs.items():
            pose_valid = pose[..., mask_index]
            gripper_valid = gripper[..., mask_index]
            position_cm = torch.linalg.vector_norm(
                prediction[..., position_slice] - target[..., position_slice], dim=-1
            ) * 100.0
            rotation_pointwise = so3_geodesic_error(
                prediction[..., rotation_slice],
                target[..., rotation_slice],
                degrees=True,
            )
            rotation_deg = mean_so3_geodesic_error(
                prediction[..., rotation_slice], target[..., rotation_slice], mask=pose_valid, degrees=True
            )
            gripper_mae = (prediction[..., gripper_index] - target[..., gripper_index]).abs()
            if torch.any(pose_valid):
                self.log(
                    f"{stage}/{arm}_position_l2_cm",
                    self._masked_mean(position_cm, pose_valid),
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(pose_valid.sum()),
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/{arm}_rotation_geodesic_deg",
                    rotation_deg,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(pose_valid.sum()),
                    sync_dist=True,
                )
            if torch.any(gripper_valid):
                self.log(
                    f"{stage}/{arm}_gripper_mae",
                    self._masked_mean(gripper_mae, gripper_valid),
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(gripper_valid.sum()),
                    sync_dist=True,
                )
            position_values.append(position_cm)
            position_masks.append(pose_valid)
            rotation_values.append(rotation_pointwise)
            rotation_masks.append(pose_valid)
            gripper_values.append(gripper_mae)
            gripper_masks.append(gripper_valid)
        combined_position = torch.cat(position_values, dim=1)
        combined_position_mask = torch.cat(position_masks, dim=1)
        combined_gripper = torch.cat(gripper_values, dim=1)
        combined_gripper_mask = torch.cat(gripper_masks, dim=1)
        combined_rotation = torch.cat(rotation_values, dim=1)
        combined_rotation_mask = torch.cat(rotation_masks, dim=1)
        self.log(
            f"{stage}/position_l2_cm",
            self._masked_mean(combined_position, combined_position_mask),
            on_step=False,
            on_epoch=True,
            batch_size=int(combined_position_mask.sum()),
            sync_dist=True,
        )
        self.log(
            f"{stage}/rotation_geodesic_deg",
            self._masked_mean(combined_rotation, combined_rotation_mask),
            on_step=False,
            on_epoch=True,
            batch_size=int(combined_rotation_mask.sum()),
            sync_dist=True,
        )
        self.log(
            f"{stage}/gripper_mae",
            self._masked_mean(combined_gripper, combined_gripper_mask),
            on_step=False,
            on_epoch=True,
            batch_size=int(combined_gripper_mask.sum()),
            sync_dist=True,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> Tensor:
        if self.variant == "a1" and not torch.all(batch["condition_valid"]):
            raise ValueError("A1 batch contains an invalid frozen condition posterior mean")
        output = self(
            batch["trajectory"],
            batch["current_state"],
            batch["condition_mean"] if self.variant == "a1" else None,
            time_mask=batch["time_mask"],
        )
        losses = dual_arm_trajectory_reconstruction_loss(
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
        batch_size = int(batch["trajectory"].shape[0])
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_loss(f"{stage}/position_loss", losses.position, losses.position_valid_elements)
        self._log_loss(f"{stage}/rotation_loss", losses.rotation, losses.rotation_valid_elements)
        self._log_loss(f"{stage}/gripper_loss", losses.gripper, losses.gripper_valid_elements)
        if stage != "train":
            self._physical_metrics(self._denormalize(output.reconstruction), batch["trajectory_raw"], batch, stage)
        return losses.total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "test")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["action_ae_metadata"] = {
            "schema_version": "robotwin-action-tcn-ae-v1",
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
            "condition_source": self.condition_source,
            "condition_checkpoint_sha256": self.condition_checkpoint_sha256,
            "loss": {
                "type": "masked_dual_arm_smooth_l1",
                "beta": self.beta,
                "weights": {
                    "position": self.position_weight,
                    "rotation_rotvec": self.rotation_weight,
                    "gripper": self.gripper_weight,
                },
            },
        }
