"""Masked dual-arm trajectory losses for RoboTwin ActionAE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True)
class DualArmTrajectoryLoss:
    total: Tensor
    position: Tensor
    rotation: Tensor
    gripper: Tensor
    position_valid_elements: Tensor
    rotation_valid_elements: Tensor
    gripper_valid_elements: Tensor


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor]:
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    pointwise = functional.smooth_l1_loss(safe_prediction, safe_target, beta=beta, reduction="none")
    count = valid.sum()
    value = pointwise.sum() / count.clamp_min(1).to(pointwise.dtype)
    return torch.where(count > 0, value, pointwise.sum() * 0.0), count


def dual_arm_trajectory_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    time_mask: Tensor,
    arm_pose_mask: Tensor,
    arm_gripper_mask: Tensor,
    beta: float = 0.1,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
    gripper_weight: float = 0.5,
) -> DualArmTrajectoryLoss:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 14:
        raise ValueError(f"RoboTwin trajectories must have matching [B,T,14] shapes, got {prediction.shape}/{target.shape}")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    batch, horizon = prediction.shape[:2]
    if time_mask.shape != (batch, horizon):
        raise ValueError("time_mask must be [B,T]")
    if arm_pose_mask.shape != (batch, horizon, 2) or arm_gripper_mask.shape != (batch, horizon, 2):
        raise ValueError("arm masks must be [B,T,2]")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    weights = (position_weight, rotation_weight, gripper_weight)
    if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
        raise ValueError(f"loss weights must be finite and non-negative, got {weights}")

    time = time_mask.to(device=prediction.device, dtype=torch.bool)
    pose = arm_pose_mask.to(device=prediction.device, dtype=torch.bool) & time.unsqueeze(-1)
    gripper = arm_gripper_mask.to(device=prediction.device, dtype=torch.bool) & time.unsqueeze(-1)
    position_prediction = torch.cat((prediction[..., 0:3], prediction[..., 7:10]), dim=-1)
    position_target = torch.cat((target[..., 0:3], target[..., 7:10]), dim=-1)
    position_valid = torch.cat(
        (pose[..., 0:1].expand(-1, -1, 3), pose[..., 1:2].expand(-1, -1, 3)), dim=-1
    )
    rotation_prediction = torch.cat((prediction[..., 3:6], prediction[..., 10:13]), dim=-1)
    rotation_target = torch.cat((target[..., 3:6], target[..., 10:13]), dim=-1)
    rotation_valid = position_valid
    gripper_prediction = torch.stack((prediction[..., 6], prediction[..., 13]), dim=-1)
    gripper_target = torch.stack((target[..., 6], target[..., 13]), dim=-1)

    position_loss, position_count = _masked_smooth_l1(
        position_prediction, position_target, position_valid, beta=beta
    )
    rotation_loss, rotation_count = _masked_smooth_l1(
        rotation_prediction, rotation_target, rotation_valid, beta=beta
    )
    gripper_loss, gripper_count = _masked_smooth_l1(
        gripper_prediction, gripper_target, gripper, beta=beta
    )
    total = (
        float(position_weight) * position_loss
        + float(rotation_weight) * rotation_loss
        + float(gripper_weight) * gripper_loss
    )
    return DualArmTrajectoryLoss(
        total=total,
        position=position_loss,
        rotation=rotation_loss,
        gripper=gripper_loss,
        position_valid_elements=position_count,
        rotation_valid_elements=rotation_count,
        gripper_valid_elements=gripper_count,
    )
