"""Masked reconstruction losses and physical rotation evaluation for ActionAE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryLossOutput:
    total: Tensor
    position: Tensor
    rotation: Tensor
    gripper: Tensor
    position_valid_elements: Tensor
    rotation_valid_elements: Tensor
    gripper_valid_elements: Tensor


def _validate_trajectories(prediction: Tensor, target: Tensor) -> tuple[int, int]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 3 or prediction.shape[-1] != 7:
        raise ValueError(f"canonical trajectories must be [B,T,7], got {tuple(prediction.shape)}")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    return int(prediction.shape[0]), int(prediction.shape[1])


def _mask(mask: Tensor, *, name: str, batch: int, horizon: int, device: torch.device) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if mask.shape != (batch, horizon):
        raise ValueError(f"{name} must be [B,T]={batch,horizon}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    valid_steps: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor]:
    valid = valid_steps.unsqueeze(-1).expand_as(prediction)
    # Invalid sidecar entries may contain sentinels/NaNs; replace them before
    # evaluating the pointwise loss so they cannot poison a masked reduction.
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    pointwise = functional.smooth_l1_loss(safe_prediction, safe_target, beta=beta, reduction="none")
    valid_count = valid.sum()
    denominator = valid_count.clamp_min(1).to(dtype=pointwise.dtype)
    value = pointwise.sum() / denominator
    value = torch.where(valid_count > 0, value, pointwise.sum() * 0.0)
    return value, valid_count


def trajectory_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    time_mask: Tensor,
    pose_mask: Tensor,
    gripper_mask: Tensor,
    beta: float = 0.1,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
    gripper_weight: float = 0.5,
) -> TrajectoryLossOutput:
    """Compute independently normalized position/rotvec/gripper Smooth-L1.

    Position and rotation share ``time_mask & pose_mask``. Gripper uses
    ``time_mask & gripper_mask`` so embodiments without a gripper can set the
    entire gripper mask to false without changing the other reductions.
    """

    batch, horizon = _validate_trajectories(prediction, target)
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError(f"beta must be finite and positive, got {beta}")
    weights = (position_weight, rotation_weight, gripper_weight)
    if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
        raise ValueError(f"loss weights must be finite and non-negative, got {weights}")

    time = _mask(time_mask, name="time_mask", batch=batch, horizon=horizon, device=prediction.device)
    pose = _mask(pose_mask, name="pose_mask", batch=batch, horizon=horizon, device=prediction.device)
    gripper_validity = _mask(
        gripper_mask,
        name="gripper_mask",
        batch=batch,
        horizon=horizon,
        device=prediction.device,
    )
    pose_validity = time & pose
    gripper_validity = time & gripper_validity

    position, position_count = _masked_smooth_l1(
        prediction[..., :3],
        target[..., :3],
        pose_validity,
        beta=beta,
    )
    rotation, rotation_count = _masked_smooth_l1(
        prediction[..., 3:6],
        target[..., 3:6],
        pose_validity,
        beta=beta,
    )
    gripper, gripper_count = _masked_smooth_l1(
        prediction[..., 6:7],
        target[..., 6:7],
        gripper_validity,
        beta=beta,
    )
    total = position_weight * position + rotation_weight * rotation + gripper_weight * gripper
    return TrajectoryLossOutput(
        total=total,
        position=position,
        rotation=rotation,
        gripper=gripper,
        position_valid_elements=position_count,
        rotation_valid_elements=rotation_count,
        gripper_valid_elements=gripper_count,
    )


def _rotvec_to_unit_quaternion(rotation_vector: Tensor) -> Tensor:
    angle_squared = rotation_vector.square().sum(dim=-1, keepdim=True)
    angle = angle_squared.sqrt()
    safe_angle = angle.clamp_min(torch.finfo(rotation_vector.dtype).eps)
    regular_scale = torch.sin(0.5 * angle) / safe_angle
    small_scale = 0.5 - angle_squared / 48.0 + angle_squared.square() / 3840.0
    scale = torch.where(angle > 1.0e-4, regular_scale, small_scale)
    quaternion = torch.cat((torch.cos(0.5 * angle), rotation_vector * scale), dim=-1)
    return functional.normalize(quaternion, dim=-1)


def so3_geodesic_error(
    predicted_rotvec: Tensor,
    target_rotvec: Tensor,
    *,
    degrees: bool = True,
) -> Tensor:
    """Pointwise shortest-path SO(3) error for rotation vectors.

    The quaternion absolute inner product makes antipodal quaternions
    equivalent. The returned shape is ``predicted_rotvec.shape[:-1]`` and the
    default unit is degrees, matching the v0 evaluation spec.
    """

    if not isinstance(predicted_rotvec, Tensor) or not isinstance(target_rotvec, Tensor):
        raise TypeError("predicted_rotvec and target_rotvec must be torch tensors")
    if predicted_rotvec.shape != target_rotvec.shape:
        raise ValueError(
            f"predicted/target rotation shape mismatch: {tuple(predicted_rotvec.shape)} vs "
            f"{tuple(target_rotvec.shape)}"
        )
    if predicted_rotvec.ndim < 2 or predicted_rotvec.shape[-1] != 3:
        raise ValueError(f"rotation vectors must end in dimension 3, got {tuple(predicted_rotvec.shape)}")
    if not predicted_rotvec.is_floating_point() or not target_rotvec.is_floating_point():
        raise TypeError("rotation vectors must be floating point")
    if predicted_rotvec.device != target_rotvec.device:
        raise ValueError("predicted_rotvec and target_rotvec must be on the same device")

    predicted_quaternion = _rotvec_to_unit_quaternion(predicted_rotvec)
    target_quaternion = _rotvec_to_unit_quaternion(target_rotvec)
    cosine = (predicted_quaternion * target_quaternion).sum(dim=-1).abs().clamp(max=1.0)
    error = 2.0 * torch.acos(cosine)
    if degrees:
        error = torch.rad2deg(error)
    return error


def mean_so3_geodesic_error(
    predicted_rotvec: Tensor,
    target_rotvec: Tensor,
    *,
    mask: Tensor | None = None,
    degrees: bool = True,
) -> Tensor:
    """Mean shortest-path SO(3) error, returning NaN for an empty mask."""

    error = so3_geodesic_error(predicted_rotvec, target_rotvec, degrees=degrees)
    if mask is None:
        return error.mean()
    expected_shape = predicted_rotvec.shape[:-1]
    if not isinstance(mask, Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if mask.shape != expected_shape:
        raise ValueError(f"mask must have shape {tuple(expected_shape)}, got {tuple(mask.shape)}")
    valid = mask.to(device=error.device, dtype=torch.bool)
    if not torch.any(valid):
        return error.new_tensor(float("nan"))
    return error[valid].mean()
