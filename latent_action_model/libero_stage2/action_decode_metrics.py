"""Metrics for decoding Stage-2 action latents through the frozen ActionVAE."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from latent_action_model.action_ae.evaluate import trajectory_metrics
from latent_action_model.action_ae.losses import trajectory_reconstruction_loss


DECODE_MODES = (
    "teacher_teacher",
    "predicted_teacher",
    "teacher_predicted",
    "predicted_predicted",
)
REPRESENTATION_STEPS = 24
PHYSICAL_VALID_COUNT = 8
PHASES_PER_CONTROL_STEP = REPRESENTATION_STEPS // PHYSICAL_VALID_COUNT
PHYSICAL_ENDPOINT_INDICES = tuple(
    range(PHASES_PER_CONTROL_STEP - 1, REPRESENTATION_STEPS, PHASES_PER_CONTROL_STEP)
)


def denormalize_current_state(
    normalized_state: Tensor,
    state_mask: Tensor,
    normalization: Mapping[str, Any],
) -> Tensor:
    """Undo the embedded ActionVAE current-state normalization."""

    if normalized_state.ndim != 2 or normalized_state.shape[1:] != (14,):
        raise ValueError("normalized current state must have shape [B,14]")
    if state_mask.shape != normalized_state.shape:
        raise ValueError("current state mask must match normalized state")
    try:
        group = normalization["current_state"]
        median = torch.as_tensor(
            group["median"], device=normalized_state.device, dtype=normalized_state.dtype
        )
        scale = torch.as_tensor(
            group["scale_mad"], device=normalized_state.device, dtype=normalized_state.dtype
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("ActionVAE normalization is missing current_state statistics") from exc
    if median.shape != (14,) or scale.shape != (14,):
        raise ValueError("ActionVAE current_state statistics must have shape [14]")
    if not bool(torch.isfinite(median).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("ActionVAE current_state statistics must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("ActionVAE current_state scale must be positive")
    raw = normalized_state * scale + median
    return torch.where(state_mask.bool(), raw, torch.zeros_like(raw))


def physical_endpoint_mask(time_mask: Tensor, physical_valid_count: Tensor) -> Tensor:
    """Select the eight real 20 Hz endpoints from the 24-slot 0.4 s bridge."""

    if time_mask.ndim != 2 or time_mask.shape[1:] != (REPRESENTATION_STEPS,):
        raise ValueError("0.4s representation time mask must have shape [B,24]")
    if physical_valid_count.shape != (time_mask.shape[0],):
        raise ValueError("physical_valid_count must have shape [B]")
    expected = torch.full_like(physical_valid_count, PHYSICAL_VALID_COUNT)
    if not bool(torch.equal(physical_valid_count, expected)):
        raise ValueError("0.4s bridge physical_valid_count must be exactly eight")
    endpoints = torch.zeros_like(time_mask, dtype=torch.bool)
    endpoints[:, PHYSICAL_ENDPOINT_INDICES] = True
    return endpoints & time_mask.bool()


def _loss_report(loss: Any) -> dict[str, float | int]:
    return {
        "total": float(loss.total),
        "position": float(loss.position),
        "rotation": float(loss.rotation),
        "gripper": float(loss.gripper),
        "position_valid_elements": int(loss.position_valid_elements),
        "rotation_valid_elements": int(loss.rotation_valid_elements),
        "gripper_valid_elements": int(loss.gripper_valid_elements),
    }


def summarize_decoded_trajectories(
    *,
    predictions_normalized: Mapping[str, Tensor],
    predictions_raw: Mapping[str, Tensor],
    target_normalized: Tensor,
    target_raw: Tensor,
    time_mask: Tensor,
    pose_mask: Tensor,
    gripper_mask: Tensor,
    current_state_raw: Tensor,
    current_state_mask: Tensor,
    physical_valid_count: Tensor,
    action_hz: Tensor,
) -> dict[str, Any]:
    """Summarize all bridge slots, physical endpoints, and the final endpoint."""

    if tuple(predictions_normalized) != DECODE_MODES:
        raise ValueError(f"normalized predictions must use modes {DECODE_MODES}")
    if tuple(predictions_raw) != DECODE_MODES:
        raise ValueError(f"raw predictions must use modes {DECODE_MODES}")
    if target_normalized.shape != target_raw.shape:
        raise ValueError("normalized and raw targets must have matching shapes")
    if target_raw.ndim != 3 or target_raw.shape[1:] != (REPRESENTATION_STEPS, 7):
        raise ValueError("trajectory targets must have shape [B,24,7]")
    batch = target_raw.shape[0]
    for name, mask in (
        ("time_mask", time_mask),
        ("pose_mask", pose_mask),
        ("gripper_mask", gripper_mask),
    ):
        if mask.shape != (batch, REPRESENTATION_STEPS):
            raise ValueError(f"{name} must have shape [B,24]")
    if current_state_raw.shape != (batch, 14) or current_state_mask.shape != (batch, 14):
        raise ValueError("current state and mask must have shape [B,14]")
    if action_hz.shape != (batch,) or not bool(torch.isfinite(action_hz).all()):
        raise ValueError("action_hz must contain one finite value per sample")
    if bool((action_hz <= 0).any()):
        raise ValueError("action_hz must be positive")

    physical_time_mask = physical_endpoint_mask(time_mask, physical_valid_count)
    initial_gripper = current_state_raw[:, 6]
    initial_gripper_valid = current_state_mask[:, 6].bool()
    report: dict[str, Any] = {}
    for mode in DECODE_MODES:
        normalized = predictions_normalized[mode]
        raw = predictions_raw[mode]
        if normalized.shape != target_normalized.shape or raw.shape != target_raw.shape:
            raise ValueError(f"{mode} prediction shape differs from target")
        normalized_loss = trajectory_reconstruction_loss(
            normalized,
            target_normalized,
            time_mask=time_mask,
            pose_mask=pose_mask,
            gripper_mask=gripper_mask,
            beta=0.1,
        )
        representation = trajectory_metrics(
            raw,
            target_raw,
            time_mask=time_mask,
            pose_mask=pose_mask,
            gripper_mask=gripper_mask,
            initial_gripper=initial_gripper,
            initial_gripper_valid=initial_gripper_valid,
            control_hz=action_hz * float(PHASES_PER_CONTROL_STEP),
        )
        physical = trajectory_metrics(
            raw,
            target_raw,
            time_mask=physical_time_mask,
            pose_mask=pose_mask,
            gripper_mask=gripper_mask,
            initial_gripper=initial_gripper,
            initial_gripper_valid=initial_gripper_valid,
            control_hz=action_hz,
        )
        report[mode] = {
            "normalized_smooth_l1": _loss_report(normalized_loss),
            "representation_slots_24": representation["all"],
            "physical_endpoints_8": physical["all"],
            "final_endpoint": physical["endpoint"],
        }
    return report


__all__ = [
    "DECODE_MODES",
    "PHYSICAL_ENDPOINT_INDICES",
    "denormalize_current_state",
    "physical_endpoint_mask",
    "summarize_decoded_trajectories",
]
