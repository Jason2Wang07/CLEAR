"""Numerically stable losses and schedules for effect-aware Stage 1.

The neighborhood target is fixed and detached.  Candidate latents come from the
global memory bank and are detached as well; gradients therefore update the
live posterior mean for each anchor in the current optimizer batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeighborhoodLossOutput:
    loss: Tensor
    teacher_perplexity: Tensor
    mean_candidate_count: Tensor
    mean_teacher_bandwidth: Tensor


def linear_neighbor_weight(
    optimizer_step: int,
    *,
    steps_per_epoch: int,
    target: float = 0.05,
    zero_epochs: int = 1,
    ramp_epochs: int = 4,
) -> float:
    """Locked V1 schedule, continuously interpolated within an epoch."""
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if target < 0 or zero_epochs < 0 or ramp_epochs <= 0:
        raise ValueError("invalid neighborhood schedule parameters")
    start = int(zero_epochs * steps_per_epoch)
    duration = int(ramp_epochs * steps_per_epoch)
    progress = min(1.0, max(0.0, (int(optimizer_step) - start) / float(duration)))
    return float(target) * progress


def warmup_cosine_multiplier(
    optimizer_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """One-epoch linear warm-up followed by cosine decay to a non-zero floor."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    step = max(0, min(int(optimizer_step), int(total_steps)))
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def _as_anchor_matrix(mu: Tensor) -> Tensor:
    if mu.ndim == 3 and mu.shape[1] == 1:
        mu = mu[:, 0]
    if mu.ndim != 2:
        raise ValueError(f"anchor_mu must be [B,D] or [B,1,D], got {tuple(mu.shape)}")
    return mu.float()


def neighborhood_distribution_loss(
    anchor_mu: Tensor,
    candidate_mu: Tensor,
    teacher_distances: Tensor,
    teacher_bandwidth: Tensor,
    *,
    student_gamma: float,
    candidate_mask: Optional[Tensor] = None,
    expected_latent_dim: int = 32,
    eps: float = 1.0e-12,
) -> NeighborhoodLossOutput:
    """Forward KL ``KL(stopgrad(p_E) || p_Z)`` on a shared candidate set.

    All probability and distance calculations are forced to FP32 as required by
    the locked V1 specification.  ``candidate_mu`` represents the detached
    full-training-set memory bank.
    """
    anchor = _as_anchor_matrix(anchor_mu)
    candidates = candidate_mu.detach().float()
    teacher_d = teacher_distances.detach().float()
    bandwidth = teacher_bandwidth.detach().float().reshape(-1)

    if candidates.ndim != 3:
        raise ValueError(f"candidate_mu must be [B,C,D], got {tuple(candidates.shape)}")
    batch, count, dim = candidates.shape
    if anchor.shape != (batch, dim):
        raise ValueError(
            f"anchor/candidate shape mismatch: anchor={tuple(anchor.shape)}, candidates={tuple(candidates.shape)}"
        )
    if dim != int(expected_latent_dim):
        raise ValueError(f"expected latent dim {expected_latent_dim}, got {dim}")
    if teacher_d.shape != (batch, count):
        raise ValueError(
            f"teacher_distances must be [B,C]={batch,count}, got {tuple(teacher_d.shape)}"
        )
    if bandwidth.shape != (batch,):
        raise ValueError(f"teacher_bandwidth must be [B], got {tuple(bandwidth.shape)}")
    if not math.isfinite(float(student_gamma)) or float(student_gamma) <= 0:
        raise ValueError(f"student_gamma must be finite and positive, got {student_gamma}")
    if not torch.isfinite(anchor).all() or not torch.isfinite(candidates).all():
        raise ValueError("latent tensors contain NaN/Inf")

    if candidate_mask is None:
        valid = torch.ones((batch, count), dtype=torch.bool, device=anchor.device)
    else:
        valid = candidate_mask.to(device=anchor.device, dtype=torch.bool)
        if valid.shape != (batch, count):
            raise ValueError(f"candidate_mask must be [B,C], got {tuple(valid.shape)}")
    valid_counts = valid.sum(dim=1)
    if torch.any(valid_counts == 0):
        bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"anchors without valid candidates: {bad[:8]}")

    teacher_d = teacher_d.to(anchor.device)
    bandwidth = bandwidth.to(anchor.device).clamp_min(float(eps))
    candidates = candidates.to(anchor.device)
    neg_inf = torch.tensor(float("-inf"), device=anchor.device, dtype=torch.float32)

    teacher_logits = -(teacher_d.square()) / (2.0 * bandwidth[:, None].square())
    teacher_logits = torch.where(valid, teacher_logits, neg_inf)
    log_p_teacher = torch.log_softmax(teacher_logits, dim=1)
    p_teacher = log_p_teacher.exp().detach()

    latent_d = (anchor[:, None, :] - candidates).square().sum(dim=-1) / float(expected_latent_dim)
    student_logits = -torch.log1p(latent_d / (float(student_gamma) ** 2))
    student_logits = torch.where(valid, student_logits, neg_inf)
    log_p_student = torch.log_softmax(student_logits, dim=1)

    # Mask before multiplication so padded entries never evaluate 0 * -inf.
    per_term = torch.where(
        valid,
        p_teacher * (log_p_teacher - log_p_student),
        torch.zeros_like(log_p_student),
    )
    per_anchor = per_term.sum(dim=1)
    loss = per_anchor.mean()

    entropy = torch.where(valid, -p_teacher * log_p_teacher, torch.zeros_like(p_teacher)).sum(dim=1)
    perplexity = entropy.exp().mean()
    return NeighborhoodLossOutput(
        loss=loss,
        teacher_perplexity=perplexity,
        mean_candidate_count=valid_counts.float().mean(),
        mean_teacher_bandwidth=bandwidth.mean(),
    )
