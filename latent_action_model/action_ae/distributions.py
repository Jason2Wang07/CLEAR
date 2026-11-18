"""Diagonal-Gaussian contracts shared by ActionVAE and Stage 2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiagonalGaussian:
    """A validated diagonal Gaussian with tensors shaped ``[B,D]``."""

    mean: Tensor
    logvar: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mean, Tensor) or not isinstance(self.logvar, Tensor):
            raise TypeError("Gaussian mean and logvar must be torch tensors")
        if self.mean.shape != self.logvar.shape or self.mean.ndim != 2:
            raise ValueError(
                "Gaussian mean/logvar must have matching [B,D] shapes, got "
                f"{tuple(self.mean.shape)} and {tuple(self.logvar.shape)}"
            )
        if not self.mean.is_floating_point() or not self.logvar.is_floating_point():
            raise TypeError("Gaussian mean and logvar must be floating point")
        if self.mean.device != self.logvar.device:
            raise ValueError("Gaussian mean and logvar must be on the same device")

    @property
    def latent_dim(self) -> int:
        return int(self.mean.shape[-1])

    def detached(self) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mean.detach(), self.logvar.detach())

    def rsample(self, *, generator: torch.Generator | None = None) -> Tensor:
        epsilon = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + torch.exp(0.5 * self.logvar) * epsilon


def clamp_logvar(
    logvar: Tensor,
    *,
    minimum: float | Tensor = -6.0,
    maximum: float | Tensor = 2.0,
) -> Tensor:
    if not isinstance(logvar, Tensor) or not logvar.is_floating_point():
        raise TypeError("logvar must be a floating-point torch tensor")
    lower = torch.as_tensor(minimum, device=logvar.device, dtype=logvar.dtype)
    upper = torch.as_tensor(maximum, device=logvar.device, dtype=logvar.dtype)
    if lower.ndim > 1 or upper.ndim > 1:
        raise ValueError("logvar bounds must be scalars or one-dimensional tensors")
    if lower.ndim == 1 and lower.shape != logvar.shape[-1:]:
        raise ValueError("minimum logvar vector must match the latent dimension")
    if upper.ndim == 1 and upper.shape != logvar.shape[-1:]:
        raise ValueError("maximum logvar vector must match the latent dimension")
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("logvar bounds must be finite")
    if not bool((lower < upper).all()):
        raise ValueError("each logvar minimum must be smaller than its maximum")
    return torch.maximum(torch.minimum(logvar, upper), lower)


def split_gaussian_parameters(
    parameters: Tensor,
    *,
    latent_dim: int,
    logvar_min: float | Tensor = -6.0,
    logvar_max: float | Tensor = 2.0,
) -> DiagonalGaussian:
    if not isinstance(parameters, Tensor) or parameters.ndim != 2:
        raise ValueError("Gaussian parameters must have shape [B,2*D]")
    if tuple(parameters.shape[1:]) != (2 * int(latent_dim),):
        raise ValueError(
            f"Gaussian parameters must be [B,{2 * int(latent_dim)}], got {tuple(parameters.shape)}"
        )
    mean, logvar = parameters.split(int(latent_dim), dim=-1)
    return DiagonalGaussian(
        mean=mean,
        logvar=clamp_logvar(logvar, minimum=logvar_min, maximum=logvar_max),
    )


def diagonal_gaussian_kl(
    posterior: DiagonalGaussian,
    prior: DiagonalGaussian,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Return analytic ``KL(posterior || prior)`` normalized over latent dims."""

    if posterior.mean.shape != prior.mean.shape:
        raise ValueError(
            "posterior/prior shape mismatch: "
            f"{tuple(posterior.mean.shape)} vs {tuple(prior.mean.shape)}"
        )
    if posterior.mean.device != prior.mean.device:
        raise ValueError("posterior and prior must be on the same device")
    posterior_mean = posterior.mean.float()
    posterior_logvar = posterior.logvar.float()
    prior_mean = prior.mean.float()
    prior_logvar = prior.logvar.float()
    per_dimension = 0.5 * (
        prior_logvar
        - posterior_logvar
        + torch.exp(posterior_logvar - prior_logvar)
        + (posterior_mean - prior_mean).square() * torch.exp(-prior_logvar)
        - 1.0
    )
    per_sample = per_dimension.mean(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError("KL reduction must be 'none', 'mean', or 'sum'")


def diagonal_gaussian_nll(
    target: Tensor,
    distribution: DiagonalGaussian,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Gaussian negative log likelihood, normalized over latent dimensions."""

    if not isinstance(target, Tensor) or target.shape != distribution.mean.shape:
        raise ValueError(
            "Gaussian NLL target must match distribution [B,D] shape, got "
            f"{getattr(target, 'shape', None)} and {tuple(distribution.mean.shape)}"
        )
    if target.device != distribution.mean.device:
        raise ValueError("Gaussian NLL target and distribution must be on the same device")
    if not target.is_floating_point():
        raise TypeError("Gaussian NLL target must be floating point")
    target_float = target.float()
    mean = distribution.mean.float()
    logvar = distribution.logvar.float()
    per_dimension = 0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (target_float - mean).square() * torch.exp(-logvar)
    )
    per_sample = per_dimension.mean(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError("Gaussian NLL reduction must be 'none', 'mean', or 'sum'")


def standardize_gaussian(
    distribution: DiagonalGaussian,
    *,
    center: Tensor,
    scale: Tensor,
) -> DiagonalGaussian:
    """Apply ``z_standard = (z - center) / scale`` to a Gaussian."""

    center = torch.as_tensor(center, device=distribution.mean.device, dtype=distribution.mean.dtype).reshape(-1)
    scale = torch.as_tensor(scale, device=distribution.mean.device, dtype=distribution.mean.dtype).reshape(-1)
    if center.shape != (distribution.latent_dim,) or scale.shape != center.shape:
        raise ValueError("Gaussian standardization center/scale must match the latent dimension")
    if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
        raise ValueError("Gaussian standardization contains NaN/Inf")
    if bool((scale <= 0).any()):
        raise ValueError("Gaussian standardization scale must be positive")
    return DiagonalGaussian(
        mean=(distribution.mean - center) / scale,
        logvar=distribution.logvar - 2.0 * torch.log(scale),
    )


def scheduled_kl_weight(
    step: int,
    *,
    total_steps: int,
    target: float,
    mean_warmup_fraction: float = 0.05,
    ramp_fraction: float = 0.20,
) -> float:
    """Zero-KL mean warm-up followed by a linear ramp to ``target``."""

    if int(step) < 0 or int(total_steps) < 1:
        raise ValueError("step must be non-negative and total_steps must be positive")
    values = (target, mean_warmup_fraction, ramp_fraction)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("KL schedule values must be finite")
    if target < 0 or not 0 <= mean_warmup_fraction < 1 or not 0 < ramp_fraction <= 1:
        raise ValueError("invalid KL target or schedule fractions")
    if mean_warmup_fraction + ramp_fraction > 1:
        raise ValueError("KL warm-up and ramp fractions cannot exceed the training duration")
    warmup_steps = int(math.ceil(total_steps * mean_warmup_fraction))
    ramp_steps = max(1, int(math.ceil(total_steps * ramp_fraction)))
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (int(step) - warmup_steps + 1) / ramp_steps)
    return float(target) * progress
