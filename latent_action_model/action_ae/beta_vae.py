"""TCN beta-VAE with a fixed standard-normal latent prior."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .distributions import DiagonalGaussian
from .model import ActionAEConfig, ActionAutoencoder


@dataclass(frozen=True)
class ActionBetaVAEConfig(ActionAEConfig):
    """Architecture contract for the standard-normal Action beta-VAE."""

    variant: str = "a1"
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    posterior_logvar_init: float = -4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.logvar_min >= self.logvar_max:
            raise ValueError("Action beta-VAE logvar_min must be smaller than logvar_max")
        if not self.logvar_min < self.posterior_logvar_init < self.logvar_max:
            raise ValueError("posterior_logvar_init must lie strictly within the configured logvar bounds")


def bounded_logvar_from_logits(
    logits: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> Tensor:
    midpoint = 0.5 * (float(minimum) + float(maximum))
    half_range = 0.5 * (float(maximum) - float(minimum))
    return midpoint + half_range * torch.tanh(logits)


def _logvar_value_to_logit(value: float, *, minimum: float, maximum: float) -> float:
    midpoint = 0.5 * (float(minimum) + float(maximum))
    half_range = 0.5 * (float(maximum) - float(minimum))
    return math.atanh((float(value) - midpoint) / half_range)


@dataclass(frozen=True)
class ActionBetaVAEOutput:
    reconstruction: Tensor
    z_action: Tensor
    posterior: DiagonalGaussian


class StandardNormalActionVAE(ActionAutoencoder):
    """Encode an action trajectory into ``q(z|a,c)`` and regularize it to N(0,I)."""

    config: ActionBetaVAEConfig

    def __init__(self, config: ActionBetaVAEConfig) -> None:
        super().__init__(config)
        self.posterior_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(
            self.posterior_logvar.bias,
            _logvar_value_to_logit(
                config.posterior_logvar_init,
                minimum=config.logvar_min,
                maximum=config.logvar_max,
            ),
        )

    def posterior(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> DiagonalGaussian:
        self._validate_trajectory(trajectory)
        if current_state.device != trajectory.device:
            raise ValueError("current_state and trajectory must be on the same device")
        if current_state.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f"trajectory/current-state batch mismatch: {trajectory.shape[0]} vs {current_state.shape[0]}"
            )
        condition = self._condition(current_state, effect_mean)
        pooled = self.encoder.pooled_features(trajectory, condition, time_mask=time_mask)
        return DiagonalGaussian(
            mean=self.encoder.latent_projection(pooled),
            logvar=bounded_logvar_from_logits(
                self.posterior_logvar(pooled),
                minimum=self.config.logvar_min,
                maximum=self.config.logvar_max,
            ),
        )

    def encode(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the posterior mean used as the deterministic Stage-2 teacher."""

        posterior = self.posterior(
            trajectory,
            current_state,
            effect_mean,
            time_mask=time_mask,
        )
        return posterior.mean.unsqueeze(1)

    def forward(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        sample_posterior: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> ActionBetaVAEOutput:
        posterior = self.posterior(
            trajectory,
            current_state,
            effect_mean,
            time_mask=time_mask,
        )
        should_sample = self.training if sample_posterior is None else bool(sample_posterior)
        latent = posterior.rsample(generator=generator) if should_sample else posterior.mean
        z_action = latent.unsqueeze(1)
        reconstruction = self.decode(z_action, current_state, effect_mean)
        return ActionBetaVAEOutput(
            reconstruction=reconstruction,
            z_action=z_action,
            posterior=posterior,
        )


def standard_normal_kl_per_dimension(posterior: DiagonalGaussian) -> Tensor:
    """Return analytic KL(q || N(0,I)) as a float32 ``[B,D]`` tensor."""

    mean = posterior.mean.float()
    logvar = posterior.logvar.float()
    return 0.5 * (mean.square() + torch.exp(logvar) - 1.0 - logvar)
