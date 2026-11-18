"""Conditional deterministic and variational TCN action autoencoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .distributions import DiagonalGaussian, clamp_logvar, split_gaussian_parameters


ActionAEVariant = Literal["a0", "a1"]


@dataclass(frozen=True)
class ActionAEConfig:
    """Serializable v0 architecture configuration.

    ``a0`` conditions on current robot state only. ``a1`` additionally uses a
    detached E39 posterior mean as its effect condition.
    """

    state_dim: int
    variant: ActionAEVariant = "a0"
    horizon: int = 32
    trajectory_dim: int = 7
    latent_dim: int = 32
    effect_dim: int = 32
    hidden_dim: int = 128
    condition_dim: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        integer_fields = {
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "trajectory_dim": self.trajectory_dim,
            "latent_dim": self.latent_dim,
            "effect_dim": self.effect_dim,
            "hidden_dim": self.hidden_dim,
            "condition_dim": self.condition_dim,
            "kernel_size": self.kernel_size,
        }
        invalid = [name for name, value in integer_fields.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.variant not in ("a0", "a1"):
            raise ValueError(f"variant must be 'a0' or 'a1', got {self.variant!r}")
        # The latent/effect interface is shared with the released 32D LaWAM
        # contract.  Horizon and trajectory width are properties of the robot
        # embodiment (LIBERO is 32x7; RoboTwin is 48x14), and the TCN itself is
        # agnostic to both as long as they are fixed by the serialized config.
        locked_v0_dimensions = {
            "latent_dim": (self.latent_dim, 32),
            "effect_dim": (self.effect_dim, 32),
        }
        mismatched = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in locked_v0_dimensions.items()
            if actual != expected
        ]
        if mismatched:
            raise ValueError(f"ActionAE v0 dimensions are fixed: {', '.join(mismatched)}")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so TCN blocks preserve the horizon")
        if not self.dilations or any(int(dilation) <= 0 for dilation in self.dilations):
            raise ValueError("dilations must contain positive integers")

    @property
    def conditions_on_effect(self) -> bool:
        return self.variant == "a1"

    def to_dict(self) -> dict[str, Any]:
        """Return checkpoint-friendly model metadata."""

        return asdict(self)


@dataclass(frozen=True)
class ActionAEOutput:
    reconstruction: Tensor
    z_action: Tensor


@dataclass(frozen=True)
class ActionVAEConfig(ActionAEConfig):
    """ActionVAE v2 architecture with an explicit future-aware prior."""

    variant: ActionAEVariant = "a1"
    context_dim: int = 2048
    visual_feature_dim: int = 768
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    posterior_logvar_init: float = -4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.variant != "a1":
            raise ValueError("ActionVAE v2 requires the frozen z-effect condition (variant='a1')")
        if self.context_dim <= 0 or self.visual_feature_dim <= 0:
            raise ValueError("ActionVAE context and visual feature dimensions must be positive")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("ActionVAE logvar_min must be smaller than logvar_max")
        if not self.logvar_min <= self.posterior_logvar_init <= self.logvar_max:
            raise ValueError("posterior_logvar_init must lie within the configured logvar bounds")


@dataclass(frozen=True)
class ActionVAEOutput:
    reconstruction: Tensor
    z_action: Tensor
    posterior: DiagonalGaussian
    prior: DiagonalGaussian


def freeze_effect_model(effect_model: nn.Module) -> nn.Module:
    """Remove E39 parameters from optimization and select eval mode."""

    effect_model.requires_grad_(False)
    effect_model.eval()
    assert_effect_model_frozen(effect_model)
    return effect_model


def assert_effect_model_frozen(effect_model: nn.Module) -> None:
    """Fail fast if an effect teacher could be updated or behave stochastically."""

    trainable = [name for name, parameter in effect_model.named_parameters() if parameter.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:4])
        raise AssertionError(f"effect model has trainable parameters: {preview}")
    training_modules = [name or "<root>" for name, module in effect_model.named_modules() if module.training]
    if training_modules:
        preview = ", ".join(training_modules[:4])
        raise AssertionError(f"all effect-model modules must remain in eval mode; active: {preview}")


def prepare_effect_condition(effect_mean: Tensor, *, expected_dim: int = 32) -> Tensor:
    """Normalize cached E39 means to ``[B,D]`` and enforce stop-gradient."""

    if not isinstance(effect_mean, Tensor):
        raise TypeError(f"effect_mean must be a torch.Tensor, got {type(effect_mean).__name__}")
    if effect_mean.ndim == 3 and effect_mean.shape[1] == 1:
        effect_mean = effect_mean[:, 0]
    if effect_mean.ndim != 2 or effect_mean.shape[1] != int(expected_dim):
        raise ValueError(
            f"effect_mean must be [B,{expected_dim}] or [B,1,{expected_dim}], got {tuple(effect_mean.shape)}"
        )
    if not effect_mean.is_floating_point():
        raise TypeError("effect_mean must be floating point")
    return effect_mean.detach()


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FiLM(nn.Module):
    """Feature-wise affine modulation for temporal ``[B,C,T]`` features."""

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        return features * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class FiLMResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = _normalization_groups(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.film1 = FiLM(condition_dim, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film2 = FiLM(condition_dim, channels)
        self.activation = nn.SiLU()

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        residual = features
        features = self.activation(self.film1(self.norm1(self.conv1(features)), condition))
        features = self.film2(self.norm2(self.conv2(features)), condition)
        return self.activation(residual + features)


class ConditionMLP(nn.Module):
    """Encode state and, for A1, detached effect through independent MLPs."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.state_mlp = self._branch(config.state_dim, config.condition_dim)
        # Keep this branch in A0 as well, driven by zeros, so A0/A1 have equal
        # parameter counts while A0 still receives no effect information.
        self.effect_mlp = self._branch(config.effect_dim, config.condition_dim)
        self.output_norm = nn.LayerNorm(config.condition_dim)

    @staticmethod
    def _branch(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, current_state: Tensor, effect_mean: Tensor | None = None) -> Tensor:
        if not isinstance(current_state, Tensor):
            raise TypeError(f"current_state must be a torch.Tensor, got {type(current_state).__name__}")
        if current_state.ndim != 2 or current_state.shape[1] != self.config.state_dim:
            raise ValueError(
                f"current_state must be [B,{self.config.state_dim}], got {tuple(current_state.shape)}"
            )
        if not current_state.is_floating_point():
            raise TypeError("current_state must be floating point")

        condition = self.state_mlp(current_state)
        if self.config.conditions_on_effect:
            if effect_mean is None:
                raise ValueError("A1 requires a frozen effect_mean condition")
            effect = prepare_effect_condition(effect_mean, expected_dim=self.config.effect_dim)
            if effect.shape[0] != current_state.shape[0]:
                raise ValueError(
                    f"effect/current-state batch mismatch: {effect.shape[0]} vs {current_state.shape[0]}"
                )
            if effect.device != current_state.device:
                raise ValueError("effect_mean and current_state must be on the same device")
            if effect.dtype != current_state.dtype:
                effect = effect.to(dtype=current_state.dtype)
        elif effect_mean is not None:
            raise ValueError("A0 does not accept an effect_mean condition")
        else:
            effect = current_state.new_zeros((current_state.shape[0], self.config.effect_dim))
        condition = condition + self.effect_mlp(effect)
        return self.output_norm(condition)


class DeterministicTCNEncoder(nn.Module):
    def __init__(self, config: ActionAEConfig, *, output_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(config.trajectory_dim, config.hidden_dim, kernel_size=1)
        self.position_embedding = nn.Parameter(torch.empty(1, config.hidden_dim, config.horizon))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            FiLMResidualTCNBlock(
                config.hidden_dim,
                config.condition_dim,
                config.kernel_size,
                dilation,
            )
            for dilation in config.dilations
        )
        self.output_norm = nn.GroupNorm(_normalization_groups(config.hidden_dim), config.hidden_dim)
        self.output_dim = int(config.latent_dim if output_dim is None else output_dim)
        if self.output_dim < 1:
            raise ValueError("TCN encoder output_dim must be positive")
        self.latent_projection = nn.Linear(config.hidden_dim, self.output_dim)

    def pooled_features(
        self,
        trajectory: Tensor,
        condition: Tensor,
        time_mask: Tensor | None = None,
    ) -> Tensor:
        batch = trajectory.shape[0]
        mask = _validate_time_mask(time_mask, batch=batch, horizon=self.config.horizon, device=trajectory.device)
        if mask is not None:
            trajectory = torch.where(mask.unsqueeze(-1), trajectory, torch.zeros_like(trajectory))
        features = self.input_projection(trajectory.transpose(1, 2)) + self.position_embedding
        if mask is not None:
            channel_mask = mask.unsqueeze(1).to(dtype=features.dtype)
            features = features * channel_mask
        else:
            channel_mask = None

        for block in self.blocks:
            features = block(features, condition)
            if channel_mask is not None:
                features = features * channel_mask
        features = self.output_norm(features)

        if mask is None:
            pooled = features.mean(dim=-1)
        else:
            valid_steps = mask.sum(dim=-1)
            if torch.any(valid_steps == 0):
                bad = torch.nonzero(valid_steps == 0, as_tuple=False).flatten().tolist()
                raise ValueError(f"time_mask contains samples without valid steps: {bad[:8]}")
            pooled = (features * channel_mask).sum(dim=-1) / valid_steps.unsqueeze(-1).to(features.dtype)
        return pooled

    def forward(self, trajectory: Tensor, condition: Tensor, time_mask: Tensor | None = None) -> Tensor:
        pooled = self.pooled_features(trajectory, condition, time_mask=time_mask)
        return self.latent_projection(pooled).unsqueeze(1)


class ConditionalTCNDecoder(nn.Module):
    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, config.hidden_dim, config.horizon))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            FiLMResidualTCNBlock(
                config.hidden_dim,
                config.condition_dim,
                config.kernel_size,
                dilation,
            )
            for dilation in reversed(config.dilations)
        )
        self.output_norm = nn.GroupNorm(_normalization_groups(config.hidden_dim), config.hidden_dim)
        # Intentionally linear: robust-normalized trajectory targets are not bounded to [-1, 1].
        self.output_projection = nn.Conv1d(config.hidden_dim, config.trajectory_dim, kernel_size=1)

    def forward(self, z_action: Tensor, condition: Tensor) -> Tensor:
        if z_action.ndim != 3 or z_action.shape[1:] != (1, self.config.latent_dim):
            raise ValueError(
                f"z_action must be [B,1,{self.config.latent_dim}], got {tuple(z_action.shape)}"
            )
        features = self.latent_projection(z_action[:, 0]).unsqueeze(-1)
        features = features.expand(-1, -1, self.config.horizon) + self.position_embedding
        for block in self.blocks:
            features = block(features, condition)
        features = self.output_norm(features)
        return self.output_projection(features).transpose(1, 2)


class ActionAutoencoder(nn.Module):
    """Direct deterministic TCN-AE used by the A0 and A1 experiments."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_mlp = ConditionMLP(config)
        self.encoder = DeterministicTCNEncoder(config)
        self.decoder = ConditionalTCNDecoder(config)

    def _validate_trajectory(self, trajectory: Tensor) -> None:
        if not isinstance(trajectory, Tensor):
            raise TypeError(f"trajectory must be a torch.Tensor, got {type(trajectory).__name__}")
        expected = (self.config.horizon, self.config.trajectory_dim)
        if trajectory.ndim != 3 or trajectory.shape[1:] != expected:
            raise ValueError(f"trajectory must be [B,{expected[0]},{expected[1]}], got {tuple(trajectory.shape)}")
        if not trajectory.is_floating_point():
            raise TypeError("trajectory must be floating point")

    def _condition(self, current_state: Tensor, effect_mean: Tensor | None) -> Tensor:
        condition = self.condition_mlp(current_state, effect_mean)
        return condition

    def encode(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_trajectory(trajectory)
        if current_state.device != trajectory.device:
            raise ValueError("current_state and trajectory must be on the same device")
        if current_state.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f"trajectory/current-state batch mismatch: {trajectory.shape[0]} vs {current_state.shape[0]}"
            )
        condition = self._condition(current_state, effect_mean)
        return self.encoder(trajectory, condition, time_mask=time_mask)

    def decode(
        self,
        z_action: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(z_action, Tensor):
            raise TypeError(f"z_action must be a torch.Tensor, got {type(z_action).__name__}")
        if current_state.device != z_action.device:
            raise ValueError("current_state and z_action must be on the same device")
        condition = self._condition(current_state, effect_mean)
        if condition.shape[0] != z_action.shape[0]:
            raise ValueError(f"z_action/condition batch mismatch: {z_action.shape[0]} vs {condition.shape[0]}")
        return self.decoder(z_action, condition)

    def forward(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> ActionAEOutput:
        z_action = self.encode(
            trajectory,
            current_state,
            effect_mean,
            time_mask=time_mask,
        )
        reconstruction = self.decode(z_action, current_state, effect_mean)
        return ActionAEOutput(reconstruction=reconstruction, z_action=z_action)


def _pooled_feature(value: Tensor, *, name: str, width: int, batch: int) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch tensor")
    if value.ndim == 3:
        if value.shape[0] != batch or value.shape[-1] != width:
            raise ValueError(f"{name} must be [B,N,{width}] or [B,{width}]")
        value = value.mean(dim=1)
    if value.ndim != 2 or tuple(value.shape) != (batch, width):
        raise ValueError(f"{name} must be [B,N,{width}] or [B,{width}], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


class FutureAwareActionPrior(nn.Module):
    """Predict ``p(z_action | h, F_t, F_t+H, state, z_effect)``."""

    def __init__(self, config: ActionVAEConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.condition_dim
        self.context = nn.Sequential(nn.LayerNorm(config.context_dim), nn.Linear(config.context_dim, hidden), nn.SiLU())
        self.current_visual = nn.Sequential(
            nn.LayerNorm(config.visual_feature_dim), nn.Linear(config.visual_feature_dim, hidden), nn.SiLU()
        )
        self.future_visual = nn.Sequential(
            nn.LayerNorm(config.visual_feature_dim), nn.Linear(config.visual_feature_dim, hidden), nn.SiLU()
        )
        self.state_effect = ConditionMLP(config)
        self.output = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * config.latent_dim),
        )

    def forward(
        self,
        current_context: Tensor,
        current_visual: Tensor,
        future_visual: Tensor,
        current_state: Tensor,
        effect_mean: Tensor,
    ) -> DiagonalGaussian:
        batch = int(current_state.shape[0])
        context = _pooled_feature(
            current_context, name="current_context", width=self.config.context_dim, batch=batch
        )
        current = _pooled_feature(
            current_visual, name="current_visual", width=self.config.visual_feature_dim, batch=batch
        )
        future = _pooled_feature(
            future_visual, name="future_visual", width=self.config.visual_feature_dim, batch=batch
        )
        devices = {value.device for value in (context, current, future, current_state, effect_mean)}
        if len(devices) != 1:
            raise ValueError("ActionVAE prior inputs must be on the same device")
        condition = self.state_effect(current_state, effect_mean)
        parameters = self.output(
            torch.cat(
                (
                    self.context(context),
                    self.current_visual(current),
                    self.future_visual(future),
                    condition,
                ),
                dim=-1,
            )
        )
        return split_gaussian_parameters(
            parameters,
            latent_dim=self.config.latent_dim,
            logvar_min=self.config.logvar_min,
            logvar_max=self.config.logvar_max,
        )


class ActionVariationalAutoencoder(ActionAutoencoder):
    """ActionVAE v2 with a trajectory posterior and future-aware conditional prior."""

    config: ActionVAEConfig

    def __init__(self, config: ActionVAEConfig) -> None:
        super().__init__(config)
        # Keep the deterministic encoder, mean projection, condition encoder, and
        # decoder state-dict contract intact so a reconstruction-trained ActionAE
        # can anchor the Gaussian phase without representation drift.
        self.posterior_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, config.posterior_logvar_init)
        self.prior = FutureAwareActionPrior(config)
        self._deterministic_backbone_frozen = False

    @property
    def deterministic_backbone_frozen(self) -> bool:
        return self._deterministic_backbone_frozen

    def freeze_deterministic_backbone(self) -> None:
        for module in (self.condition_mlp, self.encoder, self.decoder):
            module.requires_grad_(False)
            module.eval()
        self._deterministic_backbone_frozen = True

    def assert_phase2_trainable_contract(self, *, train_posterior_logvar: bool = True) -> None:
        frozen_prefixes = ("condition_mlp.", "encoder.", "decoder.")
        trainable_prefixes = (
            ("posterior_logvar.", "prior.") if train_posterior_logvar else ("prior.",)
        )
        unexpected_trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith(trainable_prefixes)
        ]
        unexpectedly_frozen = [
            name
            for name, parameter in self.named_parameters()
            if name.startswith(trainable_prefixes) and not parameter.requires_grad
        ]
        if not train_posterior_logvar:
            unexpectedly_trainable_posterior = [
                name
                for name, parameter in self.named_parameters()
                if name.startswith("posterior_logvar.") and parameter.requires_grad
            ]
            unexpected_trainable.extend(unexpectedly_trainable_posterior)
        backbone_trainable = [
            name
            for name, parameter in self.named_parameters()
            if name.startswith(frozen_prefixes) and parameter.requires_grad
        ]
        if unexpected_trainable or unexpectedly_frozen or backbone_trainable:
            raise AssertionError(
                "invalid frozen ActionVAE Phase-2 contract: "
                f"unexpected_trainable={unexpected_trainable[:8]}, "
                f"unexpectedly_frozen={unexpectedly_frozen[:8]}, "
                f"backbone_trainable={backbone_trainable[:8]}"
            )

    def train(self, mode: bool = True) -> "ActionVariationalAutoencoder":
        super().train(mode)
        if self._deterministic_backbone_frozen:
            for module in (self.condition_mlp, self.encoder, self.decoder):
                module.eval()
        return self

    def posterior(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor,
        *,
        time_mask: Tensor | None = None,
    ) -> DiagonalGaussian:
        self._validate_trajectory(trajectory)
        if current_state.device != trajectory.device:
            raise ValueError("current_state and trajectory must be on the same device")
        condition = self._condition(current_state, effect_mean)
        pooled = self.encoder.pooled_features(trajectory, condition, time_mask=time_mask)
        return DiagonalGaussian(
            mean=self.encoder.latent_projection(pooled),
            logvar=clamp_logvar(
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
        """Return the posterior mean for deterministic teacher-cache compatibility."""

        if effect_mean is None:
            raise ValueError("ActionVAE v2 requires a frozen effect_mean condition")
        return self.posterior(
            trajectory,
            current_state,
            effect_mean,
            time_mask=time_mask,
        ).mean.unsqueeze(1)

    def forward(
        self,
        trajectory: Tensor,
        current_state: Tensor,
        effect_mean: Tensor,
        *,
        current_context: Tensor,
        current_visual: Tensor,
        future_visual: Tensor,
        time_mask: Tensor | None = None,
        sample_posterior: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> ActionVAEOutput:
        posterior = self.posterior(
            trajectory,
            current_state,
            effect_mean,
            time_mask=time_mask,
        )
        prior = self.prior(
            current_context,
            current_visual,
            future_visual,
            current_state,
            effect_mean,
        )
        should_sample = self.training if sample_posterior is None else bool(sample_posterior)
        latent = posterior.rsample(generator=generator) if should_sample else posterior.mean
        z_action = latent.unsqueeze(1)
        reconstruction = self.decode(z_action, current_state, effect_mean)
        return ActionVAEOutput(
            reconstruction=reconstruction,
            z_action=z_action,
            posterior=posterior,
            prior=prior,
        )


def _validate_time_mask(
    time_mask: Tensor | None,
    *,
    batch: int,
    horizon: int,
    device: torch.device,
) -> Tensor | None:
    if time_mask is None:
        return None
    if not isinstance(time_mask, Tensor):
        raise TypeError(f"time_mask must be a torch.Tensor, got {type(time_mask).__name__}")
    if time_mask.shape != (batch, horizon):
        raise ValueError(f"time_mask must be [B,T]={batch,horizon}, got {tuple(time_mask.shape)}")
    return time_mask.to(device=device, dtype=torch.bool)
