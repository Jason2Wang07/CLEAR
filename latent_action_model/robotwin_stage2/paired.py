"""Typed paired-token modules shared by RoboTwin T2 and P2."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from torch import Tensor, nn

from latent_action_model.action_ae.distributions import DiagonalGaussian, split_gaussian_parameters
from latent_action_model.action_expert_oracle.adapters import (
    EffectTokenAdapter,
    PlanTokenAdapter,
)


VALID_ARMS = frozenset({"b0", "t2", "p2"})
ACTION_HEAD_ARCHITECTURES = frozenset({"transformer", "flatten_mlp_v1"})
RELEASE_RESIDUAL_RECIPE = "libero_release_residual_abc_v1"
PRETRAIN_RESIDUAL_RECIPE = "libero_pretrain_residual_abc_50k_v1"
ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE = "robotwin_pretrain_residual_abc_50k_v1"
ROBOTWIN_SFT25_RESIDUAL_RECIPE = "robotwin_sft25_residual_abc_50k_v1"
ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE = "robotwin_sft_release_residual_abc_50k_v1"
ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE = (
    "robotwin_sft_release_lawam_action_pe_70k_v2"
)
BRIDGE_B0_RESIDUAL_RECIPE = "bridge_b0_residual_wpe_25k_v1"
RESIDUAL_RECIPES = frozenset(
    {
        RELEASE_RESIDUAL_RECIPE,
        PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
        BRIDGE_B0_RESIDUAL_RECIPE,
    }
)


def normalize_arm(value: str | None) -> str:
    arm = "b0" if value is None else str(value).lower()
    if arm not in VALID_ARMS:
        raise ValueError(f"stage2_arm must be one of {sorted(VALID_ARMS)}, got {value!r}")
    return arm


def _module_parameter_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


class _LatentBranch(nn.Module):
    def __init__(self, latent_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, latent_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        dtype = _module_parameter_dtype(self.network, value.dtype)
        return self.network(value.to(dtype=dtype))


class PairedLaWMAdapter(nn.Module):
    """Map z-effect into one release-compatible LaWM query."""

    def __init__(self, *, latent_dim: int = 32, adapter_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.effect_branch = _LatentBranch(self.latent_dim, int(adapter_dim))
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_effect: Tensor) -> Tensor:
        expected = (self.latent_dim,)
        if z_effect.ndim != 2 or tuple(z_effect.shape[1:]) != expected:
            raise ValueError(f"z_effect must be [B,{self.latent_dim}]")
        return self.output_norm(self.effect_branch(z_effect)).unsqueeze(1)


class PairedExpertAdapters(nn.Module):
    """Route z-action to self-attention and z-effect to cross-attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        vision_dim: int,
        latent_dim: int = 32,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        self.plan = PlanTokenAdapter(
            int(hidden_dim), learned_dim=int(latent_dim), adapter_dim=int(adapter_dim)
        )
        self.plan.physical.requires_grad_(False)
        self.effect = EffectTokenAdapter(
            int(vision_dim), effect_dim=int(latent_dim), adapter_dim=int(adapter_dim)
        )

    def forward(
        self,
        z_effect: Tensor,
        z_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if z_effect.ndim != 2 or z_effect.shape[-1] != 32:
            raise ValueError("z_effect must be [B,32]")
        if z_action.shape != z_effect.shape:
            raise ValueError("z_action must match z_effect [B,32]")
        plan, plan_mask = self.plan(z_action.unsqueeze(1), representation="learned")
        effect, effect_mask = self.effect(z_effect, detach_input=False)
        return plan, plan_mask, effect, effect_mask


class ActionResidualProjection(nn.Module):
    """Project z-action into an additive Expert hidden-state residual.

    The final projection is zero-initialized so enabling this branch preserves
    the pretrained Expert exactly until optimization updates the branch.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        adapter_dim: int = 256,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(adapter_dim)),
            nn.SiLU(),
            nn.Linear(int(adapter_dim), self.hidden_dim),
        )
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, z_action: Tensor) -> Tensor:
        if z_action.ndim != 2 or tuple(z_action.shape[1:]) != (self.latent_dim,):
            raise ValueError(f"z_action must be [B,{self.latent_dim}]")
        dtype = _module_parameter_dtype(self.network, z_action.dtype)
        return self.network(z_action.to(dtype=dtype))


def _predictor_head(
    query_dim: int,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(query_dim),
        nn.Flatten(start_dim=1),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, latent_dim),
    )


class _DenseTransitionActionHead(nn.Module):
    """Decode a Gaussian action prior from dense, same-position visual change."""

    def __init__(
        self,
        *,
        query_dim: int,
        model_dim: int,
        visual_dim: int,
        state_dim: int,
        latent_dim: int,
        num_embodiments: int,
        horizon_condition_dim: int,
        visual_grid_size: tuple[int, int],
        num_layers: int,
        num_heads: int,
        dropout: float,
        logvar_min: float | Tensor,
        logvar_max: float | Tensor,
        architecture: str = "transformer",
        query_count: int = 8,
    ) -> None:
        super().__init__()
        if architecture not in ACTION_HEAD_ARCHITECTURES:
            raise ValueError(f"Unsupported action head architecture: {architecture!r}")
        self.architecture = architecture
        self.query_count = int(query_count)
        grid_height, grid_width = visual_grid_size
        self.query_dim = int(query_dim)
        self.model_dim = int(model_dim)
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.num_embodiments = int(num_embodiments)
        self.horizon_condition_dim = int(horizon_condition_dim)
        self.visual_grid_size = (int(grid_height), int(grid_width))
        self.num_visual_tokens = int(grid_height * grid_width)
        lower = torch.as_tensor(logvar_min, dtype=torch.float32).reshape(-1)
        upper = torch.as_tensor(logvar_max, dtype=torch.float32).reshape(-1)
        if lower.numel() == 1:
            lower = lower.expand(self.latent_dim).clone()
        if upper.numel() == 1:
            upper = upper.expand(self.latent_dim).clone()
        if lower.shape != (self.latent_dim,) or upper.shape != lower.shape:
            raise ValueError("dense action logvar bounds must be scalar or [latent_dim]")
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("dense action logvar bounds must be finite")
        if not bool((lower < upper).all()):
            raise ValueError("each dense action logvar lower bound must be below its upper bound")
        self.register_buffer("logvar_min", lower)
        self.register_buffer("logvar_max", upper)

        self.current_norm = nn.LayerNorm(self.visual_dim)
        self.future_norm = nn.LayerNorm(self.visual_dim)
        self.delta_norm = nn.LayerNorm(self.visual_dim)
        self.transition_mlp = nn.Sequential(
            nn.Linear(3 * self.visual_dim, 2 * self.model_dim),
            nn.GELU(),
            nn.Linear(2 * self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.row_position = nn.Parameter(torch.empty(grid_height, self.model_dim))
        self.column_position = nn.Parameter(torch.empty(grid_width, self.model_dim))

        self.plan_norm = nn.LayerNorm(self.query_dim)
        self.plan_projection = nn.Linear(self.query_dim, self.model_dim)
        self.state_condition = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.effect_condition = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.embodiment_condition = nn.Embedding(self.num_embodiments, self.model_dim)
        self.horizon_condition = nn.Sequential(
            nn.Linear(self.horizon_condition_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.query_condition_norm = nn.LayerNorm(self.model_dim)

        if self.architecture == "transformer":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.model_dim,
                nhead=int(num_heads),
                dim_feedforward=4 * self.model_dim,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=int(num_layers),
                norm=nn.LayerNorm(self.model_dim),
            )
            self.pool_query = nn.Parameter(torch.empty(1, 1, self.model_dim))
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=int(num_heads),
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            # Fixed-order flattening retains patch correspondence without attention.
            self.visual_bottleneck = nn.Linear(self.model_dim, self.latent_dim)
            fusion_dim = self.num_visual_tokens * self.latent_dim + self.query_count * self.model_dim
            self.fusion_mlp = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, self.model_dim),
                nn.GELU(),
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
            )
        self.pool_norm = nn.LayerNorm(self.model_dim)
        self.action_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 2 * self.latent_dim),
        )

        nn.init.normal_(self.row_position, mean=0.0, std=0.02)
        nn.init.normal_(self.column_position, mean=0.0, std=0.02)
        if self.architecture == "transformer":
            nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

    def _transition_tokens(self, current_visual: Tensor, future_visual: Tensor) -> Tensor:
        delta = future_visual - current_visual
        tokens = self.transition_mlp(
            torch.cat(
                (
                    self.current_norm(current_visual),
                    self.future_norm(future_visual),
                    self.delta_norm(delta),
                ),
                dim=-1,
            )
        )
        position = (
            self.row_position[:, None, :] + self.column_position[None, :, :]
        ).reshape(1, self.num_visual_tokens, self.model_dim)
        return tokens + position

    def forward(
        self,
        h_plan: Tensor,
        *,
        current_visual: Tensor,
        future_visual: Tensor,
        state: Tensor,
        effect: Tensor,
        embodiment_id: Tensor,
        horizon_condition: Tensor,
    ) -> DiagonalGaussian:
        batch = int(h_plan.shape[0])
        expected_visual = (batch, self.num_visual_tokens, self.visual_dim)
        if current_visual.shape != expected_visual or future_visual.shape != expected_visual:
            raise ValueError(
                "current_visual and future_visual must preserve the dense visual grid with "
                f"shape [B,{self.num_visual_tokens},{self.visual_dim}]"
            )
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if state.shape != (batch, self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        if effect.shape != (batch, self.latent_dim):
            raise ValueError(f"effect must be [B,{self.latent_dim}]")
        if embodiment_id.shape != (batch,) or embodiment_id.dtype not in {torch.int32, torch.int64}:
            raise ValueError("embodiment_id must be an int32/int64 tensor with shape [B]")
        if bool(((embodiment_id < 0) | (embodiment_id >= self.num_embodiments)).any()):
            raise ValueError(f"embodiment_id must be in [0,{self.num_embodiments})")
        if horizon_condition.shape != (batch, self.horizon_condition_dim):
            raise ValueError(
                f"horizon_condition must be [B,{self.horizon_condition_dim}]"
            )
        if not horizon_condition.is_floating_point() or not bool(
            torch.isfinite(horizon_condition).all()
        ):
            raise ValueError("horizon_condition must contain finite floating-point values")

        parameter_dtype = self.plan_projection.weight.dtype
        h_plan = h_plan.to(dtype=parameter_dtype)
        current_visual = current_visual.to(dtype=parameter_dtype)
        future_visual = future_visual.to(dtype=parameter_dtype)
        state = state.to(dtype=parameter_dtype)
        effect = effect.to(dtype=parameter_dtype)
        horizon_condition = horizon_condition.to(dtype=parameter_dtype)

        memory = self._transition_tokens(current_visual, future_visual)
        condition = (
            self.state_condition(state)
            + self.effect_condition(effect)
            + self.embodiment_condition(embodiment_id)
            + self.horizon_condition(horizon_condition)
        )
        queries = self.query_condition_norm(
            self.plan_projection(self.plan_norm(h_plan)) + condition.unsqueeze(1)
        )
        if self.architecture == "transformer":
            decoded = self.decoder(tgt=queries, memory=memory)
            pool_query = self.pool_query.expand(batch, -1, -1)
            pooled, _ = self.attention_pool(pool_query, decoded, decoded, need_weights=False)
            pooled = pooled[:, 0]
        else:
            if h_plan.shape[1] != self.query_count:
                raise ValueError(f"MLP head requires {self.query_count} plan queries")
            pooled = self.fusion_mlp(torch.cat(
                (queries.flatten(1), self.visual_bottleneck(memory).flatten(1)), dim=-1
            ))
        parameters = self.action_head(self.pool_norm(pooled))
        return split_gaussian_parameters(
            parameters,
            latent_dim=self.latent_dim,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )


class PairedLatentPredictor(nn.Module):
    """Optional Effect MLP plus a versioned dense-transition action head."""

    def __init__(
        self,
        *,
        query_count: int = 8,
        query_dim: int = 2048,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        visual_dim: int = 768,
        state_dim: int = 32,
        action_distribution: str = "deterministic_v1",
        logvar_min: float | Tensor = -6.0,
        logvar_max: float | Tensor = 2.0,
        num_embodiments: int = 32,
        horizon_condition_dim: int = 1,
        visual_grid_size: tuple[int, int] = (16, 16),
        action_decoder_layers: int = 3,
        action_decoder_heads: int = 8,
        action_decoder_dropout: float = 0.0,
        enable_effect_head: bool = True,
        action_head_architecture: str = "transformer",
    ) -> None:
        super().__init__()
        if action_head_architecture not in ACTION_HEAD_ARCHITECTURES:
            raise ValueError(f"Unsupported action head architecture: {action_head_architecture!r}")
        if action_head_architecture != "transformer" and action_distribution != "dense_transition_gaussian_v3":
            raise ValueError("MLP architecture requires the dense Gaussian action contract")
        self.action_head_architecture = action_head_architecture
        if min(
            query_count,
            query_dim,
            hidden_dim,
            latent_dim,
            visual_dim,
            state_dim,
            num_embodiments,
            horizon_condition_dim,
            action_decoder_layers,
            action_decoder_heads,
        ) < 1:
            raise ValueError("predictor dimensions must be positive")
        if len(visual_grid_size) != 2 or min(int(value) for value in visual_grid_size) < 1:
            raise ValueError("visual_grid_size must contain two positive dimensions")
        if int(hidden_dim) % int(action_decoder_heads) != 0:
            raise ValueError("hidden_dim must be divisible by action_decoder_heads")
        if not 0.0 <= float(action_decoder_dropout) < 1.0:
            raise ValueError("action_decoder_dropout must be in [0,1)")
        lower = torch.as_tensor(logvar_min, dtype=torch.float32).reshape(-1)
        upper = torch.as_tensor(logvar_max, dtype=torch.float32).reshape(-1)
        if lower.numel() == 1:
            lower = lower.expand(int(latent_dim)).clone()
        if upper.numel() == 1:
            upper = upper.expand(int(latent_dim)).clone()
        if lower.shape != (int(latent_dim),) or upper.shape != lower.shape:
            raise ValueError("logvar bounds must be scalar or [latent_dim]")
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("logvar bounds must be finite")
        if not bool((lower < upper).all()):
            raise ValueError("each logvar_min must be smaller than logvar_max")
        self.query_count = int(query_count)
        self.query_dim = int(query_dim)
        self.latent_dim = int(latent_dim)
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.action_distribution = str(action_distribution)
        self.enable_effect_head = bool(enable_effect_head)
        self.register_buffer("logvar_min", lower)
        self.register_buffer("logvar_max", upper)
        if self.action_distribution == "gaussian_v2":
            raise ValueError(
                "gaussian_v2 identifies the retired mean-pooled action head; migrate explicitly "
                "to dense_transition_gaussian_v3 and use a matching checkpoint"
            )
        if self.action_distribution not in {"deterministic_v1", "dense_transition_gaussian_v3"}:
            raise ValueError(
                "action_distribution must be deterministic_v1 or dense_transition_gaussian_v3"
            )
        input_dim = self.query_count * self.query_dim
        self.effect_head = (
            _predictor_head(
                self.query_dim, input_dim, int(hidden_dim), self.latent_dim
            )
            if self.enable_effect_head
            else None
        )
        if self.action_distribution == "deterministic_v1":
            self.action_head = _predictor_head(
                self.query_dim, input_dim, int(hidden_dim), self.latent_dim
            )
            self.action_context_norm = None
        else:
            self.action_context_norm = None
            self.action_head = _DenseTransitionActionHead(
                query_dim=self.query_dim,
                model_dim=int(hidden_dim),
                visual_dim=self.visual_dim,
                state_dim=self.state_dim,
                latent_dim=self.latent_dim,
                num_embodiments=int(num_embodiments),
                horizon_condition_dim=int(horizon_condition_dim),
                visual_grid_size=(int(visual_grid_size[0]), int(visual_grid_size[1])),
                num_layers=int(action_decoder_layers),
                num_heads=int(action_decoder_heads),
                dropout=float(action_decoder_dropout),
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
                architecture=self.action_head_architecture,
                query_count=self.query_count,
            )
        if self.action_head_architecture != "transformer":
            logging.getLogger(__name__).info(
                "[ActionHead] architecture=%s parameters=%d future_input=predicted",
                self.action_head_architecture,
                sum(parameter.numel() for parameter in self.action_head.parameters()),
            )

    def _validate_context(self, h_act: Tensor) -> None:
        expected = (self.query_count, self.query_dim)
        if h_act.ndim != 3 or tuple(h_act.shape[1:]) != expected:
            raise ValueError(f"h_act must be [B,{expected[0]},{expected[1]}]")

    def predict_effect(self, h_act: Tensor) -> Tensor:
        if self.effect_head is None:
            raise RuntimeError("this action-only predictor has no Effect head")
        self._validate_context(h_act)
        dtype = _module_parameter_dtype(self.effect_head, h_act.dtype)
        return self.effect_head(h_act.to(dtype=dtype))

    def predict_action(
        self,
        h_act: Tensor,
        *,
        current_visual: Tensor,
        future_visual: Tensor,
        state: Tensor,
        effect: Tensor,
        embodiment_id: Tensor | None = None,
        horizon_condition: Tensor | None = None,
    ) -> DiagonalGaussian:
        self._validate_context(h_act)
        if self.action_distribution == "deterministic_v1":
            dtype = _module_parameter_dtype(self.action_head, h_act.dtype)
            mean = self.action_head(h_act.to(dtype=dtype))
            return DiagonalGaussian(mean=mean, logvar=torch.zeros_like(mean))
        if embodiment_id is None or horizon_condition is None:
            raise ValueError(
                "dense_transition_gaussian_v3 requires embodiment_id and horizon_condition"
            )
        assert isinstance(self.action_head, _DenseTransitionActionHead)
        return self.action_head(
            h_act,
            current_visual=current_visual,
            future_visual=future_visual,
            state=state,
            effect=effect,
            embodiment_id=embodiment_id,
            horizon_condition=horizon_condition,
        )

    def forward(self, h_act: Tensor) -> tuple[Tensor, Tensor]:
        """Legacy deterministic interface retained for v1 checkpoints and tests."""

        if self.action_distribution != "deterministic_v1":
            raise RuntimeError(
                "dense_transition_gaussian_v3 action prediction requires transition inputs"
            )
        if self.effect_head is None:
            raise RuntimeError("action-only predictor requires the dense future-aware route")
        effect = self.predict_effect(h_act)
        dtype = _module_parameter_dtype(self.action_head, h_act.dtype)
        action = self.action_head(h_act.to(dtype=dtype))
        return effect, action


@dataclass(frozen=True)
class PairedTokens:
    effect: Tensor
    action: Tensor
    action_logvar: Tensor | None = None

    def detached(self) -> "PairedTokens":
        return PairedTokens(
            effect=self.effect.detach(),
            action=self.action.detach(),
            action_logvar=(self.action_logvar.detach() if self.action_logvar is not None else None),
        )
