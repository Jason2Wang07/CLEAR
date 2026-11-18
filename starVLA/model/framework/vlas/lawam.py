from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.bridge_stage2.runtime_contract import (
    load_bridge_stage2_latent_contract,
    load_frozen_bridge_action_decoder,
)
from latent_action_model.bridge_stage2.temporal import (
    is_bridge_stage2_temporal_contract,
)
from latent_action_model.action_ae.distributions import DiagonalGaussian, diagonal_gaussian_kl
from latent_action_model.action_ae.losses import trajectory_reconstruction_loss
from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.libero_stage2.runtime_contract import (
    FrozenActionVAEDecoder,
    Stage2LatentAffine,
    is_libero_04s_adapter_temporal_contract,
    load_frozen_libero_action_decoder,
    load_libero_stage2_latent_contract,
    resolve_libero_stage2_temporal_contract,
)
from latent_action_model.libero_stage2.teacher_bundle import (
    LIBERO_12S_EFFECT_CHECKPOINT,
    LIBERO_12S_EFFECT_CONFIG,
)
from latent_action_model.robotwin_stage2.paired import (
    ActionResidualProjection,
    PairedExpertAdapters,
    PairedLaWMAdapter,
    PairedLatentPredictor,
    PairedTokens,
    RESIDUAL_RECIPES,
    ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
    normalize_arm,
)
from latent_action_model.robotwin_action_ae.losses import (
    dual_arm_trajectory_reconstruction_loss,
)
from latent_action_model.robotwin_stage2.runtime_contract import (
    load_frozen_robotwin_action_decoder,
    load_robotwin_stage2_latent_contract,
)
from latent_action_model.robotwin_stage2.temporal import (
    is_robotwin_stage2_temporal_contract,
    resolve_stage2_temporal_contract,
)
from starVLA.model.framework.latent_world.processor_utils import (
    build_latent_world_processor_spec,
    configure_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch
from .vlm_auto import (
    load_vlm_auto,
    remove_lm_head,
)
from starVLA.model.tools import sync_managed_modules_training_mode

from .flowmatching_expert import ConditionalFlowMatchingConfig, ConditionalFlowMatchingHead


# ============================================================================
# Config & Constants
# ============================================================================

DEFAULT_LAM_ROOT = Path("latent_action_model")


@dataclass
class LatentWorldPolicyConfig:
    """Independent LatentWorldPolicyBackend config."""

    # Flow head config
    flow_cfg: ConditionalFlowMatchingConfig = field(default_factory=ConditionalFlowMatchingConfig)

    # Action chunk (Qwen-compatible schema)
    future_action_window_size: int = 7
    past_action_window_size: int = 0
    action_horizon: int = 8

    # Base checkpoints
    hf_cache_dir: Optional[Union[str, Path]] = None
    lam_ckpt_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/checkpoints/epoch=39.ckpt")
    lam_yaml_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/dino_base_ae.yaml")
    lam_vision_model_id_override: Optional[str] = None

    # VLM dtype
    vlm_dtype: torch.dtype = torch.bfloat16
    remove_lm_head: bool = True

    # Placeholder token
    latent_action_placeholder_token: str = "<ACT_PH>"

    # World / distill losses
    perceptual_weight: float = 0.1
    enable_loss_distill: bool = True
    latent_loss_type: str = "mse"  # "mse" | "cosine"
    lam_encoder_distill_weight: float = 1.0
    # Use the manifest-bound raw Effect teacher instead of re-encoding the two
    # policy frames. Required by Bridge, whose DINO and Effect endpoints differ.
    stage2_cached_effect_distill: bool = False

    # Flow conditioning variants
    future_prediction: bool = False
    repeated_diffusion_steps: int = 4
    flow_only_mode: bool = False
    enable_flow_h_t1_scheduled_sampling: bool = False
    flow_h_t1_pred_prob_start: float = 0.0
    flow_h_t1_pred_prob_end: float = 1.0
    flow_h_t1_pred_ramp_steps: int = 20000
    detach_future_feature: bool = False

    # Independent additions
    num_action_queries: int = 8
    flow_action_num_queries: int = 8

    # RoboTwin paired-token Stage-2. ``None`` and ``b0`` preserve release behavior.
    stage2_arm: Optional[str] = None
    stage2_latent_dim: int = 32
    stage2_adapter_dim: int = 256
    stage2_predictor_hidden_dim: int = 256
    stage2_effect_distill_weight: float = 0.1
    stage2_action_distill_weight: float = 0.1
    stage2_lawam_distill_weight: float = 0.0
    stage2_condition_source: Optional[str] = None
    stage2_action_distribution: str = "deterministic_v1"
    stage2_action_logvar_min: float = -6.0
    stage2_action_logvar_max: float = 2.0
    stage2_action_inference: str = "mean"
    stage2_action_state_source: Optional[str] = None
    stage2_phase: Optional[str] = None
    stage2_temporal_contract: Optional[str] = None
    stage2_query_scheme: str = "act_effect_flow_plan_v1"
    stage2_latent_cache_root: Optional[str] = None
    stage2_latent_normalization_path: Optional[str] = None
    stage2_action_decoder_checkpoint_path: Optional[str] = None
    stage2_action_decoder_checkpoint_sha256: Optional[str] = None
    stage2_action_decoder_config_path: Optional[str] = None
    stage2_action_decoder_config_sha256: Optional[str] = None
    stage2_action_decoder_data_root: Optional[str] = None
    stage2_action_decoder_sidecar_binding: str = "exact"
    stage2_action_teacher_transfer: Optional[str] = None
    stage2_action_decoder_sidecar_reference_root: Optional[str] = None
    stage2_effect_teacher_checkpoint_path: Optional[str] = None
    stage2_effect_teacher_checkpoint_sha256: Optional[str] = None
    stage2_effect_teacher_config_path: Optional[str] = None
    stage2_effect_teacher_config_sha256: Optional[str] = None
    stage2_lawam_condition_checkpoint_path: Optional[str] = None
    stage2_lawam_condition_checkpoint_sha256: Optional[str] = None
    stage2_lawam_condition_config_path: Optional[str] = None
    stage2_lawam_condition_config_sha256: Optional[str] = None
    stage2_horizon_seconds: float = 1.2
    stage2_horizon_condition_dim: int = 3
    stage2_action_decoder_layers: int = 3
    stage2_action_head_architecture: str = "transformer"
    stage2_action_injection_enabled: bool = True
    stage2_action_decoder_heads: int = 8
    stage2_action_decoder_dropout: float = 0.0
    stage2_action_visual_grid_size: Tuple[int, int] = (16, 16)
    stage2_flow_weight: float = 0.0
    stage2_dino_weight: float = 0.0
    stage2_action_kl_weight: float = 1.0
    stage2_action_mean_weight: float = 0.5
    stage2_action_trajectory_weight: float = 0.2
    stage2_training_recipe: Optional[str] = None
    stage2_residual_warmup_start_step: int = 0
    stage2_residual_teacher_prob_start: float = 0.0
    stage2_residual_teacher_prob_end: float = 0.0
    stage2_residual_teacher_ramp_steps: int = 1
    stage2_residual_predictor_grad_start_step: int = 0
    stage2_residual_condition_dropout: float = 0.0
    # Deployment reconstructs the graph from lightweight assets, then the
    # strict full-checkpoint loader replaces every parameter and buffer.
    stage2_deployment_bootstrap: bool = False


# ============================================================================
# Lightweight Blocks
# ============================================================================


class VLMToLAMQFormer(nn.Module):
    """Refine VLM query hidden states into one latent action embedding."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, int(lam_code_dim)) * 0.02)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=int(lam_code_dim),
                    kdim=int(vlm_hidden_dim),
                    vdim=int(vlm_hidden_dim),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_qs = nn.ModuleList([nn.LayerNorm(int(lam_code_dim)) for _ in range(int(num_layers))])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(int(vlm_hidden_dim)) for _ in range(int(num_layers))])
        hidden_dim = int(int(lam_code_dim) * float(ffn_expansion_factor))
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(lam_code_dim)),
                    nn.Linear(int(lam_code_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, int(lam_code_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bsz = int(context.shape[0])
        queries = self.query.expand(bsz, -1, -1).to(device=context.device, dtype=context.dtype)
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        return self.final_norm(queries)


# ============================================================================
# VLM / LAM Load Helpers
# ============================================================================


def _load_vlm_and_processor(
    cfg: LatentWorldPolicyConfig,
    vlm_model_id: str,
) -> Tuple[nn.Module, Any, Any, int]:
    processor_spec = build_latent_world_processor_spec(policy_cfg=cfg, vlm_model_id=vlm_model_id)
    vlm, processor = load_vlm_auto(processor_spec.model_id, processor_spec.cache_dir, dtype=cfg.vlm_dtype)
    processor, tokenizer, placeholder_token_id = configure_latent_world_processor(
        processor,
        placeholder_token=processor_spec.placeholder_token,
    )

    # Ensure tokenizer size and embedding rows stay aligned after adding placeholder tokens.
    target_vocab_size = max(int(len(tokenizer)), int(placeholder_token_id + 1))
    embed = vlm.get_input_embeddings()
    if embed is None:
        raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
    embed_rows = int(embed.weight.shape[0])
    if target_vocab_size > embed_rows:
        if not hasattr(vlm, "resize_token_embeddings"):
            raise RuntimeError(
                "[LatentWorldPolicyBackend] tokenizer vocab exceeds embedding size, "
                "but model does not support `resize_token_embeddings`."
            )
        vlm.resize_token_embeddings(target_vocab_size)

    if hasattr(vlm, "generation_config") and vlm.generation_config is not None:
        vlm.generation_config.max_new_tokens = 4
    if hasattr(vlm, "config") and vlm.config is not None:
        try:
            vlm.config.loss_type = "ForCausalLMLoss"
            vlm.config.use_cache = False
        except Exception:
            pass

    if bool(getattr(cfg, "remove_lm_head", True)):
        remove_lm_head(vlm)

    return vlm, processor, tokenizer, placeholder_token_id


def _apply_flow_only_grad_to_h_vlm(
    *,
    h_vlm: torch.Tensor,
    act_placeholder_mask: torch.Tensor,
    enable_flow_only: bool,
) -> torch.Tensor:
    """
    Keep full VLM context values for flow, but split direct-flow gradients by sequence boundary:
    only tokens after the last act placeholder keep direct flow gradients.
    """
    if not bool(enable_flow_only):
        return h_vlm
    if h_vlm.dim() != 3:
        raise ValueError(f"[LatentWorldPolicyBackend] expected h_vlm [B, L, D], got {tuple(h_vlm.shape)}")
    if act_placeholder_mask is None:
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires act_placeholder_mask.")
    if act_placeholder_mask.dim() != 2:
        raise ValueError(
            f"[LatentWorldPolicyBackend] expected act_placeholder_mask [B, L], got {tuple(act_placeholder_mask.shape)}"
        )
    if act_placeholder_mask.shape[0] != h_vlm.shape[0] or act_placeholder_mask.shape[1] != h_vlm.shape[1]:
        raise ValueError(
            "[LatentWorldPolicyBackend] act_placeholder_mask shape mismatch with h_vlm: "
            f"mask={tuple(act_placeholder_mask.shape)} vs h_vlm={tuple(h_vlm.shape)}"
        )
    act_mask = act_placeholder_mask.to(device=h_vlm.device, dtype=torch.bool)
    if not torch.all(act_mask.any(dim=1)):
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires at least one act placeholder per sample.")

    seq_positions = torch.arange(h_vlm.shape[1], device=h_vlm.device).unsqueeze(0)
    last_act_idx = torch.where(act_mask, seq_positions, -1).max(dim=1).values
    post_act_mask = seq_positions > last_act_idx.unsqueeze(1)
    post_act_mask_f = post_act_mask.unsqueeze(-1).to(dtype=h_vlm.dtype)
    return h_vlm * post_act_mask_f + h_vlm.detach() * (1.0 - post_act_mask_f)


def _module_param_dtype(module: Optional[nn.Module], default: torch.dtype) -> torch.dtype:
    if module is None:
        return default
    try:
        p = next(module.parameters())
        return p.dtype
    except Exception:
        return default


def _cuda_autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@dataclass
class PolicyEncodingState:
    h_vlm: torch.Tensor
    h_effect: torch.Tensor
    h_plan: torch.Tensor
    pred_action_emb: torch.Tensor
    h_t: torch.Tensor
    h_t1_pred: torch.Tensor
    h_t1_gt: torch.Tensor
    h_t_original: torch.Tensor
    predicted_effect: Optional[torch.Tensor] = None
    paired_tokens: Optional[PairedTokens] = None


# ============================================================================
# LatentWorldPolicyBackend
# ============================================================================


class LatentWorldPolicyBackend(nn.Module):
    """Independent world-model VLA without LatentVLAModel dependency."""

    def __init__(self, model_cfg: LatentWorldPolicyConfig, *, vlm_model_id: str) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        if not isinstance(getattr(model_cfg, "stage2_action_injection_enabled", True), bool):
            raise ValueError("stage2_action_injection_enabled must be a boolean")
        self.vlm_model_id = str(vlm_model_id)

        # 1) Load VLM + processor/tokenizer and register placeholder token.
        self.vlm, self.processor, self.tokenizer, self.placeholder_token_id = _load_vlm_and_processor(
            self.model_cfg,
            self.vlm_model_id,
        )

        # 2) Load LAM.
        deployment_bootstrap = bool(self.model_cfg.stage2_deployment_bootstrap)
        self.lam = load_latent_action_model(
            self.model_cfg.lam_ckpt_path,
            self.model_cfg.lam_yaml_path,
            vision_model_id_override=self.model_cfg.lam_vision_model_id_override,
            load_checkpoint=not deployment_bootstrap,
        )

        # 3) Create trainable query and mapping head.
        self.num_action_queries = int(self.model_cfg.num_action_queries)

        vlm_cfg = getattr(self.vlm, "config", None)
        text_cfg = getattr(vlm_cfg, "text_config", None)
        # NOTE: don't nest `getattr` in default value; default expression is evaluated eagerly.
        vlm_hidden_size = getattr(text_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            vlm_hidden_size = getattr(vlm_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            raise AttributeError("[LatentWorldPolicyBackend] cannot resolve VLM hidden_size from config.")
        vlm_hidden_dim = int(vlm_hidden_size)
        lam_code_dim = int(self.lam.code_dim)
        self.act_query = nn.Parameter(torch.randn(self.num_action_queries, vlm_hidden_dim) * 0.02)
        self.flow_action_num_queries = int(self.model_cfg.flow_action_num_queries)
        self.flow_action_query = nn.Parameter(torch.randn(self.flow_action_num_queries, vlm_hidden_dim) * 0.02)
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden_dim,
            lam_code_dim=lam_code_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=4.0,
            dropout=0.0,
        )

        # 4) Align flow config to LAM output dimensions.
        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_h = int(getattr(self.lam.encoder, "grid_height", 0) or 0)
        lam_grid_w = int(getattr(self.lam.encoder, "grid_width", 0) or 0)
        lam_num_tokens = int(lam_grid_h * lam_grid_w) if lam_grid_h > 0 and lam_grid_w > 0 else int(
            self.model_cfg.flow_cfg.num_vision_tokens
        )
        if int(self.model_cfg.flow_cfg.vision_dim) != lam_vision_dim:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.vision_dim={self.model_cfg.flow_cfg.vision_dim} "
                f"!= LAM vision dim={lam_vision_dim}; auto-aligned."
            )
        self.model_cfg.flow_cfg.vision_dim = lam_vision_dim

        if int(self.model_cfg.flow_cfg.num_vision_tokens) != lam_num_tokens:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.num_vision_tokens={self.model_cfg.flow_cfg.num_vision_tokens} "
                f"!= LAM token count={lam_num_tokens}; auto-aligned."
            )
            self.model_cfg.flow_cfg.num_vision_tokens = lam_num_tokens

        # 5) Flow head.
        self.flow = ConditionalFlowMatchingHead(config=self.model_cfg.flow_cfg)
        self.flow.action_horizon = int(self.model_cfg.action_horizon)
        self.stage2_arm = normalize_arm(self.model_cfg.stage2_arm)
        raw_stage2_phase = self.model_cfg.stage2_phase
        self.stage2_phase = (
            None if raw_stage2_phase is None or not str(raw_stage2_phase).strip()
            else str(raw_stage2_phase).strip().upper()
        )
        if self.stage2_phase is not None and self.stage2_phase not in {"W", "P", "E"}:
            raise ValueError("stage2_phase must be W, P, or E")
        self._phased_stage2 = self.stage2_phase is not None
        self.stage2_training_recipe = str(
            self.model_cfg.stage2_training_recipe or ""
        ).strip()
        if self.stage2_training_recipe not in {"", *RESIDUAL_RECIPES}:
            raise ValueError(
                f"unsupported stage2_training_recipe={self.stage2_training_recipe!r}"
            )
        self._release_residual_recipe = self.stage2_training_recipe in RESIDUAL_RECIPES
        self._lawam_action_recipe = (
            self.stage2_training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
        )
        self.stage2_temporal_contract = resolve_stage2_temporal_contract(
            self.model_cfg.stage2_temporal_contract
        )
        self._robotwin_phased_stage2 = (
            self._phased_stage2
            and is_robotwin_stage2_temporal_contract(self.stage2_temporal_contract)
        )
        self._bridge_phased_stage2 = (
            self._phased_stage2
            and is_bridge_stage2_temporal_contract(self.stage2_temporal_contract)
        )
        self._stage2_token_override: Optional[PairedTokens] = None
        self.paired_lawm_adapter: Optional[PairedLaWMAdapter] = None
        self.paired_expert_adapters: Optional[PairedExpertAdapters] = None
        self.paired_latent_predictor: Optional[PairedLatentPredictor] = None
        self.stage2_action_residual: Optional[ActionResidualProjection] = None
        if not self.action_injection_enabled() and not (
            self._phased_stage2 and self._release_residual_recipe and self.stage2_arm == "p2"
        ):
            raise ValueError("Action auxiliary-only mode requires phased P2 release-residual training")
        if not self.action_injection_enabled():
            print("[Stage2] action_injection=disabled action_branch=auxiliary_only")
        self.stage2_latent_affine: Optional[Stage2LatentAffine] = None
        self.frozen_action_decoder: Optional[nn.Module] = None
        action_logvar_min = float(self.model_cfg.stage2_action_logvar_min)
        action_logvar_max = float(self.model_cfg.stage2_action_logvar_max)
        if self._phased_stage2:
            if self.stage2_arm != "p2":
                raise ValueError("phased W/P/E training requires stage2_arm=p2")
            expected_query_scheme = (
                "act_lawam_flow_plan_v1"
                if self._lawam_action_recipe
                else "act_effect_flow_plan_v1"
            )
            if self.model_cfg.stage2_query_scheme != expected_query_scheme:
                raise ValueError(
                    "phased Stage-2 query scheme mismatch: "
                    f"expected {expected_query_scheme!r}"
                )
            if self.num_action_queries != 8 or self.flow_action_num_queries != 8:
                raise ValueError("phased W/P/E requires 8 effect queries and 8 plan queries")
            if self.num_action_queries != self.flow_action_num_queries:
                raise ValueError("effect and plan query counts must match for the paired predictor")
            if self.model_cfg.stage2_action_distribution != "dense_transition_gaussian_v3":
                raise ValueError(
                    "phased W/P/E requires dense_transition_gaussian_v3"
                )
            if self.model_cfg.stage2_action_state_source not in {
                "actionvae14_padded32",
                "robotwin_actionvae14_padded32",
                "starvla32",
            }:
                raise ValueError(
                    "phased W/P/E requires an explicit stage2_action_state_source: "
                    "actionvae14_padded32, robotwin_actionvae14_padded32, or starvla32"
                )
            if not bool(self.model_cfg.future_prediction):
                raise ValueError("phased W/P/E requires future_prediction=true")
            if not bool(self.model_cfg.detach_future_feature):
                raise ValueError("phased W/P/E requires detach_future_feature=true")
            if int(self.model_cfg.stage2_horizon_condition_dim) != 3:
                raise ValueError("phased W/P/E horizon condition is [hz, seconds, valid_count]")
            if (
                float(self.model_cfg.stage2_horizon_seconds)
                != self.stage2_temporal_contract.duration_seconds
            ):
                raise ValueError(
                    "phased W/P/E stage2_horizon_seconds does not match "
                    f"{self.stage2_temporal_contract.identifier}"
                )
            if (
                float(self.model_cfg.flow_cfg.horizon_sec)
                != self.stage2_temporal_contract.duration_seconds
            ):
                raise ValueError(
                    "phased W/P/E flow horizon does not match "
                    f"{self.stage2_temporal_contract.identifier}"
                )
            configured_grid = tuple(
                int(value) for value in self.model_cfg.stage2_action_visual_grid_size
            )
            if configured_grid != (16, 16) or (lam_grid_h, lam_grid_w) != configured_grid:
                raise ValueError(
                    "phased action predictor requires a dense 16x16 LaWM/DINO grid"
                )
            condition_checkpoint = (
                self.model_cfg.stage2_lawam_condition_checkpoint_path
                if self._lawam_action_recipe
                else self.model_cfg.stage2_effect_teacher_checkpoint_path
            )
            condition_config = (
                self.model_cfg.stage2_lawam_condition_config_path
                if self._lawam_action_recipe
                else self.model_cfg.stage2_effect_teacher_config_path
            )
            if Path(self.model_cfg.lam_ckpt_path).expanduser().resolve() != Path(
                str(condition_checkpoint)
            ).expanduser().resolve():
                raise ValueError("phased Stage-2 LaWM and condition checkpoints must match")
            if Path(self.model_cfg.lam_yaml_path).expanduser().resolve() != Path(
                str(condition_config)
            ).expanduser().resolve():
                raise ValueError("phased Stage-2 LaWM and condition configs must match")
            if self._lawam_action_recipe and self.model_cfg.stage2_condition_source != "lawam_idm":
                raise ValueError("LaWAM-action Stage-2 requires stage2_condition_source=lawam_idm")
            required_runtime_paths = {
                "stage2_latent_cache_root": self.model_cfg.stage2_latent_cache_root,
                "stage2_action_decoder_checkpoint_path": self.model_cfg.stage2_action_decoder_checkpoint_path,
                "stage2_action_decoder_data_root": self.model_cfg.stage2_action_decoder_data_root,
            }
            missing_runtime_paths = [
                name for name, value in required_runtime_paths.items()
                if value is None or not str(value).strip()
            ]
            if missing_runtime_paths:
                raise ValueError(
                    f"phased W/P/E is missing runtime paths: {missing_runtime_paths}"
                )
            if deployment_bootstrap:
                latent_contract = SimpleNamespace(
                    effect_center=torch.zeros(32, dtype=torch.float32),
                    effect_scale=torch.ones(32, dtype=torch.float32),
                    action_center=torch.zeros(32, dtype=torch.float32),
                    action_scale=torch.ones(32, dtype=torch.float32),
                    action_logvar_lower=torch.full(
                        (32,), action_logvar_min, dtype=torch.float32
                    ),
                    action_logvar_upper=torch.full(
                        (32,), action_logvar_max, dtype=torch.float32
                    ),
                    temporal_contract=self.stage2_temporal_contract.identifier,
                )
            elif self._robotwin_phased_stage2:
                latent_contract = load_robotwin_stage2_latent_contract(
                    str(self.model_cfg.stage2_latent_cache_root),
                    temporal_contract=self.stage2_temporal_contract,
                    expected_action_checkpoint_sha256=str(
                        self.model_cfg.stage2_action_decoder_checkpoint_sha256
                    ),
                    expected_effect_checkpoint_sha256=(
                        None
                        if self._lawam_action_recipe
                        else str(self.model_cfg.stage2_effect_teacher_checkpoint_sha256)
                    ),
                    expected_condition_checkpoint_sha256=(
                        str(self.model_cfg.stage2_lawam_condition_checkpoint_sha256)
                        if self._lawam_action_recipe
                        else None
                    ),
                )
            elif self._bridge_phased_stage2:
                latent_contract = load_bridge_stage2_latent_contract(
                    str(self.model_cfg.stage2_latent_cache_root),
                    temporal_contract=self.stage2_temporal_contract,
                    expected_action_checkpoint_sha256=str(
                        self.model_cfg.stage2_action_decoder_checkpoint_sha256
                    ),
                    expected_effect_checkpoint_sha256=str(
                        self.model_cfg.stage2_effect_teacher_checkpoint_sha256
                    ),
                )
            else:
                latent_contract = load_libero_stage2_latent_contract(
                    str(self.model_cfg.stage2_latent_cache_root),
                    temporal_contract=self.stage2_temporal_contract,
                    **({"action_teacher_transfer": self.model_cfg.stage2_action_teacher_transfer}
                       if getattr(self.model_cfg, "stage2_action_teacher_transfer", None) else {}),
                )
            if latent_contract.temporal_contract != self.stage2_temporal_contract.identifier:
                raise ValueError("Stage-2 latent cache resolved a different temporal contract")
            self.stage2_latent_affine = Stage2LatentAffine(latent_contract)
            if deployment_bootstrap:
                from starVLA.model.framework.latent_world.deployment_bootstrap import (
                    build_checkpoint_action_decoder,
                )

                self.frozen_action_decoder = build_checkpoint_action_decoder(
                    self.model_cfg.stage2_action_decoder_config_path,
                    robotwin=self._robotwin_phased_stage2,
                )
            elif self._robotwin_phased_stage2:
                self.frozen_action_decoder = load_frozen_robotwin_action_decoder(
                    checkpoint_path=str(self.model_cfg.stage2_action_decoder_checkpoint_path),
                    action_data_root=str(self.model_cfg.stage2_action_decoder_data_root),
                    expected_checkpoint_sha256=str(
                        self.model_cfg.stage2_action_decoder_checkpoint_sha256
                    ),
                    expected_condition_source=(
                        "lawam_idm" if self._lawam_action_recipe else "e39"
                    ),
                )
            elif self._bridge_phased_stage2:
                self.frozen_action_decoder = load_frozen_bridge_action_decoder(
                    checkpoint_path=str(self.model_cfg.stage2_action_decoder_checkpoint_path),
                    action_data_root=str(self.model_cfg.stage2_action_decoder_data_root),
                    expected_checkpoint_sha256=str(
                        self.model_cfg.stage2_action_decoder_checkpoint_sha256
                    ),
                )
            else:
                self.frozen_action_decoder = load_frozen_libero_action_decoder(
                    checkpoint_path=str(self.model_cfg.stage2_action_decoder_checkpoint_path),
                    action_data_root=str(self.model_cfg.stage2_action_decoder_data_root),
                    **({"action_teacher_transfer": self.model_cfg.stage2_action_teacher_transfer}
                       if getattr(self.model_cfg, "stage2_action_teacher_transfer", None) else {}),
                )
            action_logvar_min = latent_contract.action_logvar_lower
            action_logvar_max = latent_contract.action_logvar_upper
        if self.stage2_arm in {"t2", "p2"}:
            if int(self.model_cfg.stage2_latent_dim) != 32:
                raise ValueError("Stage-2 requires 32D effect/action tokens")
            if not self._phased_stage2:
                self.paired_lawm_adapter = PairedLaWMAdapter(
                    latent_dim=32,
                    adapter_dim=int(self.model_cfg.stage2_adapter_dim),
                )
            if self._release_residual_recipe:
                self.stage2_action_residual = ActionResidualProjection(
                    latent_dim=32,
                    adapter_dim=int(self.model_cfg.stage2_adapter_dim),
                    hidden_dim=int(self.model_cfg.flow_cfg.hidden_dim),
                )
            else:
                self.paired_expert_adapters = PairedExpertAdapters(
                    hidden_dim=int(self.model_cfg.flow_cfg.hidden_dim),
                    vision_dim=int(self.model_cfg.flow_cfg.vision_dim),
                    latent_dim=32,
                    adapter_dim=int(self.model_cfg.stage2_adapter_dim),
                )
            if self.stage2_arm == "p2":
                if self.model_cfg.stage2_action_distribution not in {
                    "deterministic_v1",
                    "dense_transition_gaussian_v3",
                }:
                    raise ValueError("unsupported Stage-2 action distribution")
                if self.model_cfg.stage2_action_inference not in {"mean", "sample"}:
                    raise ValueError("stage2_action_inference must be mean or sample")
                if (
                    self.model_cfg.stage2_action_distribution == "dense_transition_gaussian_v3"
                    and not bool(self.model_cfg.future_prediction)
                ):
                    raise ValueError("dense-transition Stage-2 requires future_prediction=true")
                self.paired_latent_predictor = PairedLatentPredictor(
                    query_count=self.flow_action_num_queries,
                    query_dim=vlm_hidden_dim,
                    hidden_dim=int(self.model_cfg.stage2_predictor_hidden_dim),
                    latent_dim=32,
                    visual_dim=int(self.model_cfg.flow_cfg.vision_dim),
                    state_dim=int(self.model_cfg.flow_cfg.state_dim),
                    action_distribution=self.model_cfg.stage2_action_distribution,
                    logvar_min=action_logvar_min,
                    logvar_max=action_logvar_max,
                    num_embodiments=int(self.lam.num_embodiments),
                    horizon_condition_dim=int(self.model_cfg.stage2_horizon_condition_dim),
                    visual_grid_size=(lam_grid_h, lam_grid_w),
                    action_decoder_layers=int(self.model_cfg.stage2_action_decoder_layers),
                    action_decoder_heads=int(self.model_cfg.stage2_action_decoder_heads),
                    action_decoder_dropout=float(self.model_cfg.stage2_action_decoder_dropout),
                    enable_effect_head=not self._lawam_action_recipe,
                    action_head_architecture=self.model_cfg.stage2_action_head_architecture,
                )
            if not self._lawam_action_recipe:
                self.vlm_to_lam.requires_grad_(False)
        self.apply_stage2_freeze_contract()
        self._flow_train_step: int = 0
        self._last_stage2_residual_metrics: Dict[str, torch.Tensor] = {}
        if bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling) and not bool(self.model_cfg.detach_future_feature):
            print(
                "[LatentWorldPolicyBackend][warn] enable_flow_h_t1_scheduled_sampling=true and "
                "detach_future_feature=false: flow gradients through GT-conditioned branch can increase loss_perceptual."
            )
        self.sync_training_modes(mode=self.training)

    def apply_stage2_freeze_contract(self) -> None:
        """Reassert arm-specific freezes after the generic release freeze policy."""

        if self.stage2_arm != "t2":
            return
        self.vlm.requires_grad_(False)
        self.vlm_to_lam.requires_grad_(False)
        self.act_query.requires_grad_(False)
        self.flow_action_query.requires_grad_(False)

    def _uses_lawam_action_recipe(self) -> bool:
        """Default hand-built legacy/test backends to the pre-existing Effect path."""

        return bool(getattr(self, "_lawam_action_recipe", False))

    def action_injection_enabled(self) -> bool:
        return bool(getattr(getattr(self, "model_cfg", None), "stage2_action_injection_enabled", True))

    def apply_stage2_phase_contract(self, phase: str) -> None:
        """Apply the component-level W/P/E gradient ownership contract."""

        normalized = str(phase).strip().upper()
        if not self._phased_stage2 or normalized != self.stage2_phase:
            raise ValueError(
                f"backend/config phase mismatch: backend={self.stage2_phase!r}, requested={normalized!r}"
            )
        if normalized not in {"W", "P", "E"}:
            raise ValueError("Stage-2 phase must be W, P, or E")
        if self.paired_latent_predictor is None:
            raise RuntimeError("phased Stage-2 predictor was not initialized")

        self.lam.requires_grad_(False)
        self.flow.requires_grad_(normalized == "E")
        self.vlm_to_lam.requires_grad_(self._uses_lawam_action_recipe())
        if self.paired_lawm_adapter is not None:
            self.paired_lawm_adapter.requires_grad_(False)
        if self.paired_expert_adapters is not None:
            self.paired_expert_adapters.requires_grad_(normalized == "E")
            # The phased path always selects the learned z-action branch.  Keep
            # the unused physical-plan branch frozen after the recursive toggle,
            # otherwise Phase E exposes unused parameters to DDP.
            self.paired_expert_adapters.plan.physical.requires_grad_(False)
        if self.stage2_action_residual is not None:
            self.stage2_action_residual.requires_grad_(
                normalized in {"P", "E"} and self.action_injection_enabled()
            )
        self.paired_latent_predictor.requires_grad_(False)
        if self.paired_latent_predictor.effect_head is not None:
            self.paired_latent_predictor.effect_head.requires_grad_(True)
        self.paired_latent_predictor.action_head.requires_grad_(normalized in {"P", "E"})
        self.lam.decoder.requires_grad_(
            normalized == "E"
            if self._uses_lawam_action_recipe()
            else normalized in {"W", "E"}
        )
        self.act_query.requires_grad_(
            not (self._release_residual_recipe and normalized == "W")
        )
        self.flow_action_query.requires_grad_(normalized in {"P", "E"})
        if self.frozen_action_decoder is None or self.stage2_latent_affine is None:
            raise RuntimeError("phased Stage-2 runtime assets were not initialized")
        self.frozen_action_decoder.requires_grad_(False)
        self.frozen_action_decoder.eval()
        self.sync_training_modes(mode=self.training)

    def _inject_queries(
        self,
        *,
        inputs_embeds: torch.Tensor,
        placeholder_mask: torch.BoolTensor,
        queries: torch.Tensor,
        num_queries: int,
        name: str,
    ) -> None:
        device = inputs_embeds.device
        placeholder_mask = placeholder_mask.to(device=device, dtype=torch.bool)

        bsz, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        expected = int(bsz * num_queries)
        got = int(placeholder_mask.sum().item())
        if got != expected:
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch: got={got}, expected={expected}"
            )

        per_sample = placeholder_mask.sum(dim=1)
        if not torch.all(per_sample == int(num_queries)):
            bad = torch.nonzero(per_sample != int(num_queries), as_tuple=False).flatten()
            b = int(bad[0].item()) if bad.numel() > 0 else -1
            got_b = int(per_sample[b].item()) if b >= 0 else -1
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch for sample {b}: got={got_b}, expected={num_queries}"
            )

        idx = placeholder_mask.nonzero(as_tuple=False)
        flat = idx[:, 0] * seq_len + idx[:, 1]
        idx = idx[flat.argsort()]
        b_idx, p_idx = idx[:, 0], idx[:, 1]
        q_idx = torch.arange(int(num_queries), device=device).repeat(bsz)
        inputs_embeds[b_idx, p_idx, :] = queries[q_idx]

    def _run_vlm_stage(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        flow_placeholder_mask: torch.BoolTensor,
        act_query: torch.Tensor,
        flow_query: torch.Tensor,
    ) -> Dict[str, Any]:
        embed = self.vlm.get_input_embeddings()
        if embed is None:
            raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
        inputs_embeds = embed(input_ids)

        self._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=act_placeholder_mask,
            queries=act_query,
            num_queries=int(self.num_action_queries),
            name="act_query",
        )
        self._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=flow_placeholder_mask,
            queries=flow_query,
            num_queries=int(flow_query.shape[0]),
            name="flow_query",
        )

        # HF transformers 4.57 Qwen3VL causal wrapper drops `hidden_states`
        # from the returned output; use the underlying multimodal model output.
        vlm_out = self.vlm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if vlm_out is None:
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model forward returned None; "
                "expected `last_hidden_state`."
            )
        hidden = getattr(vlm_out, "last_hidden_state", None)
        if hidden is None:
            output_keys = list(vlm_out.keys()) if hasattr(vlm_out, "keys") else None
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model output does not expose "
                "`last_hidden_state`. "
                f"output_type={type(vlm_out).__name__}, output_keys={output_keys}."
            )
        bsz = int(hidden.shape[0])
        effect_queries = int(self.num_action_queries)
        plan_queries = int(self.flow_action_num_queries)
        h_effect = hidden[act_placeholder_mask]
        if h_effect.numel() == 0:
            raise ValueError("[LatentWorldPolicyBackend] empty effect hidden selection; check act_placeholder_mask.")
        if h_effect.shape[0] != bsz * effect_queries:
            raise ValueError(
                "[LatentWorldPolicyBackend] effect hidden count mismatch: "
                f"got={h_effect.shape[0]}, expected={bsz * effect_queries}"
            )
        h_plan = hidden[flow_placeholder_mask]
        if h_plan.numel() == 0 or h_plan.shape[0] != bsz * plan_queries:
            raise ValueError(
                "[LatentWorldPolicyBackend] plan hidden count mismatch: "
                f"got={h_plan.shape[0]}, expected={bsz * plan_queries}"
            )

        h_effect = h_effect.view(bsz, effect_queries, -1)
        h_plan = h_plan.view(bsz, plan_queries, -1)
        pred_latent = self.vlm_to_lam(h_effect)
        return {
            "h_vlm": hidden,
            "h_effect": h_effect,
            "h_plan": h_plan,
            "pred_latent": pred_latent,
            "vlm_out": vlm_out,
        }

    def _build_lam_teacher_inputs_for_distill(self, primary_video: torch.Tensor) -> torch.Tensor:
        expected_t = int(getattr(getattr(self.lam, "encoder", None), "num_frames", 0) or 0)
        if expected_t <= 0:
            return primary_video

        cur_t = int(primary_video.shape[1])
        if cur_t == expected_t:
            return primary_video
        if cur_t < expected_t:
            raise ValueError(
                f"[LatentWorldPolicyBackend] distill teacher temporal mismatch: got T={cur_t}, "
                f"but LAM encoder expects num_frames={expected_t}."
            )
        idx = torch.linspace(0, cur_t - 1, steps=expected_t, device=primary_video.device).round().long()
        return primary_video.index_select(1, idx)

    def _run_lam_teacher(self, *, primary_video: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        primary_video_t = self._build_lam_teacher_inputs_for_distill(primary_video)
        with torch.no_grad():
            lam_out = self.lam.get_latent_action(
                videos=primary_video_t,
                states=None,
                dec_videos=primary_video_t,
                predict_future_frame=False,
                embodiment_ids=embodiment_id,
            )
        # Some LAM implementations return inference tensors here. Materialize a normal
        # tensor before using it in training losses, otherwise autograd cannot save it.
        return lam_out["quantized"].detach().clone()

    def _compute_latent_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        latent_loss_type: str,
    ) -> torch.Tensor:
        if pred_latent.shape != teacher_latent.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] latent shape mismatch: pred={tuple(pred_latent.shape)}, "
                f"teacher={tuple(teacher_latent.shape)}"
            )
        if str(latent_loss_type).lower() == "mse":
            return F.mse_loss(pred_latent, teacher_latent)
        return 1 - F.cosine_similarity(pred_latent, teacher_latent, dim=-1).mean()

    def _compute_distill_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        primary_video: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        teacher_latent = self._run_lam_teacher(primary_video=primary_video, embodiment_id=embodiment_id)
        return self._compute_latent_loss(
            pred_latent=pred_latent,
            teacher_latent=teacher_latent,
            latent_loss_type=self.model_cfg.latent_loss_type,
        )

    def _decode_future_tokens_strict_single_query(
        self,
        *,
        h_t: torch.Tensor,
        pred_action_emb: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        if pred_action_emb.shape[1] != 1:
            raise ValueError(
                f"[{source}] future_prediction requires single-query latent action, "
                f"got query_dim={pred_action_emb.shape[1]}."
            )
        decoded = self.lam.decoder(h_t, pred_action_emb)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() == 4:
            decoded = decoded[:, 0, :, :] if decoded.shape[1] == 1 else decoded[:, -1, :, :]
        return decoded

    @classmethod
    def build(
        cls,
        cfg: LatentWorldPolicyConfig,
        *,
        vlm_model_id: str,
    ) -> "LatentWorldPolicyBackend":
        model = cls(cfg, vlm_model_id=vlm_model_id)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.sync_training_modes(mode=mode)
        return self

    def sync_training_modes(self, mode: bool = True) -> None:
        sync_managed_modules_training_mode(
            self.vlm,
            self.lam,
            self.flow,
            self.vlm_to_lam,
            self.paired_lawm_adapter,
            self.paired_expert_adapters,
            self.paired_latent_predictor,
            self.stage2_action_residual,
            self.stage2_latent_affine,
            self.frozen_action_decoder,
            mode=mode,
        )

    def _resolve_stage2_effect(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        h_effect: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.stage2_arm == "b0":
            return None
        if self._stage2_token_override is not None:
            override = self._stage2_token_override
            expected_shape = (int(h_effect.shape[0]), int(self.model_cfg.stage2_latent_dim))
            if tuple(override.effect.shape) != expected_shape or override.action.shape != override.effect.shape:
                raise ValueError(
                    "Stage-2 token override must match the current batch shape "
                    f"{expected_shape}, got effect={tuple(override.effect.shape)} "
                    f"action={tuple(override.action.shape)}"
                )
            return override.effect
        if self.stage2_arm == "t2":
            if "z_effect" not in prepared_batch:
                raise KeyError("T2 requires teacher z_effect")
            effect = prepared_batch["z_effect"]
            if effect.ndim != 2 or effect.shape[-1] != 32:
                raise ValueError("T2 teacher effect must have shape [B,32]")
            return effect.detach()
        if self.paired_latent_predictor is None:
            raise RuntimeError("P2 predictor was not initialized")
        if self._uses_lawam_action_recipe():
            raise RuntimeError(
                "LaWAM-action Stage-2 resolves its condition from VLMToLAM raw output"
            )
        return self.paired_latent_predictor.predict_effect(h_effect)

    def _stage2_horizon_condition(
        self,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
    ) -> torch.Tensor:
        action_hz = prepared_batch["action_hz"].float().reshape(-1)
        if not bool(torch.isfinite(action_hz).all()) or bool((action_hz <= 0).any()):
            raise ValueError("Stage-2 action_hz must contain finite positive values")
        batch = int(action_hz.shape[0])
        temporal = getattr(self, "stage2_temporal_contract", None)
        if temporal is None:
            temporal = resolve_stage2_temporal_contract(
                getattr(self.model_cfg, "stage2_temporal_contract", None)
            )
        configured_seconds = float(self.model_cfg.stage2_horizon_seconds)
        if configured_seconds != temporal.duration_seconds:
            raise ValueError(
                "stage2_horizon_seconds is incompatible with stage2_temporal_contract: "
                f"seconds={configured_seconds}, contract={temporal.identifier!r}"
            )
        seconds = torch.full_like(action_hz, configured_seconds)
        time_mask = prepared_batch.get("teacher_time_mask")
        if time_mask is not None:
            expected_time_shape = (batch, temporal.representation_steps)
            if time_mask.shape != expected_time_shape:
                raise ValueError(
                    f"teacher_time_mask must have shape {list(expected_time_shape)}"
                )
            if (
                is_libero_04s_adapter_temporal_contract(temporal)
                and not bool(time_mask.to(dtype=torch.bool).all())
            ):
                raise ValueError("0.4s adapter teacher_time_mask must keep all 24 representation slots")
            if (
                is_robotwin_stage2_temporal_contract(temporal)
                and not bool(time_mask.to(dtype=torch.bool).all())
            ):
                raise ValueError("RoboTwin teacher_time_mask must keep all 36 slots")
            if (
                is_bridge_stage2_temporal_contract(temporal)
                and not bool(time_mask.to(dtype=torch.bool).all())
            ):
                raise ValueError("Bridge teacher_time_mask must keep all 24 resampled slots")

        physical_count = prepared_batch.get("physical_valid_count")
        if physical_count is not None:
            if physical_count.shape != (batch,):
                raise ValueError("physical_valid_count must have shape [B]")
            valid_count = physical_count.to(device=action_hz.device, dtype=torch.float32)
            if not bool(torch.isfinite(valid_count).all()) or not bool(
                torch.equal(valid_count, valid_count.round())
            ):
                raise ValueError("physical_valid_count must contain finite integers")
        elif (
            is_libero_04s_adapter_temporal_contract(temporal)
            or is_bridge_stage2_temporal_contract(temporal)
        ):
            if "actions" in prepared_batch:
                raise KeyError("resampled Stage-2 training batches require physical_valid_count")
            valid_count = torch.full_like(action_hz, float(temporal.physical_valid_count))
        elif time_mask is None:
            valid_count = torch.floor(action_hz * seconds)
        else:
            valid_count = time_mask.to(device=action_hz.device, dtype=torch.float32).sum(dim=-1)

        if is_libero_04s_adapter_temporal_contract(temporal):
            expected_hz = torch.full_like(action_hz, temporal.control_hz)
            expected_count = torch.full_like(valid_count, float(temporal.physical_valid_count))
            if not bool(torch.equal(action_hz, expected_hz)):
                raise ValueError("0.4s adapter action_hz must be exactly 20")
            if not bool(torch.equal(valid_count, expected_count)):
                raise ValueError("0.4s adapter physical_valid_count must be exactly 8")
        elif is_robotwin_stage2_temporal_contract(temporal):
            expected_hz = torch.full_like(action_hz, temporal.control_hz)
            expected_count = torch.full_like(
                valid_count, float(temporal.physical_valid_count)
            )
            if not bool(torch.equal(action_hz, expected_hz)):
                raise ValueError("RoboTwin Stage-2 action_hz must be exactly 30")
            if not bool(torch.equal(valid_count, expected_count)):
                raise ValueError("RoboTwin Stage-2 physical_valid_count must be exactly 36")
        elif is_bridge_stage2_temporal_contract(temporal):
            expected_hz = torch.full_like(action_hz, temporal.control_hz)
            expected_count = torch.full_like(
                valid_count, float(temporal.physical_valid_count)
            )
            if not bool(torch.equal(action_hz, expected_hz)):
                raise ValueError("Bridge Stage-2 action_hz must be exactly 5")
            if not bool(torch.equal(valid_count, expected_count)):
                raise ValueError("Bridge Stage-2 physical_valid_count must be exactly 6")
        if bool((valid_count <= 0).any()) or bool(
            (valid_count > temporal.representation_steps).any()
        ):
            raise ValueError(
                "Stage-2 valid_action_count must be in "
                f"[1,{temporal.representation_steps}]"
            )
        return torch.stack((action_hz, seconds, valid_count), dim=-1)

    def _resolve_stage2_action(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        h_plan: torch.Tensor,
        effect: torch.Tensor,
        h_t: torch.Tensor,
        h_t1_pred: torch.Tensor,
    ) -> PairedTokens:
        if self._stage2_token_override is not None:
            return self._stage2_token_override
        if self.stage2_arm == "t2":
            if "z_action" not in prepared_batch:
                raise KeyError("T2 requires teacher z_action")
            action = prepared_batch["z_action"]
            if action.shape != effect.shape:
                raise ValueError("T2 teacher action must match effect [B,32]")
            logvar = prepared_batch.get("z_action_logvar")
            if logvar is not None and logvar.shape != action.shape:
                raise ValueError("T2 teacher action logvar must match action [B,32]")
            return PairedTokens(
                effect=effect,
                action=action.detach(),
                action_logvar=logvar.detach() if logvar is not None else None,
            )
        if self.paired_latent_predictor is None:
            raise RuntimeError("P2 predictor was not initialized")
        state_key = "state"
        if (
            self._phased_stage2
            and self.model_cfg.stage2_action_state_source
            in {"actionvae14_padded32", "robotwin_actionvae14_padded32"}
        ):
            state_key = "action_predictor_state"
        if state_key not in prepared_batch:
            raise KeyError(
                f"Stage-2 action predictor requires batch field {state_key!r} for "
                f"state source {self.model_cfg.stage2_action_state_source!r}"
            )
        action_prior = self.paired_latent_predictor.predict_action(
            h_plan,
            current_visual=h_t.detach(),
            future_visual=h_t1_pred.detach(),
            state=prepared_batch[state_key],
            effect=effect.detach(),
            embodiment_id=prepared_batch["embodiment_id"],
            horizon_condition=self._stage2_horizon_condition(prepared_batch),
        )
        action_value = action_prior.mean
        if not self.training and self.model_cfg.stage2_action_inference == "sample":
            action_value = action_prior.rsample()
        return PairedTokens(
            effect=effect,
            action=action_value,
            action_logvar=action_prior.logvar,
        )

    def _resolve_stage2_tokens(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        h_effect: torch.Tensor,
    ) -> Optional[PairedTokens]:
        """Compatibility entry point for deterministic-v1 tools and scoped overrides."""

        effect = self._resolve_stage2_effect(
            prepared_batch=prepared_batch, h_effect=h_effect
        )
        if effect is None:
            return None
        if self._stage2_token_override is not None:
            return self._stage2_token_override
        if self.stage2_arm == "t2":
            if "z_action" not in prepared_batch:
                raise KeyError("T2 requires teacher z_action")
            return PairedTokens(effect=effect, action=prepared_batch["z_action"].detach())
        if self.paired_latent_predictor is None:
            raise RuntimeError("P2 predictor was not initialized")
        if self.paired_latent_predictor.action_distribution != "deterministic_v1":
            raise RuntimeError("dense-transition tokens require the sequential future-aware route")
        _, action = self.paired_latent_predictor(h_effect)
        return PairedTokens(effect=effect, action=action)

    @contextmanager
    def override_stage2_tokens(self, *, effect: torch.Tensor, action: torch.Tensor):
        """Temporarily replace the resolved pair after VLM prediction and before adapters."""

        if self.stage2_arm == "b0":
            raise RuntimeError("B0 has no Stage-2 tokens to override")
        if self._stage2_token_override is not None:
            raise RuntimeError("nested Stage-2 token overrides are not supported")
        if effect.ndim != 2 or effect.shape[-1] != int(self.model_cfg.stage2_latent_dim):
            raise ValueError("Stage-2 effect override must be [B, latent_dim]")
        if action.shape != effect.shape:
            raise ValueError("Stage-2 action override must match the effect override")
        self._stage2_token_override = PairedTokens(
            effect=effect.detach(),
            action=action.detach(),
        )
        try:
            yield
        finally:
            self._stage2_token_override = None

    def set_flow_train_step(self, step: int) -> None:
        self._flow_train_step = max(0, int(step))

    def _stage2_action_residual_condition(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        predicted_action: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self._release_residual_recipe or self.stage2_action_residual is None:
            raise RuntimeError("action residual requested outside the release-residual recipe")
        if predicted_action.ndim != 2 or predicted_action.shape[-1] != 32:
            raise ValueError("predicted z-action must be [B,32] for residual conditioning")
        if not self.action_injection_enabled():
            # Keep the plan loss graph, but do not call the unused projection or RNG curriculum.
            self._last_stage2_residual_metrics = {
                name: predicted_action.new_tensor(value)
                for name, value in {
                    "stage2_residual_enabled": 0.0,
                    "stage2_residual_teacher_probability": 0.0,
                    "stage2_residual_teacher_fraction": 0.0,
                    "stage2_residual_keep_fraction": 0.0,
                    "stage2_residual_predictor_detached": 1.0,
                }.items()
            }
            return None

        step = int(self._flow_train_step)
        enabled = self.stage2_phase == "E"
        teacher_prob = 0.0
        detach_predictor = False
        if self.stage2_phase == "P":
            start_step = int(self.model_cfg.stage2_residual_warmup_start_step)
            enabled = step >= start_step
            ramp_steps = max(1, int(self.model_cfg.stage2_residual_teacher_ramp_steps))
            ratio = min(1.0, max(0.0, float(step - start_step) / float(ramp_steps)))
            teacher_start = float(self.model_cfg.stage2_residual_teacher_prob_start)
            teacher_end = float(self.model_cfg.stage2_residual_teacher_prob_end)
            teacher_prob = teacher_start + ratio * (teacher_end - teacher_start)
            teacher_prob = max(0.0, min(1.0, teacher_prob)) if enabled else 0.0
            detach_predictor = (
                step < int(self.model_cfg.stage2_residual_predictor_grad_start_step)
            )

        condition = predicted_action.detach() if detach_predictor else predicted_action
        teacher_fraction = torch.zeros(
            (), device=predicted_action.device, dtype=torch.float32
        )
        if self.training and enabled and teacher_prob > 0.0:
            teacher = prepared_batch.get("z_action")
            if teacher is None or teacher.shape != predicted_action.shape:
                raise ValueError("residual teacher z-action must match predicted [B,32]")
            teacher_mask = (
                torch.rand(
                    predicted_action.shape[0], 1, device=predicted_action.device
                )
                < teacher_prob
            )
            condition = torch.where(
                teacher_mask,
                teacher.to(device=condition.device, dtype=condition.dtype).detach(),
                condition,
            )
            teacher_fraction = teacher_mask.float().mean()

        residual = self.stage2_action_residual(condition)
        residual = residual * float(enabled)
        keep_fraction = torch.ones(
            (), device=predicted_action.device, dtype=torch.float32
        )
        dropout = float(self.model_cfg.stage2_residual_condition_dropout)
        if self.training and enabled and dropout > 0.0:
            keep = (
                torch.rand(residual.shape[0], 1, device=residual.device) >= dropout
            )
            residual = residual * keep.to(dtype=residual.dtype)
            keep_fraction = keep.float().mean()

        self._last_stage2_residual_metrics = {
            "stage2_residual_enabled": torch.tensor(
                float(enabled), device=predicted_action.device
            ),
            "stage2_residual_teacher_probability": torch.tensor(
                teacher_prob, device=predicted_action.device
            ),
            "stage2_residual_teacher_fraction": teacher_fraction,
            "stage2_residual_keep_fraction": keep_fraction,
            "stage2_residual_predictor_detached": torch.tensor(
                float(detach_predictor), device=predicted_action.device
            ),
        }
        return residual

    def _flow_h_t1_pred_prob(self) -> float:
        if not bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling):
            return 1.0
        start = float(self.model_cfg.flow_h_t1_pred_prob_start)
        end = float(self.model_cfg.flow_h_t1_pred_prob_end)
        ramp_steps = max(1, int(self.model_cfg.flow_h_t1_pred_ramp_steps))
        ratio = min(1.0, float(self._flow_train_step) / float(ramp_steps))
        prob = start + (end - start) * ratio
        return float(max(0.0, min(1.0, prob)))

    def _build_flow_future_condition(
        self,
        *,
        h_t1_pred: torch.Tensor,
        h_t1_gt: torch.Tensor,
    ) -> torch.Tensor:
        if h_t1_pred.shape != h_t1_gt.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] h_t1 shape mismatch: pred={tuple(h_t1_pred.shape)}, gt={tuple(h_t1_gt.shape)}"
            )

        if not self.training:
            return h_t1_pred

        prob_pred = self._flow_h_t1_pred_prob()
        bsz = int(h_t1_pred.shape[0])
        pred_mask = (torch.rand(bsz, 1, 1, device=h_t1_pred.device) < prob_pred)
        cond_future = torch.where(pred_mask, h_t1_pred, h_t1_gt)

        if not bool(self.model_cfg.detach_future_feature):
            gt_mask = (~pred_mask).to(dtype=h_t1_pred.dtype)
            # Straight-through bridge:
            cond_future = cond_future + gt_mask * (h_t1_pred - h_t1_pred.detach())
        else:
            cond_future = cond_future.detach()

        return cond_future

    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = self.vlm.save_pretrained(str(save_dir), **kwargs)
        ckpt_path = save_dir / "pytorch_model.pt"
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt_path)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[LatentWorldPolicyBackend] Full checkpoint saved to: {ckpt_path} (num_params={num_params})")
        return out

    # ------------------
    # Forward
    # ------------------
    def _prepare_queries(self, *, device: torch.device, vlm_stage_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        flow_query = getattr(self, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldPolicyBackend] flow_action_query is None; check policy backend initialization.")
        vlm_embed_dtype = _module_param_dtype(self.vlm.get_input_embeddings(), default=vlm_stage_dtype)
        act_query = self.act_query.to(device=device, dtype=vlm_embed_dtype)
        flow_query = flow_query.to(device=device, dtype=vlm_embed_dtype)
        return act_query, flow_query

    def _prepare_train_batch(self, *, batch: LatentWorldPolicyTrainBatch) -> LatentWorldPolicyTrainBatch:
        return batch

    def _prepare_infer_batch(self, *, batch: LatentWorldPolicyInferBatch) -> LatentWorldPolicyInferBatch:
        return batch

    def _run_shared_encoding_core(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        primary_visual_input: torch.Tensor,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        vlm_stage_dtype = self.model_cfg.vlm_dtype
        lam_stage_dtype = torch.bfloat16

        device = prepared_batch["input_ids"].device
        act_query, flow_query = self._prepare_queries(device=device, vlm_stage_dtype=vlm_stage_dtype)

        with _cuda_autocast(vlm_stage_dtype):
            vlm_out_dict = self._run_vlm_stage(
                input_ids=prepared_batch["input_ids"],
                attention_mask=prepared_batch["attention_mask"],
                pixel_values=prepared_batch["pixel_values"],
                image_grid_thw=prepared_batch["image_grid_thw"],
                act_placeholder_mask=prepared_batch["act_placeholder_mask"],
                flow_placeholder_mask=prepared_batch["flow_placeholder_mask"],
                act_query=act_query,
                flow_query=flow_query,
            )
        h_vlm = vlm_out_dict["h_vlm"]
        h_effect = vlm_out_dict["h_effect"]
        h_plan = vlm_out_dict["h_plan"]
        pred_action_emb = vlm_out_dict["pred_latent"]
        if self._uses_lawam_action_recipe():
            if self.stage2_latent_affine is None:
                raise RuntimeError("LaWAM-action latent affine was not initialized")
            predicted_effect = self.stage2_latent_affine.condition_to_standardized(
                pred_action_emb[:, 0].float()
            )
        else:
            predicted_effect = self._resolve_stage2_effect(
                prepared_batch=prepared_batch,
                h_effect=h_effect,
            )
        if predicted_effect is not None and not self._uses_lawam_action_recipe():
            if self._phased_stage2:
                if self.stage2_latent_affine is None:
                    raise RuntimeError("phased Stage-2 latent affine was not initialized")
                # LaWM was trained in raw Effect-teacher space. Distillation, the
                # Expert, and the action head use the cache-standardized token.
                pred_action_emb = self.stage2_latent_affine.effect_to_raw(
                    predicted_effect
                ).unsqueeze(1)
            else:
                if self.paired_lawm_adapter is None:
                    raise RuntimeError("paired LaWM adapter was not initialized")
                pred_action_emb = self.paired_lawm_adapter(predicted_effect)

        with _cuda_autocast(lam_stage_dtype):
            if lam_features_with_no_grad:
                with torch.no_grad():
                    features = self.lam.extract_vision_features(primary_visual_input)
            else:
                features = self.lam.extract_vision_features(primary_visual_input)

            if features is None:
                raise ValueError(f"[{source}] lam visual feature extraction returned None; check LAM config.")
            h_t_original = features[:, 0, :, :]
            h_t = h_t_original
            h_t1_gt = features[:, -1, :, :]

            if self.model_cfg.future_prediction:
                h_t1_pred = self._decode_future_tokens_strict_single_query(
                    h_t=h_t,
                    pred_action_emb=pred_action_emb,
                    source=source,
                )
            else:
                h_t1_pred = h_t

        paired_tokens = None
        # Phase W has no action-head objective.  Keep the randomly initialized
        # action predictor out of both training and validation forward passes.
        predict_action = not (self._phased_stage2 and self.stage2_phase == "W")
        if predicted_effect is not None and predict_action:
            paired_tokens = self._resolve_stage2_action(
                prepared_batch=prepared_batch,
                h_plan=h_plan,
                effect=predicted_effect,
                h_t=h_t,
                h_t1_pred=h_t1_pred,
            )

        return PolicyEncodingState(
            h_vlm=h_vlm,
            h_effect=h_effect,
            h_plan=h_plan,
            pred_action_emb=pred_action_emb,
            h_t=h_t,
            h_t1_pred=h_t1_pred,
            h_t1_gt=h_t1_gt,
            h_t_original=h_t_original,
            predicted_effect=predicted_effect,
            paired_tokens=paired_tokens,
        )

    def _run_shared_encoding_train(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_video"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def _run_shared_encoding_infer(
        self,
        *,
        prepared_batch: LatentWorldPolicyInferBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_image"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def _compute_stage2_plan_losses(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        shared: PolicyEncodingState,
    ) -> Dict[str, torch.Tensor]:
        if not self._phased_stage2 or self.stage2_phase not in {"P", "E"}:
            raise RuntimeError("Stage-2 plan loss is only defined for phases P/E")
        if shared.paired_tokens is None or shared.predicted_effect is None:
            raise RuntimeError("P/E encoding did not produce predicted Effect/Action tokens")
        if shared.paired_tokens.action_logvar is None:
            raise RuntimeError("dense Stage-2 action predictor did not return logvar")
        required = (
            "z_action",
            "z_action_logvar",
            "teacher_trajectory",
            "teacher_current_state",
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        )
        missing = [name for name in required if name not in prepared_batch]
        if missing:
            raise KeyError(f"P/E plan loss is missing teacher fields: {missing}")
        if self.stage2_latent_affine is None or self.frozen_action_decoder is None:
            raise RuntimeError("P/E frozen ActionVAE runtime assets were not initialized")

        teacher = DiagonalGaussian(
            mean=prepared_batch["z_action"].float().detach(),
            logvar=prepared_batch["z_action_logvar"].float().detach(),
        )
        prior = DiagonalGaussian(
            mean=shared.paired_tokens.action.float(),
            logvar=shared.paired_tokens.action_logvar.float(),
        )
        loss_kl = diagonal_gaussian_kl(teacher, prior)
        loss_mean = F.smooth_l1_loss(prior.mean, teacher.mean, beta=0.1)

        current_state = prepared_batch["teacher_current_state"].float()
        current_state_mask = prepared_batch["teacher_current_state_mask"].to(
            device=current_state.device, dtype=torch.bool
        )
        if current_state.shape != current_state_mask.shape or current_state.shape[1:] != (14,):
            raise ValueError("teacher current state/mask must both have shape [B,14]")
        current_state = torch.where(
            current_state_mask, current_state, torch.zeros_like(current_state)
        )
        z_action_raw = self.stage2_latent_affine.action_to_raw(prior.mean)
        condition_raw = (
            shared.pred_action_emb[:, 0].float().detach()
            if self._uses_lawam_action_recipe()
            else self.stage2_latent_affine.effect_to_raw(
                shared.predicted_effect.float().detach()
            )
        )
        predicted_trajectory = self.frozen_action_decoder(
            z_action_raw,
            current_state,
            condition_raw,
        )
        if bool(getattr(self, "_robotwin_phased_stage2", False)):
            trajectory = dual_arm_trajectory_reconstruction_loss(
                predicted_trajectory.float(),
                prepared_batch["teacher_trajectory"].float().detach(),
                time_mask=prepared_batch["teacher_time_mask"],
                arm_pose_mask=prepared_batch["teacher_pose_mask"],
                arm_gripper_mask=prepared_batch["teacher_gripper_mask"],
                beta=0.1,
            )
        else:
            trajectory = trajectory_reconstruction_loss(
                predicted_trajectory.float(),
                prepared_batch["teacher_trajectory"].float().detach(),
                time_mask=prepared_batch["teacher_time_mask"],
                pose_mask=prepared_batch["teacher_pose_mask"],
                gripper_mask=prepared_batch["teacher_gripper_mask"],
                beta=0.1,
            )
        loss_plan = (
            float(self.model_cfg.stage2_action_kl_weight) * loss_kl
            + float(self.model_cfg.stage2_action_mean_weight) * loss_mean
            + float(self.model_cfg.stage2_action_trajectory_weight) * trajectory.total
        )
        return {
            "loss_plan": loss_plan,
            "loss_action_kl": loss_kl,
            "loss_action_mean": loss_mean,
            "loss_action_trajectory": trajectory.total,
            "loss_action_trajectory_position": trajectory.position,
            "loss_action_trajectory_rotation": trajectory.rotation,
            "loss_action_trajectory_gripper": trajectory.gripper,
        }

    def _compute_flow_matching_loss(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        shared: PolicyEncodingState,
    ) -> torch.Tensor:
        h_vlm_for_flow = _apply_flow_only_grad_to_h_vlm(
            h_vlm=shared.h_vlm,
            act_placeholder_mask=prepared_batch["act_placeholder_mask"],
            enable_flow_only=bool(self.model_cfg.flow_only_mode),
        )
        attn_flow = prepared_batch["attention_mask"] == 1
        repeat_steps = int(self.model_cfg.repeated_diffusion_steps)
        h_t_rep = shared.h_t.repeat(
            repeat_steps, *([1] * (shared.h_t.ndim - 1))
        )
        if self._phased_stage2:
            # L_FM may train both predicted-token paths, the Expert/adapters, and
            # VLM, but never reaches LaWM through its future-feature condition.
            h_t1_cond_rep = shared.h_t1_pred.detach().repeat(
                repeat_steps, *([1] * (shared.h_t1_pred.ndim - 1))
            )
        else:
            h_t1_pred_rep = shared.h_t1_pred.repeat(
                repeat_steps, *([1] * (shared.h_t1_pred.ndim - 1))
            )
            h_t1_gt_rep = shared.h_t1_gt.repeat(
                repeat_steps, *([1] * (shared.h_t1_gt.ndim - 1))
            )
            h_t1_cond_rep = self._build_flow_future_condition(
                h_t1_pred=h_t1_pred_rep,
                h_t1_gt=h_t1_gt_rep,
            )
        h_vlm_rep = h_vlm_for_flow.repeat(
            repeat_steps, *([1] * (h_vlm_for_flow.ndim - 1))
        )
        state_rep = prepared_batch["state"].repeat(
            repeat_steps, *([1] * (prepared_batch["state"].ndim - 1))
        )
        state_mask_rep = prepared_batch["state_mask"].repeat(
            repeat_steps, *([1] * (prepared_batch["state_mask"].ndim - 1))
        )
        actions_rep = prepared_batch["actions"].repeat(
            repeat_steps, *([1] * (prepared_batch["actions"].ndim - 1))
        )
        actions_mask_rep = prepared_batch["actions_mask"].repeat(
            repeat_steps, *([1] * (prepared_batch["actions_mask"].ndim - 1))
        )
        action_hz_rep = prepared_batch["action_hz"].repeat(repeat_steps)
        embodiment_rep = prepared_batch["embodiment_id"].repeat(repeat_steps)
        attn_rep = attn_flow.repeat(
            repeat_steps, *([1] * (attn_flow.ndim - 1))
        )

        plan_tokens_rep = None
        plan_mask_rep = None
        effect_tokens_rep = None
        effect_mask_rep = None
        action_residual_rep = None
        if shared.paired_tokens is not None:
            if self._release_residual_recipe:
                action_residual = self._stage2_action_residual_condition(
                    prepared_batch=prepared_batch,
                    predicted_action=shared.paired_tokens.action,
                )
                if action_residual is not None:
                    action_residual_rep = action_residual.repeat(repeat_steps, 1)
            else:
                if self.paired_expert_adapters is None:
                    raise RuntimeError("paired Expert adapters were not initialized")
                plan_tokens, plan_mask, effect_tokens, effect_mask = self.paired_expert_adapters(
                    shared.paired_tokens.effect,
                    shared.paired_tokens.action,
                )
                plan_tokens_rep = plan_tokens.repeat(repeat_steps, 1, 1)
                plan_mask_rep = plan_mask.repeat(repeat_steps, 1)
                effect_tokens_rep = effect_tokens.repeat(repeat_steps, 1, 1)
                effect_mask_rep = effect_mask.repeat(repeat_steps, 1)

        return self.flow(
            h_t=h_t_rep,
            h_t1_star=h_t1_cond_rep,
            h_vlm=h_vlm_rep,
            state=state_rep,
            actions=actions_rep,
            action_hz=action_hz_rep,
            embodiment_id=embodiment_rep,
            state_mask=state_mask_rep,
            actions_mask=actions_mask_rep,
            attention_mask=attn_rep,
            plan_tokens=plan_tokens_rep,
            plan_token_mask=plan_mask_rep,
            effect_tokens=effect_tokens_rep,
            effect_token_mask=effect_mask_rep,
            action_residual=action_residual_rep,
        )

    def forward(
        self,
        *,
        batch: LatentWorldPolicyTrainBatch,
    ) -> Dict[str, torch.Tensor]:
        lam_stage_dtype = torch.bfloat16
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyTrainBatch,
            self._prepare_train_batch(batch=batch),
        )
        device = prepared_batch["input_ids"].device

        shared = self._run_shared_encoding_train(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.forward",
            lam_features_with_no_grad=True,
        )

        zero = torch.zeros((), device=device, dtype=torch.float32)
        self._last_stage2_residual_metrics = {}
        with _cuda_autocast(lam_stage_dtype):
            loss_distill = zero
            loss_lawam_distill = zero
            if self.stage2_arm == "b0" and bool(self.model_cfg.enable_loss_distill):
                if bool(self.model_cfg.stage2_cached_effect_distill):
                    if "z_effect_raw" not in prepared_batch:
                        raise KeyError(
                            "stage2_cached_effect_distill requires manifest-bound z_effect_raw"
                        )
                    teacher_effect = prepared_batch["z_effect_raw"].float().detach().unsqueeze(1)
                    loss_distill = self._compute_latent_loss(
                        pred_latent=shared.pred_action_emb.float(),
                        teacher_latent=teacher_effect,
                        latent_loss_type=self.model_cfg.latent_loss_type,
                    )
                else:
                    loss_distill = self._compute_distill_loss(
                        pred_latent=shared.pred_action_emb,
                        primary_video=prepared_batch["primary_video"],
                        embodiment_id=prepared_batch["embodiment_id"],
                    )
            if self._uses_lawam_action_recipe():
                if "z_lawam_raw" not in prepared_batch:
                    raise KeyError(
                        "LaWAM-action Stage-2 requires manifest-bound z_lawam_raw"
                    )
                loss_lawam_distill = self._compute_latent_loss(
                    pred_latent=shared.pred_action_emb.float(),
                    teacher_latent=(
                        prepared_batch["z_lawam_raw"].float().detach().unsqueeze(1)
                    ),
                    latent_loss_type=self.model_cfg.latent_loss_type,
                )

            if self.model_cfg.future_prediction:
                loss_perceptual = F.mse_loss(shared.h_t1_pred, shared.h_t1_gt)
            else:
                loss_perceptual = zero

            loss_effect_distill = zero
            loss_action_distill = zero
            plan_losses: Dict[str, torch.Tensor] = {
                "loss_plan": zero,
                "loss_action_kl": zero,
                "loss_action_mean": zero,
                "loss_action_trajectory": zero,
                "loss_action_trajectory_position": zero,
                "loss_action_trajectory_rotation": zero,
                "loss_action_trajectory_gripper": zero,
            }
            if self.stage2_arm == "p2":
                if shared.predicted_effect is None:
                    raise RuntimeError("P2 encoding did not produce its condition latent")
                if self._uses_lawam_action_recipe():
                    if "z_lawam" not in prepared_batch:
                        raise KeyError("LaWAM-action P2 requires teacher z_lawam")
                else:
                    if "z_effect" not in prepared_batch:
                        raise KeyError("P2 requires teacher z_effect")
                    loss_effect_distill = F.smooth_l1_loss(
                        shared.predicted_effect.float(),
                        prepared_batch["z_effect"].float().detach(),
                        beta=0.1,
                    )
                if self._phased_stage2 and self.stage2_phase in {"P", "E"}:
                    plan_losses = self._compute_stage2_plan_losses(
                        prepared_batch=prepared_batch,
                        shared=shared,
                    )
                    loss_action_distill = plan_losses["loss_plan"]
                elif self._phased_stage2:
                    # Phase W deliberately does not execute the action predictor.
                    loss_action_distill = zero
                elif self.model_cfg.stage2_action_distribution == "gaussian_v2":
                    if shared.paired_tokens is None:
                        raise RuntimeError("P2 forward did not produce paired tokens")
                    if "z_action_logvar" not in prepared_batch:
                        raise KeyError("gaussian_v2 P2 requires teacher z_action_logvar")
                    if shared.paired_tokens.action_logvar is None:
                        raise RuntimeError("gaussian_v2 P2 did not predict action logvar")
                    teacher = DiagonalGaussian(
                        mean=prepared_batch["z_action"].float().detach(),
                        logvar=prepared_batch["z_action_logvar"].float().detach(),
                    )
                    prior = DiagonalGaussian(
                        mean=shared.paired_tokens.action.float(),
                        logvar=shared.paired_tokens.action_logvar.float(),
                    )
                    loss_action_distill = diagonal_gaussian_kl(teacher, prior)
                else:
                    if shared.paired_tokens is None or "z_action" not in prepared_batch:
                        raise KeyError("P2 requires predicted and teacher z_action")
                    loss_action_distill = F.smooth_l1_loss(
                        shared.paired_tokens.action.float(),
                        prepared_batch["z_action"].float().detach(),
                        beta=0.1,
                    )

        with _cuda_autocast(flow_stage_dtype):
            compute_flow = (
                not self._phased_stage2
                or self.stage2_phase == "E"
                or (self._release_residual_recipe and self.stage2_phase == "P")
            )
            loss_flow = (
                self._compute_flow_matching_loss(
                    prepared_batch=prepared_batch,
                    shared=shared,
                )
                if compute_flow
                else zero
            )
            if self._phased_stage2:
                loss_total = (
                    float(self.model_cfg.stage2_flow_weight) * loss_flow
                    + float(self.model_cfg.stage2_effect_distill_weight)
                    * loss_effect_distill
                    + float(self.model_cfg.stage2_action_distill_weight)
                    * loss_action_distill
                    + float(self.model_cfg.stage2_dino_weight) * loss_perceptual
                    + float(getattr(self.model_cfg, "stage2_lawam_distill_weight", 0.0))
                    * loss_lawam_distill
                )
            else:
                loss_total = (
                    loss_flow
                    + self.model_cfg.perceptual_weight * loss_perceptual
                    + self.model_cfg.lam_encoder_distill_weight * loss_distill
                    + self.model_cfg.stage2_effect_distill_weight * loss_effect_distill
                    + self.model_cfg.stage2_action_distill_weight * loss_action_distill
                )

        return {
            "loss_flow": loss_flow,
            "loss_perceptual": loss_perceptual,
            "loss_dino": loss_perceptual,
            "loss_distill": loss_distill,
            "loss_lawam_distill": loss_lawam_distill,
            "loss_effect_distill": loss_effect_distill,
            "loss_action_distill": loss_action_distill,
            **plan_losses,
            "loss_vlm": zero,
            "loss_total": loss_total,
            **self._last_stage2_residual_metrics,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        batch: LatentWorldPolicyInferBatch,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        return_intermediates: bool = False,
        return_padded: bool = False,
        action_seeds: Optional[Sequence[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyInferBatch,
            self._prepare_infer_batch(batch=batch),
        )

        if guidance_scale is None:
            guidance_scale = float(self.flow.config.cfg_guidance_scale)
        if num_inference_steps is None:
            num_inference_steps = int(self.flow.config.num_inference_steps)

        shared = self._run_shared_encoding_infer(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.predict_action",
            lam_features_with_no_grad=False,
        )
        attn_flow = prepared_batch["attention_mask"] == 1

        with _cuda_autocast(flow_stage_dtype):
            plan_tokens = None
            plan_mask = None
            effect_tokens = None
            effect_mask = None
            action_residual = None
            if shared.paired_tokens is not None:
                if self._release_residual_recipe:
                    if self.stage2_action_residual is None:
                        raise RuntimeError("action residual projection was not initialized")
                    if self.action_injection_enabled():
                        action_residual = self.stage2_action_residual(
                            shared.paired_tokens.action
                        )
                else:
                    if self.paired_expert_adapters is None:
                        raise RuntimeError("paired Expert adapters were not initialized")
                    plan_tokens, plan_mask, effect_tokens, effect_mask = self.paired_expert_adapters(
                        shared.paired_tokens.effect,
                        shared.paired_tokens.action,
                    )
            actions = self.flow.sample_actions_cfg(
                h_t=shared.h_t,
                h_t1_star=shared.h_t1_pred,
                h_vlm=shared.h_vlm,
                state=prepared_batch["state"],
                state_mask=prepared_batch["state_mask"],
                action_hz=prepared_batch["action_hz"],
                embodiment_id=prepared_batch["embodiment_id"],
                cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                attention_mask=attn_flow,
                return_padded=bool(return_padded),
                plan_tokens=plan_tokens,
                plan_token_mask=plan_mask,
                effect_tokens=effect_tokens,
                effect_token_mask=effect_mask,
                action_seeds=action_seeds,
                action_residual=action_residual,
            )

        if not return_intermediates:
            return actions

        num_tokens = shared.h_t_original.shape[1]
        hw = int(num_tokens ** 0.5)
        if hw * hw != num_tokens:
            hw = 16
        intermediates = {
            "h_t": shared.h_t_original.detach().cpu(),
            "h_t1_pred": shared.h_t1_pred.detach().cpu(),
            "vision_tokens_hw": (hw, hw),
        }
        return actions, intermediates
