"""Validated W/P/E contracts for paired-latent Stage-2 training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from latent_action_model.libero_stage2.runtime_contract import (
    LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT,
    LiberoStage2TemporalContract,
    is_libero_04s_adapter_temporal_contract,
    load_libero_stage2_latent_contract,
    resolve_libero_stage2_temporal_contract,
)
from latent_action_model.bridge_stage2.runtime_contract import (
    load_bridge_stage2_latent_contract,
)
from latent_action_model.bridge_stage2.temporal import (
    BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT,
    is_bridge_stage2_temporal_contract,
)
from latent_action_model.robotwin_stage2.paired import (
    ACTION_HEAD_ARCHITECTURES,
    BRIDGE_B0_RESIDUAL_RECIPE,
    PRETRAIN_RESIDUAL_RECIPE,
    RELEASE_RESIDUAL_RECIPE,
    RESIDUAL_RECIPES,
    ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
    ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
    ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
    ROBOTWIN_SFT25_RESIDUAL_RECIPE,
)
from latent_action_model.robotwin_stage2.runtime_contract import (
    load_robotwin_stage2_latent_contract,
)
from latent_action_model.robotwin_stage2.temporal import (
    ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT,
    is_robotwin_stage2_temporal_contract,
    resolve_stage2_temporal_contract,
)


class Stage2Phase(str, Enum):
    """The three independently checkpointed Stage-2 optimization phases."""

    W = "W"
    P = "P"
    E = "E"

    @classmethod
    def parse(cls, value: str | "Stage2Phase") -> "Stage2Phase":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"stage2_phase must be one of {[phase.value for phase in cls]}, got {value!r}"
            ) from exc


@dataclass(frozen=True)
class Stage2LossWeights:
    flow: float
    effect_distill: float
    action_distill: float
    dino: float
    lawam_distill: float = 0.0
    action_kl: float = 1.0
    action_mean: float = 0.5
    action_trajectory: float = 0.2

    def __post_init__(self) -> None:
        values = asdict(self)
        invalid = {
            name: value
            for name, value in values.items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        }
        if invalid:
            raise ValueError(f"Stage-2 loss weights must be finite and non-negative: {invalid}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2PhaseContract:
    phase: Stage2Phase
    losses: Stage2LossWeights
    train_lawm: bool
    train_action_predictor: bool
    train_expert: bool
    default_vlm_lr: float
    expected_parent_phase: Stage2Phase | None
    training_recipe: str = ""
    action_head_architecture: str = "transformer"
    action_teacher_transfer: str | None = None
    action_injection_enabled: bool = True


PHASE_CONTRACTS: Mapping[Stage2Phase, Stage2PhaseContract] = {
    Stage2Phase.W: Stage2PhaseContract(
        phase=Stage2Phase.W,
        losses=Stage2LossWeights(
            flow=0.0,
            effect_distill=1.0,
            action_distill=0.0,
            dino=1.0,
        ),
        train_lawm=True,
        train_action_predictor=False,
        train_expert=False,
        default_vlm_lr=1.0e-5,
        expected_parent_phase=None,
    ),
    Stage2Phase.P: Stage2PhaseContract(
        phase=Stage2Phase.P,
        losses=Stage2LossWeights(
            flow=0.0,
            effect_distill=0.25,
            action_distill=1.0,
            dino=0.25,
        ),
        train_lawm=False,
        train_action_predictor=True,
        train_expert=False,
        default_vlm_lr=5.0e-6,
        expected_parent_phase=Stage2Phase.W,
    ),
    Stage2Phase.E: Stage2PhaseContract(
        phase=Stage2Phase.E,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.1,
            action_distill=0.1,
            dino=0.1,
        ),
        train_lawm=True,
        train_action_predictor=True,
        train_expert=True,
        default_vlm_lr=2.0e-6,
        expected_parent_phase=Stage2Phase.P,
    ),
}

PHASE_SCHEDULES: Mapping[Stage2Phase, tuple[int, int]] = {
    Stage2Phase.W: (5_000, 300),
    Stage2Phase.P: (20_000, 1_000),
    Stage2Phase.E: (10_000, 500),
}
PHASE_SCHEDULES_04S: Mapping[Stage2Phase, tuple[int, int]] = {
    Stage2Phase.W: (3_571, 214),
    Stage2Phase.P: (14_286, 857),
    Stage2Phase.E: (7_143, 429),
}
RELEASE_RESIDUAL_PHASE_CONTRACTS: Mapping[Stage2Phase, Stage2PhaseContract] = {
    Stage2Phase.W: Stage2PhaseContract(
        phase=Stage2Phase.W,
        losses=Stage2LossWeights(
            flow=0.0,
            effect_distill=1.0,
            action_distill=0.0,
            dino=1.0,
        ),
        train_lawm=True,
        train_action_predictor=False,
        train_expert=False,
        default_vlm_lr=0.0,
        expected_parent_phase=None,
        training_recipe=RELEASE_RESIDUAL_RECIPE,
    ),
    Stage2Phase.P: Stage2PhaseContract(
        phase=Stage2Phase.P,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.1,
            action_distill=0.5,
            dino=0.1,
        ),
        train_lawm=False,
        train_action_predictor=True,
        train_expert=False,
        default_vlm_lr=1.0e-5,
        expected_parent_phase=Stage2Phase.W,
        training_recipe=RELEASE_RESIDUAL_RECIPE,
    ),
    Stage2Phase.E: Stage2PhaseContract(
        phase=Stage2Phase.E,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.1,
            action_distill=0.25,
            dino=0.1,
        ),
        train_lawm=True,
        train_action_predictor=True,
        train_expert=True,
        default_vlm_lr=1.0e-5,
        expected_parent_phase=Stage2Phase.P,
        training_recipe=RELEASE_RESIDUAL_RECIPE,
    ),
}
RELEASE_RESIDUAL_PHASE_SCHEDULES: Mapping[Stage2Phase, tuple[int, int]] = {
    Stage2Phase.W: (2_000, 120),
    Stage2Phase.P: (8_000, 480),
    Stage2Phase.E: (15_000, 900),
}
PRETRAIN_RESIDUAL_PHASE_CONTRACTS: Mapping[Stage2Phase, Stage2PhaseContract] = {
    Stage2Phase.W: Stage2PhaseContract(
        phase=Stage2Phase.W,
        losses=Stage2LossWeights(
            flow=0.0,
            effect_distill=1.0,
            action_distill=0.0,
            dino=1.0,
        ),
        train_lawm=True,
        train_action_predictor=False,
        train_expert=False,
        default_vlm_lr=0.0,
        expected_parent_phase=None,
        training_recipe=PRETRAIN_RESIDUAL_RECIPE,
    ),
    Stage2Phase.P: Stage2PhaseContract(
        phase=Stage2Phase.P,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.1,
            action_distill=0.5,
            dino=0.1,
        ),
        train_lawm=False,
        train_action_predictor=True,
        train_expert=False,
        default_vlm_lr=5.0e-5,
        expected_parent_phase=Stage2Phase.W,
        training_recipe=PRETRAIN_RESIDUAL_RECIPE,
    ),
    Stage2Phase.E: Stage2PhaseContract(
        phase=Stage2Phase.E,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.1,
            action_distill=0.25,
            dino=0.1,
        ),
        train_lawm=True,
        train_action_predictor=True,
        train_expert=True,
        default_vlm_lr=1.0e-4,
        expected_parent_phase=Stage2Phase.P,
        training_recipe=PRETRAIN_RESIDUAL_RECIPE,
    ),
}
PRETRAIN_RESIDUAL_PHASE_SCHEDULES: Mapping[Stage2Phase, tuple[int, int]] = {
    Stage2Phase.W: (4_000, 240),
    Stage2Phase.P: (16_000, 960),
    Stage2Phase.E: (30_000, 1_800),
}
LAWAM_ACTION_PHASE_CONTRACTS: Mapping[Stage2Phase, Stage2PhaseContract] = {
    Stage2Phase.P: Stage2PhaseContract(
        phase=Stage2Phase.P,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.0,
            action_distill=0.5,
            dino=0.1,
            lawam_distill=0.1,
        ),
        train_lawm=False,
        train_action_predictor=True,
        train_expert=False,
        default_vlm_lr=5.0e-5,
        expected_parent_phase=None,
        training_recipe=ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
    ),
    Stage2Phase.E: Stage2PhaseContract(
        phase=Stage2Phase.E,
        losses=Stage2LossWeights(
            flow=1.0,
            effect_distill=0.0,
            action_distill=0.25,
            dino=0.1,
            lawam_distill=0.1,
        ),
        train_lawm=True,
        train_action_predictor=True,
        train_expert=True,
        default_vlm_lr=1.0e-4,
        expected_parent_phase=Stage2Phase.P,
        training_recipe=ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
    ),
}
LAWAM_ACTION_PHASE_SCHEDULES: Mapping[Stage2Phase, tuple[int, int]] = {
    Stage2Phase.P: (20_000, 1_200),
    Stage2Phase.E: (50_000, 3_000),
}

STAGE2_ACTION_STATE_SOURCES = frozenset(
    {"actionvae14_padded32", "robotwin_actionvae14_padded32", "starvla32"}
)


STAGE2_CHECKPOINT_CONTRACT_SCHEMA = "lawam-stage2-phase-checkpoint-v1"
STAGE2_MODULE_STRUCTURE = "lawam-stage2-full-wpe-dense-v2"
RELEASE_RESIDUAL_MODULE_STRUCTURE = "lawam-stage2-release-residual-dense-v1"
LAWAM_ACTION_MODULE_STRUCTURE = "lawam-stage2-lawam-action-pe-dense-v1"
# These are portable identity suffixes, not site-specific checkpoint locations.
LIBERO_SFT_RELEASE_CHECKPOINT = Path(
    "lawam_libero_sft_release/final_model/pytorch_model.pt"
)
LIBERO_PRETRAIN_CHECKPOINT = Path("lawam_pretrain/final_model/pytorch_model.pt")
LIBERO_PRETRAIN_CHECKPOINT_BYTES = 7_174_214_645
LIBERO_PRETRAIN_CHECKPOINT_SHA256 = (
    "9fe985abf0ef3d0868f9cc330f7248ade01aaf5891f94528a593eb17d6ba14cc"
)
ROBOTWIN_SFT_RELEASE_CHECKPOINT_BYTES = 7_174_214_645
ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256 = (
    "a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb"
)
ROBOTWIN_12S_EFFECT_CHECKPOINT_SHA256 = (
    "66e872b5f6a67537b6cf33ad4d0b7e65652057e3e44a726f0a00c8205dd5c9a0"
)
ROBOTWIN_12S_EFFECT_CONFIG_SHA256 = (
    "1a1094f79c93c8fb0c77b32db29fc221eaefce932c3da3cac4cb74ea25b08bae"
)
ROBOTWIN_12S_DENSE_S1_ACTION_CONFIG_SHA256 = (
    "d2708adc9d90721adcb0f4a65dee52d4473eafe2ec4d1dd1281af2495f8bbbc1"
)
ROBOTWIN_12S_DENSE_S1_ACTION_CHECKPOINT_SHA256 = (
    "fac6da745cff1706bc9037e54ba799fa71557f5847fdc1b20e28ad458e7d6bc6"
)
ROBOTWIN_12S_LAWAM_ACTION_CONFIG_SHA256 = (
    "0dde8fcd740a3192ae8d4845c9a9f564e8e4bb203155250ce4bbefb398b43788"
)
ROBOTWIN_12S_LAWAM_ACTION_CHECKPOINT_SHA256 = (
    "48a3fd7984a8b9edec0290318b19c3e8023e51ce0e720f8ecfd55299d739c2ef"
)
ROBOTWIN_LAWAM_IDM_CHECKPOINT_SHA256 = (
    "57c31dcba13c2998c4e866cdcfc427f14ae3c1b762a586778276e2100b48af88"
)
ROBOTWIN_LAWAM_IDM_CONFIG_SHA256 = (
    "e5f98738d1c6dc8d8643d5a73d44ebd46385fbb2fc4031d46a02ad384a353449"
)
BRIDGE_12S_EFFECT_CHECKPOINT_SHA256 = (
    "bbb2aae91353c2d21009adc9027281b5d42c7c1c539d5acddabce3ebca0867b0"
)
BRIDGE_12S_EFFECT_CONFIG_SHA256 = (
    "eb939d8428573ba0ce14af415c7b6ee46fc6d6b680e78998598c69ae5b0ede63"
)
BRIDGE_12S_ACTION_CHECKPOINT_SHA256 = (
    "5edf2556414cfcc1f645b58c4d3767b7e746dd2e665ad451c84e700f6f7c1c29"
)
BRIDGE_12S_ACTION_CONFIG_SHA256 = (
    "640b7ee116e05fcea961ff7f4444777659806a6f5b687650dbe6f7e7eaff6b18"
)
BRIDGE_B0_SELECTION_TAG = "bridge_b0_25k_v1"
STAGE2_SELECTION_SCHEMA = "lawam-stage2-validation-selection-v1"
STAGE2_LAUNCH_PREFLIGHT_SCHEMA = "lawam-libero-stage2-launch-preflight-v1"
STAGE2_PREFLIGHT_RECEIPT_ENV = "LAWAM_STAGE2_PREFLIGHT_RECEIPT"


_LOSS_FIELDS = {
    "flow": "stage2_flow_weight",
    "effect_distill": "stage2_effect_distill_weight",
    "action_distill": "stage2_action_distill_weight",
    "dino": "stage2_dino_weight",
    "lawam_distill": "stage2_lawam_distill_weight",
    "action_kl": "stage2_action_kl_weight",
    "action_mean": "stage2_action_mean_weight",
    "action_trajectory": "stage2_action_trajectory_weight",
}


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(mapping, key, default)


def resolve_stage2_config_temporal_contract(cfg: Any) -> Any:
    """Resolve matching action/data markers, preserving legacy missing markers."""

    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    data_cfg = _get(_get(cfg, "datasets", None), "vla_data", None)
    action_raw = _get(action_cfg, "stage2_temporal_contract", None)
    data_raw = _get(data_cfg, "stage2_temporal_contract", None)
    action_contract = resolve_stage2_temporal_contract(action_raw)
    data_contract = resolve_stage2_temporal_contract(data_raw)
    if action_contract.identifier != data_contract.identifier:
        raise ValueError(
            "framework.action_model and datasets.vla_data stage2_temporal_contract "
            "must match exactly: "
            f"model={action_contract.identifier!r}, data={data_contract.identifier!r}"
        )
    if (
        is_libero_04s_adapter_temporal_contract(action_contract)
        or is_bridge_stage2_temporal_contract(action_contract)
    ):
        if action_raw != action_contract.identifier or data_raw != action_contract.identifier:
            raise ValueError(
                "this exact-row temporal contract requires an explicit marker "
                "in both framework.action_model and datasets.vla_data"
            )
    return action_contract


def resolve_stage2_phase_schedule(
    phase: Stage2Phase | str,
    temporal_contract: Any,
    training_recipe: str | None = None,
) -> tuple[int, int]:
    resolved_phase = Stage2Phase.parse(phase)
    temporal = resolve_stage2_temporal_contract(temporal_contract)
    recipe = str(training_recipe or "").strip()
    if recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE:
        if not is_robotwin_stage2_temporal_contract(temporal):
            raise ValueError(f"{recipe} requires the RoboTwin official 1.2s temporal contract")
        if resolved_phase not in LAWAM_ACTION_PHASE_SCHEDULES:
            raise ValueError(f"{recipe} defines only P/E phases")
        return LAWAM_ACTION_PHASE_SCHEDULES[resolved_phase]
    if recipe in RESIDUAL_RECIPES:
        if recipe == BRIDGE_B0_RESIDUAL_RECIPE:
            if not is_bridge_stage2_temporal_contract(temporal):
                raise ValueError(
                    f"{recipe} requires the Bridge official 1.2s temporal contract"
                )
            return RELEASE_RESIDUAL_PHASE_SCHEDULES[resolved_phase]
        if recipe in {
            ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
            ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
            ROBOTWIN_SFT25_RESIDUAL_RECIPE,
        }:
            if not is_robotwin_stage2_temporal_contract(temporal):
                raise ValueError(
                    f"{recipe} requires the RoboTwin official 1.2s temporal contract"
                )
            return PRETRAIN_RESIDUAL_PHASE_SCHEDULES[resolved_phase]
        if not is_libero_04s_adapter_temporal_contract(temporal):
            raise ValueError(
                f"{recipe} requires the official 0.4s temporal contract"
            )
        schedules = (
            PRETRAIN_RESIDUAL_PHASE_SCHEDULES
            if recipe == PRETRAIN_RESIDUAL_RECIPE
            else RELEASE_RESIDUAL_PHASE_SCHEDULES
        )
        return schedules[resolved_phase]
    if recipe:
        raise ValueError(f"unsupported stage2_training_recipe={recipe!r}")
    schedules = (
        PHASE_SCHEDULES_04S
        if is_libero_04s_adapter_temporal_contract(temporal)
        else PHASE_SCHEDULES
    )
    return schedules[resolved_phase]


def resolve_stage2_phase_contract(action_model_cfg: Any) -> Stage2PhaseContract | None:
    """Resolve and validate the action-model phase/loss contract.

    Missing ``stage2_phase`` preserves non-phased release and legacy ablation runs.
    Active phases may tune enabled loss coefficients, but may not enable a loss that
    the phase is defined to exclude.
    """

    raw_phase = _get(action_model_cfg, "stage2_phase", None)
    if raw_phase is None or str(raw_phase).strip() == "":
        return None
    phase = Stage2Phase.parse(raw_phase)
    training_recipe = str(
        _get(action_model_cfg, "stage2_training_recipe", "") or ""
    ).strip()
    if training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE:
        if phase not in LAWAM_ACTION_PHASE_CONTRACTS:
            raise ValueError(f"{training_recipe} defines only P/E phases")
        defaults = LAWAM_ACTION_PHASE_CONTRACTS[phase]
    elif training_recipe in {RELEASE_RESIDUAL_RECIPE, BRIDGE_B0_RESIDUAL_RECIPE}:
        defaults = RELEASE_RESIDUAL_PHASE_CONTRACTS[phase]
    elif training_recipe in {
        PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
    }:
        defaults = PRETRAIN_RESIDUAL_PHASE_CONTRACTS[phase]
    elif training_recipe:
        raise ValueError(f"unsupported stage2_training_recipe={training_recipe!r}")
    else:
        defaults = PHASE_CONTRACTS[phase]
    default_losses = defaults.losses.to_dict()
    resolved_values: dict[str, float] = {}
    for loss_name, field_name in _LOSS_FIELDS.items():
        raw_value = _get(action_model_cfg, field_name, default_losses[loss_name])
        try:
            resolved_values[loss_name] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric, got {raw_value!r}") from exc
    losses = Stage2LossWeights(**resolved_values)

    for loss_name in (
        "flow",
        "effect_distill",
        "action_distill",
        "dino",
        "lawam_distill",
    ):
        default_value = float(default_losses[loss_name])
        actual_value = float(getattr(losses, loss_name))
        if default_value == 0.0 and actual_value != 0.0:
            raise ValueError(
                f"Phase {phase.value} excludes {loss_name}; "
                f"{_LOSS_FIELDS[loss_name]} must be 0, got {actual_value}"
            )
        if default_value > 0.0 and actual_value <= 0.0:
            raise ValueError(
                f"Phase {phase.value} requires {loss_name}; "
                f"{_LOSS_FIELDS[loss_name]} must be positive"
            )
    if phase in {Stage2Phase.P, Stage2Phase.E} and losses.action_kl <= 0.0:
        raise ValueError("P/E require stage2_action_kl_weight > 0")

    architecture = str(_get(action_model_cfg, "stage2_action_head_architecture", "transformer"))
    if architecture not in ACTION_HEAD_ARCHITECTURES:
        raise ValueError(f"Unsupported action head architecture: {architecture!r}")
    injection_enabled = _get(action_model_cfg, "stage2_action_injection_enabled", True)
    if not isinstance(injection_enabled, bool):
        raise ValueError("stage2_action_injection_enabled must be a boolean")
    if not injection_enabled and (
        training_recipe not in RESIDUAL_RECIPES or _get(action_model_cfg, "stage2_arm", None) != "p2"
    ):
        raise ValueError("Action auxiliary-only mode requires a phased P2 release-residual recipe")
    return Stage2PhaseContract(
        phase=phase,
        losses=losses,
        train_lawm=defaults.train_lawm,
        train_action_predictor=defaults.train_action_predictor,
        train_expert=defaults.train_expert,
        default_vlm_lr=defaults.default_vlm_lr,
        expected_parent_phase=defaults.expected_parent_phase,
        training_recipe=training_recipe,
        action_head_architecture=architecture,
        action_teacher_transfer=_get(action_model_cfg, "stage2_action_teacher_transfer", None),
        action_injection_enabled=injection_enabled,
    )


def resolve_stage2_module_structure(contract: Stage2PhaseContract) -> str:
    if contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE:
        structure = LAWAM_ACTION_MODULE_STRUCTURE
    elif contract.training_recipe in RESIDUAL_RECIPES:
        structure = RELEASE_RESIDUAL_MODULE_STRUCTURE
    else:
        structure = STAGE2_MODULE_STRUCTURE
    if contract.action_head_architecture != "transformer":
        structure += "-" + contract.action_head_architecture
    if contract.action_teacher_transfer:
        from latent_action_model.libero_stage2.robotwin_action_transfer import checked_transfer
        checked_transfer(contract.action_teacher_transfer)
        structure += "-robotwin-action-transfer-v1"
    if not contract.action_injection_enabled:
        structure += "-action-aux-only-v1"
    return structure


def _resolve_vlm_lr(learning_rate_cfg: Any) -> float | None:
    group = _get(learning_rate_cfg, "vlm", None)
    if group is None:
        return None
    raw_lr = _get(group, "lr", group if isinstance(group, (int, float)) else None)
    return None if raw_lr is None else float(raw_lr)


def validate_stage2_training_config(cfg: Any) -> Stage2PhaseContract | None:
    """Fail fast on a physically inconsistent phased training configuration."""

    framework_cfg = _get(cfg, "framework", None)
    action_model_cfg = _get(framework_cfg, "action_model", None)
    contract = resolve_stage2_phase_contract(action_model_cfg)
    if contract is None:
        return None
    temporal = resolve_stage2_config_temporal_contract(cfg)
    is_lawam_action = (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    )

    required_action_values = {
        "stage2_arm": "p2",
        "stage2_action_distribution": "dense_transition_gaussian_v3",
        "stage2_query_scheme": (
            "act_lawam_flow_plan_v1"
            if is_lawam_action
            else "act_effect_flow_plan_v1"
        ),
        "future_prediction": True,
        "detach_future_feature": True,
        "stage2_horizon_condition_dim": 3,
        "stage2_horizon_seconds": temporal.duration_seconds,
    }
    if is_lawam_action:
        required_action_values["stage2_condition_source"] = "lawam_idm"
        unexpected_effect_fields = {
            name: _get(action_model_cfg, name, None)
            for name in (
                "stage2_effect_teacher_checkpoint_path",
                "stage2_effect_teacher_checkpoint_sha256",
                "stage2_effect_teacher_config_path",
                "stage2_effect_teacher_config_sha256",
            )
            if _get(action_model_cfg, name, None) not in (None, "")
        }
        if unexpected_effect_fields:
            raise ValueError(
                "LaWAM-action Stage-2 runs without an Effect teacher: "
                f"unexpected_fields={unexpected_effect_fields}"
            )
    action_mismatches = {
        name: (_get(action_model_cfg, name, None), expected)
        for name, expected in required_action_values.items()
        if _get(action_model_cfg, name, None) != expected
    }
    if action_mismatches:
        raise ValueError(f"Phased Stage-2 action-model contract mismatch: {action_mismatches}")
    flow_horizon = float(
        _get(_get(action_model_cfg, "flow_cfg", None), "horizon_sec", -1.0)
    )
    if flow_horizon != temporal.duration_seconds:
        raise ValueError(
            "framework.action_model.flow_cfg.horizon_sec must match the selected "
            f"temporal contract: got={flow_horizon}, expected={temporal.duration_seconds}"
        )
    action_state_source = str(
        _get(action_model_cfg, "stage2_action_state_source", "")
    ).strip()
    if action_state_source not in STAGE2_ACTION_STATE_SOURCES:
        raise ValueError(
            "stage2_action_state_source remains unresolved; choose exactly one of "
            f"{sorted(STAGE2_ACTION_STATE_SOURCES)}, got {action_state_source!r}"
        )
    data_cfg = _get(_get(cfg, "datasets", None), "vla_data", None)
    transfer = _get(action_model_cfg, "stage2_action_teacher_transfer", None)
    if transfer != _get(data_cfg, "stage2_action_teacher_transfer", None):
        raise ValueError("model/dataset ActionVAE transfer markers differ")
    dataset_state_source = str(
        _get(data_cfg, "stage2_action_state_source", "")
    ).strip()
    if dataset_state_source not in STAGE2_ACTION_STATE_SOURCES:
        raise ValueError(
            "datasets.vla_data.stage2_action_state_source remains unresolved; choose exactly "
            f"one of {sorted(STAGE2_ACTION_STATE_SOURCES)}, got {dataset_state_source!r}"
        )
    if dataset_state_source != action_state_source:
        raise ValueError(
            "framework.action_model and datasets.vla_data stage2_action_state_source must "
            f"match exactly: model={action_state_source!r}, dataset={dataset_state_source!r}"
        )
    expected_state_source = (
        "robotwin_actionvae14_padded32"
        if is_robotwin_stage2_temporal_contract(temporal)
        else "actionvae14_padded32"
    )
    if action_state_source != expected_state_source:
        raise ValueError(
            f"{temporal.identifier} requires stage2_action_state_source="
            f"{expected_state_source!r}, got {action_state_source!r}"
        )
    if float(_get(data_cfg, "sec_chunk", -1.0)) != temporal.duration_seconds:
        raise ValueError(
            "datasets.vla_data.sec_chunk must match the selected temporal contract: "
            f"got={_get(data_cfg, 'sec_chunk', None)!r}, expected={temporal.duration_seconds}"
        )
    explicit_logvar_bounds = [
        name
        for name in ("stage2_action_logvar_min", "stage2_action_logvar_max")
        if _get(action_model_cfg, name, None) is not None
    ]
    if explicit_logvar_bounds:
        raise ValueError(
            "Stage-2 predictor logvar bounds must come from the selected ActionVAE "
            f"checkpoint, not YAML fields: {explicit_logvar_bounds}"
        )

    required_paths = [
        "stage2_action_decoder_checkpoint_path",
        "stage2_action_decoder_config_path",
        "stage2_action_decoder_data_root",
        "stage2_latent_cache_root",
        "stage2_latent_normalization_path",
    ]
    required_paths.extend(
        (
            "stage2_lawam_condition_checkpoint_path",
            "stage2_lawam_condition_config_path",
        )
        if is_lawam_action
        else (
            "stage2_effect_teacher_checkpoint_path",
            "stage2_effect_teacher_config_path",
        )
    )
    missing_paths = [
        name
        for name in required_paths
        if not str(_get(action_model_cfg, name, "")).strip()
        or str(_get(action_model_cfg, name, "")).startswith("<REQUIRED_")
    ]
    required_hashes = [
        "stage2_action_decoder_checkpoint_sha256",
        "stage2_action_decoder_config_sha256",
    ]
    required_hashes.extend(
        (
            "stage2_lawam_condition_checkpoint_sha256",
            "stage2_lawam_condition_config_sha256",
        )
        if is_lawam_action
        else (
            "stage2_effect_teacher_checkpoint_sha256",
            "stage2_effect_teacher_config_sha256",
        )
    )
    invalid_hashes = []
    for name in required_hashes:
        digest = str(_get(action_model_cfg, name, "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            invalid_hashes.append(name)
    if missing_paths or invalid_hashes:
        raise ValueError(
            "Phased Stage-2 teacher/normalization contract is incomplete: "
            f"missing_paths={missing_paths}, invalid_hashes={invalid_hashes}"
        )
    if (
        is_robotwin_stage2_temporal_contract(temporal)
        or is_bridge_stage2_temporal_contract(temporal)
    ):
        sidecar_binding = str(
            _get(action_model_cfg, "stage2_action_decoder_sidecar_binding", "exact")
        )
        allowed_bindings = {"contract"} if is_lawam_action else {"exact", "semantic"}
        if sidecar_binding not in allowed_bindings:
            raise ValueError(
                "stage2_action_decoder_sidecar_binding must be one of "
                f"{sorted(allowed_bindings)} for this recipe"
            )
        reference_root = str(
            _get(
                action_model_cfg,
                "stage2_action_decoder_sidecar_reference_root",
                "",
            )
        )
        if sidecar_binding == "semantic" and (
            not reference_root or reference_root.startswith("<REQUIRED_")
        ):
            raise ValueError(
                "semantic RoboTwin sidecar binding requires an explicit reference root"
            )
        if is_lawam_action and reference_root:
            raise ValueError(
                "lawam_idm contract binding does not use a byte-bound sidecar reference root"
            )

    trainer_cfg = _get(cfg, "trainer", None)
    max_train_steps = int(_get(trainer_cfg, "max_train_steps", -1))
    warmup_steps = int(_get(trainer_cfg, "num_warmup_steps", -1))
    expected_schedule = resolve_stage2_phase_schedule(
        contract.phase,
        temporal,
        contract.training_recipe,
    )
    if (max_train_steps, warmup_steps) != expected_schedule:
        raise ValueError(
            f"Phase {contract.phase.value} schedule must be max/warmup={expected_schedule}, "
            f"got {(max_train_steps, warmup_steps)}"
        )
    if int(_get(trainer_cfg, "eval_interval", -1)) != 500:
        raise ValueError("Phased Stage-2 requires eval_interval=500 for validation selection")
    if int(_get(trainer_cfg, "save_interval", -1)) != 500:
        raise ValueError("Phased Stage-2 requires save_interval=500 while retaining final last")
    process_checkpoint_keep_limit = _get(
        trainer_cfg, "process_checkpoint_keep_limit", None
    )
    if contract.training_recipe in {
        PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
    }:
        if process_checkpoint_keep_limit != 1:
            raise ValueError(
                "pretrain-residual Stage-2 requires "
                "trainer.process_checkpoint_keep_limit=1"
            )
    elif process_checkpoint_keep_limit is not None and int(
        process_checkpoint_keep_limit
    ) < 1:
        raise ValueError("trainer.process_checkpoint_keep_limit must be >= 1")
    if contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE:
        if str(_get(cfg, "wandb_mode", "")).strip().lower() != "online":
            raise ValueError("LaWAM-action training requires wandb_mode=online")
        if _get(cfg, "wandb_required", None) is not True:
            raise ValueError("LaWAM-action training requires wandb_required=true")
        preserve_steps = tuple(
            int(step)
            for step in (_get(trainer_cfg, "preserve_checkpoint_steps", []) or [])
        )
        expected_preserve_steps = (
            (30_000,) if contract.phase is Stage2Phase.E else ()
        )
        if preserve_steps != expected_preserve_steps:
            raise ValueError(
                "LaWAM-action checkpoint preservation contract mismatch: "
                f"expected {list(expected_preserve_steps)}, got {list(preserve_steps)}"
            )
        if any(
            step <= 0
            or step >= max_train_steps
            or step % int(_get(trainer_cfg, "save_interval", -1)) != 0
            for step in preserve_steps
        ):
            raise ValueError(
                "preserve_checkpoint_steps must be positive save-interval multiples "
                "strictly below max_train_steps"
            )
    freeze_cfg = _get(trainer_cfg, "freeze", None)
    if contract.training_recipe in RESIDUAL_RECIPES:
        if contract.phase is Stage2Phase.W:
            expected_freeze = {
                "freeze_vision_backbone": True,
                "freeze_llm_backbone": True,
                "keep_llm_first_n_layers": 16,
                "freeze_last_llm_layer": True,
                "freeze_embedding": True,
                "unfreeze_vision_merger": False,
                "unfreeze_llm_last_n_layers": 0,
            }
        else:
            expected_freeze = {
                "freeze_vision_backbone": False,
                "freeze_llm_backbone": False,
                "keep_llm_first_n_layers": 16,
                "freeze_last_llm_layer": True,
                "freeze_embedding": True,
                "unfreeze_vision_merger": True,
                "unfreeze_llm_last_n_layers": -1,
            }
    else:
        expected_freeze = {
            "freeze_vision_backbone": True,
            "freeze_llm_backbone": True,
            "keep_llm_first_n_layers": 16,
            "freeze_last_llm_layer": True,
            "freeze_embedding": True,
            "unfreeze_vision_merger": False,
            "unfreeze_llm_last_n_layers": 2,
        }
    mismatches = {
        name: (_get(freeze_cfg, name, None), expected)
        for name, expected in expected_freeze.items()
        if _get(freeze_cfg, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"Phased Stage-2 freeze contract mismatch: {mismatches}")

    if contract.training_recipe in RESIDUAL_RECIPES:
        expected_residual = {
            Stage2Phase.W: {
                "stage2_residual_warmup_start_step": 0,
                "stage2_residual_teacher_prob_start": 0.0,
                "stage2_residual_teacher_prob_end": 0.0,
                "stage2_residual_teacher_ramp_steps": 1,
                "stage2_residual_predictor_grad_start_step": 0,
                "stage2_residual_condition_dropout": 0.0,
            },
            Stage2Phase.P: {
                "stage2_residual_warmup_start_step": (
                    12_000
                    if contract.training_recipe
                    in {
                        PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
                        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
                    }
                    else 6_000
                ),
                "stage2_residual_teacher_prob_start": 0.5,
                "stage2_residual_teacher_prob_end": 0.0,
                "stage2_residual_teacher_ramp_steps": (
                    2_000
                    if contract.training_recipe
                    in {
                        PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
                        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
                    }
                    else 1_000
                ),
                "stage2_residual_predictor_grad_start_step": (
                    14_000
                    if contract.training_recipe
                    in {
                        PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
                        ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
                        ROBOTWIN_SFT25_RESIDUAL_RECIPE,
                    }
                    else 7_000
                ),
                "stage2_residual_condition_dropout": 0.0,
            },
            Stage2Phase.E: {
                "stage2_residual_warmup_start_step": 0,
                "stage2_residual_teacher_prob_start": 0.0,
                "stage2_residual_teacher_prob_end": 0.0,
                "stage2_residual_teacher_ramp_steps": 1,
                "stage2_residual_predictor_grad_start_step": 0,
                "stage2_residual_condition_dropout": 0.1,
            },
        }[contract.phase]
        residual_mismatches = {
            name: (_get(action_model_cfg, name, None), expected)
            for name, expected in expected_residual.items()
            if _get(action_model_cfg, name, None) != expected
        }
        if residual_mismatches:
            raise ValueError(
                "release-residual curriculum contract mismatch: "
                f"{residual_mismatches}"
            )

    expected_lawm_flag = contract.train_lawm
    actual_lawm_flag = bool(_get(freeze_cfg, "unfreeze_lam_decoder", False))
    if actual_lawm_flag != expected_lawm_flag:
        raise ValueError(
            f"Phase {contract.phase.value} requires unfreeze_lam_decoder={expected_lawm_flag}, "
            f"got {actual_lawm_flag}"
        )

    configured_vlm_lr = _resolve_vlm_lr(_get(trainer_cfg, "learning_rate", None))
    if configured_vlm_lr != contract.default_vlm_lr:
        raise ValueError(
            f"Phase {contract.phase.value} VLM LR must be {contract.default_vlm_lr}, "
            f"got {configured_vlm_lr}"
        )

    init_checkpoint = str(_get(trainer_cfg, "pretrained_checkpoint", ""))
    if not init_checkpoint or init_checkpoint.startswith("<REQUIRED_"):
        raise ValueError(
            f"Phase {contract.phase.value} requires an explicit pretrained_checkpoint"
        )
    if (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
        and contract.phase is Stage2Phase.P
    ):
        configured_init_sha = str(
            _get(trainer_cfg, "pretrained_checkpoint_sha256", "")
        ).lower()
        if configured_init_sha != ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256:
            raise ValueError(
                "RoboTwin LaWAM-action Phase P must declare the official SFT release SHA256"
            )
    elif contract.training_recipe in RESIDUAL_RECIPES and contract.phase is Stage2Phase.W:
        if contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE:
            expected_init = None
        elif contract.training_recipe == ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE:
            configured_init_sha = str(
                _get(trainer_cfg, "pretrained_checkpoint_sha256", "")
            ).lower()
            if configured_init_sha != ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256:
                raise ValueError(
                    "RoboTwin SFT-release Phase A/W must declare the official release SHA256"
                )
            expected_init = None
        elif contract.training_recipe == ROBOTWIN_SFT25_RESIDUAL_RECIPE:
            expected_init = "lawam_robotwin_sft_release/final_model/pytorch_model.pt"
        elif contract.training_recipe in {
            PRETRAIN_RESIDUAL_RECIPE,
            ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE,
        }:
            expected_init = LIBERO_PRETRAIN_CHECKPOINT
        else:
            expected_init = LIBERO_SFT_RELEASE_CHECKPOINT
        if expected_init is not None and not Path(init_checkpoint).as_posix().endswith(
            str(expected_init)
        ):
            raise ValueError(
                f"{contract.training_recipe} Phase A/W must initialize from "
                f"{expected_init}"
            )

    raw_parent = _get(trainer_cfg, "stage2_parent_phase", None)
    expected_parent = contract.expected_parent_phase
    if expected_parent is None:
        if contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE:
            if str(raw_parent).strip().upper() != "B0":
                raise ValueError("Bridge Phase W must declare trainer.stage2_parent_phase=B0")
            parent_selection = str(_get(trainer_cfg, "stage2_parent_selection", ""))
            if not parent_selection or parent_selection.startswith("<REQUIRED_"):
                raise ValueError("Bridge Phase W requires the B0 validation-best selection")
        elif raw_parent not in (None, "", "none", "null"):
            raise ValueError(
                f"Phase {contract.phase.value} must start from its release checkpoint, "
                "not a Stage-2 parent phase"
            )
    else:
        if raw_parent is None:
            raise ValueError(
                f"Phase {contract.phase.value} requires trainer.stage2_parent_phase="
                f"{expected_parent.value}"
            )
        parent = Stage2Phase.parse(raw_parent)
        if parent is not expected_parent:
            raise ValueError(
                f"Phase {contract.phase.value} must initialize from Phase {expected_parent.value}, "
                f"got {parent.value}"
            )
        parent_selection = str(_get(trainer_cfg, "stage2_parent_selection", ""))
        if not parent_selection or parent_selection.startswith("<REQUIRED_"):
            raise ValueError(
                f"Phase {contract.phase.value} requires the parent validation-total-loss selection"
            )

    strict_init = bool(_get(trainer_cfg, "strict_finetune_init", False))
    load_pretrained_lam = bool(_get(trainer_cfg, "load_pretrained_lam", True))
    load_pretrained_flow = bool(_get(trainer_cfg, "load_pretrained_policy_flow", True))
    if not load_pretrained_flow:
        raise ValueError("All phased Stage-2 runs must preserve the pretrained/parent Expert flow")
    if (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
        and contract.phase is Stage2Phase.P
    ):
        if strict_init or not load_pretrained_lam:
            raise ValueError(
                "LaWAM-action Phase P requires relaxed release initialization with "
                "load_pretrained_lam=true"
            )
    elif contract.phase is Stage2Phase.W and contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE:
        if not strict_init or not load_pretrained_lam:
            raise ValueError(
                "Bridge Phase W must strict-load the complete B0 validation-best checkpoint"
            )
    elif contract.phase is Stage2Phase.W:
        if strict_init or load_pretrained_lam:
            raise ValueError(
                "Phase W requires strict_finetune_init=false and load_pretrained_lam=false "
                "so the selected Effect/LaWM teacher is not overwritten by release weights"
            )
    elif not strict_init or not load_pretrained_lam:
        raise ValueError(
            f"Phase {contract.phase.value} requires strict_finetune_init=true and "
            "load_pretrained_lam=true for an exact parent-phase transition"
        )
    return contract


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_path(value: Any) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def _validate_selected_teacher_binding(action_cfg: Any) -> None:
    """Lock every Stage-2 teacher entry point to its selected domain pair."""

    temporal = resolve_stage2_temporal_contract(
        _get(action_cfg, "stage2_temporal_contract", None)
    )
    from latent_action_model.libero_stage2 import robotwin_action_transfer as action_transfer
    transferred = action_transfer.checked_transfer(_get(action_cfg, "stage2_action_teacher_transfer", None))
    if transferred and temporal.identifier != "libero_original_plus_04s_adapter_v1":
        raise ValueError("RoboTwin-to-LIBERO transfer requires the Original+Plus Full temporal contract")
    recipe = str(_get(action_cfg, "stage2_training_recipe", "") or "")
    if recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE:
        path_pairs = (
            ("lam_ckpt_path", "stage2_lawam_condition_checkpoint_path"),
            ("lam_yaml_path", "stage2_lawam_condition_config_path"),
        )
        path_mismatches = {
            f"{left}!={right}": (
                str(_canonical_path(_get(action_cfg, left, None))),
                str(_canonical_path(_get(action_cfg, right, None))),
            )
            for left, right in path_pairs
            if _canonical_path(_get(action_cfg, left, None))
            != _canonical_path(_get(action_cfg, right, None))
        }
        unexpected_effect_fields = {
            name: _get(action_cfg, name, None)
            for name in (
                "stage2_effect_teacher_checkpoint_path",
                "stage2_effect_teacher_checkpoint_sha256",
                "stage2_effect_teacher_config_path",
                "stage2_effect_teacher_config_sha256",
            )
            if _get(action_cfg, name, None) not in (None, "")
        }
        expected_hashes = {
            "stage2_lawam_condition_checkpoint_sha256": (
                ROBOTWIN_LAWAM_IDM_CHECKPOINT_SHA256
            ),
            "stage2_lawam_condition_config_sha256": ROBOTWIN_LAWAM_IDM_CONFIG_SHA256,
            "stage2_action_decoder_checkpoint_sha256": (
                ROBOTWIN_12S_LAWAM_ACTION_CHECKPOINT_SHA256
            ),
            "stage2_action_decoder_config_sha256": (
                ROBOTWIN_12S_LAWAM_ACTION_CONFIG_SHA256
            ),
        }
        hash_mismatches = {
            name: (str(_get(action_cfg, name, "")).lower(), expected)
            for name, expected in expected_hashes.items()
            if str(_get(action_cfg, name, "")).lower() != expected
        }
        if path_mismatches or unexpected_effect_fields or hash_mismatches:
            raise ValueError(
                "LaWAM-action Stage-2 must use the official LaWAM IDM condition and "
                "selected LaWAM-conditioned ActionVAE without an Effect teacher: "
                f"path_mismatches={path_mismatches}, "
                f"unexpected_effect_fields={unexpected_effect_fields}, "
                f"hash_mismatches={hash_mismatches}"
            )
        return
    if is_bridge_stage2_temporal_contract(temporal):
        path_pairs = (
            ("lam_ckpt_path", "stage2_effect_teacher_checkpoint_path"),
            ("lam_yaml_path", "stage2_effect_teacher_config_path"),
        )
        path_mismatches = {
            f"{left}!={right}": (
                str(_canonical_path(_get(action_cfg, left, None))),
                str(_canonical_path(_get(action_cfg, right, None))),
            )
            for left, right in path_pairs
            if _canonical_path(_get(action_cfg, left, None))
            != _canonical_path(_get(action_cfg, right, None))
        }
        expected_hashes = {
            "stage2_effect_teacher_checkpoint_sha256": BRIDGE_12S_EFFECT_CHECKPOINT_SHA256,
            "stage2_effect_teacher_config_sha256": BRIDGE_12S_EFFECT_CONFIG_SHA256,
            "stage2_action_decoder_checkpoint_sha256": BRIDGE_12S_ACTION_CHECKPOINT_SHA256,
            "stage2_action_decoder_config_sha256": BRIDGE_12S_ACTION_CONFIG_SHA256,
        }
        hash_mismatches = {
            name: (str(_get(action_cfg, name, "")).lower(), expected)
            for name, expected in expected_hashes.items()
            if str(_get(action_cfg, name, "")).lower() != expected
        }
        if path_mismatches or hash_mismatches:
            raise ValueError(
                "Phased Bridge Stage-2 must use selected E00 Effect/LaWM and "
                "step-0007503 standard-normal ActionVAE: "
                f"path_mismatches={path_mismatches}, hash_mismatches={hash_mismatches}"
            )
        return
    if (
        is_robotwin_stage2_temporal_contract(temporal)
        or is_bridge_stage2_temporal_contract(temporal)
    ):
        dense_s1 = (
            temporal.identifier
            == ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT
        )
        path_pairs = (
            ("lam_ckpt_path", "stage2_effect_teacher_checkpoint_path"),
            ("lam_yaml_path", "stage2_effect_teacher_config_path"),
        )
        path_mismatches = {
            f"{left}!={right}": (
                str(_canonical_path(_get(action_cfg, left, None))),
                str(_canonical_path(_get(action_cfg, right, None))),
            )
            for left, right in path_pairs
            if _canonical_path(_get(action_cfg, left, None))
            != _canonical_path(_get(action_cfg, right, None))
        }
        expected_hashes = {
            "stage2_effect_teacher_checkpoint_sha256": (
                ROBOTWIN_12S_EFFECT_CHECKPOINT_SHA256
            ),
            "stage2_effect_teacher_config_sha256": ROBOTWIN_12S_EFFECT_CONFIG_SHA256,
        }
        if dense_s1:
            expected_hashes.update(
                stage2_action_decoder_checkpoint_sha256=(
                    ROBOTWIN_12S_DENSE_S1_ACTION_CHECKPOINT_SHA256
                ),
                stage2_action_decoder_config_sha256=(
                    ROBOTWIN_12S_DENSE_S1_ACTION_CONFIG_SHA256
                ),
            )
        hash_mismatches = {
            name: (str(_get(action_cfg, name, "")).lower(), expected)
            for name, expected in expected_hashes.items()
            if str(_get(action_cfg, name, "")).lower() != expected
        }
        if path_mismatches or hash_mismatches:
            raise ValueError(
                "Phased RoboTwin Stage-2 must use selected epoch-09 Effect/LaWM and "
                "one SHA-bound standard-normal 1.2s ActionVAE: "
                f"path_mismatches={path_mismatches}, hash_mismatches={hash_mismatches}"
            )
        return

    from latent_action_model.libero_stage2.teacher_bundle import (
        LIBERO_12S_ACTION_CHECKPOINT,
        LIBERO_12S_ACTION_CHECKPOINT_SHA256,
        LIBERO_12S_ACTION_CONFIG,
        LIBERO_12S_ACTION_CONFIG_SHA256,
        LIBERO_12S_ACTION_DATA_ROOT,
        LIBERO_12S_EFFECT_CHECKPOINT,
        LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        LIBERO_12S_EFFECT_CONFIG,
        LIBERO_12S_EFFECT_CONFIG_SHA256,
    )

    expected_paths = {
        "lam_ckpt_path": LIBERO_12S_EFFECT_CHECKPOINT.resolve(),
        "stage2_effect_teacher_checkpoint_path": LIBERO_12S_EFFECT_CHECKPOINT.resolve(),
        "lam_yaml_path": LIBERO_12S_EFFECT_CONFIG.resolve(),
        "stage2_effect_teacher_config_path": LIBERO_12S_EFFECT_CONFIG.resolve(),
        "stage2_action_decoder_checkpoint_path": LIBERO_12S_ACTION_CHECKPOINT.resolve(),
        "stage2_action_decoder_config_path": LIBERO_12S_ACTION_CONFIG.resolve(),
        "stage2_action_decoder_data_root": LIBERO_12S_ACTION_DATA_ROOT.resolve(),
    }
    if transferred:
        if _get(action_cfg, "stage2_condition_source", None) != "lawam_idm":
            raise ValueError("9.9 transfer requires official LaWAM IDM conditions")
        expected_paths.update(
            lam_ckpt_path=action_transfer.CONDITION_CHECKPOINT.resolve(),
            lam_yaml_path=action_transfer.CONDITION_CONFIG.resolve(),
            stage2_effect_teacher_checkpoint_path=action_transfer.CONDITION_CHECKPOINT.resolve(),
            stage2_effect_teacher_config_path=action_transfer.CONDITION_CONFIG.resolve(),
            stage2_action_decoder_checkpoint_path=action_transfer.CHECKPOINT.resolve(),
            stage2_action_decoder_config_path=action_transfer.CONFIG.resolve(),
        )
    path_mismatches = {
        name: (str(_canonical_path(_get(action_cfg, name, None))), str(expected))
        for name, expected in expected_paths.items()
        if _canonical_path(_get(action_cfg, name, None)) != expected
    }
    expected_hashes = {
        "stage2_effect_teacher_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "stage2_effect_teacher_config_sha256": LIBERO_12S_EFFECT_CONFIG_SHA256,
        "stage2_action_decoder_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
        "stage2_action_decoder_config_sha256": LIBERO_12S_ACTION_CONFIG_SHA256,
    }
    if transferred:
        expected_hashes.update(
            stage2_effect_teacher_checkpoint_sha256=action_transfer.CONDITION_SHA256,
            stage2_effect_teacher_config_sha256=action_transfer.CONDITION_CONFIG_SHA256,
            stage2_action_decoder_checkpoint_sha256=action_transfer.CHECKPOINT_SHA256,
            stage2_action_decoder_config_sha256=action_transfer.CONFIG_SHA256,
        )
    hash_mismatches = {
        name: (str(_get(action_cfg, name, "")).lower(), expected)
        for name, expected in expected_hashes.items()
        if str(_get(action_cfg, name, "")).lower() != expected
    }
    if path_mismatches or hash_mismatches:
        raise ValueError(
            "Phased Stage-2 must use the validation-selected E39 Effect teacher and "
            "standard-normal LIBERO 1.2 s ActionVAE: "
            f"path_mismatches={path_mismatches}, hash_mismatches={hash_mismatches}"
        )


def _config_sha256(cfg: Any) -> str:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    # Resume location is a runtime cursor, validated by the training-state contract.
    # The launcher audits before discovering/passing this cursor to the trainer.
    if isinstance(cfg, Mapping) and isinstance(cfg.get("trainer"), Mapping):
        cfg = dict(cfg)
        cfg["trainer"] = dict(cfg["trainer"])
        cfg["trainer"].pop("resume_training_state", None)
    encoded = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path, *, sha256: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256 is not None:
        result["sha256"] = sha256
    return result


def _require_existing_path(value: Any, *, name: str, directory: bool) -> Path:
    raw = str(value or "")
    if not raw or raw.startswith("<REQUIRED_"):
        raise ValueError(f"{name} must be explicitly configured")
    path = Path(raw).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} {kind} does not exist: {path}")
    return path


def _validate_teacher_asset_hashes(action_cfg: Any) -> dict[str, Mapping[str, Any]]:
    fields = [
        ("stage2_action_decoder_checkpoint_path", "stage2_action_decoder_checkpoint_sha256"),
        ("stage2_action_decoder_config_path", "stage2_action_decoder_config_sha256"),
    ]
    if (
        str(_get(action_cfg, "stage2_training_recipe", "") or "")
        == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    ):
        fields.extend(
            (
                (
                    "stage2_lawam_condition_checkpoint_path",
                    "stage2_lawam_condition_checkpoint_sha256",
                ),
                (
                    "stage2_lawam_condition_config_path",
                    "stage2_lawam_condition_config_sha256",
                ),
            )
        )
    else:
        fields.extend(
            (
                (
                    "stage2_effect_teacher_checkpoint_path",
                    "stage2_effect_teacher_checkpoint_sha256",
                ),
                (
                    "stage2_effect_teacher_config_path",
                    "stage2_effect_teacher_config_sha256",
                ),
            )
        )
    validated: dict[str, Mapping[str, Any]] = {}
    for path_field, hash_field in fields:
        path = _require_existing_path(
            _get(action_cfg, path_field, None), name=path_field, directory=False
        )
        expected_digest = str(_get(action_cfg, hash_field, "")).lower()
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"{hash_field} mismatch for {path}: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        validated[hash_field] = _file_identity(path, sha256=actual_digest)
    return validated


def validate_stage2_runtime_assets(
    cfg: Any,
    contract: Stage2PhaseContract | None = None,
) -> Mapping[str, Any] | None:
    """Validate selected Teachers and the official exact-row Gaussian cache."""

    framework_cfg = _get(cfg, "framework", None)
    action_cfg = _get(framework_cfg, "action_model", None)
    contract = contract or resolve_stage2_phase_contract(action_cfg)
    if contract is None:
        return None
    temporal = resolve_stage2_config_temporal_contract(cfg)
    _validate_selected_teacher_binding(action_cfg)
    data_cfg = _get(_get(cfg, "datasets", None), "vla_data", None)

    raw_backend = _get(data_cfg, "dataset_backend", None)
    backend = "" if raw_backend is None else str(raw_backend).strip()
    is_robotwin = is_robotwin_stage2_temporal_contract(temporal)
    is_bridge = is_bridge_stage2_temporal_contract(temporal)
    is_bundle_cache = is_robotwin or is_bridge
    is_adapter_04s = is_libero_04s_adapter_temporal_contract(temporal)
    if is_bundle_cache:
        if backend:
            raise ValueError(
                "exact-row Stage-2 training uses the LeRobot mixture backend "
                "with datasets.vla_data.dataset_backend unset"
            )
        _require_existing_path(
            _get(data_cfg, "stage2_manifest", None),
            name="stage2_manifest",
            directory=True,
        )
    elif is_adapter_04s:
        if backend:
            raise ValueError(
                "the 0.4s adapter must use the baseline LeRobot mixture backend "
                "with datasets.vla_data.dataset_backend unset"
            )
    elif backend != "libero_official_stage2_v1":
        raise ValueError(
            "legacy phased LIBERO Stage-2 requires "
            "dataset_backend=libero_official_stage2_v1"
        )
    if not is_bundle_cache and _get(data_cfg, "stage2_manifest", None) is not None:
        raise ValueError("LIBERO Stage-2 must not use the RoboTwin stage2_manifest route")
    if not is_adapter_04s and not is_bundle_cache:
        try:
            from starVLA.dataloader.libero_official_stage2 import (
                LIBERO_OFFICIAL_STAGE2_BACKEND,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "libero_official_stage2_v1 dataset backend is configured but not implemented"
            ) from exc
        if LIBERO_OFFICIAL_STAGE2_BACKEND != backend:
            raise RuntimeError("official LIBERO Stage-2 backend marker mismatch")
        _require_existing_path(
            _get(data_cfg, "stage2_action_sidecar_root", None),
            name="stage2_action_sidecar_root",
            directory=True,
        )
    _require_existing_path(
        _get(data_cfg, "stage2_source_root", None),
        name="stage2_source_root",
        directory=True,
    )
    action_normalization: Path | None = None
    if not is_adapter_04s and not is_bundle_cache:
        action_normalization = _require_existing_path(
            _get(data_cfg, "stage2_action_normalization", None),
            name="stage2_action_normalization",
            directory=False,
        )
    if int(_get(data_cfg, "num_frames", -1)) != 2:
        raise ValueError("phased Stage-2 requires num_frames=2")
    if float(_get(data_cfg, "sec_chunk", -1.0)) != temporal.duration_seconds:
        raise ValueError(
            f"{temporal.identifier} requires sec_chunk={temporal.duration_seconds}"
        )

    action_cache_root = _require_existing_path(
        _get(action_cfg, "stage2_latent_cache_root", None),
        name="stage2_latent_cache_root",
        directory=True,
    )
    paired_cache_path = _require_existing_path(
        _get(data_cfg, "stage2_paired_cache", None),
        name="stage2_paired_cache",
        directory=not is_bundle_cache,
    )
    cache_root = action_cache_root if is_bundle_cache else paired_cache_path
    expected_paired_cache = (
        action_cache_root / "paired_latents.npz" if is_bundle_cache else action_cache_root
    )
    if paired_cache_path != expected_paired_cache:
        raise ValueError(
            "stage2_latent_cache_root and dataset stage2_paired_cache must resolve "
            "to the same full cache (RoboTwin dataset path is paired_latents.npz)"
        )
    expected_rows = int(_get(data_cfg, "stage2_expected_rows", -1))
    is_lawam_action = (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    )
    expected_teacher_hashes = {
        ("condition" if is_lawam_action else "effect"): str(
            _get(
                action_cfg,
                (
                    "stage2_lawam_condition_checkpoint_sha256"
                    if is_lawam_action
                    else "stage2_effect_teacher_checkpoint_sha256"
                ),
                "",
            )
        ),
        "action": str(_get(action_cfg, "stage2_action_decoder_checkpoint_sha256", "")),
    }
    if is_bundle_cache:
        if expected_rows != temporal.row_count:
            raise ValueError(
                f"{temporal.identifier} cache must bind exactly {temporal.row_count} rows"
            )
        configured_provenance = _get(data_cfg, "stage2_paired_cache_provenance", None)
        if configured_provenance not in (None, ""):
            raise ValueError(
                "exact-row Stage-2 provenance is teacher_bundle.json; "
                "stage2_paired_cache_provenance must be unset"
            )
        normalization_path = _canonical_path(
            _get(action_cfg, "stage2_latent_normalization_path", None)
        )
        if normalization_path != cache_root / "latent_normalization.json":
            raise ValueError(
                "stage2_latent_normalization_path must be the selected "
                "cache latent_normalization.json"
            )
        latent_contract = (
            load_robotwin_stage2_latent_contract(
                cache_root,
                temporal_contract=temporal,
                expected_action_checkpoint_sha256=expected_teacher_hashes["action"],
                expected_effect_checkpoint_sha256=(
                    None if is_lawam_action else expected_teacher_hashes["effect"]
                ),
                expected_condition_checkpoint_sha256=(
                    expected_teacher_hashes["condition"] if is_lawam_action else None
                ),
            )
            if is_robotwin
            else load_bridge_stage2_latent_contract(
                cache_root,
                temporal_contract=temporal,
                expected_action_checkpoint_sha256=expected_teacher_hashes["action"],
                expected_effect_checkpoint_sha256=expected_teacher_hashes["effect"],
            )
        )
        if latent_contract.cache_root != cache_root:
            raise ValueError("latent runtime contract resolved a different cache root")
        bundle = json.loads((cache_root / "teacher_bundle.json").read_text(encoding="utf-8"))
        if is_robotwin:
            configured_sidecar_binding = str(
                _get(action_cfg, "stage2_action_decoder_sidecar_binding", "exact")
            )
            bundle_sidecar = bundle.get("sidecar", {})
            if bundle_sidecar.get("binding", "exact") != configured_sidecar_binding:
                raise ValueError("RoboTwin cache sidecar binding mode differs from the config")
        else:
            configured_sidecar_binding = "exact"
            bundle_sidecar = bundle.get("sidecar", {})
        if is_robotwin and configured_sidecar_binding == "semantic":
            from latent_action_model.robotwin_stage2.sidecar_binding import (
                sha256_file as sidecar_sha256_file,
                verify_reference_bundle,
            )

            reference_root = _require_existing_path(
                _get(
                    action_cfg,
                    "stage2_action_decoder_sidecar_reference_root",
                    None,
                ),
                name="stage2_action_decoder_sidecar_reference_root",
                directory=True,
            )
            verify_reference_bundle(reference_root)
            expected_reference_sha = bundle_sidecar.get("binding_metadata", {}).get(
                "reference_checksums_sha256"
            )
            actual_reference_sha = sidecar_sha256_file(
                reference_root / "checksums.sha256"
            )
            if expected_reference_sha != actual_reference_sha:
                raise ValueError(
                    "RoboTwin cache was built against a different semantic sidecar reference"
                )
        if is_robotwin and configured_sidecar_binding == "contract":
            binding_metadata = bundle_sidecar.get("binding_metadata", {})
            if (
                binding_metadata.get("condition_source") != "lawam_idm"
                or int(binding_metadata.get("rows", -1)) != temporal.row_count
                or binding_metadata.get("byte_identical_sidecar_required") is not False
            ):
                raise ValueError(
                    "RoboTwin LaWAM-IDM cache has an invalid portable sidecar contract"
                )
        decoder_data_root = _canonical_path(
            _get(action_cfg, "stage2_action_decoder_data_root", None)
        )
        sidecar_root = _canonical_path(bundle.get("sidecar", {}).get("root", None))
        if decoder_data_root != sidecar_root:
            raise ValueError(
                "frozen ActionVAE decoder data root must match the cache sidecar"
            )
        validated_teacher_assets = _validate_teacher_asset_hashes(action_cfg)
        return {
            "stage2_temporal_contract": temporal.identifier,
            "dataset_backend": (
                "robotwin_exact_row_lerobot_mixture"
                if is_robotwin
                else "bridge_exact_row_lerobot_mixture"
            ),
            "cache_root": str(cache_root),
            "paired_cache": str(paired_cache_path),
            "rows": expected_rows,
            "action_normalization": str(
                decoder_data_root / "stats" / "normalization.json"
            ),
            "teacher_checkpoint_sha256": expected_teacher_hashes,
            "validated_teacher_assets": validated_teacher_assets,
            "action_logvar_support": {
                "raw": [
                    latent_contract.action_raw_logvar_lower,
                    latent_contract.action_raw_logvar_upper,
                ],
                "standardized_envelope": [
                    float(latent_contract.action_logvar_lower.min()),
                    float(latent_contract.action_logvar_upper.max()),
                ],
                "normalization_sha256": latent_contract.normalization_sha256,
            },
        }

    cache_files = {
        name: _require_existing_path(cache_root / name, name=f"cache/{name}", directory=False)
        for name in (
            "_SUCCESS",
            "provenance.json",
            "normalization.json",
            "manifest.parquet",
            "latents.h5",
        )
    }
    configured_provenance = Path(
        str(_get(data_cfg, "stage2_paired_cache_provenance", ""))
    ).expanduser().resolve()
    if configured_provenance != cache_files["provenance.json"]:
        raise ValueError(
            "stage2_paired_cache_provenance must bind the selected cache provenance.json"
        )
    success = json.loads(cache_files["_SUCCESS"].read_text(encoding="utf-8"))
    provenance = json.loads(cache_files["provenance.json"].read_text(encoding="utf-8"))
    if (
        success.get("schema_version") != temporal.cache_schema_version
        or provenance.get("schema_version") != temporal.cache_schema_version
        or success.get("release_level") != "full"
        or success.get("all_gates_passed") is not True
        or provenance.get("ready") is not True
    ):
        raise ValueError(
            "Stage-2 requires a completed full cache release using schema "
            f"{temporal.cache_schema_version}"
        )
    provenance_rows = int(provenance.get("row_count", provenance.get("rows", -1)))
    if (
        expected_rows != temporal.row_count
        or int(success.get("row_count", -1)) != expected_rows
        or provenance_rows != expected_rows
    ):
        raise ValueError(
            f"{temporal.identifier} cache must bind exactly {temporal.row_count} rows"
        )

    for payload_name, payload in (("_SUCCESS", success), ("provenance", provenance)):
        recorded = payload.get("teacher_checkpoint_sha256")
        if not isinstance(recorded, Mapping) or dict(recorded) != expected_teacher_hashes:
            raise ValueError(
                f"{payload_name} teacher checkpoint SHA contract mismatch: "
                f"recorded={recorded}, expected={expected_teacher_hashes}"
            )

    normalization_path = Path(
        str(_get(action_cfg, "stage2_latent_normalization_path", ""))
    ).expanduser().resolve()
    if normalization_path != cache_files["normalization.json"]:
        raise ValueError(
            "stage2_latent_normalization_path must be the selected cache normalization.json"
        )
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization.get("schema_version") != temporal.normalization_schema_version:
        raise ValueError(
            "Stage-2 latent normalization schema mismatch: "
            f"expected={temporal.normalization_schema_version!r}, "
            f"got={normalization.get('schema_version')!r}"
        )

    transfer = _get(action_cfg, "stage2_action_teacher_transfer", None)
    if transfer != _get(data_cfg, "stage2_action_teacher_transfer", None):
        raise ValueError("model/dataset ActionVAE transfer markers differ")
    latent_contract = load_libero_stage2_latent_contract(
        cache_root,
        temporal_contract=temporal,
        expected_rows=expected_rows,
        expected_schema_version=temporal.cache_schema_version,
        **({"action_teacher_transfer": transfer} if transfer else {}),
    )
    if latent_contract.cache_root != cache_root:
        raise ValueError("Stage-2 latent runtime contract resolved a different cache root")

    import h5py

    with h5py.File(cache_files["latents.h5"], "r") as handle:
        if (
            handle.attrs.get("schema_version") != temporal.cache_schema_version
            or bool(handle.attrs.get("complete", False)) is not True
        ):
            raise ValueError("latent-cache HDF5 is incomplete or uses the wrong schema")
        arrays = ("z_effect", "z_action_mean", "z_action_logvar")
        missing_arrays = sorted(set(arrays).difference(handle.keys()))
        if missing_arrays:
            raise ValueError(f"Gaussian latent cache is missing arrays: {missing_arrays}")
        bad_shapes = {
            name: tuple(handle[name].shape)
            for name in arrays
            if tuple(handle[name].shape) != (expected_rows, 32)
        }
        if bad_shapes:
            raise ValueError(f"Gaussian latent cache shape mismatch: {bad_shapes}")

    validated_teacher_assets = _validate_teacher_asset_hashes(action_cfg)
    decoder_data_root = Path(
        str(_get(action_cfg, "stage2_action_decoder_data_root", ""))
    ).expanduser().resolve()
    if not is_adapter_04s:
        sidecar_root = Path(
            str(_get(data_cfg, "stage2_action_sidecar_root", ""))
        ).expanduser().resolve()
        if decoder_data_root != sidecar_root:
            raise ValueError(
                "frozen ActionVAE decoder data root must match official action sidecar"
            )

    return {
        "stage2_temporal_contract": temporal.identifier,
        "dataset_backend": backend or "baseline_lerobot_mixture",
        "cache_root": str(cache_root),
        "rows": expected_rows,
        "action_normalization": (
            None if action_normalization is None else str(action_normalization)
        ),
        "teacher_checkpoint_sha256": expected_teacher_hashes,
        "validated_teacher_assets": validated_teacher_assets,
        "action_logvar_support": {
            "raw": [
                latent_contract.action_raw_logvar_lower,
                latent_contract.action_raw_logvar_upper,
            ],
            "standardized_envelope": [
                float(latent_contract.action_logvar_lower.min()),
                float(latent_contract.action_logvar_upper.max()),
            ],
            "normalization_sha256": latent_contract.normalization_sha256,
        },
    }


def stage2_checkpoint_contract_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    return path.with_name(f"{path.name}.stage2.json")


def write_stage2_checkpoint_contract(
    *,
    checkpoint: str | Path,
    cfg: Any,
    completed_steps: int,
) -> Path | None:
    """Bind a model-only checkpoint to its immutable W/P/E lineage."""

    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    contract = resolve_stage2_phase_contract(action_cfg)
    if contract is None:
        return None
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if int(completed_steps) < 0:
        raise ValueError("completed_steps must be non-negative")
    trainer_cfg = _get(cfg, "trainer", None)
    temporal = resolve_stage2_config_temporal_contract(cfg)
    module_structure = resolve_stage2_module_structure(contract)
    payload = {
        "schema_version": STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
        "module_structure": module_structure,
        "stage2_phase": contract.phase.value,
        "stage2_temporal_contract": temporal.identifier,
        "stage2_parent_phase": _get(trainer_cfg, "stage2_parent_phase", None),
        "run_id": str(_get(cfg, "run_id", "")),
        "completed_steps": int(completed_steps),
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "losses": contract.losses.to_dict(),
    }
    if contract.training_recipe in RESIDUAL_RECIPES:
        payload["stage2_training_recipe"] = contract.training_recipe
        payload["vlm_update"] = {
            "method": (
                "frozen" if contract.phase is Stage2Phase.W else "full_parameter"
            ),
            "retained_llm_blocks": list(range(16)),
            "vision_backbone": (
                "frozen" if contract.phase is Stage2Phase.W else "trainable"
            ),
            "vision_merger": (
                "frozen" if contract.phase is Stage2Phase.W else "trainable"
            ),
            "embedding": "frozen",
        }
    else:
        payload["vlm_update"] = {
            "method": "full_parameter",
            "retained_llm_blocks": [14, 15],
            "vision_backbone": "frozen",
            "vision_merger": "frozen",
            "embedding": "frozen",
        }
    output = stage2_checkpoint_contract_path(checkpoint_path)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_stage2_parent_checkpoint(
    cfg: Any,
    contract: Stage2PhaseContract | None = None,
) -> Mapping[str, Any] | None:
    """Verify a P/E parent checkpoint's sidecar, phase, size, and SHA256."""

    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    contract = contract or resolve_stage2_phase_contract(action_cfg)
    if contract is None:
        return None
    trainer_cfg = _get(cfg, "trainer", None)
    temporal = resolve_stage2_config_temporal_contract(cfg)
    checkpoint = Path(str(_get(trainer_cfg, "pretrained_checkpoint", ""))).expanduser().resolve()
    if (
        contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE
        and contract.phase is Stage2Phase.W
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Bridge B0 parent checkpoint does not exist: {checkpoint}")
        selection_path = Path(
            str(_get(trainer_cfg, "stage2_parent_selection", ""))
        ).expanduser().resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Bridge Phase W requires the B0 validation selection: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection.get("selected") if isinstance(selection, Mapping) else None
        if not isinstance(selected, Mapping):
            raise ValueError("Bridge B0 validation selection has no selected checkpoint")
        actual_sha256 = _sha256_file(checkpoint)
        expected = {
            "schema_version": STAGE2_SELECTION_SCHEMA,
            "stage2_arm": "b0",
            "selection_tag": BRIDGE_B0_SELECTION_TAG,
            "selection_split": "validation",
            "selection_metric": "val_loss_total",
            "selection_mode": "min",
            "test_metrics_read": False,
        }
        mismatches = {
            name: (selection.get(name), value)
            for name, value in expected.items()
            if selection.get(name) != value
        }
        selected_path = Path(str(selected.get("checkpoint", ""))).expanduser().resolve()
        if selected_path != checkpoint:
            mismatches["selected.checkpoint"] = (str(selected_path), str(checkpoint))
        if selected.get("checkpoint_sha256") != actual_sha256:
            mismatches["selected.checkpoint_sha256"] = (
                selected.get("checkpoint_sha256"),
                actual_sha256,
            )
        if mismatches:
            raise ValueError(f"Bridge B0 validation selection mismatch: {mismatches}")
        return {
            "stage2_phase": "B0",
            "stage2_temporal_contract": temporal.identifier,
            "checkpoint_sha256": actual_sha256,
            "selection": str(selection_path),
        }
    if contract.expected_parent_phase is None:
        return None
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 parent checkpoint does not exist: {checkpoint}")
    sidecar = stage2_checkpoint_contract_path(checkpoint)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Stage-2 parent checkpoint has no phase sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Stage-2 checkpoint sidecar root must be an object: {sidecar}")
    expected = {
        "schema_version": STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
        "module_structure": resolve_stage2_module_structure(contract),
        "stage2_phase": contract.expected_parent_phase.value,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_bytes": int(checkpoint.stat().st_size),
    }
    if contract.training_recipe:
        expected["stage2_training_recipe"] = contract.training_recipe
    if (
        is_libero_04s_adapter_temporal_contract(temporal)
        or "stage2_temporal_contract" in payload
    ):
        expected["stage2_temporal_contract"] = temporal.identifier
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Stage-2 parent checkpoint contract mismatch: {mismatches}")
    expected_sha256 = str(payload.get("checkpoint_sha256", ""))
    actual_sha256 = _sha256_file(checkpoint)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "Stage-2 parent checkpoint SHA256 mismatch: "
            f"sidecar={expected_sha256!r}, actual={actual_sha256!r}"
        )

    selection_path = Path(
        str(_get(trainer_cfg, "stage2_parent_selection", ""))
    ).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Phase {contract.phase.value} requires the parent validation selection: "
            f"{selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise TypeError(f"Stage-2 selection root must be an object: {selection_path}")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Stage-2 parent selection has no selected checkpoint record")
    selection_expected = {
        "schema_version": STAGE2_SELECTION_SCHEMA,
        "stage2_phase": contract.expected_parent_phase.value,
        "selection_split": "validation",
        "selection_metric": "val_loss_total",
        "selection_mode": "min",
        "test_metrics_read": False,
    }
    if contract.training_recipe:
        selection_expected["stage2_training_recipe"] = contract.training_recipe
    selection_mismatches = {
        name: (selection.get(name), value)
        for name, value in selection_expected.items()
        if selection.get(name) != value
    }
    selected_checkpoint = Path(str(selected.get("checkpoint", ""))).expanduser().resolve()
    if selected_checkpoint != checkpoint:
        selection_mismatches["selected.checkpoint"] = (
            str(selected_checkpoint),
            str(checkpoint),
        )
    if selected.get("checkpoint_sha256") != actual_sha256:
        selection_mismatches["selected.checkpoint_sha256"] = (
            selected.get("checkpoint_sha256"),
            actual_sha256,
        )
    if selection_mismatches:
        raise ValueError(
            f"Stage-2 parent validation selection mismatch: {selection_mismatches}"
        )
    return payload


def _stage2_preflight_asset_paths(
    cfg: Any,
    contract: Stage2PhaseContract,
) -> dict[str, Path]:
    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    data_cfg = _get(_get(cfg, "datasets", None), "vla_data", None)
    trainer_cfg = _get(cfg, "trainer", None)
    temporal = resolve_stage2_config_temporal_contract(cfg)
    cache_root = _canonical_path(_get(action_cfg, "stage2_latent_cache_root", None))
    is_lawam_action = (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    )
    assets = {
        "action_checkpoint": _canonical_path(
            _get(action_cfg, "stage2_action_decoder_checkpoint_path", None)
        ),
        "action_config": _canonical_path(
            _get(action_cfg, "stage2_action_decoder_config_path", None)
        ),
        "init_checkpoint": _canonical_path(
            _get(trainer_cfg, "pretrained_checkpoint", None)
        ),
    }
    if is_lawam_action:
        assets.update(
            condition_checkpoint=_canonical_path(
                _get(action_cfg, "stage2_lawam_condition_checkpoint_path", None)
            ),
            condition_config=_canonical_path(
                _get(action_cfg, "stage2_lawam_condition_config_path", None)
            ),
        )
    else:
        assets.update(
            effect_checkpoint=_canonical_path(
                _get(action_cfg, "stage2_effect_teacher_checkpoint_path", None)
            ),
            effect_config=_canonical_path(
                _get(action_cfg, "stage2_effect_teacher_config_path", None)
            ),
        )
    if (
        is_robotwin_stage2_temporal_contract(temporal)
        or is_bridge_stage2_temporal_contract(temporal)
    ):
        assets.update(
            cache_success=cache_root / "_SUCCESS",
            cache_checksums=cache_root / "checksums.sha256",
            cache_bundle=cache_root / "teacher_bundle.json",
            cache_normalization=cache_root / "latent_normalization.json",
            cache_latents=cache_root / "paired_latents.npz",
            action_normalization=_canonical_path(
                _get(action_cfg, "stage2_action_decoder_data_root", None)
            )
            / "stats"
            / "normalization.json",
        )
    else:
        assets.update(
            cache_success=cache_root / "_SUCCESS",
            cache_provenance=cache_root / "provenance.json",
            cache_normalization=cache_root / "normalization.json",
            cache_manifest=cache_root / "manifest.parquet",
            cache_latents=cache_root / "latents.h5",
        )
    if temporal.identifier == LIBERO_OFFICIAL_12S_TEMPORAL_CONTRACT:
        assets["action_normalization"] = _canonical_path(
            _get(data_cfg, "stage2_action_normalization", None)
        )
    if (
        contract.expected_parent_phase is not None
        or (
            contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE
            and contract.phase is Stage2Phase.W
        )
    ):
        assets["parent_selection"] = _canonical_path(
            _get(trainer_cfg, "stage2_parent_selection", None)
        )
        if contract.expected_parent_phase is not None:
            assets["parent_checkpoint_sidecar"] = stage2_checkpoint_contract_path(
                assets["init_checkpoint"]
            )
    return assets


def write_stage2_launch_preflight_receipt(
    *,
    cfg: Any,
    output: str | Path,
) -> Path:
    """Run all expensive checks once and bind their result to the merged config."""

    contract = validate_stage2_training_config(cfg)
    if contract is None:
        raise ValueError("Stage-2 launch preflight requires a phased Stage-2 config")
    runtime = validate_stage2_runtime_assets(cfg, contract)
    if runtime is None:
        raise RuntimeError("Stage-2 runtime validation unexpectedly returned no result")
    parent = validate_stage2_parent_checkpoint(cfg, contract)

    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    is_lawam_action = (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    )
    teacher_sha_by_role = {
        "action_checkpoint": str(
            _get(action_cfg, "stage2_action_decoder_checkpoint_sha256", "")
        ),
        "action_config": str(
            _get(action_cfg, "stage2_action_decoder_config_sha256", "")
        ),
    }
    if is_lawam_action:
        teacher_sha_by_role.update(
            condition_checkpoint=str(
                _get(action_cfg, "stage2_lawam_condition_checkpoint_sha256", "")
            ),
            condition_config=str(
                _get(action_cfg, "stage2_lawam_condition_config_sha256", "")
            ),
        )
    else:
        teacher_sha_by_role.update(
            effect_checkpoint=str(
                _get(action_cfg, "stage2_effect_teacher_checkpoint_sha256", "")
            ),
            effect_config=str(
                _get(action_cfg, "stage2_effect_teacher_config_sha256", "")
            ),
        )
    assets: dict[str, Mapping[str, Any]] = {}
    for role, path in _stage2_preflight_asset_paths(cfg, contract).items():
        if not path.is_file():
            raise FileNotFoundError(f"Stage-2 preflight asset does not exist: role={role}, path={path}")
        digest = teacher_sha_by_role.get(role)
        if role == "init_checkpoint" and parent is not None:
            digest = str(parent.get("checkpoint_sha256", ""))
        elif (
            role == "init_checkpoint"
            and contract.training_recipe
            in {
                ROBOTWIN_SFT25_RESIDUAL_RECIPE,
                ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
                ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
            }
            and contract.phase
            is (
                Stage2Phase.P
                if contract.training_recipe
                == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
                else Stage2Phase.W
            )
        ):
            if path.stat().st_size != ROBOTWIN_SFT_RELEASE_CHECKPOINT_BYTES:
                raise ValueError(
                    "official RoboTwin SFT checkpoint byte size mismatch: "
                    f"expected={ROBOTWIN_SFT_RELEASE_CHECKPOINT_BYTES}, "
                    f"actual={path.stat().st_size}"
                )
            actual_digest = _sha256_file(path)
            if actual_digest != ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256:
                raise ValueError(
                    "official RoboTwin SFT checkpoint SHA256 mismatch: "
                    f"expected={ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256}, "
                    f"actual={actual_digest}"
                )
            digest = ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256
        elif (
            role == "init_checkpoint"
            and contract.training_recipe
            in {PRETRAIN_RESIDUAL_RECIPE, ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE}
            and contract.phase is Stage2Phase.W
        ):
            if path.stat().st_size != LIBERO_PRETRAIN_CHECKPOINT_BYTES:
                raise ValueError(
                    "official pretrain checkpoint byte size mismatch: "
                    f"expected={LIBERO_PRETRAIN_CHECKPOINT_BYTES}, "
                    f"actual={path.stat().st_size}"
                )
            actual_digest = _sha256_file(path)
            if actual_digest != LIBERO_PRETRAIN_CHECKPOINT_SHA256:
                raise ValueError(
                    "official pretrain checkpoint SHA256 mismatch: "
                    f"expected={LIBERO_PRETRAIN_CHECKPOINT_SHA256}, "
                    f"actual={actual_digest}"
                )
            digest = LIBERO_PRETRAIN_CHECKPOINT_SHA256
        assets[role] = _file_identity(path, sha256=digest)

    payload = {
        "schema_version": STAGE2_LAUNCH_PREFLIGHT_SCHEMA,
        "ready": True,
        "stage2_phase": contract.phase.value,
        "stage2_temporal_contract": resolve_stage2_config_temporal_contract(cfg).identifier,
        "config_sha256": _config_sha256(cfg),
        "module_structure": resolve_stage2_module_structure(contract),
        "runtime": dict(runtime),
        "assets": assets,
    }
    if contract.training_recipe:
        payload["stage2_training_recipe"] = contract.training_recipe
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def validate_stage2_launch_preflight_receipt(
    cfg: Any,
    receipt: str | Path,
    contract: Stage2PhaseContract | None = None,
) -> Mapping[str, Any]:
    """Validate a single-process preflight receipt without rehashing large files."""

    action_cfg = _get(_get(cfg, "framework", None), "action_model", None)
    contract = contract or resolve_stage2_phase_contract(action_cfg)
    if contract is None:
        raise ValueError("Stage-2 launch receipt cannot be used by a non-phased config")
    temporal = resolve_stage2_config_temporal_contract(cfg)
    _validate_selected_teacher_binding(action_cfg)
    receipt_path = Path(receipt).expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Stage-2 launch preflight receipt does not exist: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Stage-2 launch preflight root must be an object: {receipt_path}")
    expected = {
        "schema_version": STAGE2_LAUNCH_PREFLIGHT_SCHEMA,
        "ready": True,
        "stage2_phase": contract.phase.value,
        "config_sha256": _config_sha256(cfg),
        "module_structure": resolve_stage2_module_structure(contract),
    }
    if contract.training_recipe:
        expected["stage2_training_recipe"] = contract.training_recipe
    if (
        is_libero_04s_adapter_temporal_contract(temporal)
        or "stage2_temporal_contract" in payload
    ):
        expected["stage2_temporal_contract"] = temporal.identifier
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Stage-2 launch preflight receipt mismatch: {mismatches}")

    recorded_assets = payload.get("assets")
    if not isinstance(recorded_assets, Mapping):
        raise ValueError("Stage-2 launch preflight receipt has no asset identities")
    expected_paths = _stage2_preflight_asset_paths(cfg, contract)
    if set(recorded_assets) != set(expected_paths):
        raise ValueError(
            "Stage-2 launch preflight asset roles mismatch: "
            f"recorded={sorted(recorded_assets)}, expected={sorted(expected_paths)}"
        )
    is_lawam_action = (
        contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
    )
    teacher_sha_by_role = {
        "action_checkpoint": str(
            _get(action_cfg, "stage2_action_decoder_checkpoint_sha256", "")
        ),
        "action_config": str(
            _get(action_cfg, "stage2_action_decoder_config_sha256", "")
        ),
    }
    if is_lawam_action:
        teacher_sha_by_role.update(
            condition_checkpoint=str(
                _get(action_cfg, "stage2_lawam_condition_checkpoint_sha256", "")
            ),
            condition_config=str(
                _get(action_cfg, "stage2_lawam_condition_config_sha256", "")
            ),
        )
    else:
        teacher_sha_by_role.update(
            effect_checkpoint=str(
                _get(action_cfg, "stage2_effect_teacher_checkpoint_sha256", "")
            ),
            effect_config=str(
                _get(action_cfg, "stage2_effect_teacher_config_sha256", "")
            ),
        )
    if (
        contract.training_recipe == BRIDGE_B0_RESIDUAL_RECIPE
        and contract.phase is Stage2Phase.W
    ):
        selection_path = _canonical_path(
            _get(_get(cfg, "trainer", None), "stage2_parent_selection", None)
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        teacher_sha_by_role["init_checkpoint"] = str(
            selection.get("selected", {}).get("checkpoint_sha256", "")
        )
    elif (
        contract.training_recipe
        in {PRETRAIN_RESIDUAL_RECIPE, ROBOTWIN_PRETRAIN_RESIDUAL_RECIPE}
        and contract.phase is Stage2Phase.W
    ):
        teacher_sha_by_role["init_checkpoint"] = LIBERO_PRETRAIN_CHECKPOINT_SHA256
    elif (
        contract.training_recipe
        in {
            ROBOTWIN_SFT25_RESIDUAL_RECIPE,
            ROBOTWIN_SFT_RELEASE_RESIDUAL_RECIPE,
            ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE,
        }
        and contract.phase
        is (
            Stage2Phase.P
            if contract.training_recipe == ROBOTWIN_SFT_RELEASE_LAWAM_ACTION_RECIPE
            else Stage2Phase.W
        )
    ):
        teacher_sha_by_role["init_checkpoint"] = ROBOTWIN_SFT_RELEASE_CHECKPOINT_SHA256
    asset_mismatches: dict[str, Any] = {}
    for role, expected_path in expected_paths.items():
        record = recorded_assets.get(role)
        if not isinstance(record, Mapping):
            asset_mismatches[role] = (record, "mapping")
            continue
        if not expected_path.is_file():
            asset_mismatches[role] = ("missing", str(expected_path))
            continue
        stat = expected_path.stat()
        expected_identity = {
            "path": str(expected_path),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        for name, value in expected_identity.items():
            if record.get(name) != value:
                asset_mismatches[f"{role}.{name}"] = (record.get(name), value)
        expected_sha = teacher_sha_by_role.get(role)
        if expected_sha is not None and record.get("sha256") != expected_sha:
            asset_mismatches[f"{role}.sha256"] = (record.get("sha256"), expected_sha)
    if asset_mismatches:
        raise ValueError(
            "Stage-2 assets changed after single-process preflight: "
            f"{asset_mismatches}"
        )
    return payload


def apply_stage2_phase_freeze_contract(model: Any, cfg: Any) -> Any:
    """Reassert the backend-owned phase policy after the generic release freeze."""

    contract = validate_stage2_training_config(cfg)
    if contract is None:
        return model
    backend = getattr(model, "policy_backend", model)
    apply_contract = getattr(backend, "apply_stage2_phase_contract", None)
    if not callable(apply_contract):
        raise RuntimeError(
            "stage2_phase is configured but the policy backend has no "
            "apply_stage2_phase_contract() implementation"
        )
    apply_contract(contract.phase.value)
    return model
