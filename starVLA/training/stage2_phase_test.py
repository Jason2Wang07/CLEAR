from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from . import stage2_phase as stage2
from .stage2_phase import (
    LIBERO_PRETRAIN_CHECKPOINT,
    LIBERO_SFT_RELEASE_CHECKPOINT,
    PHASE_SCHEDULES,
    PHASE_SCHEDULES_04S,
    PRETRAIN_RESIDUAL_PHASE_SCHEDULES,
    RELEASE_RESIDUAL_MODULE_STRUCTURE,
    RELEASE_RESIDUAL_PHASE_SCHEDULES,
    STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
    STAGE2_LAUNCH_PREFLIGHT_SCHEMA,
    STAGE2_MODULE_STRUCTURE,
    STAGE2_SELECTION_SCHEMA,
    Stage2Phase,
    resolve_stage2_config_temporal_contract,
    resolve_stage2_phase_schedule,
    stage2_checkpoint_contract_path,
    validate_stage2_launch_preflight_receipt,
    validate_stage2_parent_checkpoint,
    validate_stage2_training_config,
)
from latent_action_model.robotwin_stage2.paired import (
    PRETRAIN_RESIDUAL_RECIPE,
    RELEASE_RESIDUAL_RECIPE,
)


def _phase_w_config(*, adapter_04s: bool) -> dict:
    seconds = 0.4 if adapter_04s else 1.2
    schedule = PHASE_SCHEDULES_04S[Stage2Phase.W] if adapter_04s else PHASE_SCHEDULES[Stage2Phase.W]
    action = {
        "stage2_phase": "W",
        "stage2_arm": "p2",
        "stage2_action_distribution": "dense_transition_gaussian_v3",
        "stage2_query_scheme": "act_effect_flow_plan_v1",
        "future_prediction": True,
        "detach_future_feature": True,
        "stage2_horizon_condition_dim": 3,
        "stage2_horizon_seconds": seconds,
        "stage2_action_state_source": "actionvae14_padded32",
        "flow_cfg": {"horizon_sec": seconds},
        "stage2_action_decoder_checkpoint_path": "/placeholder/action.ckpt",
        "stage2_action_decoder_config_path": "/placeholder/action.yaml",
        "stage2_action_decoder_data_root": "/placeholder/action-data",
        "stage2_latent_cache_root": "/placeholder/cache",
        "stage2_latent_normalization_path": "/placeholder/cache/normalization.json",
        "stage2_effect_teacher_checkpoint_path": "/placeholder/effect.ckpt",
        "stage2_effect_teacher_config_path": "/placeholder/effect.yaml",
        "stage2_action_decoder_checkpoint_sha256": "a" * 64,
        "stage2_action_decoder_config_sha256": "b" * 64,
        "stage2_effect_teacher_checkpoint_sha256": "c" * 64,
        "stage2_effect_teacher_config_sha256": "d" * 64,
    }
    data = {
        "stage2_action_state_source": "actionvae14_padded32",
        "sec_chunk": seconds,
    }
    if adapter_04s:
        marker = "libero_baseline_04s_adapter_v1"
        action["stage2_temporal_contract"] = marker
        data["stage2_temporal_contract"] = marker
    return {
        "framework": {"action_model": action},
        "datasets": {"vla_data": data},
        "trainer": {
            "max_train_steps": schedule[0],
            "num_warmup_steps": schedule[1],
            "eval_interval": 500,
            "save_interval": 500,
            "pretrained_checkpoint": "/placeholder/release.pt",
            "strict_finetune_init": False,
            "load_pretrained_lam": False,
            "load_pretrained_policy_flow": True,
            "learning_rate": {"vlm": {"lr": 1.0e-5}},
            "freeze": {
                "freeze_vision_backbone": True,
                "freeze_llm_backbone": True,
                "keep_llm_first_n_layers": 16,
                "freeze_last_llm_layer": True,
                "freeze_embedding": True,
                "unfreeze_vision_merger": False,
                "unfreeze_llm_last_n_layers": 2,
                "unfreeze_lam_decoder": True,
            },
        },
    }


@pytest.mark.parametrize("adapter_04s", (False, True))
def test_training_config_accepts_legacy_and_adapter_contracts(adapter_04s: bool) -> None:
    cfg = _phase_w_config(adapter_04s=adapter_04s)

    contract = validate_stage2_training_config(cfg)

    assert contract is not None and contract.phase is Stage2Phase.W
    temporal = resolve_stage2_config_temporal_contract(cfg)
    assert temporal.duration_seconds == (0.4 if adapter_04s else 1.2)


def test_adapter_phase_schedules_match_25k_step_sample_budget() -> None:
    marker = "libero_baseline_04s_adapter_v1"
    assert resolve_stage2_phase_schedule("W", marker) == (3_571, 214)
    assert resolve_stage2_phase_schedule("P", marker) == (14_286, 857)
    assert resolve_stage2_phase_schedule("E", marker) == (7_143, 429)
    assert sum(PHASE_SCHEDULES_04S[phase][0] for phase in Stage2Phase) == 25_000


def _release_residual_config(phase: Stage2Phase) -> dict:
    cfg = _phase_w_config(adapter_04s=True)
    action = cfg["framework"]["action_model"]
    trainer = cfg["trainer"]
    action.update(
        stage2_training_recipe=RELEASE_RESIDUAL_RECIPE,
        stage2_residual_warmup_start_step=0,
        stage2_residual_teacher_prob_start=0.0,
        stage2_residual_teacher_prob_end=0.0,
        stage2_residual_teacher_ramp_steps=1,
        stage2_residual_predictor_grad_start_step=0,
        stage2_residual_condition_dropout=0.0,
    )
    trainer.update(
        max_train_steps=RELEASE_RESIDUAL_PHASE_SCHEDULES[phase][0],
        num_warmup_steps=RELEASE_RESIDUAL_PHASE_SCHEDULES[phase][1],
    )
    action["stage2_phase"] = phase.value
    losses = {
        Stage2Phase.W: (0.0, 1.0, 0.0, 1.0),
        Stage2Phase.P: (1.0, 0.1, 0.5, 0.1),
        Stage2Phase.E: (1.0, 0.1, 0.25, 0.1),
    }[phase]
    (
        action["stage2_flow_weight"],
        action["stage2_effect_distill_weight"],
        action["stage2_action_distill_weight"],
        action["stage2_dino_weight"],
    ) = losses
    if phase is Stage2Phase.W:
        trainer["pretrained_checkpoint"] = str(LIBERO_SFT_RELEASE_CHECKPOINT)
        trainer["learning_rate"]["vlm"]["lr"] = 0.0
        trainer["freeze"]["unfreeze_llm_last_n_layers"] = 0
    else:
        trainer.update(
            pretrained_checkpoint="/placeholder/parent.pt",
            stage2_parent_phase=("W" if phase is Stage2Phase.P else "P"),
            stage2_parent_selection="/placeholder/selection.json",
            strict_finetune_init=True,
            load_pretrained_lam=True,
        )
        trainer["learning_rate"]["vlm"]["lr"] = 1.0e-5
        trainer["freeze"].update(
            freeze_vision_backbone=False,
            freeze_llm_backbone=False,
            unfreeze_vision_merger=True,
            unfreeze_llm_last_n_layers=-1,
            unfreeze_lam_decoder=(phase is Stage2Phase.E),
        )
    if phase is Stage2Phase.P:
        action.update(
            stage2_residual_warmup_start_step=6_000,
            stage2_residual_teacher_prob_start=0.5,
            stage2_residual_teacher_ramp_steps=1_000,
            stage2_residual_predictor_grad_start_step=7_000,
        )
    elif phase is Stage2Phase.E:
        action["stage2_residual_condition_dropout"] = 0.1
    return cfg


def test_release_residual_schedule_is_exact_25k_budget() -> None:
    marker = "libero_baseline_04s_adapter_v1"
    for phase, schedule in RELEASE_RESIDUAL_PHASE_SCHEDULES.items():
        assert resolve_stage2_phase_schedule(
            phase, marker, RELEASE_RESIDUAL_RECIPE
        ) == schedule
    assert sum(value[0] for value in RELEASE_RESIDUAL_PHASE_SCHEDULES.values()) == 25_000


@pytest.mark.parametrize("phase", tuple(Stage2Phase))
def test_release_residual_training_contracts_are_isolated(phase: Stage2Phase) -> None:
    cfg = _release_residual_config(phase)

    contract = validate_stage2_training_config(cfg)

    assert contract is not None
    assert contract.training_recipe == RELEASE_RESIDUAL_RECIPE
    assert stage2.resolve_stage2_module_structure(contract) == RELEASE_RESIDUAL_MODULE_STRUCTURE


def test_release_residual_phase_b_rejects_top_two_vlm_only() -> None:
    cfg = _release_residual_config(Stage2Phase.P)
    cfg["trainer"]["freeze"].update(
        freeze_vision_backbone=True,
        freeze_llm_backbone=True,
        unfreeze_vision_merger=False,
        unfreeze_llm_last_n_layers=2,
    )

    with pytest.raises(ValueError, match="freeze contract mismatch"):
        validate_stage2_training_config(cfg)


def _pretrain_residual_config(phase: Stage2Phase) -> dict:
    cfg = _release_residual_config(phase)
    action = cfg["framework"]["action_model"]
    trainer = cfg["trainer"]
    action["stage2_training_recipe"] = PRETRAIN_RESIDUAL_RECIPE
    trainer.update(
        max_train_steps=PRETRAIN_RESIDUAL_PHASE_SCHEDULES[phase][0],
        num_warmup_steps=PRETRAIN_RESIDUAL_PHASE_SCHEDULES[phase][1],
        process_checkpoint_keep_limit=1,
    )
    if phase is Stage2Phase.W:
        trainer["pretrained_checkpoint"] = str(LIBERO_PRETRAIN_CHECKPOINT)
    elif phase is Stage2Phase.P:
        trainer["learning_rate"]["vlm"]["lr"] = 5.0e-5
        action.update(
            stage2_residual_warmup_start_step=12_000,
            stage2_residual_teacher_ramp_steps=2_000,
            stage2_residual_predictor_grad_start_step=14_000,
        )
    else:
        trainer["learning_rate"]["vlm"]["lr"] = 1.0e-4
    return cfg


def test_pretrain_residual_schedule_is_exact_50k_budget() -> None:
    marker = "libero_baseline_04s_adapter_v1"
    for phase, schedule in PRETRAIN_RESIDUAL_PHASE_SCHEDULES.items():
        assert resolve_stage2_phase_schedule(
            phase, marker, PRETRAIN_RESIDUAL_RECIPE
        ) == schedule
    assert sum(value[0] for value in PRETRAIN_RESIDUAL_PHASE_SCHEDULES.values()) == 50_000


@pytest.mark.parametrize("phase", tuple(Stage2Phase))
def test_pretrain_residual_training_contracts_are_isolated(phase: Stage2Phase) -> None:
    cfg = _pretrain_residual_config(phase)

    contract = validate_stage2_training_config(cfg)

    assert contract is not None
    assert contract.training_recipe == PRETRAIN_RESIDUAL_RECIPE
    assert stage2.resolve_stage2_module_structure(contract) == RELEASE_RESIDUAL_MODULE_STRUCTURE


def test_pretrain_residual_phase_a_rejects_libero_sft_release_init() -> None:
    cfg = _pretrain_residual_config(Stage2Phase.W)
    cfg["trainer"]["pretrained_checkpoint"] = str(LIBERO_SFT_RELEASE_CHECKPOINT)

    with pytest.raises(ValueError, match="lawam_pretrain"):
        validate_stage2_training_config(cfg)


def test_pretrain_residual_phase_b_rejects_release_lr_and_timeline() -> None:
    cfg = _pretrain_residual_config(Stage2Phase.P)
    cfg["trainer"]["learning_rate"]["vlm"]["lr"] = 1.0e-5
    cfg["framework"]["action_model"].update(
        stage2_residual_warmup_start_step=6_000,
        stage2_residual_teacher_ramp_steps=1_000,
        stage2_residual_predictor_grad_start_step=7_000,
    )

    with pytest.raises(ValueError, match="curriculum contract mismatch|VLM LR"):
        validate_stage2_training_config(cfg)


def test_pretrain_residual_requires_single_rolling_process_checkpoint() -> None:
    cfg = _pretrain_residual_config(Stage2Phase.W)
    del cfg["trainer"]["process_checkpoint_keep_limit"]

    with pytest.raises(ValueError, match="process_checkpoint_keep_limit=1"):
        validate_stage2_training_config(cfg)


def test_adapter_requires_matching_explicit_action_and_data_markers() -> None:
    cfg = _phase_w_config(adapter_04s=True)
    del cfg["datasets"]["vla_data"]["stage2_temporal_contract"]

    with pytest.raises(ValueError, match="must match exactly"):
        validate_stage2_training_config(cfg)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("model_horizon", "action-model contract mismatch"),
        ("flow_horizon", "flow_cfg.horizon_sec"),
        ("data_horizon", "sec_chunk"),
        ("schedule", "schedule must be"),
    ),
)
def test_adapter_rejects_temporal_and_schedule_mixes(mutation: str, error: str) -> None:
    cfg = deepcopy(_phase_w_config(adapter_04s=True))
    if mutation == "model_horizon":
        cfg["framework"]["action_model"]["stage2_horizon_seconds"] = 1.2
    elif mutation == "flow_horizon":
        cfg["framework"]["action_model"]["flow_cfg"]["horizon_sec"] = 1.2
    elif mutation == "data_horizon":
        cfg["datasets"]["vla_data"]["sec_chunk"] = 1.2
    else:
        cfg["trainer"]["max_train_steps"] = 5_000
        cfg["trainer"]["num_warmup_steps"] = 300

    with pytest.raises(ValueError, match=error):
        validate_stage2_training_config(cfg)


def _phase_p_config(*, adapter_04s: bool, checkpoint: Path, selection: Path) -> dict:
    cfg = _phase_w_config(adapter_04s=adapter_04s)
    cfg["framework"]["action_model"]["stage2_phase"] = "P"
    schedule = PHASE_SCHEDULES_04S[Stage2Phase.P] if adapter_04s else PHASE_SCHEDULES[Stage2Phase.P]
    trainer = cfg["trainer"]
    trainer.update(
        max_train_steps=schedule[0],
        num_warmup_steps=schedule[1],
        pretrained_checkpoint=str(checkpoint),
        stage2_parent_phase="W",
        stage2_parent_selection=str(selection),
        strict_finetune_init=True,
        load_pretrained_lam=True,
    )
    trainer["learning_rate"]["vlm"]["lr"] = 5.0e-6
    trainer["freeze"]["unfreeze_lam_decoder"] = False
    return cfg


def _write_parent_contract(
    checkpoint: Path,
    selection: Path,
    *,
    temporal_marker: str | None,
) -> None:
    checkpoint.write_bytes(b"stage2-parent")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "schema_version": STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
        "module_structure": STAGE2_MODULE_STRUCTURE,
        "stage2_phase": "W",
        "checkpoint_filename": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
    }
    if temporal_marker is not None:
        payload["stage2_temporal_contract"] = temporal_marker
    stage2_checkpoint_contract_path(checkpoint).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    selection.write_text(
        json.dumps(
            {
                "schema_version": STAGE2_SELECTION_SCHEMA,
                "stage2_phase": "W",
                "selection_split": "validation",
                "selection_metric": "val_loss_total",
                "selection_mode": "min",
                "test_metrics_read": False,
                "selected": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                },
            }
        ),
        encoding="utf-8",
    )


def test_legacy_parent_sidecar_without_temporal_marker_remains_compatible(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "parent.pt"
    selection = tmp_path / "selection.json"
    _write_parent_contract(checkpoint, selection, temporal_marker=None)
    cfg = _phase_p_config(
        adapter_04s=False,
        checkpoint=checkpoint,
        selection=selection,
    )

    payload = validate_stage2_parent_checkpoint(cfg)

    assert payload is not None and payload["stage2_phase"] == "W"


def test_adapter_parent_sidecar_requires_temporal_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    selection = tmp_path / "selection.json"
    _write_parent_contract(checkpoint, selection, temporal_marker=None)
    cfg = _phase_p_config(
        adapter_04s=True,
        checkpoint=checkpoint,
        selection=selection,
    )

    with pytest.raises(ValueError, match="stage2_temporal_contract"):
        validate_stage2_parent_checkpoint(cfg)


@pytest.mark.parametrize("adapter_04s", (False, True))
def test_preflight_receipt_temporal_marker_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_04s: bool,
) -> None:
    cfg = _phase_w_config(adapter_04s=adapter_04s)
    monkeypatch.setattr(stage2, "_validate_selected_teacher_binding", lambda action: None)
    monkeypatch.setattr(stage2, "_stage2_preflight_asset_paths", lambda cfg, contract: {})
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": STAGE2_LAUNCH_PREFLIGHT_SCHEMA,
                "ready": True,
                "stage2_phase": "W",
                "config_sha256": stage2._config_sha256(cfg),
                "module_structure": STAGE2_MODULE_STRUCTURE,
                "assets": {},
            }
        ),
        encoding="utf-8",
    )

    if adapter_04s:
        with pytest.raises(ValueError, match="stage2_temporal_contract"):
            validate_stage2_launch_preflight_receipt(cfg, receipt)
    else:
        payload = validate_stage2_launch_preflight_receipt(cfg, receipt)
        assert payload["stage2_phase"] == "W"
