from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

import starVLA.training.train_starvla as train_starvla
from starVLA.training.train_starvla import (
    VLATrainer,
    prune_process_checkpoints,
    should_save_process_checkpoint,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config


def test_training_state_contract_records_all_stage2_loss_weights(tmp_path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint-contract")
    action_model = {
        "stage2_arm": "p2",
        "stage2_phase": "P",
        "stage2_training_recipe": "robotwin_sft_release_lawam_action_pe_70k_v2",
        "stage2_flow_weight": 1.0,
        "stage2_effect_distill_weight": 0.0,
        "stage2_action_distill_weight": 0.5,
        "stage2_dino_weight": 0.1,
        "stage2_lawam_distill_weight": 0.1,
        "stage2_action_kl_weight": 1.0,
        "stage2_action_mean_weight": 0.5,
        "stage2_action_trajectory_weight": 0.2,
        "stage2_residual_warmup_start_step": 12_000,
        "stage2_residual_teacher_prob_start": 0.5,
        "stage2_residual_teacher_prob_end": 0.0,
        "stage2_residual_teacher_ramp_steps": 2_000,
        "stage2_residual_predictor_grad_start_step": 14_000,
        "stage2_residual_condition_dropout": 0.0,
    }
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "run_id": "lawam-action-resume-contract",
            "seed": 2026,
            "framework": {"action_model": action_model},
            "datasets": {"vla_data": {"per_device_batch_size": 32}},
            "trainer": {
                "pretrained_checkpoint": str(init_checkpoint),
                "max_train_steps": 20_000,
                "gradient_accumulation_steps": 4,
            },
        }
    )
    trainer.accelerator = SimpleNamespace(gradient_accumulation_steps=4, num_processes=8)
    trainer.vla_train_dataloader = [None] * 7

    contract = trainer._training_state_contract()

    assert contract["stage2_losses"] == {
        "flow": 1.0,
        "effect_distill": 0.0,
        "action_distill": 0.5,
        "dino": 0.1,
        "lawam_distill": 0.1,
        "action_kl": 1.0,
        "action_mean": 0.5,
        "action_trajectory": 0.2,
    }


def test_process_checkpoint_pruning_keeps_latest_and_preserves_named_outputs(
    tmp_path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for step in (500, 1000, 1500):
        checkpoint = checkpoints / f"steps_{step}_pytorch_model.pt"
        checkpoint.write_bytes(str(step).encode("ascii"))
        checkpoint.with_name(f"{checkpoint.name}.stage2.json").write_text(
            "{}", encoding="utf-8"
        )
    best = checkpoints / "best_validation_pytorch_model.pt"
    best.write_bytes(b"best")
    training_state = checkpoints / "steps_1000_training_state"
    training_state.mkdir()

    deleted = prune_process_checkpoints(
        checkpoints,
        keep_limit=1,
        preserve_steps=(1000,),
    )

    assert [path.name for path in deleted] == [
        "steps_500_pytorch_model.pt",
    ]
    assert (checkpoints / "steps_1000_pytorch_model.pt").is_file()
    assert (
        checkpoints / "steps_1000_pytorch_model.pt.stage2.json"
    ).is_file()
    assert (checkpoints / "steps_1500_pytorch_model.pt").is_file()
    assert (
        checkpoints / "steps_1500_pytorch_model.pt.stage2.json"
    ).is_file()
    assert best.is_file()
    assert training_state.is_dir()


def test_process_checkpoint_pruning_rejects_invalid_preserved_step(tmp_path) -> None:
    with pytest.raises(ValueError, match="preserve_steps must be positive"):
        prune_process_checkpoints(tmp_path, keep_limit=1, preserve_steps=(0,))


def test_named_checkpoint_milestone_does_not_change_periodic_interval() -> None:
    milestones = (1250, 2500, 5000, 12500, 25000)

    assert should_save_process_checkpoint(
        1250,
        save_interval=2500,
        preserve_steps=milestones,
    )
    assert should_save_process_checkpoint(
        2500,
        save_interval=2500,
        preserve_steps=milestones,
    )
    assert not should_save_process_checkpoint(
        3750,
        save_interval=2500,
        preserve_steps=milestones,
    )


def test_named_checkpoint_milestone_validation() -> None:
    with pytest.raises(ValueError, match="save_interval must be > 0"):
        should_save_process_checkpoint(1, save_interval=0)
    with pytest.raises(ValueError, match="must contain positive steps"):
        should_save_process_checkpoint(1, save_interval=10, preserve_steps=(0,))


def test_required_online_wandb_failure_is_fatal(monkeypatch, tmp_path) -> None:
    trainer = object.__new__(VLATrainer)
    trainer.use_wandb = True
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.config = SimpleNamespace(
        output_dir=str(tmp_path),
        wandb_mode="online",
        wandb_required=True,
        wandb_project="robotwin-test",
        wandb_entity=None,
    )
    monkeypatch.setattr(
        train_starvla.wandb,
        "init",
        lambda **_: (_ for _ in ()).throw(RuntimeError("not logged in")),
    )

    with pytest.raises(RuntimeError, match="Required W&B online initialization failed"):
        trainer._init_wandb()


@pytest.mark.parametrize("libero_phase", [None, "w", "p", "e"])
def test_complete_training_state_round_trip_and_data_cursor(tmp_path, libero_phase) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint-contract")
    cfg = OmegaConf.create(
        {
            "run_id": "resume-smoke",
            "seed": 17,
            "output_dir": str(tmp_path),
            "framework": {"action_model": {"stage2_arm": "p2"}},
            "datasets": {"vla_data": {"per_device_batch_size": 2}},
            "trainer": {
                "pretrained_checkpoint": str(init_checkpoint),
                "max_train_steps": 20,
                "gradient_accumulation_steps": 1,
                "save_training_state": True,
            },
        }
    )
    if libero_phase is not None:
        template = Path(__file__).resolve().parents[2] / "latent_action_model/config" / f"libero_stage2_release_residual_phase_{libero_phase}.yaml"
        cfg.framework.action_model = OmegaConf.load(template).framework.action_model
        assert "stage2_lawam_distill_weight" not in cfg.framework.action_model
        cfg.framework.action_model.stage2_action_head_architecture = "flatten_mlp_v1"
        cfg = wrap_config(cfg)
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lambda step: 1.0 - step / 100.0)
    loader = DataLoader(
        TensorDataset(torch.arange(10, dtype=torch.float32).reshape(10, 1)),
        batch_size=2,
        shuffle=False,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    trainer = VLATrainer(cfg, model, loader, optimizer, scheduler, accelerator)
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    (tmp_path / "checkpoints").mkdir()

    loss = model(torch.ones(2, 1)).sum()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    trainer.completed_steps = 3
    expected_model = {
        key: value.detach().clone()
        for key, value in accelerator.unwrap_model(model).state_dict().items()
    }
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())
    trainer._save_training_state()
    if libero_phase is not None:
        assert trainer._training_state_contract()["stage2_losses"]["lawam_distill"] == 0.0

    state_dir = tmp_path / "checkpoints" / "steps_3_training_state"
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    optimizer.param_groups[0]["lr"] = 123.0
    scheduler.last_epoch = 999

    trainer._restore_training_state(str(state_dir))
    actual_model = accelerator.unwrap_model(model).state_dict()
    assert trainer.completed_steps == 3
    assert all(torch.equal(expected_model[key], actual_model[key]) for key in expected_model)
    assert optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, expected_state in expected_optimizer["state"].items():
        for key, expected_value in expected_state.items():
            actual_value = optimizer.state_dict()["state"][parameter_id][key]
            if torch.is_tensor(expected_value):
                assert torch.equal(expected_value, actual_value)
            else:
                assert expected_value == actual_value
    assert scheduler.state_dict() == expected_scheduler

    trainer._create_data_iterators()
    resumed_batch = trainer._get_next_batch()[0]
    assert torch.equal(resumed_batch.cpu(), torch.tensor([[6.0], [7.0]]))
