from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework import base_framework as base_framework_module
from starVLA.model.framework.base_framework import baseframework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class _Lam(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.register_buffer("running_scale", torch.zeros(1))


class _Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = _Lam()
        self.lam_extra = nn.Linear(2, 2, bias=False)
        self.flow = nn.Linear(2, 2, bias=False)
        self.stage2_arm = "p2"
        self.model_cfg = SimpleNamespace(stage2_phase="E")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_backend = _Backend()
        self.config = SimpleNamespace(framework=SimpleNamespace(name="test"))


def _fill(model: _Model, *, lam: float, lam_extra: float, flow: float) -> None:
    with torch.no_grad():
        model.policy_backend.lam.projection.weight.fill_(lam)
        model.policy_backend.lam.running_scale.fill_(lam)
        model.policy_backend.lam_extra.weight.fill_(lam_extra)
        model.policy_backend.flow.weight.fill_(flow)


def test_load_pretrained_lam_defaults_to_legacy_overwrite(tmp_path):
    checkpoint_model = _Model()
    _fill(checkpoint_model, lam=1.0, lam_extra=2.0, flow=3.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_model.state_dict(), checkpoint_path)

    model = _Model()
    _fill(model, lam=7.0, lam_extra=8.0, flow=9.0)
    TrainerUtils.load_finetune_init_weights(model, str(checkpoint_path))

    assert torch.all(model.policy_backend.lam.projection.weight == 1.0)
    assert torch.all(model.policy_backend.lam.running_scale == 1.0)
    assert torch.all(model.policy_backend.lam_extra.weight == 2.0)
    assert torch.all(model.policy_backend.flow.weight == 3.0)


def test_disable_pretrained_lam_preserves_exact_prefix_only(tmp_path):
    checkpoint_model = _Model()
    _fill(checkpoint_model, lam=1.0, lam_extra=2.0, flow=3.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_model.state_dict(), checkpoint_path)

    model = _Model()
    _fill(model, lam=7.0, lam_extra=8.0, flow=9.0)
    TrainerUtils.load_finetune_init_weights(
        model,
        str(checkpoint_path),
        load_pretrained_lam=False,
    )

    assert torch.all(model.policy_backend.lam.projection.weight == 7.0)
    assert torch.all(model.policy_backend.lam.running_scale == 7.0)
    assert torch.all(model.policy_backend.lam_extra.weight == 2.0)
    assert torch.all(model.policy_backend.flow.weight == 3.0)


def test_strict_stage2_load_verifies_and_removes_framework_aliases(tmp_path):
    checkpoint_model = _Model()
    _fill(checkpoint_model, lam=1.0, lam_extra=2.0, flow=3.0)
    checkpoint = checkpoint_model.state_dict()
    checkpoint["policy_action_head.weight"] = checkpoint[
        "policy_backend.flow.weight"
    ]
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    model = _Model()
    _fill(model, lam=7.0, lam_extra=8.0, flow=9.0)
    TrainerUtils.load_finetune_init_weights(
        model,
        str(checkpoint_path),
        strict_structure=True,
    )

    assert torch.all(model.policy_backend.lam.projection.weight == 1.0)
    assert torch.all(model.policy_backend.lam_extra.weight == 2.0)
    assert torch.all(model.policy_backend.flow.weight == 3.0)


def test_strict_stage2_load_rejects_unequal_framework_alias(tmp_path):
    checkpoint_model = _Model()
    checkpoint = checkpoint_model.state_dict()
    checkpoint["policy_action_head.weight"] = torch.full_like(
        checkpoint["policy_backend.flow.weight"], 99.0
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="framework alias mismatch"):
        TrainerUtils.load_finetune_init_weights(
            _Model(),
            str(checkpoint_path),
            strict_structure=True,
        )


def test_phased_from_pretrained_bootstraps_before_framework_build(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "phase-e.pt"
    checkpoint_path.parent.mkdir()
    torch.save({}, checkpoint_path)
    checkpoint_path.with_name(f"{checkpoint_path.name}.stage2.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    config = {
        "framework": {
            "name": "LaWAM",
            "action_model": {"stage2_phase": "E"},
        },
        "trainer": {"pretrained_checkpoint": "/training/parent.pt"},
    }
    captured = {}

    class _BuiltModel(nn.Module):
        pass

    def _build_framework(*, cfg):
        captured["bootstrap"] = bool(
            cfg.framework.action_model.stage2_deployment_bootstrap
        )
        return _BuiltModel()

    monkeypatch.setattr(
        base_framework_module,
        "read_mode_config",
        lambda _: (config, {"agilex": {}}),
    )
    monkeypatch.setattr(
        "starVLA.model.framework.build_framework",
        _build_framework,
    )
    monkeypatch.setattr(
        "starVLA.training.trainer_utils.trainer_tools.apply_training_freeze_policy",
        lambda model, _: model,
    )
    monkeypatch.setattr(
        TrainerUtils,
        "load_finetune_init_weights",
        staticmethod(lambda model, *_args, **_kwargs: model),
    )

    model = baseframework.from_pretrained(str(checkpoint_path))

    assert isinstance(model, _BuiltModel)
    assert captured == {"bootstrap": True}
    assert model.norm_stats == {"agilex": {}}
