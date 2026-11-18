from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.latent_world.deployment_bootstrap import build_checkpoint_action_decoder
from starVLA.model.framework.share_tools import resolve_release_paths
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("robotwin,horizon,width", [(False, 24, 7), (True, 36, 14)])
def test_decoder_bootstrap_needs_only_architecture_and_preserves_state(robotwin, horizon, width):
    filename = ("action_beta_vae_robotwin_v1_12s_dense_s1_a1.yaml" if robotwin
                else "action_beta_vae_libero_v1_12s_a1.yaml")
    config = ROOT / "configs/reference" / filename
    original = build_checkpoint_action_decoder(config, robotwin=robotwin)
    restored = build_checkpoint_action_decoder(config, robotwin=robotwin)
    restored.load_state_dict(original.state_dict(), strict=True)
    action, state, effect = torch.randn(2, 32), torch.randn(2, 14), torch.randn(2, 32)
    with torch.no_grad():
        first = original(action, state, effect)
        second = restored(action, state, effect)
    assert first.shape == (2, horizon, width)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not any(p.requires_grad for p in restored.parameters())


def test_bundle_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_release_paths({"config": "repo://configs/reference/dino_large_vae.yaml",
                                      "stats": ["bundle://stats.json"], "model": "Qwen/model"},
                                     run_dir=tmp_path / "bundle")
    assert resolved["config"] == str(ROOT / "configs/reference/dino_large_vae.yaml")
    assert resolved["stats"] == [str(tmp_path / "bundle/stats.json")]
    assert resolved["model"] == "Qwen/model"


def test_deployment_rejects_missing_learned_modules_even_in_phase_w(tmp_path):
    model = nn.Module()
    model.config = SimpleNamespace(framework=SimpleNamespace(name="test"))
    model.policy_backend = nn.Module()
    backend = model.policy_backend
    backend.stage2_arm = "p2"
    backend.model_cfg = SimpleNamespace(stage2_phase="W", stage2_deployment_bootstrap=True)
    backend.lam = nn.Linear(2, 2)
    backend.flow = nn.Linear(2, 2)
    backend.stage2_latent_affine = nn.Linear(2, 2)
    checkpoint = {k: v for k, v in model.state_dict().items() if "stage2_latent_affine" not in k}
    path = tmp_path / "incomplete.pt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="strict Stage-2 checkpoint structure mismatch"):
        TrainerUtils.load_finetune_init_weights(model, str(path), strict_structure=True)
