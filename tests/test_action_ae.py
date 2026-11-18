from __future__ import annotations

import math
import json
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from latent_action_model.action_ae import (
    ActionAEConfig,
    ActionAELightningModule,
    ActionAutoencoder,
    assert_effect_model_frozen,
    freeze_effect_model,
    mean_so3_geodesic_error,
    prepare_effect_condition,
    trajectory_reconstruction_loss,
)


def _small_config(state_dim: int, variant: str = "a0") -> ActionAEConfig:
    return ActionAEConfig(
        state_dim=state_dim,
        variant=variant,
        hidden_dim=16,
        condition_dim=12,
        dilations=(1, 2),
    )


@pytest.mark.parametrize("state_dim", [5, 14])
def test_a0_shapes_are_deterministic_for_different_state_dims(state_dim: int) -> None:
    torch.manual_seed(3)
    model = ActionAutoencoder(_small_config(state_dim)).eval()
    trajectory = torch.randn(3, 32, 7)
    state = torch.randn(3, state_dim)

    first = model(trajectory, state)
    second = model(trajectory, state)

    assert first.z_action.shape == (3, 1, 32)
    assert first.reconstruction.shape == (3, 32, 7)
    torch.testing.assert_close(first.z_action, second.z_action, rtol=0, atol=0)
    torch.testing.assert_close(first.reconstruction, second.reconstruction, rtol=0, atol=0)
    assert model.config.to_dict()["state_dim"] == state_dim


def test_a0_and_a1_have_equal_capacity() -> None:
    a0 = ActionAutoencoder(_small_config(14, "a0"))
    a1 = ActionAutoencoder(_small_config(14, "a1"))
    assert sum(parameter.numel() for parameter in a0.parameters()) == sum(
        parameter.numel() for parameter in a1.parameters()
    )


def test_state_condition_modulates_both_encoder_and_decoder() -> None:
    torch.manual_seed(5)
    model = ActionAutoencoder(_small_config(6)).eval()
    trajectory = torch.randn(2, 32, 7)
    state_a = torch.zeros(2, 6)
    state_b = torch.ones(2, 6)

    latent_a = model.encode(trajectory, state_a)
    latent_b = model.encode(trajectory, state_b)
    assert not torch.allclose(latent_a, latent_b)

    reconstruction_a = model.decode(latent_a, state_a)
    reconstruction_b = model.decode(latent_a, state_b)
    assert not torch.allclose(reconstruction_a, reconstruction_b)


def test_encoder_masks_invalid_values_before_temporal_convolutions() -> None:
    torch.manual_seed(6)
    model = ActionAutoencoder(_small_config(4)).eval()
    trajectory = torch.randn(2, 32, 7)
    time_mask = torch.ones(2, 32, dtype=torch.bool)
    time_mask[:, -4:] = False
    state = torch.randn(2, 4)
    reference = model.encode(trajectory, state, time_mask=time_mask)

    trajectory[:, -4:] = float("nan")
    with_invalid_padding = model.encode(trajectory, state, time_mask=time_mask)

    assert torch.isfinite(with_invalid_padding).all()
    torch.testing.assert_close(reference, with_invalid_padding)


def test_a1_accepts_e39_layout_and_stops_effect_gradient() -> None:
    torch.manual_seed(8)
    model = ActionAutoencoder(_small_config(9, variant="a1"))
    trajectory = torch.randn(2, 32, 7, requires_grad=True)
    state = torch.randn(2, 9, requires_grad=True)
    effect = torch.randn(2, 1, 32, requires_grad=True)

    output = model(trajectory, state, effect)
    output.reconstruction.square().mean().backward()

    assert output.z_action.shape == (2, 1, 32)
    assert trajectory.grad is not None
    assert state.grad is not None
    assert effect.grad is None
    assert prepare_effect_condition(effect).shape == (2, 32)
    assert not prepare_effect_condition(effect).requires_grad


def test_a0_a1_condition_contracts_fail_fast() -> None:
    trajectory = torch.randn(2, 32, 7)
    state = torch.randn(2, 4)
    effect = torch.randn(2, 32)
    with pytest.raises(ValueError, match="does not accept"):
        ActionAutoencoder(_small_config(4, "a0"))(trajectory, state, effect)
    with pytest.raises(ValueError, match="requires"):
        ActionAutoencoder(_small_config(4, "a1"))(trajectory, state)
    with pytest.raises(ValueError, match="current_state"):
        ActionAutoencoder(_small_config(4))(trajectory, torch.randn(2, 5))


def test_decoder_output_is_not_tanh_bounded() -> None:
    model = ActionAutoencoder(_small_config(3))
    with torch.no_grad():
        model.decoder.output_projection.weight.zero_()
        model.decoder.output_projection.bias.fill_(2.5)
    output = model(torch.zeros(1, 32, 7), torch.zeros(1, 3))
    torch.testing.assert_close(output.reconstruction, torch.full((1, 32, 7), 2.5))


def test_effect_model_freeze_helper_and_assertion() -> None:
    effect_model = nn.Sequential(nn.Linear(4, 32), nn.Dropout(p=0.5))
    effect_model.train()
    returned = freeze_effect_model(effect_model)

    assert returned is effect_model
    assert not effect_model.training
    assert all(not parameter.requires_grad for parameter in effect_model.parameters())
    assert_effect_model_frozen(effect_model)

    effect_model[0].weight.requires_grad_(True)
    with pytest.raises(AssertionError, match="trainable"):
        assert_effect_model_frozen(effect_model)
    effect_model[0].weight.requires_grad_(False)
    effect_model[1].train()
    with pytest.raises(AssertionError, match="eval mode"):
        assert_effect_model_frozen(effect_model)


def test_masked_modality_losses_normalize_by_valid_elements() -> None:
    prediction = torch.zeros(1, 2, 7, requires_grad=True)
    target = torch.full((1, 2, 7), float("nan"))
    target[:, 0, :3] = torch.tensor([0.2, -0.2, 0.4])
    target[:, 0, 3:6] = torch.tensor([0.05, -0.05, 0.1])
    target[:, 0, 6] = 0.3
    valid = torch.tensor([[True, False]])

    result = trajectory_reconstruction_loss(
        prediction,
        target,
        time_mask=valid,
        pose_mask=torch.ones_like(valid),
        gripper_mask=torch.ones_like(valid),
    )
    expected_position = functional.smooth_l1_loss(
        prediction[:, 0, :3], target[:, 0, :3], beta=0.1, reduction="mean"
    )
    expected_rotation = functional.smooth_l1_loss(
        prediction[:, 0, 3:6], target[:, 0, 3:6], beta=0.1, reduction="mean"
    )
    expected_gripper = functional.smooth_l1_loss(
        prediction[:, 0, 6:7], target[:, 0, 6:7], beta=0.1, reduction="mean"
    )

    torch.testing.assert_close(result.position, expected_position)
    torch.testing.assert_close(result.rotation, expected_rotation)
    torch.testing.assert_close(result.gripper, expected_gripper)
    torch.testing.assert_close(result.total, expected_position + expected_rotation + 0.5 * expected_gripper)
    assert result.position_valid_elements.item() == 3
    assert result.rotation_valid_elements.item() == 3
    assert result.gripper_valid_elements.item() == 1
    result.total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_absent_gripper_has_zero_loss_without_affecting_pose() -> None:
    prediction = torch.zeros(2, 32, 7)
    target = torch.ones_like(prediction)
    result = trajectory_reconstruction_loss(
        prediction,
        target,
        time_mask=torch.ones(2, 32, dtype=torch.bool),
        pose_mask=torch.ones(2, 32, dtype=torch.bool),
        gripper_mask=torch.zeros(2, 32, dtype=torch.bool),
    )

    assert result.gripper.item() == 0.0
    assert result.gripper_valid_elements.item() == 0
    torch.testing.assert_close(result.total, result.position + result.rotation)


def test_so3_geodesic_error_uses_shortest_rotation_and_mask() -> None:
    predicted = torch.tensor([[[0.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi]]], dtype=torch.float64)
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -math.pi]]], dtype=torch.float64)

    first_only = mean_so3_geodesic_error(
        predicted,
        target,
        mask=torch.tensor([[True, False]]),
    )
    antipodal = mean_so3_geodesic_error(predicted[:, 1:], target[:, 1:], degrees=False)

    torch.testing.assert_close(first_only, torch.tensor(90.0, dtype=torch.float64), atol=1.0e-8, rtol=0)
    torch.testing.assert_close(antipodal, torch.tensor(0.0, dtype=torch.float64), atol=1.0e-8, rtol=0)
    assert torch.isnan(
        mean_so3_geodesic_error(predicted, target, mask=torch.zeros(1, 2, dtype=torch.bool))
    )


def test_lightning_checkpoint_embeds_normalization_and_model_contract(tmp_path) -> None:
    normalization = {
        "fit_split": "train",
        "position": {"median": [0.0] * 3, "scale_mad": [1.0] * 3},
        "rotation_rotvec": {"median": [0.0] * 3, "scale_mad": [1.0] * 3},
        "gripper": {"median": [0.0], "scale_mad": [1.0]},
    }
    sidecar_root = tmp_path / "sidecar"
    path = sidecar_root / "stats" / "normalization.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(normalization), encoding="utf-8")
    (sidecar_root / "schema.json").write_text('{}\n', encoding="utf-8")
    (sidecar_root / "checksums.sha256").write_text('test\n', encoding="utf-8")
    (sidecar_root / "provenance.json").write_text(
        json.dumps(
            {
                "effect_manifest_sha256": "a" * 64,
                "e39": {
                    "sha256": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    module = ActionAELightningModule(
        path,
        variant="a1",
        hidden_dim=16,
        condition_dim=12,
        dilations=(1, 2),
    )
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    metadata = checkpoint["action_ae_metadata"]
    assert metadata["schema_version"] == "action-tcn-ae-v0"
    assert metadata["model_config"]["variant"] == "a1"
    assert metadata["model_config"]["latent_dim"] == 32
    assert metadata["normalization"] == normalization
    assert len(metadata["normalization_sha256"]) == 64
    assert len(metadata["sidecar_provenance_sha256"]) == 64
    assert metadata["effect_manifest_sha256"] == "a" * 64
    assert metadata["e39_cache_sha256"] == "b" * 64
    assert metadata["e39_checkpoint_sha256"] == "c" * 64


def test_validation_skips_gripper_metrics_when_batch_has_no_gripper(tmp_path) -> None:
    normalization = {
        "fit_split": "train",
        "position": {"median": [0.0] * 3, "scale_mad": [1.0] * 3},
        "rotation_rotvec": {"median": [0.0] * 3, "scale_mad": [1.0] * 3},
        "gripper": {"median": [0.0], "scale_mad": [1.0]},
    }
    path = tmp_path / "stats" / "normalization.json"
    path.parent.mkdir()
    path.write_text(json.dumps(normalization), encoding="utf-8")
    module = ActionAELightningModule(
        path,
        hidden_dim=16,
        condition_dim=12,
        dilations=(1, 2),
    )
    module.log = Mock()
    batch = {
        "trajectory": torch.zeros(2, 32, 7),
        "trajectory_raw": torch.zeros(2, 32, 7),
        "current_state": torch.zeros(2, 14),
        "e39_mean": torch.zeros(2, 32),
        "e39_valid": torch.ones(2, dtype=torch.bool),
        "time_mask": torch.ones(2, 32, dtype=torch.bool),
        "pose_mask": torch.ones(2, 32, dtype=torch.bool),
        "gripper_mask": torch.zeros(2, 32, dtype=torch.bool),
    }

    loss = module.validation_step(batch, 0)

    assert torch.isfinite(loss)
    logged_names = {call.args[0] for call in module.log.call_args_list}
    assert "val/gripper_loss" not in logged_names
    assert "val/gripper_mae_m" not in logged_names
    for call in module.log.call_args_list:
        assert torch.isfinite(call.args[1]).all()
