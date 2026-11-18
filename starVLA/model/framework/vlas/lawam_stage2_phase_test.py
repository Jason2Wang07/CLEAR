from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from latent_action_model.action_ae.distributions import DiagonalGaussian
from latent_action_model.robotwin_stage2.paired import (
    ActionResidualProjection,
    PairedExpertAdapters,
    PairedTokens,
    RELEASE_RESIDUAL_RECIPE,
)
from starVLA.model.framework.vlas.lawam import (
    LatentWorldPolicyBackend,
    PolicyEncodingState,
)
from starVLA.model.framework.latent_world.runtime.output_mapper import (
    map_policy_train_output,
)


class _TestLatentAffine(nn.Module):
    def effect_to_raw(self, value: Tensor) -> Tensor:
        return 1.5 * value + 0.25

    def action_to_raw(self, value: Tensor) -> Tensor:
        return 2.0 * value - 0.5

    def condition_to_standardized(self, value: Tensor) -> Tensor:
        return (value - 0.25) / 1.5


class _TestFrozenDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(
        self,
        action: Tensor,
        current_state: Tensor,
        effect: Tensor,
    ) -> Tensor:
        step = (
            action[:, :7]
            + 0.5 * effect[:, :7]
            + 0.1 * current_state[:, :7]
        )
        return self.scale * step.unsqueeze(1).expand(-1, 24, -1)


class _RecordingActionPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.last_inputs: dict[str, Tensor] = {}

    def predict_action(
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
        self.last_inputs = {
            "h_plan": h_plan,
            "current_visual": current_visual,
            "future_visual": future_visual,
            "state": state,
            "effect": effect,
            "embodiment_id": embodiment_id,
            "horizon_condition": horizon_condition,
        }
        batch = h_plan.shape[0]
        value = (
            h_plan.mean(dim=(1, 2))
            + current_visual.mean(dim=(1, 2))
            + future_visual.mean(dim=(1, 2))
            + state.mean(dim=-1)
            + effect.mean(dim=-1)
        )
        mean = (self.scale * value).unsqueeze(-1).expand(batch, 32)
        return DiagonalGaussian(mean=mean, logvar=mean - 0.3)


class _TestPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.effect_head = nn.Linear(2, 2)
        self.action_head = nn.Linear(2, 2)
        self.unowned_parameter = nn.Parameter(torch.ones(()))


class _TestLam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


class _EncodingLam(nn.Module):
    def extract_vision_features(self, visual: Tensor) -> Tensor:
        batch = int(visual.shape[0])
        return torch.zeros(batch, 2, 2, 4, device=visual.device)


class _GradientProbeFlow(nn.Module):
    def forward(self, **inputs: Tensor) -> Tensor:
        return inputs["h_vlm"].sum() + inputs["h_t1_star"].sum()


class _RecordingFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_inputs: dict[str, Tensor | None] = {}

    def forward(self, **inputs: Tensor | None) -> Tensor:
        self.last_inputs = inputs
        residual = inputs["action_residual"]
        assert isinstance(residual, Tensor)
        return residual.sum()


class _RecordingResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_condition: Tensor | None = None

    def forward(self, condition: Tensor) -> Tensor:
        self.last_condition = condition
        return torch.ones(condition.shape[0], 2, device=condition.device)


def _bare_backend() -> LatentWorldPolicyBackend:
    backend = LatentWorldPolicyBackend.__new__(LatentWorldPolicyBackend)
    nn.Module.__init__(backend)
    return backend


def _encoding_state(
    *,
    predicted_effect: Tensor,
    action_mean: Tensor,
    action_logvar: Tensor,
) -> PolicyEncodingState:
    batch = predicted_effect.shape[0]
    return PolicyEncodingState(
        h_vlm=torch.zeros(batch, 3, 4),
        h_effect=torch.zeros(batch, 8, 4),
        h_plan=torch.zeros(batch, 8, 4),
        pred_action_emb=torch.zeros(batch, 1, 32),
        h_t=torch.zeros(batch, 2, 4),
        h_t1_pred=torch.zeros(batch, 2, 4),
        h_t1_gt=torch.zeros(batch, 2, 4),
        h_t_original=torch.zeros(batch, 2, 4),
        predicted_effect=predicted_effect,
        paired_tokens=PairedTokens(
            effect=predicted_effect,
            action=action_mean,
            action_logvar=action_logvar,
        ),
    )


def _plan_batch(batch: int = 2) -> dict[str, Tensor]:
    return {
        "z_action": torch.full((batch, 32), 0.35),
        "z_action_logvar": torch.full((batch, 32), -0.4),
        "teacher_trajectory": torch.full((batch, 24, 7), 0.15),
        "teacher_current_state": torch.linspace(-0.2, 0.5, 14).repeat(batch, 1),
        "teacher_time_mask": torch.ones(batch, 24, dtype=torch.bool),
        "teacher_pose_mask": torch.ones(batch, 24, dtype=torch.bool),
        "teacher_gripper_mask": torch.ones(batch, 24, dtype=torch.bool),
        "teacher_current_state_mask": torch.ones(batch, 14, dtype=torch.bool),
    }


def test_stage2_plan_loss_has_exact_composition_and_gradient_boundary() -> None:
    backend = _bare_backend()
    backend._phased_stage2 = True
    backend.stage2_phase = "P"
    backend.model_cfg = SimpleNamespace(
        stage2_action_kl_weight=1.0,
        stage2_action_mean_weight=0.5,
        stage2_action_trajectory_weight=0.2,
    )
    backend.stage2_latent_affine = _TestLatentAffine()
    backend.frozen_action_decoder = _TestFrozenDecoder().requires_grad_(False)

    predicted_effect = torch.full((2, 32), 0.2, requires_grad=True)
    action_mean = torch.full((2, 32), -0.1, requires_grad=True)
    action_logvar = torch.full((2, 32), 0.25, requires_grad=True)
    shared = _encoding_state(
        predicted_effect=predicted_effect,
        action_mean=action_mean,
        action_logvar=action_logvar,
    )

    losses = backend._compute_stage2_plan_losses(
        prepared_batch=_plan_batch(),
        shared=shared,
    )

    expected = (
        losses["loss_action_kl"]
        + 0.5 * losses["loss_action_mean"]
        + 0.2 * losses["loss_action_trajectory"]
    )
    torch.testing.assert_close(losses["loss_plan"], expected, rtol=0.0, atol=0.0)

    losses["loss_plan"].backward()
    assert action_mean.grad is not None and torch.count_nonzero(action_mean.grad) > 0
    assert action_logvar.grad is not None and torch.count_nonzero(action_logvar.grad) > 0
    assert predicted_effect.grad is None
    assert backend.frozen_action_decoder.scale.grad is None


def test_lawam_action_plan_loss_detaches_raw_condition() -> None:
    backend = _bare_backend()
    backend._phased_stage2 = True
    backend._lawam_action_recipe = True
    backend.stage2_phase = "P"
    backend.model_cfg = SimpleNamespace(
        stage2_action_kl_weight=1.0,
        stage2_action_mean_weight=0.5,
        stage2_action_trajectory_weight=0.2,
    )
    backend.stage2_latent_affine = _TestLatentAffine()
    backend.frozen_action_decoder = _TestFrozenDecoder().requires_grad_(False)

    raw_lawam = torch.full((2, 1, 32), 0.4, requires_grad=True)
    action_mean = torch.full((2, 32), -0.1, requires_grad=True)
    action_logvar = torch.full((2, 32), 0.25, requires_grad=True)
    shared = _encoding_state(
        predicted_effect=torch.full((2, 32), 0.1, requires_grad=True),
        action_mean=action_mean,
        action_logvar=action_logvar,
    )
    shared.pred_action_emb = raw_lawam

    losses = backend._compute_stage2_plan_losses(
        prepared_batch=_plan_batch(),
        shared=shared,
    )
    losses["loss_plan"].backward()

    assert action_mean.grad is not None and torch.count_nonzero(action_mean.grad) > 0
    assert action_logvar.grad is not None and torch.count_nonzero(action_logvar.grad) > 0
    assert raw_lawam.grad is None


def test_stage2_action_permanently_detaches_transition_and_effect_inputs() -> None:
    backend = _bare_backend()
    predictor = _RecordingActionPredictor()
    backend._stage2_token_override = None
    backend.stage2_arm = "p2"
    backend._phased_stage2 = True
    backend.stage2_phase = "P"
    backend.paired_latent_predictor = predictor
    backend.model_cfg = SimpleNamespace(
        stage2_action_state_source="starvla32",
        stage2_action_inference="mean",
        stage2_horizon_seconds=1.2,
    )

    h_plan = torch.randn(2, 8, 4, requires_grad=True)
    effect = torch.randn(2, 32, requires_grad=True)
    current = torch.randn(2, 256, 3, requires_grad=True)
    future = torch.randn(2, 256, 3, requires_grad=True)
    state = torch.randn(2, 32)
    batch = {
        "state": state,
        "embodiment_id": torch.tensor([0, 1], dtype=torch.long),
        "action_hz": torch.tensor([20.0, 20.0]),
        "teacher_time_mask": torch.ones(2, 24, dtype=torch.bool),
    }

    paired = backend._resolve_stage2_action(
        prepared_batch=batch,
        h_plan=h_plan,
        effect=effect,
        h_t=current,
        h_t1_pred=future,
    )
    torch.testing.assert_close(
        predictor.last_inputs["horizon_condition"],
        torch.tensor([[20.0, 1.2, 24.0], [20.0, 1.2, 24.0]]),
    )
    paired.action.sum().backward()

    assert h_plan.grad is not None and torch.count_nonzero(h_plan.grad) > 0
    assert effect.grad is None
    assert current.grad is None
    assert future.grad is None
    assert predictor.last_inputs["effect"].requires_grad is False
    assert predictor.last_inputs["current_visual"].requires_grad is False
    assert predictor.last_inputs["future_visual"].requires_grad is False


def test_stage2_04s_horizon_uses_physical_count_not_representation_mask() -> None:
    backend = _bare_backend()
    backend.model_cfg = SimpleNamespace(
        stage2_temporal_contract="libero_baseline_04s_adapter_v1",
        stage2_horizon_seconds=0.4,
    )
    batch = {
        "action_hz": torch.tensor([20.0, 20.0]),
        "teacher_time_mask": torch.ones(2, 24, dtype=torch.bool),
        "physical_valid_count": torch.tensor([8, 8]),
        "actions": torch.zeros(2, 8, 7),
    }

    condition = backend._stage2_horizon_condition(batch)

    torch.testing.assert_close(
        condition,
        torch.tensor([[20.0, 0.4, 8.0], [20.0, 0.4, 8.0]]),
    )


def test_stage2_04s_inference_resolves_physical_count_from_contract() -> None:
    backend = _bare_backend()
    backend.model_cfg = SimpleNamespace(
        stage2_temporal_contract="libero_baseline_04s_adapter_v1",
        stage2_horizon_seconds=0.4,
    )

    condition = backend._stage2_horizon_condition(
        {"action_hz": torch.tensor([20.0])}
    )

    torch.testing.assert_close(condition, torch.tensor([[20.0, 0.4, 8.0]]))


@pytest.mark.parametrize(
    "temporal_contract",
    ("robotwin_official_12s_v1", "robotwin_official_12s_dense_s1_v1"),
)
def test_stage2_robotwin_horizon_condition_is_30hz_12s_h36(
    temporal_contract: str,
) -> None:
    backend = _bare_backend()
    backend.model_cfg = SimpleNamespace(
        stage2_temporal_contract=temporal_contract,
        stage2_horizon_seconds=1.2,
    )
    batch = {
        "action_hz": torch.tensor([30.0, 30.0]),
        "teacher_time_mask": torch.ones(2, 36, dtype=torch.bool),
        "physical_valid_count": torch.tensor([36, 36]),
    }

    condition = backend._stage2_horizon_condition(batch)

    torch.testing.assert_close(
        condition,
        torch.tensor([[30.0, 1.2, 36.0], [30.0, 1.2, 36.0]]),
    )


@pytest.mark.parametrize(
    ("action_hz", "valid_count", "mask_count", "error"),
    (
        (20.0, 36, 36, "exactly 30"),
        (30.0, 35, 36, "exactly 36"),
        (30.0, 36, 35, "all 36 slots"),
    ),
)
def test_stage2_robotwin_horizon_rejects_temporal_drift(
    action_hz: float,
    valid_count: int,
    mask_count: int,
    error: str,
) -> None:
    backend = _bare_backend()
    backend.model_cfg = SimpleNamespace(
        stage2_temporal_contract="robotwin_official_12s_v1",
        stage2_horizon_seconds=1.2,
    )
    time_mask = torch.zeros(1, 36, dtype=torch.bool)
    time_mask[:, :mask_count] = True
    batch = {
        "action_hz": torch.tensor([action_hz]),
        "teacher_time_mask": time_mask,
        "physical_valid_count": torch.tensor([valid_count]),
    }

    with pytest.raises(ValueError, match=error):
        backend._stage2_horizon_condition(batch)


@pytest.mark.parametrize(
    ("seconds", "physical_count", "mask_count", "error"),
    (
        (1.2, 8, 24, "stage2_horizon_seconds is incompatible"),
        (0.4, None, 24, "require physical_valid_count"),
        (0.4, 24, 24, "must be exactly 8"),
        (0.4, 8, 8, "keep all 24 representation slots"),
    ),
)
def test_stage2_04s_horizon_rejects_mixed_contracts(
    seconds: float,
    physical_count: int | None,
    mask_count: int,
    error: str,
) -> None:
    backend = _bare_backend()
    backend.model_cfg = SimpleNamespace(
        stage2_temporal_contract="libero_baseline_04s_adapter_v1",
        stage2_horizon_seconds=seconds,
    )
    time_mask = torch.zeros(1, 24, dtype=torch.bool)
    time_mask[:, :mask_count] = True
    batch = {
        "action_hz": torch.tensor([20.0]),
        "teacher_time_mask": time_mask,
        "actions": torch.zeros(1, 8, 7),
    }
    if physical_count is not None:
        batch["physical_valid_count"] = torch.tensor([physical_count])

    with pytest.raises((KeyError, ValueError), match=error):
        backend._stage2_horizon_condition(batch)


@pytest.mark.parametrize("training", (True, False))
def test_phase_w_never_executes_random_action_predictor(training: bool) -> None:
    backend = _bare_backend()
    backend.training = training
    backend._phased_stage2 = True
    backend.stage2_phase = "W"
    backend.model_cfg = SimpleNamespace(
        vlm_dtype=torch.float32,
        future_prediction=True,
    )
    backend.stage2_latent_affine = _TestLatentAffine()
    backend.lam = _EncodingLam()
    backend._prepare_queries = lambda **_: (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    backend._run_vlm_stage = lambda **_: {
        "h_vlm": torch.zeros(1, 2, 4),
        "h_effect": torch.zeros(1, 1, 4),
        "h_plan": torch.zeros(1, 1, 4),
        "pred_latent": torch.zeros(1, 1, 32),
    }
    backend._resolve_stage2_effect = lambda **_: torch.zeros(1, 32)
    backend._decode_future_tokens_strict_single_query = (
        lambda **_: torch.zeros(1, 2, 4)
    )

    action_calls = 0

    def _unexpected_action_call(**_: object) -> PairedTokens:
        nonlocal action_calls
        action_calls += 1
        raise AssertionError("Phase W must not execute its random action predictor")

    backend._resolve_stage2_action = _unexpected_action_call
    prepared_batch = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "act_placeholder_mask": torch.ones(1, 1, dtype=torch.bool),
        "flow_placeholder_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    shared = backend._run_shared_encoding_core(
        prepared_batch=prepared_batch,
        primary_visual_input=torch.zeros(1, 2, 3, 2, 2),
        source="phase-w-regression-test",
        lam_features_with_no_grad=True,
    )

    assert shared.paired_tokens is None
    assert action_calls == 0


def test_stage2_flow_condition_cannot_backpropagate_through_predicted_future() -> None:
    backend = _bare_backend()
    backend._phased_stage2 = True
    backend.model_cfg = SimpleNamespace(
        flow_only_mode=False,
        repeated_diffusion_steps=1,
    )
    backend.flow = _GradientProbeFlow()
    backend.paired_expert_adapters = None
    backend._release_residual_recipe = False

    h_vlm = torch.randn(1, 3, 4, requires_grad=True)
    h_t1_pred = torch.randn(1, 2, 4, requires_grad=True)
    shared = PolicyEncodingState(
        h_vlm=h_vlm,
        h_effect=torch.zeros(1, 8, 4),
        h_plan=torch.zeros(1, 8, 4),
        pred_action_emb=torch.zeros(1, 1, 32),
        h_t=torch.zeros(1, 2, 4),
        h_t1_pred=h_t1_pred,
        h_t1_gt=torch.zeros(1, 2, 4),
        h_t_original=torch.zeros(1, 2, 4),
        predicted_effect=torch.zeros(1, 32),
        paired_tokens=None,
    )
    batch = {
        "act_placeholder_mask": torch.zeros(1, 3, dtype=torch.bool),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "state": torch.zeros(1, 1, 32),
        "state_mask": torch.ones(1, 1, 32, dtype=torch.bool),
        "actions": torch.zeros(1, 24, 32),
        "actions_mask": torch.ones(1, 24, 32, dtype=torch.bool),
        "action_hz": torch.tensor([20.0]),
        "embodiment_id": torch.tensor([0], dtype=torch.long),
    }

    loss = backend._compute_flow_matching_loss(
        prepared_batch=batch,
        shared=shared,
    )
    loss.backward()

    assert h_vlm.grad is not None and torch.count_nonzero(h_vlm.grad) > 0
    assert h_t1_pred.grad is None


def _release_residual_model_cfg(**overrides: object) -> SimpleNamespace:
    values = {
        "flow_only_mode": False,
        "repeated_diffusion_steps": 1,
        "stage2_residual_warmup_start_step": 6_000,
        "stage2_residual_teacher_prob_start": 0.5,
        "stage2_residual_teacher_prob_end": 0.0,
        "stage2_residual_teacher_ramp_steps": 1_000,
        "stage2_residual_predictor_grad_start_step": 7_000,
        "stage2_residual_condition_dropout": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _release_residual_flow_batch() -> dict[str, Tensor]:
    return {
        "act_placeholder_mask": torch.zeros(1, 3, dtype=torch.bool),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "state": torch.zeros(1, 1, 32),
        "state_mask": torch.ones(1, 1, 32, dtype=torch.bool),
        "actions": torch.zeros(1, 24, 32),
        "actions_mask": torch.ones(1, 24, 32, dtype=torch.bool),
        "action_hz": torch.tensor([20.0]),
        "embodiment_id": torch.tensor([0], dtype=torch.long),
        "z_action": torch.ones(1, 32),
    }


def test_release_residual_flow_never_injects_plan_or_effect_tokens() -> None:
    backend = _bare_backend()
    backend._phased_stage2 = True
    backend._release_residual_recipe = True
    backend.stage2_training_recipe = RELEASE_RESIDUAL_RECIPE
    backend.stage2_phase = "P"
    backend.model_cfg = _release_residual_model_cfg()
    backend.flow = _RecordingFlow()
    backend.paired_expert_adapters = None
    backend.stage2_action_residual = ActionResidualProjection(
        latent_dim=32,
        adapter_dim=4,
        hidden_dim=2,
    )
    backend._flow_train_step = 0
    backend._last_stage2_residual_metrics = {}
    shared = _encoding_state(
        predicted_effect=torch.zeros(1, 32),
        action_mean=torch.zeros(1, 32),
        action_logvar=torch.zeros(1, 32),
    )

    backend._compute_flow_matching_loss(
        prepared_batch=_release_residual_flow_batch(),
        shared=shared,
    )

    assert backend.flow.last_inputs["plan_tokens"] is None
    assert backend.flow.last_inputs["effect_tokens"] is None
    residual = backend.flow.last_inputs["action_residual"]
    assert isinstance(residual, Tensor) and residual.shape == (1, 2)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)


def test_release_residual_phase_b_curriculum_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _bare_backend()
    backend._release_residual_recipe = True
    backend.stage2_phase = "P"
    backend.model_cfg = _release_residual_model_cfg()
    backend.stage2_action_residual = _RecordingResidual()
    backend._last_stage2_residual_metrics = {}
    predicted = torch.zeros(2, 32, requires_grad=True)
    batch = {"z_action": torch.full((2, 32), 0.75)}

    backend._flow_train_step = 5_999
    disabled = backend._stage2_action_residual_condition(
        prepared_batch=batch,
        predicted_action=predicted,
    )
    torch.testing.assert_close(disabled, torch.zeros_like(disabled))
    assert backend._last_stage2_residual_metrics["stage2_residual_enabled"].item() == 0.0

    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    backend._flow_train_step = 6_000
    backend._stage2_action_residual_condition(
        prepared_batch=batch,
        predicted_action=predicted,
    )
    torch.testing.assert_close(
        backend.stage2_action_residual.last_condition,
        batch["z_action"],
    )
    assert backend._last_stage2_residual_metrics["stage2_residual_teacher_probability"].item() == 0.5
    assert backend._last_stage2_residual_metrics["stage2_residual_predictor_detached"].item() == 1.0

    backend._flow_train_step = 7_000
    backend._stage2_action_residual_condition(
        prepared_batch=batch,
        predicted_action=predicted,
    )
    torch.testing.assert_close(
        backend.stage2_action_residual.last_condition,
        predicted,
    )
    assert backend.stage2_action_residual.last_condition.requires_grad
    assert backend._last_stage2_residual_metrics["stage2_residual_teacher_probability"].item() == 0.0
    assert backend._last_stage2_residual_metrics["stage2_residual_predictor_detached"].item() == 0.0


def test_release_residual_phase_c_applies_condition_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _bare_backend()
    backend.training = True
    backend._release_residual_recipe = True
    backend.stage2_phase = "E"
    backend.model_cfg = _release_residual_model_cfg(
        stage2_residual_condition_dropout=0.1
    )
    backend.stage2_action_residual = _RecordingResidual()
    backend._flow_train_step = 0
    backend._last_stage2_residual_metrics = {}
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))

    residual = backend._stage2_action_residual_condition(
        prepared_batch={"z_action": torch.ones(2, 32)},
        predicted_action=torch.zeros(2, 32),
    )

    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert backend._last_stage2_residual_metrics["stage2_residual_keep_fraction"].item() == 0.0


def test_release_residual_diagnostics_survive_training_output_mapping() -> None:
    base = {
        "loss_total": torch.zeros(()),
        "loss_flow": torch.zeros(()),
        "loss_perceptual": torch.zeros(()),
        "loss_distill": torch.zeros(()),
        "loss_vlm": torch.zeros(()),
    }
    diagnostics = {
        "stage2_residual_enabled": torch.tensor(1.0),
        "stage2_residual_teacher_probability": torch.tensor(0.5),
        "stage2_residual_teacher_fraction": torch.tensor(0.25),
        "stage2_residual_keep_fraction": torch.tensor(0.9),
        "stage2_residual_predictor_detached": torch.tensor(1.0),
    }

    mapped = map_policy_train_output({**base, **diagnostics})

    for key, value in diagnostics.items():
        torch.testing.assert_close(mapped[key], value)


def _phase_backend(phase: str) -> LatentWorldPolicyBackend:
    backend = _bare_backend()
    backend._phased_stage2 = True
    backend._release_residual_recipe = False
    backend.stage2_phase = phase
    backend.lam = _TestLam()
    backend.flow = nn.Linear(2, 2)
    backend.vlm_to_lam = nn.Linear(2, 2)
    backend.paired_lawm_adapter = nn.Linear(2, 2)
    backend.paired_expert_adapters = PairedExpertAdapters(
        hidden_dim=2,
        vision_dim=2,
        latent_dim=32,
        adapter_dim=4,
    )
    backend.paired_latent_predictor = _TestPredictor()
    backend.vlm = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    backend.vlm[0].requires_grad_(False)
    backend.vlm[1].requires_grad_(True)
    backend.act_query = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    backend.flow_action_query = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    backend.stage2_latent_affine = _TestLatentAffine()
    backend.frozen_action_decoder = _TestFrozenDecoder()
    backend.stage2_action_residual = None
    return backend


def _all_require_grad(module: nn.Module, expected: bool) -> None:
    parameters = tuple(module.parameters())
    assert parameters
    assert all(parameter.requires_grad is expected for parameter in parameters)


@pytest.mark.parametrize(
    ("phase", "decoder", "action_head", "flow", "expert", "plan_query"),
    (
        ("W", True, False, False, False, False),
        ("P", False, True, False, False, True),
        ("E", True, True, True, True, True),
    ),
)
def test_stage2_phase_contract_owns_exact_component_set(
    phase: str,
    decoder: bool,
    action_head: bool,
    flow: bool,
    expert: bool,
    plan_query: bool,
) -> None:
    backend = _phase_backend(phase)
    backend.apply_stage2_phase_contract(phase)

    _all_require_grad(backend.lam.encoder, False)
    _all_require_grad(backend.lam.decoder, decoder)
    _all_require_grad(backend.flow, flow)
    _all_require_grad(backend.vlm_to_lam, False)
    _all_require_grad(backend.paired_lawm_adapter, False)
    _all_require_grad(backend.paired_expert_adapters.plan.physical, False)
    _all_require_grad(backend.paired_expert_adapters.plan.learned, expert)
    _all_require_grad(backend.paired_expert_adapters.effect, expert)
    _all_require_grad(backend.paired_latent_predictor.effect_head, True)
    _all_require_grad(backend.paired_latent_predictor.action_head, action_head)
    assert backend.paired_latent_predictor.unowned_parameter.requires_grad is False
    assert backend.act_query.requires_grad is True
    assert backend.flow_action_query.requires_grad is plan_query
    _all_require_grad(backend.frozen_action_decoder, False)
    assert backend.frozen_action_decoder.training is False

    # Generic freeze policy owns which VLM blocks are open; the phase policy
    # must preserve that top-block selection in every phase.
    _all_require_grad(backend.vlm[0], False)
    _all_require_grad(backend.vlm[1], True)


@pytest.mark.parametrize("phase", ("P", "E"))
def test_lawam_action_phase_contract_trains_mapping_without_effect_head(
    phase: str,
) -> None:
    backend = _phase_backend(phase)
    backend._lawam_action_recipe = True
    backend._release_residual_recipe = True
    backend.paired_latent_predictor.effect_head = None
    backend.paired_expert_adapters = None
    backend.stage2_action_residual = nn.Linear(32, 2)

    backend.apply_stage2_phase_contract(phase)

    _all_require_grad(backend.vlm_to_lam, True)
    _all_require_grad(backend.paired_latent_predictor.action_head, True)
    _all_require_grad(backend.stage2_action_residual, True)
    _all_require_grad(backend.lam.encoder, False)
    _all_require_grad(backend.lam.decoder, phase == "E")
    _all_require_grad(backend.flow, phase == "E")
    assert backend.paired_latent_predictor.effect_head is None
    assert backend.act_query.requires_grad is True
    assert backend.flow_action_query.requires_grad is True


class _ForwardHarness(LatentWorldPolicyBackend):
    def __init__(self, phase: str) -> None:
        nn.Module.__init__(self)
        weights = {
            "W": (0.0, 1.0, 0.0, 1.0),
            "P": (0.0, 0.25, 1.0, 0.25),
            "E": (1.0, 0.1, 0.1, 0.1),
        }[phase]
        self.stage2_phase = phase
        self._phased_stage2 = True
        self._release_residual_recipe = False
        self.stage2_arm = "p2"
        self.model_cfg = SimpleNamespace(
            vlm_dtype=torch.float32,
            future_prediction=True,
            enable_loss_distill=False,
            stage2_action_distribution="dense_transition_gaussian_v3",
            stage2_flow_weight=weights[0],
            stage2_effect_distill_weight=weights[1],
            stage2_action_distill_weight=weights[2],
            stage2_dino_weight=weights[3],
        )
        self.plan_calls = 0
        self.flow_calls = 0

    def _prepare_train_batch(self, *, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return batch

    def _run_shared_encoding_train(self, **_: object) -> PolicyEncodingState:
        effect = torch.full((2, 32), 0.3)
        return PolicyEncodingState(
            h_vlm=torch.zeros(2, 3, 4),
            h_effect=torch.zeros(2, 8, 4),
            h_plan=torch.zeros(2, 8, 4),
            pred_action_emb=torch.zeros(2, 1, 32),
            h_t=torch.zeros(2, 2, 4),
            h_t1_pred=torch.full((2, 2, 4), 0.4),
            h_t1_gt=torch.full((2, 2, 4), -0.1),
            h_t_original=torch.zeros(2, 2, 4),
            predicted_effect=effect,
            paired_tokens=PairedTokens(
                effect=effect,
                action=torch.full((2, 32), -0.2),
                action_logvar=torch.full((2, 32), 0.1),
            ),
        )

    def _compute_stage2_plan_losses(self, **_: object) -> dict[str, Tensor]:
        self.plan_calls += 1
        loss_plan = torch.tensor(2.75)
        return {
            "loss_plan": loss_plan,
            "loss_action_kl": torch.tensor(1.0),
            "loss_action_mean": torch.tensor(2.0),
            "loss_action_trajectory": torch.tensor(3.0),
            "loss_action_trajectory_position": torch.tensor(4.0),
            "loss_action_trajectory_rotation": torch.tensor(5.0),
            "loss_action_trajectory_gripper": torch.tensor(6.0),
        }

    def _compute_flow_matching_loss(self, **_: object) -> Tensor:
        self.flow_calls += 1
        return torch.tensor(4.25)


@pytest.mark.parametrize(
    ("phase", "effect_weight", "plan_weight", "dino_weight", "flow_weight"),
    (
        ("W", 1.0, 0.0, 1.0, 0.0),
        ("P", 0.25, 1.0, 0.25, 0.0),
        ("E", 0.1, 0.1, 0.1, 1.0),
    ),
)
def test_stage2_forward_aggregates_exact_phase_losses_and_skips_disabled_branches(
    phase: str,
    effect_weight: float,
    plan_weight: float,
    dino_weight: float,
    flow_weight: float,
) -> None:
    backend = _ForwardHarness(phase)
    batch = {
        "input_ids": torch.zeros(2, 1, dtype=torch.long),
        "z_effect": torch.zeros(2, 32),
    }

    output = backend(batch=batch)
    expected_effect = F.smooth_l1_loss(
        torch.full((2, 32), 0.3),
        torch.zeros(2, 32),
        beta=0.1,
    )
    expected_dino = F.mse_loss(
        torch.full((2, 2, 4), 0.4),
        torch.full((2, 2, 4), -0.1),
    )
    torch.testing.assert_close(output["loss_effect_distill"], expected_effect)
    torch.testing.assert_close(output["loss_dino"], expected_dino)
    expected = (
        flow_weight * output["loss_flow"]
        + effect_weight * output["loss_effect_distill"]
        + plan_weight * output["loss_action_distill"]
        + dino_weight * output["loss_dino"]
    )
    torch.testing.assert_close(output["loss_total"], expected, rtol=0.0, atol=0.0)
    assert backend.plan_calls == (0 if phase == "W" else 1)
    assert backend.flow_calls == (1 if phase == "E" else 0)
    if phase == "W":
        torch.testing.assert_close(output["loss_plan"], torch.zeros(()))
        torch.testing.assert_close(output["loss_action_distill"], torch.zeros(()))
    if phase in {"W", "P"}:
        torch.testing.assert_close(output["loss_flow"], torch.zeros(()))


def test_lawam_action_forward_uses_lawam_distill_and_no_effect_loss() -> None:
    backend = _ForwardHarness("P")
    backend._lawam_action_recipe = True
    backend._release_residual_recipe = True
    backend.model_cfg.stage2_flow_weight = 1.0
    backend.model_cfg.stage2_effect_distill_weight = 0.0
    backend.model_cfg.stage2_action_distill_weight = 0.5
    backend.model_cfg.stage2_dino_weight = 0.1
    backend.model_cfg.stage2_lawam_distill_weight = 0.1
    backend.model_cfg.latent_loss_type = "mse"
    batch = {
        "input_ids": torch.zeros(2, 1, dtype=torch.long),
        "z_lawam": torch.zeros(2, 32),
        "z_lawam_raw": torch.ones(2, 32),
    }

    output = backend(batch=batch)

    torch.testing.assert_close(output["loss_effect_distill"], torch.zeros(()))
    torch.testing.assert_close(output["loss_lawam_distill"], torch.ones(()))
    expected = (
        output["loss_flow"]
        + 0.5 * output["loss_action_distill"]
        + 0.1 * output["loss_dino"]
        + 0.1 * output["loss_lawam_distill"]
    )
    torch.testing.assert_close(output["loss_total"], expected)
    assert backend.plan_calls == 1
    assert backend.flow_calls == 1


def test_lawam_distill_survives_training_output_mapping() -> None:
    base = {
        "loss_total": torch.zeros(()),
        "loss_flow": torch.zeros(()),
        "loss_perceptual": torch.zeros(()),
        "loss_distill": torch.zeros(()),
        "loss_lawam_distill": torch.tensor(0.75),
        "loss_vlm": torch.zeros(()),
    }

    mapped = map_policy_train_output(base)

    torch.testing.assert_close(mapped["loss_lawam_distill"], torch.tensor(0.75))
