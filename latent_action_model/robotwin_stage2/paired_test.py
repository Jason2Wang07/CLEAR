import pytest
import torch

from latent_action_model.robotwin_stage2.paired import (
    ActionResidualProjection,
    PairedExpertAdapters,
    PairedLaWMAdapter,
    PairedLatentPredictor,
)


def test_action_residual_projection_is_zero_initialized_and_trainable() -> None:
    projection = ActionResidualProjection(
        latent_dim=4,
        adapter_dim=8,
        hidden_dim=16,
    )
    z_action = torch.randn(3, 4, requires_grad=True)

    output = projection(z_action)

    assert output.shape == (3, 16)
    assert torch.count_nonzero(output) == 0
    target = torch.randn_like(output)
    torch.nn.functional.mse_loss(output, target).backward()
    final = projection.network[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0


def test_action_residual_projection_rejects_wrong_latent_shape() -> None:
    projection = ActionResidualProjection(latent_dim=4, adapter_dim=8, hidden_dim=16)
    with pytest.raises(ValueError, match=r"z_action must be \[B,4\]"):
        projection(torch.randn(2, 5))


def test_lawm_adapter_accepts_effect_only() -> None:
    adapter = PairedLaWMAdapter(latent_dim=32, adapter_dim=64)
    effect = torch.randn(3, 32, requires_grad=True)

    output = adapter(effect)
    assert output.shape == (3, 1, 32)
    output.square().mean().backward()
    assert effect.grad is not None
    assert torch.isfinite(effect.grad).all()

    with pytest.raises(TypeError):
        adapter(effect, torch.randn_like(effect))


def test_expert_adapters_keep_typed_effect_action_routes() -> None:
    adapter = PairedExpertAdapters(
        hidden_dim=64,
        vision_dim=48,
        latent_dim=32,
        adapter_dim=64,
    )
    effect = torch.randn(3, 32, requires_grad=True)
    action = torch.randn(3, 32, requires_grad=True)

    plan, plan_mask, effect_tokens, effect_mask = adapter(effect, action)
    assert plan.shape[0] == effect.shape[0]
    assert effect_tokens.shape[0] == effect.shape[0]
    assert plan_mask.dtype == torch.bool
    assert effect_mask.dtype == torch.bool
    (plan.square().mean() + effect_tokens.square().mean()).backward()
    assert effect.grad is not None and torch.isfinite(effect.grad).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def _dense_transition_predictor() -> PairedLatentPredictor:
    return PairedLatentPredictor(
        query_count=2,
        query_dim=8,
        hidden_dim=16,
        latent_dim=32,
        visual_dim=6,
        state_dim=4,
        num_embodiments=5,
        horizon_condition_dim=3,
        visual_grid_size=(2, 3),
        action_decoder_layers=3,
        action_decoder_heads=4,
        action_distribution="dense_transition_gaussian_v3",
    )


def test_lawam_action_predictor_has_no_effect_head() -> None:
    predictor = PairedLatentPredictor(
        query_count=2,
        query_dim=8,
        hidden_dim=16,
        latent_dim=32,
        visual_dim=6,
        state_dim=4,
        num_embodiments=5,
        horizon_condition_dim=3,
        visual_grid_size=(2, 3),
        action_decoder_layers=2,
        action_decoder_heads=4,
        action_distribution="dense_transition_gaussian_v3",
        enable_effect_head=False,
    )

    assert predictor.effect_head is None
    with pytest.raises(RuntimeError, match="no Effect head"):
        predictor.predict_effect(torch.randn(1, 2, 8))
    distribution = predictor.predict_action(
        torch.randn(1, 2, 8),
        current_visual=torch.randn(1, 6, 6),
        future_visual=torch.randn(1, 6, 6),
        state=torch.randn(1, 4),
        effect=torch.randn(1, 32),
        embodiment_id=torch.tensor([1]),
        horizon_condition=torch.tensor([[30.0, 1.2, 36.0]]),
    )
    assert distribution.mean.shape == (1, 32)
    assert distribution.logvar.shape == (1, 32)


def test_dense_transition_predictor_preserves_patches_and_backpropagates() -> None:
    predictor = _dense_transition_predictor()
    context = torch.randn(3, 2, 8, requires_grad=True)
    current = torch.randn(3, 6, 6, requires_grad=True)
    future = torch.randn(3, 6, 6, requires_grad=True)
    state = torch.randn(3, 4, requires_grad=True)
    effect = predictor.predict_effect(context).detach().requires_grad_()
    horizon = torch.randn(3, 3, requires_grad=True)
    action = predictor.predict_action(
        context,
        current_visual=current,
        future_visual=future,
        state=state,
        effect=effect,
        embodiment_id=torch.tensor([0, 2, 4]),
        horizon_condition=horizon,
    )
    assert action.mean.shape == (3, 32)
    assert action.logvar.shape == action.mean.shape
    assert bool((action.logvar >= predictor.logvar_min).all())
    assert bool((action.logvar <= predictor.logvar_max).all())
    (action.mean.square().mean() + action.logvar.square().mean()).backward()
    for value in (context, current, future, state, effect, horizon):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert float(value.grad.abs().sum()) > 0


def test_dense_transition_predictor_follows_parameter_dtype_after_bf16_cast() -> None:
    predictor = _dense_transition_predictor().to(dtype=torch.bfloat16).eval()
    context = torch.randn(2, 2, 8, dtype=torch.float32)

    effect = predictor.predict_effect(context)
    action = predictor.predict_action(
        context,
        current_visual=torch.randn(2, 6, 6, dtype=torch.float32),
        future_visual=torch.randn(2, 6, 6, dtype=torch.float32),
        state=torch.randn(2, 4, dtype=torch.float32),
        effect=effect.float(),
        embodiment_id=torch.tensor([0, 4]),
        horizon_condition=torch.tensor(
            [[20.0, 0.4, 8.0], [20.0, 0.4, 8.0]], dtype=torch.float32
        ),
    )

    assert effect.dtype == torch.bfloat16
    assert action.mean.dtype == torch.bfloat16
    assert action.logvar.dtype == torch.bfloat16


def test_dense_transition_predictor_applies_per_dimension_logvar_bounds() -> None:
    lower = torch.linspace(-7.0, -5.0, 32)
    upper = torch.linspace(1.0, 6.0, 32)
    predictor = PairedLatentPredictor(
        query_count=2,
        query_dim=8,
        hidden_dim=16,
        latent_dim=32,
        visual_dim=6,
        state_dim=4,
        num_embodiments=3,
        horizon_condition_dim=3,
        visual_grid_size=(2, 3),
        action_decoder_layers=1,
        action_decoder_heads=4,
        action_distribution="dense_transition_gaussian_v3",
        logvar_min=lower,
        logvar_max=upper,
    )
    distribution = predictor.predict_action(
        torch.randn(2, 2, 8),
        current_visual=torch.randn(2, 6, 6),
        future_visual=torch.randn(2, 6, 6),
        state=torch.randn(2, 4),
        effect=torch.randn(2, 32),
        embodiment_id=torch.tensor([0, 2]),
        horizon_condition=torch.tensor([[20.0, 1.2, 24.0], [20.0, 1.2, 24.0]]),
    )
    assert torch.all(distribution.logvar >= lower)
    assert torch.all(distribution.logvar <= upper)


def test_dense_transition_predictor_uses_same_position_change() -> None:
    torch.manual_seed(19)
    predictor = _dense_transition_predictor().eval()
    context = torch.randn(1, 2, 8)
    current = torch.randn(1, 6, 6)
    future = torch.randn(1, 6, 6)
    conditions = {
        "state": torch.randn(1, 4),
        "effect": torch.randn(1, 32),
        "embodiment_id": torch.tensor([1]),
        "horizon_condition": torch.randn(1, 3),
    }
    original = predictor.predict_action(
        context,
        current_visual=current,
        future_visual=future,
        **conditions,
    )
    swapped_future = future.clone()
    swapped_future[:, [0, 1]] = swapped_future[:, [1, 0]]
    swapped = predictor.predict_action(
        context,
        current_visual=current,
        future_visual=swapped_future,
        **conditions,
    )

    # A mean-pooled future head is invariant to this swap. Dense same-position
    # transition tokens are not, because each future patch is paired with F_t^i.
    assert not torch.allclose(original.mean, swapped.mean)


def test_dense_transition_predictor_rejects_pooled_visual_and_missing_conditions() -> None:
    predictor = _dense_transition_predictor()
    inputs = {
        "h_act": torch.randn(2, 2, 8),
        "current_visual": torch.randn(2, 6, 6),
        "future_visual": torch.randn(2, 6, 6),
        "state": torch.randn(2, 4),
        "effect": torch.randn(2, 32),
    }
    with pytest.raises(ValueError, match="embodiment_id and horizon_condition"):
        predictor.predict_action(**inputs)
    with pytest.raises(ValueError, match="preserve the dense visual grid"):
        predictor.predict_action(
            **{**inputs, "future_visual": torch.randn(2, 6)},
            embodiment_id=torch.tensor([0, 1]),
            horizon_condition=torch.randn(2, 3),
        )


def test_retired_gaussian_v2_config_fails_fast() -> None:
    with pytest.raises(ValueError, match="retired mean-pooled action head"):
        PairedLatentPredictor(action_distribution="gaussian_v2")
