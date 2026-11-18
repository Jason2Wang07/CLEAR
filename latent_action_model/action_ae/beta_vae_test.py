from __future__ import annotations

import torch

from latent_action_model.action_ae.beta_vae import (
    ActionBetaVAEConfig,
    StandardNormalActionVAE,
    bounded_logvar_from_logits,
    standard_normal_kl_per_dimension,
)
from latent_action_model.action_ae.distributions import DiagonalGaussian


def test_standard_normal_action_vae_shapes_and_deterministic_mean() -> None:
    model = StandardNormalActionVAE(
        ActionBetaVAEConfig(
            state_dim=14,
            variant="a1",
            horizon=24,
            hidden_dim=16,
            condition_dim=16,
            dilations=(1, 2),
        )
    ).eval()
    batch = 3
    trajectory = torch.randn(batch, 24, 7)
    state = torch.randn(batch, 14)
    effect = torch.randn(batch, 32)
    output = model(trajectory, state, effect)
    posterior = model.posterior(trajectory, state, effect)
    assert output.reconstruction.shape == (batch, 24, 7)
    assert output.z_action.shape == (batch, 1, 32)
    assert torch.equal(output.z_action[:, 0], posterior.mean)
    assert torch.equal(model.encode(trajectory, state, effect)[:, 0], posterior.mean)


def test_standard_normal_kl_is_zero_only_for_standard_normal() -> None:
    standard = DiagonalGaussian(torch.zeros(2, 4), torch.zeros(2, 4))
    shifted = DiagonalGaussian(torch.ones(2, 4), torch.zeros(2, 4))
    assert torch.equal(standard_normal_kl_per_dimension(standard), torch.zeros(2, 4))
    assert torch.all(standard_normal_kl_per_dimension(shifted) > 0)


def test_sampled_reconstruction_updates_all_beta_vae_components() -> None:
    model = StandardNormalActionVAE(
        ActionBetaVAEConfig(
            state_dim=14,
            variant="a1",
            horizon=24,
            hidden_dim=16,
            condition_dim=16,
            dilations=(1, 2),
        )
    ).train()
    output = model(
        torch.randn(2, 24, 7),
        torch.randn(2, 14),
        torch.randn(2, 32),
        sample_posterior=True,
    )
    loss = output.reconstruction.square().mean() + standard_normal_kl_per_dimension(output.posterior).mean()
    loss.backward()
    gradients = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert any(name.startswith("encoder.") for name in gradients)
    assert any(name.startswith("posterior_logvar.") for name in gradients)
    assert any(name.startswith("decoder.") for name in gradients)


def test_bounded_logvar_has_consistent_boundary_gradients() -> None:
    logits = torch.tensor([[-3.0, 0.0, 3.0]], requires_grad=True)
    logvar = bounded_logvar_from_logits(logits, minimum=-6.0, maximum=2.0)
    assert torch.all(logvar > -6.0)
    assert torch.all(logvar < 2.0)
    logvar.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)


def test_posterior_logvar_init_is_distribution_value() -> None:
    model = StandardNormalActionVAE(
        ActionBetaVAEConfig(
            state_dim=14,
            variant="a1",
            horizon=24,
            hidden_dim=16,
            condition_dim=16,
            dilations=(1, 2),
        )
    ).eval()
    posterior = model.posterior(
        torch.randn(2, 24, 7),
        torch.randn(2, 14),
        torch.randn(2, 32),
    )
    assert torch.allclose(posterior.logvar, torch.full_like(posterior.logvar, -4.0))
