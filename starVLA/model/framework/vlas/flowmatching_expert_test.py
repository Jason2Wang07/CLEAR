import pytest
import torch

from .flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
)


def _residual_probe() -> ConditionalFlowMatchingHead:
    head = ConditionalFlowMatchingHead.__new__(ConditionalFlowMatchingHead)
    torch.nn.Module.__init__(head)
    head.config = ConditionalFlowMatchingConfig(hidden_dim=4)
    return head


def test_none_and_zero_action_residual_are_identical() -> None:
    head = _residual_probe()
    features = torch.randn(2, 3, 4)

    without_residual = head._add_action_residual(features, None)
    with_zero_residual = head._add_action_residual(features, torch.zeros(2, 4))

    torch.testing.assert_close(without_residual, with_zero_residual, rtol=0.0, atol=0.0)


def test_nonzero_action_residual_receives_gradient() -> None:
    head = _residual_probe()
    features = torch.randn(2, 3, 4)
    residual = torch.randn(2, 4, requires_grad=True)

    head._add_action_residual(features, residual).square().sum().backward()

    assert residual.grad is not None
    assert torch.count_nonzero(residual.grad) > 0


def test_action_residual_rejects_wrong_shape() -> None:
    head = _residual_probe()
    features = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match=r"action_residual.*\[B,4\]"):
        head._add_action_residual(features, torch.zeros(2, 3))
