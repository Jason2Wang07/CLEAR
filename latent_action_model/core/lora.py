"""Minimal LoRA weight parametrizations for Stage-1 effect-encoder tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRAWeight(nn.Module):
    """Add a trainable low-rank delta to a frozen 2-D weight tensor."""

    def __init__(self, out_features: int, in_features: int, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}.")
        if rank > min(out_features, in_features):
            raise ValueError(
                f"LoRA rank {rank} exceeds the smaller weight dimension "
                f"{min(out_features, in_features)}."
            )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, int(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(int(out_features), self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.lora_B, self.lora_A)
        return weight + delta.to(dtype=weight.dtype) * self.scaling


@dataclass(frozen=True)
class LoRAInjection:
    target_names: tuple[str, ...]
    parameter_count: int


def _register_weight(module: nn.Module, *, rank: int, alpha: float) -> None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        raise TypeError(f"LoRA target must expose a 2-D weight parameter, got {type(module).__name__}.")
    parametrize.register_parametrization(
        module,
        "weight",
        LoRAWeight(int(weight.shape[0]), int(weight.shape[1]), rank=rank, alpha=alpha),
    )


def inject_lora_into_encoder(encoder: nn.Module, *, rank: int, alpha: float) -> LoRAInjection:
    """Parametrize every encoder Linear weight and MultiheadAttention QKV weight."""

    if rank < 1:
        raise ValueError(f"LoRA rank must be positive, got {rank}.")
    targets: list[str] = []
    modules = list(encoder.named_modules())
    for name, module in modules:
        if isinstance(module, nn.Linear):
            _register_weight(module, rank=rank, alpha=alpha)
            targets.append(f"{name}.weight" if name else "weight")
        if isinstance(module, nn.MultiheadAttention):
            weight = module.in_proj_weight
            if weight is None or weight.ndim != 2:
                raise ValueError(
                    f"LoRA requires packed MultiheadAttention projections; {name!r} uses separate weights."
                )
            parametrize.register_parametrization(
                module,
                "in_proj_weight",
                LoRAWeight(
                    int(weight.shape[0]),
                    int(weight.shape[1]),
                    rank=rank,
                    alpha=alpha,
                ),
            )
            targets.append(f"{name}.in_proj_weight" if name else "in_proj_weight")

    if not targets:
        raise RuntimeError("No eligible Linear or MultiheadAttention weights found in the effect encoder.")
    parameters = [
        parameter
        for module in encoder.modules()
        if isinstance(module, LoRAWeight)
        for parameter in module.parameters()
    ]
    return LoRAInjection(
        target_names=tuple(targets),
        parameter_count=sum(parameter.numel() for parameter in parameters),
    )


def is_lora_parameter_name(name: str) -> bool:
    return ".parametrizations.weight.0.lora_" in name or ".parametrizations.in_proj_weight.0.lora_" in name
