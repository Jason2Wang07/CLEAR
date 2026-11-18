"""Independent gated adapters for oracle plan and frozen E39 effect tokens."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


PlanRepresentation = Literal["physical", "learned"]


class _GatedTokenBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adapter_dim: int,
        *,
        num_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.position_embedding: nn.Parameter | None
        if num_tokens is None:
            self.position_embedding = None
        else:
            self.position_embedding = nn.Parameter(torch.empty(1, int(num_tokens), self.output_dim))
            nn.init.normal_(self.position_embedding, std=0.02)
        # sigmoid(0)=0.5 keeps the condition active while allowing either attenuation or growth.
        self.gate_logit = nn.Parameter(torch.zeros(()))

    @property
    def gate(self) -> Tensor:
        return self.gate_logit.sigmoid()

    def forward(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokens must have shape [B,N,{self.input_dim}], got {tuple(tokens.shape)}"
            )
        if self.position_embedding is not None and tokens.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                f"tokens must contain {self.position_embedding.shape[1]} ordered tokens, "
                f"got {tokens.shape[1]}"
            )
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if feature_mask is not None:
            if feature_mask.shape != tokens.shape:
                raise ValueError(
                    f"feature_mask must match tokens, got {tuple(feature_mask.shape)} vs {tuple(tokens.shape)}"
                )
            feature_mask = feature_mask.to(device=tokens.device, dtype=torch.bool)
            tokens = tokens * feature_mask.to(dtype=tokens.dtype)
            inferred_token_mask = feature_mask.any(dim=-1)
            token_mask = inferred_token_mask if token_mask is None else token_mask & inferred_token_mask
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"token_mask must have shape {tuple(tokens.shape[:2])}, got {tuple(token_mask.shape)}"
                )
            token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        adapted = self.adapter(tokens)
        if self.position_embedding is not None:
            adapted = adapted + self.position_embedding.to(device=adapted.device, dtype=adapted.dtype)
        adapted = adapted * self.gate
        adapted = adapted * token_mask.unsqueeze(-1).to(dtype=adapted.dtype)
        return adapted, token_mask


class PlanTokenAdapter(nn.Module):
    """Project physical or learned plans into self-attended action-expert tokens.

    Both branches are always instantiated so E0-E5 have identical parameter
    capacity even when a particular experiment uses only one representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        physical_dim: int = 8,
        learned_dim: int = 32,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        self.physical = _GatedTokenBranch(
            physical_dim, hidden_dim, adapter_dim, num_tokens=4
        )
        self.learned = _GatedTokenBranch(
            learned_dim, hidden_dim, adapter_dim, num_tokens=1
        )

    def forward(
        self,
        tokens: Tensor,
        *,
        representation: PlanRepresentation,
        token_mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if representation == "physical":
            return self.physical(tokens, token_mask=token_mask, feature_mask=feature_mask)
        if representation == "learned":
            if feature_mask is not None:
                raise ValueError("learned plans accept token_mask, not feature_mask")
            return self.learned(tokens, token_mask=token_mask)
        raise ValueError(f"unsupported plan representation: {representation!r}")


class EffectTokenAdapter(nn.Module):
    """Project frozen E39 means into an independently gated cross-attention token."""

    def __init__(self, vision_dim: int, *, effect_dim: int = 32, adapter_dim: int = 256) -> None:
        super().__init__()
        self.effect = _GatedTokenBranch(effect_dim, vision_dim, adapter_dim)

    def forward(
        self,
        effect_mean: Tensor,
        *,
        token_mask: Tensor | None = None,
        detach_input: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if effect_mean.ndim == 2:
            effect_mean = effect_mean.unsqueeze(1)
        if detach_input:
            effect_mean = effect_mean.detach()
        return self.effect(effect_mean, token_mask=token_mask)


class OracleConditionAdapters(nn.Module):
    """Fixed-capacity adapter bundle shared by all E0-E5 conditions."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        vision_dim: int,
        physical_dim: int = 8,
        learned_dim: int = 32,
        effect_dim: int = 32,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        self.plan = PlanTokenAdapter(
            hidden_dim,
            physical_dim=physical_dim,
            learned_dim=learned_dim,
            adapter_dim=adapter_dim,
        )
        self.effect = EffectTokenAdapter(
            vision_dim,
            effect_dim=effect_dim,
            adapter_dim=adapter_dim,
        )
