"""Locked BridgeData V2 temporal contract for Stage-2."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


TEMPORAL_SCHEMA_VERSION = "bridge-stage2-temporal-v1"
BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT = "bridge_official_12s_v1"
BRIDGE_STAGE2_CACHE_SCHEMA_VERSION = "bridge-stage2-paired-teacher-cache-v1"
BRIDGE_STAGE2_NORMALIZATION_SCHEMA_VERSION = (
    "bridge-stage2-latent-standardization-v1"
)
BRIDGE_STAGE2_ROW_COUNT = 545_462
BRIDGE_STAGE2_SPLIT_ROWS = {
    "train": 435_141,
    "validation": 54_728,
    "test": 55_593,
}
BRIDGE_STAGE2_TEMPORAL_CONTRACTS = frozenset(
    {BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT}
)
BRIDGE_MANIFEST_TEMPORAL_CONTRACT = (
    "actions=t:t+6,dino=t+5,effect=t+6,replan_steps=1"
)


@dataclass(frozen=True)
class BridgeStage2TemporalContract:
    identifier: str = BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT
    cache_schema_version: str = BRIDGE_STAGE2_CACHE_SCHEMA_VERSION
    normalization_schema_version: str = BRIDGE_STAGE2_NORMALIZATION_SCHEMA_VERSION
    row_count: int = BRIDGE_STAGE2_ROW_COUNT
    control_hz: float = 5.0
    duration_seconds: float = 1.2
    effect_teacher_delta_steps: int = 6
    dino_future_delta_steps: int = 5
    representation_steps: int = 24
    representation_hz: float = 20.0
    physical_valid_count: int = 6
    trajectory_dim: int = 7
    state_dim: int = 14
    arm_count: int = 1
    replan_steps: int = 1

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.representation_hz <= 0:
            raise ValueError("Bridge control and representation rates must be positive")
        if not math.isclose(
            self.duration_seconds * self.control_hz,
            self.physical_valid_count,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("Bridge 1.2s must contain exactly six 5 Hz transitions")
        if not math.isclose(
            self.duration_seconds * self.representation_hz,
            self.representation_steps,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("Bridge ActionVAE representation must contain 24 slots")
        if self.effect_teacher_delta_steps != self.physical_valid_count:
            raise ValueError("Bridge effect teacher/action endpoint must be t+6")
        if self.dino_future_delta_steps != self.physical_valid_count - 1:
            raise ValueError("Bridge DINO future must follow the release t+5 indexing")
        if self.effect_teacher_delta_steps == self.dino_future_delta_steps:
            raise ValueError("Bridge effect and DINO endpoints must remain distinct")
        if (self.trajectory_dim, self.state_dim, self.arm_count) != (7, 14, 1):
            raise ValueError("Bridge ActionVAE must use the locked single-arm 24x7/14D contract")
        if self.replan_steps != 1:
            raise ValueError("Bridge SIMPLER evaluation must replan every control step")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": TEMPORAL_SCHEMA_VERSION, **asdict(self)}

    def validate_manifest_row(self, row: Mapping[str, Any]) -> None:
        required = {
            "start_step",
            "end_step",
            "control_hz",
            "dino_start_step",
            "dino_future_step",
            "effect_teacher_step",
            "source_action_steps",
            "action_hz",
            "duration_seconds",
            "temporal_contract",
            "physical_valid_count",
            "language_instruction",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise KeyError(f"Bridge row is missing temporal fields: {missing}")
        start = int(row["start_step"])
        if int(row["end_step"]) - start != self.physical_valid_count:
            raise ValueError("Bridge window must contain six source transitions")
        if int(row["dino_start_step"]) != start:
            raise ValueError("Bridge dino_start_step must equal start_step")
        if int(row["dino_future_step"]) != start + self.dino_future_delta_steps:
            raise ValueError("Bridge DINO future must be t+5")
        if int(row["effect_teacher_step"]) != start + self.effect_teacher_delta_steps:
            raise ValueError("Bridge effect teacher must be t+6")
        if int(row["source_action_steps"]) != self.physical_valid_count:
            raise ValueError("Bridge source_action_steps must be six")
        for name in ("control_hz", "action_hz"):
            if not math.isclose(
                float(row[name]), self.control_hz, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(f"Bridge {name} must be exactly 5 Hz")
        if not math.isclose(
            float(row["duration_seconds"]),
            self.duration_seconds,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("Bridge duration_seconds must be 1.2")
        if int(row["physical_valid_count"]) != self.physical_valid_count:
            raise ValueError("Bridge physical_valid_count must be six")
        if str(row["temporal_contract"]) != BRIDGE_MANIFEST_TEMPORAL_CONTRACT:
            raise ValueError("Bridge manifest temporal_contract marker mismatch")
        if not str(row["language_instruction"]).strip():
            raise ValueError("Bridge Stage-2 excludes empty-language rows")

    def expected_action_mask(self, *, batch_size: int) -> tuple[int, int, int, int]:
        return (
            int(batch_size),
            self.representation_steps,
            self.physical_valid_count,
            self.trajectory_dim,
        )


DEFAULT_TEMPORAL_CONTRACT = BridgeStage2TemporalContract()


def is_bridge_stage2_temporal_contract(value: Any) -> bool:
    identifier = (
        value.identifier if isinstance(value, BridgeStage2TemporalContract) else value
    )
    return str(identifier or "").strip() in BRIDGE_STAGE2_TEMPORAL_CONTRACTS


def resolve_bridge_stage2_temporal_contract(
    value: str | BridgeStage2TemporalContract | None,
) -> BridgeStage2TemporalContract:
    if isinstance(value, BridgeStage2TemporalContract):
        return value
    identifier = str(value or "").strip() or BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT
    if identifier not in BRIDGE_STAGE2_TEMPORAL_CONTRACTS:
        raise ValueError(
            f"unsupported Bridge stage2_temporal_contract {identifier!r}; "
            f"expected {BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT!r}"
        )
    return DEFAULT_TEMPORAL_CONTRACT


__all__ = [
    "BRIDGE_MANIFEST_TEMPORAL_CONTRACT",
    "BRIDGE_OFFICIAL_12S_TEMPORAL_CONTRACT",
    "BRIDGE_STAGE2_CACHE_SCHEMA_VERSION",
    "BRIDGE_STAGE2_NORMALIZATION_SCHEMA_VERSION",
    "BRIDGE_STAGE2_ROW_COUNT",
    "BRIDGE_STAGE2_SPLIT_ROWS",
    "BridgeStage2TemporalContract",
    "DEFAULT_TEMPORAL_CONTRACT",
    "is_bridge_stage2_temporal_contract",
    "resolve_bridge_stage2_temporal_contract",
]
