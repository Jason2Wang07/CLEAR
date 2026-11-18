"""Locked 30 Hz temporal contract for RoboTwin Stage-2."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


TEMPORAL_SCHEMA_VERSION = "robotwin-stage2-temporal-v1"
ROBOTWIN_OFFICIAL_12S_TEMPORAL_CONTRACT = "robotwin_official_12s_v1"
ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT = (
    "robotwin_official_12s_dense_s1_v1"
)
ROBOTWIN_STAGE2_CACHE_SCHEMA_VERSION = "robotwin-stage2-paired-teacher-cache-v3"
ROBOTWIN_STAGE2_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin-stage2-latent-standardization-v2"
)
ROBOTWIN_STAGE2_ROW_COUNT = 155_029
ROBOTWIN_STAGE2_DENSE_S1_ROW_COUNT = 5_087_247
ROBOTWIN_STAGE2_TEMPORAL_CONTRACTS = frozenset(
    {
        ROBOTWIN_OFFICIAL_12S_TEMPORAL_CONTRACT,
        ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT,
    }
)
RELEASE_FUTURE_INDEX_RULE = "two-frame-action-endpoint-last-index"


@dataclass(frozen=True)
class RoboTwinStage2TemporalContract:
    identifier: str = ROBOTWIN_OFFICIAL_12S_TEMPORAL_CONTRACT
    cache_schema_version: str = ROBOTWIN_STAGE2_CACHE_SCHEMA_VERSION
    normalization_schema_version: str = ROBOTWIN_STAGE2_NORMALIZATION_SCHEMA_VERSION
    row_count: int = ROBOTWIN_STAGE2_ROW_COUNT
    control_hz: float = 30.0
    teacher_action_steps: int = 36
    teacher_horizon_sec: float = 1.2
    teacher_end_delta: int = 36
    expert_valid_action_steps: int = 36
    expert_padded_action_steps: int = 50
    expert_horizon_sec: float = 1.2
    release_future_delta: int = 35
    native_action_dim: int = 16
    flow_action_dim: int = 32

    @property
    def duration_seconds(self) -> float:
        return self.teacher_horizon_sec

    @property
    def effect_teacher_delta_steps(self) -> int:
        return self.teacher_end_delta

    @property
    def dino_future_delta_steps(self) -> int:
        return self.release_future_delta

    @property
    def representation_steps(self) -> int:
        return self.teacher_action_steps

    @property
    def physical_valid_count(self) -> int:
        return self.expert_valid_action_steps

    @property
    def trajectory_dim(self) -> int:
        return 14

    @property
    def state_dim(self) -> int:
        return 14

    @property
    def arm_count(self) -> int:
        return 2

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if math.floor(self.teacher_horizon_sec * self.control_hz) != self.teacher_action_steps:
            raise ValueError("Teacher horizon must resolve to 36 actions at 30 Hz")
        if math.floor(self.expert_horizon_sec * self.control_hz) != self.expert_valid_action_steps:
            raise ValueError("Expert horizon must resolve to 36 actions at 30 Hz")
        if self.teacher_end_delta != self.teacher_action_steps:
            raise ValueError("Teacher endpoint must be observation t+36")
        if self.release_future_delta != self.expert_valid_action_steps - 1:
            raise ValueError("two-frame release future must be observation t+35")
        if self.teacher_end_delta == self.release_future_delta:
            raise ValueError("Teacher and release future endpoints must remain distinct")
        if self.expert_padded_action_steps < self.expert_valid_action_steps:
            raise ValueError("padded Expert horizon is shorter than its valid horizon")
        if self.native_action_dim > self.flow_action_dim:
            raise ValueError("native actions cannot exceed the Flow action width")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": TEMPORAL_SCHEMA_VERSION, **asdict(self)}

    def validate_manifest_row(self, row: Mapping[str, Any]) -> None:
        required = {"start_step", "end_step", "control_hz"}
        missing = sorted(required.difference(row))
        if missing:
            raise KeyError(f"RoboTwin row is missing temporal fields: {missing}")
        start = int(row["start_step"])
        end = int(row["end_step"])
        if end - start != self.teacher_action_steps:
            raise ValueError(
                f"Teacher window must contain 36 actions, got end-start={end - start}"
            )
        if abs(float(row["control_hz"]) - self.control_hz) > 1e-6:
            raise ValueError(f"RoboTwin Stage-2 requires 30 Hz, got {row['control_hz']}")
        if "teacher_end_step" in row and int(row["teacher_end_step"]) != start + 36:
            raise ValueError("teacher_end_step must equal start_step+36")
        if "release_future_step" in row and int(row["release_future_step"]) != start + 35:
            raise ValueError("release_future_step must equal start_step+35")

    def expected_action_mask(self, *, batch_size: int) -> tuple[int, int, int, int]:
        return (
            int(batch_size),
            self.expert_padded_action_steps,
            self.expert_valid_action_steps,
            self.native_action_dim,
        )


DEFAULT_TEMPORAL_CONTRACT = RoboTwinStage2TemporalContract()
DENSE_S1_TEMPORAL_CONTRACT = RoboTwinStage2TemporalContract(
    identifier=ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT,
    row_count=ROBOTWIN_STAGE2_DENSE_S1_ROW_COUNT,
)


def is_robotwin_stage2_temporal_contract(value: Any) -> bool:
    identifier = value.identifier if isinstance(value, RoboTwinStage2TemporalContract) else value
    return str(identifier or "").strip() in ROBOTWIN_STAGE2_TEMPORAL_CONTRACTS


def resolve_robotwin_stage2_temporal_contract(
    value: str | RoboTwinStage2TemporalContract | None,
) -> RoboTwinStage2TemporalContract:
    if isinstance(value, RoboTwinStage2TemporalContract):
        return value
    identifier = str(value or "").strip() or ROBOTWIN_OFFICIAL_12S_TEMPORAL_CONTRACT
    if identifier not in ROBOTWIN_STAGE2_TEMPORAL_CONTRACTS:
        raise ValueError(
            "unsupported RoboTwin stage2_temporal_contract "
            f"{identifier!r}; expected one of {sorted(ROBOTWIN_STAGE2_TEMPORAL_CONTRACTS)!r}"
        )
    return (
        DENSE_S1_TEMPORAL_CONTRACT
        if identifier == ROBOTWIN_OFFICIAL_12S_DENSE_S1_TEMPORAL_CONTRACT
        else DEFAULT_TEMPORAL_CONTRACT
    )


def resolve_stage2_temporal_contract(value: Any):
    """Dispatch one phased Stage-2 temporal identifier to its domain contract."""

    from latent_action_model.bridge_stage2.temporal import (
        BridgeStage2TemporalContract,
        is_bridge_stage2_temporal_contract,
        resolve_bridge_stage2_temporal_contract,
    )

    if isinstance(value, BridgeStage2TemporalContract) or is_bridge_stage2_temporal_contract(
        value
    ):
        return resolve_bridge_stage2_temporal_contract(value)
    if isinstance(value, RoboTwinStage2TemporalContract) or is_robotwin_stage2_temporal_contract(
        value
    ):
        return resolve_robotwin_stage2_temporal_contract(value)
    from latent_action_model.libero_stage2.runtime_contract import (
        resolve_libero_stage2_temporal_contract,
    )

    return resolve_libero_stage2_temporal_contract(value)
