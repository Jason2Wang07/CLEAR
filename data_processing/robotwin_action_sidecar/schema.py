"""Frozen storage and semantic contract for RoboTwin ActionAE v1."""

from __future__ import annotations


SCHEMA_VERSION = "robotwin-z-action-sidecar-v1"
NORMALIZATION_SCHEMA_VERSION = "robotwin-z-action-normalization-v1"
ARTIFACT_NAME = "robotwin_z_action_sidecar_v1"
SCHEMA_VERSION_12S = "robotwin-z-action-sidecar-v1-12s"
SCHEMA_VERSION_12S_LAWAM = "robotwin-z-action-sidecar-v2-12s-lawam-idm"
NORMALIZATION_SCHEMA_VERSION_12S = "robotwin-z-action-normalization-v1-12s"
ARTIFACT_NAME_12S = "robotwin_z_action_sidecar_v1_12s"
ARTIFACT_NAME_12S_LAWAM = "robotwin_z_action_sidecar_v2_12s_lawam_idm"
HORIZON = 48
CONTROL_HZ = 30.0
TRAJECTORY_DIM = 14
STATE_DIM = 14
EFFECT_DIM = 32
NATIVE_ACTION_DIM = 16
ARM_COUNT = 2
ANCHOR_STEPS = (12, 24, 36, 48)


def versions_for_horizon(
    horizon: int, *, condition_source: str = "e39"
) -> tuple[str, str, str]:
    value = int(horizon)
    if condition_source not in {"e39", "lawam_idm"}:
        raise ValueError(f"unsupported condition source {condition_source!r}")
    if value == HORIZON:
        if condition_source != "e39":
            raise ValueError("lawam_idm sidecars are only reviewed for the 36-step horizon")
        return SCHEMA_VERSION, NORMALIZATION_SCHEMA_VERSION, ARTIFACT_NAME
    if value == 36:
        if condition_source == "lawam_idm":
            return (
                SCHEMA_VERSION_12S_LAWAM,
                NORMALIZATION_SCHEMA_VERSION_12S,
                ARTIFACT_NAME_12S_LAWAM,
            )
        return SCHEMA_VERSION_12S, NORMALIZATION_SCHEMA_VERSION_12S, ARTIFACT_NAME_12S
    raise ValueError(f"unsupported RoboTwin horizon {value}; expected 36 or {HORIZON}")


def anchor_steps_for_horizon(horizon: int) -> tuple[int, int, int, int]:
    value = int(horizon)
    if value < 4 or value % 4:
        raise ValueError(f"horizon must be a positive multiple of four, got {value}")
    quarter = value // 4
    return (quarter, 2 * quarter, 3 * quarter, value)

MANIFEST_FIELDS: dict[str, str] = {
    "sample_id": "string",
    "group_id": "string",
    "split": "string",
    "task_id": "string",
    "source_id": "string",
    "source_commit": "string",
    "controller_id": "string",
    "episode_id": "integer",
    "start_step": "integer",
    "end_step": "integer",
    "effect_shard_path": "string",
    "effect_shard_row": "integer",
    "shard_path": "string",
    "shard_row": "integer",
    "window_sha256": "string",
    "e39_valid": "boolean",
}

def condition_field_names(condition_source: str) -> tuple[str, str]:
    if condition_source == "e39":
        return "e39_mean", "e39_valid"
    if condition_source == "lawam_idm":
        return "lawam_mean", "lawam_valid"
    raise ValueError(f"unsupported condition source {condition_source!r}")


def fields_for_horizon(
    horizon: int = HORIZON, *, condition_source: str = "e39"
) -> dict[str, tuple[str, tuple[int, ...]]]:
    value = int(horizon)
    anchors = anchor_steps_for_horizon(value)
    mean_field, valid_field = condition_field_names(condition_source)
    fields = {
    "sample_id": ("string", ()),
    "group_id": ("string", ()),
    "split": ("string", ()),
    "task_id": ("string", ()),
    "source_id": ("string", ()),
    "source_commit": ("string", ()),
    "source_sha256": ("string", ()),
    "controller_id": ("string", ()),
    "embodiment_id": ("string", ()),
    "episode_id": ("int64", ()),
    "episode_seed": ("int64", ()),
    "start_step": ("int32", ()),
    "end_step": ("int32", ()),
    "control_hz": ("float32", ()),
    "dual_arm_pose_start": ("float32", (NATIVE_ACTION_DIM,)),
    "dual_arm_pose_end": ("float32", (NATIVE_ACTION_DIM,)),
    "waypoints": ("float32", (value, TRAJECTORY_DIM)),
    "time_mask": ("uint8", (value,)),
    "arm_pose_mask": ("uint8", (value, ARM_COUNT)),
    "arm_gripper_mask": ("uint8", (value, ARM_COUNT)),
    "native_actions": ("float32", (value, NATIVE_ACTION_DIM)),
    "native_action_mask": ("uint8", (value, NATIVE_ACTION_DIM)),
    "current_state": ("float32", (STATE_DIM,)),
    "current_state_mask": ("uint8", (STATE_DIM,)),
    mean_field: ("float32", (EFFECT_DIM,)),
    valid_field: ("uint8", ()),
    "physical_anchors": ("float32", (len(anchors), TRAJECTORY_DIM)),
    "physical_anchor_mask": ("uint8", (len(anchors), TRAJECTORY_DIM)),
    "window_sha256": ("string", ()),
    }
    return fields


FIELDS: dict[str, tuple[str, tuple[int, ...]]] = fields_for_horizon()


def schema_document(
    *, horizon: int = HORIZON, condition_source: str = "e39", dense: bool = False
) -> dict:
    value = int(horizon)
    schema_version, normalization_version, artifact_name = versions_for_horizon(
        value, condition_source=condition_source
    )
    fields = fields_for_horizon(value, condition_source=condition_source)
    mean_field, valid_field = condition_field_names(condition_source)
    condition_semantics = (
        "deterministic posterior mean from the frozen validation-selected RoboTwin Effect Encoder"
        if condition_source == "e39"
        else "deterministic posterior mean from the frozen official LaWAM release IDM"
    )
    manifest_fields = {
        name: dtype
        for name, dtype in MANIFEST_FIELDS.items()
        if name != "e39_valid" and (not dense or not name.startswith("effect_shard_"))
    }
    manifest_fields[valid_field] = "boolean"
    document = {
        "schema_version": schema_version,
        "artifact_name": artifact_name,
        "horizon_actions": value,
        "control_hz": CONTROL_HZ,
        "duration_seconds": value / CONTROL_HZ,
        "latent": {
            "name": "z_action",
            "batch_shape": ["B", 1, EFFECT_DIM],
            "deterministic": True,
        },
        "waypoint": {
            "batch_shape": ["B", value, TRAJECTORY_DIM],
            "alignment": "post_action; waypoint[t] is source observation.state at start_step+t+1",
            "layout": [
                "left R_start^T @ (position_t-position_start) xyz [m]",
                "left Log(R_start^T @ R_t) rotvec [rad]",
                "left physical gripper value",
                "right R_start^T @ (position_t-position_start) xyz [m]",
                "right Log(R_start^T @ R_t) rotvec [rad]",
                "right physical gripper value",
            ],
            "source_pose_convention": "per arm xyz+qwxyz+gripper",
            "decoder_output_activation": "identity",
        },
        "current_state": {
            "shape": [STATE_DIM],
            "layout": "effect-release state14_start: per arm world xyz, world rotvec, gripper",
        },
        "native_actions": {
            "shape": [value, NATIVE_ACTION_DIM],
            "semantics": "raw dual-arm absolute EEF command from source action rows start:end",
            "layout": "left xyz+qwxyz+gripper, right xyz+qwxyz+gripper",
        },
        "condition": {
            "source": condition_source,
            "mean_field": mean_field,
            "valid_field": valid_field,
            "shape": [EFFECT_DIM],
            "semantics": condition_semantics,
            "sampling": False,
        },
        "masks": {
            "time_mask": "valid post-action timestep",
            "arm_pose_mask": "valid xyz and rotvec, separately for left/right arms",
            "arm_gripper_mask": "valid physical gripper value, separately for left/right arms",
            "native_action_mask": "valid raw source action dimensions",
        },
        "normalization": {
            "schema_version": normalization_version,
            "fit_split": "train",
            "estimator": "median with max(1.4826*MAD, RMS-about-median, 1e-8) scale",
            "per_arm": True,
            "per_episode": False,
        },
        "loss_v1": {
            "type": "masked dual-arm Smooth-L1",
            "beta": 0.1,
            "weights": {"position": 1.0, "rotation_rotvec": 1.0, "gripper": 0.5},
            "reduction": "modalities are normalized by their own valid scalar-element counts across both arms",
        },
        "conditions": {
            "A0": ["current_state"],
            "A1": ["current_state", f"stopgrad({mean_field})"],
        },
        "manifest": {
            "path": "manifests/all.parquet",
            "required_fields": manifest_fields,
            "additional_fields": True,
        },
        "fields": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in fields.items()
        },
    }
    if condition_source == "e39":
        document["e39"] = {
            "shape": [EFFECT_DIM],
            "semantics": condition_semantics,
            "sampling": False,
        }
    return document
