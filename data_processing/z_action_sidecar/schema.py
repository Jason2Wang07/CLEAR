"""Frozen storage and semantic contract for ManiSkill z-action sidecar v1."""

from __future__ import annotations

SCHEMA_VERSION = "maniskill-z-action-sidecar-v1"
DENSE_SCHEMA_VERSION = "maniskill-z-action-sidecar-v2"
ARTIFACT_NAME = "maniskill_z_action_sidecar_v1"
DENSE_ARTIFACT_NAME = "maniskill_z_action_sidecar_dense_v2"
HORIZON = 32
CONTROL_HZ = 20.0
WAYPOINT_DIM = 7
STATE_DIM = 14
E39_DIM = 32
MAX_NATIVE_ACTION_DIM = 8
ANCHOR_STEPS = (8, 16, 24, 32)
TARGET_CONTACT_FORCE_THRESHOLD = 1e-6

# Parquet implementations may choose different physical representations (for
# example, pandas currently reads Arrow strings as ``object``).  The manifest
# contract therefore freezes logical types while allowing upstream passthrough
# columns to remain in the published table.
MANIFEST_FIELDS: dict[str, str] = {
    "sample_id": "string",
    "group_id": "string",
    "split": "string",
    "task_id": "string",
    "source_id": "string",
    "source_commit": "string",
    "controller_id": "string",
    "episode_id": "integer",
    "episode_seed": "integer",
    "start_step": "integer",
    "end_step": "integer",
    "control_hz": "number",
    "effect_shard_path": "string",
    "effect_shard_row": "integer",
    "shard_path": "string",
    "shard_row": "integer",
    "window_sha256": "string",
    "e39_valid": "boolean",
}

# Each HDF5 row is one official effect window. Shapes exclude the row axis.
def anchor_steps_for_horizon(horizon: int) -> tuple[int, int, int, int]:
    """Return four evenly spaced, 1-based physical anchors."""

    value = int(horizon)
    if value < 4 or value % 4:
        raise ValueError(f"horizon must be a positive multiple of four, got {value}")
    quarter = value // 4
    return (quarter, 2 * quarter, 3 * quarter, value)


def fields_for_horizon(
    horizon: int = HORIZON,
    *,
    anchor_steps: tuple[int, ...] | None = None,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    value = int(horizon)
    anchors = anchor_steps_for_horizon(value) if anchor_steps is None else tuple(anchor_steps)
    if not anchors or anchors[-1] != value or any(step < 1 or step > value for step in anchors):
        raise ValueError(f"invalid anchor steps {anchors!r} for horizon {value}")
    return {
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
    "eef_pose_start": ("float32", (7,)),
    "eef_pose_end": ("float32", (7,)),
    "waypoints": ("float32", (value, WAYPOINT_DIM)),
    "time_mask": ("uint8", (value,)),
    "pose_mask": ("uint8", (value,)),
    "gripper_mask": ("uint8", (value,)),
    "contact": ("float32", (value,)),
    "contact_peak_force": ("float32", (value,)),
    "contact_impulse": ("float32", (value, 6)),
    "contact_mask": ("uint8", (value,)),
    "grasp": ("float32", (value,)),
    "grasp_mask": ("uint8", (value,)),
    "native_actions": ("float32", (value, MAX_NATIVE_ACTION_DIM)),
    "native_action_mask": ("uint8", (value, MAX_NATIVE_ACTION_DIM)),
    "native_action_dim": ("uint8", ()),
    "current_state": ("float32", (STATE_DIM,)),
    "current_state_mask": ("uint8", (STATE_DIM,)),
    "e39_mean": ("float32", (E39_DIM,)),
    "e39_valid": ("uint8", ()),
    "physical_anchors": ("float32", (len(anchors), 8)),
    "physical_anchor_mask": ("uint8", (len(anchors), 8)),
    "replay_object_error": ("float32", (value,)),
    "replay_robot_error": ("float32", (value,)),
    "window_sha256": ("string", ()),
    }


FIELDS: dict[str, tuple[str, tuple[int, ...]]] = fields_for_horizon()


def schema_document(
    *,
    horizon: int = HORIZON,
    control_hz: float = CONTROL_HZ,
    schema_version: str = SCHEMA_VERSION,
    artifact_name: str = ARTIFACT_NAME,
    anchor_steps: tuple[int, ...] | None = None,
) -> dict:
    value = int(horizon)
    anchors = anchor_steps_for_horizon(value) if anchor_steps is None else tuple(anchor_steps)
    fields = fields_for_horizon(value, anchor_steps=anchors)
    return {
        "schema_version": schema_version,
        "artifact_name": artifact_name,
        "horizon_actions": value,
        "control_hz": float(control_hz),
        "duration_seconds": value / float(control_hz),
        "latent": {
            "name": "z_action",
            "batch_shape": ["B", 1, E39_DIM],
            "deterministic": True,
        },
        "waypoint": {
            "batch_shape": ["B", value, WAYPOINT_DIM],
            "alignment": "post_action; waypoint[t] is source state start_step+t+1",
            "layout": [
                "R_start^T @ (position_t - position_start) [m]",
                "Log(R_start^T @ R_t) [rad]",
                "physical gripper width [m]",
            ],
            "source_pose_convention": "xyz+qwxyz",
            "decoder_output_activation": "identity",
        },
        "current_state": {
            "shape": [STATE_DIM],
            "layout": "lawam state14_start: world xyz, world rotvec, physical gripper width, reserved zeros",
        },
        "native_actions": {
            "shape": [value, MAX_NATIVE_ACTION_DIM],
            "semantics": "raw source-policy output passed to env.step before controller saturation/scaling",
            "execution": "pass valid dimensions unchanged to ManiSkill; do not pre-clip to env.action_space bounds",
            "pd_ee_delta_pose": "ManiSkill clips translation/gripper component-wise and rotation by vector norm internally",
        },
        "e39": {
            "shape": [E39_DIM],
            "semantics": "deterministic posterior mean from frozen epoch-39 Stage-1 checkpoint",
            "sampling": False,
        },
        "physical_anchors": {
            "steps_1_based": list(anchors),
            "shape": [len(anchors), 8],
            "layout": "canonical waypoint 7D + binary target-contact label",
        },
        "contact": {
            "semantics": "target-object pairwise contact recorded during exact transition replay",
            "binary_rule": f"contact_peak_force > {TARGET_CONTACT_FORCE_THRESHOLD}",
            "scope": "target contact only; not arbitrary obstacle/environment contact",
            "raw_fields": ["contact_peak_force", "contact_impulse"],
        },
        "masks": {
            "time_mask": "valid post-action timestep",
            "pose_mask": "valid local position and relative rotation",
            "gripper_mask": "physical gripper exists; false for PandaStick/PushT",
            "contact_mask": "exact replay transition exists and passes replay tolerance",
            "grasp_mask": "contact_mask and embodiment has a gripper",
            "native_action_mask": "valid controller-specific action elements",
        },
        "normalization": {
            "fit_split": "train",
            "estimator": "median and 1.4826 * MAD",
            "target_groups": ["position", "rotation_rotvec", "gripper"],
            "per_episode": False,
        },
        "loss_v0": {
            "type": "masked Smooth-L1",
            "beta": 0.1,
            "weights": {"position": 1.0, "rotation_rotvec": 1.0, "gripper": 0.5},
            "reduction": "each modality divided by its own valid scalar-element count before weighting",
        },
        "conditions": {
            "A0": ["current_state"],
            "A1": ["current_state", "stopgrad(e39_mean)"],
        },
        "manifest": {
            "path": "manifests/all.parquet",
            "required_fields": dict(MANIFEST_FIELDS),
            "additional_fields": True,
        },
        "fields": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in fields.items()
        },
    }


def dense_schema_document(
    *,
    horizon: int = HORIZON,
    control_hz: float = CONTROL_HZ,
    schema_version: str = DENSE_SCHEMA_VERSION,
    artifact_name: str = DENSE_ARTIFACT_NAME,
    anchor_steps: tuple[int, ...] | None = None,
) -> dict:
    """Return the v2 contract for mask-padded dense successful-demo windows."""

    document = schema_document(
        horizon=horizon,
        control_hz=control_hz,
        schema_version=schema_version,
        artifact_name=artifact_name,
        anchor_steps=anchor_steps,
    )
    document.update(
        {
            "schema_version": schema_version,
            "artifact_name": artifact_name,
            "duration_seconds": {
                "maximum": int(horizon) / float(control_hz),
                "per_row": "valid_steps / control_hz",
            },
        }
    )
    document["waypoint"]["alignment"] = (
        "post_action; valid waypoint[t] is source state start_step+t+1; "
        "indices t >= valid_steps are zero padding"
    )
    document["e39"]["semantics"] = (
        "optional frozen Stage-1 posterior mean; zero with e39_valid=0 until dense E39 is cached"
    )
    document["contact"] = {
        "semantics": "unavailable without a replay cache that passes the source-backend tolerance",
        "binary_rule": None,
        "scope": "contact, force, impulse, and grasp values are zero with masks disabled",
        "raw_fields": ["contact_peak_force", "contact_impulse"],
    }
    document["masks"].update(
        {
            "time_mask": "true for the first valid_steps post-action timesteps only",
            "pose_mask": "time_mask for saved-state FK waypoints",
            "gripper_mask": "time_mask when a physical gripper exists",
            "contact_mask": "false when exact replay labels are unavailable",
            "grasp_mask": "false when exact replay labels are unavailable",
            "native_action_mask": (
                "valid controller-specific elements for the first valid_steps timesteps only"
            ),
        }
    )
    document["physical_anchors"]["masking"] = (
        "an anchor is valid only when its fixed 1-based timestep is <= valid_steps; "
        "contact is masked when replay labels are unavailable"
    )
    document["manifest"]["dense_required_fields"] = {
        "valid_steps": "integer",
        "padded_horizon": "integer",
        "requires_padding": "boolean",
    }
    return document
