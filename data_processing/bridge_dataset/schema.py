"""Frozen BridgeData V2 contracts for the 1.2-second latent pipeline."""

from __future__ import annotations

import copy

from lawam_effect_dataset.schema import schema_document as base_effect_schema_document
from z_action_sidecar.schema import dense_schema_document, fields_for_horizon


EFFECT_SCHEMA_VERSION = "bridge-effect-windows-v1-12s"
SIDECAR_SCHEMA_VERSION = "bridge-z-action-sidecar-v1-12s-resampled24"
NORMALIZATION_SCHEMA_VERSION = "bridge-z-action-normalization-v1-12s-resampled24"
RELEASE_SCHEMA_VERSION = "bridge-actionvae-release-v1-12s-resampled24"
STAGE2_MANIFEST_SCHEMA_VERSION = "bridge-stage2-manifest-v1-12s"

SOURCE_CONTROL_HZ = 5.0
PHYSICAL_ACTION_STEPS = 6
REPRESENTATION_HZ = 20.0
REPRESENTATION_STEPS = 24
WAYPOINT_DIM = 7
SOURCE_ACTION_DIM = 7
MAX_NATIVE_ACTION_DIM = 8
TAU_SECONDS = 1.2
PRIMARY_CAMERA = "observation.images.image_0"


def effect_schema_document() -> dict:
    document = copy.deepcopy(base_effect_schema_document())
    document.update(
        {
            "schema_version": EFFECT_SCHEMA_VERSION,
            "artifact_name": "bridge_effect_windows_v1_12s",
            "source_dataset": "BrunoM42/bridge_orig_lerobot",
            "tau_seconds": TAU_SECONDS,
        }
    )
    document["image"] = {
        "camera": PRIMARY_CAMERA,
        "source_shape": [256, 256, 3],
        "stored_shape": [3, 256, 256],
        "resize": "none",
        "vertical_flip": False,
        "views": "single primary view",
    }
    document["transition_alignment"] = {
        "start": "state/image[start_step]",
        "actions": "action[start_step:start_step+6]",
        "end": "state/image[start_step+6]",
        "last_source_action": "excluded because it has no next observation",
    }
    document["source_state"] = {
        "layout": ["xyz", "roll-pitch-yaw", "reserved pad", "normalized gripper"],
        "pose_conversion": "Rz(yaw) @ Ry(pitch) @ Rx(roll) @ bridge_default_rotation",
    }
    document["effect"] = {
        "layout": [
            "start-frame EEF translation delta xyz [m]",
            "start-frame EEF relative principal rotation vector [rad]",
            "physical two-finger aperture delta [m]",
            "five reserved zeros",
        ],
        "mask": "first seven dimensions valid",
    }
    document["unavailable_labels"] = [
        "target_segmentation",
        "active_object_pose",
        "contact",
        "grasp",
    ]
    return document


def action_fields() -> dict[str, tuple[str, tuple[int, ...]]]:
    fields = dict(fields_for_horizon(REPRESENTATION_STEPS, anchor_steps=(6, 12, 18, 24)))
    fields["physical_valid_count"] = ("uint8", ())
    return fields


def sidecar_schema_document() -> dict:
    document = copy.deepcopy(
        dense_schema_document(
            horizon=REPRESENTATION_STEPS,
            control_hz=REPRESENTATION_HZ,
            schema_version=SIDECAR_SCHEMA_VERSION,
            artifact_name="bridge_z_action_sidecar_v1_12s_resampled24",
            anchor_steps=(6, 12, 18, 24),
        )
    )
    document.update(
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "source_dataset": "BrunoM42/bridge_orig_lerobot",
            "duration_seconds": TAU_SECONDS,
            "horizon_actions": REPRESENTATION_STEPS,
            "representation_steps": REPRESENTATION_STEPS,
            "representation_hz": REPRESENTATION_HZ,
            "physical_action_steps": PHYSICAL_ACTION_STEPS,
            "source_control_hz": SOURCE_CONTROL_HZ,
        }
    )
    document["waypoint"].update(
        {
            "alignment": (
                "24 post-phase representation slots obtained by four-way subdivision of each "
                "of six source transitions"
            ),
            "interpolation": {
                "position": "linear",
                "rotation": "shortest-arc quaternion SLERP",
                "gripper": "source-start ZOH for phases 1-3; source endpoint at phase 4",
            },
        }
    )
    document["native_actions"].update(
        {
            "semantics": "raw Bridge delta xyz, delta RPY, absolute open-gripper target",
            "valid_representation_indices_zero_based": [3, 7, 11, 15, 19, 23],
            "execution": "six source actions only; interpolated slots are never executable actions",
        }
    )
    document["physical_valid_count"] = {
        "shape": [],
        "dtype": "uint8",
        "value": PHYSICAL_ACTION_STEPS,
        "semantics": "number of real 5 Hz source transitions represented by the row",
    }
    document["gripper"] = {
        "source": "normalized Bridge gripper state",
        "physical_aperture_m": "0.020 + 0.050 * source_gripper",
        "clipping": False,
    }
    document["manifest"]["physical_window_contract"] = (
        "end_step-start_step=6 while horizon_actions=24; source and representation clocks are distinct"
    )
    document["fields"]["physical_valid_count"] = {"dtype": "uint8", "shape": []}
    return document


def stage2_modality_document() -> dict:
    """Single-primary-view mapping consumed by BridgeDataConfig."""

    scalar_state = {
        name: {
            "start": start,
            "end": end,
            "absolute": True,
            "dtype": "float32",
            "original_key": "observation.state",
        }
        for name, start, end in (
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("eef_orientation_rotvec", 3, 6),
            ("gripper", 7, 8),
        )
    }
    scalar_state["eef_orientation_rotvec"]["rotation_type"] = "euler_angles_rpy"
    scalar_action = {
        name: {
            "start": start,
            "end": end,
            "absolute": absolute,
            "dtype": "float32",
            "original_key": "action",
        }
        for name, start, end, absolute in (
            ("x", 0, 1, False),
            ("y", 1, 2, False),
            ("z", 2, 3, False),
            ("eef_orientation_rotvec", 3, 6, False),
            ("gripper", 6, 7, True),
        )
    }
    scalar_action["eef_orientation_rotvec"]["rotation_type"] = "euler_angles_rpy"
    return {
        "state": scalar_state,
        "action": scalar_action,
        "video": {
            "primary_view": {"original_key": PRIMARY_CAMERA},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
