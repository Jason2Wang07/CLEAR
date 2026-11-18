"""Versioned LIBERO storage contracts built on the frozen ActionVAE shapes."""

from __future__ import annotations

import copy

from lawam_effect_dataset.schema import schema_document as base_effect_schema_document
from z_action_sidecar.schema import HORIZON, anchor_steps_for_horizon, dense_schema_document


EFFECT_SCHEMA_VERSION = "libero-effect-windows-v1"
SIDECAR_SCHEMA_VERSION = "libero-z-action-sidecar-v1"
NORMALIZATION_SCHEMA_VERSION = "libero-z-action-normalization-v1"
EFFECT_SCHEMA_VERSION_12S = "libero-effect-windows-v1-12s"
SIDECAR_SCHEMA_VERSION_12S = "libero-z-action-sidecar-v1-12s"
NORMALIZATION_SCHEMA_VERSION_12S = "libero-z-action-normalization-v1-12s"


def versions_for_horizon(horizon_actions: int) -> tuple[str, str, str]:
    value = int(horizon_actions)
    if value == HORIZON:
        return EFFECT_SCHEMA_VERSION, SIDECAR_SCHEMA_VERSION, NORMALIZATION_SCHEMA_VERSION
    if value == 24:
        return (
            EFFECT_SCHEMA_VERSION_12S,
            SIDECAR_SCHEMA_VERSION_12S,
            NORMALIZATION_SCHEMA_VERSION_12S,
        )
    raise ValueError(f"unsupported LIBERO horizon {value}; expected 24 or {HORIZON}")


def effect_schema_document(*, horizon_actions: int = HORIZON) -> dict:
    value = int(horizon_actions)
    effect_version, _, _ = versions_for_horizon(value)
    document = copy.deepcopy(base_effect_schema_document())
    document.update(
        {
            "schema_version": effect_version,
            "artifact_name": "libero_effect_windows_v1" if value == HORIZON else "libero_effect_windows_v1_12s",
            "source_dataset": "LIBERO official HDF5",
            "tau_seconds": value / 20.0,
        }
    )
    document["image"] = {
        "camera": "agentview_rgb",
        "source_shape": [128, 128, 3],
        "stored_shape": [3, 256, 256],
        "resize": "bilinear",
        "source_image_convention_attribute": "opengl",
        "vertical_flip": False,
    }
    document["transition_alignment"] = {
        "start": "obs[start_step]",
        "actions": "actions[start_step:end_step]",
        "end": "obs[end_step]",
        "last_source_action": "excluded because no next observation is stored",
    }
    document["effect"] = {
        "family": "task_id",
        "layout": [
            "start-frame EEF translation delta xyz [m]",
            "start-frame EEF relative rotation vector [rad]",
            "physical gripper width delta [m]",
            "five reserved zeros",
        ],
        "mask": "first seven dimensions valid; object/contact dimensions unavailable",
    }
    document["unavailable_labels"] = [
        "target_segmentation",
        "active_object_pose",
        "contact",
        "grasp",
    ]
    return document


def sidecar_schema_document(*, horizon_actions: int = HORIZON) -> dict:
    value = int(horizon_actions)
    _, sidecar_version, _ = versions_for_horizon(value)
    anchors = anchor_steps_for_horizon(value)
    document = copy.deepcopy(
        dense_schema_document(
            horizon=value,
            control_hz=20.0,
            schema_version=sidecar_version,
            artifact_name=(
                "libero_z_action_sidecar_v1"
                if value == HORIZON
                else "libero_z_action_sidecar_v1_12s"
            ),
            anchor_steps=anchors,
        )
    )
    document.update(
        {
            "schema_version": sidecar_version,
            "source_dataset": "LIBERO official HDF5",
            "duration_seconds": value / 20.0,
        }
    )
    document["waypoint"]["alignment"] = (
        "post_action; waypoint[t] is saved LIBERO obs at start_step+t+1"
    )
    document["native_actions"].update(
        {
            "semantics": (
                "raw normalized 7D LIBERO OSC_POSE delta command stored in the official HDF5"
            ),
            "execution": "requires the matching LIBERO OSC_POSE controller configuration",
        }
    )
    document["e39"]["semantics"] = (
        "zero with e39_valid=0 until the selected LIBERO Stage-1 posterior cache is joined"
    )
    document["contact"] = {
        "semantics": "not present in official HDF5; values are zero and masks are disabled",
        "binary_rule": None,
        "scope": "unavailable",
        "raw_fields": [],
    }
    document["masks"].update(
        {
            "time_mask": f"all {value} saved post-action observations are valid",
            "pose_mask": f"all {value} saved EEF poses are valid",
            "gripper_mask": f"all {value} physical gripper widths are valid",
            "contact_mask": "always false until simulator replay enrichment",
            "grasp_mask": "always false until simulator replay enrichment",
        }
    )
    document["manifest"].pop("dense_required_fields", None)
    return document
