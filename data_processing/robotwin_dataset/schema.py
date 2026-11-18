"""Versioned RoboTwin Stage-1 effect release schema."""

from __future__ import annotations

import copy

from lawam_effect_dataset.schema import schema_document as base_schema_document


SCHEMA_VERSION = "robotwin-effect-windows-v1"
SCHEMA_VERSION_12S = "robotwin-effect-windows-v1-12s"


def schema_version_for_horizon(horizon: int) -> str:
    value = int(horizon)
    if value == 48:
        return SCHEMA_VERSION
    if value == 36:
        return SCHEMA_VERSION_12S
    raise ValueError(f"unsupported RoboTwin effect horizon {value}; expected 36 or 48")


def schema_document(*, horizon: int = 48, control_hz: float = 30.0) -> dict:
    value = int(horizon)
    schema_version = schema_version_for_horizon(value)
    document = copy.deepcopy(base_schema_document())
    document.update(
        {
            "schema_version": schema_version,
            "artifact_name": (
                "robotwin_effect_windows_v1" if value == 48 else "robotwin_effect_windows_v1_12s"
            ),
            "source_dataset": "jialei02/robotwin_merged LeRobot v3",
            "tau_seconds": value / float(control_hz),
        }
    )
    document["image"] = {
        "camera": "cam_high",
        "source_shape": [480, 640, 3],
        "stored_shape": [3, 256, 256],
        "resize": "max-4:3 centered crop then antialiased bilinear resize",
        "vertical_flip": False,
    }
    document["episode_family"] = {
        "contract": "55 contiguous 500-episode source blocks mapped to 50 benchmark families",
        "extra_variant_blocks": {
            "50": "adjust_bottle",
            "51": "hanging_mug",
            "52": "place_a2b_left",
            "53": "place_fan",
            "54": "rotate_qrcode",
        },
    }
    document["state14"] = {
        "layout": [
            "left world delta xyz / relative rotvec / gripper delta",
            "right world delta xyz / relative rotvec / gripper delta",
        ],
        "dimensions_per_arm": 7,
        "mask": "all 14 dimensions valid",
    }
    document["effect"] = {
        "layout": [
            "left start-frame translation delta xyz and relative rotation vector",
            "right start-frame translation delta xyz and relative rotation vector",
        ],
        "mask": "all 12 dimensions valid; DINO teacher remains image-only",
    }
    document["legacy_single_arm_fields"] = "eef_pose_* and gripper_* store the left arm only"
    return document
