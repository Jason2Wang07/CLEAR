"""Frozen v1 storage schema."""

from __future__ import annotations

SCHEMA_VERSION = "lawam-effect-windows-v1"

FIELDS = {
    "sample_id": ["string", []], "task_id": ["string", []], "effect_family": ["string", []],
    "source_id": ["string", []], "source_commit": ["string", []], "controller_id": ["string", []],
    "episode_id": ["int64", []], "episode_seed": ["int64", []], "start_step": ["int32", []],
    "end_step": ["int32", []], "control_hz": ["float32", []], "tau": ["float32", []], "split": ["string", []],
    "rgb_start": ["uint8", [3, 256, 256]], "rgb_end": ["uint8", [3, 256, 256]],
    "target_seg_start": ["uint32", [256, 256]], "target_seg_end": ["uint32", [256, 256]],
    "object_pose_start": ["float32", [7]], "object_pose_end": ["float32", [7]],
    "reference_pose_start": ["float32", [7]], "reference_pose_end": ["float32", [7]],
    "active_object_id": ["string", []],
    "aux_object_poses_start": ["float32", [3, 7]], "aux_object_poses_end": ["float32", [3, 7]],
    "aux_object_mask": ["uint8", [3]],
    "effect_raw": ["float32", [12]], "effect_mask": ["uint8", [12]], "effect_valid": ["uint8", []],
    "effect_direction": ["string", []], "effect_magnitude": ["float32", []],
    "eef_pose_start": ["float32", [7]], "eef_pose_end": ["float32", [7]],
    "gripper_start": ["float32", []], "gripper_end": ["float32", []],
    "state14_start": ["float32", [14]], "state14_end": ["float32", [14]],
    "delta_state14": ["float32", [14]], "state_mask": ["uint8", [14]], "embodiment_id": ["string", []],
    "contact_summary": ["float32", [4]], "grasp_summary": ["float32", [4]],
    "phase": ["string", []], "outcome": ["string", []], "quality": ["float32", []],
    "replay_valid": ["uint8", []], "render_valid": ["uint8", []], "source_sha256": ["string", []],
}


def schema_document() -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "image_layout": "CHW", "pose_convention": "xyz+qwxyz",
        "tau_seconds": 1.6, "effect_storage_width": 12,
        "optional_fields": [
            "reference_pose_start", "reference_pose_end", "aux_object_poses_start",
            "aux_object_poses_end", "aux_object_mask", "depth_start", "depth_end",
        ],
        "fields": {name: {"dtype": spec[0], "shape": spec[1]} for name, spec in FIELDS.items()},
    }
