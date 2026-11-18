"""Exact-row Stage-2 manifest and paired latent lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .export_teacher_cache import CACHE_SCHEMA_VERSION, sha256_file
from .temporal import resolve_robotwin_stage2_temporal_contract


MODE_TO_SPLIT = {"train": "train", "all": "train", "val": "validation", "test": "test"}
REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "split",
    "task_id",
    "episode_id",
    "start_step",
    "end_step",
    "control_hz",
    "window_sha256",
}


def _resolve_manifest(path: str | Path, split: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidate = path / f"{split}.parquet"
        if not candidate.is_file() and (path / "manifests").is_dir():
            candidate = path / "manifests" / f"{split}.parquet"
        path = candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class RoboTwinStage2RowIndex:
    """Bind one policy split to exactly the selected H36 teacher windows."""

    def __init__(
        self,
        *,
        manifest: str | Path,
        paired_cache: str | Path,
        mode: str,
        split_override: str | None = None,
        temporal_contract: str | None = None,
    ) -> None:
        split = str(split_override or MODE_TO_SPLIT.get(str(mode).lower(), "")).lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"cannot resolve Stage-2 split from mode={mode!r}")
        self.split = split
        self.temporal = resolve_robotwin_stage2_temporal_contract(temporal_contract)
        self.manifest_path = _resolve_manifest(manifest, split)
        frame = pd.read_parquet(self.manifest_path)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"Stage-2 manifest is missing columns: {missing}")
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"Stage-2 manifest must contain only split={split}")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Stage-2 manifest contains duplicate sample IDs")
        for row in frame.to_dict("records"):
            self.temporal.validate_manifest_row(row)
        self.frame = frame.reset_index(drop=True)

        self.paired_cache_path = Path(paired_cache).expanduser().resolve()
        if not self.paired_cache_path.is_file():
            raise FileNotFoundError(self.paired_cache_path)
        with np.load(self.paired_cache_path, allow_pickle=False) as payload:
            condition_cache_name = (
                "lawam" if "z_lawam" in payload.files else "effect"
            )
            condition_key = f"z_{condition_cache_name}"
            condition_raw_key = f"{condition_key}_raw"
            required = {
                "sample_id",
                "split",
                condition_key,
                condition_raw_key,
                "z_action",
                "z_action_logvar",
                "teacher_trajectory",
                "teacher_current_state",
                "teacher_time_mask",
                "teacher_pose_mask",
                "teacher_gripper_mask",
                "teacher_current_state_mask",
                "physical_valid_count",
            }
            missing_cache = sorted(required.difference(payload.files))
            if missing_cache:
                raise ValueError(f"paired teacher cache is missing fields: {missing_cache}")
            cache_ids = np.asarray(payload["sample_id"]).astype(str)
            cache_splits = np.asarray(payload["split"]).astype(str)
            select = cache_splits == split
            split_ids = cache_ids[select]
            condition = np.asarray(payload[condition_key][select], dtype=np.float32)
            condition_raw = np.asarray(
                payload[condition_raw_key][select], dtype=np.float32
            )
            action = np.asarray(payload["z_action"][select], dtype=np.float32)
            action_logvar = np.asarray(payload["z_action_logvar"][select], dtype=np.float32)
            trajectory = np.asarray(payload["teacher_trajectory"][select], dtype=np.float32)
            current_state = np.asarray(payload["teacher_current_state"][select], dtype=np.float32)
            time_mask = np.asarray(payload["teacher_time_mask"][select], dtype=np.uint8)
            pose_mask = np.asarray(payload["teacher_pose_mask"][select], dtype=np.uint8)
            gripper_mask = np.asarray(payload["teacher_gripper_mask"][select], dtype=np.uint8)
            current_state_mask = np.asarray(
                payload["teacher_current_state_mask"][select], dtype=np.uint8
            )
            physical_valid_count = np.asarray(
                payload["physical_valid_count"][select], dtype=np.int16
            )
        if condition.shape != (len(split_ids), 32) or action.shape != condition.shape:
            raise ValueError("paired teacher cache latent shape mismatch")
        if condition_raw.shape != condition.shape:
            raise ValueError("paired teacher cache raw condition shape mismatch")
        expected_rows = len(split_ids)
        expected_shapes = {
            "z_action_logvar": (expected_rows, 32),
            "teacher_trajectory": (expected_rows, 36, 14),
            "teacher_current_state": (expected_rows, 14),
            "teacher_time_mask": (expected_rows, 36),
            "teacher_pose_mask": (expected_rows, 36, 2),
            "teacher_gripper_mask": (expected_rows, 36, 2),
            "teacher_current_state_mask": (expected_rows, 14),
            "physical_valid_count": (expected_rows,),
        }
        actual_shapes = {
            "z_action_logvar": action_logvar.shape,
            "teacher_trajectory": trajectory.shape,
            "teacher_current_state": current_state.shape,
            "teacher_time_mask": time_mask.shape,
            "teacher_pose_mask": pose_mask.shape,
            "teacher_gripper_mask": gripper_mask.shape,
            "teacher_current_state_mask": current_state_mask.shape,
            "physical_valid_count": physical_valid_count.shape,
        }
        mismatched_shapes = {
            name: (actual_shapes[name], expected)
            for name, expected in expected_shapes.items()
            if actual_shapes[name] != expected
        }
        if mismatched_shapes:
            raise ValueError(f"RoboTwin Stage-2 teacher field shape mismatch: {mismatched_shapes}")
        for name, value in (
            ("teacher_time_mask", time_mask),
            ("teacher_pose_mask", pose_mask),
            ("teacher_gripper_mask", gripper_mask),
            ("teacher_current_state_mask", current_state_mask),
        ):
            if not np.isin(value, (0, 1)).all():
                raise ValueError(f"{name} must be binary")
        if not np.all(physical_valid_count == self.temporal.physical_valid_count):
            raise ValueError(
                "RoboTwin physical_valid_count must match the selected temporal contract"
            )
        manifest_ids = self.frame["sample_id"].astype(str).to_numpy()
        if set(split_ids.tolist()) != set(manifest_ids.tolist()):
            raise ValueError("Stage-2 manifest/cache sample IDs do not exact-set join")
        if len(set(split_ids.tolist())) != len(split_ids):
            raise ValueError("paired teacher cache contains duplicate split sample IDs")
        lookup = {sample_id: index for index, sample_id in enumerate(split_ids.tolist())}
        order = np.asarray([lookup[value] for value in manifest_ids.tolist()], dtype=np.int64)
        self.sample_ids = manifest_ids.astype("U32")
        self.condition_cache_name = condition_cache_name
        self.z_condition = np.ascontiguousarray(condition[order])
        self.z_condition_raw = np.ascontiguousarray(condition_raw[order])
        if condition_cache_name == "effect":
            self.z_effect = self.z_condition
        self.z_action = np.ascontiguousarray(action[order])
        self.z_action_logvar = np.ascontiguousarray(action_logvar[order])
        self.teacher_trajectory = np.ascontiguousarray(trajectory[order])
        self.teacher_current_state = np.ascontiguousarray(current_state[order])
        self.teacher_time_mask = np.ascontiguousarray(time_mask[order])
        self.teacher_pose_mask = np.ascontiguousarray(pose_mask[order])
        self.teacher_gripper_mask = np.ascontiguousarray(gripper_mask[order])
        self.teacher_current_state_mask = np.ascontiguousarray(current_state_mask[order])
        self.physical_valid_count = np.ascontiguousarray(physical_valid_count[order])
        if (
            not np.isfinite(self.z_condition).all()
            or not np.isfinite(self.z_condition_raw).all()
            or not np.isfinite(self.z_action).all()
        ):
            raise ValueError("paired teacher cache contains NaN/Inf")
        if not np.isfinite(self.z_action_logvar).all():
            raise ValueError("paired teacher cache action logvar contains NaN/Inf")
        if not np.isfinite(self.teacher_trajectory).all() or not np.isfinite(
            self.teacher_current_state
        ).all():
            raise ValueError("paired teacher cache trajectory/state contains NaN/Inf")
        self.provenance = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "split": split,
            "rows": len(self.frame),
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256_file(self.manifest_path),
            "paired_cache": str(self.paired_cache_path),
            "paired_cache_sha256": sha256_file(self.paired_cache_path),
            "exact_join": True,
        }

    def __len__(self) -> int:
        return len(self.frame)

    def row(self, index: int) -> dict[str, Any]:
        normalized = int(index) % len(self)
        return self.frame.iloc[normalized].to_dict()

    def attach(
        self,
        sample: dict[str, Any],
        *,
        index: int,
        episode_id: int,
        start_step: int,
    ) -> dict[str, Any]:
        normalized = int(index) % len(self)
        row = self.frame.iloc[normalized]
        if int(row["episode_id"]) != int(episode_id) or int(row["start_step"]) != int(start_step):
            raise ValueError("policy row identity changed before paired latent attachment")
        sample["sample_id"] = str(row["sample_id"])
        condition_key = f"z_{self.condition_cache_name}"
        sample[condition_key] = torch.from_numpy(
            self.z_condition[normalized].copy()
        )
        sample[f"{condition_key}_raw"] = torch.from_numpy(
            self.z_condition_raw[normalized].copy()
        )
        sample["z_action"] = torch.from_numpy(self.z_action[normalized].copy())
        sample["z_action_logvar"] = torch.from_numpy(
            self.z_action_logvar[normalized].copy()
        )
        sample["teacher_trajectory"] = torch.from_numpy(
            self.teacher_trajectory[normalized].copy()
        )
        sample["teacher_current_state"] = torch.from_numpy(
            self.teacher_current_state[normalized].copy()
        )
        sample["teacher_time_mask"] = torch.from_numpy(
            self.teacher_time_mask[normalized].copy()
        ).bool()
        sample["teacher_pose_mask"] = torch.from_numpy(
            self.teacher_pose_mask[normalized].copy()
        ).bool()
        sample["teacher_gripper_mask"] = torch.from_numpy(
            self.teacher_gripper_mask[normalized].copy()
        ).bool()
        sample["teacher_current_state_mask"] = torch.from_numpy(
            self.teacher_current_state_mask[normalized].copy()
        ).bool()
        predictor_state = torch.zeros(32, dtype=torch.float32)
        predictor_state[:14] = sample["teacher_current_state"]
        sample["action_predictor_state"] = predictor_state
        sample["physical_valid_count"] = int(self.physical_valid_count[normalized])
        return sample
