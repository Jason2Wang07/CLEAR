"""Exact-row Bridge Stage-2 manifest and paired-teacher lookup."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .temporal import (
    BRIDGE_STAGE2_SPLIT_ROWS,
    resolve_bridge_stage2_temporal_contract,
)


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
    "language_instruction",
    "dino_start_step",
    "dino_future_step",
    "effect_teacher_step",
    "source_action_steps",
    "action_hz",
    "duration_seconds",
    "temporal_contract",
    "physical_valid_count",
}


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest(path: str | Path, split: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        candidate = resolved / f"{split}.parquet"
        if not candidate.is_file() and (resolved / "manifests").is_dir():
            candidate = resolved / "manifests" / f"{split}.parquet"
        resolved = candidate
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


class BridgeStage2RowIndex:
    """Bind one policy split to the exact 5 Hz non-empty-language Bridge rows."""

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
        if split not in BRIDGE_STAGE2_SPLIT_ROWS:
            raise ValueError(f"cannot resolve Bridge Stage-2 split from mode={mode!r}")
        self.split = split
        self.temporal = resolve_bridge_stage2_temporal_contract(temporal_contract)
        self.manifest_path = _resolve_manifest(manifest, split)
        frame = pd.read_parquet(self.manifest_path)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"Bridge Stage-2 manifest is missing columns: {missing}")
        if len(frame) != BRIDGE_STAGE2_SPLIT_ROWS[split]:
            raise ValueError(
                f"Bridge {split} manifest must contain {BRIDGE_STAGE2_SPLIT_ROWS[split]} rows, "
                f"got {len(frame)}"
            )
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"Bridge Stage-2 manifest must contain only split={split}")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Bridge Stage-2 manifest contains duplicate sample IDs")
        for row in frame.to_dict("records"):
            self.temporal.validate_manifest_row(row)
        self.frame = frame.reset_index(drop=True)

        self.paired_cache_path = Path(paired_cache).expanduser().resolve()
        if not self.paired_cache_path.is_file():
            raise FileNotFoundError(self.paired_cache_path)
        with np.load(self.paired_cache_path, allow_pickle=False) as payload:
            required = {
                "sample_id",
                "split",
                "z_effect_raw",
                "z_effect",
                "z_action_raw",
                "z_action",
                "z_action_logvar_raw",
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
                raise ValueError(f"Bridge paired teacher cache is missing fields: {missing_cache}")
            cache_ids = np.asarray(payload["sample_id"]).astype(str)
            cache_splits = np.asarray(payload["split"]).astype(str)
            select = cache_splits == split
            split_ids = cache_ids[select]
            arrays = {
                name: np.asarray(payload[name][select], dtype=dtype)
                for name, dtype in (
                    ("z_effect_raw", np.float32),
                    ("z_effect", np.float32),
                    ("z_action_raw", np.float32),
                    ("z_action", np.float32),
                    ("z_action_logvar_raw", np.float32),
                    ("z_action_logvar", np.float32),
                    ("teacher_trajectory", np.float32),
                    ("teacher_current_state", np.float32),
                    ("teacher_time_mask", np.uint8),
                    ("teacher_pose_mask", np.uint8),
                    ("teacher_gripper_mask", np.uint8),
                    ("teacher_current_state_mask", np.uint8),
                    ("physical_valid_count", np.int16),
                )
            }

        rows = len(split_ids)
        expected_shapes = {
            "z_effect_raw": (rows, 32),
            "z_effect": (rows, 32),
            "z_action_raw": (rows, 32),
            "z_action": (rows, 32),
            "z_action_logvar_raw": (rows, 32),
            "z_action_logvar": (rows, 32),
            "teacher_trajectory": (rows, 24, 7),
            "teacher_current_state": (rows, 14),
            "teacher_time_mask": (rows, 24),
            "teacher_pose_mask": (rows, 24),
            "teacher_gripper_mask": (rows, 24),
            "teacher_current_state_mask": (rows, 14),
            "physical_valid_count": (rows,),
        }
        mismatches = {
            name: (tuple(arrays[name].shape), expected)
            for name, expected in expected_shapes.items()
            if tuple(arrays[name].shape) != expected
        }
        if mismatches:
            raise ValueError(f"Bridge Stage-2 teacher field shape mismatch: {mismatches}")
        if rows != BRIDGE_STAGE2_SPLIT_ROWS[split]:
            raise ValueError(f"Bridge paired cache has the wrong {split} row count: {rows}")
        for name in (
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        ):
            if not np.isin(arrays[name], (0, 1)).all():
                raise ValueError(f"Bridge {name} must be binary")
        expected_state_mask = np.zeros((rows, 14), dtype=bool)
        expected_state_mask[:, :7] = True
        if not np.array_equal(
            arrays["teacher_current_state_mask"].astype(bool), expected_state_mask
        ):
            raise ValueError("Bridge teacher current-state mask must be true7/false7")
        if not arrays["teacher_time_mask"].astype(bool).all():
            raise ValueError("Bridge resampled ActionVAE trajectory must keep all 24 slots valid")
        if not np.all(arrays["physical_valid_count"] == self.temporal.physical_valid_count):
            raise ValueError("Bridge physical_valid_count must be six")
        float_fields = (
            "z_effect_raw",
            "z_effect",
            "z_action_raw",
            "z_action",
            "z_action_logvar_raw",
            "z_action_logvar",
            "teacher_trajectory",
            "teacher_current_state",
        )
        if any(not np.isfinite(arrays[name]).all() for name in float_fields):
            raise ValueError("Bridge paired teacher cache contains NaN/Inf")

        manifest_ids = self.frame["sample_id"].astype(str).to_numpy()
        if len(set(split_ids.tolist())) != rows:
            raise ValueError("Bridge paired cache contains duplicate split sample IDs")
        if set(split_ids.tolist()) != set(manifest_ids.tolist()):
            raise ValueError("Bridge Stage-2 manifest/cache sample IDs do not exact-set join")
        lookup = {sample_id: index for index, sample_id in enumerate(split_ids.tolist())}
        order = np.asarray([lookup[value] for value in manifest_ids.tolist()], dtype=np.int64)
        self.sample_ids = manifest_ids.astype("U64")
        for name, value in arrays.items():
            setattr(self, name, np.ascontiguousarray(value[order]))
        self.provenance = {
            "schema_version": self.temporal.cache_schema_version,
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
        return self.frame.iloc[int(index) % len(self)].to_dict()

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
            raise ValueError("Bridge policy row identity changed before teacher attachment")
        expected_language = str(row["language_instruction"]).strip()
        if str(sample.get("lang", "")).strip() != expected_language:
            raise ValueError("Bridge policy language differs from the exact Stage-2 manifest row")
        sample["sample_id"] = str(row["sample_id"])
        for name in ("z_effect_raw", "z_effect", "z_action", "z_action_logvar"):
            sample[name] = torch.from_numpy(getattr(self, name)[normalized].copy())
        for name in (
            "teacher_trajectory",
            "teacher_current_state",
        ):
            sample[name] = torch.from_numpy(getattr(self, name)[normalized].copy())
        for name in (
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        ):
            sample[name] = torch.from_numpy(getattr(self, name)[normalized].copy()).bool()
        predictor_state = torch.zeros(32, dtype=torch.float32)
        predictor_state[:14] = sample["teacher_current_state"]
        sample["action_predictor_state"] = predictor_state
        sample["physical_valid_count"] = int(self.physical_valid_count[normalized])
        return sample


__all__ = ["BridgeStage2RowIndex", "REQUIRED_MANIFEST_COLUMNS", "sha256_file"]
