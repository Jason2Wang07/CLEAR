#!/usr/bin/env python3
"""Audit aligned LIBERO effect and ActionAE releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from libero_dataset.converter import _window_digest
from libero_dataset.schema import (
    versions_for_horizon,
)
from lawam_effect_dataset.schema import FIELDS as EFFECT_FIELDS
from z_action_sidecar.audit import _check_published_checksums, _endpoint_from_canonical, _pose_error
from z_action_sidecar.schema import fields_for_horizon


AUDIT_SCHEMA = "libero-actionvae-release-audit-v1"
IDENTITY_COLUMNS = (
    "sample_id",
    "group_id",
    "split",
    "task_id",
    "source_id",
    "episode_id",
    "start_step",
    "end_step",
)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _row_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _row_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return np.asarray(value)


def _manifest(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "manifests/all.parquet")
    return frame.sort_values("sample_id").reset_index(drop=True)


def _required_files(root: Path, *, action: bool) -> list[str]:
    common = [
        "_SUCCESS",
        "schema.json",
        "coverage_report.json",
        "provenance.json",
        "checksums.sha256",
        "manifests/all.parquet",
        "manifests/train.parquet",
        "manifests/validation.parquet",
        "manifests/test.parquet",
    ]
    common.append("stats/normalization.json" if action else "stats/state_normalization.json")
    return [name for name in common if not (root / name).is_file()]


def _audit_manifest(frame: pd.DataFrame, *, label: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    required = set(IDENTITY_COLUMNS) | {"shard_path", "shard_row"}
    missing = sorted(required.difference(frame.columns))
    issues.extend(f"{label}:manifest:missing:{name}" for name in missing)
    if missing:
        return issues, {}
    if frame.sample_id.astype(str).duplicated().any():
        issues.append(f"{label}:manifest:duplicate_sample_id")
    unknown = sorted(set(frame.split.astype(str)).difference({"train", "validation", "test"}))
    if unknown:
        issues.append(f"{label}:manifest:unknown_splits:{unknown}")
    episode_splits = frame.groupby(["source_id", "episode_id"])["split"].nunique()
    group_splits = frame.groupby("group_id")["split"].nunique()
    if int((episode_splits > 1).sum()):
        issues.append(f"{label}:manifest:episode_split_leakage:{int((episode_splits > 1).sum())}")
    if int((group_splits > 1).sum()):
        issues.append(f"{label}:manifest:group_split_leakage:{int((group_splits > 1).sum())}")
    task_split_rows = frame.groupby(["task_id", "split"]).size().unstack(fill_value=0)
    for split in ("train", "validation", "test"):
        if split not in task_split_rows or bool((task_split_rows[split] == 0).any()):
            issues.append(f"{label}:manifest:task_without_{split}")
    return issues, {
        "rows": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "episodes": int(frame.groupby(["source_id", "episode_id"]).ngroups),
        "groups": int(frame.group_id.nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in frame.split.astype(str).value_counts().sort_index().items()
        },
    }


def _audit_shards(
    root: Path,
    manifest: pd.DataFrame,
    *,
    fields: dict[str, Any],
    schema_version: str,
    action: bool,
    horizon: int,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    seen: set[str] = set()
    e39_valid = 0
    endpoint_position_max = 0.0
    endpoint_rotation_max = 0.0
    for relative, part in manifest.groupby("shard_path", sort=True):
        path = root / str(relative)
        if not path.is_file():
            issues.append(f"shard:missing:{relative}")
            continue
        with h5py.File(path, "r") as handle:
            if _decode(handle.attrs.get("schema_version", "")) != schema_version:
                issues.append(f"shard:schema:{relative}:{handle.attrs.get('schema_version')}")
            if not bool(handle.attrs.get("complete", False)):
                issues.append(f"shard:incomplete:{relative}")
            missing = sorted(set(fields).difference(handle.keys()))
            if missing:
                issues.append(f"shard:fields:{relative}:{missing}")
                continue
            row_count = len(handle["sample_id"])
            for name, spec in fields.items():
                dtype, shape = spec
                expected_shape = (row_count, *shape)
                if tuple(handle[name].shape) != expected_shape:
                    issues.append(f"shard:shape:{relative}:{name}:{handle[name].shape}")
                if dtype != "string" and handle[name].dtype != np.dtype(dtype):
                    issues.append(f"shard:dtype:{relative}:{name}:{handle[name].dtype}")
            required_values = (
                set(fields)
                if action
                else {"sample_id", "split", "start_step", "end_step", "rgb_start", "rgb_end", "effect_mask"}
            )
            values = {name: np.asarray(handle[name][...]) for name in required_values}
            for item in part.itertuples(index=False):
                row = int(item.shard_row)
                sample_id = str(item.sample_id)
                if row < 0 or row >= row_count:
                    issues.append(f"shard:row_oob:{sample_id}:{row}")
                    continue
                stored_id = _decode(values["sample_id"][row])
                stored_split = _decode(values["split"][row])
                if stored_id != sample_id or stored_split != str(item.split):
                    issues.append(f"shard:identity:{sample_id}:{stored_id}:{stored_split}")
                    continue
                if sample_id in seen:
                    issues.append(f"shard:duplicate_sample_id:{sample_id}")
                seen.add(sample_id)
                if int(values["end_step"][row]) - int(values["start_step"][row]) != horizon:
                    issues.append(f"window:horizon:{sample_id}")
                if action:
                    time = np.asarray(values["time_mask"][row], dtype=np.uint8)
                    pose = np.asarray(values["pose_mask"][row], dtype=np.uint8)
                    gripper = np.asarray(values["gripper_mask"][row], dtype=np.uint8)
                    if not time.all() or not pose.all() or not gripper.all():
                        issues.append(f"mask:trajectory:{sample_id}")
                    if np.asarray(values["contact_mask"][row]).any() or np.asarray(values["grasp_mask"][row]).any():
                        issues.append(f"mask:unavailable_contact_or_grasp:{sample_id}")
                    row_payload = {
                        name: _row_value(values[name][row])
                        for name in fields
                    }
                    expected_digest = _window_digest(row_payload)
                    stored_digest = _decode(values["window_sha256"][row])
                    if stored_digest != expected_digest or str(item.window_sha256) != stored_digest:
                        issues.append(f"window:sha256:{sample_id}")
                    reconstructed = _endpoint_from_canonical(
                        np.asarray(values["eef_pose_start"][row]),
                        np.asarray(values["waypoints"][row])[-1],
                    )
                    position_error, rotation_error = _pose_error(
                        reconstructed,
                        np.asarray(values["eef_pose_end"][row]),
                    )
                    endpoint_position_max = max(endpoint_position_max, position_error)
                    endpoint_rotation_max = max(endpoint_rotation_max, rotation_error)
                    if position_error > 1e-5 or rotation_error > 1e-5:
                        issues.append(f"window:endpoint:{sample_id}:{position_error}:{rotation_error}")
                    value = int(values["e39_valid"][row])
                    if value not in (0, 1):
                        issues.append(f"e39:nonbinary:{sample_id}:{value}")
                    if value and not np.isfinite(values["e39_mean"][row]).all():
                        issues.append(f"e39:nonfinite:{sample_id}")
                    e39_valid += value
                else:
                    if not np.asarray(values["rgb_start"][row]).any() or not np.asarray(values["rgb_end"][row]).any():
                        issues.append(f"rgb:zero:{sample_id}")
                    if not np.array_equal(
                        np.asarray(values["effect_mask"][row], dtype=np.uint8),
                        np.asarray([1] * 7 + [0] * 5, dtype=np.uint8),
                    ):
                        issues.append(f"effect:mask:{sample_id}")
    if len(seen) != len(manifest):
        issues.append(f"shard:coverage:{len(seen)}:{len(manifest)}")
    return issues, {
        "seen_rows": len(seen),
        "e39_valid_rows": e39_valid if action else None,
        "endpoint_position_error_max_m": endpoint_position_max if action else None,
        "endpoint_rotation_error_max_rad": endpoint_rotation_max if action else None,
    }


def audit_release(
    root: Path,
    *,
    effect_root: Path | None = None,
    full: bool = False,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    release = root.expanduser().resolve()
    if (release / "effect").is_dir() and (release / "action").is_dir():
        resolved_effect_root = release / "effect"
        action_root = release / "action"
    else:
        if effect_root is None:
            raise ValueError("an independent action release requires --effect-root")
        resolved_effect_root = effect_root.expanduser().resolve()
        action_root = release
    issues = [
        f"effect:missing:{name}"
        for name in _required_files(resolved_effect_root, action=False)
    ]
    issues.extend(f"action:missing:{name}" for name in _required_files(action_root, action=True))
    if issues:
        return {"schema": AUDIT_SCHEMA, "valid": False, "issues": issues, "root": str(release)}

    effect_schema = json.loads((resolved_effect_root / "schema.json").read_text(encoding="utf-8"))
    action_schema = json.loads((action_root / "schema.json").read_text(encoding="utf-8"))
    normalization = json.loads((action_root / "stats/normalization.json").read_text(encoding="utf-8"))
    state_normalization = json.loads(
        (resolved_effect_root / "stats/state_normalization.json").read_text(encoding="utf-8")
    )
    horizon = int(action_schema.get("horizon_actions", 0))
    try:
        effect_schema_version, sidecar_schema_version, normalization_schema_version = (
            versions_for_horizon(horizon)
        )
        action_fields = fields_for_horizon(horizon)
    except ValueError as exc:
        issues.append(f"action:horizon:{exc}")
        effect_schema_version = str(effect_schema.get("schema_version", ""))
        sidecar_schema_version = str(action_schema.get("schema_version", ""))
        normalization_schema_version = str(normalization.get("schema_version", ""))
        action_fields = {}
    if effect_schema.get("schema_version") != effect_schema_version:
        issues.append(f"effect:schema:{effect_schema.get('schema_version')}")
    if action_schema.get("schema_version") != sidecar_schema_version:
        issues.append(f"action:schema:{action_schema.get('schema_version')}")
    if normalization.get("schema_version") != normalization_schema_version:
        issues.append(f"action:normalization_schema:{normalization.get('schema_version')}")
    if normalization.get("fit_split") != "train" or state_normalization.get("fit_split") != "train":
        issues.append("normalization:not_train_only")
    if verify_checksums:
        issues.extend(
            f"effect:{item}" for item in _check_published_checksums(resolved_effect_root)
        )
        issues.extend(f"action:{item}" for item in _check_published_checksums(action_root))

    effect = _manifest(resolved_effect_root)
    action = _manifest(action_root)
    effect_issues, effect_summary = _audit_manifest(effect, label="effect")
    action_issues, action_summary = _audit_manifest(action, label="action")
    issues.extend(effect_issues)
    issues.extend(action_issues)
    if all(column in effect and column in action for column in IDENTITY_COLUMNS):
        for column in IDENTITY_COLUMNS:
            if not effect[column].astype(str).equals(action[column].astype(str)):
                issues.append(f"alignment:{column}")

    effect_shard_issues, effect_shards = _audit_shards(
        resolved_effect_root,
        effect,
        fields=EFFECT_FIELDS,
        schema_version=effect_schema_version,
        action=False,
        horizon=horizon,
    )
    action_shard_issues, action_shards = _audit_shards(
        action_root,
        action,
        fields=action_fields,
        schema_version=sidecar_schema_version,
        action=True,
        horizon=horizon,
    )
    issues.extend(f"effect:{item}" for item in effect_shard_issues)
    issues.extend(f"action:{item}" for item in action_shard_issues)

    if full:
        if effect_summary.get("tasks") != 130 or effect_summary.get("episodes") != 6500:
            issues.append(
                f"full:coverage:tasks={effect_summary.get('tasks')}:episodes={effect_summary.get('episodes')}"
            )
        episode_counts = (
            effect[["task_id", "source_id", "episode_id", "split"]]
            .drop_duplicates()
            .groupby(["task_id", "split"])
            .size()
            .unstack(fill_value=0)
        )
        expected = {"train": 40, "validation": 5, "test": 5}
        for split, count in expected.items():
            if split not in episode_counts or not bool((episode_counts[split] == count).all()):
                issues.append(f"full:task_episode_split:{split}:expected={count}")

    return {
        "schema": AUDIT_SCHEMA,
        "valid": not issues,
        "full_contract": bool(full),
        "root": str(release),
        "effect": {**effect_summary, **effect_shards},
        "action": {**action_summary, **action_shards},
        "issues": issues[:500],
        "issue_count": len(issues),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--effect-root", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_release(
        args.root,
        effect_root=args.effect_root,
        full=args.full,
        verify_checksums=not args.skip_checksums,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
