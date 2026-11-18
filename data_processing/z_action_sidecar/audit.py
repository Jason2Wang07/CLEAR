"""Integrity audit for a published z-action sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy.spatial.transform import Rotation

from .schema import ANCHOR_STEPS, FIELDS, HORIZON, MANIFEST_FIELDS, SCHEMA_VERSION


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _check_published_checksums(root: Path) -> list[str]:
    path = root / "checksums.sha256"
    if not path.is_file():
        return ["missing:checksums.sha256"]
    issues: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            issues.append(f"checksums:{line_number}:syntax")
            continue
        target = root / relative
        if not target.is_file():
            issues.append(f"checksums:{line_number}:missing:{relative}")
        elif _sha256_file(target) != expected:
            issues.append(f"checksums:{line_number}:mismatch:{relative}")
    return issues


def _endpoint_from_canonical(start_pose: np.ndarray, waypoint: np.ndarray) -> np.ndarray:
    start = np.asarray(start_pose, dtype=np.float64)
    point = np.asarray(waypoint, dtype=np.float64)
    r0 = Rotation.from_quat(start[[4, 5, 6, 3]])
    position = start[:3] + r0.apply(point[:3])
    rotation = r0 * Rotation.from_rotvec(point[3:6])
    xyzw = rotation.as_quat()
    return np.concatenate((position, xyzw[[3, 0, 1, 2]]))


def _pose_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    ra = Rotation.from_quat(a[[4, 5, 6, 3]])
    rb = Rotation.from_quat(b[[4, 5, 6, 3]])
    return float(np.linalg.norm(a[:3] - b[:3])), float((ra.inv() * rb).magnitude())


def _manifest_schema_issues(schema: Mapping[str, Any]) -> list[str]:
    manifest_schema = schema.get("manifest")
    if manifest_schema is None:
        # The first immutable v1 publication predates the embedded manifest
        # declaration.  Its rows are still checked against the frozen contract
        # in this module.
        return []
    if not isinstance(manifest_schema, Mapping):
        return ["schema:manifest:not_object"]
    required = manifest_schema.get("required_fields")
    if not isinstance(required, Mapping):
        return ["schema:manifest:required_fields:not_object"]

    issues: list[str] = []
    for name, logical_dtype in MANIFEST_FIELDS.items():
        documented = required.get(name)
        if documented != logical_dtype:
            issues.append(f"schema:manifest:{name}:{documented!r}!={logical_dtype!r}")
    for name in sorted(set(required).difference(MANIFEST_FIELDS)):
        issues.append(f"schema:manifest:unexpected_required_field:{name}")
    if manifest_schema.get("additional_fields") is not True:
        issues.append("schema:manifest:additional_fields:not_allowed")
    return issues


def _matches_logical_dtype(series: pd.Series, logical_dtype: str) -> bool:
    dtype = series.dtype
    if logical_dtype == "string":
        if is_object_dtype(dtype):
            return bool(series.dropna().map(lambda value: isinstance(value, str)).all())
        return bool(is_string_dtype(dtype))
    if logical_dtype == "integer":
        return bool(is_integer_dtype(dtype) and not is_bool_dtype(dtype))
    if logical_dtype == "number":
        return bool(is_numeric_dtype(dtype) and not is_bool_dtype(dtype))
    if logical_dtype == "boolean":
        return bool(is_bool_dtype(dtype))
    raise ValueError(f"unsupported manifest logical dtype: {logical_dtype}")


def _manifest_contract_issues(manifest: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for name, logical_dtype in MANIFEST_FIELDS.items():
        if name not in manifest.columns:
            issues.append(f"manifest:missing_field:{name}")
            continue
        series = manifest[name]
        if not _matches_logical_dtype(series, logical_dtype):
            issues.append(f"manifest:dtype:{name}:{series.dtype}!={logical_dtype}")
        null_count = int(series.isna().sum())
        if null_count:
            issues.append(f"manifest:null:{name}:{null_count}")
    return issues


def _manifest_split_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    episode_leakage: list[str] = []
    group_leakage: list[str] = []
    episode_count = group_count = 0
    if {"source_id", "episode_id", "split"}.issubset(manifest.columns):
        episode_splits = manifest.groupby(["source_id", "episode_id"], dropna=False)["split"].nunique(
            dropna=False
        )
        episode_count = len(episode_splits)
        episode_leakage = sorted(
            f"{source}:{episode}"
            for (source, episode), split_count in episode_splits.items()
            if split_count > 1
        )
    if {"group_id", "split"}.issubset(manifest.columns):
        group_splits = manifest.groupby("group_id", dropna=False)["split"].nunique(dropna=False)
        group_count = len(group_splits)
        group_leakage = sorted(str(group_id) for group_id, split_count in group_splits.items() if split_count > 1)
    return {
        "episodes": int(episode_count),
        "episode_split_leakage": episode_leakage,
        "groups": int(group_count),
        "group_split_leakage": group_leakage,
    }


def audit_sidecar(root: Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    issues: list[str] = []
    for relative in ("_SUCCESS", "schema.json", "coverage_report.json", "provenance.json", "stats/normalization.json", "manifests/all.parquet"):
        if not (root / relative).is_file():
            issues.append(f"missing:{relative}")
    if issues:
        return {"schema_version": SCHEMA_VERSION, "valid": False, "issues": issues, "rows": 0}

    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    if schema.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version:{schema.get('schema_version')}")
    normalization = json.loads((root / "stats/normalization.json").read_text(encoding="utf-8"))
    if normalization.get("fit_split") != "train":
        issues.append(f"normalization_fit_split:{normalization.get('fit_split')}")
    if verify_checksums:
        issues.extend(_check_published_checksums(root))

    manifest = pd.read_parquet(root / "manifests/all.parquet")
    contract_issues = _manifest_schema_issues(schema) + _manifest_contract_issues(manifest)
    issues.extend(contract_issues)
    split_audit = _manifest_split_audit(manifest)
    episode_leakage = split_audit["episode_split_leakage"]
    group_leakage = split_audit["group_split_leakage"]
    if "sample_id" in manifest and manifest["sample_id"].duplicated().any():
        issues.append("manifest:duplicate_sample_id")
    if "split" in manifest and set(manifest["split"].astype(str)).difference({"train", "validation", "test"}):
        issues.append("manifest:unknown_split")
    if episode_leakage:
        issues.append(f"manifest:episode_leakage:{len(episode_leakage)}")
    if group_leakage:
        issues.append(f"manifest:group_split_leakage:{len(group_leakage)}")

    split_counts = (
        manifest["split"].astype(str).value_counts().sort_index().to_dict()
        if "split" in manifest
        else {}
    )
    sources = int(manifest["source_id"].nunique()) if "source_id" in manifest else 0
    if contract_issues:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "rows": len(manifest),
            "seen_rows": 0,
            "sources": sources,
            "episodes": split_audit["episodes"],
            "groups": split_audit["groups"],
            "split_counts": split_counts,
            "episode_leakage_count": len(episode_leakage),
            "episode_split_leakage": episode_leakage[:32],
            "group_leakage_count": len(group_leakage),
            "group_split_leakage": group_leakage[:32],
            "e39_valid_rows": 0,
            "gripper_valid_steps": 0,
            "contact_valid_steps": 0,
            "endpoint_position_error_max_m": 0.0,
            "endpoint_rotation_error_max_rad": 0.0,
            "issues": issues[:200],
            "issue_count": len(issues),
        }

    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    seen: set[str] = set()
    endpoint_position_max = endpoint_rotation_max = 0.0
    e39_valid = gripper_valid = contact_valid = 0
    for relative_path, part in manifest.groupby("shard_path", sort=True):
        path = root / str(relative_path)
        if not path.is_file():
            issues.append(f"manifest:missing_shard:{relative_path}")
            continue
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != SCHEMA_VERSION or not handle.attrs.get("complete"):
                issues.append(f"shard:metadata:{relative_path}")
            missing = sorted(set(FIELDS).difference(handle.keys()))
            if missing:
                issues.append(f"shard:fields:{relative_path}:{missing}")
                continue
            length = len(handle["sample_id"])
            for name, (dtype, shape) in FIELDS.items():
                dataset = handle[name]
                if dataset.shape != (length, *shape):
                    issues.append(f"shard:shape:{relative_path}:{name}:{dataset.shape}")
                if dtype != "string" and dataset.dtype != np.dtype(dtype):
                    issues.append(f"shard:dtype:{relative_path}:{name}:{dataset.dtype}")
            for item in part.itertuples(index=False):
                row = int(item.shard_row)
                if row < 0 or row >= length:
                    issues.append(f"manifest:row_oob:{item.sample_id}")
                    continue
                sample_id = _decode(handle["sample_id"][row])
                if sample_id != str(item.sample_id):
                    issues.append(f"manifest:identity:{item.sample_id}:{sample_id}")
                    continue
                if sample_id in seen:
                    issues.append(f"shard:duplicate_sample_id:{sample_id}")
                seen.add(sample_id)
                if int(handle["end_step"][row]) - int(handle["start_step"][row]) != HORIZON:
                    issues.append(f"window:horizon:{sample_id}")

                arrays = {name: handle[name][row] for name in (
                    "waypoints", "time_mask", "pose_mask", "gripper_mask", "contact",
                    "contact_peak_force", "contact_impulse", "contact_mask", "grasp", "grasp_mask",
                    "native_actions", "native_action_mask", "current_state", "e39_mean",
                    "physical_anchors", "physical_anchor_mask", "replay_object_error", "replay_robot_error",
                )}
                numeric = [np.asarray(value) for value in arrays.values()]
                if any(not np.isfinite(value).all() for value in numeric):
                    issues.append(f"window:nonfinite:{sample_id}")
                for name in ("time_mask", "pose_mask", "gripper_mask", "contact_mask", "grasp_mask", "native_action_mask", "physical_anchor_mask"):
                    if not np.isin(arrays[name], (0, 1)).all():
                        issues.append(f"mask:not_binary:{sample_id}:{name}")
                time = arrays["time_mask"].astype(bool)
                if not time.all():
                    issues.append(f"mask:time_not_full:{sample_id}")
                if np.any(arrays["pose_mask"] > arrays["time_mask"]):
                    issues.append(f"mask:pose_without_time:{sample_id}")
                if np.any(arrays["gripper_mask"] > arrays["time_mask"]):
                    issues.append(f"mask:gripper_without_time:{sample_id}")
                if np.any(arrays["contact_mask"] > arrays["time_mask"]):
                    issues.append(f"mask:contact_without_time:{sample_id}")
                if np.any(arrays["grasp_mask"] > arrays["contact_mask"]):
                    issues.append(f"mask:grasp_without_contact_validity:{sample_id}")
                native_dim = int(handle["native_action_dim"][row])
                expected_native = np.zeros_like(arrays["native_action_mask"])
                expected_native[:, :native_dim] = 1
                if not np.array_equal(arrays["native_action_mask"], expected_native):
                    issues.append(f"mask:native:{sample_id}")
                indices = np.asarray(ANCHOR_STEPS) - 1
                expected_anchors = np.concatenate((arrays["waypoints"][indices], arrays["contact"][indices, None]), axis=1)
                if not np.allclose(arrays["physical_anchors"], expected_anchors, rtol=0.0, atol=1e-7):
                    issues.append(f"anchors:values:{sample_id}")

                reconstructed = _endpoint_from_canonical(handle["eef_pose_start"][row], arrays["waypoints"][-1])
                position_error, rotation_error = _pose_error(reconstructed, handle["eef_pose_end"][row])
                endpoint_position_max = max(endpoint_position_max, position_error)
                endpoint_rotation_max = max(endpoint_rotation_max, rotation_error)
                if position_error > 1e-5 or rotation_error > 1e-5:
                    issues.append(f"endpoint:canonical:{sample_id}:{position_error}:{rotation_error}")
                e39_valid += int(handle["e39_valid"][row])
                gripper_valid += int(np.asarray(arrays["gripper_mask"]).sum())
                contact_valid += int(np.asarray(arrays["contact_mask"]).sum())

    if len(seen) != len(manifest):
        issues.append(f"coverage:{len(seen)}:{len(manifest)}")
    coverage = json.loads((root / "coverage_report.json").read_text(encoding="utf-8"))
    if coverage.get("e39", {}).get("required") and e39_valid != len(manifest):
        issues.append(f"e39:coverage:{e39_valid}:{len(manifest)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not issues,
        "rows": len(manifest),
        "seen_rows": len(seen),
        "sources": sources,
        "episodes": split_audit["episodes"],
        "groups": split_audit["groups"],
        "split_counts": split_counts,
        "episode_leakage_count": len(episode_leakage),
        "episode_split_leakage": episode_leakage[:32],
        "group_leakage_count": len(group_leakage),
        "group_split_leakage": group_leakage[:32],
        "e39_valid_rows": e39_valid,
        "gripper_valid_steps": gripper_valid,
        "contact_valid_steps": contact_valid,
        "endpoint_position_error_max_m": endpoint_position_max,
        "endpoint_rotation_error_max_rad": endpoint_rotation_max,
        "issues": issues[:200],
        "issue_count": len(issues),
    }
