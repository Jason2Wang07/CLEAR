"""Audit immutable RoboTwin ActionAE A0/A1 releases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .schema import (
    anchor_steps_for_horizon,
    condition_field_names,
    fields_for_horizon,
    versions_for_horizon,
)


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


def _checksum_issues(root: Path) -> list[str]:
    path = root / "checksums.sha256"
    if not path.is_file():
        return ["missing:checksums.sha256"]
    issues = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file():
            issues.append(f"checksum_missing:{relative}")
        elif _sha256_file(target) != expected:
            issues.append(f"checksum_mismatch:{relative}")
    return issues


def _canonical_endpoints(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 16 or last.shape != first.shape:
        raise ValueError(f"dual-arm endpoints must both be [N,16], got {first.shape}/{last.shape}")
    result = np.empty((len(first), 14), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (8, 7)):
        start_arm = first[:, source_offset : source_offset + 8]
        end_arm = last[:, source_offset : source_offset + 8]
        start_quaternion = start_arm[:, [4, 5, 6, 3]]
        end_quaternion = end_arm[:, [4, 5, 6, 3]]
        start_quaternion /= np.linalg.norm(start_quaternion, axis=1, keepdims=True)
        end_quaternion /= np.linalg.norm(end_quaternion, axis=1, keepdims=True)
        start_rotation = Rotation.from_quat(start_quaternion)
        end_rotation = Rotation.from_quat(end_quaternion)
        result[:, target_offset : target_offset + 3] = start_rotation.inv().apply(
            end_arm[:, :3] - start_arm[:, :3]
        )
        result[:, target_offset + 3 : target_offset + 6] = (
            start_rotation.inv() * end_rotation
        ).as_rotvec()
        result[:, target_offset + 6] = end_arm[:, 7]
    return result


def audit_release(
    root: str | Path,
    *,
    effect_root: str | Path | None = None,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    release = Path(root).expanduser().resolve()
    effect = Path(effect_root).expanduser().resolve() if effect_root is not None else None
    issues: list[str] = []
    required = (
        "_SUCCESS", "schema.json", "provenance.json", "coverage_report.json",
        "stats/normalization.json", "manifests/all.parquet",
    )
    for relative in required:
        if not (release / relative).is_file():
            issues.append(f"missing:{relative}")
    if issues:
        return {"valid": False, "issues": issues, "issue_count": len(issues), "rows": 0}
    if verify_checksums:
        issues.extend(_checksum_issues(release))
    schema = json.loads((release / "schema.json").read_text(encoding="utf-8"))
    horizon = int(schema.get("horizon_actions", 0))
    condition_source = str(schema.get("condition", {}).get("source", "e39"))
    try:
        mean_field, valid_field = condition_field_names(condition_source)
        schema_version, normalization_schema_version, _ = versions_for_horizon(
            horizon, condition_source=condition_source
        )
        fields = fields_for_horizon(horizon, condition_source=condition_source)
        anchor_steps = anchor_steps_for_horizon(horizon)
    except ValueError as exc:
        issues.append(f"horizon:{exc}")
        schema_version = str(schema.get("schema_version", ""))
        normalization_schema_version = ""
        fields = {}
        anchor_steps = ()
        mean_field, valid_field = "e39_mean", "e39_valid"
    if schema.get("schema_version") != schema_version:
        issues.append(f"schema_version:{schema.get('schema_version')}")
    condition_schema = schema.get("condition")
    if condition_source == "lawam_idm":
        if not isinstance(condition_schema, dict):
            issues.append("missing_condition_schema")
        else:
            if condition_schema.get("mean_field") != mean_field:
                issues.append("condition_mean_field")
            if condition_schema.get("valid_field") != valid_field:
                issues.append("condition_valid_field")
            if condition_schema.get("sampling") is not False:
                issues.append("condition_sampling")
    normalization = json.loads((release / "stats/normalization.json").read_text(encoding="utf-8"))
    if normalization.get("schema_version") != normalization_schema_version:
        issues.append(f"normalization_schema:{normalization.get('schema_version')}")
    if normalization.get("fit_split") != "train":
        issues.append("normalization_not_train")
    for name in (
        "left_position", "left_rotation_rotvec", "left_gripper", "right_position",
        "right_rotation_rotvec", "right_gripper", "current_state",
    ):
        scale = np.asarray(normalization.get(name, {}).get("scale", []), dtype=np.float64)
        if not scale.size or not np.isfinite(scale).all() or np.any(scale <= 0):
            issues.append(f"normalization_invalid:{name}")

    manifest = pd.read_parquet(release / "manifests/all.parquet")
    if manifest.sample_id.duplicated().any():
        issues.append("manifest_duplicate_sample_id")
    if effect is not None:
        effect_manifest = pd.read_parquet(effect / "manifests/all.parquet")
        if set(manifest.sample_id.astype(str)) != set(effect_manifest.sample_id.astype(str)):
            issues.append("effect_exact_set_mismatch")
        if len(manifest) != len(effect_manifest):
            issues.append("effect_row_count_mismatch")
    split_counts = Counter(manifest.split.astype(str))
    episode_splits = manifest.groupby("episode_id").split.nunique()
    if len(episode_splits) and int(episode_splits.max()) != 1:
        issues.append("episode_split_leakage")

    condition_valid_count = 0
    endpoint_max_error = 0.0
    visited = 0
    for relative, group in manifest.groupby("shard_path", sort=True):
        path = release / str(relative)
        if not path.is_file():
            issues.append(f"missing_shard:{relative}")
            continue
        with h5py.File(path, "r") as handle:
            if _decode(handle.attrs.get("schema_version", "")) != schema_version or not handle.attrs.get("complete"):
                issues.append(f"invalid_shard:{relative}")
                continue
            rows = int(handle["sample_id"].shape[0])
            for name, (_, shape) in fields.items():
                if name not in handle or tuple(handle[name].shape) != (rows, *shape):
                    issues.append(f"field_shape:{relative}:{name}")
            shard_rows = group["shard_row"].astype(np.int64).to_numpy()
            if np.any(shard_rows < 0) or np.any(shard_rows >= rows):
                issues.append(f"shard_row:{relative}")
                continue
            seen = np.zeros(rows, dtype=bool)
            seen[shard_rows] = True
            stored_sample_ids = np.asarray([_decode(value) for value in handle["sample_id"][...]], dtype=str)
            stored_splits = np.asarray([_decode(value) for value in handle["split"][...]], dtype=str)
            stored_checksums = np.asarray([_decode(value) for value in handle["window_sha256"][...]], dtype=str)
            expected_sample_ids = group["sample_id"].astype(str).to_numpy()
            if not np.array_equal(stored_sample_ids[shard_rows], expected_sample_ids):
                issues.append(f"sample_identity:{relative}")
            if not np.array_equal(stored_splits[shard_rows], group["split"].astype(str).to_numpy()):
                issues.append(f"split_identity:{relative}")
            if not np.array_equal(
                stored_checksums[shard_rows], group["window_sha256"].astype(str).to_numpy()
            ):
                issues.append(f"window_identity:{relative}")

            arrays = {
                name: handle[name][...]
                for name in (
                    "dual_arm_pose_start", "dual_arm_pose_end", "waypoints", "time_mask",
                    "arm_pose_mask", "arm_gripper_mask", "native_actions", "native_action_mask",
                    "current_state", "current_state_mask", mean_field, valid_field,
                    "physical_anchors", "physical_anchor_mask",
                )
            }
            for field in ("waypoints", "native_actions", "current_state", mean_field):
                if not np.isfinite(arrays[field]).all():
                    issues.append(f"nonfinite:{relative}:{field}")
            for field in (
                "time_mask", "arm_pose_mask", "arm_gripper_mask", "native_action_mask",
                "current_state_mask", "physical_anchor_mask", valid_field,
            ):
                if not np.isin(arrays[field], (0, 1)).all():
                    issues.append(f"nonbinary:{relative}:{field}")
            if not np.all(arrays["time_mask"] == 1):
                issues.append(f"time_mask:{relative}")
            condition_values = arrays[valid_field].astype(np.int64)
            condition_valid_count += int(condition_values.sum())
            if not np.array_equal(
                condition_values[shard_rows].astype(bool), group[valid_field].astype(bool).to_numpy()
            ):
                issues.append(f"condition_manifest:{relative}")
            expected_endpoints = _canonical_endpoints(
                arrays["dual_arm_pose_start"], arrays["dual_arm_pose_end"]
            )
            endpoint_errors = np.max(np.abs(expected_endpoints - arrays["waypoints"][:, -1]), axis=1)
            endpoint_max_error = max(endpoint_max_error, float(endpoint_errors.max(initial=0.0)))
            if np.any(endpoint_errors > 2.0e-6):
                issues.append(f"endpoint:{relative}:{float(endpoint_errors.max())}")
            anchor_indices = np.asarray(anchor_steps, dtype=np.int64) - 1
            if not np.array_equal(arrays["physical_anchors"], arrays["waypoints"][:, anchor_indices]):
                issues.append(f"anchors:{relative}")
            visited += len(group)
            if not seen.all():
                issues.append(f"unreferenced_shard_rows:{relative}:{int((~seen).sum())}")

    coverage = json.loads((release / "coverage_report.json").read_text(encoding="utf-8"))
    if visited != len(manifest):
        issues.append(f"visited_rows:{visited}:{len(manifest)}")
    if int(coverage.get("condition_valid_rows", coverage.get("e39_valid_rows", -1))) != condition_valid_count:
        issues.append("coverage_condition_count")
    if condition_valid_count not in (0, len(manifest)):
        issues.append("partial_condition_coverage")
    provenance = json.loads((release / "provenance.json").read_text(encoding="utf-8"))
    condition_provenance = provenance.get("condition", provenance.get(condition_source))
    if condition_valid_count == len(manifest) and not isinstance(condition_provenance, dict):
        issues.append("missing_condition_provenance")
    if condition_valid_count == 0 and condition_provenance is not None:
        issues.append("unexpected_condition_provenance")
    if condition_source == "lawam_idm" and isinstance(condition_provenance, dict):
        expected_values = {
            "source": "lawam_idm",
            "teacher": "official_lawam_release_idm",
            "posterior_statistic": "mean",
            "sampling": False,
        }
        for name, expected in expected_values.items():
            if condition_provenance.get(name) != expected:
                issues.append(f"condition_provenance:{name}")
        for name in ("checkpoint_sha256", "model_config_sha256"):
            digest = condition_provenance.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                issues.append(f"condition_provenance:{name}")
    report = {
        "schema": "robotwin-action-audit-v1",
        "valid": not issues,
        "issues": issues[:200],
        "issue_count": len(issues),
        "rows": len(manifest),
        "split_counts": dict(sorted(split_counts.items())),
        "condition_source": condition_source,
        "condition_valid_rows": condition_valid_count,
        "episode_split_disjoint": "episode_split_leakage" not in issues,
        "effect_exact_set": (
            "effect_exact_set_mismatch" not in issues if effect is not None else None
        ),
        "endpoint_max_abs_error": endpoint_max_error,
    }
    if condition_source == "e39":
        report["e39_valid_rows"] = condition_valid_count
    else:
        report["lawam_valid_rows"] = condition_valid_count
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--effect-root", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_release(args.root, effect_root=args.effect_root, verify_checksums=not args.skip_checksums)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
