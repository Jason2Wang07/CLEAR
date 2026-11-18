"""Audit the frozen Bridge 1.2-second ActionVAE representation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


SCHEMA_VERSION = "bridge-z-action-sidecar-v1-12s-resampled24"
NORMALIZATION_VERSION = "bridge-z-action-normalization-v1-12s-resampled24"
HORIZON = 24
PHYSICAL_STEPS = 6
SOURCE_HZ = 5.0
REPRESENTATION_HZ = 20.0
TRAJECTORY_DIM = 7
NATIVE_DIM = 7
NATIVE_INDICES = np.asarray([3, 7, 11, 15, 19, 23], dtype=np.int64)
ANCHOR_INDICES = np.asarray([5, 11, 17, 23], dtype=np.int64)
OFFICIAL_SPLIT_COUNTS = {"train": 640_479, "validation": 80_012, "test": 80_126}


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _expected_native_mask(rows: int) -> np.ndarray:
    expected = np.zeros((rows, HORIZON, 8), dtype=np.uint8)
    expected[:, NATIVE_INDICES, :NATIVE_DIM] = 1
    return expected


def _rotation_slerp_error(waypoints: np.ndarray) -> float:
    maximum = 0.0
    fractions = np.asarray([0.25, 0.50, 0.75, 1.0], dtype=np.float64)
    for row in np.asarray(waypoints, dtype=np.float64):
        rotations = Rotation.from_rotvec(row[:, 3:6])
        identity = Rotation.identity()
        for physical_step in range(PHYSICAL_STEPS):
            stop = 4 * (physical_step + 1)
            start_rotation = identity if physical_step == 0 else rotations[stop - 5]
            end_rotation = rotations[stop - 1]
            expected = Slerp(
                [0.0, 1.0],
                Rotation.from_quat(
                    np.stack((start_rotation.as_quat(), end_rotation.as_quat()))
                ),
            )(fractions)
            actual = rotations[stop - 4 : stop]
            error = (expected.inv() * actual).magnitude()
            maximum = max(maximum, float(np.max(error)))
    return maximum


def validate_trajectory_arrays(
    *,
    waypoints: np.ndarray,
    current_state: np.ndarray,
    atol: float = 2.0e-5,
) -> dict[str, float]:
    """Validate 6x4 interpolation without treating interpolated slots as actions."""

    trajectory = np.asarray(waypoints, dtype=np.float64)
    state = np.asarray(current_state, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (HORIZON, TRAJECTORY_DIM):
        raise ValueError(f"waypoints must be [B,{HORIZON},{TRAJECTORY_DIM}]")
    if state.shape != (len(trajectory), 14):
        raise ValueError("current_state must be [B,14]")
    if not np.isfinite(trajectory).all() or not np.isfinite(state).all():
        raise ValueError("Bridge trajectory/state contains NaN or Inf")

    groups = trajectory.reshape(len(trajectory), PHYSICAL_STEPS, 4, TRAJECTORY_DIM)
    position_start = np.zeros((len(trajectory), PHYSICAL_STEPS, 3), dtype=np.float64)
    position_start[:, 1:] = groups[:, :-1, 3, :3]
    position_end = groups[:, :, 3, :3]
    fractions = np.asarray([0.25, 0.50, 0.75, 1.0], dtype=np.float64)
    expected_position = position_start[:, :, None, :] + fractions[None, None, :, None] * (
        position_end - position_start
    )[:, :, None, :]
    position_error = float(np.max(np.abs(groups[..., :3] - expected_position)))

    gripper_start = np.empty((len(trajectory), PHYSICAL_STEPS), dtype=np.float64)
    gripper_start[:, 0] = state[:, 6]
    gripper_start[:, 1:] = groups[:, :-1, 3, 6]
    gripper_error = float(
        np.max(np.abs(groups[:, :, :3, 6] - gripper_start[:, :, None]))
    )
    rotation_error = _rotation_slerp_error(trajectory)
    if position_error > atol:
        raise ValueError(f"Bridge position interpolation error {position_error} exceeds {atol}")
    if gripper_error > atol:
        raise ValueError(f"Bridge gripper ZOH error {gripper_error} exceeds {atol}")
    if rotation_error > atol:
        raise ValueError(f"Bridge rotation SLERP error {rotation_error} exceeds {atol}")
    return {
        "position_linear_max_abs": position_error,
        "rotation_slerp_max_rad": rotation_error,
        "gripper_zoh_max_abs_m": gripper_error,
    }


def _schema_issues(schema: Mapping[str, Any]) -> list[str]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "duration_seconds": 1.2,
        "horizon_actions": HORIZON,
        "representation_steps": HORIZON,
        "representation_hz": REPRESENTATION_HZ,
        "physical_action_steps": PHYSICAL_STEPS,
        "source_control_hz": SOURCE_HZ,
    }
    issues = [
        f"schema:{name}:{schema.get(name)!r}!={value!r}"
        for name, value in expected.items()
        if schema.get(name) != value
    ]
    interpolation = schema.get("waypoint", {}).get("interpolation", {})
    expected_interpolation = {
        "position": "linear",
        "rotation": "shortest-arc quaternion SLERP",
        "gripper": "source-start ZOH for phases 1-3; source endpoint at phase 4",
    }
    for name, value in expected_interpolation.items():
        if interpolation.get(name) != value:
            issues.append(f"schema:interpolation:{name}")
    if schema.get("native_actions", {}).get("valid_representation_indices_zero_based") != (
        NATIVE_INDICES.tolist()
    ):
        issues.append("schema:native_action_indices")
    if schema.get("physical_valid_count", {}).get("value") != PHYSICAL_STEPS:
        issues.append("schema:physical_valid_count")
    return issues


def _manifest_issues(
    root: Path,
    manifest: pd.DataFrame,
    *,
    require_e39: bool,
    official: bool,
) -> list[str]:
    issues: list[str] = []
    required = {
        "sample_id",
        "split",
        "start_step",
        "end_step",
        "control_hz",
        "shard_path",
        "shard_row",
        "e39_valid",
        "physical_valid_count",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        return [f"manifest:missing:{name}" for name in missing]
    if manifest.sample_id.astype(str).duplicated().any():
        issues.append("manifest:duplicate_sample_id")
    if not np.all((manifest.end_step - manifest.start_step).to_numpy() == PHYSICAL_STEPS):
        issues.append("manifest:physical_window_not_6")
    if not np.allclose(manifest.control_hz.to_numpy(dtype=np.float64), SOURCE_HZ):
        issues.append("manifest:source_control_hz_not_5")
    if not np.all(manifest.physical_valid_count.to_numpy(dtype=np.int64) == PHYSICAL_STEPS):
        issues.append("manifest:physical_valid_count_not_6")
    if require_e39 and not manifest.e39_valid.astype(bool).all():
        issues.append("manifest:e39_incomplete")
    if official:
        counts = {
            str(name): int(count)
            for name, count in manifest.split.astype(str).value_counts().items()
        }
        if counts != OFFICIAL_SPLIT_COUNTS:
            issues.append(f"manifest:official_split_counts:{counts}")
    for split in OFFICIAL_SPLIT_COUNTS:
        split_path = root / "manifests" / f"{split}.parquet"
        if not split_path.is_file():
            issues.append(f"manifest:missing_split:{split}")
            continue
        actual = pd.read_parquet(split_path, columns=["sample_id"])["sample_id"].astype(str)
        expected = manifest.loc[manifest.split.astype(str) == split, "sample_id"].astype(str)
        if actual.tolist() != expected.tolist():
            issues.append(f"manifest:split_partition:{split}")
    return issues


def _verify_checksums(root: Path) -> list[str]:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        return ["checksums:missing"]
    issues: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            issues.append(f"checksums:missing:{relative}")
        elif _sha256_file(path) != expected:
            issues.append(f"checksums:mismatch:{relative}")
    return issues


def audit_bridge_action_release(
    data_root: str | Path,
    *,
    require_e39: bool = False,
    official: bool = False,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    required = [
        root / "_SUCCESS",
        root / "schema.json",
        root / "provenance.json",
        root / "stats/normalization.json",
        root / "manifests/all.parquet",
    ]
    issues = [f"missing:{path}" for path in required if not path.is_file()]
    if issues:
        return {"schema": "bridge-actionvae-contract-audit-v1", "valid": False, "issues": issues}

    schema = _load_json(root / "schema.json")
    normalization = _load_json(root / "stats/normalization.json")
    manifest = pd.read_parquet(root / "manifests/all.parquet")
    issues.extend(_schema_issues(schema))
    if normalization.get("schema_version") != NORMALIZATION_VERSION:
        issues.append("normalization:schema")
    if normalization.get("fit_split") != "train":
        issues.append("normalization:not_train_only")
    train_rows = int((manifest.split.astype(str) == "train").sum())
    if int(normalization.get("sample_count", -1)) != train_rows:
        issues.append("normalization:sample_count")
    issues.extend(
        _manifest_issues(root, manifest, require_e39=require_e39, official=official)
    )

    expected_native = _expected_native_mask(1)[0]
    geometry_samples: list[np.ndarray] = []
    geometry_states: list[np.ndarray] = []
    checked_rows = 0
    for relative, rows in manifest.groupby("shard_path", sort=True):
        path = root / str(relative)
        if not path.is_file():
            issues.append(f"shard:missing:{relative}")
            continue
        with h5py.File(path, "r") as handle:
            indices = rows.shard_row.to_numpy(dtype=np.int64)
            stored_ids = np.asarray(handle["sample_id"])[indices].astype(str)
            if stored_ids.tolist() != rows.sample_id.astype(str).tolist():
                issues.append(f"shard:identity:{relative}")
                continue
            checked_rows += len(indices)
            for name in ("time_mask", "pose_mask", "gripper_mask"):
                if not np.all(np.asarray(handle[name])[indices] == 1):
                    issues.append(f"shard:{name}:{relative}")
            physical = np.asarray(handle["physical_valid_count"])[indices]
            if not np.all(physical == PHYSICAL_STEPS):
                issues.append(f"shard:physical_valid_count:{relative}")
            native_dim = np.asarray(handle["native_action_dim"])[indices]
            if not np.all(native_dim == NATIVE_DIM):
                issues.append(f"shard:native_action_dim:{relative}")
            native_mask = np.asarray(handle["native_action_mask"])[indices]
            if not np.array_equal(native_mask, np.broadcast_to(expected_native, native_mask.shape)):
                issues.append(f"shard:native_action_mask:{relative}")
            native = np.asarray(handle["native_actions"])[indices]
            if not np.all(native[native_mask == 0] == 0):
                issues.append(f"shard:native_action_padding:{relative}")
            e39_valid = np.asarray(handle["e39_valid"])[indices]
            if require_e39 and not np.all(e39_valid == 1):
                issues.append(f"shard:e39_valid:{relative}")
            e39 = np.asarray(handle["e39_mean"])[indices]
            if np.any((e39_valid == 1)[:, None] & ~np.isfinite(e39)):
                issues.append(f"shard:e39_nonfinite:{relative}")
            waypoints = np.asarray(handle["waypoints"])[indices]
            states = np.asarray(handle["current_state"])[indices]
            sample_indices = np.unique(np.asarray([0, len(indices) // 2, len(indices) - 1]))
            geometry_samples.append(waypoints[sample_indices])
            geometry_states.append(states[sample_indices])
            anchors = np.asarray(handle["physical_anchors"])[indices, :, :TRAJECTORY_DIM]
            if not np.allclose(anchors, waypoints[:, ANCHOR_INDICES], atol=1.0e-6, rtol=0.0):
                issues.append(f"shard:physical_anchors:{relative}")

    interpolation: dict[str, float] = {}
    if geometry_samples:
        try:
            interpolation = validate_trajectory_arrays(
                waypoints=np.concatenate(geometry_samples),
                current_state=np.concatenate(geometry_states),
            )
        except ValueError as exc:
            issues.append(f"interpolation:{exc}")
    if checked_rows != len(manifest):
        issues.append(f"shard:coverage:{checked_rows}!={len(manifest)}")
    if verify_checksums:
        issues.extend(_verify_checksums(root))
    return {
        "schema": "bridge-actionvae-contract-audit-v1",
        "valid": not issues,
        "root": str(root),
        "rows": len(manifest),
        "split_counts": {
            str(name): int(count)
            for name, count in manifest.split.astype(str).value_counts().sort_index().items()
        },
        "require_e39": bool(require_e39),
        "official": bool(official),
        "checksums_verified": bool(verify_checksums),
        "representation": {
            "source_hz": SOURCE_HZ,
            "physical_steps": PHYSICAL_STEPS,
            "representation_hz": REPRESENTATION_HZ,
            "representation_steps": HORIZON,
            "duration_seconds": HORIZON / REPRESENTATION_HZ,
            "native_indices": NATIVE_INDICES.tolist(),
            "interpolation": interpolation,
        },
        "issues": issues[:200],
        "issue_count": len(issues),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--require-e39", action="store_true")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_bridge_action_release(
        args.data_root,
        require_e39=args.require_e39,
        official=args.official,
        verify_checksums=args.verify_checksums,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
