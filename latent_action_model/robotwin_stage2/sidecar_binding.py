"""Portable integrity gates for the dense RoboTwin ActionVAE sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SEMANTIC_BINDING_SCHEMA = "robotwin-dense-sidecar-semantic-binding-v1"
LAWAM_IDM_CONTRACT_BINDING_SCHEMA = (
    "robotwin-lawam-idm-dense-sidecar-contract-binding-v1"
)
LAWAM_IDM_SIDECAR_SCHEMA = "robotwin-z-action-sidecar-v2-12s-lawam-idm"
EXPECTED_SOURCE_DATASET = "jialei02/robotwin_merged"
EXPECTED_SOURCE_REVISION = "560f9e9fecc7dbf826afd6042b418d632e319df1"
EXPECTED_EFFECT_CHECKPOINT_SHA256 = (
    "66e872b5f6a67537b6cf33ad4d0b7e65652057e3e44a726f0a00c8205dd5c9a0"
)
EXPECTED_EFFECT_CONFIG_SHA256 = (
    "1a1094f79c93c8fb0c77b32db29fc221eaefce932c3da3cac4cb74ea25b08bae"
)
EXPECTED_EFFECT_SELECTION_SHA256 = (
    "4ad7a8fcc2e56cae12e93f4d1ddf7b625145c231726695c4cafd031bd8308d4b"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "255a793877156dc668fdb69b1e74f1bf2415a109ae287821f8f4b5f50e605896"
)
EXPECTED_SPLIT_ROWS = {
    "train": 4_065_338,
    "validation": 510_933,
    "test": 510_976,
}
EXPECTED_ROWS = sum(EXPECTED_SPLIT_ROWS.values())
REFERENCE_FILES = (
    "schema.json",
    "stats/normalization.json",
    "coverage_report.json",
    "recovery_audit.json",
    "provenance.json",
)
REQUIRED_CANDIDATE_FILES = (
    "_SUCCESS",
    "checksums.sha256",
    *REFERENCE_FILES,
    "manifests/all.parquet",
    "manifests/train.parquet",
    "manifests/validation.parquet",
    "manifests/test.parquet",
)


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _read_checksum_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        cells = line.split("  ", 1)
        if len(cells) != 2:
            raise ValueError(f"invalid checksum record at {path}:{line_number}")
        digest, relative = cells[0].lower(), cells[1]
        rel = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or rel.is_absolute()
            or ".." in rel.parts
            or str(rel) in {"", "."}
        ):
            raise ValueError(f"unsafe checksum record at {path}:{line_number}")
        normalized = str(rel)
        if normalized in records:
            raise ValueError(f"duplicate checksum path: {normalized}")
        records[normalized] = digest
    if not records:
        raise ValueError(f"checksum manifest is empty: {path}")
    return records


def verify_sidecar_checksum_inventory(sidecar_root: str | Path) -> dict[str, Any]:
    """Hash every candidate file and require a closed checksum inventory."""

    root = Path(sidecar_root).expanduser().resolve()
    checksums_path = root / "checksums.sha256"
    if not checksums_path.is_file():
        raise FileNotFoundError(checksums_path)
    records = _read_checksum_records(checksums_path)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != checksums_path
        and ".cache" not in path.relative_to(root).parts
    }
    if set(records) != actual_files:
        raise ValueError(
            "dense sidecar checksum inventory is not closed: "
            f"unlisted={sorted(actual_files.difference(records))[:8]}, "
            f"missing={sorted(set(records).difference(actual_files))[:8]}"
        )
    mismatches: dict[str, tuple[str, str]] = {}
    for relative, expected in records.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            mismatches[relative] = (expected, actual)
    if mismatches:
        raise ValueError(
            f"dense sidecar content checksum mismatch: {dict(list(mismatches.items())[:8])}"
        )
    return {
        "checksums_sha256": sha256_file(checksums_path),
        "files": len(records),
        "bytes": sum((root / relative).stat().st_size for relative in records),
    }


def verify_reference_bundle(reference_root: str | Path) -> dict[str, str]:
    root = Path(reference_root).expanduser().resolve()
    checksums_path = root / "checksums.sha256"
    success_path = root / "_SUCCESS"
    for relative in (*REFERENCE_FILES, "checksums.sha256", "_SUCCESS"):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    records = _read_checksum_records(checksums_path)
    if set(records) != set(REFERENCE_FILES):
        raise ValueError("semantic reference checksum file set mismatch")
    actual = {relative: sha256_file(root / relative) for relative in REFERENCE_FILES}
    if actual != records:
        raise ValueError("semantic reference bundle content checksum mismatch")
    if success_path.read_text(encoding="ascii").strip().lower() != sha256_file(checksums_path):
        raise ValueError("semantic reference _SUCCESS does not bind checksums.sha256")
    if actual["stats/normalization.json"] != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("semantic reference contains the wrong compatibility normalization")
    return actual


def _video_suffix(value: Any) -> str:
    text = str(value)
    marker = "/videos/"
    return text[text.index(marker) + 1 :] if marker in text else text


def _repair_projection(provenance: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    rows = provenance.get(name)
    if not isinstance(rows, list):
        raise ValueError(f"dense sidecar provenance.{name} must be a list")
    projected = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"dense sidecar provenance.{name} contains a non-object")
        item = {key: value for key, value in row.items() if key != "video_path"}
        item["video_path"] = _video_suffix(row.get("video_path", ""))
        projected.append(item)
    return sorted(projected, key=lambda value: json.dumps(value, sort_keys=True))


def _provenance_projection(provenance: Mapping[str, Any]) -> dict[str, Any]:
    e39 = provenance.get("e39")
    normalization = provenance.get("normalization")
    if not isinstance(e39, Mapping) or not isinstance(normalization, Mapping):
        raise ValueError("dense sidecar provenance is missing E09/normalization metadata")
    return {
        "schema": provenance.get("schema"),
        "source_dataset": provenance.get("source_dataset"),
        "source_commit": provenance.get("source_commit"),
        "source_info": provenance.get("source_info"),
        "window_stride": provenance.get("window_stride"),
        "horizon_frames": provenance.get("horizon_frames"),
        "split_seed": provenance.get("split_seed"),
        "split_unit": provenance.get("split_unit"),
        "transition_alignment": provenance.get("transition_alignment"),
        "trajectory_target": provenance.get("trajectory_target"),
        "rgb_storage": provenance.get("rgb_storage"),
        "rgb_repair_policy": provenance.get("rgb_repair_policy"),
        "rgb_frame_repairs": _repair_projection(provenance, "rgb_frame_repairs"),
        "corrupt_packets": _repair_projection(provenance, "corrupt_packets"),
        "normalization": {
            key: normalization.get(key) for key in ("policy", "sha256", "reason")
        },
        "e39": {
            key: e39.get(key)
            for key in (
                "teacher",
                "posterior_statistic",
                "sampling",
                "checkpoint_sha256",
                "model_config_sha256",
                "selection_report_sha256",
                "feature_reuse",
                "precision",
            )
        },
    }


def _coverage_projection(coverage: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"shards"}
    return {key: value for key, value in coverage.items() if key not in excluded}


def _validate_manifest_metadata(
    sidecar_root: Path,
    schema: Mapping[str, Any],
    *,
    valid_field: str = "e39_valid",
) -> None:
    import pyarrow.parquet as parquet

    expected = {"all": EXPECTED_ROWS, **EXPECTED_SPLIT_ROWS}
    required_columns = {
        "sample_id",
        "group_id",
        "split",
        "task_id",
        "source_id",
        "controller_id",
        "episode_id",
        "start_step",
        "end_step",
        "shard_path",
        "shard_row",
        "window_sha256",
        valid_field,
    }
    for split, expected_rows in expected.items():
        path = sidecar_root / "manifests" / f"{split}.parquet"
        metadata = parquet.ParquetFile(path).metadata
        if metadata.num_rows != expected_rows:
            raise ValueError(
                f"dense sidecar {split} manifest has {metadata.num_rows} rows, "
                f"expected {expected_rows}"
            )
        if split == "all":
            columns = set(parquet.ParquetFile(path).schema_arrow.names)
            missing = sorted(required_columns.difference(columns))
            if missing:
                raise ValueError(f"dense sidecar all manifest is missing columns: {missing}")


def validate_semantic_sidecar_binding(
    sidecar_root: str | Path,
    reference_root: str | Path,
    *,
    action_metadata: Mapping[str, Any] | None = None,
    effect_checkpoint: str | Path | None = None,
    effect_config: str | Path | None = None,
    effect_selection: str | Path | None = None,
    verify_content_checksums: bool = True,
) -> dict[str, Any]:
    """Reject a rebuilt sidecar unless its complete semantic contract matches."""

    sidecar = Path(sidecar_root).expanduser().resolve()
    reference = Path(reference_root).expanduser().resolve()
    for relative in REQUIRED_CANDIDATE_FILES:
        if not (sidecar / relative).is_file():
            raise FileNotFoundError(sidecar / relative)
    reference_hashes = verify_reference_bundle(reference)
    integrity = (
        verify_sidecar_checksum_inventory(sidecar)
        if verify_content_checksums
        else {
            "checksums_sha256": sha256_file(sidecar / "checksums.sha256"),
            "files": len(_read_checksum_records(sidecar / "checksums.sha256")),
            "bytes": None,
        }
    )

    candidate_schema = _read_json(sidecar / "schema.json")
    reference_schema = _read_json(reference / "schema.json")
    if candidate_schema != reference_schema:
        raise ValueError("dense sidecar schema is not semantically identical to the reference")
    if (
        candidate_schema.get("control_hz") != 30
        or candidate_schema.get("duration_seconds") != 1.2
        or candidate_schema.get("horizon_actions") != 36
        or candidate_schema.get("waypoint", {}).get("alignment")
        != "post_action; waypoint[t] is source observation.state at start_step+t+1"
        or candidate_schema.get("fields", {}).get("current_state", {}).get("shape") != [14]
        or candidate_schema.get("fields", {}).get("e39_mean", {}).get("shape") != [32]
        or candidate_schema.get("fields", {}).get("waypoints", {}).get("shape") != [36, 14]
        or candidate_schema.get("fields", {}).get("time_mask", {}).get("shape") != [36]
    ):
        raise ValueError("dense sidecar schema violates the locked H36/state14/effect32 contract")

    normalization_sha = sha256_file(sidecar / "stats/normalization.json")
    if normalization_sha != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("dense sidecar compatibility normalization SHA256 mismatch")
    if _read_json(sidecar / "stats/normalization.json") != _read_json(
        reference / "stats/normalization.json"
    ):
        raise ValueError("dense sidecar normalization differs semantically from the reference")

    coverage = _read_json(sidecar / "coverage_report.json")
    reference_coverage = _read_json(reference / "coverage_report.json")
    if _coverage_projection(coverage) != _coverage_projection(reference_coverage):
        raise ValueError("dense sidecar coverage semantics differ from the reference")
    if (
        coverage.get("valid") is not True
        or coverage.get("rows") != EXPECTED_ROWS
        or coverage.get("e39_valid_rows") != EXPECTED_ROWS
        or coverage.get("split_counts") != EXPECTED_SPLIT_ROWS
        or coverage.get("episodes") != 27_500
        or coverage.get("window_stride") != 1
        or coverage.get("horizon_actions") != 36
        or coverage.get("control_hz") != 30
        or coverage.get("duration_seconds") != 1.2
        or coverage.get("rgb_frame_repair_count") != 6
        or coverage.get("corrupt_packet_count") != 6
    ):
        raise ValueError("dense sidecar coverage violates the locked release")

    audit = _read_json(sidecar / "recovery_audit.json")
    if (
        audit.get("valid") is not True
        or audit.get("issue_count") != 0
        or audit.get("issues") != []
        or audit.get("rows") != EXPECTED_ROWS
        or audit.get("e39_valid_rows") != EXPECTED_ROWS
        or audit.get("split_counts") != EXPECTED_SPLIT_ROWS
        or audit.get("episode_split_disjoint") is not True
    ):
        raise ValueError("dense sidecar recovery audit is not clean and episode-disjoint")

    provenance = _read_json(sidecar / "provenance.json")
    reference_provenance = _read_json(reference / "provenance.json")
    if _provenance_projection(provenance) != _provenance_projection(reference_provenance):
        raise ValueError("dense sidecar provenance semantics differ from the reference")
    projected = _provenance_projection(provenance)
    if (
        projected["source_dataset"] != EXPECTED_SOURCE_DATASET
        or projected["source_commit"] != EXPECTED_SOURCE_REVISION
        or projected["source_info"]
        != {"total_episodes": 27_500, "total_frames": 6_077_247, "fps": 30}
        or projected["split_seed"] != 20260818
        or projected["split_unit"] != "episode within benchmark effect family"
        or projected["window_stride"] != 1
        or projected["horizon_frames"] != 36
        or projected["normalization"]["sha256"] != EXPECTED_NORMALIZATION_SHA256
        or projected["e39"]["checkpoint_sha256"] != EXPECTED_EFFECT_CHECKPOINT_SHA256
        or projected["e39"]["model_config_sha256"] != EXPECTED_EFFECT_CONFIG_SHA256
        or projected["e39"]["selection_report_sha256"] != EXPECTED_EFFECT_SELECTION_SHA256
        or projected["e39"]["posterior_statistic"] != "mean"
        or projected["e39"]["sampling"] is not False
    ):
        raise ValueError("dense sidecar source/E09 semantics violate the locked release")
    _validate_manifest_metadata(sidecar, candidate_schema)

    explicit_assets = {
        "effect_checkpoint": (effect_checkpoint, EXPECTED_EFFECT_CHECKPOINT_SHA256),
        "effect_config": (effect_config, EXPECTED_EFFECT_CONFIG_SHA256),
        "effect_selection": (effect_selection, EXPECTED_EFFECT_SELECTION_SHA256),
    }
    for name, (value, expected_sha) in explicit_assets.items():
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{name} SHA256 mismatch for semantic sidecar binding")

    if action_metadata is not None:
        if action_metadata.get("normalization_sha256") != EXPECTED_NORMALIZATION_SHA256:
            raise ValueError("ActionVAE metadata uses a different normalization")
        if action_metadata.get("e39_checkpoint_sha256") != EXPECTED_EFFECT_CHECKPOINT_SHA256:
            raise ValueError("ActionVAE metadata uses a different Effect teacher")

    return {
        "schema": SEMANTIC_BINDING_SCHEMA,
        "sidecar_root": str(sidecar),
        "reference_root": str(reference),
        "reference_checksums_sha256": sha256_file(reference / "checksums.sha256"),
        "reference_files": reference_hashes,
        "candidate_integrity": integrity,
        "rows": EXPECTED_ROWS,
        "split_rows": dict(EXPECTED_SPLIT_ROWS),
        "source_revision": EXPECTED_SOURCE_REVISION,
        "normalization_sha256": normalization_sha,
        "effect_checkpoint_sha256": EXPECTED_EFFECT_CHECKPOINT_SHA256,
    }


def validate_lawam_idm_sidecar_contract(
    sidecar_root: str | Path,
    *,
    action_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a rebuilt LaWAM-IDM sidecar without byte-identical shard hashes."""

    sidecar = Path(sidecar_root).expanduser().resolve()
    required = (
        "_SUCCESS",
        "schema.json",
        "stats/normalization.json",
        "coverage_report.json",
        "recovery_audit.json",
        "provenance.json",
        "manifests/all.parquet",
        "manifests/train.parquet",
        "manifests/validation.parquet",
        "manifests/test.parquet",
    )
    for relative in required:
        if not (sidecar / relative).is_file():
            raise FileNotFoundError(sidecar / relative)

    schema = _read_json(sidecar / "schema.json")
    condition_schema = schema.get("condition")
    fields = schema.get("fields")
    if (
        schema.get("schema_version") != LAWAM_IDM_SIDECAR_SCHEMA
        or not isinstance(condition_schema, Mapping)
        or condition_schema.get("source") != "lawam_idm"
        or condition_schema.get("mean_field") != "lawam_mean"
        or condition_schema.get("valid_field") != "lawam_valid"
        or condition_schema.get("sampling") is not False
        or not isinstance(fields, Mapping)
        or fields.get("lawam_mean", {}).get("shape") != [32]
        or fields.get("current_state", {}).get("shape") != [14]
        or fields.get("waypoints", {}).get("shape") != [36, 14]
        or fields.get("time_mask", {}).get("shape") != [36]
        or float(schema.get("control_hz", -1)) != 30.0
        or float(schema.get("duration_seconds", -1)) != 1.2
        or int(schema.get("horizon_actions", -1)) != 36
    ):
        raise ValueError("LaWAM-IDM sidecar schema violates the H36/condition32 contract")

    coverage = _read_json(sidecar / "coverage_report.json")
    if (
        coverage.get("valid") is not True
        or coverage.get("condition_source") != "lawam_idm"
        or int(coverage.get("rows", -1)) != EXPECTED_ROWS
        or int(coverage.get("condition_valid_rows", -1)) != EXPECTED_ROWS
        or coverage.get("split_counts") != EXPECTED_SPLIT_ROWS
        or int(coverage.get("episodes", -1)) != 27_500
        or int(coverage.get("window_stride", -1)) != 1
        or int(coverage.get("horizon_actions", -1)) != 36
        or float(coverage.get("control_hz", -1)) != 30.0
    ):
        raise ValueError("LaWAM-IDM sidecar coverage violates the dense stride-1 release")

    audit = _read_json(sidecar / "recovery_audit.json")
    if (
        audit.get("valid") is not True
        or audit.get("condition_source") != "lawam_idm"
        or int(audit.get("issue_count", -1)) != 0
        or audit.get("issues") != []
        or int(audit.get("rows", -1)) != EXPECTED_ROWS
        or int(audit.get("condition_valid_rows", -1)) != EXPECTED_ROWS
        or audit.get("split_counts") != EXPECTED_SPLIT_ROWS
        or audit.get("episode_split_disjoint") is not True
    ):
        raise ValueError("LaWAM-IDM sidecar recovery audit is incomplete")

    provenance = _read_json(sidecar / "provenance.json")
    condition = provenance.get("condition", provenance.get("lawam_idm"))
    if (
        provenance.get("source_dataset") != EXPECTED_SOURCE_DATASET
        or provenance.get("source_commit") != EXPECTED_SOURCE_REVISION
        or provenance.get("source_info")
        != {"total_episodes": 27_500, "total_frames": 6_077_247, "fps": 30}
        or int(provenance.get("window_stride", -1)) != 1
        or int(provenance.get("horizon_frames", -1)) != 36
        or not isinstance(condition, Mapping)
        or condition.get("source") != "lawam_idm"
        or condition.get("posterior_statistic") != "mean"
        or condition.get("sampling") is not False
    ):
        raise ValueError("LaWAM-IDM sidecar source/condition provenance is incompatible")

    normalization_sha = sha256_file(sidecar / "stats/normalization.json")
    condition_sha = str(condition.get("checkpoint_sha256", "")).lower()
    if len(condition_sha) != 64:
        raise ValueError("LaWAM-IDM sidecar condition checkpoint identity is missing")
    if action_metadata is not None:
        expected = {
            "condition_source": "lawam_idm",
            "condition_checkpoint_sha256": condition_sha,
            "normalization_sha256": normalization_sha,
        }
        mismatches = {
            key: (action_metadata.get(key), value)
            for key, value in expected.items()
            if action_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "ActionVAE and LaWAM-IDM sidecar contracts differ: "
                f"{mismatches}"
            )

    _validate_manifest_metadata(sidecar, schema, valid_field="lawam_valid")
    return {
        "schema": LAWAM_IDM_CONTRACT_BINDING_SCHEMA,
        "sidecar_root": str(sidecar),
        "rows": EXPECTED_ROWS,
        "split_rows": dict(EXPECTED_SPLIT_ROWS),
        "source_revision": EXPECTED_SOURCE_REVISION,
        "condition_source": "lawam_idm",
        "condition_checkpoint_sha256": condition_sha,
        "normalization_sha256": normalization_sha,
        "byte_identical_sidecar_required": False,
    }


__all__ = [
    "EXPECTED_EFFECT_CHECKPOINT_SHA256",
    "EXPECTED_EFFECT_CONFIG_SHA256",
    "EXPECTED_EFFECT_SELECTION_SHA256",
    "EXPECTED_NORMALIZATION_SHA256",
    "EXPECTED_ROWS",
    "EXPECTED_SOURCE_REVISION",
    "EXPECTED_SPLIT_ROWS",
    "SEMANTIC_BINDING_SCHEMA",
    "LAWAM_IDM_CONTRACT_BINDING_SCHEMA",
    "LAWAM_IDM_SIDECAR_SCHEMA",
    "sha256_file",
    "validate_semantic_sidecar_binding",
    "validate_lawam_idm_sidecar_contract",
    "verify_reference_bundle",
    "verify_sidecar_checksum_inventory",
]
