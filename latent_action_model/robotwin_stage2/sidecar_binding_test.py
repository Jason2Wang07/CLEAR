from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from . import sidecar_binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_inventory_is_closed_and_content_verified(tmp_path: Path) -> None:
    root = tmp_path / "sidecar"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"dense-sidecar")
    (root / "checksums.sha256").write_text(
        f"{_sha256(payload)}  payload.bin\n", encoding="ascii"
    )

    report = sidecar_binding.verify_sidecar_checksum_inventory(root)

    assert report["files"] == 1
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content checksum mismatch"):
        sidecar_binding.verify_sidecar_checksum_inventory(root)


def test_checksum_inventory_rejects_unlisted_file(tmp_path: Path) -> None:
    root = tmp_path / "sidecar"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"listed")
    (root / "extra.bin").write_bytes(b"unlisted")
    (root / "checksums.sha256").write_text(
        f"{_sha256(payload)}  payload.bin\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="inventory is not closed"):
        sidecar_binding.verify_sidecar_checksum_inventory(root)


def test_reference_bundle_is_success_marker_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reference"
    for relative in sidecar_binding.REFERENCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    normalization = root / "stats/normalization.json"
    monkeypatch.setattr(
        sidecar_binding, "EXPECTED_NORMALIZATION_SHA256", _sha256(normalization)
    )
    checksums = root / "checksums.sha256"
    checksums.write_text(
        "".join(
            f"{_sha256(root / relative)}  {relative}\n"
            for relative in sidecar_binding.REFERENCE_FILES
        ),
        encoding="ascii",
    )
    (root / "_SUCCESS").write_text(_sha256(checksums) + "\n", encoding="ascii")

    sidecar_binding.verify_reference_bundle(root)

    (root / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="_SUCCESS"):
        sidecar_binding.verify_reference_bundle(root)


def test_lawam_idm_contract_binding_does_not_require_sidecar_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rebuilt"
    required = (
        "_SUCCESS",
        "stats/normalization.json",
        "manifests/all.parquet",
        "manifests/train.parquet",
        "manifests/validation.parquet",
        "manifests/test.parquet",
    )
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"rebuilt:{relative}\n", encoding="utf-8")
    payloads = {
        "schema.json": {
            "schema_version": sidecar_binding.LAWAM_IDM_SIDECAR_SCHEMA,
            "condition": {
                "source": "lawam_idm",
                "mean_field": "lawam_mean",
                "valid_field": "lawam_valid",
                "sampling": False,
            },
            "fields": {
                "lawam_mean": {"shape": [32]},
                "current_state": {"shape": [14]},
                "waypoints": {"shape": [36, 14]},
                "time_mask": {"shape": [36]},
            },
            "control_hz": 30.0,
            "duration_seconds": 1.2,
            "horizon_actions": 36,
        },
        "coverage_report.json": {
            "valid": True,
            "condition_source": "lawam_idm",
            "rows": sidecar_binding.EXPECTED_ROWS,
            "condition_valid_rows": sidecar_binding.EXPECTED_ROWS,
            "split_counts": sidecar_binding.EXPECTED_SPLIT_ROWS,
            "episodes": 27_500,
            "window_stride": 1,
            "horizon_actions": 36,
            "control_hz": 30.0,
        },
        "recovery_audit.json": {
            "valid": True,
            "condition_source": "lawam_idm",
            "issue_count": 0,
            "issues": [],
            "rows": sidecar_binding.EXPECTED_ROWS,
            "condition_valid_rows": sidecar_binding.EXPECTED_ROWS,
            "split_counts": sidecar_binding.EXPECTED_SPLIT_ROWS,
            "episode_split_disjoint": True,
        },
        "provenance.json": {
            "source_dataset": sidecar_binding.EXPECTED_SOURCE_DATASET,
            "source_commit": sidecar_binding.EXPECTED_SOURCE_REVISION,
            "source_info": {
                "total_episodes": 27_500,
                "total_frames": 6_077_247,
                "fps": 30,
            },
            "window_stride": 1,
            "horizon_frames": 36,
            "condition": {
                "source": "lawam_idm",
                "posterior_statistic": "mean",
                "sampling": False,
                "checkpoint_sha256": "a" * 64,
                "model_config_sha256": "b" * 64,
            },
        },
    }
    for relative, payload in payloads.items():
        (root / relative).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sidecar_binding, "_validate_manifest_metadata", lambda *args, **kwargs: None)

    report = sidecar_binding.validate_lawam_idm_sidecar_contract(root)

    assert report["byte_identical_sidecar_required"] is False
    assert report["rows"] == 5_087_247
    assert "checksums_sha256" not in report
