"""Exact-set and identity audit for LIBERO Stage 2 cache manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


JOIN_AUDIT_SCHEMA_VERSION = "libero-stage2-exact-join-audit-v1"
JOINED_MANIFEST_SCHEMA_VERSION = "libero-stage2-joined-manifest-v1"
IDENTITY_COLUMNS = (
    "sample_id",
    "split",
    "task_id",
    "episode_id",
    "start_step",
    "end_step",
    "window_sha256",
)


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path, *, name: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} manifest is missing identity columns: {missing}")
    frame = frame.copy()
    for column in ("sample_id", "split", "task_id", "window_sha256"):
        frame[column] = frame[column].astype(str)
    for column in ("episode_id", "start_step", "end_step"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame


def _examples(values: Sequence[Any], limit: int) -> list[str]:
    return [str(value) for value in list(values)[:limit]]


def audit_exact_join(
    *,
    base_manifest: str | Path,
    cache_manifests: Mapping[str, str | Path],
    example_limit: int = 20,
) -> dict[str, Any]:
    if not cache_manifests:
        raise ValueError("exact join audit requires at least one cache manifest")
    if example_limit < 1:
        raise ValueError("example_limit must be positive")
    base_path = Path(base_manifest).expanduser().resolve()
    base = _read_manifest(base_path, name="base")
    base_duplicates = sorted(base.loc[base.sample_id.duplicated(keep=False), "sample_id"].unique())
    base_by_id = base.drop_duplicates("sample_id", keep="first").set_index("sample_id", drop=False)
    issues: list[str] = []
    if base_duplicates:
        issues.append("base_duplicate_sample_ids")
    components: dict[str, Any] = {}
    for name, raw_path in sorted(cache_manifests.items()):
        if not str(name).strip() or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in str(name)):
            raise ValueError(f"cache name must use lowercase ASCII identifier characters: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        cache = _read_manifest(path, name=name)
        duplicates = sorted(cache.loc[cache.sample_id.duplicated(keep=False), "sample_id"].unique())
        cache_by_id = cache.drop_duplicates("sample_id", keep="first").set_index("sample_id", drop=False)
        base_ids = set(base_by_id.index)
        cache_ids = set(cache_by_id.index)
        missing_ids = sorted(base_ids.difference(cache_ids))
        extra_ids = sorted(cache_ids.difference(base_ids))
        common_ids = sorted(base_ids.intersection(cache_ids))
        mismatch_counts: dict[str, int] = {}
        mismatch_examples: dict[str, list[str]] = {}
        if common_ids:
            base_common = base_by_id.loc[common_ids]
            cache_common = cache_by_id.loc[common_ids]
            for column in IDENTITY_COLUMNS[1:]:
                mismatch = base_common[column].to_numpy() != cache_common[column].to_numpy()
                mismatch_ids = base_common.index[mismatch].tolist()
                mismatch_counts[column] = len(mismatch_ids)
                mismatch_examples[column] = _examples(mismatch_ids, example_limit)
        else:
            mismatch_counts = {column: 0 for column in IDENTITY_COLUMNS[1:]}
            mismatch_examples = {column: [] for column in IDENTITY_COLUMNS[1:]}
        component_issues = []
        if duplicates:
            component_issues.append("duplicate_sample_ids")
        if missing_ids:
            component_issues.append("missing_sample_ids")
        if extra_ids:
            component_issues.append("extra_sample_ids")
        for column, count in mismatch_counts.items():
            if count:
                component_issues.append(f"identity_mismatch:{column}")
        issues.extend(f"{name}:{issue}" for issue in component_issues)
        components[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(cache),
            "unique_sample_ids": len(cache_ids),
            "duplicate_count": len(duplicates),
            "duplicate_examples": _examples(duplicates, example_limit),
            "missing_count": len(missing_ids),
            "missing_examples": _examples(missing_ids, example_limit),
            "extra_count": len(extra_ids),
            "extra_examples": _examples(extra_ids, example_limit),
            "identity_mismatch_counts": mismatch_counts,
            "identity_mismatch_examples": mismatch_examples,
            "issues": component_issues,
            "exact_set_and_identity_match": not component_issues,
        }
    return {
        "schema_version": JOIN_AUDIT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "rows": len(base),
            "unique_sample_ids": len(base_by_id),
            "duplicate_count": len(base_duplicates),
            "duplicate_examples": _examples(base_duplicates, example_limit),
        },
        "identity_columns": list(IDENTITY_COLUMNS),
        "components": components,
        "issues": issues,
        "issue_count": len(issues),
        "ready": not issues,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_join_audit(path: Path, report: Mapping[str, Any]) -> None:
    _write_json_atomic(path, report, label="exact join audit")


def publish_joined_manifest(
    *,
    output_root: str | Path,
    base_manifest: str | Path,
    cache_manifests: Mapping[str, str | Path],
    audit_report: Mapping[str, Any],
) -> dict[str, Any]:
    if audit_report.get("schema_version") != JOIN_AUDIT_SCHEMA_VERSION:
        raise ValueError("join audit schema mismatch")
    if audit_report.get("ready") is not True or audit_report.get("issue_count") != 0:
        raise ValueError("refusing to publish a joined manifest from a failed exact join audit")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite joined manifest release: {output}")
    base = _read_manifest(Path(base_manifest), name="base")
    if base.sample_id.duplicated().any():
        raise ValueError("base manifest changed after audit and now contains duplicate sample IDs")
    joined = base.set_index("sample_id", drop=False)
    column_map: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(cache_manifests.items()):
        cache = _read_manifest(Path(raw_path), name=name).set_index("sample_id", drop=False)
        cache = cache.loc[joined.index]
        extras = [column for column in cache.columns if column not in IDENTITY_COLUMNS]
        mapping = {column: f"{name}__{column}" for column in extras}
        if set(mapping.values()).intersection(joined.columns):
            raise ValueError(f"joined cache columns collide for component {name}")
        joined = joined.join(cache[extras].rename(columns=mapping), how="left", validate="one_to_one")
        column_map[name] = mapping
    joined = joined.reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        manifest_path = temporary / "manifest.parquet"
        joined.to_parquet(manifest_path, index=False)
        audit_path = temporary / "join_audit.json"
        audit_path.write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance = {
            "schema_version": JOINED_MANIFEST_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(joined),
            "identity_columns": list(IDENTITY_COLUMNS),
            "column_map": column_map,
            "base_manifest": {
                "path": str(Path(base_manifest).expanduser().resolve()),
                "sha256": sha256_file(base_manifest),
            },
            "cache_manifests": {
                name: {
                    "path": str(Path(path).expanduser().resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(cache_manifests.items())
            },
        }
        provenance_path = temporary / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        names = ("join_audit.json", "manifest.parquet", "provenance.json")
        checksums = "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in names)
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text(checksums, encoding="ascii")
        success = {
            "schema_version": JOINED_MANIFEST_SCHEMA_VERSION,
            "row_count": len(joined),
            "manifest_sha256": sha256_file(manifest_path),
            "checksums_sha256": sha256_file(checksums_path),
            "all_gates_passed": True,
        }
        (temporary / "_SUCCESS").write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        return {"output": str(output), **success}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _cache_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("cache manifests must use NAME=PATH")
    return name, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, action="append", type=_cache_argument)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--joined-output", type=Path)
    parser.add_argument("--example-limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cache_manifests = dict(args.cache)
    if len(cache_manifests) != len(args.cache):
        raise ValueError("cache names must be unique")
    report = audit_exact_join(
        base_manifest=args.base_manifest,
        cache_manifests=cache_manifests,
        example_limit=args.example_limit,
    )
    write_join_audit(args.output_report, report)
    published = None
    if args.joined_output is not None and report["ready"]:
        published = publish_joined_manifest(
            output_root=args.joined_output,
            base_manifest=args.base_manifest,
            cache_manifests=cache_manifests,
            audit_report=report,
        )
    print(
        json.dumps(
            {
                "output_report": str(args.output_report.expanduser().resolve()),
                "ready": report["ready"],
                "issue_count": report["issue_count"],
                "joined_output": published,
            },
            sort_keys=True,
        )
    )
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
