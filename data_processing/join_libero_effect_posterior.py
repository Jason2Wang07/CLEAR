#!/usr/bin/env python3
"""Join selected effect posterior means into a schema-preserving immutable A1 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from libero_dataset.converter import _window_digest as _libero_window_digest
from z_action_sidecar.io import write_shard_atomic
from z_action_sidecar.schema import E39_DIM, fields_for_horizon


def _sidecar_contract(
    schema_version: str,
    horizon: int,
) -> tuple[
    Mapping[str, tuple[str, tuple[int, ...]]],
    Callable[[Mapping[str, Any]], str],
    str,
]:
    if schema_version == "bridge-z-action-sidecar-v1-12s-resampled24":
        from bridge_dataset.converter import _window_digest as bridge_window_digest
        from bridge_dataset.schema import REPRESENTATION_STEPS, action_fields

        if int(horizon) != int(REPRESENTATION_STEPS):
            raise ValueError(
                f"Bridge sidecar horizon must be {REPRESENTATION_STEPS}, got {horizon}"
            )
        return action_fields(), bridge_window_digest, "bridge"
    return fields_for_horizon(horizon), _libero_window_digest, "libero"


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root)}" for path in paths]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("posterior cache lacks metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema") != "lawam-effect-posterior-means-v1":
            raise ValueError(f"posterior cache schema mismatch: {metadata.get('schema')!r}")
        ids = archive["sample_ids"].astype(np.str_)
        splits = archive["splits"].astype(np.str_)
        means = np.asarray(archive["posterior_mean"], dtype=np.float32)
    if means.ndim == 3 and means.shape[1:] == (1, E39_DIM):
        means = means[:, 0]
    if means.shape != (len(ids), E39_DIM) or splits.shape != (len(ids),):
        raise ValueError(
            f"posterior cache shapes are invalid: ids={ids.shape}, splits={splits.shape}, means={means.shape}"
        )
    if not np.isfinite(means).all():
        raise ValueError("posterior means contain NaN/Inf")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("posterior cache contains duplicate sample_id values")
    unknown = sorted(set(splits.tolist()).difference({"train", "validation", "test"}))
    if unknown:
        raise ValueError(f"posterior cache contains unknown splits: {unknown}")
    return (
        dict(zip(ids.tolist(), means, strict=True)),
        dict(zip(ids.tolist(), splits.tolist(), strict=True)),
        metadata,
    )


def _read_rows(
    path: Path,
    fields: Mapping[str, tuple[str, tuple[int, ...]]],
) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"source shard is incomplete: {path}")
        missing = sorted(set(fields).difference(handle.keys()))
        if missing:
            raise ValueError(f"source shard {path} is missing fields: {missing}")
        count = len(handle["sample_id"])
        rows: list[dict[str, Any]] = []
        for index in range(count):
            rows.append({name: _decode(handle[name][index]) for name in fields})
    return rows


def join_release(
    source_root: Path,
    posterior_cache: Path,
    output: Path,
    *,
    effect_manifest: Path,
) -> dict[str, Any]:
    source = source_root.expanduser().resolve()
    cache = posterior_cache.expanduser().resolve()
    destination = output.expanduser().resolve()
    effect_manifest = effect_manifest.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    for required in (
        source / "_SUCCESS",
        source / "schema.json",
        source / "provenance.json",
        source / "coverage_report.json",
        source / "manifests/all.parquet",
        source / "stats/normalization.json",
        source / "checksums.sha256",
        cache,
        effect_manifest,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    horizon = int(source_schema.get("horizon_actions", 0))
    sidecar_schema_version = str(source_schema.get("schema_version", ""))
    fields, window_digest, domain = _sidecar_contract(sidecar_schema_version, horizon)
    manifest = pd.read_parquet(source / "manifests/all.parquet").sort_values("sample_id").reset_index(drop=True)
    if manifest.sample_id.duplicated().any():
        raise ValueError("source manifest contains duplicate sample_id values")
    effect = pd.read_parquet(effect_manifest).sort_values("sample_id").reset_index(drop=True)
    if effect.sample_id.astype(str).tolist() != manifest.sample_id.astype(str).tolist():
        raise ValueError("effect/action manifests do not have identical sample_id sets")
    if effect.split.astype(str).tolist() != manifest.split.astype(str).tolist():
        raise ValueError("effect/action manifests do not have identical split ownership")
    means, cache_splits, metadata = _load_cache(cache)
    manifest_ids = manifest.sample_id.astype(str).tolist()
    manifest_set = set(manifest_ids)
    cache_set = set(means)
    if manifest_set != cache_set:
        missing = sorted(manifest_set - cache_set)[:8]
        extra = sorted(cache_set - manifest_set)[:8]
        raise ValueError(f"posterior exact-set join failed: missing={missing}, extra={extra}")
    split_mismatches = [
        sample_id
        for sample_id, split in manifest[["sample_id", "split"]].itertuples(index=False, name=None)
        if cache_splits[str(sample_id)] != str(split)
    ]
    if split_mismatches:
        raise ValueError(f"posterior split mismatch for {len(split_mismatches)} rows: {split_mismatches[:8]}")

    checkpoint = Path(str(metadata.get("checkpoint", ""))).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"posterior metadata checkpoint is missing: {checkpoint}")

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, temporary)
        (temporary / "_SUCCESS").unlink(missing_ok=True)
        digests: dict[str, str] = {}
        for relative, part in manifest.groupby("shard_path", sort=True):
            source_shard = source / str(relative)
            target_shard = temporary / str(relative)
            rows = _read_rows(source_shard, fields)
            for item in part.itertuples(index=False):
                row_index = int(item.shard_row)
                sample_id = str(item.sample_id)
                stored_id = str(_decode(rows[row_index]["sample_id"]))
                if stored_id != sample_id:
                    raise ValueError(
                        f"source shard identity mismatch at {relative}:{row_index}: {stored_id!r} != {sample_id!r}"
                    )
                rows[row_index]["e39_mean"] = means[sample_id]
                rows[row_index]["e39_valid"] = np.uint8(1)
                rows[row_index]["window_sha256"] = window_digest(rows[row_index])
                digests[sample_id] = str(rows[row_index]["window_sha256"])
            target_shard.unlink()
            write_shard_atomic(
                target_shard,
                rows,
                schema_version=sidecar_schema_version,
                fields=fields,
            )

        manifest["e39_valid"] = True
        manifest["window_sha256"] = [digests[sample_id] for sample_id in manifest_ids]
        manifests = temporary / "manifests"
        manifest.to_parquet(manifests / "all.parquet", index=False)
        for split in ("train", "validation", "test"):
            manifest[manifest.split.astype(str) == split].to_parquet(
                manifests / f"{split}.parquet", index=False
            )

        schema = json.loads((temporary / "schema.json").read_text(encoding="utf-8"))
        selection = metadata.get("checkpoint_selection", {})
        if not isinstance(selection, dict):
            raise ValueError("posterior checkpoint_selection metadata must be a mapping")
        selection_status = str(selection.get("status", "legacy_validation_selected"))
        is_ablation = selection_status == "explicit_ablation"
        domain_label = "Bridge" if domain == "bridge" else "LIBERO"
        schema["e39"]["semantics"] = (
            f"deterministic posterior mean from an explicit {domain_label} Stage-1 ablation checkpoint"
            if is_ablation
            else f"deterministic posterior mean from the validation-selected {domain_label} Stage-1 checkpoint"
        )
        _atomic_json(temporary / "schema.json", schema)
        provenance = json.loads((temporary / "provenance.json").read_text(encoding="utf-8"))
        provenance.update(
            {
                "parent_action_release": str(source),
                "parent_action_checksums_sha256": _sha256_file(source / "checksums.sha256"),
                "effect_manifest_sha256": _sha256_file(effect_manifest),
                "e39": {
                    "semantic_name": (
                        f"ablation_{domain}_effect_posterior_mean"
                        if is_ablation
                        else f"selected_{domain}_effect_posterior_mean"
                    ),
                    "path": str(cache),
                    "sha256": _sha256_file(cache),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "teacher": str(metadata.get("teacher", "unspecified")),
                    "selection_status": selection_status,
                    "selection_report": selection.get("report"),
                    "selection_report_sha256": selection.get("report_sha256"),
                    "selection_rationale": selection.get("rationale"),
                    "rows": len(manifest),
                },
            }
        )
        _atomic_json(temporary / "provenance.json", provenance)
        coverage = json.loads((temporary / "coverage_report.json").read_text(encoding="utf-8"))
        coverage["e39_valid_rows"] = len(manifest)
        _atomic_json(temporary / "coverage_report.json", coverage)
        (temporary / "_SUCCESS").write_text(
            f"{domain}-actionvae-action-a1-release-v1\n",
            encoding="utf-8",
        )
        _write_checksums(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": f"{domain}-effect-posterior-join-report-v1",
        "valid": True,
        "rows": len(manifest),
        "split_counts": {
            str(key): int(value)
            for key, value in manifest.split.astype(str).value_counts().sort_index().items()
        },
        "output": str(destination),
        "posterior_cache_sha256": _sha256_file(cache),
        "checkpoint_sha256": _sha256_file(checkpoint),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--posterior-cache", required=True, type=Path)
    parser.add_argument("--effect-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = join_release(
        args.source_root,
        args.posterior_cache,
        args.output,
        effect_manifest=args.effect_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
