"""Build an immutable RoboTwin ActionAE sidecar aligned to the effect release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from robotwin_dataset.geometry import dual_arm_contract
from robotwin_dataset.source import load_source_contract, source_commit

from .geometry import canonical_dual_arm_waypoints
from .io import write_shard_atomic
from .schema import (
    CONTROL_HZ,
    EFFECT_DIM,
    HORIZON,
    NATIVE_ACTION_DIM,
    TRAJECTORY_DIM,
    anchor_steps_for_horizon,
    fields_for_horizon,
    schema_document,
    versions_for_horizon,
)
from .statistics import normalization_from_train_shards, write_normalization


SPLITS = ("train", "validation", "test")
CONTROLLER_ID = "robotwin_dual_arm_absolute_eef_wxyz"


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_digest(commit: str, relative_path: str, size: int) -> str:
    return hashlib.sha256(f"{commit}\0{relative_path}\0{int(size)}".encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{_sha256_file(path)}  {path.relative_to(root)}" for path in paths]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _list16(column: Any, *, name: str, path: Path) -> np.ndarray:
    values = column.combine_chunks()
    offsets = values.offsets.to_numpy(zero_copy_only=False)
    if len(offsets) != len(values) + 1 or not np.all(np.diff(offsets) == 16):
        raise ValueError(f"{path}: {name} rows must all have width 16")
    array = values.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 16)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def _window_digest(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("sample_id", "split", "source_commit", "controller_id"):
        digest.update(str(row[name]).encode("utf-8"))
        digest.update(b"\0")
    for name in (
        "episode_id",
        "start_step",
        "end_step",
        "dual_arm_pose_start",
        "dual_arm_pose_end",
        "waypoints",
        "native_actions",
        "current_state",
        "current_state_mask",
    ):
        digest.update(np.ascontiguousarray(row[name]).tobytes())
    return digest.hexdigest()


def _load_effect_contracts(
    effect_root: Path,
    manifest: pd.DataFrame,
    *,
    effect_schema_version: str,
) -> dict[str, dict[str, np.ndarray]]:
    contracts: dict[str, dict[str, np.ndarray]] = {}
    for relative, group in manifest.groupby("shard_path", sort=True):
        path = effect_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"effect shard does not exist: {path}")
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != effect_schema_version or not handle.attrs.get("complete"):
                raise ValueError(f"invalid RoboTwin effect shard: {path}")
            for manifest_index, item in group.iterrows():
                row = int(item["shard_row"])
                sample_id = str(item["sample_id"])
                if row < 0 or row >= handle["sample_id"].shape[0]:
                    raise IndexError(f"effect manifest row {manifest_index} references {path}:{row}")
                if _decode(handle["sample_id"][row]) != sample_id:
                    raise ValueError(f"effect manifest/HDF5 sample mismatch for {sample_id}")
                if _decode(handle["split"][row]) != str(item["split"]):
                    raise ValueError(f"effect manifest/HDF5 split mismatch for {sample_id}")
                contracts[sample_id] = {
                    "state14_start": np.asarray(handle["state14_start"][row], dtype=np.float32),
                    "state14_end": np.asarray(handle["state14_end"][row], dtype=np.float32),
                    "delta_state14": np.asarray(handle["delta_state14"][row], dtype=np.float32),
                    "state_mask": np.asarray(handle["state_mask"][row], dtype=np.uint8),
                }
    if len(contracts) != len(manifest):
        raise RuntimeError(f"effect contract coverage mismatch: {len(contracts)} vs {len(manifest)}")
    return contracts


class _ShardWriter:
    def __init__(
        self,
        root: Path,
        shard_rows: int,
        *,
        schema_version: str,
        fields: Mapping[str, tuple[str, tuple[int, ...]]],
    ) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.root = root
        self.shard_rows = int(shard_rows)
        self.schema_version = schema_version
        self.fields = fields
        self.buffers: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
            split: [] for split in SPLITS
        }
        self.indices = {split: 0 for split in SPLITS}
        self.manifest: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any], effect_manifest: dict[str, Any]) -> None:
        split = str(row["split"])
        if split not in self.buffers:
            raise ValueError(f"unsupported split {split!r}")
        self.buffers[split].append((row, effect_manifest))
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        paired = self.buffers[split]
        if not paired:
            return
        relative = Path("shards") / split / f"part-{self.indices[split]:05d}.h5"
        rows = [item[0] for item in paired]
        write_shard_atomic(
            self.root / relative,
            rows,
            schema_version=self.schema_version,
            fields=self.fields,
        )
        for shard_row, (row, effect) in enumerate(paired):
            self.manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "split": split,
                    "task_id": row["task_id"],
                    "effect_family": effect["effect_family"],
                    "source_id": row["source_id"],
                    "source_commit": row["source_commit"],
                    "controller_id": row["controller_id"],
                    "episode_id": row["episode_id"],
                    "episode_seed": row["episode_seed"],
                    "start_step": row["start_step"],
                    "end_step": row["end_step"],
                    "control_hz": row["control_hz"],
                    "tau": effect["tau"],
                    "effect_direction": effect["effect_direction"],
                    "effect_magnitude": effect["effect_magnitude"],
                    "language_instruction": effect["language_instruction"],
                    "suite": effect["suite"],
                    "effect_shard_path": str(effect["shard_path"]),
                    "effect_shard_row": int(effect["shard_row"]),
                    "shard_path": relative.as_posix(),
                    "shard_row": shard_row,
                    "window_sha256": row["window_sha256"],
                    "e39_valid": False,
                }
            )
        self.buffers[split] = []
        self.indices[split] += 1

    def finish(self) -> None:
        for split in SPLITS:
            self.flush(split)


def _write_manifests(root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(manifests / "all.parquet", index=False)
    for split in SPLITS:
        frame[frame.split == split].to_parquet(manifests / f"{split}.parquet", index=False)


def _validate_effect_manifest(
    effect_root: Path,
    *,
    horizon: int,
    max_rows: int | None,
) -> pd.DataFrame:
    path = effect_root / "manifests/all.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = pd.read_parquet(path)
    required = {
        "sample_id", "group_id", "split", "task_id", "effect_family", "source_id",
        "source_commit", "controller_id", "episode_id", "episode_seed", "start_step",
        "end_step", "control_hz", "tau", "effect_direction", "effect_magnitude",
        "language_instruction", "suite", "shard_path", "shard_row",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"effect manifest is missing required columns: {missing}")
    if manifest.sample_id.duplicated().any():
        raise ValueError("effect manifest contains duplicate sample_id values")
    if set(manifest.split.astype(str)) != set(SPLITS):
        raise ValueError("effect manifest must contain train/validation/test")
    if not np.all(manifest.end_step.astype(np.int64) - manifest.start_step.astype(np.int64) == horizon):
        raise ValueError(f"effect manifest contains a non-{horizon}-step window")
    if set(manifest.controller_id.astype(str)) != {CONTROLLER_ID}:
        raise ValueError("effect manifest has an incompatible controller")
    if max_rows is not None:
        if max_rows < 3:
            raise ValueError("max_rows must be at least three")
        per_split = max(1, int(max_rows) // len(SPLITS))
        manifest = pd.concat(
            [manifest[manifest.split == split].sort_values("sample_id").head(per_split) for split in SPLITS],
            ignore_index=True,
        )
    return manifest.sort_values("sample_id").reset_index(drop=True)


def build_release(
    source_root: str | Path,
    effect_root: str | Path,
    output: str | Path,
    *,
    shard_rows: int = 256,
    max_rows: int | None = None,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    effect = Path(effect_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    if not (effect / "_SUCCESS").is_file():
        raise FileNotFoundError(f"effect release is incomplete: {effect}")
    effect_schema = json.loads((effect / "schema.json").read_text(encoding="utf-8"))
    horizon = int(round(float(effect_schema.get("tau_seconds", 0.0)) * CONTROL_HZ))
    if horizon < 1:
        raise ValueError(f"effect schema has invalid tau_seconds: {effect_schema.get('tau_seconds')!r}")
    sidecar_schema_version, normalization_schema_version, _ = versions_for_horizon(horizon)
    action_fields = fields_for_horizon(horizon)
    anchor_steps = anchor_steps_for_horizon(horizon)
    effect_schema_version = str(effect_schema.get("schema_version", ""))
    info, episodes = load_source_contract(source)
    commit = source_commit(source)
    manifest = _validate_effect_manifest(effect, horizon=horizon, max_rows=max_rows)
    if set(manifest.source_commit.astype(str)) != {commit}:
        raise ValueError("effect release and source checkout commits differ")

    episode_records = {
        int(episodes["episode_index"][index]): {name: values[index] for name, values in episodes.items()}
        for index in range(len(episodes["episode_index"]))
    }
    file_starts: dict[tuple[int, int], int] = {}
    for episode_id, record in episode_records.items():
        del episode_id
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        start = int(record["dataset_from_index"])
        file_starts[key] = min(start, file_starts.get(key, start))

    contracts = _load_effect_contracts(
        effect,
        manifest,
        effect_schema_version=effect_schema_version,
    )
    specs_by_file: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in manifest.to_dict(orient="records"):
        episode_id = int(item["episode_id"])
        record = episode_records.get(episode_id)
        if record is None:
            raise ValueError(f"effect manifest references unknown episode {episode_id}")
        if int(item["end_step"]) >= int(record["length"]):
            raise ValueError(f"effect window endpoint is outside episode {episode_id}")
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        specs_by_file[key].append({"effect": item, "episode": record})

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir()
    writer = _ShardWriter(
        temporary,
        shard_rows,
        schema_version=sidecar_schema_version,
        fields=action_fields,
    )
    source_inventory: list[dict[str, Any]] = []
    maximum_contract_error = 0.0
    try:
        for file_number, key in enumerate(sorted(specs_by_file), start=1):
            chunk, file_index = key
            path = source / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            table = parquet.read_table(
                path,
                columns=["observation.state", "action", "frame_index", "episode_index", "index"],
            )
            states = _list16(table.column("observation.state"), name="observation.state", path=path)
            actions = _list16(table.column("action"), name="action", path=path)
            global_indices = table.column("index").combine_chunks().to_numpy().astype(np.int64, copy=False)
            episode_indices = table.column("episode_index").combine_chunks().to_numpy().astype(np.int64, copy=False)
            frame_indices = table.column("frame_index").combine_chunks().to_numpy().astype(np.int64, copy=False)
            if len(states) != len(actions) or len(states) != len(global_indices):
                raise ValueError(f"source parquet columns have inconsistent lengths: {path}")
            relative = path.relative_to(source).as_posix()
            source_digest = _inventory_digest(commit, relative, path.stat().st_size)
            source_inventory.append(
                {"relative_path": relative, "bytes": path.stat().st_size, "inventory_sha256": source_digest}
            )
            file_start = file_starts[key]
            file_specs = sorted(
                specs_by_file[key],
                key=lambda item: (int(item["effect"]["episode_id"]), int(item["effect"]["start_step"])),
            )
            print(f"[action] {file_number}/{len(specs_by_file)} {path.name}: windows={len(file_specs)}", flush=True)
            for spec in file_specs:
                effect_row = spec["effect"]
                episode = spec["episode"]
                episode_id = int(effect_row["episode_id"])
                start = int(effect_row["start_step"])
                end = int(effect_row["end_step"])
                global_start = int(episode["dataset_from_index"]) + start
                local_start = global_start - file_start
                local_end = local_start + horizon
                if local_start < 0 or local_end >= len(states):
                    raise IndexError(f"window {effect_row['sample_id']} lies outside {path}")
                expected_global = np.arange(global_start, global_start + horizon + 1, dtype=np.int64)
                if not np.array_equal(global_indices[local_start : local_end + 1], expected_global):
                    raise ValueError(f"global source index mismatch for {effect_row['sample_id']}")
                if not np.all(episode_indices[local_start : local_end + 1] == episode_id):
                    raise ValueError(f"episode boundary crossed by {effect_row['sample_id']}")
                if not np.array_equal(
                    frame_indices[local_start : local_end + 1],
                    np.arange(start, end + 1, dtype=np.int64),
                ):
                    raise ValueError(f"frame index mismatch for {effect_row['sample_id']}")

                source_states = states[local_start : local_end + 1]
                source_actions = actions[local_start:local_end]
                state14_start, state14_end, delta_state14, state_mask, _ = dual_arm_contract(
                    source_states[0], source_states[-1]
                )
                contract = contracts[str(effect_row["sample_id"])]
                contract_error = max(
                    float(np.max(np.abs(state14_start - contract["state14_start"]))),
                    float(np.max(np.abs(state14_end - contract["state14_end"]))),
                    float(np.max(np.abs(delta_state14 - contract["delta_state14"]))),
                )
                maximum_contract_error = max(maximum_contract_error, contract_error)
                if contract_error > 2.0e-6 or not np.array_equal(state_mask, contract["state_mask"]):
                    raise ValueError(
                        f"source/effect state contract mismatch for {effect_row['sample_id']}: {contract_error}"
                    )

                waypoints = canonical_dual_arm_waypoints(source_states[0], source_states[1:])
                if waypoints.shape != (horizon, TRAJECTORY_DIM):
                    raise AssertionError(f"invalid canonical waypoint shape {waypoints.shape}")
                active = np.ones(horizon, dtype=np.uint8)
                arm_active = np.ones((horizon, 2), dtype=np.uint8)
                anchors = waypoints[np.asarray(anchor_steps, dtype=np.int64) - 1].copy()
                row: dict[str, Any] = {
                    "sample_id": str(effect_row["sample_id"]),
                    "group_id": str(effect_row["group_id"]),
                    "split": str(effect_row["split"]),
                    "task_id": str(effect_row["task_id"]),
                    "source_id": str(effect_row["source_id"]),
                    "source_commit": str(effect_row["source_commit"]),
                    "source_sha256": source_digest,
                    "controller_id": str(effect_row["controller_id"]),
                    "embodiment_id": "agilex",
                    "episode_id": episode_id,
                    "episode_seed": int(effect_row["episode_seed"]),
                    "start_step": start,
                    "end_step": end,
                    "control_hz": CONTROL_HZ,
                    "dual_arm_pose_start": source_states[0].copy(),
                    "dual_arm_pose_end": source_states[-1].copy(),
                    "waypoints": waypoints,
                    "time_mask": active,
                    "arm_pose_mask": arm_active.copy(),
                    "arm_gripper_mask": arm_active.copy(),
                    "native_actions": source_actions.copy(),
                    "native_action_mask": np.ones((horizon, NATIVE_ACTION_DIM), dtype=np.uint8),
                    "current_state": contract["state14_start"].copy(),
                    "current_state_mask": contract["state_mask"].copy(),
                    "e39_mean": np.zeros(EFFECT_DIM, dtype=np.float32),
                    "e39_valid": 0,
                    "physical_anchors": anchors,
                    "physical_anchor_mask": np.ones_like(anchors, dtype=np.uint8),
                }
                row["window_sha256"] = _window_digest(row)
                writer.add(row, effect_row)
        writer.finish()
        if len(writer.manifest) != len(manifest):
            raise RuntimeError(f"action/effect row count mismatch: {len(writer.manifest)} vs {len(manifest)}")
        _write_manifests(temporary, writer.manifest)
        normalization = normalization_from_train_shards(
            temporary,
            temporary_dir=temporary / ".normalization-workspace",
            horizon=horizon,
            schema_version=normalization_schema_version,
        )
        write_normalization(temporary / "stats/normalization.json", normalization)
        _atomic_json(temporary / "schema.json", schema_document(horizon=horizon))

        split_counts = Counter(str(row["split"]) for row in writer.manifest)
        family_counts = Counter(str(row["effect_family"]) for row in writer.manifest)
        coverage = {
            "schema": "robotwin-action-coverage-v1",
            "valid": True,
            "rows": len(writer.manifest),
            "split_counts": dict(sorted(split_counts.items())),
            "families": len(family_counts),
            "family_counts": dict(sorted(family_counts.items())),
            "horizon_actions": horizon,
            "control_hz": CONTROL_HZ,
            "duration_seconds": horizon / CONTROL_HZ,
            "e39_valid_rows": 0,
            "effect_exact_set": set(manifest.sample_id.astype(str)) == {row["sample_id"] for row in writer.manifest},
            "maximum_source_effect_state_error": maximum_contract_error,
        }
        _atomic_json(temporary / "coverage_report.json", coverage)
        provenance = {
            "schema": "robotwin-action-provenance-v1",
            "source_root": str(source),
            "source_dataset": "jialei02/robotwin_merged",
            "source_commit": commit,
            "source_info": {
                "total_episodes": info["total_episodes"],
                "total_frames": info["total_frames"],
                "fps": info["fps"],
            },
            "source_inventory": source_inventory,
            "source_inventory_note": "inventory_sha256 binds source commit, relative path, and byte size",
            "effect_root": str(effect),
            "effect_manifest_sha256": _sha256_file(effect / "manifests/all.parquet"),
            "effect_checksums_sha256": _sha256_file(effect / "checksums.sha256"),
            "effect_schema_sha256": _sha256_file(effect / "schema.json"),
            "transition_alignment": "observation.state[t] -> action[t] -> observation.state[t+1]",
            "trajectory_target": "actual post-action dual-arm state trajectory in start-frame coordinates",
            "split_ownership": "inherited exactly from robotwin_effect_v1; episode-disjoint and family-stratified",
            "sidecar_schema_version": sidecar_schema_version,
            "normalization_schema_version": normalization["schema_version"],
            "e39": None,
        }
        _atomic_json(temporary / "provenance.json", provenance)
        (temporary / "_SUCCESS").write_text("robotwin-actionvae-action-a0-release-v1\n", encoding="utf-8")
        _write_checksums(temporary)
        os.replace(temporary, destination)
        return {**coverage, "output": str(destination)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--effect-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-rows", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    if args.shard_rows < 1:
        parser.error("--shard-rows must be positive")
    if args.max_rows is not None and args.max_rows < 3:
        parser.error("--max-rows must be at least three")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_release(
        args.source_root,
        args.effect_root,
        args.output,
        shard_rows=args.shard_rows,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
