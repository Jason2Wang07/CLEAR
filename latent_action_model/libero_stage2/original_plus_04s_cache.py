"""Build the jointly normalized Original+LIBERO-Plus 0.4s Stage-2 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from .baseline_04s_cache import (
    ACTIONVAE_12S_REPRESENTATION_STEPS,
    CACHE_SCHEMA_VERSION as BASELINE_CACHE_SCHEMA_VERSION,
    CANONICAL_WAYPOINT_DIM,
    CONTROL_HZ,
    DINO_FUTURE_DELTA_STEPS,
    EFFECT_TEACHER_DELTA_STEPS,
    HORIZON_SECONDS,
    LATENT_DIM,
    LIBERO_04S_CONTROL_STEPS,
    LIBERO_12S_ACTION_CHECKPOINT,
    LIBERO_12S_ACTION_CHECKPOINT_SHA256,
    LIBERO_12S_ACTION_SCHEMA_VERSION,
    LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
    MergedLibero04sSource,
    SOURCE_DATASET_NAME as ORIGINAL_DATASET_NAME,
    SOURCE_EPISODE_COUNT as ORIGINAL_EPISODE_COUNT,
    SOURCE_ROW_COUNT as ORIGINAL_ROW_COUNT,
    VALIDATION_ROW_COUNT as ORIGINAL_VALIDATION_ROW_COUNT,
    Baseline04sTeacherCache,
    _copy_rank_shards,
    _distributed_context,
    _encode_rank_shard,
    _json_read,
    _json_write_atomic,
    _load_teachers,
    _sha256_json,
    _split_range,
    _teacher_paths_and_hashes,
    sha256_file,
    source_release_fingerprint,
)
from .interventions import LatentStandardizer
from .runtime_contract import (
    LATENT_CACHE_SCHEMA_VERSION_ORIGINAL_PLUS_04S,
    LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_ORIGINAL_PLUS_04S,
    LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT,
    LIBERO_ORIGINAL_PLUS_04S_ROW_COUNT,
)


CACHE_SCHEMA_VERSION = LATENT_CACHE_SCHEMA_VERSION_ORIGINAL_PLUS_04S
NORMALIZATION_SCHEMA_VERSION = (
    LATENT_NORMALIZATION_BUNDLE_SCHEMA_VERSION_ORIGINAL_PLUS_04S
)
TEMPORAL_CONTRACT = LIBERO_ORIGINAL_PLUS_04S_ADAPTER_TEMPORAL_CONTRACT
PLUS_DATASET_NAME = "libero_plus_lerobot"
PLUS_ROW_COUNT = 2_238_036
PLUS_EPISODE_COUNT = 14_347
PLUS_TASK_COUNT = 40
PLUS_VALIDATION_EPISODE_COUNT = 143
PLUS_VALIDATION_ROW_COUNT = 22_981
JOINT_ROW_COUNT = LIBERO_ORIGINAL_PLUS_04S_ROW_COUNT
JOINT_EPISODE_COUNT = ORIGINAL_EPISODE_COUNT + PLUS_EPISODE_COUNT
JOINT_VALIDATION_ROW_COUNT = (
    ORIGINAL_VALIDATION_ROW_COUNT + PLUS_VALIDATION_ROW_COUNT
)
DEFAULT_ORIGINAL_ROOT = Path("/data/share/libero_merged_no_noops_20hz")
DEFAULT_PLUS_ROOT = Path(
    "/202609/202609/transfer_250010178/ActionVAE/datasets/libero_plus_lerobot"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/202609/202609/transfer_250010178/ActionVAE/stage2_cache/"
    "libero_original_plus_04s_v1/full-2511501"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def plus_release_fingerprint(root: str | Path) -> dict[str, Any]:
    source = Path(root).expanduser().resolve()
    required = [
        source / "meta/info.json",
        source / "meta/episodes.jsonl",
        source / "meta/episodes_stats.jsonl",
        source / "meta/tasks.jsonl",
        source / "meta/modality.json",
        source / "meta/stats_gr00t.json",
    ]
    metadata_files = [path for path in required if path.is_file()]
    parquet_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    video_files = sorted((source / "videos").glob("chunk-*/*/*.mp4"))
    missing_metadata = [str(path) for path in required if not path.is_file()]
    observed = {
        "metadata": len(metadata_files),
        "parquet": len(parquet_files),
        "video": len(video_files),
    }
    expected = {
        "metadata": len(required),
        "parquet": PLUS_EPISODE_COUNT,
        "video": 2 * PLUS_EPISODE_COUNT,
    }
    if observed != expected:
        raise FileNotFoundError(
            "incomplete LIBERO-Plus release while fingerprinting: "
            f"observed={observed}, expected={expected}, "
            f"missing_metadata={missing_metadata}"
        )
    files = [*metadata_files, *parquet_files, *video_files]
    aggregate = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(source).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": "libero-plus-release-fingerprint-v1",
        "root": str(source),
        "file_count": len(entries),
        "files": entries,
        "aggregate_sha256": aggregate.hexdigest(),
    }


class LiberoPlus04sSource(MergedLibero04sSource):
    """LIBERO-Plus v2.1 source using the same physical 0.4s bridge."""

    def __init__(self, root: str | Path, *, max_samples: int | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = PLUS_DATASET_NAME
        info = _json_read(self.root / "meta/info.json")
        expected_info = {
            "codebase_version": "v2.1",
            "total_frames": PLUS_ROW_COUNT,
            "total_episodes": PLUS_EPISODE_COUNT,
            "total_tasks": PLUS_TASK_COUNT,
            "fps": int(CONTROL_HZ),
            "chunks_size": 1000,
        }
        mismatches = {
            name: (info.get(name), expected)
            for name, expected in expected_info.items()
            if info.get(name) != expected
        }
        if self.root.name != PLUS_DATASET_NAME or mismatches:
            raise ValueError(
                f"LIBERO-Plus source contract mismatch: root={self.root.name!r}, "
                f"fields={mismatches}"
            )
        self._chunk_size = int(info["chunks_size"])

        task_rows = _read_jsonl(self.root / "meta/tasks.jsonl")
        self.task_text = {
            int(row["task_index"]): str(row["task"]) for row in task_rows
        }
        task_by_text = {text: index for index, text in self.task_text.items()}
        if len(self.task_text) != PLUS_TASK_COUNT or len(task_by_text) != PLUS_TASK_COUNT:
            raise ValueError("LIBERO-Plus task metadata is not a one-to-one 40-task map")

        episode_records = sorted(
            _read_jsonl(self.root / "meta/episodes.jsonl"),
            key=lambda row: int(row["episode_index"]),
        )
        if [int(row["episode_index"]) for row in episode_records] != list(
            range(PLUS_EPISODE_COUNT)
        ):
            raise ValueError("LIBERO-Plus episode indices must be contiguous")
        lengths = np.asarray([int(row["length"]) for row in episode_records], dtype=np.int64)
        if bool((lengths <= 0).any()) or int(lengths.sum()) != PLUS_ROW_COUNT:
            raise ValueError("LIBERO-Plus episode lengths do not cover total_frames")
        starts = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
        ends = starts + lengths
        episodes = []
        for row, start, end in zip(episode_records, starts, ends):
            tasks = row.get("tasks")
            if not isinstance(tasks, list) or len(tasks) != 1 or str(tasks[0]) not in task_by_text:
                raise ValueError(
                    f"LIBERO-Plus episode {row['episode_index']} has invalid task metadata"
                )
            episodes.append(
                {
                    **row,
                    "dataset_from_index": int(start),
                    "dataset_to_index": int(end),
                    "task_index": int(task_by_text[str(tasks[0])]),
                }
            )
        self.episodes = pd.DataFrame(episodes)
        self._episode_by_id = {
            int(row.episode_index): row for row in self.episodes.itertuples(index=False)
        }
        self._episode_record_by_id = {
            int(row["episode_index"]): row for row in episodes
        }

        self.states = np.empty((PLUS_ROW_COUNT, 8), dtype=np.float32)
        self.actions = np.empty((PLUS_ROW_COUNT, 7), dtype=np.float32)
        self.timestamps = np.empty(PLUS_ROW_COUNT, dtype=np.float64)
        self.frame_indices = np.empty(PLUS_ROW_COUNT, dtype=np.int64)
        self.episode_indices = np.empty(PLUS_ROW_COUNT, dtype=np.int64)
        self.task_indices = np.empty(PLUS_ROW_COUNT, dtype=np.int64)
        observed_global_indices = np.empty(PLUS_ROW_COUNT, dtype=np.int64)
        for record in episodes:
            episode_id = int(record["episode_index"])
            start = int(record["dataset_from_index"])
            end = int(record["dataset_to_index"])
            path = self.root / (
                f"data/chunk-{episode_id // self._chunk_size:03d}/"
                f"episode_{episode_id:06d}.parquet"
            )
            frame = pd.read_parquet(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ],
            )
            if len(frame) != end - start:
                raise ValueError(f"LIBERO-Plus episode length mismatch: {episode_id}")
            self.states[start:end] = np.stack(frame["observation.state"].to_numpy())
            self.actions[start:end] = np.stack(frame["action"].to_numpy())
            self.timestamps[start:end] = frame["timestamp"].to_numpy(dtype=np.float64)
            self.frame_indices[start:end] = frame["frame_index"].to_numpy(dtype=np.int64)
            self.episode_indices[start:end] = frame["episode_index"].to_numpy(dtype=np.int64)
            self.task_indices[start:end] = frame["task_index"].to_numpy(dtype=np.int64)
            observed_global_indices[start:end] = frame["index"].to_numpy(dtype=np.int64)
            if not np.array_equal(
                self.frame_indices[start:end], np.arange(end - start, dtype=np.int64)
            ):
                raise ValueError(f"LIBERO-Plus frame_index mismatch: {episode_id}")
            if not np.all(self.episode_indices[start:end] == episode_id):
                raise ValueError(f"LIBERO-Plus episode_index mismatch: {episode_id}")
            if not np.all(self.task_indices[start:end] == int(record["task_index"])):
                raise ValueError(f"LIBERO-Plus task_index mismatch: {episode_id}")
        if not np.array_equal(observed_global_indices, np.arange(PLUS_ROW_COUNT)):
            raise ValueError("LIBERO-Plus global index must be contiguous and row aligned")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("LIBERO-Plus state/action contains NaN/Inf")

        raw_rotations = self.states[:, 3:6].astype(np.float64)
        raw_norm = np.linalg.norm(raw_rotations, axis=1)
        raw_rotation = Rotation.from_rotvec(raw_rotations)
        canonical_rotation = Rotation.from_rotvec(raw_rotation.as_rotvec())
        equivalence_error = (
            raw_rotation.inv() * canonical_rotation
        ).magnitude()
        max_error = float(np.max(equivalence_error))
        if max_error > 1e-10:
            raise ValueError(
                "LIBERO-Plus principal-rotvec canonicalization changed orientation: "
                f"max_error={max_error}"
            )
        self.canonical_rotvec = canonical_rotation.as_rotvec().astype(np.float32)
        self.canonicalization = {
            "rule": "principal rotation vector preserving physical orientation",
            "raw_norm_gt_pi_count": int(np.sum(raw_norm > np.pi + 1e-5)),
            "raw_row_count": int(len(raw_norm)),
            "raw_norm_max": float(np.max(raw_norm)),
            "canonical_norm_max": float(
                np.max(np.linalg.norm(self.canonical_rotvec.astype(np.float64), axis=1))
            ),
            "max_rotation_equivalence_error_rad": max_error,
            "equivalence_gate_rad": 1e-10,
        }

        if max_samples is not None and max_samples < 2:
            raise ValueError("max_samples must be at least two")
        self.selected_count = min(
            PLUS_ROW_COUNT,
            PLUS_ROW_COUNT if max_samples is None else int(max_samples),
        )
        self.release_level = "full" if self.selected_count == PLUS_ROW_COUNT else "smoke"
        validation_ids = self.episodes.tail(PLUS_VALIDATION_EPISODE_COUNT)[
            "episode_index"
        ].astype(int)
        self.validation_episode_ids = set(validation_ids.tolist())
        validation_rows = int(
            self.episodes.tail(PLUS_VALIDATION_EPISODE_COUNT)["length"].sum()
        )
        if validation_rows != PLUS_VALIDATION_ROW_COUNT:
            raise ValueError("LIBERO-Plus validation tail changed")

    def decode_effect_pairs(self, source_indices: Sequence[int]) -> np.ndarray:
        from starVLA.dataloader.gr00t_lerobot.video import (
            get_frames_by_timestamps_single_seek,
        )

        result = np.empty((len(source_indices), 2, 256, 256, 3), dtype=np.uint8)
        grouped: dict[Path, list[tuple[int, float, float]]] = {}
        for offset, raw_index in enumerate(source_indices):
            source_index = int(raw_index)
            row, start_step = self._episode_location(source_index)
            tail = int(row.length) - 1
            effect_step = min(start_step + EFFECT_TEACHER_DELTA_STEPS, tail)
            episode_id = int(row.episode_index)
            path = self.root / (
                f"videos/chunk-{episode_id // self._chunk_size:03d}/"
                f"observation.images.front/episode_{episode_id:06d}.mp4"
            )
            grouped.setdefault(path, []).append(
                (offset, start_step / CONTROL_HZ, effect_step / CONTROL_HZ)
            )
        for path, requests in grouped.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            timestamps = np.asarray(
                [value for _, start, end in requests for value in (start, end)],
                dtype=np.float64,
            )
            decoded = get_frames_by_timestamps_single_seek(
                str(path), timestamps, video_backend="pyav"
            )
            if decoded.shape != (len(requests) * 2, 256, 256, 3):
                raise ValueError(f"decoded frame shape mismatch for {path}: {decoded.shape}")
            decoded = decoded.astype(np.uint8, copy=False).reshape(
                len(requests), 2, 256, 256, 3
            )
            for request_index, (offset, _, _) in enumerate(requests):
                result[offset] = decoded[request_index]
        return result


class OriginalPlus04sSource:
    """One row space over Original followed by Plus, with source-qualified identity."""

    def __init__(
        self,
        original_root: str | Path,
        plus_root: str | Path,
        *,
        max_samples_per_source: int | None = None,
    ) -> None:
        self.original = MergedLibero04sSource(
            original_root, max_samples=max_samples_per_source
        )
        self.plus = LiberoPlus04sSource(
            plus_root, max_samples=max_samples_per_source
        )
        self.sources = (self.original, self.plus)
        self.offsets = (0, self.original.selected_count)
        self.selected_count = sum(source.selected_count for source in self.sources)
        self.release_level = (
            "full"
            if all(source.release_level == "full" for source in self.sources)
            else "smoke"
        )
        self.root = self.plus.root.parent
        self.canonicalization = {
            source.dataset_name: source.canonicalization for source in self.sources
        }

    def _partition(
        self, source_indices: Sequence[int]
    ) -> list[tuple[MergedLibero04sSource, np.ndarray, np.ndarray]]:
        values = np.asarray(source_indices, dtype=np.int64)
        if values.ndim != 1 or bool((values < 0).any()) or bool(
            (values >= self.selected_count).any()
        ):
            raise IndexError("joint source index is out of range")
        result = []
        for source, offset in zip(self.sources, self.offsets):
            positions = np.flatnonzero(
                (values >= offset) & (values < offset + source.selected_count)
            )
            if len(positions):
                result.append((source, positions, values[positions] - offset))
        return result

    def structural_batch(self, source_indices: Sequence[int], normalization: Any) -> dict[str, np.ndarray]:
        count = len(source_indices)
        output: dict[str, np.ndarray] = {}
        for source, positions, local_indices in self._partition(source_indices):
            values = source.structural_batch(local_indices.tolist(), normalization)
            for name, value in values.items():
                if name not in output:
                    output[name] = np.empty((count, *value.shape[1:]), dtype=value.dtype)
                output[name][positions] = value
        return output

    def decode_effect_pairs(self, source_indices: Sequence[int]) -> np.ndarray:
        output = np.empty((len(source_indices), 2, 256, 256, 3), dtype=np.uint8)
        for source, positions, local_indices in self._partition(source_indices):
            output[positions] = source.decode_effect_pairs(local_indices.tolist())
        return output

    def manifest(self) -> pd.DataFrame:
        frames = []
        for source, offset in zip(self.sources, self.offsets):
            frame = source.manifest().copy()
            frame["dataset_source_index"] = frame["source_index"].astype(np.int64)
            frame["combined_source_index"] = (
                frame["dataset_source_index"] + int(offset)
            )
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        result["latent_row"] = np.arange(len(result), dtype=np.int64)
        if result[["dataset_name", "episode_id", "start_step"]].duplicated().any():
            raise ValueError("joint Original+Plus identity is not unique")
        return result


def _temporal_metadata() -> dict[str, Any]:
    return {
        "stage2_temporal_contract": TEMPORAL_CONTRACT,
        "control_hz": CONTROL_HZ,
        "duration_seconds": HORIZON_SECONDS,
        "source_action_steps": LIBERO_04S_CONTROL_STEPS,
        "dino_future_delta_steps": DINO_FUTURE_DELTA_STEPS,
        "effect_teacher_delta_steps": EFFECT_TEACHER_DELTA_STEPS,
        "representation_steps": ACTIONVAE_12S_REPRESENTATION_STEPS,
        "physical_valid_count": LIBERO_04S_CONTROL_STEPS,
        "trajectory_representation": (
            "start-frame-relative canonical SE(3); position piecewise-linear, "
            "orientation shortest-path quaternion SLERP, gripper zero-order-hold"
        ),
        "tail_policy": "episode-local endpoint clamp",
    }


def _root_contract(
    *,
    source: OriginalPlus04sSource,
    source_fingerprints: Mapping[str, Any],
    teachers: Mapping[str, Any],
    world_size: int,
    precision: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage2_temporal_contract": TEMPORAL_CONTRACT,
        "source_row_count": JOINT_ROW_COUNT,
        "selected_row_count": source.selected_count,
        "release_level": source.release_level,
        "source_fingerprints": dict(source_fingerprints),
        "teachers": dict(teachers),
        "temporal_contract": _temporal_metadata(),
        "identity": ["dataset_name", "episode_id", "start_step"],
        "world_size": world_size,
        "precision": precision,
    }


def _publish_cache(
    *,
    source: OriginalPlus04sSource,
    output_root: Path,
    work_root: Path,
    contract: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    from . import robotwin_action_transfer as transfer

    transferred = transfer.checked_transfer(contract.get("action_teacher_transfer"))
    action_path = transfer.CHECKPOINT if transferred else LIBERO_12S_ACTION_CHECKPOINT
    action_sha = transfer.CHECKPOINT_SHA256 if transferred else LIBERO_12S_ACTION_CHECKPOINT_SHA256
    condition_sha = transfer.CONDITION_SHA256 if transferred else LIBERO_12S_EFFECT_CHECKPOINT_SHA256
    action_schema = transfer.SCHEMA if transferred else LIBERO_12S_ACTION_SCHEMA_VERSION
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable cache: {output_root}")
    publishing = output_root.parent / f".{output_root.name}.publishing"
    if publishing.exists():
        if not resume:
            raise FileExistsError(f"publishing directory exists; pass --resume: {publishing}")
        shutil.rmtree(publishing)
    publishing.mkdir(parents=True)
    row_count = source.selected_count
    contract_digest = _sha256_json(contract)
    h5_path = publishing / "latents.h5"
    try:
        with h5py.File(h5_path, "w") as output:
            if transferred:
                output.attrs["action_teacher_transfer"] = transfer.TRANSFER_ID
                output.attrs["condition_source"] = "lawam_idm"
            output.attrs["schema_version"] = CACHE_SCHEMA_VERSION
            output.attrs["stage2_temporal_contract"] = TEMPORAL_CONTRACT
            output.attrs["release_level"] = source.release_level
            output.attrs["complete"] = False
            chunk_rows = max(1, min(256, row_count))
            from .baseline_04s_cache import _RAW_SHARD_SPECS

            for name, (tail, dtype) in _RAW_SHARD_SPECS.items():
                output.create_dataset(
                    name,
                    shape=(row_count, *tail),
                    dtype=dtype,
                    chunks=(chunk_rows, *tail),
                )
            _copy_rank_shards(
                output,
                work_root=work_root,
                row_count=row_count,
                world_size=int(contract["world_size"]),
                contract_digest=contract_digest,
                schema_version=CACHE_SCHEMA_VERSION,
                temporal_contract=TEMPORAL_CONTRACT,
            )
            if not np.array_equal(output["global_row"][:], np.arange(row_count)):
                raise ValueError("merged cache global-row identity mismatch")
            effect_raw = output["z_effect_raw"][:]
            action_raw = output["z_action_mean_raw"][:]
            action_logvar_raw = output["z_action_logvar_raw"][:]
            effect_standardizer = LatentStandardizer.fit(effect_raw, split="train")
            action_standardizer = LatentStandardizer.fit(action_raw, split="train")
            z_effect = effect_standardizer.transform(effect_raw)
            z_action = action_standardizer.transform(action_raw)
            action_scale = np.maximum(
                np.asarray(action_standardizer.std, dtype=np.float32),
                np.float32(action_standardizer.epsilon),
            )
            z_action_logvar = action_logvar_raw - 2.0 * np.log(action_scale)
            for name, value in (
                ("z_effect", z_effect),
                ("z_action", z_action),
                ("z_action_mean", z_action),
                ("z_action_logvar", z_action_logvar),
                ("z_action_raw", action_raw),
            ):
                output.create_dataset(
                    name,
                    data=np.asarray(value, dtype=np.float32),
                    dtype=np.float32,
                    chunks=(chunk_rows, LATENT_DIM),
                )
            output.attrs["complete"] = True
            output.flush()

        manifest = source.manifest()
        manifest_path = publishing / "manifest.parquet"
        manifest.to_parquet(manifest_path, index=False)
        fit_membership = (
            "all_rows" if source.release_level == "full" else "selected_smoke_rows"
        )
        normalization = {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "stage2_temporal_contract": TEMPORAL_CONTRACT,
            "fit_split": "train",
            "fit_membership": fit_membership,
            "fit_row_count": row_count,
            "fit_dataset_row_counts": {
                name: int(count)
                for name, count in manifest["dataset_name"].value_counts().items()
            },
            "z_effect": effect_standardizer.to_dict(),
            "z_action": action_standardizer.to_dict(),
            "action_logvar_standardization": (
                "logvar_raw-2*log(max(action_std,epsilon))"
            ),
        }
        normalization_path = publishing / "normalization.json"
        _json_write_atomic(normalization_path, normalization)
        validation_count = int(manifest["validation_membership"].sum())
        provenance = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage2_temporal_contract": TEMPORAL_CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_level": source.release_level,
            "row_count": row_count,
            "train_membership_count": int(manifest["train_membership"].sum()),
            "validation_membership_count": validation_count,
            "identity": ["dataset_name", "episode_id", "start_step"],
            "source": {
                "dataset_name": "libero_all",
                "row_count": JOINT_ROW_COUNT,
                "episode_count": JOINT_EPISODE_COUNT,
                "dataset_row_counts": {
                    ORIGINAL_DATASET_NAME: ORIGINAL_ROW_COUNT,
                    PLUS_DATASET_NAME: PLUS_ROW_COUNT,
                },
                "dataset_episode_counts": {
                    ORIGINAL_DATASET_NAME: ORIGINAL_EPISODE_COUNT,
                    PLUS_DATASET_NAME: PLUS_EPISODE_COUNT,
                },
                "release_fingerprints": contract["source_fingerprints"],
                "temporal_contract": _temporal_metadata(),
                "orientation_canonicalization": source.canonicalization,
                "action_checkpoint": str(action_path.resolve()),
                "action_checkpoint_sha256": action_sha,
                "action_checkpoint_schema_version": action_schema,
                "action_prior": "standard_normal",
            },
            "teachers": contract["teachers"],
            "teacher_checkpoint_sha256": {
                "effect": condition_sha,
                "action": action_sha,
            },
            "action_teacher": {
                "schema_version": action_schema,
                "prior": "standard_normal",
                "variant": "a1",
                "source_horizon_steps": 36 if transferred else 24,
                "source_duration_seconds": 1.2,
            },
            "normalization": {
                "schema_version": NORMALIZATION_SCHEMA_VERSION,
                "fit_split": "train",
                "fit_membership": fit_membership,
                "fit_row_count": row_count,
                "scope": "joint_original_plus",
            },
            "artifacts": {
                "normalization": {
                    "path": "normalization.json",
                    "sha256": sha256_file(normalization_path),
                }
            },
            "action_distribution": {
                "latent_dim": LATENT_DIM,
                "parameter_dim": 2 * LATENT_DIM,
                "layout": "mean_then_logvar",
                "teacher_statistics": {
                    "action_scale_min": float(action_scale.min()),
                    "action_scale_max": float(action_scale.max()),
                    "standardized_logvar_min": float(np.min(z_action_logvar)),
                    "standardized_logvar_max": float(np.max(z_action_logvar)),
                },
            },
            "distributed": {
                "world_size": int(contract["world_size"]),
                "partition": "contiguous-balanced-global-row-ranges",
                "hdf5_write_policy": (
                    "one exclusive shard per rank; rank0 read-only merge"
                ),
            },
            "issue_count": 0,
            "ready": True,
        }
        provenance_path = publishing / "provenance.json"
        if transferred:
            provenance.update(action_teacher_transfer=transfer.TRANSFER_ID, condition_source="lawam_idm",
                              transfer_contract=contract["transfer_contract"])
            provenance["legacy_field_semantics"] = "z_effect fields store standardized official z_lawam posterior means"
        _json_write_atomic(provenance_path, provenance)
        artifact_names = (
            "latents.h5",
            "manifest.parquet",
            "normalization.json",
            "provenance.json",
        )
        checksums_path = publishing / "checksums.sha256"
        checksums_path.write_text(
            "".join(
                f"{sha256_file(publishing / name)}  {name}\n"
                for name in artifact_names
            ),
            encoding="ascii",
        )
        success = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage2_temporal_contract": TEMPORAL_CONTRACT,
            "release_level": source.release_level,
            "row_count": row_count,
            "train_membership_count": row_count,
            "validation_membership_count": validation_count,
            "checksums_sha256": sha256_file(checksums_path),
            "teacher_checkpoint_sha256": {
                "effect": condition_sha,
                "action": action_sha,
            },
            "all_gates_passed": True,
        }
        if transferred:
            success.update(action_teacher_transfer=transfer.TRANSFER_ID, condition_source="lawam_idm")
        _json_write_atomic(publishing / "_SUCCESS", success)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(publishing, output_root)
        shutil.rmtree(work_root, ignore_errors=True)
        return {"output_root": str(output_root), **success}
    except BaseException:
        shutil.rmtree(publishing, ignore_errors=True)
        raise


class OriginalPlus04sTeacherCache(Baseline04sTeacherCache):
    def __init__(self, cache_root: str | Path, *, require_full: bool = True,
                 action_teacher_transfer: str | None = None) -> None:
        super().__init__(
            cache_root,
            require_full=require_full,
            expected_schema_version=CACHE_SCHEMA_VERSION,
            expected_temporal_contract=TEMPORAL_CONTRACT,
            expected_normalization_schema=NORMALIZATION_SCHEMA_VERSION,
            expected_full_rows=JOINT_ROW_COUNT,
            expected_validation_rows=JOINT_VALIDATION_ROW_COUNT,
            action_teacher_transfer=action_teacher_transfer,
        )


def build_original_plus_04s_cache(
    *,
    original_root: str | Path = DEFAULT_ORIGINAL_ROOT,
    plus_root: str | Path = DEFAULT_PLUS_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 64,
    precision: str = "bf16",
    max_samples_per_source: int | None = None,
    resume: bool = False,
    progress_every: int = 10,
) -> dict[str, Any]:
    if batch_size < 1 or progress_every < 0:
        raise ValueError("batch_size must be positive and progress_every non-negative")
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16, or fp32")
    context = _distributed_context()
    output = Path(output_root).expanduser().resolve()
    work_root = output.parent / f".{output.name}.inprogress"
    if context.is_main:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable cache: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if work_root.exists() and not resume:
            raise FileExistsError(f"cache work directory exists; pass --resume: {work_root}")
        work_root.mkdir(parents=True, exist_ok=True)
    context.barrier()

    source = OriginalPlus04sSource(
        original_root,
        plus_root,
        max_samples_per_source=max_samples_per_source,
    )
    contract_path = work_root / "contract.json"
    if context.is_main and not contract_path.exists():
        fingerprints = {
            ORIGINAL_DATASET_NAME: source_release_fingerprint(source.original.root),
            PLUS_DATASET_NAME: plus_release_fingerprint(source.plus.root),
        }
        contract = _root_contract(
            source=source,
            source_fingerprints=fingerprints,
            teachers=_teacher_paths_and_hashes(),
            world_size=context.world_size,
            precision=precision,
        )
        _json_write_atomic(contract_path, contract)
    context.barrier()
    contract = _json_read(contract_path)
    expected_local = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage2_temporal_contract": TEMPORAL_CONTRACT,
        "source_row_count": JOINT_ROW_COUNT,
        "selected_row_count": source.selected_count,
        "release_level": source.release_level,
        "world_size": context.world_size,
        "precision": precision,
    }
    for name, expected in expected_local.items():
        if contract.get(name) != expected:
            raise ValueError(f"cache root/resume contract mismatch for {name}")
    device = torch.device(
        f"cuda:{context.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and precision != "fp32":
        raise RuntimeError("GPU cache extraction is required for bf16/fp16")
    shard_result = _encode_rank_shard(
        source=source,  # type: ignore[arg-type]
        work_root=work_root,
        contract=contract,
        rank=context.rank,
        world_size=context.world_size,
        batch_size=batch_size,
        device=device,
        precision=precision,
        resume=resume,
        progress_every=progress_every,
    )
    context.barrier()
    result: dict[str, Any] = {"rank_shard": shard_result}
    if context.is_main:
        result = _publish_cache(
            source=source,
            output_root=output,
            work_root=work_root,
            contract=contract,
            resume=resume,
        )
    context.barrier()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--plus-root", type=Path, default=DEFAULT_PLUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples-per-source", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_original_plus_04s_cache(
        original_root=args.original_root,
        plus_root=args.plus_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        precision=args.precision,
        max_samples_per_source=args.max_samples_per_source,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
