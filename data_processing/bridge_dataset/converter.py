"""Resumable primary-view Bridge conversion into paired effect/action releases."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import av
import h5py
import numpy as np
import pandas as pd

from lawam_effect_dataset.geometry import relative_pose
from lawam_effect_dataset.shards import write_shard_atomic as write_effect_shard
from lawam_effect_dataset.windows import group_id, stable_id
from z_action_sidecar.io import write_shard_atomic as write_action_shard
from z_action_sidecar.schema import E39_DIM
from z_action_sidecar.statistics import robust_group

from .geometry import bridge_window_geometry
from .schema import (
    EFFECT_SCHEMA_VERSION,
    MAX_NATIVE_ACTION_DIM,
    NORMALIZATION_SCHEMA_VERSION,
    PHYSICAL_ACTION_STEPS,
    RELEASE_SCHEMA_VERSION,
    REPRESENTATION_HZ,
    REPRESENTATION_STEPS,
    SIDECAR_SCHEMA_VERSION,
    SOURCE_CONTROL_HZ,
    STAGE2_MANIFEST_SCHEMA_VERSION,
    TAU_SECONDS,
    action_fields,
    effect_schema_document,
    sidecar_schema_document,
    stage2_modality_document,
)
from .source import BridgeEpisode, SOURCE_DATASET_ID, SOURCE_REVISION, audit_source, load_source


SPLITS = ("train", "validation", "test")
CONTROLLER_ID = "bridge_delta_xyz_delta_rpy_absolute_gripper"
ANCHOR_INDICES = np.asarray([5, 11, 17, 23], dtype=np.int64)
NATIVE_REPRESENTATION_INDICES = np.asarray([3, 7, 11, 15, 19, 23], dtype=np.int64)

_SOURCE_STATES: np.ndarray | None = None
_SOURCE_ACTIONS: np.ndarray | None = None


def _sha256_file(path: str | Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _window_digest(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    identity = {
        name: row[name]
        for name in (
            "sample_id",
            "source_id",
            "source_commit",
            "source_sha256",
            "episode_id",
            "start_step",
            "end_step",
            "controller_id",
            "physical_valid_count",
        )
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in (
        "waypoints",
        "time_mask",
        "pose_mask",
        "gripper_mask",
        "native_actions",
        "native_action_mask",
        "current_state",
        "current_state_mask",
        "e39_mean",
    ):
        value = np.ascontiguousarray(row[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _identity_pose() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _effect_row(
    episode: BridgeEpisode,
    *,
    start: int,
    geometry: Any,
    rgb_start: np.ndarray,
    rgb_end: np.ndarray,
    video_sha256: str,
) -> dict[str, Any]:
    end = int(start) + PHYSICAL_ACTION_STEPS
    source_id = "bridge_orig_lerobot"
    sample_id = stable_id(source_id, episode.episode_id, int(start), end, version="bridge-v1-12s-s2")
    task_id = f"bridge/task-{episode.task_index:05d}"
    identity = _identity_pose()
    zeros_pose = np.zeros((3, 7), dtype=np.float32)
    image_start = np.asarray(rgb_start, dtype=np.uint8)
    image_end = np.asarray(rgb_end, dtype=np.uint8)
    if image_start.shape != (256, 256, 3) or image_end.shape != (256, 256, 3):
        raise ValueError("Bridge primary images must be uint8 [256,256,3]")
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "effect_family": geometry.effect_family,
        "source_id": source_id,
        "source_commit": SOURCE_REVISION,
        "controller_id": CONTROLLER_ID,
        "episode_id": episode.episode_id,
        "episode_seed": episode.episode_id,
        "start_step": int(start),
        "end_step": end,
        "control_hz": SOURCE_CONTROL_HZ,
        "tau": TAU_SECONDS,
        "split": episode.split,
        "rgb_start": np.transpose(image_start, (2, 0, 1)).copy(),
        "rgb_end": np.transpose(image_end, (2, 0, 1)).copy(),
        "target_seg_start": np.zeros((256, 256), dtype=np.uint32),
        "target_seg_end": np.zeros((256, 256), dtype=np.uint32),
        "object_pose_start": identity.copy(),
        "object_pose_end": identity.copy(),
        "reference_pose_start": identity.copy(),
        "reference_pose_end": identity.copy(),
        "active_object_id": "",
        "aux_object_poses_start": zeros_pose.copy(),
        "aux_object_poses_end": zeros_pose.copy(),
        "aux_object_mask": np.zeros(3, dtype=np.uint8),
        "effect_raw": geometry.effect_raw,
        "effect_mask": geometry.effect_mask,
        "effect_valid": 1,
        "effect_direction": geometry.effect_direction,
        "effect_magnitude": geometry.effect_magnitude,
        "eef_pose_start": geometry.poses[0],
        "eef_pose_end": geometry.poses[-1],
        "gripper_start": float(geometry.widths[0]),
        "gripper_end": float(geometry.widths[-1]),
        "state14_start": geometry.state14_start,
        "state14_end": geometry.state14_end,
        "delta_state14": geometry.delta_state14,
        "state_mask": geometry.state_mask,
        "embodiment_id": "widowx",
        "contact_summary": np.zeros(4, dtype=np.float32),
        "grasp_summary": np.zeros(4, dtype=np.float32),
        "phase": "demonstration",
        "outcome": "success",
        "quality": 1.0,
        "replay_valid": 0,
        "render_valid": 1,
        "source_sha256": video_sha256,
        "group_id": group_id(source_id, episode.episode_id),
        "language_instruction": episode.instruction,
        "teacher_compatibility": geometry.teacher_compatibility,
        "source_data_index": episode.data_from + int(start),
    }


def _action_row(
    effect: Mapping[str, Any],
    *,
    geometry: Any,
    source_actions: np.ndarray,
) -> dict[str, Any]:
    native = np.zeros((REPRESENTATION_STEPS, MAX_NATIVE_ACTION_DIM), dtype=np.float32)
    native_mask = np.zeros_like(native, dtype=np.uint8)
    native[NATIVE_REPRESENTATION_INDICES, :7] = np.asarray(source_actions, dtype=np.float32)
    native_mask[NATIVE_REPRESENTATION_INDICES, :7] = 1
    active = np.ones(REPRESENTATION_STEPS, dtype=np.uint8)
    inactive = np.zeros(REPRESENTATION_STEPS, dtype=np.uint8)
    anchors = np.zeros((4, 8), dtype=np.float32)
    anchors[:, :7] = geometry.waypoints[ANCHOR_INDICES]
    anchor_mask = np.zeros_like(anchors, dtype=np.uint8)
    anchor_mask[:, :7] = 1
    row: dict[str, Any] = {
        "sample_id": str(effect["sample_id"]),
        "group_id": str(effect["group_id"]),
        "split": str(effect["split"]),
        "task_id": str(effect["task_id"]),
        "source_id": str(effect["source_id"]),
        "source_commit": str(effect["source_commit"]),
        "source_sha256": str(effect["source_sha256"]),
        "controller_id": str(effect["controller_id"]),
        "embodiment_id": "widowx",
        "episode_id": int(effect["episode_id"]),
        "episode_seed": int(effect["episode_seed"]),
        "start_step": int(effect["start_step"]),
        "end_step": int(effect["end_step"]),
        "control_hz": SOURCE_CONTROL_HZ,
        "eef_pose_start": np.asarray(effect["eef_pose_start"], dtype=np.float32),
        "eef_pose_end": np.asarray(effect["eef_pose_end"], dtype=np.float32),
        "waypoints": np.asarray(geometry.waypoints, dtype=np.float32),
        "time_mask": active.copy(),
        "pose_mask": active.copy(),
        "gripper_mask": active.copy(),
        "contact": np.zeros(REPRESENTATION_STEPS, dtype=np.float32),
        "contact_peak_force": np.zeros(REPRESENTATION_STEPS, dtype=np.float32),
        "contact_impulse": np.zeros((REPRESENTATION_STEPS, 6), dtype=np.float32),
        "contact_mask": inactive.copy(),
        "grasp": np.zeros(REPRESENTATION_STEPS, dtype=np.float32),
        "grasp_mask": inactive.copy(),
        "native_actions": native,
        "native_action_mask": native_mask,
        "native_action_dim": 7,
        "current_state": np.asarray(effect["state14_start"], dtype=np.float32),
        "current_state_mask": np.asarray(effect["state_mask"], dtype=np.uint8),
        "e39_mean": np.zeros(E39_DIM, dtype=np.float32),
        "e39_valid": 0,
        "physical_anchors": anchors,
        "physical_anchor_mask": anchor_mask,
        "replay_object_error": np.zeros(REPRESENTATION_STEPS, dtype=np.float32),
        "replay_robot_error": np.zeros(REPRESENTATION_STEPS, dtype=np.float32),
        "physical_valid_count": PHYSICAL_ACTION_STEPS,
    }
    row["window_sha256"] = _window_digest(row)
    return row


class _VideoShardWriter:
    def __init__(self, root: Path, *, video_file_index: int, shard_rows: int) -> None:
        self.root = root
        self.video_file_index = int(video_file_index)
        self.shard_rows = int(shard_rows)
        self.buffers: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
            split: [] for split in SPLITS
        }
        self.indices = {split: 0 for split in SPLITS}
        self.effect_manifest: list[dict[str, Any]] = []
        self.action_manifest: list[dict[str, Any]] = []
        self.shard_checksums: dict[str, str] = {}
        self.train_stats: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, effect: dict[str, Any], action: dict[str, Any]) -> None:
        split = str(effect["split"])
        self.buffers[split].append((effect, action))
        if split == "train":
            for name, value in (
                ("effect_delta", effect["delta_state14"]),
                ("effect_mask", effect["state_mask"]),
                ("waypoints", action["waypoints"]),
                ("time_mask", action["time_mask"]),
                ("pose_mask", action["pose_mask"]),
                ("gripper_mask", action["gripper_mask"]),
                ("current_state", action["current_state"]),
                ("current_state_mask", action["current_state_mask"]),
            ):
                self.train_stats[name].append(np.asarray(value))
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    @staticmethod
    def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: row[name]
            for name in (
                "sample_id",
                "group_id",
                "split",
                "task_id",
                "source_id",
                "source_commit",
                "controller_id",
                "episode_id",
                "episode_seed",
                "start_step",
                "end_step",
                "control_hz",
            )
        }

    def flush(self, split: str) -> None:
        paired = self.buffers[split]
        if not paired:
            return
        index = self.indices[split]
        filename = f"part-v{self.video_file_index:03d}-{index:04d}.h5"
        relative = Path("shards") / split / filename
        effect_path = self.root / "effect" / relative
        action_path = self.root / "action" / relative
        effects = [item[0] for item in paired]
        actions = [item[1] for item in paired]
        write_effect_shard(effect_path, effects, schema_version=EFFECT_SCHEMA_VERSION)
        write_action_shard(
            action_path,
            actions,
            schema_version=SIDECAR_SCHEMA_VERSION,
            fields=action_fields(),
        )
        self.shard_checksums[effect_path.relative_to(self.root).as_posix()] = _sha256_file(effect_path)
        self.shard_checksums[action_path.relative_to(self.root).as_posix()] = _sha256_file(action_path)
        for shard_row, (effect, action) in enumerate(paired):
            effect_manifest = self._identity(effect)
            effect_manifest.update(
                {
                    "effect_family": effect["effect_family"],
                    "teacher_compatibility": effect["teacher_compatibility"],
                    "effect_direction": effect["effect_direction"],
                    "effect_magnitude": effect["effect_magnitude"],
                    "view": "primary_view",
                    "tau": effect["tau"],
                    "language_instruction": effect["language_instruction"],
                    "suite": "bridge",
                    "source_data_index": effect["source_data_index"],
                    "shard_path": relative.as_posix(),
                    "shard_row": shard_row,
                }
            )
            self.effect_manifest.append(effect_manifest)
            action_manifest = self._identity(action)
            action_manifest.update(
                {
                    "effect_family": effect["effect_family"],
                    "teacher_compatibility": effect["teacher_compatibility"],
                    "language_instruction": effect["language_instruction"],
                    "suite": "bridge",
                    "source_data_index": effect["source_data_index"],
                    "effect_shard_path": (Path("../effect") / relative).as_posix(),
                    "effect_shard_row": shard_row,
                    "shard_path": relative.as_posix(),
                    "shard_row": shard_row,
                    "window_sha256": action["window_sha256"],
                    "e39_valid": False,
                    "physical_valid_count": PHYSICAL_ACTION_STEPS,
                }
            )
            self.action_manifest.append(action_manifest)
        self.buffers[split] = []
        self.indices[split] += 1

    def finish(self, parts_root: Path) -> tuple[Path, Path, Path]:
        for split in SPLITS:
            self.flush(split)
        effect_manifest = parts_root / "manifests" / f"effect-v{self.video_file_index:03d}.parquet"
        action_manifest = parts_root / "manifests" / f"action-v{self.video_file_index:03d}.parquet"
        stats_path = parts_root / "stats" / f"train-v{self.video_file_index:03d}.npz"
        _atomic_parquet(effect_manifest, pd.DataFrame(self.effect_manifest))
        _atomic_parquet(action_manifest, pd.DataFrame(self.action_manifest))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = stats_path.with_name(f".{stats_path.name}.{os.getpid()}.tmp")
        arrays = {
            name: np.stack(values)
            for name, values in self.train_stats.items()
            if values
        }
        if not arrays:
            arrays = {
                "effect_delta": np.empty((0, 14), dtype=np.float32),
                "effect_mask": np.empty((0, 14), dtype=np.uint8),
                "waypoints": np.empty((0, REPRESENTATION_STEPS, 7), dtype=np.float32),
                "time_mask": np.empty((0, REPRESENTATION_STEPS), dtype=np.uint8),
                "pose_mask": np.empty((0, REPRESENTATION_STEPS), dtype=np.uint8),
                "gripper_mask": np.empty((0, REPRESENTATION_STEPS), dtype=np.uint8),
                "current_state": np.empty((0, 14), dtype=np.float32),
                "current_state_mask": np.empty((0, 14), dtype=np.uint8),
            }
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            os.replace(temporary, stats_path)
        finally:
            temporary.unlink(missing_ok=True)
        return effect_manifest, action_manifest, stats_path


def _require_worker_source() -> tuple[np.ndarray, np.ndarray]:
    if _SOURCE_STATES is None or _SOURCE_ACTIONS is None:
        raise RuntimeError("Bridge worker source arrays were not initialized")
    return _SOURCE_STATES, _SOURCE_ACTIONS


def _convert_episode(
    episode: BridgeEpisode,
    frames: Sequence[np.ndarray],
    *,
    video_sha256: str,
    window_stride: int,
    writer: _VideoShardWriter,
) -> int:
    states, actions = _require_worker_source()
    if len(frames) != episode.length:
        raise ValueError(
            f"episode {episode.episode_id} decoded {len(frames)} frames, expected {episode.length}"
        )
    count = 0
    for start in range(0, episode.length - PHYSICAL_ACTION_STEPS, int(window_stride)):
        absolute = episode.data_from + start
        raw_states = states[absolute : absolute + PHYSICAL_ACTION_STEPS + 1]
        raw_actions = actions[absolute : absolute + PHYSICAL_ACTION_STEPS]
        geometry = bridge_window_geometry(raw_states)
        effect = _effect_row(
            episode,
            start=start,
            geometry=geometry,
            rgb_start=frames[start],
            rgb_end=frames[start + PHYSICAL_ACTION_STEPS],
            video_sha256=video_sha256,
        )
        action = _action_row(effect, geometry=geometry, source_actions=raw_actions)
        writer.add(effect, action)
        count += 1
    return count


def _convert_video_file(
    source_root: str,
    work_root: str,
    video_file_index: int,
    episodes: Sequence[BridgeEpisode],
    window_stride: int,
    shard_rows: int,
) -> dict[str, Any]:
    root = Path(source_root)
    work = Path(work_root)
    parts = work / ".parts"
    marker = parts / "markers" / f"video-{int(video_file_index):03d}.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        required = [work / value for value in payload.get("required_files", [])]
        if payload.get("complete") is True and all(path.is_file() for path in required):
            return payload
        raise ValueError(f"invalid resumable Bridge part marker: {marker}")

    # A worker can be preempted after publishing some immutable shards but
    # before writing its completion marker. Video indices are unique owners,
    # so remove only this unfinished video's outputs before retrying it.
    part_stem = f"part-v{int(video_file_index):03d}-"
    incomplete_paths = [
        *(work / "effect" / "shards").glob(f"*/{part_stem}*.h5"),
        *(work / "action" / "shards").glob(f"*/{part_stem}*.h5"),
        parts / "manifests" / f"effect-v{int(video_file_index):03d}.parquet",
        parts / "manifests" / f"action-v{int(video_file_index):03d}.parquet",
        parts / "stats" / f"train-v{int(video_file_index):03d}.npz",
    ]
    for path in incomplete_paths:
        Path(path).unlink(missing_ok=True)

    video_path = root / f"videos/observation.images.image_0/chunk-000/file-{video_file_index:03d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    video_sha256 = _sha256_file(video_path)
    selected = sorted(episodes, key=lambda episode: episode.video_from_frame)
    writer = _VideoShardWriter(work, video_file_index=video_file_index, shard_rows=shard_rows)
    episode_cursor = 0
    active_frames: list[np.ndarray] = []
    decoded_count = 0
    emitted = 0
    last_required_frame = selected[-1].video_to_frame
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_count = frame_index + 1
            if episode_cursor >= len(selected):
                break
            episode = selected[episode_cursor]
            if frame_index < episode.video_from_frame:
                continue
            if frame_index >= episode.video_to_frame:
                raise ValueError(
                    f"primary video skipped episode boundary for episode {episode.episode_id}"
                )
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (256, 256, 3) or array.dtype != np.uint8:
                raise ValueError(f"unexpected Bridge frame {array.dtype} {array.shape} in {video_path}")
            active_frames.append(array)
            if frame_index + 1 == episode.video_to_frame:
                emitted += _convert_episode(
                    episode,
                    active_frames,
                    video_sha256=video_sha256,
                    window_stride=window_stride,
                    writer=writer,
                )
                episode_cursor += 1
                active_frames = []
            if frame_index + 1 >= last_required_frame and episode_cursor == len(selected):
                break
    if episode_cursor != len(selected):
        missing = selected[episode_cursor]
        raise ValueError(
            f"primary video ended before episode {missing.episode_id}; decoded={decoded_count} path={video_path}"
        )
    effect_manifest, action_manifest, stats_path = writer.finish(parts)
    required_files = [
        effect_manifest.relative_to(work).as_posix(),
        action_manifest.relative_to(work).as_posix(),
        stats_path.relative_to(work).as_posix(),
        *sorted(writer.shard_checksums),
    ]
    payload = {
        "schema": "bridge-conversion-video-part-v1",
        "complete": True,
        "video_file_index": int(video_file_index),
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "episodes": len(selected),
        "rows": emitted,
        "decoded_frames": decoded_count,
        "effect_manifest": effect_manifest.relative_to(work).as_posix(),
        "action_manifest": action_manifest.relative_to(work).as_posix(),
        "stats": stats_path.relative_to(work).as_posix(),
        "required_files": required_files,
        "shard_checksums": dict(sorted(writer.shard_checksums.items())),
    }
    _atomic_json(marker, payload)
    print(
        f"[bridge-video] file={video_file_index:03d} episodes={len(selected)} rows={emitted} "
        f"decoded={decoded_count}",
        flush=True,
    )
    return payload


def _assign_teacher_pools(frame: pd.DataFrame, maximum_pool_rows: int = 512) -> pd.Series:
    result = pd.Series(index=frame.index, dtype="object")
    for (split, compatibility), group in frame.groupby(["split", "teacher_compatibility"], sort=True):
        bucket_count = max(1, int(math.ceil(len(group) / float(maximum_pool_rows))))
        episode_rows = [
            (int(episode_id), indices.to_numpy(dtype=np.int64))
            for episode_id, indices in group.groupby("episode_id", sort=True).groups.items()
        ]
        episode_rows.sort(
            key=lambda item: (
                -len(item[1]),
                hashlib.blake2b(
                    f"bridge-teacher-pool\0{split}\0{compatibility}\0{item[0]}".encode(),
                    digest_size=16,
                ).digest(),
                item[0],
            )
        )
        buckets: list[list[np.ndarray]] = [[] for _ in range(bucket_count)]
        bucket_sizes = [0] * bucket_count
        for _, indices in episode_rows:
            bucket = min(range(bucket_count), key=lambda index: (bucket_sizes[index], index))
            buckets[bucket].append(indices)
            bucket_sizes[bucket] += len(indices)
        for bucket, chunks in enumerate(buckets):
            if not chunks:
                raise RuntimeError("teacher-pool bucketing emitted an empty bucket")
            indices = np.concatenate(chunks)
            result.loc[indices] = f"{split}/{compatibility}/b{bucket:04d}"
    if result.isna().any():
        raise RuntimeError("teacher-pool assignment missed manifest rows")
    return result.astype(str)


def _write_manifests(root: Path, frame: pd.DataFrame) -> None:
    ordered = frame.sort_values("sample_id").reset_index(drop=True)
    _atomic_parquet(root / "manifests/all.parquet", ordered)
    for split in SPLITS:
        _atomic_parquet(
            root / f"manifests/{split}.parquet",
            ordered[ordered.split == split].reset_index(drop=True),
        )


def _load_train_stats(work: Path, parts: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    loaded: dict[str, list[np.ndarray]] = defaultdict(list)
    for part in parts:
        with np.load(work / str(part["stats"]), allow_pickle=False) as archive:
            for name in archive.files:
                if len(archive[name]):
                    loaded[name].append(np.asarray(archive[name]))
    required = {
        "effect_delta",
        "effect_mask",
        "waypoints",
        "time_mask",
        "pose_mask",
        "gripper_mask",
        "current_state",
        "current_state_mask",
    }
    missing = sorted(required.difference(loaded))
    if missing:
        raise ValueError(f"Bridge train statistics are missing arrays: {missing}")
    return {name: np.concatenate(values, axis=0) for name, values in loaded.items()}


def _effect_normalization(delta: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    group = robust_group(delta, mask)
    return {
        "fit_split": "train",
        "field": "delta_state14",
        "count": group["count"],
        "median": group["median"],
        "scale_mad": group["scale_mad"],
    }


def _action_normalization(stats: Mapping[str, np.ndarray]) -> dict[str, Any]:
    waypoints = np.asarray(stats["waypoints"], dtype=np.float32)
    time = np.asarray(stats["time_mask"], dtype=bool)
    pose = np.asarray(stats["pose_mask"], dtype=bool)
    gripper = np.asarray(stats["gripper_mask"], dtype=bool)
    state = np.asarray(stats["current_state"], dtype=np.float32)
    state_mask = np.asarray(stats["current_state_mask"], dtype=bool)
    pose_time = time & pose
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "fit_split": "train",
        "estimator": "median_and_1.4826_mad",
        "sample_count": len(waypoints),
        "position": robust_group(
            waypoints[..., :3], np.repeat(pose_time[..., None], 3, axis=-1)
        ),
        "rotation_rotvec": robust_group(
            waypoints[..., 3:6], np.repeat(pose_time[..., None], 3, axis=-1)
        ),
        "gripper": robust_group(
            waypoints[..., 6:7], gripper[..., None] & time[..., None]
        ),
        "current_state": robust_group(state, state_mask),
    }


def _stage2_manifest(action_manifest: pd.DataFrame) -> pd.DataFrame:
    frame = action_manifest[action_manifest.language_instruction.astype(str).str.strip() != ""].copy()
    frame["dino_start_step"] = frame.start_step.astype(np.int64)
    frame["dino_future_step"] = frame.start_step.astype(np.int64) + (PHYSICAL_ACTION_STEPS - 1)
    frame["effect_teacher_step"] = frame.end_step.astype(np.int64)
    frame["source_action_steps"] = PHYSICAL_ACTION_STEPS
    frame["action_hz"] = SOURCE_CONTROL_HZ
    frame["duration_seconds"] = TAU_SECONDS
    frame["temporal_contract"] = "actions=t:t+6,dino=t+5,effect=t+6,replan_steps=1"
    def stage2_relative_shard(value: Any, artifact: str) -> str:
        raw = Path(str(value))
        shard_positions = [index for index, part in enumerate(raw.parts) if part == "shards"]
        if not shard_positions:
            raise ValueError(f"cannot derive {artifact} shard suffix from {value!r}")
        suffix = Path(*raw.parts[shard_positions[-1] :])
        # Paths are relative to stage2/manifests/*.parquet.
        return (Path("../..") / artifact / suffix).as_posix()

    frame["effect_shard_path"] = frame.effect_shard_path.map(
        lambda value: stage2_relative_shard(value, "effect")
    )
    frame["action_shard_path"] = frame.shard_path.map(
        lambda value: stage2_relative_shard(value, "action")
    )
    return frame


def _publish_stage2_overlay(stage2_root: Path, source_root: Path) -> None:
    overlay = stage2_root / "lerobot"
    overlay.mkdir(parents=True, exist_ok=True)
    for name in ("data", "videos"):
        destination = overlay / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source_root / name, target_is_directory=True)
    shutil.copytree(source_root / "meta", overlay / "meta", dirs_exist_ok=True)
    _atomic_json(overlay / "meta/modality.json", stage2_modality_document())


def _write_child_checksums(child: Path, known: Mapping[str, str], root: Path) -> None:
    rows: dict[str, str] = {}
    prefix = f"{child.name}/"
    for relative, digest in known.items():
        if relative.startswith(prefix):
            rows[relative[len(prefix) :]] = digest
    del root
    discovered: list[Path] = []
    for directory, child_dirs, filenames in os.walk(child, followlinks=False):
        directory_path = Path(directory)
        child_dirs[:] = [
            name for name in child_dirs if not (directory_path / name).is_symlink()
        ]
        for filename in filenames:
            path = directory_path / filename
            if not path.is_symlink() and path.name != "checksums.sha256":
                discovered.append(path)
    for path in sorted(discovered):
        relative = path.relative_to(child).as_posix()
        rows.setdefault(relative, _sha256_file(path))
    lines = [f"{digest}  {relative}" for relative, digest in sorted(rows.items())]
    (child / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_release(
    work: Path,
    *,
    source_root: Path,
    source_audit: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    effect_parts = [pd.read_parquet(work / str(part["effect_manifest"])) for part in parts]
    action_parts = [pd.read_parquet(work / str(part["action_manifest"])) for part in parts]
    effect = pd.concat(effect_parts, ignore_index=True)
    action = pd.concat(action_parts, ignore_index=True)
    if len(effect) != len(action) or set(effect.sample_id) != set(action.sample_id):
        raise ValueError("Bridge effect/action part manifests are not aligned")
    effect["teacher_pool"] = _assign_teacher_pools(effect)
    pool_by_sample = dict(zip(effect.sample_id.astype(str), effect.teacher_pool.astype(str)))
    action["teacher_pool"] = action.sample_id.astype(str).map(pool_by_sample)
    if action.teacher_pool.isna().any():
        raise ValueError("Bridge action manifest is missing teacher pools")
    _write_manifests(work / "effect", effect)
    _write_manifests(work / "action", action)

    stage2 = _stage2_manifest(action)
    _write_manifests(work / "stage2", stage2)
    _atomic_json(
        work / "stage2/schema.json",
        {
            "schema_version": STAGE2_MANIFEST_SCHEMA_VERSION,
            "source_dataset": SOURCE_DATASET_ID,
            "source_revision": SOURCE_REVISION,
            "views": ["primary_view"],
            "physical_action_steps": PHYSICAL_ACTION_STEPS,
            "action_hz": SOURCE_CONTROL_HZ,
            "duration_seconds": TAU_SECONDS,
            "dino_future_delta_steps": PHYSICAL_ACTION_STEPS - 1,
            "effect_teacher_delta_steps": PHYSICAL_ACTION_STEPS,
            "replan_steps": 1,
            "filter": "complete windows with non-empty language",
        },
    )
    _publish_stage2_overlay(work / "stage2", source_root)

    stats = _load_train_stats(work, parts)
    _atomic_json(
        work / "effect/stats/state_normalization.json",
        _effect_normalization(stats["effect_delta"], stats["effect_mask"]),
    )
    _atomic_json(work / "action/stats/normalization.json", _action_normalization(stats))
    _atomic_json(work / "effect/schema.json", effect_schema_document())
    _atomic_json(work / "action/schema.json", sidecar_schema_document())

    split_counts = Counter(effect.split.astype(str))
    stage2_split_counts = Counter(stage2.split.astype(str))
    train_pools = effect[effect.split == "train"].groupby("teacher_pool").size()
    train_pool_episode_counts = (
        effect[effect.split == "train"].groupby("teacher_pool").episode_id.nunique()
    )
    if len(train_pools) == 0 or int(train_pools.max()) > 512:
        raise ValueError("Bridge train teacher-pool contract failed")
    if contract.get("max_episodes") is None and int(train_pool_episode_counts.min()) <= 16:
        raise ValueError("Bridge train teacher pools require more than 16 source episodes")
    expected_rows = 800_617 if contract.get("max_episodes") is None else None
    expected_stage2_rows = 545_462 if contract.get("max_episodes") is None else None
    if expected_rows is not None and len(effect) != expected_rows:
        raise ValueError(f"official Bridge stride-2 row count changed: {len(effect)} != {expected_rows}")
    if expected_stage2_rows is not None and len(stage2) != expected_stage2_rows:
        raise ValueError(
            f"official Bridge language row count changed: {len(stage2)} != {expected_stage2_rows}"
        )
    coverage = {
        "schema": "bridge-actionvae-coverage-v1",
        "valid": True,
        "rows": len(effect),
        "stage2_language_rows": len(stage2),
        "episodes": len({int(value) for value in effect.episode_id}),
        "split_counts": dict(sorted(split_counts.items())),
        "stage2_split_counts": dict(sorted(stage2_split_counts.items())),
        "effect_families": int(effect.effect_family.nunique()),
        "teacher_pools": int(effect.teacher_pool.nunique()),
        "train_teacher_pool_min_rows": int(train_pools.min()),
        "train_teacher_pool_max_rows": int(train_pools.max()),
        "train_teacher_pool_min_episodes": int(train_pool_episode_counts.min()),
        "window_stride": int(contract["window_stride"]),
        "physical_action_steps": PHYSICAL_ACTION_STEPS,
        "representation_steps": REPRESENTATION_STEPS,
        "source_control_hz": SOURCE_CONTROL_HZ,
        "representation_hz": REPRESENTATION_HZ,
        "tau_seconds": TAU_SECONDS,
    }
    provenance = {
        "schema": "bridge-actionvae-provenance-v1",
        "source_root": str(source_root),
        "source_dataset": SOURCE_DATASET_ID,
        "source_revision": SOURCE_REVISION,
        "source_audit": dict(source_audit),
        "conversion_contract": dict(contract),
        "camera": "observation.images.image_0 only",
        "pose_contract": "Rz(yaw)@Ry(pitch)@Rx(roll)@DEFAULT_ROTATION, then xyz+qwxyz",
        "gripper_contract": "two-finger aperture_m=0.020+0.050*g without clipping",
        "action_representation": (
            "six 5Hz transitions resampled to 24 representation slots; native actions valid only at "
            "indices 3,7,11,15,19,23"
        ),
        "stage2_temporal_contract": "future DINO t+5, effect teacher/action endpoint t+6, replan_steps=1",
        "split_contract": "global deterministic episode-level 80/10/10",
    }
    for artifact in ("effect", "action", "stage2"):
        child = work / artifact
        _atomic_json(child / "coverage_report.json", {**coverage, "artifact": artifact})
        _atomic_json(child / "provenance.json", {**provenance, "artifact": artifact})
        (child / "_SUCCESS").write_text(f"{RELEASE_SCHEMA_VERSION}:{artifact}\n", encoding="utf-8")
    _atomic_json(work / "coverage_report.json", coverage)
    _atomic_json(work / "provenance.json", provenance)
    _atomic_json(work / "audit.json", dict(source_audit))
    (work / "_SUCCESS").write_text(f"{RELEASE_SCHEMA_VERSION}\n", encoding="utf-8")

    known_checksums: dict[str, str] = {}
    for part in parts:
        known_checksums.update({str(k): str(v) for k, v in part["shard_checksums"].items()})
    shutil.rmtree(work / ".parts")
    for artifact in ("effect", "action", "stage2"):
        _write_child_checksums(work / artifact, known_checksums, work)
    root_paths = [
        work / "_SUCCESS",
        work / "audit.json",
        work / "coverage_report.json",
        work / "provenance.json",
        work / "effect/checksums.sha256",
        work / "action/checksums.sha256",
        work / "stage2/checksums.sha256",
    ]
    (work / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256_file(path)}  {path.relative_to(work).as_posix()}" for path in root_paths
        )
        + "\n",
        encoding="utf-8",
    )
    return coverage


def build_release(
    source_root: str | Path,
    output: str | Path,
    *,
    split_seed: int = 20260803,
    max_episodes: int | None = None,
    window_stride: int = 2,
    workers: int = 16,
    shard_rows: int = 256,
) -> dict[str, Any]:
    global _SOURCE_STATES, _SOURCE_ACTIONS

    if int(window_stride) != 2:
        raise ValueError("the confirmed Bridge release contract requires window_stride=2")
    if workers < 1 or shard_rows < 1:
        raise ValueError("workers and shard_rows must be positive")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        report = audit_release(destination, verify_checksums=False)
        if not report["valid"]:
            raise FileExistsError(f"existing Bridge release is invalid: {destination}")
        return report
    source = load_source(source_root, split_seed=split_seed, max_episodes=max_episodes)
    source_report = audit_source(
        source_root,
        split_seed=split_seed,
        max_episodes=max_episodes,
        window_stride=window_stride,
    )
    contract = {
        "schema": "bridge-conversion-contract-v1",
        "source_root": str(source.root),
        "source_revision": SOURCE_REVISION,
        "split_seed": int(split_seed),
        "max_episodes": max_episodes,
        "window_stride": int(window_stride),
        "shard_rows": int(shard_rows),
        "single_primary_view": True,
        "physical_action_steps": PHYSICAL_ACTION_STEPS,
        "representation_steps": REPRESENTATION_STEPS,
        "gripper_aperture": "0.020+0.050*g_no_clip",
        "stage2_temporal": "dino_t_plus_5_effect_t_plus_6_replan_1",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f".{destination.name}.work")
    if work.exists():
        contract_path = work / ".parts/contract.json"
        if not contract_path.is_file() or json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"stale Bridge conversion work directory has a different contract: {work}")
    else:
        (work / ".parts").mkdir(parents=True)
        _atomic_json(work / ".parts/contract.json", contract)

    by_video: dict[int, list[BridgeEpisode]] = defaultdict(list)
    for episode in source.episodes:
        if episode.length > PHYSICAL_ACTION_STEPS:
            by_video[episode.video_file_index].append(episode)
    if not by_video:
        raise ValueError("selected Bridge episodes contain no complete 1.2-second windows")
    _SOURCE_STATES = source.states
    _SOURCE_ACTIONS = source.actions
    jobs = [
        (str(source.root), str(work), video_index, episodes, int(window_stride), int(shard_rows))
        for video_index, episodes in sorted(by_video.items())
    ]
    parts: list[dict[str, Any]] = []
    if workers == 1 or len(jobs) == 1:
        for job in jobs:
            parts.append(_convert_video_file(*job))
    else:
        # Inherit the large source arrays copy-on-write instead of duplicating
        # them into each worker through a spawn context.
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(jobs)),
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            futures = {executor.submit(_convert_video_file, *job): job[2] for job in jobs}
            for future in as_completed(futures):
                parts.append(future.result())
    parts.sort(key=lambda part: int(part["video_file_index"]))
    if sum(int(part["rows"]) for part in parts) != int(source_report["complete_windows"]):
        raise ValueError("Bridge worker row total does not match the source audit")
    coverage = _finalize_release(
        work,
        source_root=source.root,
        source_audit=source_report,
        parts=parts,
        contract=contract,
    )
    os.replace(work, destination)
    print(json.dumps({"output": str(destination), **coverage}, sort_keys=True), flush=True)
    return {"output": str(destination), **coverage}


def _verify_checksum_file(root: Path, checksum_file: Path) -> list[str]:
    issues: list[str] = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            issues.append(f"missing:{path}")
        elif _sha256_file(path) != expected:
            issues.append(f"checksum:{path}")
    return issues


def audit_release(root: str | Path, *, verify_checksums: bool = False) -> dict[str, Any]:
    release = Path(root).expanduser().resolve()
    issues: list[str] = []
    required = [
        release / "_SUCCESS",
        release / "effect/schema.json",
        release / "action/schema.json",
        release / "stage2/schema.json",
        release / "effect/manifests/all.parquet",
        release / "action/manifests/all.parquet",
        release / "stage2/manifests/all.parquet",
    ]
    issues.extend(f"missing:{path}" for path in required if not path.is_file())
    if issues:
        return {"schema": "bridge-release-audit-v1", "valid": False, "issues": issues}
    effect_schema = json.loads((release / "effect/schema.json").read_text(encoding="utf-8"))
    action_schema = json.loads((release / "action/schema.json").read_text(encoding="utf-8"))
    if effect_schema.get("schema_version") != EFFECT_SCHEMA_VERSION:
        issues.append("effect_schema")
    if action_schema.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        issues.append("action_schema")
    effect = pd.read_parquet(release / "effect/manifests/all.parquet")
    action = pd.read_parquet(release / "action/manifests/all.parquet")
    stage2 = pd.read_parquet(release / "stage2/manifests/all.parquet")
    if effect.sample_id.tolist() != action.sample_id.tolist():
        issues.append("paired_manifest_order")
    if len(set(effect.sample_id.astype(str))) != len(effect):
        issues.append("duplicate_sample_id")
    if not np.all((effect.end_step - effect.start_step).to_numpy() == PHYSICAL_ACTION_STEPS):
        issues.append("physical_window")
    if (stage2.language_instruction.astype(str).str.strip() == "").any():
        issues.append("empty_stage2_language")
    if not set(stage2.sample_id.astype(str)).issubset(set(action.sample_id.astype(str))):
        issues.append("stage2_not_action_subset")
    expected_temporal = {
        "dino_future_step": stage2.start_step.astype(np.int64) + (PHYSICAL_ACTION_STEPS - 1),
        "effect_teacher_step": stage2.start_step.astype(np.int64) + PHYSICAL_ACTION_STEPS,
        "source_action_steps": pd.Series(PHYSICAL_ACTION_STEPS, index=stage2.index),
    }
    for name, expected in expected_temporal.items():
        if name not in stage2 or not np.array_equal(
            stage2[name].to_numpy(dtype=np.int64), expected.to_numpy(dtype=np.int64)
        ):
            issues.append(f"stage2_temporal:{name}")
    manifest_parent = release / "stage2" / "manifests"
    for column in ("effect_shard_path", "action_shard_path"):
        if column not in stage2:
            issues.append(f"stage2_missing:{column}")
            continue
        for raw_path in sorted(set(stage2[column].astype(str))):
            if not (manifest_parent / raw_path).resolve().is_file():
                issues.append(f"stage2_shard_path:{column}:{raw_path}")
                break
    for artifact, version in (("effect", EFFECT_SCHEMA_VERSION), ("action", SIDECAR_SCHEMA_VERSION)):
        for path in sorted((release / artifact / "shards").glob("*/*.h5")):
            with h5py.File(path, "r") as handle:
                if not bool(handle.attrs.get("complete", False)) or str(handle.attrs.get("schema_version", "")) != version:
                    issues.append(f"incomplete_shard:{path}")
    if verify_checksums:
        for artifact in ("effect", "action", "stage2"):
            checksum_file = release / artifact / "checksums.sha256"
            if not checksum_file.is_file():
                issues.append(f"missing:{checksum_file}")
            else:
                issues.extend(_verify_checksum_file(release / artifact, checksum_file))
        root_checksum = release / "checksums.sha256"
        if root_checksum.is_file():
            issues.extend(_verify_checksum_file(release, root_checksum))
        else:
            issues.append(f"missing:{root_checksum}")
    return {
        "schema": "bridge-release-audit-v1",
        "valid": not issues,
        "root": str(release),
        "rows": len(effect),
        "stage2_language_rows": len(stage2),
        "effect_shards": len(list((release / "effect/shards").glob("*/*.h5"))),
        "action_shards": len(list((release / "action/shards").glob("*/*.h5"))),
        "checksums_verified": bool(verify_checksums),
        "issues": issues,
    }
