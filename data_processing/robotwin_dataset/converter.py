"""Stream the RoboTwin LeRobot release into immutable Stage-1 HDF5 windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

from lawam_effect_dataset.shards import write_shard_atomic
from lawam_effect_dataset.statistics import robust_normalization
from lawam_effect_dataset.windows import group_id, stable_id

from .geometry import dual_arm_contract, effect_direction, normalize_pose_wxyz, resize_cam_high_batch
from .schema import schema_document, schema_version_for_horizon
from .source import (
    BENCHMARK_FAMILIES,
    CONTROL_HZ,
    HORIZON,
    WINDOW_STRIDE,
    family_for_episode,
    load_source_contract,
    source_commit,
    stratified_episode_splits,
    window_starts,
)


CONTROLLER_ID = "robotwin_dual_arm_absolute_eef_wxyz"
SPLITS = ("train", "validation", "test")


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _identity_pose() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _inventory_digest(commit: str, relative_path: str, size: int) -> str:
    return hashlib.sha256(f"{commit}\0{relative_path}\0{int(size)}".encode("utf-8")).hexdigest()


def _decode_selected_frames_once(
    path: Path,
    frame_indices: Sequence[int],
    *,
    resize_batch_size: int = 64,
) -> dict[int, np.ndarray]:
    targets = np.asarray(sorted(set(int(value) for value in frame_indices)), dtype=np.int64)
    if len(targets) == 0:
        return {}
    if int(targets[0]) < 0:
        raise ValueError(f"negative video frame requested from {path}")
    result: dict[int, np.ndarray] = {}
    pending_indices: list[int] = []
    pending_frames: list[np.ndarray] = []

    def flush() -> None:
        if not pending_frames:
            return
        resized = resize_cam_high_batch(pending_frames)
        for index, image in zip(pending_indices, resized):
            result[index] = np.ascontiguousarray(image)
        pending_indices.clear()
        pending_frames.clear()

    cursor = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for decoded_index, frame in enumerate(container.decode(stream)):
            while cursor < len(targets) and int(targets[cursor]) < decoded_index:
                raise RuntimeError(f"video decoder skipped frame {int(targets[cursor])} in {path}")
            if cursor >= len(targets):
                break
            if decoded_index != int(targets[cursor]):
                continue
            pending_indices.append(decoded_index)
            pending_frames.append(frame.to_ndarray(format="rgb24"))
            cursor += 1
            if len(pending_frames) >= int(resize_batch_size):
                flush()
            if cursor >= len(targets):
                break
    flush()
    if cursor != len(targets):
        missing = targets[cursor : cursor + 8].tolist()
        raise RuntimeError(f"video {path} ended before requested frames {missing}")
    return result


def _decode_selected_frames_by_pts(
    path: Path,
    frame_indices: Sequence[int],
    *,
    resize_batch_size: int = 64,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    targets = np.asarray(sorted(set(int(value) for value in frame_indices)), dtype=np.int64)
    if len(targets) == 0:
        return {}, {
            "video_path": str(path),
            "missing_frame_indices": [],
            "corrupt_packet_approx_indices": [],
        }
    if int(targets[0]) < 0:
        raise ValueError(f"negative video frame requested from {path}")

    result: dict[int, np.ndarray] = {}
    seen_indices: set[int] = set()
    corrupt_packet_indices: list[int | None] = []
    pending_indices: list[int] = []
    pending_frames: list[np.ndarray] = []

    def flush() -> None:
        if not pending_frames:
            return
        resized = resize_cam_high_batch(pending_frames)
        for index, image in zip(pending_indices, resized):
            result[index] = np.ascontiguousarray(image)
        pending_indices.clear()
        pending_frames.clear()

    target_set = set(targets.tolist())
    previous_index = -1
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is None or stream.average_rate is None:
            raise RuntimeError(f"video {path} has no stable time_base/average_rate for PTS recovery")
        index_scale = stream.time_base * stream.average_rate
        complete = False
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                approximate = None if packet.pts is None else round(packet.pts * index_scale)
                corrupt_packet_indices.append(approximate)
                continue
            for frame in frames:
                if frame.pts is None:
                    raise RuntimeError(f"video {path} produced a frame without PTS during recovery")
                position = frame.pts * index_scale
                decoded_index = round(position)
                if abs(float(position - decoded_index)) >= 0.5:
                    raise RuntimeError(f"video {path} has ambiguous PTS-to-frame mapping at PTS {frame.pts}")
                if decoded_index <= previous_index:
                    raise RuntimeError(f"video {path} PTS frame indices are not strictly increasing")
                previous_index = decoded_index
                seen_indices.add(decoded_index)
                if decoded_index in target_set:
                    pending_indices.append(decoded_index)
                    pending_frames.append(frame.to_ndarray(format="rgb24"))
                    if len(pending_frames) >= int(resize_batch_size):
                        flush()
                    if len(result) + len(pending_indices) == len(target_set):
                        complete = True
                        break
            if complete:
                break
    flush()
    missing_targets = sorted(target_set.difference(result))
    if missing_targets:
        raise RuntimeError(
            f"video {path} corruption overlaps requested frame indices {missing_targets[:8]}"
        )
    missing_indices = sorted(set(range(int(targets[-1]) + 1)).difference(seen_indices))
    return result, {
        "video_path": str(path),
        "recovery": "packet_skip_with_pts_frame_identity",
        "missing_frame_indices_before_last_target": missing_indices,
        "corrupt_packet_approx_indices": corrupt_packet_indices,
        "requested_frame_overlap": [],
    }


def _decode_selected_frames(
    path: Path,
    frame_indices: Sequence[int],
    *,
    resize_batch_size: int = 64,
    max_attempts: int = 3,
    repair_log: list[dict[str, Any]] | None = None,
) -> dict[int, np.ndarray]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            return _decode_selected_frames_once(
                path,
                frame_indices,
                resize_batch_size=resize_batch_size,
            )
        except av.error.InvalidDataError as exc:
            if attempt == max_attempts:
                print(
                    f"[video-repair] {path} normal decode failed {max_attempts} times; "
                    "trying strict PTS recovery",
                    flush=True,
                )
                recovered, repair = _decode_selected_frames_by_pts(
                    path,
                    frame_indices,
                    resize_batch_size=resize_batch_size,
                )
                if repair_log is not None:
                    repair_log.append(repair)
                return recovered
            print(
                f"[video-retry] {path} attempt={attempt + 1}/{max_attempts} "
                f"after {type(exc).__name__}: {exc}",
                flush=True,
            )
    raise AssertionError("unreachable")


class _EffectShardWriter:
    def __init__(self, root: Path, shard_rows: int, *, schema_version: str) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.root = root
        self.shard_rows = int(shard_rows)
        self.schema_version = str(schema_version)
        self.buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        self.indices = {split: 0 for split in SPLITS}
        self.manifest: list[dict[str, Any]] = []
        self.stats_rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        split = str(row["split"])
        self.buffers[split].append(row)
        self.stats_rows.append(
            {
                "split": split,
                "effect_valid": row["effect_valid"],
                "delta_state14": row["delta_state14"],
                "state_mask": row["state_mask"],
            }
        )
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        relative = Path("shards") / split / f"part-{self.indices[split]:05d}.h5"
        write_shard_atomic(self.root / relative, rows, schema_version=self.schema_version)
        for shard_row, row in enumerate(rows):
            self.manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "split": split,
                    "task_id": row["task_id"],
                    "effect_family": row["effect_family"],
                    "source_id": row["source_id"],
                    "source_commit": row["source_commit"],
                    "controller_id": row["controller_id"],
                    "episode_id": row["episode_id"],
                    "episode_seed": row["episode_seed"],
                    "start_step": row["start_step"],
                    "end_step": row["end_step"],
                    "control_hz": row["control_hz"],
                    "tau": row["tau"],
                    "effect_direction": row["effect_direction"],
                    "effect_magnitude": row["effect_magnitude"],
                    "view": "cam_high",
                    "language_instruction": row["language_instruction"],
                    "suite": "robotwin",
                    "shard_path": relative.as_posix(),
                    "shard_row": shard_row,
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


def _selected_episode_rows(
    episodes: dict[str, list[Any]],
    *,
    max_episodes: int | None,
) -> list[dict[str, Any]]:
    count = len(episodes["episode_index"])
    if max_episodes is not None:
        if max_episodes < 3:
            raise ValueError("max_episodes must be at least three")
        count = min(count, int(max_episodes))
    records: list[dict[str, Any]] = []
    for index in range(count):
        episode_id = int(episodes["episode_index"][index])
        tasks = episodes["tasks"][index]
        if not isinstance(tasks, list) or len(tasks) != 1 or not str(tasks[0]).strip():
            raise ValueError(f"episode {episode_id} must have exactly one non-empty task instruction")
        record = {name: values[index] for name, values in episodes.items()}
        record["episode_id"] = episode_id
        record["family"] = family_for_episode(episode_id)
        record["instruction"] = str(tasks[0]).strip()
        records.append(record)
    return records


def _attach_splits(records: list[dict[str, Any]], seed: int) -> None:
    by_family: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_family[str(record["family"])].append(int(record["episode_id"]))
    split_maps = {
        family: stratified_episode_splits(family, ids, seed)
        for family, ids in by_family.items()
    }
    for record in records:
        record["split"] = split_maps[str(record["family"])][int(record["episode_id"])]


def _window_specs(
    records: list[dict[str, Any]],
    stride: int,
    *,
    horizon: int = HORIZON,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for record in records:
        episode_id = int(record["episode_id"])
        data_start = int(record["dataset_from_index"])
        video_start = int(round(float(record["videos/observation.images.cam_high/from_timestamp"]) * CONTROL_HZ))
        length = int(record["length"])
        duration = float(record["videos/observation.images.cam_high/to_timestamp"]) - float(
            record["videos/observation.images.cam_high/from_timestamp"]
        )
        if not np.isclose(duration, length / CONTROL_HZ, atol=1.0 / CONTROL_HZ, rtol=0.0):
            raise ValueError(f"episode {episode_id} video duration {duration} disagrees with length {length}")
        for start in window_starts(length, horizon=horizon, stride=stride).tolist():
            end = int(start) + int(horizon)
            specs.append(
                {
                    **record,
                    "start": int(start),
                    "end": end,
                    "state_start_index": data_start + int(start),
                    "state_end_index": data_start + end,
                    "video_start_index": video_start + int(start),
                    "video_end_index": video_start + end,
                }
            )
    return specs


def _load_selected_states(
    source: Path,
    episodes: dict[str, list[Any]],
    specs: list[dict[str, Any]],
) -> dict[int, np.ndarray]:
    requested_by_file: dict[tuple[int, int], set[int]] = defaultdict(set)
    for spec in specs:
        key = (int(spec["data/chunk_index"]), int(spec["data/file_index"]))
        requested_by_file[key].update((int(spec["state_start_index"]), int(spec["state_end_index"])))

    file_starts: dict[tuple[int, int], int] = {}
    for index, chunk in enumerate(episodes["data/chunk_index"]):
        key = (int(chunk), int(episodes["data/file_index"][index]))
        start = int(episodes["dataset_from_index"][index])
        file_starts[key] = min(start, file_starts.get(key, start))

    result: dict[int, np.ndarray] = {}
    for key in sorted(requested_by_file):
        chunk, file_index = key
        path = source / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        table = parquet.read_table(path, columns=["observation.state", "index"])
        globals_requested = np.asarray(sorted(requested_by_file[key]), dtype=np.int64)
        local = globals_requested - int(file_starts[key])
        if np.any(local < 0) or np.any(local >= table.num_rows):
            raise IndexError(f"state endpoint outside source parquet {path}")
        selected_indices = table.column("index").take(pa.array(local)).to_numpy()
        if not np.array_equal(selected_indices.astype(np.int64), globals_requested):
            raise ValueError(f"global index contract mismatch in {path}")
        selected = table.column("observation.state").take(pa.array(local)).combine_chunks()
        offsets = selected.offsets.to_numpy(zero_copy_only=False)
        if not np.all(np.diff(offsets) == 16):
            raise ValueError(f"observation.state rows are not 16D in {path}")
        values = selected.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 16)
        if not np.isfinite(values).all():
            raise ValueError(f"observation.state contains NaN/Inf in {path}")
        result.update((int(index), np.ascontiguousarray(value)) for index, value in zip(globals_requested, values))
        print(f"[state] {path.name}: loaded {len(values)} selected endpoints", flush=True)
    return result


def _effect_row(
    spec: Mapping[str, Any],
    *,
    state_start: np.ndarray,
    state_end: np.ndarray,
    rgb_start: np.ndarray,
    rgb_end: np.ndarray,
    commit: str,
    source_digest: str,
    horizon: int,
) -> dict[str, Any]:
    episode_id = int(spec["episode_id"])
    start = int(spec["start"])
    end = int(spec["end"])
    family = str(spec["family"])
    source_id = "robotwin_merged"
    sample_id = stable_id(source_id, episode_id, start, end, version="robotwin-v1")
    state14_start, state14_end, delta_state14, state_mask, effect_raw = dual_arm_contract(
        state_start, state_end
    )
    left_start = normalize_pose_wxyz(state_start[:8], name="left_start")
    left_end = normalize_pose_wxyz(state_end[:8], name="left_end")
    identity = _identity_pose()
    zeros_pose = np.zeros((3, 7), dtype=np.float32)
    return {
        "sample_id": sample_id,
        "task_id": f"robotwin/{family}",
        "effect_family": family,
        "source_id": source_id,
        "source_commit": commit,
        "controller_id": CONTROLLER_ID,
        "episode_id": episode_id,
        "episode_seed": episode_id,
        "start_step": start,
        "end_step": end,
        "control_hz": CONTROL_HZ,
        "tau": int(horizon) / CONTROL_HZ,
        "split": str(spec["split"]),
        "rgb_start": rgb_start,
        "rgb_end": rgb_end,
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
        "effect_raw": effect_raw,
        "effect_mask": np.ones(12, dtype=np.uint8),
        "effect_valid": 1,
        "effect_direction": effect_direction(delta_state14),
        "effect_magnitude": float(np.linalg.norm(effect_raw)),
        "eef_pose_start": left_start[:7].astype(np.float32),
        "eef_pose_end": left_end[:7].astype(np.float32),
        "gripper_start": float(left_start[7]),
        "gripper_end": float(left_end[7]),
        "state14_start": state14_start,
        "state14_end": state14_end,
        "delta_state14": delta_state14,
        "state_mask": state_mask,
        "embodiment_id": "agilex",
        "contact_summary": np.zeros(4, dtype=np.float32),
        "grasp_summary": np.zeros(4, dtype=np.float32),
        "phase": "demonstration",
        "outcome": "success",
        "quality": 1.0,
        "replay_valid": 0,
        "render_valid": 1,
        "source_sha256": source_digest,
        "group_id": group_id(source_id, episode_id),
        "language_instruction": str(spec["instruction"]),
    }


def build_release(
    source_root: str | Path,
    output: str | Path,
    *,
    max_episodes: int | None = None,
    horizon_frames: int = HORIZON,
    window_stride: int = WINDOW_STRIDE,
    shard_rows: int = 256,
    split_seed: int = 20260818,
) -> dict[str, Any]:
    horizon = int(horizon_frames)
    if horizon < 1 or window_stride < 1:
        raise ValueError("horizon_frames and window_stride must be positive")
    schema_version = schema_version_for_horizon(horizon)
    source = Path(source_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    info, episodes = load_source_contract(source)
    records = _selected_episode_rows(episodes, max_episodes=max_episodes)
    _attach_splits(records, split_seed)
    specs = _window_specs(records, int(window_stride), horizon=horizon)
    if not specs:
        raise ValueError(f"selected RoboTwin episodes contain no complete {horizon}-frame windows")
    states = _load_selected_states(source, episodes, specs)
    commit = source_commit(source)

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    writer = _EffectShardWriter(temporary, shard_rows, schema_version=schema_version)
    source_inventory: list[dict[str, Any]] = []
    video_decode_repairs: list[dict[str, Any]] = []
    by_video: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        key = (
            int(spec["videos/observation.images.cam_high/chunk_index"]),
            int(spec["videos/observation.images.cam_high/file_index"]),
        )
        by_video[key].append(spec)
    try:
        for file_number, key in enumerate(sorted(by_video), start=1):
            chunk, file_index = key
            path = source / (
                f"videos/observation.images.cam_high/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            file_specs = sorted(
                by_video[key], key=lambda item: (int(item["episode_id"]), int(item["start"]))
            )
            requested = [
                index
                for spec in file_specs
                for index in (int(spec["video_start_index"]), int(spec["video_end_index"]))
            ]
            print(
                f"[video] {file_number}/{len(by_video)} {path.name}: "
                f"windows={len(file_specs)} unique_frames={len(set(requested))}",
                flush=True,
            )
            images = _decode_selected_frames(path, requested, repair_log=video_decode_repairs)
            relative = path.relative_to(source).as_posix()
            digest = _inventory_digest(commit, relative, path.stat().st_size)
            source_inventory.append(
                {"relative_path": relative, "bytes": path.stat().st_size, "inventory_sha256": digest}
            )
            for spec in file_specs:
                writer.add(
                    _effect_row(
                        spec,
                        state_start=states[int(spec["state_start_index"])],
                        state_end=states[int(spec["state_end_index"])],
                        rgb_start=images[int(spec["video_start_index"])],
                        rgb_end=images[int(spec["video_end_index"])],
                        commit=commit,
                        source_digest=digest,
                        horizon=horizon,
                    )
                )
        writer.finish()
        _write_manifests(temporary, writer.manifest)
        # RoboTwin contains long stationary intervals and discrete gripper
        # changes. MAD alone can be nearly zero and catastrophically amplify
        # valid motion, so use a train-only RMS floor for this release.
        stats = robust_normalization(
            writer.stats_rows,
            "delta_state14",
            "state_mask",
            rms_floor=True,
        )
        _atomic_json(temporary / "stats/state_normalization.json", stats)
        _atomic_json(
            temporary / "schema.json",
            schema_document(horizon=horizon, control_hz=CONTROL_HZ),
        )

        split_counts = Counter(str(row["split"]) for row in writer.manifest)
        family_counts = Counter(str(row["effect_family"]) for row in writer.manifest)
        episode_split = {int(record["episode_id"]): str(record["split"]) for record in records}
        coverage = {
            "schema": "robotwin-effect-coverage-v1",
            "valid": True,
            "rows": len(writer.manifest),
            "episodes": len(records),
            "families": len(family_counts),
            "expected_full_families": len(BENCHMARK_FAMILIES),
            "split_counts": dict(sorted(split_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "window_stride": int(window_stride),
            "horizon_frames": horizon,
            "control_hz": CONTROL_HZ,
            "tau_seconds": horizon / CONTROL_HZ,
            "episode_split_disjoint": len(episode_split) == len(records),
            "video_decode_repair_count": len(video_decode_repairs),
            "video_decode_missing_frame_count": sum(
                len(item["missing_frame_indices_before_last_target"])
                for item in video_decode_repairs
            ),
        }
        _atomic_json(temporary / "coverage_report.json", coverage)
        provenance = {
            "schema": "robotwin-effect-provenance-v1",
            "source_root": str(source),
            "source_dataset": "jialei02/robotwin_merged",
            "source_commit": commit,
            "source_info": {
                "total_episodes": info["total_episodes"],
                "total_frames": info["total_frames"],
                "fps": info["fps"],
            },
            "source_inventory": source_inventory,
            "video_decode_repairs": video_decode_repairs,
            "source_inventory_note": (
                "inventory_sha256 binds HF commit, relative path, and byte size; it is not a full MP4 content hash"
            ),
            "camera": "cam_high",
            "split_seed": int(split_seed),
            "split_unit": "episode within benchmark effect family",
            "family_contract": "55 source blocks mapped to 50 benchmark families",
            "window_stride": int(window_stride),
            "horizon_frames": horizon,
            "state_contract": "dual-arm 2x(position3, relative-rotvec3, gripper1)",
            "teacher_contract": "DINO residual is computed from rgb_end-rgb_start only",
        }
        _atomic_json(temporary / "provenance.json", provenance)
        (temporary / "_SUCCESS").write_text(f"{schema_version}\n", encoding="utf-8")
        _write_checksums(temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        print(json.dumps({"output": str(destination), **coverage}, sort_keys=True), flush=True)
        return coverage
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--horizon-frames", type=int, default=HORIZON)
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--shard-rows", type=int, default=256)
    parser.add_argument("--split-seed", type=int, default=20260818)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_release(
        args.source_root,
        args.output,
        max_episodes=args.max_episodes,
        horizon_frames=args.horizon_frames,
        window_stride=args.window_stride,
        shard_rows=args.shard_rows,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
