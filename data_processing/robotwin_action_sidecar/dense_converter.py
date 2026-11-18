"""Build a dense RoboTwin ActionAE A1 sidecar without materializing RGB effect shards.

Each source frame is decoded and embedded by frozen DINO at most once.  For an
episode of length ``L``, the exporter emits every complete ``[t, t+horizon]``
window, so the row count is ``L - horizon`` rather than the sparse
non-overlapping count used by the original sidecar builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import torch
import torch.distributed as dist
from scipy.spatial.transform import Rotation

from lawam_effect_dataset.windows import group_id, stable_id
from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from robotwin_dataset.converter import CONTROLLER_ID, _attach_splits, _inventory_digest, _selected_episode_rows
from robotwin_dataset.geometry import resize_cam_high_batch
from robotwin_dataset.source import CONTROL_HZ, load_source_contract, source_commit

from .converter import _list16, _window_digest
from .io import write_shard_atomic
from .schema import (
    EFFECT_DIM,
    NATIVE_ACTION_DIM,
    TRAJECTORY_DIM,
    anchor_steps_for_horizon,
    condition_field_names,
    fields_for_horizon,
    schema_document,
    versions_for_horizon,
)
from .audit import audit_release


SPLITS = ("train", "validation", "test")
SOURCE_ID = "robotwin_merged"
EMBODIMENT_ID = "agilex"
EMBODIMENT_INDEX = 1


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


def _manifest_schema(*, condition_source: str = "e39") -> pa.Schema:
    _, valid_field = condition_field_names(condition_source)
    return pa.schema(
        [
            ("sample_id", pa.string()),
            ("group_id", pa.string()),
            ("split", pa.string()),
            ("task_id", pa.string()),
            ("effect_family", pa.string()),
            ("source_id", pa.string()),
            ("source_commit", pa.string()),
            ("controller_id", pa.string()),
            ("episode_id", pa.int64()),
            ("episode_seed", pa.int64()),
            ("start_step", pa.int32()),
            ("end_step", pa.int32()),
            ("control_hz", pa.float32()),
            ("tau", pa.float32()),
            ("language_instruction", pa.string()),
            ("suite", pa.string()),
            ("shard_path", pa.string()),
            ("shard_row", pa.int32()),
            ("window_sha256", pa.string()),
            (valid_field, pa.bool_()),
        ]
    )


class _DenseShardWriter:
    def __init__(
        self,
        root: Path,
        part_root: Path,
        *,
        rank: int,
        shard_rows: int,
        horizon: int,
        condition_source: str = "e39",
    ) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.root = root
        self.part_root = part_root
        self.rank = int(rank)
        self.shard_rows = int(shard_rows)
        self.horizon = int(horizon)
        self.condition_source = str(condition_source)
        self.mean_field, self.valid_field = condition_field_names(self.condition_source)
        self.schema_version, _, _ = versions_for_horizon(
            self.horizon, condition_source=self.condition_source
        )
        self.fields = fields_for_horizon(
            self.horizon, condition_source=self.condition_source
        )
        self.buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        self.indices = {split: 0 for split in SPLITS}
        self.counts = Counter()
        self.manifest_schema = _manifest_schema(condition_source=self.condition_source)
        self.manifest_writers: dict[str, parquet.ParquetWriter] = {}

    def add(self, row: dict[str, Any], manifest: dict[str, Any]) -> None:
        split = str(row["split"])
        if split not in self.buffers or split != str(manifest["split"]):
            raise ValueError(f"invalid dense row split {split!r}")
        self.buffers[split].append({"row": row, "manifest": manifest})
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def _manifest_writer(self, split: str) -> parquet.ParquetWriter:
        writer = self.manifest_writers.get(split)
        if writer is None:
            path = self.part_root / "manifests" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = parquet.ParquetWriter(path, self.manifest_schema, compression="zstd")
            self.manifest_writers[split] = writer
        return writer

    def flush(self, split: str) -> None:
        buffered = self.buffers[split]
        if not buffered:
            return
        relative = Path("shards") / split / (
            f"rank-{self.rank:02d}-part-{self.indices[split]:05d}.h5"
        )
        rows = [item["row"] for item in buffered]
        write_shard_atomic(
            self.root / relative,
            rows,
            schema_version=self.schema_version,
            fields=self.fields,
        )
        manifest_rows = []
        for shard_row, item in enumerate(buffered):
            manifest = dict(item["manifest"])
            manifest.update(
                shard_path=relative.as_posix(),
                shard_row=shard_row,
                window_sha256=str(item["row"]["window_sha256"]),
                **{self.valid_field: True},
            )
            manifest_rows.append(manifest)
        table = pa.Table.from_pylist(manifest_rows, schema=self.manifest_schema)
        self._manifest_writer(split).write_table(table, row_group_size=len(table))
        self.counts[split] += len(buffered)
        self.buffers[split] = []
        self.indices[split] += 1

    def finish(self) -> dict[str, int]:
        for split in SPLITS:
            self.flush(split)
        for writer in self.manifest_writers.values():
            writer.close()
        for split in SPLITS:
            if split not in self.manifest_writers:
                path = self.part_root / "manifests" / f"{split}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                parquet.write_table(
                    pa.Table.from_pylist([], schema=self.manifest_schema),
                    path,
                    compression="zstd",
                )
        return {split: int(self.counts[split]) for split in SPLITS}


def _combine_manifests(
    work: Path, world_size: int, *, condition_source: str = "e39"
) -> dict[str, int]:
    output_root = work / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    schema = _manifest_schema(condition_source=condition_source)
    counts: dict[str, int] = {}
    all_writer = parquet.ParquetWriter(output_root / "all.parquet", schema, compression="zstd")
    try:
        for split in SPLITS:
            split_writer = parquet.ParquetWriter(
                output_root / f"{split}.parquet", schema, compression="zstd"
            )
            count = 0
            try:
                for rank in range(world_size):
                    path = work / "parts" / f"rank-{rank:02d}" / "manifests" / f"{split}.parquet"
                    source = parquet.ParquetFile(path)
                    for row_group in range(source.num_row_groups):
                        table = source.read_row_group(row_group)
                        split_writer.write_table(table, row_group_size=len(table))
                        all_writer.write_table(table, row_group_size=len(table))
                        count += len(table)
            finally:
                split_writer.close()
            counts[split] = count
    finally:
        all_writer.close()
    return counts


def _world_state14(states: np.ndarray) -> np.ndarray:
    source = np.asarray(states, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 16 or not np.isfinite(source).all():
        raise ValueError(f"states must be finite [T,16], got {source.shape}")
    result = np.empty((len(source), 14), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (8, 7)):
        arm = source[:, source_offset : source_offset + 8]
        quaternion = arm[:, [4, 5, 6, 3]]
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1.0e-8):
            raise ValueError("source states contain a zero quaternion")
        quaternion = quaternion / norms
        result[:, target_offset : target_offset + 3] = arm[:, :3]
        result[:, target_offset + 3 : target_offset + 6] = Rotation.from_quat(
            quaternion
        ).as_rotvec()
        result[:, target_offset + 6] = arm[:, 7]
    return result


def canonical_dense_windows(states: np.ndarray, horizon: int) -> np.ndarray:
    """Vectorized equivalent of canonical_dual_arm_waypoints for all stride-1 windows."""

    source = np.asarray(states, dtype=np.float64)
    horizon = int(horizon)
    count = len(source) - horizon
    if source.ndim != 2 or source.shape[1] != 16 or count < 1:
        raise ValueError(f"expected states [L,16] with L>{horizon}, got {source.shape}")
    post = np.lib.stride_tricks.sliding_window_view(
        source[1:], horizon, axis=0
    )[:count].transpose(0, 2, 1)
    result = np.empty((count, horizon, TRAJECTORY_DIM), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (8, 7)):
        start_arm = source[:count, source_offset : source_offset + 8]
        post_arm = post[..., source_offset : source_offset + 8]
        start_q = start_arm[:, [4, 5, 6, 3]]
        post_q = post_arm[..., [4, 5, 6, 3]]
        start_norm = np.linalg.norm(start_q, axis=-1, keepdims=True)
        post_norm = np.linalg.norm(post_q, axis=-1, keepdims=True)
        if np.any(start_norm < 1.0e-8) or np.any(post_norm < 1.0e-8):
            raise ValueError("source states contain a zero quaternion")
        start_rotation = Rotation.from_quat(start_q / start_norm).as_matrix()
        post_rotation = Rotation.from_quat(
            (post_q / post_norm).reshape(-1, 4)
        ).as_matrix().reshape(count, horizon, 3, 3)
        inverse_start = np.swapaxes(start_rotation, -1, -2)
        delta_position = post_arm[..., :3] - start_arm[:, None, :3]
        result[..., target_offset : target_offset + 3] = np.einsum(
            "nij,ntj->nti", inverse_start, delta_position
        )
        relative_rotation = np.einsum("nij,ntjk->ntik", inverse_start, post_rotation)
        result[..., target_offset + 3 : target_offset + 6] = Rotation.from_matrix(
            relative_rotation.reshape(-1, 3, 3)
        ).as_rotvec().reshape(count, horizon, 3)
        result[..., target_offset + 6] = post_arm[..., 7]
    return result


def _nearest_fill(frames: np.ndarray, valid: np.ndarray) -> list[dict[str, int]]:
    missing = np.flatnonzero(~valid)
    if not len(missing):
        return []
    available = np.flatnonzero(valid)
    if not len(available):
        raise RuntimeError("an episode has no decodable RGB frames")
    repairs: list[dict[str, int]] = []
    for index in missing.tolist():
        position = int(np.searchsorted(available, index))
        candidates = available[max(0, position - 1) : min(len(available), position + 1)]
        replacement = int(min(candidates.tolist(), key=lambda value: (abs(value - index), value)))
        frames[index] = frames[replacement]
        valid[index] = True
        repairs.append({"missing_episode_frame": int(index), "replacement_episode_frame": replacement})
    return repairs


def _decode_video_episodes(
    path: Path,
    records: list[dict[str, Any]],
    callback: Callable[[dict[str, Any], np.ndarray, list[dict[str, int]]], None],
    *,
    resize_batch_size: int = 64,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["video_start_frame"]))
    if not ordered:
        return {"video_path": str(path), "corrupt_packets": [], "frame_repairs": []}
    record_index = 0
    current = ordered[0]
    current_start = int(current["video_start_frame"])
    current_end = current_start + int(current["length"])
    resized = np.empty((int(current["length"]), 3, 256, 256), dtype=np.uint8)
    valid = np.zeros(int(current["length"]), dtype=bool)
    pending_indices: list[int] = []
    pending_frames: list[np.ndarray] = []
    corrupt_packets: list[int | None] = []
    frame_repairs: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_frames:
            return
        output = resize_cam_high_batch(pending_frames)
        resized[np.asarray(pending_indices, dtype=np.int64)] = output
        valid[np.asarray(pending_indices, dtype=np.int64)] = True
        pending_indices.clear()
        pending_frames.clear()

    def finish_current() -> None:
        flush()
        repairs = _nearest_fill(resized, valid)
        for repair in repairs:
            frame_repairs.append(
                {
                    "episode_id": int(current["episode_id"]),
                    "video_path": str(path),
                    **repair,
                }
            )
        callback(current, resized, repairs)

    def advance(decoded_index: int) -> bool:
        nonlocal record_index, current, current_start, current_end, resized, valid
        while decoded_index >= current_end:
            finish_current()
            record_index += 1
            if record_index >= len(ordered):
                return False
            current = ordered[record_index]
            current_start = int(current["video_start_frame"])
            current_end = current_start + int(current["length"])
            resized = np.empty((int(current["length"]), 3, 256, 256), dtype=np.uint8)
            valid = np.zeros(int(current["length"]), dtype=bool)
        return True

    previous_index = -1
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None or stream.average_rate is None:
            raise RuntimeError(f"video {path} lacks stable PTS metadata")
        index_scale = stream.time_base * stream.average_rate
        complete = False
        for packet in container.demux(stream):
            try:
                decoded_frames = packet.decode()
            except av.error.InvalidDataError:
                approximate = None if packet.pts is None else round(packet.pts * index_scale)
                corrupt_packets.append(approximate)
                continue
            for frame in decoded_frames:
                if frame.pts is None:
                    raise RuntimeError(f"video {path} returned a frame without PTS")
                position = frame.pts * index_scale
                decoded_index = round(position)
                if abs(float(position - decoded_index)) >= 0.5 or decoded_index <= previous_index:
                    raise RuntimeError(f"video {path} has invalid PTS frame ordering")
                previous_index = decoded_index
                if decoded_index < current_start:
                    continue
                if not advance(decoded_index):
                    complete = True
                    break
                if current_start <= decoded_index < current_end:
                    pending_indices.append(decoded_index - current_start)
                    pending_frames.append(frame.to_ndarray(format="rgb24"))
                    if len(pending_frames) >= resize_batch_size:
                        flush()
            if complete:
                break
    if record_index < len(ordered):
        finish_current()
        record_index += 1
    if record_index != len(ordered):
        raise RuntimeError(f"video {path} ended before all selected episodes were decoded")
    return {
        "video_path": str(path),
        "corrupt_packets": corrupt_packets,
        "frame_repairs": frame_repairs,
    }


class _DenseEffectEncoder:
    def __init__(
        self,
        *,
        checkpoint: Path,
        model_config: Path,
        vision_model: Path | None,
        device: torch.device,
        precision: str,
        vision_batch_size: int,
        effect_batch_size: int,
        condition_source: str = "e39",
    ) -> None:
        self.device = device
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.vision_batch_size = int(vision_batch_size)
        self.effect_batch_size = int(effect_batch_size)
        self.condition_source = str(condition_source)
        self.model = load_latent_action_model(
            checkpoint,
            model_config,
            vision_model_id_override=str(vision_model.resolve()) if vision_model is not None else None,
        ).to(device).eval()
        if self.model.vq_type not in {"vae", "beta_vae"}:
            raise ValueError("dense condition export requires a Gaussian latent encoder")
        if int(self.model.num_frames) != 2:
            raise ValueError("dense condition export requires a two-frame IDM")
        if int(self.model.num_queries) != 1:
            raise ValueError("dense condition export requires exactly one latent query")
        if int(getattr(self.model.vq, "code_dim", -1)) != EFFECT_DIM:
            raise ValueError(f"dense condition export requires a {EFFECT_DIM}D latent")
        if bool(self.model.encoder.add_state):
            raise ValueError(
                "dense direct-feature export currently requires enc_add_state=false"
            )

    @torch.inference_mode()
    def encode_episode(self, frames: np.ndarray, states14: np.ndarray, horizon: int) -> np.ndarray:
        frame_tensor = torch.from_numpy(np.ascontiguousarray(frames))
        feature_batches = []
        for start in range(0, len(frame_tensor), self.vision_batch_size):
            raw = frame_tensor[start : start + self.vision_batch_size]
            valid_count = len(raw)
            if valid_count < self.vision_batch_size:
                raw = torch.cat(
                    (raw, raw[-1:].expand(self.vision_batch_size - valid_count, -1, -1, -1)),
                    dim=0,
                )
            raw = raw.to(
                self.device, non_blocking=True
            )
            videos, _ = gpu_two_view_video_aug(raw[:, None], training=False, dual_view_aug=False)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                features = self.model.vision_encoder.encode(
                    videos[:, 0], n=self.model.latent_layer_to_use
                )
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = features[:, 0] if features.ndim == 4 else features
            feature_batches.append(features[:valid_count])
        features = torch.cat(feature_batches, dim=0)
        count = len(frames) - int(horizon)
        if count < 1 or len(states14) != len(frames):
            raise ValueError("episode frames/states do not satisfy the dense horizon contract")
        means = np.empty((count, EFFECT_DIM), dtype=np.float32)
        for start in range(0, count, self.effect_batch_size):
            end = min(count, start + self.effect_batch_size)
            pairs = torch.stack(
                (features[start:end], features[start + horizon : end + horizon]), dim=1
            )
            endpoint_states = torch.from_numpy(
                np.stack((states14[start:end], states14[start + horizon : end + horizon]), axis=1)
            ).to(self.device, non_blocking=True)
            embodiment = torch.full(
                (end - start,), EMBODIMENT_INDEX, dtype=torch.long, device=self.device
            )
            valid_count = end - start
            if valid_count < self.effect_batch_size:
                padding = self.effect_batch_size - valid_count
                pairs = torch.cat((pairs, pairs[-1:].expand(padding, -1, -1, -1)), dim=0)
                endpoint_states = torch.cat(
                    (endpoint_states, endpoint_states[-1:].expand(padding, -1, -1)), dim=0
                )
                embodiment = torch.cat(
                    (embodiment, embodiment[-1:].expand(padding)), dim=0
                )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                nodes = self.model.encoder(pairs, endpoint_states, embodiment_id=embodiment)
            with torch.autocast(device_type=self.device.type, enabled=False):
                mu, _ = self.model.vq.posterior_stats(nodes.float())
            means[start:end] = mu[:valid_count, 0].float().cpu().numpy()
        if not np.isfinite(means).all():
            raise RuntimeError("dense E39 posterior contains NaN/Inf")
        return means


class _ParquetStateCache:
    def __init__(self, source: Path, file_starts: Mapping[tuple[int, int], int], commit: str) -> None:
        self.source = source
        self.file_starts = file_starts
        self.commit = commit
        self.key: tuple[int, int] | None = None
        self.states: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.digest: str | None = None
        self.inventory: dict[tuple[int, int], dict[str, Any]] = {}

    def load(self, key: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, int, str]:
        if self.key != key:
            chunk, file_index = key
            path = self.source / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
            table = parquet.read_table(
                path,
                columns=["observation.state", "action", "episode_index", "frame_index", "index"],
            )
            self.states = _list16(table.column("observation.state"), name="observation.state", path=path)
            self.actions = _list16(table.column("action"), name="action", path=path)
            global_indices = table.column("index").combine_chunks().to_numpy().astype(np.int64)
            expected = np.arange(self.file_starts[key], self.file_starts[key] + len(global_indices))
            if not np.array_equal(global_indices, expected):
                raise ValueError(f"source parquet global index mismatch: {path}")
            relative = path.relative_to(self.source).as_posix()
            self.digest = _inventory_digest(self.commit, relative, path.stat().st_size)
            self.inventory[key] = {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "inventory_sha256": self.digest,
            }
            self.key = key
        assert self.states is not None and self.actions is not None and self.digest is not None
        return self.states, self.actions, int(self.file_starts[key]), self.digest


def _episode_rows(
    record: Mapping[str, Any],
    *,
    states: np.ndarray,
    actions: np.ndarray,
    condition_mean: np.ndarray,
    source_digest: str,
    commit: str,
    horizon: int,
    condition_source: str = "e39",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = len(states) - horizon
    if count < 1 or actions.shape != states.shape or condition_mean.shape != (count, EFFECT_DIM):
        raise ValueError("episode source/condition shapes do not satisfy the dense window contract")
    mean_field, valid_field = condition_field_names(condition_source)
    trajectories = canonical_dense_windows(states, horizon)
    native_windows = np.lib.stride_tricks.sliding_window_view(
        actions, horizon, axis=0
    )[:count].transpose(0, 2, 1)
    state14 = _world_state14(states)
    anchor_indices = np.asarray(anchor_steps_for_horizon(horizon), dtype=np.int64) - 1
    active = np.ones(horizon, dtype=np.uint8)
    arm_active = np.ones((horizon, 2), dtype=np.uint8)
    native_active = np.ones((horizon, NATIVE_ACTION_DIM), dtype=np.uint8)
    state_active = np.ones(14, dtype=np.uint8)
    family = str(record["family"])
    episode_id = int(record["episode_id"])
    split = str(record["split"])
    task_id = f"robotwin/{family}"
    group = group_id(SOURCE_ID, episode_id)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for start in range(count):
        end = start + horizon
        sample_id = stable_id(SOURCE_ID, episode_id, start, end, version="robotwin-v1")
        trajectory = np.ascontiguousarray(trajectories[start])
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "group_id": group,
            "split": split,
            "task_id": task_id,
            "source_id": SOURCE_ID,
            "source_commit": commit,
            "source_sha256": source_digest,
            "controller_id": CONTROLLER_ID,
            "embodiment_id": EMBODIMENT_ID,
            "episode_id": episode_id,
            "episode_seed": episode_id,
            "start_step": start,
            "end_step": end,
            "control_hz": CONTROL_HZ,
            "dual_arm_pose_start": np.ascontiguousarray(states[start]),
            "dual_arm_pose_end": np.ascontiguousarray(states[end]),
            "waypoints": trajectory,
            "time_mask": active,
            "arm_pose_mask": arm_active,
            "arm_gripper_mask": arm_active,
            "native_actions": np.ascontiguousarray(native_windows[start]),
            "native_action_mask": native_active,
            "current_state": np.ascontiguousarray(state14[start]),
            "current_state_mask": state_active,
            mean_field: np.ascontiguousarray(condition_mean[start]),
            valid_field: 1,
            "physical_anchors": np.ascontiguousarray(trajectory[anchor_indices]),
            "physical_anchor_mask": np.ones((len(anchor_indices), TRAJECTORY_DIM), dtype=np.uint8),
        }
        row["window_sha256"] = _window_digest(row)
        manifests.append(
            {
                "sample_id": sample_id,
                "group_id": group,
                "split": split,
                "task_id": task_id,
                "effect_family": family,
                "source_id": SOURCE_ID,
                "source_commit": commit,
                "controller_id": CONTROLLER_ID,
                "episode_id": episode_id,
                "episode_seed": episode_id,
                "start_step": start,
                "end_step": end,
                "control_hz": CONTROL_HZ,
                "tau": horizon / CONTROL_HZ,
                "language_instruction": str(record["instruction"]),
                "suite": "robotwin",
            }
        )
        rows.append(row)
    return rows, manifests


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


def _barrier() -> None:
    if dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    rank, world_size, local_rank = _distributed_context()
    source = args.source_root.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    work = destination.with_name(f".{destination.name}.dense-build")
    if rank == 0:
        if destination.exists():
            raise FileExistsError(f"immutable output already exists: {destination}")
        if work.exists():
            if not args.reset_work_root:
                raise FileExistsError(f"stale dense work root exists: {work}")
            shutil.rmtree(work)
        (work / "parts").mkdir(parents=True)
    _barrier()

    info, episodes = load_source_contract(source)
    records = _selected_episode_rows(episodes, max_episodes=args.max_episodes)
    _attach_splits(records, args.split_seed)
    commit = source_commit(source)
    file_starts: dict[tuple[int, int], int] = {}
    for index, chunk in enumerate(episodes["data/chunk_index"]):
        key = (int(chunk), int(episodes["data/file_index"][index]))
        start = int(episodes["dataset_from_index"][index])
        file_starts[key] = min(start, file_starts.get(key, start))
    by_video: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        length = int(record["length"])
        if length <= args.horizon_frames:
            continue
        item = dict(record)
        item["video_start_frame"] = int(
            round(float(record["videos/observation.images.cam_high/from_timestamp"]) * CONTROL_HZ)
        )
        key = (
            int(record["videos/observation.images.cam_high/chunk_index"]),
            int(record["videos/observation.images.cam_high/file_index"]),
        )
        by_video[key].append(item)
    video_keys = sorted(by_video)
    if args.max_videos is not None:
        video_keys = video_keys[: args.max_videos]
    selected_keys = [key for index, key in enumerate(video_keys) if index % world_size == rank]
    part_root = work / "parts" / f"rank-{rank:02d}"
    part_root.mkdir(parents=True, exist_ok=True)
    writer = _DenseShardWriter(
        work,
        part_root,
        rank=rank,
        shard_rows=args.shard_rows,
        horizon=args.horizon_frames,
        condition_source=args.condition_source,
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    encoder = _DenseEffectEncoder(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        vision_model=args.vision_model,
        device=device,
        precision=args.precision,
        vision_batch_size=args.vision_batch_size,
        effect_batch_size=args.effect_batch_size,
        condition_source=args.condition_source,
    )
    cache = _ParquetStateCache(source, file_starts, commit)
    video_reports: list[dict[str, Any]] = []
    episodes_processed = 0
    for video_number, key in enumerate(selected_keys, start=1):
        chunk, file_index = key
        video_path = source / (
            f"videos/observation.images.cam_high/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        )
        video_records = by_video[key]
        data_keys = {
            (int(record["data/chunk_index"]), int(record["data/file_index"]))
            for record in video_records
        }
        if len(data_keys) != 1:
            raise ValueError(f"video {video_path} crosses source parquet files: {data_keys}")
        data_key = next(iter(data_keys))
        source_states, source_actions, file_start, source_digest = cache.load(data_key)
        print(
            f"DENSE_EXPORT rank={rank} video={video_number}/{len(selected_keys)} "
            f"path={video_path.name} episodes={len(video_records)}",
            flush=True,
        )

        def process_episode(
            record: dict[str, Any], frames: np.ndarray, repairs: list[dict[str, int]]
        ) -> None:
            nonlocal episodes_processed
            del repairs
            global_start = int(record["dataset_from_index"])
            local_start = global_start - file_start
            length = int(record["length"])
            episode_states = np.ascontiguousarray(source_states[local_start : local_start + length])
            episode_actions = np.ascontiguousarray(source_actions[local_start : local_start + length])
            if episode_states.shape != (length, 16) or episode_actions.shape != (length, 16):
                raise IndexError(f"episode {record['episode_id']} lies outside its source parquet")
            states14 = _world_state14(episode_states)
            condition_mean = encoder.encode_episode(frames, states14, args.horizon_frames)
            rows, manifests = _episode_rows(
                record,
                states=episode_states,
                actions=episode_actions,
                condition_mean=condition_mean,
                source_digest=source_digest,
                commit=commit,
                horizon=args.horizon_frames,
                condition_source=args.condition_source,
            )
            for row, manifest in zip(rows, manifests):
                writer.add(row, manifest)
            episodes_processed += 1

        video_reports.append(_decode_video_episodes(video_path, video_records, process_episode))

    counts = writer.finish()
    part_report = {
        "rank": rank,
        "world_size": world_size,
        "video_files": len(selected_keys),
        "episodes": episodes_processed,
        "split_counts": counts,
        "shard_counts": {split: int(writer.indices[split]) for split in SPLITS},
        "source_inventory": list(cache.inventory.values()),
        "video_reports": video_reports,
    }
    _atomic_json(part_root / "report.json", part_report)
    (part_root / "_SUCCESS").write_text("robotwin-dense-rank-v1\n", encoding="utf-8")
    _barrier()

    final_report: dict[str, Any] = part_report
    if rank == 0:
        for worker_rank in range(world_size):
            success = work / "parts" / f"rank-{worker_rank:02d}" / "_SUCCESS"
            if not success.is_file():
                raise RuntimeError(f"dense exporter rank {worker_rank} did not complete")
        split_counts = _combine_manifests(
            work, world_size, condition_source=args.condition_source
        )
        total_rows = sum(split_counts.values())
        expected_rows = sum(
            max(0, int(record["length"]) - args.horizon_frames)
            for key in video_keys
            for record in by_video[key]
        )
        if total_rows != expected_rows:
            raise RuntimeError(f"dense row count mismatch: emitted={total_rows} expected={expected_rows}")
        normalization_target = work / "stats" / "normalization.json"
        normalization_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.normalization_source, normalization_target)
        if _sha256_file(normalization_target) != _sha256_file(args.normalization_source):
            raise RuntimeError("copied compatibility normalization changed bytes")
        _atomic_json(
            work / "schema.json",
            schema_document(
                horizon=args.horizon_frames,
                condition_source=args.condition_source,
                dense=True,
            ),
        )
        reports = [
            json.loads(
                (work / "parts" / f"rank-{worker_rank:02d}" / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            for worker_rank in range(world_size)
        ]
        repairs = [
            repair
            for report in reports
            for video in report["video_reports"]
            for repair in video["frame_repairs"]
        ]
        corrupt_packets = [
            {"video_path": video["video_path"], "packet_frame": packet}
            for report in reports
            for video in report["video_reports"]
            for packet in video["corrupt_packets"]
        ]
        family_counts = Counter()
        for key in video_keys:
            for record in by_video[key]:
                family_counts[str(record["family"])] += max(
                    0, int(record["length"]) - args.horizon_frames
                )
        coverage = {
            "schema": "robotwin-action-dense-coverage-v1",
            "valid": True,
            "rows": total_rows,
            "split_counts": split_counts,
            "episodes": sum(int(report["episodes"]) for report in reports),
            "video_files": len(video_keys),
            "families": len(family_counts),
            "family_counts": dict(sorted(family_counts.items())),
            "horizon_actions": args.horizon_frames,
            "window_stride": 1,
            "control_hz": CONTROL_HZ,
            "duration_seconds": args.horizon_frames / CONTROL_HZ,
            "condition_source": args.condition_source,
            "condition_valid_rows": total_rows,
            "shards": sum(
                int(value)
                for report in reports
                for value in report["shard_counts"].values()
            ),
            "rgb_frame_repair_count": len(repairs),
            "corrupt_packet_count": len(corrupt_packets),
        }
        if args.condition_source == "e39":
            coverage["e39_valid_rows"] = total_rows
        else:
            coverage["lawam_valid_rows"] = total_rows
        _atomic_json(work / "coverage_report.json", coverage)
        selection_report_sha = (
            _sha256_file(args.selection_report) if args.selection_report is not None else None
        )
        condition_provenance = {
            "teacher": (
                "dino_residual_v1" if args.condition_source == "e39" else "official_lawam_release_idm"
            ),
            "posterior_statistic": "mean",
            "sampling": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "model_config": str(args.model_config.resolve()),
            "model_config_sha256": _sha256_file(args.model_config),
            "vision_model": str(args.vision_model.resolve()) if args.vision_model is not None else None,
            "precision": args.precision,
            "selection_report": str(args.selection_report.resolve()) if args.selection_report is not None else None,
            "selection_report_sha256": selection_report_sha,
            "feature_reuse": "each decoded frame is embedded once per episode, then paired as (t,t+36)",
        }
        provenance = {
            "schema": "robotwin-action-dense-provenance-v1",
            "source_root": str(source),
            "source_dataset": "jialei02/robotwin_merged",
            "source_commit": commit,
            "source_info": {
                "total_episodes": info["total_episodes"],
                "total_frames": info["total_frames"],
                "fps": info["fps"],
            },
            "window_stride": 1,
            "horizon_frames": args.horizon_frames,
            "split_seed": args.split_seed,
            "split_unit": "episode within benchmark effect family",
            "transition_alignment": "observation.state[t] -> action[t] -> observation.state[t+1]",
            "trajectory_target": "36 actual post-action states in start-frame coordinates",
            "rgb_storage": "none; source video is streamed and DINO features are consumed immediately",
            "rgb_repair_policy": "nearest valid PTS frame within the same episode; ties choose the earlier frame",
            "rgb_frame_repairs": repairs,
            "corrupt_packets": corrupt_packets,
            "normalization": {
                "policy": "frozen_1.2s_compatibility_bridge",
                "source": str(args.normalization_source.resolve()),
                "sha256": _sha256_file(normalization_target),
                "reason": "same source episodes, horizon, controller, and trajectory semantics",
            },
            "condition": {"source": args.condition_source, **condition_provenance},
        }
        provenance[args.condition_source] = condition_provenance
        _atomic_json(work / "provenance.json", provenance)
        artifact_version = "v2" if args.condition_source == "lawam_idm" else "v1"
        (work / "_SUCCESS").write_text(
            f"robotwin-actionvae-{args.condition_source}-dense-stride1-{artifact_version}\n",
            encoding="utf-8",
        )
        recovery_audit = audit_release(work, verify_checksums=False)
        if recovery_audit.get("valid") is not True or recovery_audit.get("issue_count") != 0:
            raise RuntimeError(
                "dense sidecar failed its final semantic audit: "
                f"{recovery_audit.get('issues', [])[:8]}"
            )
        _atomic_json(work / "recovery_audit.json", recovery_audit)
        _write_checksums(work)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work, destination)
        final_report = {**coverage, "output": str(destination)}
        print(json.dumps(final_report, sort_keys=True), flush=True)
    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return final_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument(
        "--condition-source", choices=("e39", "lawam_idm"), default="e39"
    )
    parser.add_argument("--normalization-source", required=True, type=Path)
    parser.add_argument("--horizon-frames", type=int, default=36)
    parser.add_argument("--shard-rows", type=int, default=8192)
    parser.add_argument("--split-seed", type=int, default=20260818)
    parser.add_argument("--vision-batch-size", type=int, default=128)
    parser.add_argument("--effect-batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--reset-work-root", action="store_true")
    args = parser.parse_args(argv)
    for name in ("source_root", "checkpoint", "model_config", "normalization_source"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.vision_model is not None and not args.vision_model.exists():
        parser.error(f"--vision-model does not exist: {args.vision_model}")
    if args.selection_report is not None and not args.selection_report.is_file():
        parser.error(f"--selection-report does not exist: {args.selection_report}")
    if args.horizon_frames != 36:
        parser.error("the reviewed dense RoboTwin recipe requires horizon_frames=36")
    if min(args.shard_rows, args.vision_batch_size, args.effect_batch_size) < 1:
        parser.error("shard and inference batch sizes must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    build_release(parse_args(argv))


if __name__ == "__main__":
    main()
