"""Build and read the 0.4s LIBERO Stage-2 teacher cache.

The source sample identity remains the merged LeRobot ``(episode_id,
start_step)`` pair.  This module only attaches frozen Effect/ActionVAE teacher
targets; it never owns sampling or retry policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy.spatial.transform import Rotation

from latent_action_model.action_ae.beta_vae import StandardNormalActionVAE
from latent_action_model.action_ae.data import ActionNormalization
from latent_action_model.action_ae.evaluate import load_action_ae_checkpoint
from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug

from .interventions import LatentStandardizer
from .state_transform import (
    ACTIONVAE_RAW_STATE_DIM,
    ACTIONVAE_VALID_STATE_DIM,
    build_libero_actionvae_raw_state,
)
from .teacher_bundle import (
    LIBERO_12S_ACTION_CHECKPOINT,
    LIBERO_12S_ACTION_CHECKPOINT_SHA256,
    LIBERO_12S_ACTION_CONFIG,
    LIBERO_12S_ACTION_CONFIG_SHA256,
    LIBERO_12S_ACTION_DATA_ROOT,
    LIBERO_12S_ACTION_NORMALIZATION_SHA256,
    LIBERO_12S_ACTION_SCHEMA_VERSION,
    LIBERO_12S_EFFECT_CHECKPOINT,
    LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
    LIBERO_12S_EFFECT_CONFIG,
    LIBERO_12S_EFFECT_CONFIG_SHA256,
)
from .temporal_adapter_04s import (
    ACTIONVAE_12S_REPRESENTATION_STEPS,
    CANONICAL_WAYPOINT_DIM,
    LIBERO_04S_CONTROL_STEPS,
    adapt_libero_04s_states,
)


CACHE_SCHEMA_VERSION = "libero-stage2-latent-cache-v3"
NORMALIZATION_SCHEMA_VERSION = "libero-stage2-latent-normalization-bundle-v3"
TEMPORAL_CONTRACT = "libero_baseline_04s_adapter_v1"
SOURCE_DATASET_NAME = "libero_merged_no_noops_20hz"
SOURCE_ROW_COUNT = 273_465
SOURCE_EPISODE_COUNT = 1_693
VALIDATION_ROW_COUNT = 1_941
CONTROL_HZ = 20.0
HORIZON_SECONDS = 0.4
DINO_FUTURE_DELTA_STEPS = 7
EFFECT_TEACHER_DELTA_STEPS = 8
LATENT_DIM = 32
LIBERO_EMBODIMENT_ID = 25
DEFAULT_DATASET_ROOT = Path("data/libero_merged_no_noops_20hz")
DEFAULT_OUTPUT_ROOT = Path(
    "data/libero/policy_cache"
)

_MANIFEST_REQUIRED = {
    "latent_row",
    "sample_id",
    "episode_id",
    "start_step",
    "source_index",
    "train_membership",
    "validation_membership",
}
_ATTACH_FIELDS = (
    "z_effect",
    "z_effect_raw",
    "z_action",
    "z_action_logvar",
    "teacher_trajectory",
    "teacher_current_state",
    "teacher_time_mask",
    "teacher_pose_mask",
    "teacher_gripper_mask",
    "teacher_current_state_mask",
)
_RAW_SHARD_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "global_row": ((), np.dtype(np.int64)),
    "z_effect_raw": ((LATENT_DIM,), np.dtype(np.float32)),
    "z_action_mean_raw": ((LATENT_DIM,), np.dtype(np.float32)),
    "z_action_logvar_raw": ((LATENT_DIM,), np.dtype(np.float32)),
    "teacher_trajectory": (
        (ACTIONVAE_12S_REPRESENTATION_STEPS, CANONICAL_WAYPOINT_DIM),
        np.dtype(np.float32),
    ),
    "teacher_trajectory_raw": (
        (ACTIONVAE_12S_REPRESENTATION_STEPS, CANONICAL_WAYPOINT_DIM),
        np.dtype(np.float32),
    ),
    "teacher_current_state": ((ACTIONVAE_RAW_STATE_DIM,), np.dtype(np.float32)),
    "teacher_current_state_raw": ((ACTIONVAE_RAW_STATE_DIM,), np.dtype(np.float32)),
    "teacher_time_mask": ((ACTIONVAE_12S_REPRESENTATION_STEPS,), np.dtype(np.uint8)),
    "teacher_pose_mask": ((ACTIONVAE_12S_REPRESENTATION_STEPS,), np.dtype(np.uint8)),
    "teacher_gripper_mask": ((ACTIONVAE_12S_REPRESENTATION_STEPS,), np.dtype(np.uint8)),
    "teacher_current_state_mask": ((ACTIONVAE_RAW_STATE_DIM,), np.dtype(np.uint8)),
    "native_actions": ((LIBERO_04S_CONTROL_STEPS, 7), np.dtype(np.float32)),
    "native_action_mask": ((LIBERO_04S_CONTROL_STEPS, 7), np.dtype(np.uint8)),
    "physical_valid_count": ((), np.dtype(np.uint8)),
    "source_available_action_count": ((), np.dtype(np.uint8)),
    "effect_future_step": ((), np.dtype(np.int32)),
    "dino_future_step": ((), np.dtype(np.int32)),
    "episode_tail_step": ((), np.dtype(np.int32)),
    "effect_future_clamped": ((), np.dtype(np.uint8)),
    "dino_future_clamped": ((), np.dtype(np.uint8)),
}


def _json_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: str | Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sample_id(
    episode_id: int,
    start_step: int,
    *,
    dataset_name: str = SOURCE_DATASET_NAME,
) -> str:
    return f"{dataset_name}:{int(episode_id)}:{int(start_step)}"


def _split_range(count: int, rank: int, world_size: int) -> tuple[int, int]:
    if count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed range")
    return count * rank // world_size, count * (rank + 1) // world_size


def _source_release_files(root: Path) -> list[Path]:
    required = [
        root / "meta/info.json",
        root / "meta/modality.json",
        root / "meta/tasks.parquet",
    ]
    required.extend(sorted((root / "meta/episodes").glob("*/*.parquet")))
    required.extend(sorted((root / "data").glob("*/*.parquet")))
    required.extend(sorted((root / "videos").glob("*/*/*.mp4")))
    missing = [path for path in required[:3] if not path.is_file()]
    if missing or not any("/data/" in str(path) for path in required) or not any(
        "/videos/" in str(path) for path in required
    ):
        raise FileNotFoundError(f"incomplete merged LeRobot release: {missing or root}")
    return required


def source_release_fingerprint(root: str | Path) -> dict[str, Any]:
    source = Path(root).expanduser().resolve()
    files = _source_release_files(source)
    entries = []
    aggregate = hashlib.sha256()
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
        "schema_version": "libero-merged-release-fingerprint-v1",
        "root": str(source),
        "file_count": len(entries),
        "files": entries,
        "aggregate_sha256": aggregate.hexdigest(),
    }


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()


def _distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("multi-rank cache production requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(rank, world_size, local_rank, initialized_here)


class MergedLibero04sSource:
    """Validated in-memory scalar/state index plus batched AV1 endpoint reader."""

    def __init__(self, root: str | Path, *, max_samples: int | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = SOURCE_DATASET_NAME
        info = _json_read(self.root / "meta/info.json")
        if self.root.name != SOURCE_DATASET_NAME:
            raise ValueError(f"dataset directory must be named {SOURCE_DATASET_NAME!r}")
        if float(info.get("fps", 0)) != CONTROL_HZ:
            raise ValueError("merged LIBERO source must be 20Hz")
        if int(info.get("total_frames", -1)) != SOURCE_ROW_COUNT:
            raise ValueError("merged LIBERO source row count mismatch")
        if int(info.get("total_episodes", -1)) != SOURCE_EPISODE_COUNT:
            raise ValueError("merged LIBERO source episode count mismatch")

        data_paths = sorted((self.root / "data").glob("*/*.parquet"))
        episode_paths = sorted((self.root / "meta/episodes").glob("*/*.parquet"))
        if not data_paths or not episode_paths:
            raise FileNotFoundError("merged LIBERO parquet release is incomplete")
        frame = pd.concat(
            [
                pd.read_parquet(
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
                for path in data_paths
            ],
            ignore_index=True,
        )
        episodes = pd.concat(
            [pd.read_parquet(path) for path in episode_paths], ignore_index=True
        ).sort_values("dataset_from_index", kind="stable")
        if len(frame) != SOURCE_ROW_COUNT or len(episodes) != SOURCE_EPISODE_COUNT:
            raise ValueError("merged LIBERO parquet rows disagree with info.json")
        source_index = frame["index"].to_numpy(dtype=np.int64)
        if not np.array_equal(source_index, np.arange(SOURCE_ROW_COUNT, dtype=np.int64)):
            raise ValueError("merged LIBERO global index must be contiguous and row aligned")
        self.states = np.stack(frame["observation.state"].to_numpy()).astype(
            np.float32, copy=False
        )
        self.actions = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)
        self.timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        self.frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
        self.episode_indices = frame["episode_index"].to_numpy(dtype=np.int64)
        self.task_indices = frame["task_index"].to_numpy(dtype=np.int64)
        if self.states.shape != (SOURCE_ROW_COUNT, 8) or self.actions.shape != (
            SOURCE_ROW_COUNT,
            7,
        ):
            raise ValueError("merged LIBERO state/action tensor shape mismatch")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("merged LIBERO state/action contains NaN/Inf")

        expected_from = np.concatenate(
            ([0], episodes["dataset_to_index"].to_numpy(dtype=np.int64)[:-1])
        )
        if not np.array_equal(
            episodes["dataset_from_index"].to_numpy(dtype=np.int64), expected_from
        ) or int(episodes.iloc[-1]["dataset_to_index"]) != SOURCE_ROW_COUNT:
            raise ValueError("episode ranges do not form an exact source partition")
        self.episodes = episodes.reset_index(drop=True)
        self._episode_by_id = {
            int(row.episode_index): row
            for row in self.episodes.itertuples(index=False)
        }
        self._episode_record_by_id = {
            int(record["episode_index"]): record
            for record in self.episodes.to_dict(orient="records")
        }
        for row in self.episodes.itertuples(index=False):
            start = int(row.dataset_from_index)
            end = int(row.dataset_to_index)
            expected_steps = np.arange(end - start, dtype=np.int64)
            if not np.array_equal(self.frame_indices[start:end], expected_steps):
                raise ValueError(f"episode {row.episode_index} has non-contiguous frame_index")
            if not np.all(self.episode_indices[start:end] == int(row.episode_index)):
                raise ValueError(f"episode {row.episode_index} row identity mismatch")

        tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
        self.task_text = {
            int(value): str(index)
            for index, value in zip(tasks.index.tolist(), tasks["task_index"].tolist())
        }
        if set(np.unique(self.task_indices)).difference(self.task_text):
            raise ValueError("tasks.parquet does not cover all source task indices")

        raw_rotations = self.states[:, 3:6].astype(np.float64)
        raw_norm = np.linalg.norm(raw_rotations, axis=1)
        raw_rotation = Rotation.from_rotvec(raw_rotations)
        canonical_rotation = Rotation.from_rotvec(raw_rotation.as_rotvec())
        equivalence_error = (raw_rotation.inv() * canonical_rotation).magnitude()
        max_error = float(np.max(equivalence_error))
        if max_error > 1e-10:
            raise ValueError(
                "principal-rotvec canonicalization changed physical orientation: "
                f"max_error={max_error}"
            )
        self.canonical_rotvec = raw_rotation.as_rotvec().astype(np.float32)
        self.canonicalization = {
            "rule": (
                "Rotation.from_rotvec(raw).as_quat() followed by canonical principal "
                "rotvec; no norm clipping or component truncation"
            ),
            "raw_norm_gt_pi_count": int(np.sum(raw_norm > np.pi + 1e-5)),
            "raw_row_count": int(len(raw_norm)),
            "raw_norm_max": float(np.max(raw_norm)),
            "canonical_norm_max": float(
                np.max(np.linalg.norm(self.canonical_rotvec.astype(np.float64), axis=1))
            ),
            "max_rotation_equivalence_error_rad": max_error,
            "equivalence_gate_rad": 1e-10,
        }

        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.selected_count = min(
            SOURCE_ROW_COUNT,
            SOURCE_ROW_COUNT if max_samples is None else int(max_samples),
        )
        if self.selected_count < 2:
            raise ValueError("latent standardization requires at least two selected rows")
        self.release_level = "full" if self.selected_count == SOURCE_ROW_COUNT else "smoke"
        validation_episode_count = int(np.floor(SOURCE_EPISODE_COUNT * 0.01))
        self.validation_episode_ids = set(
            self.episodes.tail(validation_episode_count)["episode_index"]
            .astype(int)
            .tolist()
        )

    def _episode_location(self, source_index: int) -> tuple[Any, int]:
        episode_id = int(self.episode_indices[source_index])
        row = self._episode_by_id[episode_id]
        local_step = source_index - int(row.dataset_from_index)
        if local_step != int(self.frame_indices[source_index]):
            raise ValueError("source global/local identity mismatch")
        return row, local_step

    def raw_state14(self, source_index: int) -> tuple[np.ndarray, np.ndarray]:
        state = self.states[source_index]
        return build_libero_actionvae_raw_state(
            state[:3], self.canonical_rotvec[source_index], state[6:8]
        )

    def structural_batch(
        self,
        source_indices: Sequence[int],
        normalization: ActionNormalization,
    ) -> dict[str, np.ndarray]:
        count = len(source_indices)
        raw_trajectory = np.empty((count, 24, 7), dtype=np.float32)
        trajectory = np.empty_like(raw_trajectory)
        raw_current = np.empty((count, 14), dtype=np.float32)
        current = np.empty_like(raw_current)
        current_mask = np.empty((count, 14), dtype=np.uint8)
        endpoint_states = np.empty((count, 2, 14), dtype=np.float32)
        endpoint_state_masks = np.empty((count, 2, 14), dtype=np.uint8)
        native_actions = np.empty((count, 8, 7), dtype=np.float32)
        native_action_mask = np.ones((count, 8, 7), dtype=np.uint8)
        source_available_action_count = np.empty(count, dtype=np.uint8)
        effect_step = np.empty(count, dtype=np.int32)
        dino_step = np.empty(count, dtype=np.int32)
        tail_step = np.empty(count, dtype=np.int32)
        effect_clamped = np.empty(count, dtype=np.uint8)
        dino_clamped = np.empty(count, dtype=np.uint8)
        representation_mask = np.ones((count, 24), dtype=np.uint8)

        for offset, raw_index in enumerate(source_indices):
            source_index = int(raw_index)
            row, start_step = self._episode_location(source_index)
            episode_from = int(row.dataset_from_index)
            episode_length = int(row.length)
            tail = episode_length - 1
            state_local = np.minimum(
                start_step + np.arange(EFFECT_TEACHER_DELTA_STEPS + 1), tail
            )
            state_global = episode_from + state_local
            adapted = adapt_libero_04s_states(self.states[state_global], 0)
            if adapted.physical_valid_count != LIBERO_04S_CONTROL_STEPS:
                raise ValueError("temporal adapter changed physical_valid_count")
            raw_trajectory[offset] = adapted.waypoints

            raw_state, raw_mask = self.raw_state14(source_index)
            raw_current[offset] = raw_state
            current_mask[offset] = raw_mask.astype(np.uint8)
            effect_local = min(start_step + EFFECT_TEACHER_DELTA_STEPS, tail)
            effect_global = episode_from + effect_local
            endpoint_state, endpoint_mask = self.raw_state14(effect_global)
            endpoint_states[offset, 0] = raw_state
            endpoint_states[offset, 1] = endpoint_state
            endpoint_state_masks[offset, 0] = raw_mask
            endpoint_state_masks[offset, 1] = endpoint_mask

            action_local = np.minimum(
                start_step + np.arange(LIBERO_04S_CONTROL_STEPS), tail
            )
            native_actions[offset] = self.actions[episode_from + action_local]
            source_available_action_count[offset] = min(
                LIBERO_04S_CONTROL_STEPS, episode_length - start_step
            )
            effect_step[offset] = effect_local
            dino_step[offset] = min(start_step + DINO_FUTURE_DELTA_STEPS, tail)
            tail_step[offset] = tail
            effect_clamped[offset] = int(start_step + EFFECT_TEACHER_DELTA_STEPS > tail)
            dino_clamped[offset] = int(start_step + DINO_FUTURE_DELTA_STEPS > tail)

            time = torch.from_numpy(adapted.representation_mask.copy())
            raw_trajectory_tensor = torch.from_numpy(adapted.waypoints.copy())
            normalized_trajectory, _ = normalization.trajectory(
                raw_trajectory_tensor,
                time_mask=time,
                pose_mask=time,
                gripper_mask=time,
            )
            trajectory[offset] = normalized_trajectory.numpy()
            current[offset] = normalization.current_state(
                torch.from_numpy(raw_state.copy()), torch.from_numpy(raw_mask.copy())
            ).numpy()

        return {
            "teacher_trajectory": trajectory,
            "teacher_trajectory_raw": raw_trajectory,
            "teacher_current_state": current,
            "teacher_current_state_raw": raw_current,
            "teacher_time_mask": representation_mask,
            "teacher_pose_mask": representation_mask.copy(),
            "teacher_gripper_mask": representation_mask.copy(),
            "teacher_current_state_mask": current_mask,
            "native_actions": native_actions,
            "native_action_mask": native_action_mask,
            "physical_valid_count": np.full(count, 8, dtype=np.uint8),
            "source_available_action_count": source_available_action_count,
            "effect_future_step": effect_step,
            "dino_future_step": dino_step,
            "episode_tail_step": tail_step,
            "effect_future_clamped": effect_clamped,
            "dino_future_clamped": dino_clamped,
            "effect_endpoint_states": endpoint_states,
            "effect_endpoint_state_masks": endpoint_state_masks,
        }

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
            episode_record = self._episode_record_by_id[int(row.episode_index)]
            chunk = int(episode_record["videos/observation.images.image/chunk_index"])
            file_index = int(episode_record["videos/observation.images.image/file_index"])
            file_path = self.root / (
                "videos/observation.images.image/"
                f"chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            )
            from_timestamp = float(
                episode_record["videos/observation.images.image/from_timestamp"]
            )
            grouped.setdefault(file_path, []).append(
                (
                    offset,
                    from_timestamp + start_step / CONTROL_HZ,
                    from_timestamp + effect_step / CONTROL_HZ,
                )
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

    def manifest(self) -> pd.DataFrame:
        indices = np.arange(self.selected_count, dtype=np.int64)
        episodes = self.episode_indices[: self.selected_count]
        steps = self.frame_indices[: self.selected_count]
        task_indices = self.task_indices[: self.selected_count]
        tail_by_episode = {
            int(row.episode_index): int(row.length) - 1
            for row in self.episodes.itertuples(index=False)
        }
        tails = np.asarray([tail_by_episode[int(value)] for value in episodes], dtype=np.int64)
        effect = np.minimum(steps + EFFECT_TEACHER_DELTA_STEPS, tails)
        dino = np.minimum(steps + DINO_FUTURE_DELTA_STEPS, tails)
        frame = pd.DataFrame(
            {
                "latent_row": indices,
                "sample_id": [
                    _sample_id(
                        int(episode),
                        int(step),
                        dataset_name=self.dataset_name,
                    )
                    for episode, step in zip(episodes, steps)
                ],
                "dataset_name": self.dataset_name,
                "split": "train",
                "train_membership": True,
                "validation_membership": [
                    int(value) in self.validation_episode_ids for value in episodes
                ],
                "task_id": [str(int(value)) for value in task_indices],
                "task_index": task_indices,
                "language_instruction": [self.task_text[int(value)] for value in task_indices],
                "episode_id": episodes,
                "start_step": steps,
                "source_index": indices,
                "effect_future_unclamped_step": steps + EFFECT_TEACHER_DELTA_STEPS,
                "effect_future_step": effect,
                "effect_future_clamped": effect != steps + EFFECT_TEACHER_DELTA_STEPS,
                "dino_future_unclamped_step": steps + DINO_FUTURE_DELTA_STEPS,
                "dino_future_step": dino,
                "dino_future_clamped": dino != steps + DINO_FUTURE_DELTA_STEPS,
                "episode_tail_step": tails,
                "physical_valid_count": LIBERO_04S_CONTROL_STEPS,
                "source_available_action_count": np.minimum(
                    LIBERO_04S_CONTROL_STEPS, tails - steps + 1
                ),
                "representation_steps": ACTIONVAE_12S_REPRESENTATION_STEPS,
            }
        )
        if frame[["episode_id", "start_step"]].duplicated().any():
            raise ValueError("merged source identity is not unique")
        return frame


def _teacher_paths_and_hashes() -> dict[str, Any]:
    normalization = LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json"
    paths = {
        "effect_checkpoint": LIBERO_12S_EFFECT_CHECKPOINT.resolve(),
        "effect_config": LIBERO_12S_EFFECT_CONFIG.resolve(),
        "action_checkpoint": LIBERO_12S_ACTION_CHECKPOINT.resolve(),
        "action_config": LIBERO_12S_ACTION_CONFIG.resolve(),
        "action_normalization": normalization.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {
        "effect_checkpoint": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "effect_config": LIBERO_12S_EFFECT_CONFIG_SHA256,
        "action_checkpoint": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
        "action_config": LIBERO_12S_ACTION_CONFIG_SHA256,
        "action_normalization": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError(f"selected teacher asset hash mismatch: {name}")
    return {
        name: {"path": str(path), "sha256": actual[name]}
        for name, path in paths.items()
    }


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


def _load_teachers(device: torch.device) -> tuple[Any, Any, ActionNormalization]:
    effect = load_latent_action_model(
        LIBERO_12S_EFFECT_CHECKPOINT,
        LIBERO_12S_EFFECT_CONFIG,
        vision_model_id_override=os.environ.get("CLEAR_DINO_PATH", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ).to(device)
    effect.requires_grad_(False).eval()
    action = load_action_ae_checkpoint(
        LIBERO_12S_ACTION_CHECKPOINT,
        LIBERO_12S_ACTION_DATA_ROOT,
        device=device,
    )
    if action.checkpoint_sha256 != LIBERO_12S_ACTION_CHECKPOINT_SHA256:
        raise ValueError("selected ActionVAE checkpoint hash mismatch")
    if action.metadata.get("schema_version") != LIBERO_12S_ACTION_SCHEMA_VERSION:
        raise ValueError("selected ActionVAE schema mismatch")
    if not isinstance(action.model, StandardNormalActionVAE):
        raise TypeError("0.4s adapter requires the selected standard-normal ActionVAE")
    config = action.model.config
    if (config.horizon, config.state_dim, config.latent_dim, config.variant) != (
        24,
        14,
        32,
        "a1",
    ):
        raise ValueError("selected ActionVAE architecture mismatch")
    normalization_path = LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json"
    normalization = ActionNormalization.from_file(
        normalization_path,
        expected_schema_version="libero-z-action-normalization-v1-12s",
    )
    return effect, action, normalization


def _create_partial_shard(
    path: Path,
    *,
    start: int,
    end: int,
    rank: int,
    world_size: int,
    contract_digest: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
    temporal_contract: str = TEMPORAL_CONTRACT,
) -> None:
    row_count = end - start
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = schema_version
        handle.attrs["stage2_temporal_contract"] = temporal_contract
        handle.attrs["rank"] = rank
        handle.attrs["world_size"] = world_size
        handle.attrs["global_start"] = start
        handle.attrs["global_end"] = end
        handle.attrs["contract_sha256"] = contract_digest
        handle.attrs["complete"] = False
        chunk_rows = max(1, min(256, row_count))
        for name, (tail_shape, dtype) in _RAW_SHARD_SPECS.items():
            chunks = (chunk_rows, *tail_shape)
            handle.create_dataset(
                name,
                shape=(row_count, *tail_shape),
                dtype=dtype,
                chunks=chunks,
            )
        handle["global_row"][:] = np.arange(start, end, dtype=np.int64)
        handle.create_dataset(
            "filled",
            shape=(row_count,),
            dtype=np.uint8,
            chunks=(chunk_rows,),
            fillvalue=0,
        )
        handle.flush()


def _validate_partial_shard(
    handle: h5py.File,
    *,
    start: int,
    end: int,
    rank: int,
    world_size: int,
    contract_digest: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
    temporal_contract: str = TEMPORAL_CONTRACT,
) -> None:
    expected_attrs = {
        "schema_version": schema_version,
        "stage2_temporal_contract": temporal_contract,
        "rank": rank,
        "world_size": world_size,
        "global_start": start,
        "global_end": end,
        "contract_sha256": contract_digest,
    }
    for name, expected in expected_attrs.items():
        if handle.attrs.get(name) != expected:
            raise ValueError(f"resume shard attribute mismatch for {name}")
    expected_rows = np.arange(start, end, dtype=np.int64)
    if not np.array_equal(handle["global_row"][:], expected_rows):
        raise ValueError("resume shard global-row identity mismatch")
    for name, (tail, dtype) in _RAW_SHARD_SPECS.items():
        if name not in handle or handle[name].shape != (end - start, *tail):
            raise ValueError(f"resume shard field shape mismatch: {name}")
        if np.dtype(handle[name].dtype) != dtype:
            raise ValueError(f"resume shard field dtype mismatch: {name}")


@torch.inference_mode()
def _encode_rank_shard(
    *,
    source: MergedLibero04sSource,
    work_root: Path,
    contract: Mapping[str, Any],
    rank: int,
    world_size: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    resume: bool,
    progress_every: int,
    teacher_loader: Any = None,
) -> dict[str, Any]:
    schema_version = str(contract.get("schema_version", CACHE_SCHEMA_VERSION))
    temporal_contract = str(
        contract.get("stage2_temporal_contract", TEMPORAL_CONTRACT)
    )
    start, end = _split_range(source.selected_count, rank, world_size)
    rank_root = work_root / f"rank-{rank:05d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    contract_digest = _sha256_json(contract)
    contract_path = rank_root / "contract.json"
    rank_contract = {
        **dict(contract),
        "rank": rank,
        "global_start": start,
        "global_end": end,
        "contract_sha256": contract_digest,
    }
    if contract_path.exists():
        if _json_read(contract_path) != rank_contract:
            raise ValueError(f"rank {rank} resume contract mismatch")
    else:
        _json_write_atomic(contract_path, rank_contract)

    final_path = rank_root / f"part-{rank:05d}.h5"
    partial_path = rank_root / f"part-{rank:05d}.partial.h5"
    success_path = rank_root / "_SUCCESS"
    if final_path.exists():
        if not resume or not success_path.is_file():
            raise FileExistsError(f"rank shard already exists: {final_path}")
        with h5py.File(final_path, "r") as handle:
            _validate_partial_shard(
                handle,
                start=start,
                end=end,
                rank=rank,
                world_size=world_size,
                contract_digest=contract_digest,
                schema_version=schema_version,
                temporal_contract=temporal_contract,
            )
            if not bool(handle.attrs.get("complete", False)) or not np.all(
                handle["filled"][:] == 1
            ):
                raise ValueError(f"completed rank shard is incomplete: {final_path}")
        return _json_read(success_path)
    if not partial_path.exists():
        _create_partial_shard(
            partial_path,
            start=start,
            end=end,
            rank=rank,
            world_size=world_size,
            contract_digest=contract_digest,
            schema_version=schema_version,
            temporal_contract=temporal_contract,
        )
    elif not resume:
        raise FileExistsError(f"partial rank shard exists; pass --resume: {partial_path}")

    effect_model, loaded_action, normalization = (teacher_loader or _load_teachers)(device)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with h5py.File(partial_path, "r+") as handle:
        _validate_partial_shard(
            handle,
            start=start,
            end=end,
            rank=rank,
            world_size=world_size,
            contract_digest=contract_digest,
            schema_version=schema_version,
            temporal_contract=temporal_contract,
        )
        missing_local = np.flatnonzero(handle["filled"][:] == 0)
        for batch_number, begin in enumerate(range(0, len(missing_local), batch_size), 1):
            local_rows = missing_local[begin : begin + batch_size]
            global_rows = start + local_rows
            structure = source.structural_batch(global_rows.tolist(), normalization)
            frames = source.decode_effect_pairs(global_rows.tolist())
            videos, dec_videos = gpu_two_view_video_aug(
                torch.from_numpy(frames).to(device, non_blocking=True),
                training=False,
                dual_view_aug=False,
            )
            endpoint_states = torch.from_numpy(structure.pop("effect_endpoint_states")).to(
                device, non_blocking=True
            )
            endpoint_masks = torch.from_numpy(
                structure.pop("effect_endpoint_state_masks")
            ).to(device=device, dtype=torch.bool, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and precision != "fp32",
            ):
                effect_output = effect_model.get_latent_action(
                    videos=videos,
                    states=endpoint_states,
                    dec_videos=dec_videos,
                    state_mask=endpoint_masks,
                    embodiment_ids=torch.full(
                        (len(global_rows),),
                        LIBERO_EMBODIMENT_ID,
                        device=device,
                        dtype=torch.long,
                    ),
                    predict_future_frame=False,
                )
            effect_mean = effect_output.get("latent_mu")
            if effect_mean is None or tuple(effect_mean.shape) != (
                len(global_rows),
                1,
                LATENT_DIM,
            ):
                raise ValueError("Effect teacher returned an invalid posterior mean")
            effect_mean = effect_mean[:, 0].float()
            action_posterior = loaded_action.model.posterior(
                torch.from_numpy(structure["teacher_trajectory"]).to(
                    device, non_blocking=True
                ),
                torch.from_numpy(structure["teacher_current_state"]).to(
                    device, non_blocking=True
                ),
                effect_mean,
                time_mask=torch.from_numpy(structure["teacher_time_mask"]).to(
                    device=device, dtype=torch.bool, non_blocking=True
                ),
            )
            values: dict[str, np.ndarray] = {
                **structure,
                "z_effect_raw": effect_mean.cpu().numpy(),
                "z_action_mean_raw": action_posterior.mean.float().cpu().numpy(),
                "z_action_logvar_raw": action_posterior.logvar.float().cpu().numpy(),
            }
            for name, (tail, dtype) in _RAW_SHARD_SPECS.items():
                if name == "global_row":
                    continue
                value = np.asarray(values[name], dtype=dtype)
                if value.shape != (len(local_rows), *tail) or not np.isfinite(value).all():
                    raise ValueError(f"invalid rank cache value for {name}: {value.shape}")
                handle[name][local_rows] = value
            handle["filled"][local_rows] = 1
            handle.flush()
            if progress_every > 0 and batch_number % progress_every == 0:
                print(
                    f"BASELINE04_CACHE rank={rank} rows={int(np.sum(handle['filled'][:]))}/"
                    f"{end-start}",
                    flush=True,
                )
        if not np.all(handle["filled"][:] == 1):
            raise RuntimeError(f"rank {rank} cache extraction did not fill every row")
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(partial_path, final_path)
    success = {
        "schema_version": schema_version,
        "stage2_temporal_contract": temporal_contract,
        "rank": rank,
        "world_size": world_size,
        "global_start": start,
        "global_end": end,
        "row_count": end - start,
        "contract_sha256": contract_digest,
        "shard_sha256": sha256_file(final_path),
        "complete": True,
    }
    _json_write_atomic(success_path, success)
    return success


def _copy_rank_shards(
    output: h5py.File,
    *,
    work_root: Path,
    row_count: int,
    world_size: int,
    contract_digest: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
    temporal_contract: str = TEMPORAL_CONTRACT,
) -> None:
    seen = np.zeros(row_count, dtype=np.uint8)
    for rank in range(world_size):
        start, end = _split_range(row_count, rank, world_size)
        rank_root = work_root / f"rank-{rank:05d}"
        shard = rank_root / f"part-{rank:05d}.h5"
        success = _json_read(rank_root / "_SUCCESS")
        if (
            success.get("contract_sha256") != contract_digest
            or int(success.get("global_start", -1)) != start
            or int(success.get("global_end", -1)) != end
            or success.get("shard_sha256") != sha256_file(shard)
        ):
            raise ValueError(f"rank {rank} success marker mismatch")
        with h5py.File(shard, "r") as source:
            _validate_partial_shard(
                source,
                start=start,
                end=end,
                rank=rank,
                world_size=world_size,
                contract_digest=contract_digest,
                schema_version=schema_version,
                temporal_contract=temporal_contract,
            )
            if not bool(source.attrs.get("complete", False)):
                raise ValueError(f"rank {rank} shard is not complete")
            for name in _RAW_SHARD_SPECS:
                output[name][start:end] = source[name][:]
        if np.any(seen[start:end]):
            raise ValueError("rank shard ranges overlap")
        seen[start:end] = 1
    if not np.all(seen == 1):
        raise ValueError("rank shard ranges do not cover every selected row")


def _publish_cache(
    *,
    source: MergedLibero04sSource,
    output_root: Path,
    work_root: Path,
    contract: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
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
            output.attrs["schema_version"] = CACHE_SCHEMA_VERSION
            output.attrs["stage2_temporal_contract"] = TEMPORAL_CONTRACT
            output.attrs["release_level"] = source.release_level
            output.attrs["complete"] = False
            chunk_rows = max(1, min(256, row_count))
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
        fit_membership = "all_rows" if source.release_level == "full" else "selected_smoke_rows"
        normalization = {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "stage2_temporal_contract": TEMPORAL_CONTRACT,
            "fit_split": "train",
            "fit_membership": fit_membership,
            "fit_row_count": row_count,
            "z_effect": effect_standardizer.to_dict(),
            "z_action": action_standardizer.to_dict(),
            "action_logvar_standardization": "logvar_raw-2*log(max(action_std,epsilon))",
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
            "identity": ["episode_id", "start_step"],
            "source": {
                "dataset_root": str(source.root),
                "dataset_name": SOURCE_DATASET_NAME,
                "row_count": SOURCE_ROW_COUNT,
                "episode_count": SOURCE_EPISODE_COUNT,
                "release_fingerprint": contract["source_fingerprint"],
                "temporal_contract": _temporal_metadata(),
                "orientation_canonicalization": source.canonicalization,
                "action_checkpoint": str(LIBERO_12S_ACTION_CHECKPOINT.resolve()),
                "action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
                "action_checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
                "action_prior": "standard_normal",
            },
            "teachers": contract["teachers"],
            "teacher_checkpoint_sha256": {
                "effect": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
                "action": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            },
            "action_teacher": {
                "schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
                "prior": "standard_normal",
                "variant": "a1",
                "source_horizon_steps": 24,
                "source_duration_seconds": 1.2,
            },
            "normalization": {
                "schema_version": NORMALIZATION_SCHEMA_VERSION,
                "fit_split": "train",
                "fit_membership": fit_membership,
                "fit_row_count": row_count,
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
                "hdf5_write_policy": "one exclusive shard per rank; rank0 read-only merge",
            },
            "issue_count": 0,
            "ready": True,
        }
        provenance_path = publishing / "provenance.json"
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
                f"{sha256_file(publishing / name)}  {name}\n" for name in artifact_names
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
                "effect": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
                "action": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            },
            "all_gates_passed": True,
        }
        _json_write_atomic(publishing / "_SUCCESS", success)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(publishing, output_root)
        shutil.rmtree(work_root, ignore_errors=True)
        return {"output_root": str(output_root), **success}
    except BaseException:
        # Keep rank shards for --resume, but never expose a partial final cache.
        shutil.rmtree(publishing, ignore_errors=True)
        raise


def _root_contract(
    *,
    source: MergedLibero04sSource,
    source_fingerprint: Mapping[str, Any],
    teachers: Mapping[str, Any],
    world_size: int,
    precision: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage2_temporal_contract": TEMPORAL_CONTRACT,
        "dataset_root": str(source.root),
        "source_row_count": SOURCE_ROW_COUNT,
        "selected_row_count": source.selected_count,
        "release_level": source.release_level,
        "source_fingerprint": dict(source_fingerprint),
        "teachers": dict(teachers),
        "temporal_contract": _temporal_metadata(),
        "world_size": world_size,
        "precision": precision,
    }


def build_baseline_04s_cache(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 64,
    precision: str = "bf16",
    max_samples: int | None = None,
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

    source = MergedLibero04sSource(dataset_root, max_samples=max_samples)
    contract_path = work_root / "contract.json"
    if context.is_main and not contract_path.exists():
        fingerprint = source_release_fingerprint(source.root)
        teachers = _teacher_paths_and_hashes()
        contract = _root_contract(
            source=source,
            source_fingerprint=fingerprint,
            teachers=teachers,
            world_size=context.world_size,
            precision=precision,
        )
        _json_write_atomic(contract_path, contract)
    context.barrier()
    contract = _json_read(contract_path)
    expected_local = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage2_temporal_contract": TEMPORAL_CONTRACT,
        "dataset_root": str(source.root),
        "source_row_count": SOURCE_ROW_COUNT,
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
        source=source,
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


class Baseline04sTeacherCache:
    """Process-local, pickle-safe read-only teacher lookup by source identity."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        require_full: bool = True,
        expected_schema_version: str = CACHE_SCHEMA_VERSION,
        expected_temporal_contract: str = TEMPORAL_CONTRACT,
        expected_normalization_schema: str = NORMALIZATION_SCHEMA_VERSION,
        expected_full_rows: int = SOURCE_ROW_COUNT,
        expected_validation_rows: int | None = VALIDATION_ROW_COUNT,
        action_teacher_transfer: str | None = None,
    ) -> None:
        from .robotwin_action_transfer import CHECKPOINT_SHA256, CONDITION_SHA256, checked_transfer, check_marker

        action_sha = CHECKPOINT_SHA256 if checked_transfer(action_teacher_transfer) else LIBERO_12S_ACTION_CHECKPOINT_SHA256
        condition_sha = CONDITION_SHA256 if checked_transfer(action_teacher_transfer) else LIBERO_12S_EFFECT_CHECKPOINT_SHA256
        self.cache_root = Path(cache_root).expanduser().resolve()
        success = _json_read(self.cache_root / "_SUCCESS")
        check_marker(success, action_teacher_transfer)
        if success.get("schema_version") != expected_schema_version:
            raise ValueError("baseline 0.4s teacher cache schema mismatch")
        if success.get("stage2_temporal_contract") != expected_temporal_contract:
            raise ValueError("baseline 0.4s temporal contract mismatch")
        if success.get("all_gates_passed") is not True:
            raise ValueError("baseline 0.4s cache did not pass publication gates")
        if require_full and (
            success.get("release_level") != "full"
            or int(success.get("row_count", -1)) != expected_full_rows
            or int(success.get("train_membership_count", -1)) != expected_full_rows
            or (
                expected_validation_rows is not None
                and int(success.get("validation_membership_count", -1))
                != expected_validation_rows
            )
        ):
            raise ValueError(
                f"training requires the full {expected_full_rows}-row 0.4s cache"
            )
        checksums_path = self.cache_root / "checksums.sha256"
        if sha256_file(checksums_path) != success.get("checksums_sha256"):
            raise ValueError("baseline 0.4s checksum manifest mismatch")
        expected_checksums: dict[str, str] = {}
        for line_number, raw in enumerate(
            checksums_path.read_text(encoding="ascii").splitlines(), 1
        ):
            digest, separator, name = raw.partition("  ")
            if not separator or len(digest) != 64 or name in expected_checksums:
                raise ValueError(f"invalid checksum row {line_number}")
            expected_checksums[name] = digest
        required_artifacts = {
            "latents.h5",
            "manifest.parquet",
            "normalization.json",
            "provenance.json",
        }
        if set(expected_checksums) != required_artifacts:
            raise ValueError("baseline 0.4s checksum artifact set mismatch")
        for name, digest in expected_checksums.items():
            if sha256_file(self.cache_root / name) != digest:
                raise ValueError(f"baseline 0.4s artifact checksum mismatch: {name}")
        provenance = _json_read(self.cache_root / "provenance.json")
        check_marker(provenance, action_teacher_transfer)
        if (
            provenance.get("schema_version") != expected_schema_version
            or provenance.get("stage2_temporal_contract") != expected_temporal_contract
            or provenance.get("release_level") != success.get("release_level")
            or int(provenance.get("row_count", -1)) != int(success["row_count"])
            or provenance.get("teacher_checkpoint_sha256")
            != {
                "effect": condition_sha,
                "action": action_sha,
            }
            or provenance.get("ready") is not True
            or int(provenance.get("issue_count", -1)) != 0
        ):
            raise ValueError("baseline 0.4s provenance contract mismatch")
        temporal = provenance.get("source", {}).get("temporal_contract", {})
        for name, expected in _temporal_metadata().items():
            if name in {
                "trajectory_representation",
                "tail_policy",
                "stage2_temporal_contract",
            }:
                continue
            if temporal.get(name) != expected:
                raise ValueError(f"baseline 0.4s provenance temporal mismatch: {name}")
        normalization = _json_read(self.cache_root / "normalization.json")
        expected_membership = (
            "all_rows" if success.get("release_level") == "full" else "selected_smoke_rows"
        )
        if (
            normalization.get("schema_version") != expected_normalization_schema
            or normalization.get("stage2_temporal_contract")
            != expected_temporal_contract
            or normalization.get("fit_split") != "train"
            or normalization.get("fit_membership") != expected_membership
            or int(normalization.get("fit_row_count", -1)) != int(success["row_count"])
        ):
            raise ValueError("baseline 0.4s normalization contract mismatch")
        manifest = pd.read_parquet(self.cache_root / "manifest.parquet")
        missing = sorted(_MANIFEST_REQUIRED.difference(manifest.columns))
        if missing:
            raise ValueError(f"baseline 0.4s manifest is missing columns: {missing}")
        if len(manifest) != int(success["row_count"]):
            raise ValueError("baseline 0.4s manifest row count mismatch")
        if not bool(manifest["train_membership"].all()):
            raise ValueError("baseline 0.4s cache must mark every row as train membership")
        if (
            require_full
            and expected_validation_rows is not None
            and int(manifest["validation_membership"].sum())
            != expected_validation_rows
        ):
            raise ValueError("baseline 0.4s validation membership count mismatch")
        self._identity_columns = (
            ("dataset_name", "episode_id", "start_step")
            if "dataset_name" in manifest.columns
            else ("episode_id", "start_step")
        )
        if manifest[list(self._identity_columns)].duplicated().any():
            raise ValueError("baseline 0.4s manifest identity is not unique")
        if not np.array_equal(
            manifest["latent_row"].to_numpy(dtype=np.int64),
            np.arange(len(manifest), dtype=np.int64),
        ):
            raise ValueError("baseline 0.4s latent_row is not contiguous")
        if self._identity_columns[0] == "dataset_name":
            self._rows = {
                (str(row.dataset_name), int(row.episode_id), int(row.start_step)): int(
                    row.latent_row
                )
                for row in manifest.itertuples(index=False)
            }
            self._metadata = {
                (str(row.dataset_name), int(row.episode_id), int(row.start_step)): row._asdict()
                for row in manifest.itertuples(index=False)
            }
        else:
            self._rows = {
                (int(row.episode_id), int(row.start_step)): int(row.latent_row)
                for row in manifest.itertuples(index=False)
            }
            self._metadata = {
                (int(row.episode_id), int(row.start_step)): row._asdict()
                for row in manifest.itertuples(index=False)
            }
        self._h5_path = self.cache_root / "latents.h5"
        with h5py.File(self._h5_path, "r") as handle:
            if handle.attrs.get("schema_version") != expected_schema_version:
                raise ValueError("baseline 0.4s HDF5 schema mismatch")
            if (
                handle.attrs.get("stage2_temporal_contract")
                != expected_temporal_contract
            ):
                raise ValueError("baseline 0.4s HDF5 temporal contract mismatch")
            if handle.attrs.get("release_level") != success.get("release_level"):
                raise ValueError("baseline 0.4s HDF5 release-level mismatch")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError("baseline 0.4s HDF5 is incomplete")
            for name in _ATTACH_FIELDS:
                if name not in handle or int(handle[name].shape[0]) != len(manifest):
                    raise ValueError(f"baseline 0.4s HDF5 attach field mismatch: {name}")
            if "physical_valid_count" not in handle or tuple(
                handle["physical_valid_count"].shape
            ) != (len(manifest),):
                raise ValueError("baseline 0.4s HDF5 physical_valid_count mismatch")
            if not np.all(handle["physical_valid_count"][:] == LIBERO_04S_CONTROL_STEPS):
                raise ValueError("baseline 0.4s physical_valid_count must be exactly eight")
        self._handle: h5py.File | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._owner_pid = None

    def _local_handle(self) -> h5py.File:
        pid = os.getpid()
        if self._owner_pid != pid:
            self.close()
            self._handle = h5py.File(self._h5_path, "r")
            self._owner_pid = pid
        assert self._handle is not None
        return self._handle

    def get(
        self,
        episode_id: int,
        start_step: int,
        *,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        if self._identity_columns[0] == "dataset_name":
            if not str(dataset_name or "").strip():
                raise ValueError("dataset_name is required by this teacher cache")
            identity = (str(dataset_name), int(episode_id), int(start_step))
        else:
            identity = (int(episode_id), int(start_step))
        try:
            row = self._rows[identity]
        except KeyError as exc:
            raise KeyError(f"teacher cache has no source identity {identity}") from exc
        handle = self._local_handle()
        result: dict[str, Any] = {}
        bool_fields = {
            "teacher_time_mask",
            "teacher_pose_mask",
            "teacher_gripper_mask",
            "teacher_current_state_mask",
        }
        for name in _ATTACH_FIELDS:
            array = np.asarray(handle[name][row]).copy()
            result[name] = torch.from_numpy(array).to(
                dtype=torch.bool if name in bool_fields else torch.float32
            )
        action_predictor_state = torch.zeros(32, dtype=torch.float32)
        action_predictor_state[:14] = result["teacher_current_state"]
        result["action_predictor_state"] = action_predictor_state
        physical_valid_count = int(handle["physical_valid_count"][row])
        if physical_valid_count != LIBERO_04S_CONTROL_STEPS:
            raise ValueError("teacher cache physical_valid_count changed after validation")
        result["physical_valid_count"] = torch.tensor(
            physical_valid_count, dtype=torch.long
        )
        result["teacher_cache_metadata"] = dict(self._metadata[identity])
        return result

    def attach(
        self,
        sample: Mapping[str, Any],
        *,
        episode_id: int,
        start_step: int,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        attached = dict(sample)
        overlap = sorted(
            set(attached).intersection(
                {*_ATTACH_FIELDS, "action_predictor_state", "physical_valid_count"}
            )
        )
        if overlap:
            raise ValueError(f"sample already contains teacher fields: {overlap}")
        attached.update(
            self.get(
                episode_id,
                start_step,
                dataset_name=dataset_name,
            )
        )
        return attached

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is nondeterministic
        try:
            self.close()
        except Exception:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_baseline_04s_cache(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        precision=args.precision,
        max_samples=args.max_samples,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
