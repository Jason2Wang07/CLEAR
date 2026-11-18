"""Read-only validation and indexing for BridgeData V2 LeRobot v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

from .schema import PHYSICAL_ACTION_STEPS, SOURCE_CONTROL_HZ


SOURCE_DATASET_ID = "BrunoM42/bridge_orig_lerobot"
SOURCE_REVISION = "d85bc6f78170873650d2c1456a2fa4cd4ae6c095"
PRIMARY_VIDEO_KEY = "observation.images.image_0"


@dataclass(frozen=True)
class BridgeEpisode:
    episode_id: int
    data_from: int
    data_to: int
    length: int
    data_file_index: int
    video_file_index: int
    video_from_frame: int
    video_to_frame: int
    task_index: int
    instruction: str
    split: str


@dataclass(frozen=True)
class BridgeSource:
    root: Path
    info: dict[str, Any]
    episodes: tuple[BridgeEpisode, ...]
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    episode_indices: np.ndarray
    global_indices: np.ndarray
    task_indices: np.ndarray


def _fixed_list_array(table: Any, name: str, width: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    offsets = column.offsets.to_numpy(zero_copy_only=False)
    if not np.all(np.diff(offsets) == width):
        raise ValueError(f"{name} rows must all have width {width}")
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, width)


def _episode_splits(episode_ids: list[int], seed: int) -> dict[int, str]:
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Bridge episode IDs are not unique")

    def key(episode_id: int) -> bytes:
        return hashlib.blake2b(f"{seed}\0bridge\0{episode_id}".encode(), digest_size=16).digest()

    ordered = sorted(episode_ids, key=lambda episode_id: (key(episode_id), episode_id))
    count = len(ordered)
    validation_count = round(0.10 * count)
    test_count = round(0.10 * count)
    train_count = count - validation_count - test_count
    result: dict[int, str] = {}
    for index, episode_id in enumerate(ordered):
        result[episode_id] = (
            "train"
            if index < train_count
            else ("validation" if index < train_count + validation_count else "test")
        )
    return result


def load_source(
    source_root: str | Path,
    *,
    split_seed: int = 20260803,
    max_episodes: int | None = None,
) -> BridgeSource:
    root = Path(source_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    if not info_path.is_file() or not episode_path.is_file():
        raise FileNotFoundError(f"incomplete Bridge LeRobot root: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if (
        int(info.get("fps", -1)) != int(SOURCE_CONTROL_HZ)
        or int(info.get("total_episodes", -1)) != 53_192
        or int(info.get("total_frames", -1)) != 1_893_026
        or str(info.get("robot_type", "")) != "widowx"
    ):
        raise ValueError("Bridge info.json does not match the audited source contract")

    episode_table = parquet.read_table(episode_path)
    episode_columns = episode_table.to_pydict()
    all_episode_ids = [int(value) for value in episode_columns["episode_index"]]
    split_by_episode = _episode_splits(all_episode_ids, int(split_seed))
    selected_count = len(all_episode_ids) if max_episodes is None else min(int(max_episodes), len(all_episode_ids))
    if selected_count < 1:
        raise ValueError("max_episodes selects no Bridge episodes")

    data_tables = [
        parquet.read_table(
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
        for path in sorted((root / "data").glob("*/*.parquet"))
    ]
    if not data_tables:
        raise FileNotFoundError(f"Bridge data parquet files are missing under {root / 'data'}")
    states = np.concatenate([_fixed_list_array(table, "observation.state", 8) for table in data_tables]).astype(
        np.float32, copy=False
    )
    actions = np.concatenate([_fixed_list_array(table, "action", 7) for table in data_tables]).astype(
        np.float32, copy=False
    )

    def concat(name: str, dtype: Any) -> np.ndarray:
        return np.concatenate(
            [table.column(name).combine_chunks().to_numpy(zero_copy_only=False) for table in data_tables]
        ).astype(dtype, copy=False)

    timestamps = concat("timestamp", np.float32)
    frame_indices = concat("frame_index", np.int64)
    episode_indices = concat("episode_index", np.int64)
    global_indices = concat("index", np.int64)
    task_indices = concat("task_index", np.int64)
    expected_rows = int(info["total_frames"])
    if any(len(array) != expected_rows for array in (states, actions, timestamps, frame_indices, episode_indices, global_indices, task_indices)):
        raise ValueError("Bridge parquet row counts do not match info.json")
    if not np.array_equal(global_indices, np.arange(expected_rows, dtype=np.int64)):
        raise ValueError("Bridge global index is not contiguous")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Bridge state/action contains NaN/Inf")
    if not np.all(states[:, 6] == 0.0):
        raise ValueError("Bridge state pad dimension is not exactly zero")

    episodes: list[BridgeEpisode] = []
    for row in range(selected_count):
        episode_id = int(episode_columns["episode_index"][row])
        data_from = int(episode_columns["dataset_from_index"][row])
        data_to = int(episode_columns["dataset_to_index"][row])
        length = int(episode_columns["length"][row])
        if data_to - data_from != length:
            raise ValueError(f"episode {episode_id} data range disagrees with length")
        slc = slice(data_from, data_to)
        if not np.all(episode_indices[slc] == episode_id):
            raise ValueError(f"episode {episode_id} parquet ownership mismatch")
        if not np.array_equal(frame_indices[slc], np.arange(length, dtype=np.int64)):
            raise ValueError(f"episode {episode_id} frame_index is not contiguous")
        expected_time = np.arange(length, dtype=np.float64) / SOURCE_CONTROL_HZ
        if not np.allclose(timestamps[slc], expected_time, atol=2.0e-6, rtol=0.0):
            raise ValueError(f"episode {episode_id} timestamp contract mismatch")
        if length > 1 and not np.allclose(
            actions[data_from : data_to - 1, :6],
            states[data_from + 1 : data_to, :6] - states[data_from : data_to - 1, :6],
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise ValueError(f"episode {episode_id} action/state transition mismatch")
        task_values = np.unique(task_indices[slc])
        if len(task_values) != 1:
            raise ValueError(f"episode {episode_id} contains multiple task indices")
        tasks = episode_columns["tasks"][row]
        instruction = "" if not tasks else str(tasks[0]).strip()
        video_from = float(episode_columns[f"videos/{PRIMARY_VIDEO_KEY}/from_timestamp"][row])
        video_to = float(episode_columns[f"videos/{PRIMARY_VIDEO_KEY}/to_timestamp"][row])
        video_from_frame = int(round(video_from * SOURCE_CONTROL_HZ))
        video_to_frame = int(round(video_to * SOURCE_CONTROL_HZ))
        if video_to_frame - video_from_frame != length:
            raise ValueError(f"episode {episode_id} primary video range disagrees with length")
        episodes.append(
            BridgeEpisode(
                episode_id=episode_id,
                data_from=data_from,
                data_to=data_to,
                length=length,
                data_file_index=int(episode_columns["data/file_index"][row]),
                video_file_index=int(episode_columns[f"videos/{PRIMARY_VIDEO_KEY}/file_index"][row]),
                video_from_frame=video_from_frame,
                video_to_frame=video_to_frame,
                task_index=int(task_values[0]),
                instruction=instruction,
                split=split_by_episode[episode_id],
            )
        )
    return BridgeSource(
        root=root,
        info=info,
        episodes=tuple(episodes),
        states=states,
        actions=actions,
        timestamps=timestamps,
        frame_indices=frame_indices,
        episode_indices=episode_indices,
        global_indices=global_indices,
        task_indices=task_indices,
    )


def audit_source(
    source_root: str | Path,
    *,
    split_seed: int = 20260803,
    max_episodes: int | None = None,
    window_stride: int = 2,
) -> dict[str, Any]:
    source = load_source(source_root, split_seed=split_seed, max_episodes=max_episodes)
    windows = sum(
        len(range(0, episode.length - PHYSICAL_ACTION_STEPS, int(window_stride)))
        for episode in source.episodes
    )
    language_windows = sum(
        len(range(0, episode.length - PHYSICAL_ACTION_STEPS, int(window_stride)))
        for episode in source.episodes
        if episode.instruction
    )
    split_episode_counts = {
        split: sum(episode.split == split for episode in source.episodes)
        for split in ("train", "validation", "test")
    }
    return {
        "schema": "bridge-source-audit-v1",
        "valid": True,
        "source_root": str(source.root),
        "episodes": len(source.episodes),
        "source_frames": len(source.states),
        "complete_windows": windows,
        "language_windows": language_windows,
        "empty_language_windows": windows - language_windows,
        "window_stride": int(window_stride),
        "physical_action_steps": PHYSICAL_ACTION_STEPS,
        "control_hz": SOURCE_CONTROL_HZ,
        "split_episode_counts": split_episode_counts,
    }
