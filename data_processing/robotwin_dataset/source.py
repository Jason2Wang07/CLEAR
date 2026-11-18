"""Locked source and split contracts for the released RoboTwin LeRobot data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as parquet


CONTROL_HZ = 30.0
HORIZON = 48
WINDOW_STRIDE = 48
EPISODES_PER_BLOCK = 500
EXPECTED_EPISODES = 27_500
EXPECTED_FRAMES = 6_077_247

# The HF release is concatenated in these 500-episode blocks.  Blocks 50--54
# are additional variants of five benchmark tasks and intentionally share the
# same effect family as their original block.
BENCHMARK_FAMILIES: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

BLOCK_FAMILIES: tuple[str, ...] = BENCHMARK_FAMILIES + (
    "adjust_bottle",
    "hanging_mug",
    "place_a2b_left",
    "place_fan",
    "rotate_qrcode",
)


def family_for_episode(episode_id: int) -> str:
    episode = int(episode_id)
    if episode < 0:
        raise ValueError("episode_id must be non-negative")
    block = episode // EPISODES_PER_BLOCK
    if block >= len(BLOCK_FAMILIES):
        raise ValueError(f"episode_id {episode} is outside the locked 55-block RoboTwin release")
    return BLOCK_FAMILIES[block]


def window_starts(length: int, *, horizon: int = HORIZON, stride: int = WINDOW_STRIDE) -> np.ndarray:
    """Return starts whose endpoint observation at ``start+horizon`` exists."""

    length = int(length)
    horizon = int(horizon)
    stride = int(stride)
    if length < 0 or horizon < 1 or stride < 1:
        raise ValueError("length must be non-negative and horizon/stride must be positive")
    if length <= horizon:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, length - horizon, stride, dtype=np.int64)


def stratified_episode_splits(
    family: str,
    episode_ids: Sequence[int],
    seed: int,
) -> dict[int, str]:
    """Assign deterministic family-local, episode-disjoint 80/10/10 splits."""

    unique = [int(value) for value in episode_ids]
    if len(unique) != len(set(unique)):
        raise ValueError(f"family {family!r} contains duplicate episode IDs")
    if len(unique) < 3:
        raise ValueError(f"family {family!r} needs at least three episodes")

    def key(episode_id: int) -> tuple[bytes, int]:
        payload = f"{int(seed)}\0{family}\0{episode_id}".encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest(), episode_id

    ordered = sorted(unique, key=key)
    count = len(ordered)
    validation_count = max(1, round(0.10 * count))
    test_count = max(1, round(0.10 * count))
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError(f"family {family!r} has no train episode after splitting")
    result: dict[int, str] = {}
    for index, episode_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        result[episode_id] = split
    return result


def source_commit(root: Path) -> str:
    metadata = root / ".cache/huggingface/download/meta/info.json.metadata"
    if not metadata.is_file():
        return "unknown"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else "unknown"


def load_source_contract(root: str | Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    source = Path(root).expanduser().resolve()
    info_path = source / "meta/info.json"
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    modality_path = source / "meta/modality.json"
    if not info_path.is_file() or not episodes_path.is_file() or not modality_path.is_file():
        raise FileNotFoundError(f"incomplete RoboTwin LeRobot release: {source}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    required_info = {
        "codebase_version": "v3.0",
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "fps": int(CONTROL_HZ),
    }
    for name, expected in required_info.items():
        if info.get(name) != expected:
            raise ValueError(f"RoboTwin info.{name}={info.get(name)!r}, expected {expected!r}")
    state = info.get("features", {}).get("observation.state", {})
    if state.get("shape") != [16] or state.get("dtype") != "float32":
        raise ValueError("RoboTwin observation.state must be float32[16]")
    camera = info.get("features", {}).get("observation.images.cam_high", {})
    if camera.get("shape") != [3, 480, 640] or camera.get("dtype") != "video":
        raise ValueError("RoboTwin cam_high must be a 640x480 RGB video")

    columns = (
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.cam_high/chunk_index",
        "videos/observation.images.cam_high/file_index",
        "videos/observation.images.cam_high/from_timestamp",
        "videos/observation.images.cam_high/to_timestamp",
        "tasks",
        "length",
    )
    table = parquet.read_table(episodes_path, columns=list(columns))
    if table.num_rows != EXPECTED_EPISODES:
        raise ValueError(f"RoboTwin episode table has {table.num_rows} rows, expected {EXPECTED_EPISODES}")
    episodes = table.to_pydict()
    expected_ids = np.arange(EXPECTED_EPISODES, dtype=np.int64)
    actual_ids = np.asarray(episodes["episode_index"], dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("RoboTwin episode_index must be contiguous and ordered")
    lengths = np.asarray(episodes["length"], dtype=np.int64)
    starts = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
    stops = np.asarray(episodes["dataset_to_index"], dtype=np.int64)
    if np.any(lengths <= HORIZON) or not np.array_equal(stops - starts, lengths):
        raise ValueError("RoboTwin episode lengths or global data ranges are inconsistent")
    if int(lengths.sum()) != EXPECTED_FRAMES:
        raise ValueError("RoboTwin episode lengths do not sum to info.total_frames")
    if len(BLOCK_FAMILIES) * EPISODES_PER_BLOCK != EXPECTED_EPISODES:
        raise AssertionError("locked RoboTwin block mapping does not cover the release")
    return info, episodes
