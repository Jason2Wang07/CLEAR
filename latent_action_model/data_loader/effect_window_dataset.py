"""Dataset and I/O-aware batch sampling for Stage-1 effect windows.

The release stores one manifest row per HDF5 row.  The manifest is the source
of truth for split membership; tensors are read from the referenced shard.
HDF5 files are opened lazily and handles are strictly process-local so a
``DataLoader`` worker never reuses a handle inherited from its parent.
"""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

try:
    import h5py as _h5py
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch in tests
    _h5py = None
    _H5PY_IMPORT_ERROR: Exception | None = exc
else:
    _H5PY_IMPORT_ERROR = None

try:
    import pyarrow.parquet as _parquet
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    _parquet = None
    _PYARROW_IMPORT_ERROR: Exception | None = exc
else:
    _PYARROW_IMPORT_ERROR = None


_SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
_MANIFEST_FILENAMES = {"train": "train.parquet", "validation": "validation.parquet", "test": "test.parquet"}
_DEFAULT_EMBODIMENT_IDS = {"panda": 25, "agilex": 1, "widowx": 18}
_REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "group_id",
    "task_id",
    "effect_family",
    "source_id",
    "controller_id",
    "episode_id",
    "start_step",
    "end_step",
    "split",
    "shard_path",
    "shard_row",
}
_REQUIRED_H5_FIELDS = (
    "sample_id",
    "split",
    "embodiment_id",
    "rgb_start",
    "rgb_end",
    "state14_start",
    "state14_end",
    "delta_state14",
    "state_mask",
    "effect_raw",
    "effect_mask",
)


def _require_h5py() -> None:
    if _h5py is None:
        raise ImportError(
            "EffectWindowDataset requires h5py to read Stage-1 HDF5 shards. "
            "Install it in the active training environment (for example: `pip install h5py`)."
        ) from _H5PY_IMPORT_ERROR


def _require_pyarrow() -> None:
    if _parquet is None:
        raise ImportError(
            "EffectWindowDataset requires pyarrow to read the official Parquet manifests. "
            "Install it in the active training environment (for example: `pip install pyarrow`)."
        ) from _PYARROW_IMPORT_ERROR


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    try:
        return _SPLIT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}."
        ) from exc


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_string(value.item())
    return str(value)


def _as_contiguous_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    # Copying detaches the tensor from any h5py-owned temporary buffer.
    return torch.from_numpy(array.copy()).to(dtype=dtype)


class EffectWindowDataset(Dataset[Dict[str, Any]]):
    """Read a single official Stage-1 split from its Parquet manifest.

    Args:
        data_root: Root containing ``manifests/``, ``shards/`` and ``stats/``.
        split: Official split. ``val`` is accepted as an alias for
            ``validation`` but still reads ``validation.parquet``.
        max_open_shards: Maximum number of process-local HDF5 handles retained
            in the least-recently-used cache.
        verify_row_identity: Check HDF5 ``sample_id`` and ``split`` against the
            manifest on every read.
        embodiment_ids: Explicit string-to-integer mapping.  The default maps
            the release's ``panda`` label to LaWAM's Franka ID 25.
    """

    frame_shape = (2, 3, 256, 256)
    state_shape = (2, 14)
    effect_width = 12

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        max_open_shards: int = 8,
        verify_row_identity: bool = True,
        embodiment_ids: Mapping[str, int] | None = None,
        state_normalization_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        _require_h5py()
        _require_pyarrow()
        if max_open_shards < 1:
            raise ValueError(f"max_open_shards must be >= 1, got {max_open_shards}.")

        self.data_root = Path(data_root).expanduser().resolve()
        self.split = _normalize_split(split)
        self.max_open_shards = int(max_open_shards)
        self.verify_row_identity = bool(verify_row_identity)
        if state_normalization_path is None:
            self.state_normalization_path = self.data_root / "stats" / "state_normalization.json"
        else:
            normalization_path = Path(state_normalization_path).expanduser()
            if not normalization_path.is_absolute():
                normalization_path = self.data_root / normalization_path
            self.state_normalization_path = normalization_path.resolve()
        mapping = _DEFAULT_EMBODIMENT_IDS if embodiment_ids is None else embodiment_ids
        self.embodiment_ids = {str(key).strip().lower(): int(value) for key, value in mapping.items()}

        manifest_path = self.data_root / "manifests" / _MANIFEST_FILENAMES[self.split]
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Official {self.split!r} manifest does not exist: {manifest_path}"
            )
        self.manifest_path = manifest_path
        table = _parquet.read_table(manifest_path)
        missing_columns = sorted(_REQUIRED_MANIFEST_COLUMNS.difference(table.column_names))
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {missing_columns}."
            )
        columns = table.to_pydict()
        self._manifest_columns: Dict[str, list[Any]] = {
            name: list(values) for name, values in columns.items()
        }
        self._length = int(table.num_rows)

        manifest_splits = {str(value) for value in self._manifest_columns["split"]}
        if manifest_splits != {self.split} and self._length > 0:
            raise ValueError(
                f"Manifest {manifest_path} must contain only split={self.split!r}; "
                f"found {sorted(manifest_splits)}."
            )

        sample_ids = [str(value) for value in self._manifest_columns["sample_id"]]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Manifest {manifest_path} contains duplicate sample_id values.")
        self._sample_ids = sample_ids

        self._shard_rows = [int(value) for value in self._manifest_columns["shard_row"]]
        if any(row < 0 for row in self._shard_rows):
            raise ValueError(f"Manifest {manifest_path} contains a negative shard_row.")

        resolved_by_manifest_path: Dict[str, str] = {}
        self._shard_paths: list[str] = []
        for raw_path in self._manifest_columns["shard_path"]:
            raw_key = str(raw_path)
            resolved = resolved_by_manifest_path.get(raw_key)
            if resolved is None:
                resolved = str(self._resolve_shard_path(raw_key))
                resolved_by_manifest_path[raw_key] = resolved
            self._shard_paths.append(resolved)

        locations = list(zip(self._shard_paths, self._shard_rows))
        if len(set(locations)) != len(locations):
            raise ValueError(f"Manifest {manifest_path} references a shard row more than once.")

        self._state_median, self._state_scale = self._load_state_normalization()
        self._h5_handles: OrderedDict[str, Any] = OrderedDict()
        self._handle_owner_pid: int | None = None

    def _resolve_shard_path(self, manifest_value: str) -> Path:
        raw = Path(manifest_value).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(self.data_root / raw)

        # Release manifests may contain absolute paths from the machine that
        # built the package.  Preserve the suffix beginning at `shards/` so a
        # relocated package remains readable.
        parts = raw.parts
        shard_positions = [index for index, part in enumerate(parts) if part == "shards"]
        if shard_positions:
            candidates.append(self.data_root.joinpath(*parts[shard_positions[-1] :]))
        candidates.append(self.data_root / "shards" / self.split / raw.name)

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
        formatted = ", ".join(str(path) for path in unique_candidates)
        raise FileNotFoundError(
            f"Could not resolve shard path {manifest_value!r} for split={self.split!r}. "
            f"Tried: {formatted}"
        )

    def _load_state_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        stats_path = self.state_normalization_path
        if not stats_path.is_file():
            raise FileNotFoundError(f"State normalization file does not exist: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if stats.get("field") != "delta_state14":
            raise ValueError(
                f"Expected state normalization field='delta_state14' in {stats_path}, "
                f"got {stats.get('field')!r}."
            )
        if stats.get("fit_split") != "train":
            raise ValueError(
                f"State normalization in {stats_path} must be fit on train, "
                f"got {stats.get('fit_split')!r}."
            )
        median = np.asarray(stats.get("median"), dtype=np.float32)
        scale = np.asarray(stats.get("scale_mad"), dtype=np.float32)
        if median.shape != (14,) or scale.shape != (14,):
            raise ValueError(
                f"Expected 14-D median and scale_mad in {stats_path}; "
                f"got median={median.shape}, scale={scale.shape}."
            )
        if not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"State normalization values in {stats_path} must be finite with scale_mad > 0.")
        return median, scale

    def __len__(self) -> int:
        return self._length

    @property
    def sample_locations(self) -> tuple[tuple[str, int], ...]:
        """Resolved ``(shard_path, shard_row)`` pairs in manifest order."""

        return tuple(zip(self._shard_paths, self._shard_rows))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Stable sample IDs in official manifest order."""

        return tuple(self._sample_ids)

    def manifest_metadata(self, index: int) -> Dict[str, Any]:
        """Return manifest metadata without opening the referenced HDF5 shard."""

        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"EffectWindowDataset index {index} out of range for length {self._length}.")
        return self._metadata_for_index(index, self._shard_paths[index], self._shard_rows[index])

    @property
    def handle_owner_pid(self) -> int | None:
        """PID owning the current handle cache, exposed for diagnostics/tests."""

        return self._handle_owner_pid

    @property
    def open_shard_count(self) -> int:
        return len(self._h5_handles)

    def _close_cached_handles(self) -> None:
        handles = getattr(self, "_h5_handles", None)
        if handles is not None:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            handles.clear()

    def close(self) -> None:
        self._close_cached_handles()
        self._handle_owner_pid = None

    def reset_worker_state(self) -> None:
        """Discard inherited handles and initialize an empty cache for this PID."""

        self._close_cached_handles()
        self._h5_handles = OrderedDict()
        self._handle_owner_pid = os.getpid()

    def _ensure_process_local_cache(self) -> None:
        pid = os.getpid()
        if self._handle_owner_pid != pid:
            self.reset_worker_state()

    def _get_h5_handle(self, shard_path: str) -> Any:
        self._ensure_process_local_cache()
        cached = self._h5_handles.pop(shard_path, None)
        if cached is not None:
            self._h5_handles[shard_path] = cached
            return cached

        handle = _h5py.File(shard_path, "r")
        try:
            missing = [name for name in _REQUIRED_H5_FIELDS if name not in handle]
            if missing:
                raise ValueError(f"Shard {shard_path} is missing required fields: {missing}.")
            self._h5_handles[shard_path] = handle
            while len(self._h5_handles) > self.max_open_shards:
                _, evicted = self._h5_handles.popitem(last=False)
                evicted.close()
        except Exception:
            handle.close()
            raise
        return handle

    def _metadata_for_index(self, index: int, shard_path: str, shard_row: int) -> Dict[str, Any]:
        excluded = {"effect_raw", "effect_mask", "shard_path", "shard_row"}
        metadata = {
            name: values[index]
            for name, values in self._manifest_columns.items()
            if name not in excluded
        }
        metadata.update(
            {
                "dataset_index": index,
                "manifest_shard_path": str(self._manifest_columns["shard_path"][index]),
                "shard_path": shard_path,
                "shard_row": shard_row,
            }
        )
        return metadata

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"EffectWindowDataset index {index} out of range for length {self._length}.")

        shard_path = self._shard_paths[index]
        shard_row = self._shard_rows[index]
        handle = self._get_h5_handle(shard_path)
        shard_length = int(handle["sample_id"].shape[0])
        if shard_row >= shard_length:
            raise IndexError(
                f"Manifest row {index} references shard_row={shard_row}, but {shard_path} has {shard_length} rows."
            )

        sample_id = _decode_h5_string(handle["sample_id"][shard_row])
        h5_split = _decode_h5_string(handle["split"][shard_row])
        if self.verify_row_identity:
            if sample_id != self._sample_ids[index]:
                raise ValueError(
                    f"Manifest/HDF5 sample mismatch at dataset index {index}: "
                    f"manifest={self._sample_ids[index]!r}, h5={sample_id!r}."
                )
            if h5_split != self.split:
                raise ValueError(
                    f"Manifest/HDF5 split mismatch at dataset index {index}: "
                    f"expected={self.split!r}, h5={h5_split!r}."
                )

        frames = torch.stack(
            (
                _as_contiguous_tensor(handle["rgb_start"][shard_row], torch.uint8),
                _as_contiguous_tensor(handle["rgb_end"][shard_row], torch.uint8),
            ),
            dim=0,
        )
        states = torch.stack(
            (
                _as_contiguous_tensor(handle["state14_start"][shard_row], torch.float32),
                _as_contiguous_tensor(handle["state14_end"][shard_row], torch.float32),
            ),
            dim=0,
        )
        delta_raw = _as_contiguous_tensor(handle["delta_state14"][shard_row], torch.float32)
        state_dim_mask = _as_contiguous_tensor(handle["state_mask"][shard_row], torch.bool)
        effect_raw = _as_contiguous_tensor(handle["effect_raw"][shard_row], torch.float32)
        effect_mask = _as_contiguous_tensor(handle["effect_mask"][shard_row], torch.bool)

        if tuple(frames.shape) != self.frame_shape:
            raise ValueError(
                f"Expected frames {self.frame_shape}, got {tuple(frames.shape)} "
                f"in {shard_path}:{shard_row}."
            )
        if tuple(states.shape) != self.state_shape:
            raise ValueError(
                f"Expected states {self.state_shape}, got {tuple(states.shape)} "
                f"in {shard_path}:{shard_row}."
            )
        if tuple(delta_raw.shape) != (14,) or tuple(state_dim_mask.shape) != (14,):
            raise ValueError(
                f"Expected delta_state14/state_mask shape (14,), got "
                f"{tuple(delta_raw.shape)}/{tuple(state_dim_mask.shape)} in {shard_path}:{shard_row}."
            )
        if tuple(effect_raw.shape) != (self.effect_width,) or tuple(effect_mask.shape) != (self.effect_width,):
            raise ValueError(
                f"Expected effect_raw/effect_mask shape ({self.effect_width},), got "
                f"{tuple(effect_raw.shape)}/{tuple(effect_mask.shape)} in {shard_path}:{shard_row}."
            )

        median = torch.from_numpy(self._state_median)
        scale = torch.from_numpy(self._state_scale)
        delta_normalized = ((delta_raw - median) / scale) * state_dim_mask.to(torch.float32)
        state_mask = state_dim_mask.unsqueeze(0).expand(2, -1).clone()

        embodiment_label = _decode_h5_string(handle["embodiment_id"][shard_row]).strip().lower()
        if embodiment_label not in self.embodiment_ids:
            raise ValueError(
                f"Unknown embodiment_id {embodiment_label!r} in {shard_path}:{shard_row}; "
                "provide an explicit embodiment_ids mapping instead of silently assigning an ID."
            )
        embodiment_id = self.embodiment_ids[embodiment_label]
        metadata = self.manifest_metadata(index)
        metadata["embodiment_label"] = embodiment_label

        return {
            "frames": frames,
            "states": states,
            "delta_state14_raw": delta_raw,
            "delta_state14": delta_normalized,
            "delta_proprio": delta_normalized,
            "state_mask": state_mask,
            "state_dim_mask": state_dim_mask,
            "effect_raw": effect_raw,
            "effect_mask": effect_mask,
            "embodiment_id": embodiment_id,
            "metadata": metadata,
        }

    def __getstate__(self) -> Dict[str, Any]:
        # Spawn workers pickle datasets.  Never serialize a live h5py handle.
        state = self.__dict__.copy()
        state["_h5_handles"] = OrderedDict()
        state["_handle_owner_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._h5_handles = OrderedDict()
        self._handle_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def effect_window_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Stack effect-window samples and provide LaWAM compatibility aliases."""

    if not batch:
        raise ValueError("effect_window_collate received an empty batch.")
    frames = torch.stack([sample["frames"] for sample in batch], dim=0)
    states = torch.stack([sample["states"] for sample in batch], dim=0)
    delta_raw = torch.stack([sample["delta_state14_raw"] for sample in batch], dim=0)
    delta_normalized = torch.stack([sample["delta_state14"] for sample in batch], dim=0)
    state_mask = torch.stack([sample["state_mask"] for sample in batch], dim=0)
    state_dim_mask = torch.stack([sample["state_dim_mask"] for sample in batch], dim=0)
    effect_raw = torch.stack([sample["effect_raw"] for sample in batch], dim=0)
    effect_mask = torch.stack([sample["effect_mask"] for sample in batch], dim=0)
    embodiment_ids = torch.tensor([int(sample["embodiment_id"]) for sample in batch], dtype=torch.long)
    metadata = [dict(sample["metadata"]) for sample in batch]

    return {
        "videos": frames,
        "frames": frames,
        "states": states,
        "proprio": states,
        "delta_state14_raw": delta_raw,
        "delta_state14": delta_normalized,
        "delta_proprio": delta_normalized,
        "state_mask": state_mask,
        "state_dim_mask": state_dim_mask,
        "proprio_mask": state_dim_mask.to(torch.float32),
        "effect_raw": effect_raw,
        "effect_mask": effect_mask,
        "embodiment_ids": embodiment_ids,
        "metadata": metadata,
        "sample_ids": [str(item["sample_id"]) for item in metadata],
        "group_ids": [str(item["group_id"]) for item in metadata],
        "task_ids": [str(item["task_id"]) for item in metadata],
        "effect_families": [str(item["effect_family"]) for item in metadata],
        "dataset_names": [str(item["source_id"]) for item in metadata],
        "trajectory_ids": [int(item["episode_id"]) for item in metadata],
        "base_indices": torch.tensor([int(item["start_step"]) for item in metadata], dtype=torch.long),
        "dataset_indices": torch.tensor([int(item["dataset_index"]) for item in metadata], dtype=torch.long),
        "shard_rows": torch.tensor([int(item["shard_row"]) for item in metadata], dtype=torch.long),
        "shard_paths": [str(item["shard_path"]) for item in metadata],
    }


class ShardAwareBatchSampler(Sampler[list[int]]):
    """Build contiguous shard-row blocks, then partition global batches by rank.

    Every dataset index appears at most once globally in an epoch.  With
    ``drop_last=True``, ``batch_size`` is the global batch size and each rank
    receives ``batch_size / world_size`` samples per step.  Every rank therefore
    performs the same number of steps.  With
    ``drop_last=False``, no padding or duplication is introduced; ranks that do
    not own a slice of the final incomplete global batch simply have one fewer
    batch.
    """

    def __init__(
        self,
        dataset: EffectWindowDataset,
        batch_size: int,
        *,
        block_size: int | None = None,
        shuffle: bool = True,
        seed: int = 20260803,
        drop_last: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        order: str = "shard",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if block_size is None:
            block_size = 4 * int(batch_size)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}.")
        if order not in {"shard", "manifest"}:
            raise ValueError(f"order must be 'shard' or 'manifest', got {order!r}.")
        if order == "manifest" and shuffle:
            raise ValueError("order='manifest' requires shuffle=False.")

        distributed_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        resolved_world_size = (
            int(world_size)
            if world_size is not None
            else (torch.distributed.get_world_size() if distributed_ready else 1)
        )
        resolved_rank = (
            int(rank) if rank is not None else (torch.distributed.get_rank() if distributed_ready else 0)
        )
        if resolved_world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {resolved_world_size}.")
        if not 0 <= resolved_rank < resolved_world_size:
            raise ValueError(f"rank must be in [0, {resolved_world_size}), got {resolved_rank}.")
        if batch_size % resolved_world_size != 0:
            raise ValueError(
                f"Global batch_size={batch_size} must be divisible by world_size={resolved_world_size}."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.local_batch_size = self.batch_size // resolved_world_size
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.rank = resolved_rank
        self.world_size = resolved_world_size
        self.order = order
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        if self.order == "manifest":
            return list(range(len(self.dataset)))

        by_shard: Dict[str, list[tuple[int, int]]] = defaultdict(list)
        for dataset_index, (shard_path, shard_row) in enumerate(self.dataset.sample_locations):
            by_shard[shard_path].append((int(shard_row), dataset_index))

        blocks: list[list[int]] = []
        for shard_path in sorted(by_shard):
            ordered = [index for _, index in sorted(by_shard[shard_path])]
            blocks.extend(
                ordered[start : start + self.block_size]
                for start in range(0, len(ordered), self.block_size)
            )
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(blocks)
        return [index for block in blocks for index in block]

    def __iter__(self) -> Iterator[list[int]]:
        ordered = self._ordered_indices()
        full_global_batches = len(ordered) // self.batch_size
        for global_batch_index in range(full_global_batches):
            global_start = global_batch_index * self.batch_size
            local_start = global_start + self.rank * self.local_batch_size
            yield ordered[local_start : local_start + self.local_batch_size]

        if self.drop_last:
            return
        tail_start = full_global_batches * self.batch_size
        remainder = len(ordered) - tail_start
        tail_per_rank, extra = divmod(remainder, self.world_size)
        local_count = tail_per_rank + int(self.rank < extra)
        local_offset = self.rank * tail_per_rank + min(self.rank, extra)
        if local_count > 0:
            local_start = tail_start + local_offset
            yield ordered[local_start : local_start + local_count]

    def __len__(self) -> int:
        full_global_batches, remainder = divmod(len(self.dataset), self.batch_size)
        if self.drop_last:
            return full_global_batches
        tail_per_rank, extra = divmod(remainder, self.world_size)
        owns_tail = tail_per_rank > 0 or self.rank < extra
        return full_global_batches + int(owns_tail)


class MixedBalancedBatchSampler(Sampler[list[int]]):
    """Build global batches with exact natural/balanced halves.

    Natural rows are sampled from a per-epoch uniform permutation. Balanced
    rows are drawn hierarchically: family -> task -> family-relative magnitude
    quartile -> sample. Sampling with replacement is intentional for the
    balanced half so small tasks are not silently dropped. Every rank builds
    the same deterministic global stream and returns its disjoint local slice.
    """

    def __init__(
        self,
        dataset: EffectWindowDataset,
        batch_size: int,
        *,
        natural_fraction: float = 0.5,
        seed: int = 20260803,
        drop_last: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        distributed_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.world_size = int(
            world_size if world_size is not None else (torch.distributed.get_world_size() if distributed_ready else 1)
        )
        self.rank = int(rank if rank is not None else (torch.distributed.get_rank() if distributed_ready else 0))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid rank/world_size")
        if batch_size < 1 or batch_size % self.world_size:
            raise ValueError("global batch_size must be positive and divisible by world_size")
        if not 0.0 < float(natural_fraction) < 1.0:
            raise ValueError("natural_fraction must be strictly between zero and one")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.local_batch_size = self.batch_size // self.world_size
        self.local_natural = int(round(self.local_batch_size * float(natural_fraction)))
        self.local_balanced = self.local_batch_size - self.local_natural
        if self.local_natural < 1 or self.local_balanced < 1:
            raise ValueError("each rank must receive at least one natural and one balanced sample")
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        columns = dataset._manifest_columns
        required = {"effect_family", "task_id", "effect_magnitude"}
        missing = sorted(required.difference(columns))
        if missing:
            raise ValueError(f"mixed-balanced sampling requires manifest columns: {missing}")
        self._families = np.asarray([str(value) for value in columns["effect_family"]], dtype=np.str_)
        self._tasks = np.asarray([str(value) for value in columns["task_id"]], dtype=np.str_)
        self._magnitudes = np.asarray(columns["effect_magnitude"], dtype=np.float64)
        if not np.all(np.isfinite(self._magnitudes)):
            raise ValueError("effect_magnitude contains NaN or infinity")

        self._groups: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for family in sorted(set(self._families.tolist())):
            family_indices = np.flatnonzero(self._families == family).astype(np.int64)
            quantiles = np.quantile(self._magnitudes[family_indices], [0.25, 0.50, 0.75])
            bins = np.searchsorted(quantiles, self._magnitudes[family_indices], side="left")
            task_groups: dict[str, dict[int, np.ndarray]] = {}
            for task in sorted(set(self._tasks[family_indices].tolist())):
                task_mask = self._tasks[family_indices] == task
                task_indices = family_indices[task_mask]
                task_bins = bins[task_mask]
                available = {
                    int(bin_id): task_indices[task_bins == bin_id]
                    for bin_id in sorted(set(task_bins.tolist()))
                    if np.any(task_bins == bin_id)
                }
                if available:
                    task_groups[task] = available
            if task_groups:
                self._groups[family] = task_groups
        if not self._groups:
            raise ValueError("mixed-balanced sampler found no non-empty hierarchy")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _draw_balanced(self, rng: np.random.Generator, count: int) -> list[int]:
        families = sorted(self._groups)
        result: list[int] = []
        for _ in range(count):
            family = families[int(rng.integers(len(families)))]
            tasks = sorted(self._groups[family])
            task = tasks[int(rng.integers(len(tasks)))]
            bins = sorted(self._groups[family][task])
            bin_id = bins[int(rng.integers(len(bins)))]
            members = self._groups[family][task][bin_id]
            result.append(int(members[int(rng.integers(len(members)))]))
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        natural = rng.permutation(len(self.dataset)).astype(np.int64)
        natural_cursor = 0
        for _ in range(len(self)):
            global_natural_count = self.local_natural * self.world_size
            if natural_cursor + global_natural_count > len(natural):
                natural = np.concatenate((natural[natural_cursor:], rng.permutation(len(self.dataset))))
                natural_cursor = 0
            natural_batch = natural[natural_cursor : natural_cursor + global_natural_count]
            natural_cursor += global_natural_count
            balanced_batch = self._draw_balanced(rng, self.local_balanced * self.world_size)

            n_start = self.rank * self.local_natural
            b_start = self.rank * self.local_balanced
            local = natural_batch[n_start : n_start + self.local_natural].tolist()
            local.extend(balanced_batch[b_start : b_start + self.local_balanced])
            # Grouping a rank's rows by shard improves HDF5 locality without
            # changing batch composition.
            local.sort(key=lambda index: self.dataset.sample_locations[index])
            yield local

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
