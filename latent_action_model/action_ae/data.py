"""Lazy, split-preserving readers for versioned single-arm z-action sidecars."""

from __future__ import annotations

import copy
import inspect
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyarrow.parquet as parquet
import torch
try:
    from lightning import LightningDataModule
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise

    class LightningDataModule:  # type: ignore[no-redef]
        """Import shim for inference runtimes that do not install Lightning."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "ActionSidecarDataModule requires the optional 'lightning' package"
            )
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from latent_action_model.data_loader.effect_window_dataset import ShardAwareBatchSampler

from .context import ActionPriorContextCache, PRIOR_CONTEXT_KEYS


SIDECAR_SCHEMA_VERSION = "maniskill-z-action-sidecar-v1"
SUPPORTED_SIDECAR_SCHEMA_VERSIONS = {
    SIDECAR_SCHEMA_VERSION,
    "maniskill-z-action-sidecar-v2",
    "libero-z-action-sidecar-v1",
    "libero-z-action-sidecar-v1-12s",
    "bridge-z-action-sidecar-v1-12s-resampled24",
}
NORMALIZATION_SCHEMA_VERSION = "maniskill-z-action-normalization-v1"
SUPPORTED_NORMALIZATION_SCHEMA_VERSIONS = {
    NORMALIZATION_SCHEMA_VERSION,
    "libero-z-action-normalization-v1",
    "libero-z-action-normalization-v1-12s",
    "bridge-z-action-normalization-v1-12s-resampled24",
}
NORMALIZATION_SCHEMA_BY_SIDECAR = {
    "maniskill-z-action-sidecar-v1": NORMALIZATION_SCHEMA_VERSION,
    "maniskill-z-action-sidecar-v2": NORMALIZATION_SCHEMA_VERSION,
    "libero-z-action-sidecar-v1": "libero-z-action-normalization-v1",
    "libero-z-action-sidecar-v1-12s": "libero-z-action-normalization-v1-12s",
    "bridge-z-action-sidecar-v1-12s-resampled24": (
        "bridge-z-action-normalization-v1-12s-resampled24"
    ),
}
HORIZON = 32
TRAJECTORY_DIM = 7
STATE_DIM = 14
EFFECT_DIM = 32
NATIVE_ACTION_DIM = 8
ANCHOR_COUNT = 4

_SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
_MANIFEST_FILENAMES = {"train": "train.parquet", "validation": "validation.parquet", "test": "test.parquet"}
_REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "group_id",
    "split",
    "task_id",
    "source_id",
    "controller_id",
    "episode_id",
    "start_step",
    "end_step",
    "shard_path",
    "shard_row",
}
def _h5_field_shapes(horizon: int) -> dict[str, tuple[int, ...]]:
    return {
    "sample_id": (),
    "split": (),
    "embodiment_id": (),
    "waypoints": (horizon, TRAJECTORY_DIM),
    "time_mask": (horizon,),
    "pose_mask": (horizon,),
    "gripper_mask": (horizon,),
    "contact": (horizon,),
    "contact_mask": (horizon,),
    "grasp": (horizon,),
    "grasp_mask": (horizon,),
    "native_actions": (horizon, NATIVE_ACTION_DIM),
    "native_action_mask": (horizon, NATIVE_ACTION_DIM),
    "native_action_dim": (),
    "current_state": (STATE_DIM,),
    "current_state_mask": (STATE_DIM,),
    "e39_mean": (EFFECT_DIM,),
    "e39_valid": (),
    "physical_anchors": (ANCHOR_COUNT, 8),
    "physical_anchor_mask": (ANCHOR_COUNT, 8),
    "window_sha256": (),
    }


_H5_FIELD_SHAPES: dict[str, tuple[int, ...]] = _h5_field_shapes(HORIZON)
_MASK_FIELDS = (
    "time_mask",
    "pose_mask",
    "gripper_mask",
    "contact_mask",
    "grasp_mask",
    "native_action_mask",
    "current_state_mask",
    "physical_anchor_mask",
)


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    try:
        return _SPLIT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}") from exc


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_string(value.item())
    return str(value)


def _tensor(value: Any, dtype: torch.dtype) -> Tensor:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array.copy()).to(dtype=dtype)


def _binary_mask(value: Any, *, name: str) -> Tensor:
    array = np.asarray(value)
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return _tensor(array, torch.bool)


@dataclass(frozen=True)
class RobustGroupStats:
    median: Tensor
    scale: Tensor
    count: Tensor


class ActionNormalization:
    """Validated train-only median/MAD statistics used by ActionAE."""

    _WIDTHS = {"position": 3, "rotation_rotvec": 3, "gripper": 1, "current_state": STATE_DIM}

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        source: Path,
        expected_schema_version: str | None = None,
    ) -> None:
        self.source = Path(source)
        self._metadata = copy.deepcopy(dict(metadata))
        if metadata.get("schema_version") not in SUPPORTED_NORMALIZATION_SCHEMA_VERSIONS:
            raise ValueError(
                "normalization schema must be one of "
                f"{sorted(SUPPORTED_NORMALIZATION_SCHEMA_VERSIONS)!r}, "
                f"got {metadata.get('schema_version')!r}"
            )
        if expected_schema_version is not None and metadata.get("schema_version") != expected_schema_version:
            raise ValueError(
                f"normalization schema for this sidecar must be {expected_schema_version!r}, "
                f"got {metadata.get('schema_version')!r}"
            )
        if metadata.get("fit_split") != "train":
            raise ValueError(f"normalization must be fit on train, got {metadata.get('fit_split')!r}")
        if metadata.get("estimator") != "median_and_1.4826_mad":
            raise ValueError(f"unexpected normalization estimator {metadata.get('estimator')!r}")
        if int(metadata.get("sample_count", 0)) <= 0:
            raise ValueError("normalization sample_count must be positive")

        self.groups: dict[str, RobustGroupStats] = {}
        for name, width in self._WIDTHS.items():
            group = metadata.get(name)
            if not isinstance(group, Mapping):
                raise ValueError(f"normalization is missing group {name!r}")
            median = torch.as_tensor(group.get("median"), dtype=torch.float32)
            scale = torch.as_tensor(group.get("scale_mad"), dtype=torch.float32)
            count = torch.as_tensor(group.get("count"), dtype=torch.long)
            if median.shape != (width,) or scale.shape != (width,) or count.shape != (width,):
                raise ValueError(
                    f"normalization group {name!r} must have width {width}; "
                    f"got median={tuple(median.shape)}, scale={tuple(scale.shape)}, count={tuple(count.shape)}"
                )
            if not torch.isfinite(median).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError(f"normalization group {name!r} must contain finite median and positive scale")
            if torch.any(count < 0):
                raise ValueError(f"normalization group {name!r} has a negative valid-element count")
            self.groups[name] = RobustGroupStats(median=median, scale=scale, count=count)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_schema_version: str | None = None,
    ) -> ActionNormalization:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"normalization file does not exist: {source}")
        with source.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        return cls(
            metadata,
            source=source,
            expected_schema_version=expected_schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated copy suitable for checkpoint metadata."""

        return copy.deepcopy(self._metadata)

    @staticmethod
    def _normalize(values: Tensor, valid: Tensor, stats: RobustGroupStats, *, name: str) -> Tensor:
        if valid.shape == values.shape:
            expanded_valid = valid
        elif valid.shape == values.shape[:-1]:
            expanded_valid = valid.unsqueeze(-1).expand_as(values)
        else:
            raise ValueError(
                f"{name} mask shape {tuple(valid.shape)} is incompatible with values {tuple(values.shape)}"
            )
        if torch.any(expanded_valid & ~torch.isfinite(values)):
            raise ValueError(f"{name} contains NaN/Inf in valid elements")
        normalized = (values - stats.median) / stats.scale
        return torch.where(expanded_valid, normalized, torch.zeros_like(normalized))

    def trajectory(
        self,
        raw: Tensor,
        *,
        time_mask: Tensor,
        pose_mask: Tensor,
        gripper_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if raw.ndim != 2 or raw.shape[1] != TRAJECTORY_DIM:
            raise ValueError(f"raw trajectory must be [T,{TRAJECTORY_DIM}], got {tuple(raw.shape)}")
        if any(mask.shape != (raw.shape[0],) for mask in (time_mask, pose_mask, gripper_mask)):
            raise ValueError("trajectory masks must match the raw trajectory horizon")
        pose_valid = time_mask & pose_mask
        gripper_valid = time_mask & gripper_mask
        position = self._normalize(raw[:, :3], pose_valid, self.groups["position"], name="position")
        rotation = self._normalize(
            raw[:, 3:6],
            pose_valid,
            self.groups["rotation_rotvec"],
            name="rotation_rotvec",
        )
        gripper = self._normalize(raw[:, 6:7], gripper_valid, self.groups["gripper"], name="gripper")
        trajectory_mask = torch.cat(
            (
                pose_valid[:, None].expand(-1, 6),
                gripper_valid[:, None],
            ),
            dim=-1,
        )
        return torch.cat((position, rotation, gripper), dim=-1), trajectory_mask

    def current_state(self, raw: Tensor, mask: Tensor) -> Tensor:
        if raw.shape != (STATE_DIM,) or mask.shape != (STATE_DIM,):
            raise ValueError(f"current state/mask must both be {(STATE_DIM,)}")
        return self._normalize(
            raw,
            mask,
            self.groups["current_state"],
            name="current_state",
        )


class ActionSidecarDataset(Dataset[dict[str, Any]]):
    """Read one immutable sidecar split with process-local HDF5 handles."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        max_open_shards: int = 8,
        verify_row_identity: bool = True,
        require_e39: bool = False,
        prior_context_path: str | Path | None = None,
        require_prior_context: bool = False,
        subset_size: int | None = None,
        subset_seed: int = 20260803,
    ) -> None:
        super().__init__()
        if max_open_shards < 1:
            raise ValueError(f"max_open_shards must be >= 1, got {max_open_shards}")
        if subset_size is not None and subset_size < 1:
            raise ValueError(f"subset_size must be positive when provided, got {subset_size}")
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = _normalize_split(split)
        self.max_open_shards = int(max_open_shards)
        self.verify_row_identity = bool(verify_row_identity)
        self.require_e39 = bool(require_e39)
        self.prior_context_path = (
            Path(prior_context_path).expanduser().resolve() if prior_context_path is not None else None
        )
        self.require_prior_context = bool(require_prior_context)
        if self.require_prior_context and self.prior_context_path is None:
            raise ValueError("ActionVAE requires prior_context_path")
        self.subset_seed = int(subset_seed)

        schema_path = self.data_root / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"sidecar schema does not exist: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.schema_version = str(schema.get("schema_version", ""))
        if self.schema_version not in SUPPORTED_SIDECAR_SCHEMA_VERSIONS:
            raise ValueError(
                f"sidecar schema must be one of {sorted(SUPPORTED_SIDECAR_SCHEMA_VERSIONS)!r}, "
                f"got {self.schema_version!r}"
            )
        self.horizon = int(schema.get("horizon_actions", HORIZON))
        if self.horizon < 1:
            raise ValueError(f"sidecar schema has invalid horizon_actions={self.horizon}")
        if schema.get("waypoint", {}).get("batch_shape") != ["B", self.horizon, TRAJECTORY_DIM]:
            raise ValueError("sidecar schema has an incompatible waypoint contract")
        self.physical_action_steps = int(schema.get("physical_action_steps", self.horizon))
        if not 1 <= self.physical_action_steps <= self.horizon:
            raise ValueError(
                "sidecar schema physical_action_steps must be in "
                f"[1,{self.horizon}], got {self.physical_action_steps}"
            )
        self.representation_hz = float(
            schema.get("representation_hz", schema.get("control_hz", 0.0))
        )
        self.source_control_hz = float(
            schema.get("source_control_hz", schema.get("control_hz", 0.0))
        )
        if self.representation_hz <= 0 or self.source_control_hz <= 0:
            raise ValueError("sidecar schema must define positive source and representation rates")
        self._h5_field_shapes = _h5_field_shapes(self.horizon)
        self._has_physical_valid_count = "physical_valid_count" in schema.get("fields", {})
        if self._has_physical_valid_count:
            self._h5_field_shapes["physical_valid_count"] = ()

        self.manifest_path = self.data_root / "manifests" / _MANIFEST_FILENAMES[self.split]
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"official {self.split!r} manifest does not exist: {self.manifest_path}")
        table = parquet.read_table(self.manifest_path)
        missing = sorted(_REQUIRED_MANIFEST_COLUMNS.difference(table.column_names))
        if missing:
            raise ValueError(f"manifest {self.manifest_path} is missing required columns: {missing}")
        all_columns = {name: list(values) for name, values in table.to_pydict().items()}
        full_length = int(table.num_rows)
        manifest_splits = {str(value) for value in all_columns["split"]}
        if full_length and manifest_splits != {self.split}:
            raise ValueError(
                f"manifest {self.manifest_path} must contain only split={self.split!r}; "
                f"found {sorted(manifest_splits)}"
            )
        all_sample_ids = [str(value) for value in all_columns["sample_id"]]
        if len(set(all_sample_ids)) != len(all_sample_ids):
            raise ValueError(f"manifest {self.manifest_path} contains duplicate sample_id values")
        all_shard_rows = [int(value) for value in all_columns["shard_row"]]
        if any(row < 0 for row in all_shard_rows):
            raise ValueError(f"manifest {self.manifest_path} contains a negative shard_row")
        window_lengths = np.asarray(all_columns["end_step"], dtype=np.int64) - np.asarray(
            all_columns["start_step"], dtype=np.int64
        )
        if len(window_lengths) and not np.all(window_lengths == self.physical_action_steps):
            raise ValueError(
                f"manifest {self.manifest_path} contains windows that do not match "
                f"physical_action_steps {self.physical_action_steps}"
            )

        if subset_size is None or subset_size == full_length:
            selected = list(range(full_length))
        else:
            if subset_size > full_length:
                raise ValueError(f"subset_size={subset_size} exceeds {self.split} split size {full_length}")
            rng = np.random.default_rng(self.subset_seed)
            selected = sorted(rng.choice(full_length, size=int(subset_size), replace=False).tolist())
        self._manifest_indices = selected
        self._manifest_columns = {
            name: [values[index] for index in selected] for name, values in all_columns.items()
        }
        self._length = len(selected)
        self._sample_ids = [str(value) for value in self._manifest_columns["sample_id"]]
        self._shard_rows = [int(value) for value in self._manifest_columns["shard_row"]]
        resolved_by_manifest_path: dict[str, str] = {}
        self._shard_paths = []
        for value in self._manifest_columns["shard_path"]:
            manifest_path = str(value)
            resolved = resolved_by_manifest_path.get(manifest_path)
            if resolved is None:
                resolved = self._resolve_shard_path(manifest_path)
                resolved_by_manifest_path[manifest_path] = resolved
            self._shard_paths.append(resolved)
        locations = list(zip(self._shard_paths, self._shard_rows))
        if len(set(locations)) != len(locations):
            raise ValueError(f"manifest {self.manifest_path} references a shard row more than once")

        stats_path = self.data_root / "stats" / "normalization.json"
        self.normalization = ActionNormalization.from_file(
            stats_path,
            expected_schema_version=NORMALIZATION_SCHEMA_BY_SIDECAR[self.schema_version],
        )
        self._h5_handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._handle_owner_pid: int | None = None
        self._memory_cache: tuple[dict[str, Any], ...] | None = None
        self.prior_context = (
            ActionPriorContextCache(
                self.prior_context_path,
                sample_ids=self._sample_ids,
                split=self.split,
            )
            if self.prior_context_path is not None
            else None
        )

    def _resolve_shard_path(self, manifest_value: str) -> str:
        raw = Path(manifest_value).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(self.data_root / raw)
        shard_positions = [index for index, part in enumerate(raw.parts) if part == "shards"]
        if shard_positions:
            candidates.append(self.data_root.joinpath(*raw.parts[shard_positions[-1] :]))
        candidates.append(self.data_root / "shards" / self.split / raw.name)
        seen: set[str] = set()
        unique: list[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"could not resolve shard path {manifest_value!r}; tried {', '.join(str(path) for path in unique)}"
        )

    def __len__(self) -> int:
        return self._length

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self._sample_ids)

    @property
    def sample_locations(self) -> tuple[tuple[str, int], ...]:
        return tuple(zip(self._shard_paths, self._shard_rows))

    @property
    def manifest_indices(self) -> tuple[int, ...]:
        """Original official-manifest row indices retained by this dataset."""

        return tuple(self._manifest_indices)

    @property
    def handle_owner_pid(self) -> int | None:
        return self._handle_owner_pid

    @property
    def open_shard_count(self) -> int:
        return len(self._h5_handles)

    def manifest_metadata(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        metadata = {name: values[index] for name, values in self._manifest_columns.items()}
        metadata.update(
            {
                "dataset_index": index,
                "manifest_index": self._manifest_indices[index],
                "manifest_shard_path": str(self._manifest_columns["shard_path"][index]),
                "shard_path": self._shard_paths[index],
                "shard_row": self._shard_rows[index],
            }
        )
        return metadata

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"ActionSidecarDataset index {index} out of range for length {self._length}")
        return int(index)

    def _close_cached_handles(self) -> None:
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        if hasattr(self, "_h5_handles"):
            self._h5_handles.clear()

    def close(self) -> None:
        self._close_cached_handles()
        if self.prior_context is not None:
            self.prior_context.close()
        self._handle_owner_pid = None

    def reset_worker_state(self) -> None:
        self._close_cached_handles()
        self._h5_handles = OrderedDict()
        self._handle_owner_pid = os.getpid()
        if self.prior_context is not None:
            self.prior_context.reset_worker_state()

    def _ensure_process_local_cache(self) -> None:
        if self._handle_owner_pid != os.getpid():
            self.reset_worker_state()

    def _get_h5_handle(self, shard_path: str) -> h5py.File:
        self._ensure_process_local_cache()
        cached = self._h5_handles.pop(shard_path, None)
        if cached is not None:
            self._h5_handles[shard_path] = cached
            return cached
        handle = h5py.File(shard_path, "r")
        try:
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"sidecar shard is incomplete: {shard_path}")
            schema_version = _decode_h5_string(handle.attrs.get("schema_version", ""))
            if schema_version != self.schema_version:
                raise ValueError(
                    f"shard {shard_path} has schema {schema_version!r}, expected {self.schema_version!r}"
                )
            missing = sorted(set(self._h5_field_shapes).difference(handle.keys()))
            if missing:
                raise ValueError(f"shard {shard_path} is missing required fields: {missing}")
            row_count = int(handle["sample_id"].shape[0])
            for name, row_shape in self._h5_field_shapes.items():
                actual_shape = tuple(handle[name].shape)
                expected_shape = (row_count, *row_shape)
                if actual_shape != expected_shape:
                    raise ValueError(f"shard field {name!r} has shape {actual_shape}, expected {expected_shape}")
            self._h5_handles[shard_path] = handle
            while len(self._h5_handles) > self.max_open_shards:
                _, evicted = self._h5_handles.popitem(last=False)
                evicted.close()
        except Exception:
            handle.close()
            raise
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        if self._memory_cache is not None:
            return self._memory_cache[index]
        shard_path = self._shard_paths[index]
        shard_row = self._shard_rows[index]
        handle = self._get_h5_handle(shard_path)
        row_count = int(handle["sample_id"].shape[0])
        if shard_row >= row_count:
            raise IndexError(
                f"manifest index {self._manifest_indices[index]} references shard_row={shard_row}, "
                f"but {shard_path} has {row_count} rows"
            )

        sample_id = _decode_h5_string(handle["sample_id"][shard_row])
        stored_split = _decode_h5_string(handle["split"][shard_row])
        if self.verify_row_identity:
            if sample_id != self._sample_ids[index]:
                raise ValueError(
                    f"manifest/HDF5 sample mismatch at dataset index {index}: "
                    f"manifest={self._sample_ids[index]!r}, h5={sample_id!r}"
                )
            if stored_split != self.split:
                raise ValueError(
                    f"manifest/HDF5 split mismatch at dataset index {index}: "
                    f"expected={self.split!r}, h5={stored_split!r}"
                )
            if "window_sha256" in self._manifest_columns:
                manifest_checksum = str(self._manifest_columns["window_sha256"][index])
                stored_checksum = _decode_h5_string(handle["window_sha256"][shard_row])
                if manifest_checksum != stored_checksum:
                    raise ValueError(
                        f"manifest/HDF5 window checksum mismatch for {sample_id}: "
                        f"{manifest_checksum!r} != {stored_checksum!r}"
                    )

        masks = {
            name: _binary_mask(handle[name][shard_row], name=name)
            for name in _MASK_FIELDS
        }
        trajectory_raw = _tensor(handle["waypoints"][shard_row], torch.float32)
        current_state_raw = _tensor(handle["current_state"][shard_row], torch.float32)
        trajectory, trajectory_mask = self.normalization.trajectory(
            trajectory_raw,
            time_mask=masks["time_mask"],
            pose_mask=masks["pose_mask"],
            gripper_mask=masks["gripper_mask"],
        )
        current_state = self.normalization.current_state(current_state_raw, masks["current_state_mask"])

        e39_value = int(np.asarray(handle["e39_valid"][shard_row]).item())
        if e39_value not in (0, 1):
            raise ValueError(f"e39_valid must be binary in {shard_path}:{shard_row}, got {e39_value}")
        e39_valid = torch.tensor(bool(e39_value), dtype=torch.bool)
        e39_mean = _tensor(handle["e39_mean"][shard_row], torch.float32)
        if e39_valid and not torch.isfinite(e39_mean).all():
            raise ValueError(f"valid e39_mean contains NaN/Inf in {shard_path}:{shard_row}")
        e39_mean = torch.where(e39_valid, e39_mean, torch.zeros_like(e39_mean))
        if self.require_e39 and not bool(e39_valid):
            raise ValueError(f"A1 requires e39_mean, but sample {sample_id!r} has e39_valid=0")

        metadata = self.manifest_metadata(index)
        metadata["embodiment_id"] = _decode_h5_string(handle["embodiment_id"][shard_row])
        metadata["source_control_hz"] = self.source_control_hz
        metadata["representation_hz"] = self.representation_hz
        physical_valid_count = self.horizon
        if self._has_physical_valid_count:
            physical_valid_count = int(
                np.asarray(handle["physical_valid_count"][shard_row]).item()
            )
        if physical_valid_count != self.physical_action_steps:
            raise ValueError(
                f"physical_valid_count must equal schema physical_action_steps="
                f"{self.physical_action_steps} in {shard_path}:{shard_row}, "
                f"got {physical_valid_count}"
            )
        result: dict[str, Any] = {
            "trajectory_raw": trajectory_raw,
            "trajectory": trajectory,
            "trajectory_mask": trajectory_mask,
            "current_state_raw": current_state_raw,
            "current_state": current_state,
            "e39_mean": e39_mean,
            "e39_valid": e39_valid,
            "contact": _tensor(handle["contact"][shard_row], torch.float32),
            "grasp": _tensor(handle["grasp"][shard_row], torch.float32),
            "native_actions": _tensor(handle["native_actions"][shard_row], torch.float32),
            "native_action_dim": torch.tensor(
                int(np.asarray(handle["native_action_dim"][shard_row]).item()), dtype=torch.long
            ),
            "physical_anchors": _tensor(handle["physical_anchors"][shard_row], torch.float32),
            "physical_valid_count": torch.tensor(physical_valid_count, dtype=torch.long),
            "metadata": metadata,
        }
        result.update(masks)
        if self.prior_context is not None:
            result.update(self.prior_context.get(sample_id))
        return result

    def preload(self) -> None:
        """Materialize the immutable split once in shard/row order."""

        if self._memory_cache is not None:
            return
        cached: list[dict[str, Any] | None] = [None] * self._length
        order = sorted(
            range(self._length),
            key=lambda index: (
                self._shard_paths[index],
                self._shard_rows[index],
                index,
            ),
        )
        for index in order:
            cached[index] = self[index]
        if any(item is None for item in cached):
            raise RuntimeError(f"in-memory preload did not fill split {self.split!r}")
        self._memory_cache = tuple(item for item in cached if item is not None)
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_handles"] = OrderedDict()
        state["_handle_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._h5_handles = OrderedDict()
        self._handle_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_STACK_KEYS = (
    "trajectory_raw",
    "trajectory",
    "trajectory_mask",
    "current_state_raw",
    "current_state",
    "e39_mean",
    "e39_valid",
    "time_mask",
    "pose_mask",
    "gripper_mask",
    "contact",
    "contact_mask",
    "grasp",
    "grasp_mask",
    "native_actions",
    "native_action_dim",
    "native_action_mask",
    "current_state_mask",
    "physical_anchors",
    "physical_anchor_mask",
    "physical_valid_count",
)


def action_sidecar_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("action_sidecar_collate received an empty batch")
    collated = {key: torch.stack([sample[key] for sample in batch], dim=0) for key in _STACK_KEYS}
    optional = [key for key in PRIOR_CONTEXT_KEYS if key in batch[0]]
    if optional and any(any(key not in sample for key in optional) for sample in batch):
        raise ValueError("ActionVAE prior context fields must be present for every row")
    collated.update({key: torch.stack([sample[key] for sample in batch], dim=0) for key in optional})
    metadata = [dict(sample["metadata"]) for sample in batch]
    collated.update(
        {
            "metadata": metadata,
            "sample_ids": [str(item["sample_id"]) for item in metadata],
            "dataset_indices": torch.tensor([int(item["dataset_index"]) for item in metadata]),
            "manifest_indices": torch.tensor([int(item["manifest_index"]) for item in metadata]),
            "shard_rows": torch.tensor([int(item["shard_row"]) for item in metadata]),
            "shard_paths": [str(item["shard_path"]) for item in metadata],
        }
    )
    return collated


def _action_sidecar_worker_init_fn(worker_id: int) -> None:
    del worker_id
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reset_worker_state"):
        info.dataset.reset_worker_state()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ActionSidecarDataModule(LightningDataModule):
    """Lightning DataModule that never changes official split ownership."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        variant: str = "a0",
        batch_size: int = 256,
        num_workers: int = 4,
        block_size: int | None = None,
        max_open_shards: int = 8,
        verify_row_identity: bool = True,
        require_e39: bool | None = None,
        prior_context_path: str | Path | None = None,
        require_prior_context: bool = False,
        train_subset_size: int | None = None,
        val_subset_size: int | None = None,
        test_subset_size: int | None = None,
        subset_seed: int = 20260803,
        drop_last: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        preload_in_memory: bool = False,
        in_order: bool = True,
        seed: int = 20260803,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"a0", "a1"}:
            raise ValueError("variant must be 'a0' or 'a1'")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if prefetch_factor is not None and prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        self.data_root = Path(data_root)
        self.variant = str(variant)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.block_size = block_size
        self.max_open_shards = int(max_open_shards)
        self.verify_row_identity = bool(verify_row_identity)
        expected_e39 = self.variant == "a1"
        if require_e39 is not None and bool(require_e39) != expected_e39:
            raise ValueError(f"variant={self.variant!r} requires require_e39={expected_e39}")
        self.require_e39 = expected_e39
        self.prior_context_path = prior_context_path
        self.require_prior_context = bool(require_prior_context)
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.test_subset_size = test_subset_size
        self.subset_seed = int(subset_seed)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.preload_in_memory = bool(preload_in_memory)
        self.in_order = bool(in_order)
        self.seed = int(seed)
        self.rank = rank
        self.world_size = world_size
        self.train_dataset: ActionSidecarDataset | None = None
        self.val_dataset: ActionSidecarDataset | None = None
        self.test_dataset: ActionSidecarDataset | None = None
        self.train_batch_sampler: ShardAwareBatchSampler | None = None
        self._epoch = 0

    def _new_dataset(self, split: str, subset_size: int | None) -> ActionSidecarDataset:
        dataset = ActionSidecarDataset(
            self.data_root,
            split,
            max_open_shards=self.max_open_shards,
            verify_row_identity=self.verify_row_identity,
            require_e39=self.require_e39,
            prior_context_path=self.prior_context_path,
            require_prior_context=self.require_prior_context,
            subset_size=subset_size,
            subset_seed=self.subset_seed,
        )
        if self.preload_in_memory:
            dataset.preload()
        return dataset

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._new_dataset("train", self.train_subset_size)
            if self.val_dataset is None:
                self.val_dataset = self._new_dataset("validation", self.val_subset_size)
        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = self._new_dataset("validation", self.val_subset_size)
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = self._new_dataset("test", self.test_subset_size)
        if stage not in (None, "fit", "validate", "test", "predict"):
            raise ValueError(f"unsupported setup stage {stage!r}")

    def _loader(
        self,
        dataset: ActionSidecarDataset,
        *,
        shuffle: bool,
        drop_last: bool,
        order: str = "shard",
    ) -> tuple[DataLoader, ShardAwareBatchSampler]:
        generator = torch.Generator().manual_seed(self.seed)
        rank = self.rank
        world_size = self.world_size
        trainer = getattr(self, "trainer", None)
        if rank is None and trainer is not None:
            rank = int(getattr(trainer, "global_rank", 0))
        if world_size is None and trainer is not None:
            world_size = int(getattr(trainer, "world_size", 1))
        sampler = ShardAwareBatchSampler(
            dataset,
            self.batch_size,
            block_size=self.block_size,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
            rank=rank,
            world_size=world_size,
            order=order,
        )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "worker_init_fn": _action_sidecar_worker_init_fn if self.num_workers > 0 else None,
            "collate_fn": action_sidecar_collate,
            "generator": generator,
        }
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            if "in_order" in inspect.signature(DataLoader).parameters:
                kwargs["in_order"] = self.in_order
        return DataLoader(**kwargs), sampler

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first")
        loader, self.train_batch_sampler = self._loader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.drop_last,
        )
        self.train_batch_sampler.set_epoch(self._epoch)
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized; call setup('fit') or setup('validate') first")
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)[0]

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized; call setup('test') first")
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)[0]

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def full_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first")
        return self._loader(
            self.train_dataset,
            shuffle=False,
            drop_last=False,
            order="manifest",
        )[0]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.train_batch_sampler is not None:
            self.train_batch_sampler.set_epoch(self._epoch)

    @property
    def normalization_metadata(self) -> dict[str, Any]:
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                return dataset.normalization.to_dict()
        raise RuntimeError("normalization is unavailable before setup")

    def teardown(self, stage: str | None = None) -> None:
        del stage
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()
