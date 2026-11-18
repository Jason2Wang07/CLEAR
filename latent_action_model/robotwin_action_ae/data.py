"""Lazy readers for immutable RoboTwin dual-arm ActionAE sidecars."""

from __future__ import annotations

import copy
import inspect
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyarrow.parquet as parquet
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from latent_action_model.data_loader.effect_window_dataset import ShardAwareBatchSampler
from latent_action_model.action_ae.context import ActionPriorContextCache, PRIOR_CONTEXT_KEYS


SCHEMA_VERSION = "robotwin-z-action-sidecar-v1"
NORMALIZATION_SCHEMA_VERSION = "robotwin-z-action-normalization-v1"
SCHEMA_VERSION_12S = "robotwin-z-action-sidecar-v1-12s"
SCHEMA_VERSION_12S_LAWAM = "robotwin-z-action-sidecar-v2-12s-lawam-idm"
NORMALIZATION_SCHEMA_VERSION_12S = "robotwin-z-action-normalization-v1-12s"
NORMALIZATION_BY_SCHEMA = {
    SCHEMA_VERSION: NORMALIZATION_SCHEMA_VERSION,
    SCHEMA_VERSION_12S: NORMALIZATION_SCHEMA_VERSION_12S,
    SCHEMA_VERSION_12S_LAWAM: NORMALIZATION_SCHEMA_VERSION_12S,
}
CONDITION_FIELDS_BY_SCHEMA = {
    SCHEMA_VERSION: ("e39", "e39_mean", "e39_valid"),
    SCHEMA_VERSION_12S: ("e39", "e39_mean", "e39_valid"),
    SCHEMA_VERSION_12S_LAWAM: ("lawam_idm", "lawam_mean", "lawam_valid"),
}
HORIZON = 48
TRAJECTORY_DIM = 14
STATE_DIM = 14
EFFECT_DIM = 32
NATIVE_ACTION_DIM = 16
ARM_COUNT = 2
ANCHOR_COUNT = 4
SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id", "group_id", "split", "task_id", "source_id", "controller_id",
    "episode_id", "start_step", "end_step", "shard_path", "shard_row",
}
def _h5_field_shapes(
    horizon: int, *, mean_field: str = "e39_mean", valid_field: str = "e39_valid"
) -> dict[str, tuple[int, ...]]:
    return {
    "sample_id": (),
    "split": (),
    "embodiment_id": (),
    "waypoints": (horizon, TRAJECTORY_DIM),
    "time_mask": (horizon,),
    "arm_pose_mask": (horizon, ARM_COUNT),
    "arm_gripper_mask": (horizon, ARM_COUNT),
    "native_actions": (horizon, NATIVE_ACTION_DIM),
    "native_action_mask": (horizon, NATIVE_ACTION_DIM),
    "current_state": (STATE_DIM,),
    "current_state_mask": (STATE_DIM,),
    mean_field: (EFFECT_DIM,),
    valid_field: (),
    "physical_anchors": (ANCHOR_COUNT, TRAJECTORY_DIM),
    "physical_anchor_mask": (ANCHOR_COUNT, TRAJECTORY_DIM),
    "window_sha256": (),
    }


H5_FIELD_SHAPES: dict[str, tuple[int, ...]] = _h5_field_shapes(HORIZON)
MASK_FIELDS = (
    "time_mask", "arm_pose_mask", "arm_gripper_mask", "native_action_mask",
    "current_state_mask", "physical_anchor_mask",
)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode(value.item())
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


class RoboTwinActionNormalization:
    GROUP_WIDTHS = {
        "left_position": 3,
        "left_rotation_rotvec": 3,
        "left_gripper": 1,
        "right_position": 3,
        "right_rotation_rotvec": 3,
        "right_gripper": 1,
        "current_state": STATE_DIM,
    }

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        source: Path,
        expected_schema_version: str = NORMALIZATION_SCHEMA_VERSION,
    ) -> None:
        self.source = Path(source)
        self.metadata = copy.deepcopy(dict(metadata))
        if metadata.get("schema_version") != expected_schema_version:
            raise ValueError(f"normalization schema must be {expected_schema_version!r}")
        if metadata.get("fit_split") != "train":
            raise ValueError("RoboTwin ActionAE normalization must be fit on train")
        if metadata.get("estimator") != "median_and_max_1.4826_mad_rms_about_median":
            raise ValueError(f"unexpected normalization estimator {metadata.get('estimator')!r}")
        if int(metadata.get("sample_count", 0)) < 1:
            raise ValueError("normalization sample_count must be positive")
        self.groups: dict[str, tuple[Tensor, Tensor]] = {}
        for name, width in self.GROUP_WIDTHS.items():
            group = metadata.get(name)
            if not isinstance(group, Mapping):
                raise ValueError(f"normalization is missing group {name!r}")
            median = torch.as_tensor(group.get("median"), dtype=torch.float32)
            scale = torch.as_tensor(group.get("scale"), dtype=torch.float32)
            if median.shape != (width,) or scale.shape != (width,):
                raise ValueError(f"normalization group {name!r} must have width {width}")
            if not torch.isfinite(median).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError(f"normalization group {name!r} has invalid median/scale")
            self.groups[name] = (median, scale)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_schema_version: str = NORMALIZATION_SCHEMA_VERSION,
    ) -> "RoboTwinActionNormalization":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        return cls(
            json.loads(source.read_text(encoding="utf-8")),
            source=source,
            expected_schema_version=expected_schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.metadata)

    def _normalize(self, values: Tensor, valid: Tensor, group: str) -> Tensor:
        median, scale = self.groups[group]
        if valid.shape == values.shape[:-1]:
            valid = valid.unsqueeze(-1).expand_as(values)
        if valid.shape != values.shape:
            raise ValueError(f"mask {tuple(valid.shape)} is incompatible with {group} {tuple(values.shape)}")
        if torch.any(valid & ~torch.isfinite(values)):
            raise ValueError(f"{group} contains NaN/Inf in valid elements")
        normalized = (values - median) / scale
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    def trajectory(
        self,
        raw: Tensor,
        *,
        time_mask: Tensor,
        arm_pose_mask: Tensor,
        arm_gripper_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if raw.ndim != 2 or raw.shape[1] != TRAJECTORY_DIM:
            raise ValueError(f"raw trajectory must be [T,{TRAJECTORY_DIM}], got {tuple(raw.shape)}")
        horizon = raw.shape[0]
        if time_mask.shape != (horizon,):
            raise ValueError("RoboTwin time mask must match the trajectory horizon")
        if arm_pose_mask.shape != (horizon, ARM_COUNT) or arm_gripper_mask.shape != (horizon, ARM_COUNT):
            raise ValueError(f"RoboTwin arm masks must be [{horizon},2]")
        left_pose = time_mask & arm_pose_mask[:, 0]
        left_gripper = time_mask & arm_gripper_mask[:, 0]
        right_pose = time_mask & arm_pose_mask[:, 1]
        right_gripper = time_mask & arm_gripper_mask[:, 1]
        parts = (
            self._normalize(raw[:, 0:3], left_pose, "left_position"),
            self._normalize(raw[:, 3:6], left_pose, "left_rotation_rotvec"),
            self._normalize(raw[:, 6:7], left_gripper, "left_gripper"),
            self._normalize(raw[:, 7:10], right_pose, "right_position"),
            self._normalize(raw[:, 10:13], right_pose, "right_rotation_rotvec"),
            self._normalize(raw[:, 13:14], right_gripper, "right_gripper"),
        )
        mask = torch.cat(
            (
                left_pose[:, None].expand(-1, 6), left_gripper[:, None],
                right_pose[:, None].expand(-1, 6), right_gripper[:, None],
            ),
            dim=-1,
        )
        return torch.cat(parts, dim=-1), mask

    def current_state(self, raw: Tensor, mask: Tensor) -> Tensor:
        if raw.shape != (STATE_DIM,) or mask.shape != (STATE_DIM,):
            raise ValueError("current state and mask must both be [14]")
        return self._normalize(raw, mask, "current_state")


class RoboTwinActionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        max_open_shards: int = 8,
        verify_row_identity: bool = True,
        require_e39: bool = False,
        condition_source: str | None = None,
        require_condition: bool | None = None,
        prior_context_path: str | Path | None = None,
        require_prior_context: bool = False,
        subset_size: int | None = None,
        subset_seed: int = 20260820,
    ) -> None:
        super().__init__()
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        normalized_split = str(split).lower()
        if normalized_split not in SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = SPLITS[normalized_split]
        self.max_open_shards = int(max_open_shards)
        self.verify_row_identity = bool(verify_row_identity)
        self.require_condition = bool(require_e39 if require_condition is None else require_condition)
        self.prior_context_path = (
            Path(prior_context_path).expanduser().resolve() if prior_context_path is not None else None
        )
        self.require_prior_context = bool(require_prior_context)
        if self.require_prior_context and self.prior_context_path is None:
            raise ValueError("RoboTwin ActionVAE requires prior_context_path")
        schema_path = self.data_root / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.schema_version = str(schema.get("schema_version", ""))
        if self.schema_version not in NORMALIZATION_BY_SCHEMA:
            raise ValueError(f"unsupported sidecar schema {self.schema_version!r}")
        detected_source, self.condition_mean_field, self.condition_valid_field = (
            CONDITION_FIELDS_BY_SCHEMA[self.schema_version]
        )
        schema_source = str(schema.get("condition", {}).get("source", detected_source))
        if schema_source != detected_source:
            raise ValueError(
                f"sidecar condition source mismatch: schema={schema_source!r}, version={detected_source!r}"
            )
        if condition_source is not None and str(condition_source) != detected_source:
            raise ValueError(
                f"requested condition_source={condition_source!r}, sidecar provides {detected_source!r}"
            )
        self.condition_source = detected_source
        self.horizon = int(schema.get("horizon_actions", HORIZON))
        if schema.get("waypoint", {}).get("batch_shape") != ["B", self.horizon, TRAJECTORY_DIM]:
            raise ValueError("sidecar has an incompatible waypoint contract")
        self.h5_field_shapes = _h5_field_shapes(
            self.horizon,
            mean_field=self.condition_mean_field,
            valid_field=self.condition_valid_field,
        )

        self.manifest_path = self.data_root / "manifests" / f"{self.split}.parquet"
        table = parquet.read_table(self.manifest_path)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(table.column_names))
        if missing:
            raise ValueError(f"manifest is missing required columns: {missing}")
        columns = {name: list(values) for name, values in table.to_pydict().items()}
        full_length = int(table.num_rows)
        if full_length and {str(value) for value in columns["split"]} != {self.split}:
            raise ValueError(f"manifest must contain only split={self.split}")
        sample_ids = [str(value) for value in columns["sample_id"]]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("manifest contains duplicate sample_id values")
        window_lengths = np.asarray(columns["end_step"], dtype=np.int64) - np.asarray(
            columns["start_step"], dtype=np.int64
        )
        if len(window_lengths) and not np.all(window_lengths == self.horizon):
            raise ValueError(f"manifest windows do not match horizon {self.horizon}")
        if subset_size is None or subset_size == full_length:
            selected = list(range(full_length))
        else:
            if subset_size < 1 or subset_size > full_length:
                raise ValueError(f"invalid subset_size={subset_size} for split size {full_length}")
            rng = np.random.default_rng(int(subset_seed))
            selected = sorted(rng.choice(full_length, size=int(subset_size), replace=False).tolist())
        self._manifest_indices = selected
        self._columns = {name: [values[index] for index in selected] for name, values in columns.items()}
        self._sample_ids = [str(value) for value in self._columns["sample_id"]]
        self._shard_rows = [int(value) for value in self._columns["shard_row"]]
        self._shard_paths = [str((self.data_root / str(value)).resolve()) for value in self._columns["shard_path"]]
        if any(not Path(path).is_file() for path in self._shard_paths):
            missing_path = next(path for path in self._shard_paths if not Path(path).is_file())
            raise FileNotFoundError(missing_path)
        locations = list(zip(self._shard_paths, self._shard_rows))
        if len(locations) != len(set(locations)):
            raise ValueError("manifest references a shard row more than once")
        self.normalization = RoboTwinActionNormalization.from_file(
            self.data_root / "stats/normalization.json",
            expected_schema_version=NORMALIZATION_BY_SCHEMA[self.schema_version],
        )
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
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

    def __len__(self) -> int:
        return len(self._sample_ids)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self._sample_ids)

    @property
    def sample_locations(self) -> tuple[tuple[str, int], ...]:
        return tuple(zip(self._shard_paths, self._shard_rows))

    @property
    def manifest_indices(self) -> tuple[int, ...]:
        return tuple(self._manifest_indices)

    @property
    def open_shard_count(self) -> int:
        return len(self._handles)

    def manifest_metadata(self, index: int) -> dict[str, Any]:
        metadata = {name: values[index] for name, values in self._columns.items()}
        metadata.update(
            dataset_index=index,
            manifest_index=self._manifest_indices[index],
            manifest_shard_path=str(self._columns["shard_path"][index]),
            shard_path=self._shard_paths[index],
            shard_row=self._shard_rows[index],
        )
        return metadata

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(index)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._handle_owner_pid = None
        if self.prior_context is not None:
            self.prior_context.close()

    def reset_worker_state(self) -> None:
        self.close()
        self._handle_owner_pid = os.getpid()
        if self.prior_context is not None:
            self.prior_context.reset_worker_state()

    def _handle(self, path: str) -> h5py.File:
        if self._handle_owner_pid != os.getpid():
            self.reset_worker_state()
        cached = self._handles.pop(path, None)
        if cached is not None:
            self._handles[path] = cached
            return cached
        handle = h5py.File(path, "r")
        try:
            if _decode(handle.attrs.get("schema_version", "")) != self.schema_version or not handle.attrs.get("complete"):
                raise ValueError(f"invalid sidecar shard: {path}")
            missing = sorted(set(self.h5_field_shapes).difference(handle.keys()))
            if missing:
                raise ValueError(f"shard {path} is missing fields: {missing}")
            rows = int(handle["sample_id"].shape[0])
            for name, shape in self.h5_field_shapes.items():
                if tuple(handle[name].shape) != (rows, *shape):
                    raise ValueError(f"shard field {name!r} has invalid shape {handle[name].shape}")
            self._handles[path] = handle
            while len(self._handles) > self.max_open_shards:
                _, old = self._handles.popitem(last=False)
                old.close()
            return handle
        except Exception:
            handle.close()
            raise

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        if self._memory_cache is not None:
            return self._memory_cache[index]
        path = self._shard_paths[index]
        row = self._shard_rows[index]
        handle = self._handle(path)
        sample_id = _decode(handle["sample_id"][row])
        if self.verify_row_identity:
            if sample_id != self._sample_ids[index] or _decode(handle["split"][row]) != self.split:
                raise ValueError(f"manifest/HDF5 identity mismatch for {self._sample_ids[index]}")
            if "window_sha256" in self._columns:
                stored = _decode(handle["window_sha256"][row])
                if stored != str(self._columns["window_sha256"][index]):
                    raise ValueError(f"window checksum mismatch for {sample_id}")
        masks = {name: _binary_mask(handle[name][row], name=name) for name in MASK_FIELDS}
        trajectory_raw = _tensor(handle["waypoints"][row], torch.float32)
        current_state_raw = _tensor(handle["current_state"][row], torch.float32)
        trajectory, trajectory_mask = self.normalization.trajectory(
            trajectory_raw,
            time_mask=masks["time_mask"],
            arm_pose_mask=masks["arm_pose_mask"],
            arm_gripper_mask=masks["arm_gripper_mask"],
        )
        current_state = self.normalization.current_state(current_state_raw, masks["current_state_mask"])
        condition_value = int(np.asarray(handle[self.condition_valid_field][row]).item())
        if condition_value not in (0, 1):
            raise ValueError(f"{self.condition_valid_field} must be binary for {sample_id}")
        condition_valid = torch.tensor(bool(condition_value), dtype=torch.bool)
        condition_mean = _tensor(handle[self.condition_mean_field][row], torch.float32)
        if condition_valid and not torch.isfinite(condition_mean).all():
            raise ValueError(
                f"valid {self.condition_mean_field} contains NaN/Inf for {sample_id}"
            )
        condition_mean = torch.where(
            condition_valid, condition_mean, torch.zeros_like(condition_mean)
        )
        if self.require_condition and not bool(condition_valid):
            raise ValueError(
                f"A1 requires {self.condition_mean_field}, but sample {sample_id} is invalid"
            )
        metadata = self.manifest_metadata(index)
        metadata["embodiment_id"] = _decode(handle["embodiment_id"][row])
        result = {
            "trajectory": trajectory,
            "trajectory_raw": trajectory_raw,
            "trajectory_mask": trajectory_mask,
            "current_state": current_state,
            "current_state_raw": current_state_raw,
            "condition_mean": condition_mean,
            "condition_valid": condition_valid,
            # Preserve the original batch API while new code uses the source-neutral names.
            "e39_mean": condition_mean,
            "e39_valid": condition_valid,
            "time_mask": masks["time_mask"],
            "arm_pose_mask": masks["arm_pose_mask"],
            "arm_gripper_mask": masks["arm_gripper_mask"],
            "native_actions": _tensor(handle["native_actions"][row], torch.float32),
            "native_action_mask": masks["native_action_mask"],
            "current_state_mask": masks["current_state_mask"],
            "physical_anchors": _tensor(handle["physical_anchors"][row], torch.float32),
            "physical_anchor_mask": masks["physical_anchor_mask"],
            "metadata": metadata,
        }
        if self.prior_context is not None:
            result.update(self.prior_context.get(sample_id))
        return result

    def preload(self) -> None:
        if self._memory_cache is None:
            samples = tuple(self[index] for index in range(len(self)))
            self.close()
            self._memory_cache = samples

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_handle_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


STACK_KEYS = (
    "trajectory", "trajectory_raw", "trajectory_mask", "current_state", "current_state_raw",
    "condition_mean", "condition_valid", "e39_mean", "e39_valid", "time_mask",
    "arm_pose_mask", "arm_gripper_mask",
    "native_actions", "native_action_mask", "current_state_mask", "physical_anchors",
    "physical_anchor_mask",
)


def robotwin_action_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("robotwin_action_collate received an empty batch")
    collated = {key: torch.stack([sample[key] for sample in batch]) for key in STACK_KEYS}
    optional = [key for key in PRIOR_CONTEXT_KEYS if key in batch[0]]
    if optional and any(any(key not in sample for key in optional) for sample in batch):
        raise ValueError("RoboTwin ActionVAE prior context fields must be present for every row")
    collated.update({key: torch.stack([sample[key] for sample in batch]) for key in optional})
    metadata = [dict(sample["metadata"]) for sample in batch]
    collated.update(
        metadata=metadata,
        sample_ids=[str(item["sample_id"]) for item in metadata],
        dataset_indices=torch.tensor([int(item["dataset_index"]) for item in metadata]),
        manifest_indices=torch.tensor([int(item["manifest_index"]) for item in metadata]),
        shard_rows=torch.tensor([int(item["shard_row"]) for item in metadata]),
        shard_paths=[str(item["shard_path"]) for item in metadata],
    )
    return collated


def _worker_init(worker_id: int) -> None:
    del worker_id
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reset_worker_state"):
        info.dataset.reset_worker_state()
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


class RoboTwinActionDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        *,
        variant: str = "a0",
        batch_size: int = 256,
        num_workers: int = 4,
        block_size: int | None = 1024,
        max_open_shards: int = 8,
        verify_row_identity: bool = True,
        require_e39: bool | None = None,
        condition_source: str | None = None,
        require_condition: bool | None = None,
        prior_context_path: str | Path | None = None,
        require_prior_context: bool = False,
        train_subset_size: int | None = None,
        val_subset_size: int | None = None,
        test_subset_size: int | None = None,
        subset_seed: int = 20260820,
        drop_last: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        preload_in_memory: bool = False,
        in_order: bool = True,
        seed: int = 20260820,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"a0", "a1"}:
            raise ValueError("variant must be a0 or a1")
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        expected_e39 = variant == "a1"
        if require_e39 is not None and bool(require_e39) != expected_e39:
            raise ValueError(f"variant={variant!r} requires require_e39={expected_e39}")
        self.data_root = Path(data_root)
        self.variant = variant
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.block_size = block_size
        self.max_open_shards = int(max_open_shards)
        self.verify_row_identity = bool(verify_row_identity)
        self.require_e39 = expected_e39
        if require_condition is not None and bool(require_condition) != expected_e39:
            raise ValueError(f"variant={variant!r} requires require_condition={expected_e39}")
        self.condition_source = condition_source
        self.require_condition = expected_e39
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
        self.train_dataset: RoboTwinActionDataset | None = None
        self.val_dataset: RoboTwinActionDataset | None = None
        self.test_dataset: RoboTwinActionDataset | None = None
        self.train_batch_sampler: ShardAwareBatchSampler | None = None
        self._epoch = 0

    def _new_dataset(self, split: str, subset_size: int | None) -> RoboTwinActionDataset:
        dataset = RoboTwinActionDataset(
            self.data_root,
            split,
            max_open_shards=self.max_open_shards,
            verify_row_identity=self.verify_row_identity,
            require_e39=self.require_e39,
            condition_source=self.condition_source,
            require_condition=self.require_condition,
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
            self.train_dataset = self.train_dataset or self._new_dataset("train", self.train_subset_size)
            self.val_dataset = self.val_dataset or self._new_dataset("validation", self.val_subset_size)
        if stage in (None, "validate"):
            self.val_dataset = self.val_dataset or self._new_dataset("validation", self.val_subset_size)
        if stage in (None, "test", "predict"):
            self.test_dataset = self.test_dataset or self._new_dataset("test", self.test_subset_size)
        if stage not in (None, "fit", "validate", "test", "predict"):
            raise ValueError(f"unsupported setup stage {stage!r}")

    def _loader(
        self,
        dataset: RoboTwinActionDataset,
        *,
        shuffle: bool,
        drop_last: bool,
        order: str = "shard",
    ) -> tuple[DataLoader, ShardAwareBatchSampler]:
        sampler = ShardAwareBatchSampler(
            dataset,
            self.batch_size,
            block_size=self.block_size,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
            rank=self.rank,
            world_size=self.world_size,
            order=order,
        )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "worker_init_fn": _worker_init if self.num_workers > 0 else None,
            "collate_fn": robotwin_action_collate,
            "generator": torch.Generator().manual_seed(self.seed),
        }
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            if "in_order" in inspect.signature(DataLoader).parameters:
                kwargs["in_order"] = self.in_order
        return DataLoader(**kwargs), sampler

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') before train_dataloader")
        loader, self.train_batch_sampler = self._loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)
        self.train_batch_sampler.set_epoch(self._epoch)
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("validation dataset is not initialized")
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)[0]

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test dataset is not initialized")
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)[0]

    def full_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train dataset is not initialized")
        return self._loader(self.train_dataset, shuffle=False, drop_last=False, order="manifest")[0]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.train_batch_sampler is not None:
            self.train_batch_sampler.set_epoch(self._epoch)

    def teardown(self, stage: str | None = None) -> None:
        del stage
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()
