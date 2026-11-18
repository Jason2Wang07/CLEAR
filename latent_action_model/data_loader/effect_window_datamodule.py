"""Lightning DataModule for the Stage-1 effect-window release."""

from __future__ import annotations

import inspect
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, get_worker_info

from .effect_window_dataset import (
    EffectWindowDataset,
    MixedBalancedBatchSampler,
    ShardAwareBatchSampler,
    effect_window_collate,
)


def _effect_window_worker_init_fn(worker_id: int) -> None:
    """Reset inherited HDF5 state and deterministically seed worker libraries."""

    del worker_id
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reset_worker_state"):
        info.dataset.reset_worker_state()

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    thread_count_text = os.environ.get("LAM_WORKER_OMP_THREADS", "1")
    thread_count = int(thread_count_text) if thread_count_text.isdigit() else 1
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = str(thread_count)
    torch.set_num_threads(max(1, thread_count))
    try:
        torch.set_num_interop_threads(max(1, min(thread_count, 2)))
    except RuntimeError:
        pass


class EffectWindowDataModule(LightningDataModule):
    """DataModule using official splits and process-local lazy HDF5 readers.

    ``batch_size`` is global.  The per-rank DataLoader batch is
    ``batch_size / world_size`` and divisibility is validated by the sampler.
    The custom sampler already owns DDP partitioning.  A Lightning Trainer
    using this DataModule should therefore set ``use_distributed_sampler=False``
    so Lightning does not replace it with a padding sampler.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 256,
        num_workers: int = 4,
        train_block_size: int | None = None,
        eval_block_size: int | None = None,
        max_open_shards: int = 8,
        drop_last: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        in_order: bool = True,
        seed: int = 20260803,
        rank: int | None = None,
        world_size: int | None = None,
        verify_row_identity: bool = True,
        train_sampling: str = "natural",
        natural_fraction: float = 0.5,
        state_normalization_path: str | Path | None = None,
        release_validation_baseline_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
        if prefetch_factor is not None and prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1 when provided, got {prefetch_factor}.")

        self.data_root = Path(data_root)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_block_size = train_block_size
        self.eval_block_size = eval_block_size
        self.max_open_shards = int(max_open_shards)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.in_order = bool(in_order)
        self.seed = int(seed)
        self.rank = rank
        self.world_size = world_size
        self.verify_row_identity = bool(verify_row_identity)
        self.state_normalization_path = state_normalization_path
        # Retained in the serialized run config as part of the data contract;
        # checkpoint selection consumes it outside the DataLoader itself.
        self.release_validation_baseline_path = release_validation_baseline_path
        if train_sampling not in {"natural", "mixed_balanced"}:
            raise ValueError("train_sampling must be 'natural' or 'mixed_balanced'")
        self.train_sampling = str(train_sampling)
        self.natural_fraction = float(natural_fraction)

        self.train_dataset: EffectWindowDataset | None = None
        self.val_dataset: EffectWindowDataset | None = None
        self.test_dataset: EffectWindowDataset | None = None
        self.train_batch_sampler: ShardAwareBatchSampler | None = None
        self.val_batch_sampler: ShardAwareBatchSampler | None = None
        self.test_batch_sampler: ShardAwareBatchSampler | None = None
        self.full_train_batch_sampler: ShardAwareBatchSampler | None = None
        self._train_epoch = 0

    def _new_dataset(self, split: str) -> EffectWindowDataset:
        return EffectWindowDataset(
            self.data_root,
            split,
            max_open_shards=self.max_open_shards,
            verify_row_identity=self.verify_row_identity,
            state_normalization_path=self.state_normalization_path,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._new_dataset("train")
            if self.val_dataset is None:
                self.val_dataset = self._new_dataset("validation")
        if stage in (None, "validate"):
            if self.val_dataset is None:
                self.val_dataset = self._new_dataset("validation")
        if stage in (None, "test", "predict"):
            if self.test_dataset is None:
                self.test_dataset = self._new_dataset("test")
        if stage not in (None, "fit", "validate", "test", "predict"):
            raise ValueError(f"Unsupported setup stage {stage!r}.")

    def _make_loader(
        self,
        dataset: EffectWindowDataset,
        *,
        shuffle: bool,
        drop_last: bool,
        block_size: int | None,
        order: str = "shard",
    ) -> tuple[DataLoader, ShardAwareBatchSampler]:
        sampler = ShardAwareBatchSampler(
            dataset,
            self.batch_size,
            block_size=block_size,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
            rank=self.rank,
            world_size=self.world_size,
            order=order,
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": effect_window_collate,
            "worker_init_fn": _effect_window_worker_init_fn if self.num_workers > 0 else None,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "generator": generator,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            if "in_order" in inspect.signature(DataLoader).parameters:
                loader_kwargs["in_order"] = self.in_order
        return DataLoader(**loader_kwargs), sampler

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first.")
        if self.train_sampling == "mixed_balanced":
            sampler = MixedBalancedBatchSampler(
                self.train_dataset,
                self.batch_size,
                natural_fraction=self.natural_fraction,
                seed=self.seed,
                drop_last=self.drop_last,
                rank=self.rank,
                world_size=self.world_size,
            )
            generator = torch.Generator().manual_seed(self.seed)
            loader_kwargs: dict[str, Any] = {
                "dataset": self.train_dataset,
                "batch_sampler": sampler,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "collate_fn": effect_window_collate,
                "worker_init_fn": _effect_window_worker_init_fn if self.num_workers > 0 else None,
                "persistent_workers": self.num_workers > 0 and self.persistent_workers,
                "generator": generator,
            }
            if self.num_workers > 0:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
                if "in_order" in inspect.signature(DataLoader).parameters:
                    loader_kwargs["in_order"] = self.in_order
            loader = DataLoader(**loader_kwargs)
            self.train_batch_sampler = sampler
        else:
            loader, self.train_batch_sampler = self._make_loader(
                self.train_dataset,
                shuffle=True,
                drop_last=self.drop_last,
                block_size=self.train_block_size,
            )
        self.train_batch_sampler.set_epoch(self._train_epoch)
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized; call setup('fit') or setup('validate') first.")
        loader, self.val_batch_sampler = self._make_loader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            block_size=self.eval_block_size,
        )
        return loader

    def full_train_dataloader(self) -> DataLoader:
        """Return train rows in manifest-index order without padding or dropping.

        This loader is intended for release-bank initialization and periodic
        full latent-bank refreshes.  In DDP, each rank owns a unique monotonic
        subsequence and the union across ranks is exactly ``range(len(train))``.
        ``dataset_indices`` in every batch provides the memory-bank write keys.
        """

        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first.")
        loader, self.full_train_batch_sampler = self._make_loader(
            self.train_dataset,
            shuffle=False,
            drop_last=False,
            block_size=self.batch_size,
            order="manifest",
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized; call setup('test') first.")
        loader, self.test_batch_sampler = self._make_loader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            block_size=self.eval_block_size,
        )
        return loader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def set_epoch(self, epoch: int) -> bool:
        self._train_epoch = int(epoch)
        if self.train_batch_sampler is None:
            return False
        self.train_batch_sampler.set_epoch(self._train_epoch)
        return True

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.set_epoch(int(getattr(trainer, "current_epoch", 0)))

    def teardown(self, stage: Optional[str] = None) -> None:
        del stage
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()
