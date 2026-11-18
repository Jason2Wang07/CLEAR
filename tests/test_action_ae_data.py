from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from latent_action_model.action_ae import (
    ActionAELightningModule,
    ActionSidecarDataModule,
    ActionSidecarDataset,
)


POSITION_MEDIAN = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
POSITION_SCALE = np.asarray([2.0, 4.0, 5.0], dtype=np.float32)
ROTATION_MEDIAN = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
ROTATION_SCALE = np.asarray([0.5, 0.25, 2.0], dtype=np.float32)
GRIPPER_MEDIAN = np.asarray([0.04], dtype=np.float32)
GRIPPER_SCALE = np.asarray([0.01], dtype=np.float32)
STATE_MEDIAN = np.arange(14, dtype=np.float32)
STATE_SCALE = np.full(14, 2.0, dtype=np.float32)


def _write_normalization(root: Path) -> None:
    stats = {
        "schema_version": "maniskill-z-action-normalization-v1",
        "fit_split": "train",
        "estimator": "median_and_1.4826_mad",
        "sample_count": 6,
        "position": {
            "count": [100, 100, 100],
            "median": POSITION_MEDIAN.tolist(),
            "scale_mad": POSITION_SCALE.tolist(),
        },
        "rotation_rotvec": {
            "count": [100, 100, 100],
            "median": ROTATION_MEDIAN.tolist(),
            "scale_mad": ROTATION_SCALE.tolist(),
        },
        "gripper": {
            "count": [50],
            "median": GRIPPER_MEDIAN.tolist(),
            "scale_mad": GRIPPER_SCALE.tolist(),
        },
        "current_state": {
            "count": [6] * 7 + [0] * 7,
            "median": STATE_MEDIAN.tolist(),
            "scale_mad": STATE_SCALE.tolist(),
        },
    }
    path = root / "stats" / "normalization.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stats), encoding="utf-8")


def _write_shard(
    root: Path,
    split: str,
    shard_name: str,
    count: int,
    *,
    invalid_e39_last: bool = False,
) -> list[dict[str, Any]]:
    relative_path = Path("shards") / split / f"{shard_name}.h5"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [f"{split}-{shard_name}-{index:03d}" for index in range(count)]
    checksums = [f"checksum-{sample_id}" for sample_id in sample_ids]
    string_dtype = h5py.string_dtype("utf-8")

    waypoints = np.zeros((count, 32, 7), dtype=np.float32)
    time_mask = np.ones((count, 32), dtype=np.uint8)
    pose_mask = np.ones((count, 32), dtype=np.uint8)
    gripper_mask = np.ones((count, 32), dtype=np.uint8)
    current_state = np.zeros((count, 14), dtype=np.float32)
    current_state_mask = np.zeros((count, 14), dtype=np.uint8)
    e39_mean = np.zeros((count, 32), dtype=np.float32)
    e39_valid = np.ones(count, dtype=np.uint8)
    for index in range(count):
        factor = float(index + 1)
        waypoints[index, :, :3] = POSITION_MEDIAN + factor * POSITION_SCALE
        waypoints[index, :, 3:6] = ROTATION_MEDIAN + factor * ROTATION_SCALE
        waypoints[index, :, 6:7] = GRIPPER_MEDIAN + factor * GRIPPER_SCALE
        time_mask[index, -1] = 0
        pose_mask[index, -2] = 0
        waypoints[index, -1] = 999.0
        waypoints[index, -2, :6] = 999.0
        if index % 2 == 1:
            gripper_mask[index] = 0
            waypoints[index, :, 6] = 999.0
        current_state[index] = STATE_MEDIAN + factor * STATE_SCALE
        current_state_mask[index, :7] = 1
        current_state[index, 7:] = 999.0
        e39_mean[index] = np.arange(32, dtype=np.float32) + factor
    if invalid_e39_last:
        e39_valid[-1] = 0
        e39_mean[-1] = 123.0

    contact_mask = time_mask.copy()
    grasp_mask = time_mask & gripper_mask
    native_action_mask = np.zeros((count, 32, 8), dtype=np.uint8)
    native_action_mask[..., :7] = time_mask[..., None]
    physical_anchor_mask = np.ones((count, 4, 8), dtype=np.uint8)
    physical_anchor_mask[..., 6] = gripper_mask[:, [7, 15, 23, 31]]

    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version="maniskill-z-action-sidecar-v1", complete=True)
        handle.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string_dtype)
        handle.create_dataset("split", data=np.asarray([split] * count, dtype=object), dtype=string_dtype)
        handle.create_dataset(
            "embodiment_id",
            data=np.asarray(["panda"] * count, dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset("waypoints", data=waypoints)
        handle.create_dataset("time_mask", data=time_mask)
        handle.create_dataset("pose_mask", data=pose_mask)
        handle.create_dataset("gripper_mask", data=gripper_mask)
        handle.create_dataset("contact", data=np.zeros((count, 32), dtype=np.float32))
        handle.create_dataset("contact_mask", data=contact_mask)
        handle.create_dataset("grasp", data=np.zeros((count, 32), dtype=np.float32))
        handle.create_dataset("grasp_mask", data=grasp_mask)
        handle.create_dataset("native_actions", data=np.zeros((count, 32, 8), dtype=np.float32))
        handle.create_dataset("native_action_mask", data=native_action_mask)
        handle.create_dataset("native_action_dim", data=np.full(count, 7, dtype=np.uint8))
        handle.create_dataset("current_state", data=current_state)
        handle.create_dataset("current_state_mask", data=current_state_mask)
        handle.create_dataset("e39_mean", data=e39_mean)
        handle.create_dataset("e39_valid", data=e39_valid)
        handle.create_dataset("physical_anchors", data=np.zeros((count, 4, 8), dtype=np.float32))
        handle.create_dataset("physical_anchor_mask", data=physical_anchor_mask)
        handle.create_dataset("window_sha256", data=np.asarray(checksums, dtype=object), dtype=string_dtype)

    return [
        {
            "sample_id": sample_id,
            "group_id": f"group-{split}-{index}",
            "split": split,
            "task_id": f"task-{index % 2}",
            "source_id": shard_name,
            "controller_id": "pd_joint_delta_pos",
            "episode_id": index,
            "start_step": 10 * index,
            "end_step": 10 * index + 32,
            "shard_path": str(relative_path),
            "shard_row": index,
            "window_sha256": checksums[index],
            "e39_valid": bool(e39_valid[index]),
        }
        for index, sample_id in enumerate(sample_ids)
    ]


@pytest.fixture()
def synthetic_sidecar(tmp_path: Path) -> Path:
    root = tmp_path / "sidecar"
    root.mkdir()
    (root / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": "maniskill-z-action-sidecar-v1",
                "control_hz": 20.0,
                "waypoint": {"batch_shape": ["B", 32, 7]},
            }
        ),
        encoding="utf-8",
    )
    _write_normalization(root)
    rows_by_split = {
        "train": _write_shard(root, "train", "a", 3)
        + _write_shard(root, "train", "b", 3, invalid_e39_last=True),
        "validation": _write_shard(root, "validation", "validation", 3),
        "test": _write_shard(root, "test", "test", 2),
    }
    manifests = root / "manifests"
    manifests.mkdir()
    for split, rows in rows_by_split.items():
        pq.write_table(pa.Table.from_pylist(rows), manifests / f"{split}.parquet")
    pq.write_table(
        pa.Table.from_pylist([row for rows in rows_by_split.values() for row in rows]),
        manifests / "all.parquet",
    )
    return root


def test_dataset_applies_train_stats_and_zeroes_every_invalid_element(synthetic_sidecar: Path) -> None:
    dataset = ActionSidecarDataset(synthetic_sidecar, "train", max_open_shards=1)
    assert dataset.open_shard_count == 0
    sample = dataset[0]

    assert sample["trajectory_raw"].shape == (32, 7)
    assert sample["trajectory"].shape == (32, 7)
    assert sample["trajectory_mask"].shape == (32, 7)
    torch.testing.assert_close(sample["trajectory"][0], torch.ones(7))
    torch.testing.assert_close(sample["trajectory"][-1], torch.zeros(7))
    torch.testing.assert_close(sample["trajectory"][-2, :6], torch.zeros(6))
    torch.testing.assert_close(sample["trajectory"][-2, 6:], torch.ones(1))
    assert torch.equal(sample["trajectory_raw"][-1], torch.full((7,), 999.0))
    torch.testing.assert_close(sample["current_state"][:7], torch.ones(7))
    torch.testing.assert_close(sample["current_state"][7:], torch.zeros(7))
    assert torch.equal(sample["current_state_raw"][7:], torch.full((7,), 999.0))
    assert sample["e39_mean"].shape == (32,)
    assert bool(sample["e39_valid"])
    assert sample["metadata"]["sample_id"] == dataset.sample_ids[0]
    assert sample["metadata"]["split"] == "train"
    assert sample["metadata"]["embodiment_id"] == "panda"
    assert dataset.open_shard_count == 1
    assert dataset.handle_owner_pid == os.getpid()
    for name in (
        "time_mask",
        "pose_mask",
        "gripper_mask",
        "contact_mask",
        "grasp_mask",
        "native_action_mask",
        "current_state_mask",
        "physical_anchor_mask",
    ):
        assert sample[name].dtype == torch.bool


def test_dataset_accepts_dense_v2_schema_with_matching_shards(synthetic_sidecar: Path) -> None:
    schema_path = synthetic_sidecar / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["schema_version"] = "maniskill-z-action-sidecar-v2"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    for shard in (synthetic_sidecar / "shards").rglob("*.h5"):
        with h5py.File(shard, "r+") as handle:
            handle.attrs["schema_version"] = "maniskill-z-action-sidecar-v2"
    dataset = ActionSidecarDataset(synthetic_sidecar, "train")
    assert dataset.schema_version == "maniskill-z-action-sidecar-v2"
    assert dataset[0]["trajectory"].shape == (32, 7)


def test_no_gripper_and_invalid_e39_are_masked_but_a1_can_require_coverage(synthetic_sidecar: Path) -> None:
    dataset = ActionSidecarDataset(synthetic_sidecar, "train")
    no_gripper = dataset[1]
    assert not torch.any(no_gripper["gripper_mask"])
    assert torch.equal(no_gripper["trajectory"][:, 6], torch.zeros(32))

    invalid = dataset[-1]
    assert not bool(invalid["e39_valid"])
    assert torch.equal(invalid["e39_mean"], torch.zeros(32))
    with pytest.raises(ValueError, match="A1 requires"):
        ActionSidecarDataset(synthetic_sidecar, "train", require_e39=True)[-1]


def _fork_dataset_read(dataset: ActionSidecarDataset, connection: Any) -> None:
    sample = dataset[1]
    connection.send(
        (
            os.getpid(),
            dataset.handle_owner_pid,
            dataset.open_shard_count,
            sample["metadata"]["sample_id"],
        )
    )
    connection.close()


def test_hdf5_handles_are_lazy_and_reopened_after_fork(synthetic_sidecar: Path) -> None:
    dataset = ActionSidecarDataset(synthetic_sidecar, "train")
    dataset[0]
    parent_pid = os.getpid()
    assert dataset.handle_owner_pid == parent_pid

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_fork_dataset_read, args=(dataset, child_connection))
    process.start()
    child_pid, owner_pid, open_count, sample_id = parent_connection.recv()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert child_pid != parent_pid
    assert owner_pid == child_pid
    assert open_count == 1
    assert sample_id == dataset.sample_ids[1]
    assert dataset.handle_owner_pid == parent_pid


def test_manifest_hdf5_identity_is_checked_at_referenced_shard_row(synthetic_sidecar: Path) -> None:
    shard_path = synthetic_sidecar / "shards" / "train" / "a.h5"
    with h5py.File(shard_path, "r+") as handle:
        handle["sample_id"][0] = "wrong-id"
    dataset = ActionSidecarDataset(synthetic_sidecar, "train")
    with pytest.raises(ValueError, match="sample mismatch"):
        dataset[0]


def test_normalization_rejects_non_train_statistics(synthetic_sidecar: Path) -> None:
    stats_path = synthetic_sidecar / "stats" / "normalization.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["fit_split"] = "validation"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="fit on train"):
        ActionSidecarDataset(synthetic_sidecar, "validation")


def test_datamodule_subsets_each_official_split_without_reassignment(synthetic_sidecar: Path) -> None:
    data = ActionSidecarDataModule(
        synthetic_sidecar,
        batch_size=2,
        num_workers=0,
        train_subset_size=3,
        val_subset_size=2,
        test_subset_size=1,
        subset_seed=17,
        drop_last=False,
    )
    data.setup(None)
    assert data.train_dataset is not None
    assert data.val_dataset is not None
    assert data.test_dataset is not None
    assert len(data.train_dataset) == 3
    assert len(data.val_dataset) == 2
    assert len(data.test_dataset) == 1
    assert all(sample_id.startswith("train-") for sample_id in data.train_dataset.sample_ids)
    assert all(sample_id.startswith("validation-") for sample_id in data.val_dataset.sample_ids)
    assert all(sample_id.startswith("test-") for sample_id in data.test_dataset.sample_ids)

    second = ActionSidecarDataModule(
        synthetic_sidecar,
        batch_size=2,
        num_workers=0,
        train_subset_size=3,
        subset_seed=17,
    )
    second.setup("fit")
    assert second.train_dataset is not None
    assert second.train_dataset.sample_ids == data.train_dataset.sample_ids

    validation_batch = next(iter(data.val_dataloader()))
    assert validation_batch["trajectory"].shape == (2, 32, 7)
    assert validation_batch["current_state"].shape == (2, 14)
    assert validation_batch["e39_mean"].shape == (2, 32)
    assert validation_batch["e39_valid"].dtype == torch.bool
    assert all(item["split"] == "validation" for item in validation_batch["metadata"])
    assert data.normalization_metadata["fit_split"] == "train"


def test_datamodule_uses_trainer_rank_to_partition_global_batches(
    synthetic_sidecar: Path,
) -> None:
    per_rank_indices: list[list[int]] = []
    for rank in range(2):
        data = ActionSidecarDataModule(
            synthetic_sidecar,
            batch_size=4,
            num_workers=0,
            drop_last=False,
            seed=17,
        )
        data.setup("fit")
        data.trainer = SimpleNamespace(global_rank=rank, world_size=2)
        loader = data.train_dataloader()
        assert data.train_batch_sampler is not None
        assert data.train_batch_sampler.batch_size == 4
        assert data.train_batch_sampler.local_batch_size == 2
        assert data.train_batch_sampler.rank == rank
        assert data.train_batch_sampler.world_size == 2
        per_rank_indices.append(
            [
                int(index)
                for batch in loader
                for index in batch["dataset_indices"].tolist()
            ]
        )

    assert set(per_rank_indices[0]).isdisjoint(per_rank_indices[1])
    assert sorted(per_rank_indices[0] + per_rank_indices[1]) == list(range(6))


def test_preload_keeps_identity_and_avoids_reopening_hdf5(
    synthetic_sidecar: Path,
) -> None:
    dataset = ActionSidecarDataset(synthetic_sidecar, "train")
    expected = dataset[0]
    dataset.preload()

    assert dataset.open_shard_count == 0
    assert dataset.handle_owner_pid is None
    cached = dataset[0]
    assert cached["metadata"]["sample_id"] == expected["metadata"]["sample_id"]
    torch.testing.assert_close(cached["trajectory"], expected["trajectory"])
    assert dataset.open_shard_count == 0


def test_lightning_epoch_hook_advances_shard_sampler(synthetic_sidecar: Path) -> None:
    data = ActionSidecarDataModule(
        synthetic_sidecar,
        batch_size=2,
        num_workers=0,
        drop_last=True,
    )
    data.setup("fit")
    data.train_dataloader()
    assert data.train_batch_sampler is not None
    assert data.train_batch_sampler.epoch == 0

    module = ActionAELightningModule(
        synthetic_sidecar / "stats" / "normalization.json",
        hidden_dim=16,
        condition_dim=12,
        dilations=(1, 2),
    )
    module._trainer = SimpleNamespace(datamodule=data, current_epoch=7)
    module.on_train_epoch_start()

    assert data.train_batch_sampler.epoch == 7
