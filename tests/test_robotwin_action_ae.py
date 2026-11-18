from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from latent_action_model.action_ae.model import ActionAEConfig, ActionAutoencoder
from latent_action_model.robotwin_action_ae.data import RoboTwinActionDataset
from latent_action_model.robotwin_action_ae.evaluate import (
    _task_balanced_delta,
    deterministic_derangement,
    grouped_derangement,
    per_sample_metrics,
)
from latent_action_model.robotwin_action_ae.losses import dual_arm_trajectory_reconstruction_loss


def _stats() -> dict:
    def group(width: int) -> dict:
        return {
            "count": [100] * width,
            "median": [0.0] * width,
            "scale": [2.0] * width,
            "scale_mad": [1.0] * width,
            "rms_about_median": [2.0] * width,
        }

    return {
        "schema_version": "robotwin-z-action-normalization-v1",
        "fit_split": "train",
        "estimator": "median_and_max_1.4826_mad_rms_about_median",
        "sample_count": 2,
        "left_position": group(3),
        "left_rotation_rotvec": group(3),
        "left_gripper": group(1),
        "right_position": group(3),
        "right_rotation_rotvec": group(3),
        "right_gripper": group(1),
        "current_state": group(14),
    }


def _synthetic_release(root: Path, *, e39_valid: bool) -> Path:
    root.mkdir()
    (root / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": "robotwin-z-action-sidecar-v1",
                "waypoint": {"batch_shape": ["B", 48, 14]},
            }
        ),
        encoding="utf-8",
    )
    (root / "stats").mkdir()
    (root / "stats/normalization.json").write_text(json.dumps(_stats()), encoding="utf-8")
    rows = []
    string = h5py.string_dtype("utf-8")
    for split in ("train", "validation", "test"):
        relative = Path("shards") / split / "part-00000.h5"
        path = root / relative
        path.parent.mkdir(parents=True)
        sample_ids = [f"{split}-{index}" for index in range(2)]
        with h5py.File(path, "w") as handle:
            handle.attrs.update(schema_version="robotwin-z-action-sidecar-v1", complete=True)
            handle.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string)
            handle.create_dataset("split", data=np.asarray([split] * 2, dtype=object), dtype=string)
            handle.create_dataset("embodiment_id", data=np.asarray(["agilex"] * 2, dtype=object), dtype=string)
            handle.create_dataset("waypoints", data=np.full((2, 48, 14), 2.0, dtype=np.float32))
            handle.create_dataset("time_mask", data=np.ones((2, 48), dtype=np.uint8))
            handle.create_dataset("arm_pose_mask", data=np.ones((2, 48, 2), dtype=np.uint8))
            handle.create_dataset("arm_gripper_mask", data=np.ones((2, 48, 2), dtype=np.uint8))
            handle.create_dataset("native_actions", data=np.zeros((2, 48, 16), dtype=np.float32))
            handle.create_dataset("native_action_mask", data=np.ones((2, 48, 16), dtype=np.uint8))
            handle.create_dataset("current_state", data=np.full((2, 14), 2.0, dtype=np.float32))
            handle.create_dataset("current_state_mask", data=np.ones((2, 14), dtype=np.uint8))
            handle.create_dataset("e39_mean", data=np.ones((2, 32), dtype=np.float32))
            handle.create_dataset("e39_valid", data=np.full(2, int(e39_valid), dtype=np.uint8))
            handle.create_dataset("physical_anchors", data=np.zeros((2, 4, 14), dtype=np.float32))
            handle.create_dataset("physical_anchor_mask", data=np.ones((2, 4, 14), dtype=np.uint8))
            handle.create_dataset("window_sha256", data=np.asarray(sample_ids, dtype=object), dtype=string)
        for index, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": f"group-{split}-{index}",
                    "split": split,
                    "task_id": "robotwin/test",
                    "source_id": "robotwin_merged",
                    "controller_id": "robotwin_dual_arm_absolute_eef_wxyz",
                    "episode_id": 10 * index + {"train": 0, "validation": 1, "test": 2}[split],
                    "start_step": 0,
                    "end_step": 48,
                    "shard_path": relative.as_posix(),
                    "shard_row": index,
                    "window_sha256": sample_id,
                    "e39_valid": e39_valid,
                }
            )
    manifests = root / "manifests"
    manifests.mkdir()
    frame = pa.Table.from_pylist(rows)
    pq.write_table(frame, manifests / "all.parquet")
    for split in ("train", "validation", "test"):
        pq.write_table(pa.Table.from_pylist([row for row in rows if row["split"] == split]), manifests / f"{split}.parquet")
    return root


def test_robotwin_model_supports_48x14_and_keeps_a0_a1_parameter_counts_equal() -> None:
    counts = []
    for variant in ("a0", "a1"):
        config = ActionAEConfig(
            state_dim=14,
            variant=variant,
            horizon=48,
            trajectory_dim=14,
            hidden_dim=16,
            condition_dim=16,
            dilations=(1, 2),
        )
        model = ActionAutoencoder(config)
        effect = torch.randn(2, 32, requires_grad=True) if variant == "a1" else None
        output = model(
            torch.randn(2, 48, 14),
            torch.randn(2, 14),
            effect,
            time_mask=torch.ones(2, 48, dtype=torch.bool),
        )
        assert output.reconstruction.shape == (2, 48, 14)
        assert output.z_action.shape == (2, 1, 32)
        output.reconstruction.sum().backward()
        if effect is not None:
            assert effect.grad is None
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts[0] == counts[1]


def test_dual_arm_loss_uses_both_arms_and_modality_masks() -> None:
    target = torch.zeros(1, 48, 14)
    prediction = target.clone()
    prediction[..., 0:3] = 1.0
    prediction[..., 7:10] = 2.0
    result = dual_arm_trajectory_reconstruction_loss(
        prediction,
        target,
        time_mask=torch.ones(1, 48, dtype=torch.bool),
        arm_pose_mask=torch.ones(1, 48, 2, dtype=torch.bool),
        arm_gripper_mask=torch.zeros(1, 48, 2, dtype=torch.bool),
    )
    assert result.position > 0
    assert result.position_valid_elements == 48 * 2 * 3
    assert result.gripper == 0
    assert result.gripper_valid_elements == 0


def test_dataset_normalizes_48x14_and_enforces_a1_coverage(tmp_path: Path) -> None:
    a0 = _synthetic_release(tmp_path / "a0", e39_valid=False)
    dataset = RoboTwinActionDataset(a0, "train")
    sample = dataset[0]
    assert sample["trajectory"].shape == (48, 14)
    assert torch.equal(sample["trajectory"], torch.ones(48, 14))
    assert torch.equal(sample["current_state"], torch.ones(14))
    assert not sample["e39_valid"]
    dataset.close()

    a1 = _synthetic_release(tmp_path / "a1", e39_valid=True)
    a1_sample = RoboTwinActionDataset(a1, "train", require_e39=True)[0]
    assert a1_sample["e39_valid"]
    assert a1_sample["e39_mean"].shape == (32,)


def test_robotwin_evaluation_shuffles_are_deterministic_and_group_local() -> None:
    first, offset = deterministic_derangement(8, 17)
    second, second_offset = deterministic_derangement(8, 17)
    assert offset == second_offset
    assert torch.equal(first, second)
    assert not torch.any(first == torch.arange(8))

    groups = ("a", "a", "a", "b", "b", "b", "b", "b")
    permutation, metadata = grouped_derangement(groups, 19)
    assert metadata["group_count"] == 2
    assert not torch.any(permutation == torch.arange(len(groups)))
    assert all(groups[index] == groups[donor] for index, donor in enumerate(permutation.tolist()))


def test_robotwin_per_sample_metrics_use_physical_dual_arm_units() -> None:
    class IdentityModule:
        beta = 0.1
        position_weight = 1.0
        rotation_weight = 1.0
        gripper_weight = 0.5

        @staticmethod
        def _denormalize(value: torch.Tensor) -> torch.Tensor:
            return value

    target = torch.zeros(1, 4, 14)
    prediction = target.clone()
    prediction[..., 0] = 0.01
    prediction[..., 7] = 0.02
    metrics = per_sample_metrics(
        IdentityModule(),  # type: ignore[arg-type]
        prediction,
        target,
        target,
        time_mask=torch.ones(1, 4, dtype=torch.bool),
        arm_pose_mask=torch.ones(1, 4, 2, dtype=torch.bool),
        arm_gripper_mask=torch.ones(1, 4, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(metrics["position_l2_cm"], torch.tensor([1.5]))
    torch.testing.assert_close(metrics["endpoint_position_l2_cm"], torch.tensor([1.5]))
    torch.testing.assert_close(metrics["rotation_geodesic_deg"], torch.zeros(1))
    torch.testing.assert_close(metrics["gripper_mae"], torch.zeros(1))


def test_robotwin_task_balanced_delta_reports_positive_candidate_improvement() -> None:
    report = _task_balanced_delta(
        torch.tensor([3.0, 5.0, 10.0, 14.0]),
        torch.tensor([2.0, 4.0, 8.0, 12.0]),
        ("a", "a", "b", "b"),
        seed=23,
        replicates=1000,
    )
    assert report["task_balanced_mean"] == 1.5
    assert report["nonnegative_task_count"] == 2
    assert report["paired_bootstrap_95ci"][0] >= 1.0
