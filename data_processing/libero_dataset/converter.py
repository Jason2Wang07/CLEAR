"""Streaming LIBERO conversion into aligned Stage-1 and ActionAE releases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from lawam_effect_dataset.geometry import delta_state14, relative_pose, state14
from lawam_effect_dataset.shards import write_shard_atomic as write_effect_shard
from lawam_effect_dataset.statistics import robust_normalization
from lawam_effect_dataset.windows import group_id, stable_id
from z_action_sidecar.geometry import canonical_waypoints
from z_action_sidecar.io import write_shard_atomic as write_action_shard
from z_action_sidecar.schema import (
    E39_DIM,
    HORIZON,
    MAX_NATIVE_ACTION_DIM,
    anchor_steps_for_horizon,
    fields_for_horizon,
)
from z_action_sidecar.statistics import sidecar_normalization

from .geometry import gripper_width, poses_from_rotvec, resize_agentview
from .schema import (
    EFFECT_SCHEMA_VERSION,
    NORMALIZATION_SCHEMA_VERSION,
    SIDECAR_SCHEMA_VERSION,
    effect_schema_document,
    sidecar_schema_document,
    versions_for_horizon,
)
from .source import audit_source, demo_sort_key, discover_task_files, iter_selected_demos, task_metadata


CONTROL_HZ = 20.0
CONTROLLER_ID = "libero_osc_pose_delta_normalized"


def stratified_episode_splits(
    task_id: str,
    episode_ids: Sequence[int],
    seed: int,
) -> dict[int, str]:
    """Assign deterministic, task-local, episode-disjoint 80/10/10 splits."""

    unique_ids = [int(value) for value in episode_ids]
    if len(set(unique_ids)) != len(unique_ids):
        raise ValueError(f"task {task_id} contains duplicate episode IDs")

    def key(episode_id: int) -> bytes:
        payload = f"{seed}\0{task_id}\0{episode_id}".encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()

    ordered = sorted(unique_ids, key=lambda episode_id: (key(episode_id), episode_id))
    count = len(ordered)
    test_count = max(1, round(0.10 * count))
    validation_count = max(1, round(0.10 * count))
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError(f"task {task_id} has too few episodes for three splits: {count}")
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


def _window_digest(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    identity = {
        name: row[name]
        for name in (
            "sample_id",
            "source_id",
            "source_commit",
            "source_sha256",
            "episode_id",
            "start_step",
            "end_step",
            "controller_id",
        )
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in (
        "waypoints",
        "time_mask",
        "pose_mask",
        "gripper_mask",
        "contact",
        "contact_peak_force",
        "contact_impulse",
        "contact_mask",
        "grasp",
        "grasp_mask",
        "native_actions",
        "native_action_mask",
        "current_state",
        "current_state_mask",
        "e39_mean",
    ):
        value = np.ascontiguousarray(row[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _identity_pose() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _effect_row(
    *,
    sample_id: str,
    group: str,
    split: str,
    task: Mapping[str, Any],
    episode_id: int,
    start: int,
    end: int,
    poses: np.ndarray,
    widths: np.ndarray,
    images: h5py.Dataset,
    source_sha256: str,
    horizon_actions: int,
) -> dict[str, Any]:
    start_pose = poses[start]
    end_pose = poses[end]
    start_width = float(widths[start])
    end_width = float(widths[end])
    start_state, state_mask = state14(start_pose, start_width)
    end_state, _ = state14(end_pose, end_width)
    state_delta = delta_state14(start_pose, end_pose, start_width, end_width)
    effect_raw = np.zeros(12, dtype=np.float32)
    effect_raw[:6] = relative_pose(start_pose, end_pose)
    effect_raw[6] = end_width - start_width
    effect_mask = np.zeros(12, dtype=np.uint8)
    effect_mask[:7] = 1
    identity = _identity_pose()
    zeros_pose = np.zeros((3, 7), dtype=np.float32)
    return {
        "sample_id": sample_id,
        "task_id": str(task["task_id"]),
        "effect_family": str(task["task_id"]),
        "source_id": str(task["source_id"]),
        "source_commit": "libero-v1",
        "controller_id": CONTROLLER_ID,
        "episode_id": episode_id,
        "episode_seed": episode_id,
        "start_step": start,
        "end_step": end,
        "control_hz": CONTROL_HZ,
        "tau": int(horizon_actions) / CONTROL_HZ,
        "split": split,
        "rgb_start": resize_agentview(images[start]),
        "rgb_end": resize_agentview(images[end]),
        "target_seg_start": np.zeros((256, 256), dtype=np.uint32),
        "target_seg_end": np.zeros((256, 256), dtype=np.uint32),
        "object_pose_start": identity.copy(),
        "object_pose_end": identity.copy(),
        "reference_pose_start": identity.copy(),
        "reference_pose_end": identity.copy(),
        "active_object_id": "",
        "aux_object_poses_start": zeros_pose.copy(),
        "aux_object_poses_end": zeros_pose.copy(),
        "aux_object_mask": np.zeros(3, dtype=np.uint8),
        "effect_raw": effect_raw,
        "effect_mask": effect_mask,
        "effect_valid": 1,
        "effect_direction": "mixed",
        "effect_magnitude": float(np.linalg.norm(effect_raw[:3])),
        "eef_pose_start": start_pose,
        "eef_pose_end": end_pose,
        "gripper_start": start_width,
        "gripper_end": end_width,
        "state14_start": start_state,
        "state14_end": end_state,
        "delta_state14": state_delta,
        "state_mask": state_mask,
        "embodiment_id": "panda",
        "contact_summary": np.zeros(4, dtype=np.float32),
        "grasp_summary": np.zeros(4, dtype=np.float32),
        "phase": "demonstration",
        "outcome": "success",
        "quality": 1.0,
        "replay_valid": 0,
        "render_valid": 1,
        "source_sha256": source_sha256,
        "group_id": group,
    }


def _action_row(
    effect: Mapping[str, Any],
    *,
    group: str,
    poses: np.ndarray,
    widths: np.ndarray,
    actions: np.ndarray,
    horizon_actions: int,
    anchor_steps: tuple[int, ...],
) -> dict[str, Any]:
    start = int(effect["start_step"])
    end = int(effect["end_step"])
    waypoints = canonical_waypoints(poses[start], poses[start + 1 : end + 1], widths[start + 1 : end + 1])
    horizon = int(horizon_actions)
    if waypoints.shape != (horizon, 7):
        raise ValueError(f"expected {horizon} LIBERO waypoints, got {waypoints.shape}")
    native_actions = np.zeros((horizon, MAX_NATIVE_ACTION_DIM), dtype=np.float32)
    native_actions[:, :7] = actions[start:end]
    native_mask = np.zeros_like(native_actions, dtype=np.uint8)
    native_mask[:, :7] = 1
    active = np.ones(horizon, dtype=np.uint8)
    inactive = np.zeros(horizon, dtype=np.uint8)
    anchor_indices = np.asarray(anchor_steps, dtype=np.int64) - 1
    anchors = np.zeros((len(anchor_steps), 8), dtype=np.float32)
    anchors[:, :7] = waypoints[anchor_indices]
    anchor_mask = np.zeros_like(anchors, dtype=np.uint8)
    anchor_mask[:, :7] = 1
    row: dict[str, Any] = {
        "sample_id": str(effect["sample_id"]),
        "group_id": group,
        "split": str(effect["split"]),
        "task_id": str(effect["task_id"]),
        "source_id": str(effect["source_id"]),
        "source_commit": str(effect["source_commit"]),
        "source_sha256": str(effect["source_sha256"]),
        "controller_id": str(effect["controller_id"]),
        "embodiment_id": "panda",
        "episode_id": int(effect["episode_id"]),
        "episode_seed": int(effect["episode_seed"]),
        "start_step": start,
        "end_step": end,
        "control_hz": CONTROL_HZ,
        "eef_pose_start": np.asarray(effect["eef_pose_start"], dtype=np.float32),
        "eef_pose_end": np.asarray(effect["eef_pose_end"], dtype=np.float32),
        "waypoints": waypoints,
        "time_mask": active.copy(),
        "pose_mask": active.copy(),
        "gripper_mask": active.copy(),
        "contact": np.zeros(horizon, dtype=np.float32),
        "contact_peak_force": np.zeros(horizon, dtype=np.float32),
        "contact_impulse": np.zeros((horizon, 6), dtype=np.float32),
        "contact_mask": inactive.copy(),
        "grasp": np.zeros(horizon, dtype=np.float32),
        "grasp_mask": inactive.copy(),
        "native_actions": native_actions,
        "native_action_mask": native_mask,
        "native_action_dim": 7,
        "current_state": np.asarray(effect["state14_start"], dtype=np.float32),
        "current_state_mask": np.asarray(effect["state_mask"], dtype=np.uint8),
        "e39_mean": np.zeros(E39_DIM, dtype=np.float32),
        "e39_valid": 0,
        "physical_anchors": anchors,
        "physical_anchor_mask": anchor_mask,
        "replay_object_error": np.zeros(horizon, dtype=np.float32),
        "replay_robot_error": np.zeros(horizon, dtype=np.float32),
    }
    row["window_sha256"] = _window_digest(row)
    return row


class _PairedShardWriter:
    def __init__(
        self,
        root: Path,
        shard_rows: int,
        *,
        action_fields: Mapping[str, tuple[str, tuple[int, ...]]],
        effect_schema_version: str,
        sidecar_schema_version: str,
    ) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.root = root
        self.shard_rows = int(shard_rows)
        self.action_fields = action_fields
        self.effect_schema_version = effect_schema_version
        self.sidecar_schema_version = sidecar_schema_version
        self.buffers: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
            split: [] for split in ("train", "validation", "test")
        }
        self.indices = {split: 0 for split in self.buffers}
        self.effect_manifest: list[dict[str, Any]] = []
        self.action_manifest: list[dict[str, Any]] = []
        self.effect_stats_rows: list[dict[str, Any]] = []
        self.action_stats_rows: list[dict[str, Any]] = []

    def add(self, effect: dict[str, Any], action: dict[str, Any]) -> None:
        split = str(effect["split"])
        self.buffers[split].append((effect, action))
        self.effect_stats_rows.append(
            {
                "split": split,
                "effect_valid": effect["effect_valid"],
                "delta_state14": effect["delta_state14"],
                "state_mask": effect["state_mask"],
            }
        )
        self.action_stats_rows.append(
            {
                "split": split,
                "waypoints": action["waypoints"],
                "time_mask": action["time_mask"],
                "pose_mask": action["pose_mask"],
                "gripper_mask": action["gripper_mask"],
                "current_state": action["current_state"],
                "current_state_mask": action["current_state_mask"],
            }
        )
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    @staticmethod
    def _identity_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
        names = (
            "sample_id",
            "group_id",
            "split",
            "task_id",
            "source_id",
            "source_commit",
            "controller_id",
            "episode_id",
            "episode_seed",
            "start_step",
            "end_step",
            "control_hz",
        )
        return {name: row[name] for name in names}

    def flush(self, split: str) -> None:
        paired = self.buffers[split]
        if not paired:
            return
        index = self.indices[split]
        filename = f"part-{index:05d}.h5"
        effect_relative = Path("shards") / split / filename
        action_relative = Path("shards") / split / filename
        effects = [pair[0] for pair in paired]
        actions = [pair[1] for pair in paired]
        write_effect_shard(
            self.root / "effect" / effect_relative,
            effects,
            schema_version=self.effect_schema_version,
        )
        write_action_shard(
            self.root / "action" / action_relative,
            actions,
            schema_version=self.sidecar_schema_version,
            fields=self.action_fields,
        )
        for shard_row, (effect, action) in enumerate(paired):
            effect_manifest = self._identity_manifest(effect)
            effect_manifest.update(
                {
                    "effect_family": effect["effect_family"],
                    "effect_direction": effect["effect_direction"],
                    "effect_magnitude": effect["effect_magnitude"],
                    "view": "agentview",
                    "tau": effect["tau"],
                    "language_instruction": effect["language_instruction"],
                    "suite": effect["suite"],
                    "shard_path": effect_relative.as_posix(),
                    "shard_row": shard_row,
                }
            )
            self.effect_manifest.append(effect_manifest)
            action_manifest = self._identity_manifest(action)
            action_manifest.update(
                {
                    "language_instruction": action["language_instruction"],
                    "suite": action["suite"],
                    "effect_shard_path": (Path("../effect") / effect_relative).as_posix(),
                    "effect_shard_row": shard_row,
                    "shard_path": action_relative.as_posix(),
                    "shard_row": shard_row,
                    "window_sha256": action["window_sha256"],
                    "e39_valid": False,
                }
            )
            self.action_manifest.append(action_manifest)
        self.buffers[split] = []
        self.indices[split] += 1

    def finish(self) -> None:
        for split in self.buffers:
            self.flush(split)


def _validate_demo(demo: h5py.Group, *, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    obs = demo["obs"]
    positions = np.asarray(obs["ee_pos"], dtype=np.float64)
    orientations = np.asarray(obs["ee_ori"], dtype=np.float64)
    stored_states = np.asarray(obs["ee_states"], dtype=np.float64)
    gripper = np.asarray(obs["gripper_states"], dtype=np.float64)
    n = len(actions)
    if actions.shape != (n, 7):
        raise ValueError(f"{label}: actions must be [N,7], got {actions.shape}")
    if positions.shape != (n, 3) or orientations.shape != (n, 3) or stored_states.shape != (n, 6):
        raise ValueError(f"{label}: malformed saved EEF state")
    if gripper.shape != (n, 2) or demo["obs/agentview_rgb"].shape != (n, 128, 128, 3):
        raise ValueError(f"{label}: malformed gripper/image state")
    if not np.isfinite(actions).all():
        raise ValueError(f"{label}: actions contain NaN/Inf")
    state_error = float(np.max(np.abs(stored_states - np.concatenate((positions, orientations), axis=1))))
    if state_error > 1e-8:
        raise ValueError(f"{label}: ee_states disagrees with ee_pos/ee_ori by {state_error}")
    return actions, poses_from_rotvec(positions, orientations), gripper_width(gripper)


def _write_manifests(root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(manifests / "all.parquet", index=False)
    for split in ("train", "validation", "test"):
        frame[frame.split == split].to_parquet(manifests / f"{split}.parquet", index=False)


def _child_coverage(coverage: Mapping[str, Any], artifact: str) -> dict[str, Any]:
    return {**coverage, "artifact": artifact}


def _child_provenance(provenance: Mapping[str, Any], artifact: str) -> dict[str, Any]:
    return {
        **provenance,
        "artifact": artifact,
        "release_root_contract": "paths in manifests are relative to this directory",
    }


def _publish_metadata(
    root: Path,
    *,
    coverage: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    for artifact in ("effect", "action"):
        child = root / artifact
        _atomic_json(child / "coverage_report.json", _child_coverage(coverage, artifact))
        _atomic_json(child / "provenance.json", _child_provenance(provenance, artifact))
        (child / "_SUCCESS").write_text(
            f"libero-actionvae-{artifact}-release-v1\n",
            encoding="utf-8",
        )
        _write_checksums(child)

    _atomic_json(root / "coverage_report.json", coverage)
    _atomic_json(root / "provenance.json", provenance)
    (root / "_SUCCESS").write_text("libero-actionvae-release-v1\n", encoding="utf-8")
    _write_checksums(root)


def build_release(
    source_root: str | Path,
    output: str | Path,
    *,
    suites: Sequence[str] | None = None,
    max_task_files: int | None = None,
    max_episodes: int | None = None,
    horizon_actions: int = HORIZON,
    window_stride: int = 8,
    shard_rows: int = 256,
    split_seed: int = 20260801,
) -> dict[str, Any]:
    horizon = int(horizon_actions)
    if horizon < 1 or window_stride < 1:
        raise ValueError("horizon_actions and window_stride must be positive")
    effect_schema_version, sidecar_schema_version, normalization_schema_version = (
        versions_for_horizon(horizon)
    )
    anchor_steps = anchor_steps_for_horizon(horizon)
    action_fields = fields_for_horizon(horizon, anchor_steps=anchor_steps)
    source = Path(source_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    files = discover_task_files(source, suites=suites, max_task_files=max_task_files)
    selected_by_file: dict[Path, list[str]] = {}
    for path, demo_name in iter_selected_demos(files, max_episodes=max_episodes):
        selected_by_file.setdefault(path, []).append(demo_name)
    if not selected_by_file:
        raise ValueError("no LIBERO episodes selected")

    audit = audit_source(
        source,
        suites=suites,
        max_task_files=max_task_files,
        max_episodes=max_episodes,
        horizon=horizon,
        window_stride=window_stride,
    )
    if not audit["valid"]:
        raise ValueError(f"LIBERO source audit failed: {audit['issues'][:8]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir()
    writer = _PairedShardWriter(
        temporary,
        shard_rows,
        action_fields=action_fields,
        effect_schema_version=effect_schema_version,
        sidecar_schema_version=sidecar_schema_version,
    )
    source_provenance: list[dict[str, Any]] = []
    try:
        split_by_task: dict[str, dict[int, str]] = {}
        for path, demo_names in selected_by_file.items():
            metadata = task_metadata(path, source)
            episode_ids = [demo_sort_key(name)[0] for name in demo_names]
            split_by_task[str(metadata["task_id"])] = stratified_episode_splits(
                str(metadata["task_id"]),
                episode_ids,
                split_seed,
            )
        for path, demo_names in selected_by_file.items():
            metadata = task_metadata(path, source)
            if not np.isclose(float(metadata["control_hz"]), CONTROL_HZ, atol=1e-6, rtol=0.0):
                raise ValueError(f"{path}: expected {CONTROL_HZ} Hz, got {metadata['control_hz']}")
            if str(metadata["image_convention"]).lower() != "opengl":
                raise ValueError(f"{path}: expected OpenGL image convention, got {metadata['image_convention']!r}")
            source_sha256 = _sha256_file(path)
            source_provenance.append(
                {
                    "relative_path": metadata["relative_path"],
                    "sha256": source_sha256,
                    "episodes": len(demo_names),
                }
            )
            with h5py.File(path, "r") as handle:
                for demo_name in sorted(demo_names, key=demo_sort_key):
                    demo = handle[f"data/{demo_name}"]
                    episode_id = demo_sort_key(demo_name)[0]
                    label = f"{metadata['relative_path']}:{demo_name}"
                    actions, poses, widths = _validate_demo(demo, label=label)
                    transition_count = len(actions) - 1
                    group = group_id(str(metadata["task_id"]), episode_id)
                    split = split_by_task[str(metadata["task_id"])][episode_id]
                    for start in range(0, transition_count - horizon + 1, window_stride):
                        end = start + horizon
                        sample_id = stable_id(
                            str(metadata["source_id"]),
                            episode_id,
                            start,
                            end,
                            version="libero-v1",
                        )
                        effect = _effect_row(
                            sample_id=sample_id,
                            group=group,
                            split=split,
                            task=metadata,
                            episode_id=episode_id,
                            start=start,
                            end=end,
                            poses=poses,
                            widths=widths,
                            images=demo["obs/agentview_rgb"],
                            source_sha256=source_sha256,
                            horizon_actions=horizon,
                        )
                        effect["language_instruction"] = metadata["language_instruction"]
                        effect["suite"] = metadata["suite"]
                        action = _action_row(
                            effect,
                            group=group,
                            poses=poses,
                            widths=widths,
                            actions=actions,
                            horizon_actions=horizon,
                            anchor_steps=anchor_steps,
                        )
                        action["language_instruction"] = metadata["language_instruction"]
                        action["suite"] = metadata["suite"]
                        writer.add(effect, action)
        writer.finish()
        if not writer.effect_manifest:
            raise ValueError(f"selected LIBERO episodes contain no complete {horizon}-step windows")
        split_counts = Counter(str(row["split"]) for row in writer.effect_manifest)
        if not split_counts.get("train"):
            raise ValueError("selected subset has no train windows; increase the subset or change split_seed")
        _write_manifests(temporary / "effect", writer.effect_manifest)
        _write_manifests(temporary / "action", writer.action_manifest)

        effect_stats = robust_normalization(
            writer.effect_stats_rows,
            "delta_state14",
            "state_mask",
        )
        action_stats = sidecar_normalization(
            writer.action_stats_rows,
            schema_version=normalization_schema_version,
        )
        _atomic_json(temporary / "effect/stats/state_normalization.json", effect_stats)
        _atomic_json(temporary / "action/stats/normalization.json", action_stats)
        _atomic_json(
            temporary / "effect/schema.json",
            effect_schema_document(horizon_actions=horizon),
        )
        _atomic_json(
            temporary / "action/schema.json",
            sidecar_schema_document(horizon_actions=horizon),
        )
        _atomic_json(temporary / "audit.json", audit)
        coverage = {
            "schema": "libero-actionvae-coverage-v1",
            "valid": True,
            "rows": len(writer.effect_manifest),
            "episodes": int(len({(row["source_id"], row["episode_id"]) for row in writer.effect_manifest})),
            "task_files": len(selected_by_file),
            "split_counts": dict(sorted(split_counts.items())),
            "window_stride": window_stride,
            "horizon_actions": horizon,
            "control_hz": CONTROL_HZ,
            "contact_valid_steps": 0,
            "grasp_valid_steps": 0,
            "e39_valid_rows": 0,
        }
        _atomic_json(temporary / "coverage_report.json", coverage)
        provenance = {
            "schema": "libero-actionvae-provenance-v1",
            "source_root": str(source),
            "source_files": source_provenance,
            "source_dataset_tag": "libero-v1",
            "transition_alignment": "obs[t] -> actions[t] -> obs[t+1]",
            "excluded_source_actions": "last action of each demo has no saved next observation",
            "image_processing": "saved display-oriented RGB, no vertical flip, bilinear 128x128 to 256x256, CHW uint8",
            "split_ownership": "episode-disjoint",
            "split_strategy": "task-stratified deterministic hash order; per task 80/10/10",
            "split_seed": split_seed,
            "effect_schema_version": effect_schema_version,
            "sidecar_schema_version": sidecar_schema_version,
            "normalization_schema_version": normalization_schema_version,
            "unavailable_supervision": ["object_pose", "target_segmentation", "contact", "grasp", "e39"],
        }
        _publish_metadata(temporary, coverage=coverage, provenance=provenance)
        os.replace(temporary, destination)
        return {**coverage, "output": str(destination)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
