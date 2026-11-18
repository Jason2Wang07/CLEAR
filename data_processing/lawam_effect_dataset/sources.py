"""Trajectory source discovery, metadata parsing, and integrity scanning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .commit_routing import replay_profile


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    task_id: str
    source_type: str
    controller_id: str
    h5_path: str
    json_path: str
    source_commit: str | None
    control_hz: float | None
    control_hz_source: str | None
    episode_count: int
    recommended_v1: bool


class SourceAdapter:
    def __init__(self, record: SourceRecord):
        self.record = record
        self.metadata = json.loads(Path(record.json_path).read_text())
        self.episodes = {int(e["episode_id"]): e for e in self.metadata.get("episodes", [])}

    def episode_ids(self, handle: h5py.File) -> list[int]:
        return sorted(int(k[5:]) for k in handle if k.startswith("traj_") and k[5:].isdigit())

    def episode_seed(self, episode_id: int) -> int:
        return int(self.episodes[episode_id]["episode_seed"])

    def state_paths(self, group: h5py.Group) -> tuple[list[str], list[str]]:
        actors = list(group.get("env_states/actors", {}).keys())
        articulations = list(group.get("env_states/articulations", {}).keys())
        return actors, articulations


def _control_hz(metadata: dict, overrides: dict[str, float], source_id: str) -> tuple[float | None, str | None]:
    if source_id in overrides:
        return float(overrides[source_id]), "override"
    kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    for key in ("control_freq", "control_hz"):
        if key in kwargs:
            return float(kwargs[key]), f"env_kwargs.{key}"
    sim_freq = kwargs.get("sim_freq")
    decimation = kwargs.get("control_decimation")
    if sim_freq is not None and decimation is not None:
        return float(sim_freq) / float(decimation), "env_kwargs.sim_freq/control_decimation"
    return None, None


def discover_sources(root: Path, overrides: dict[str, float] | None = None) -> list[SourceRecord]:
    root, overrides = Path(root), overrides or {}
    records = []
    excluded = {"AnymalC-Reach-v1", "DrawTriangle-v1", "TwoRobotPickCube-v1", "TwoRobotStackCube-v1"}
    for h5_path in sorted(root.rglob("*.h5")):
        json_path = h5_path.with_suffix(".json")
        if not json_path.exists():
            continue
        rel = h5_path.relative_to(root)
        task_id, source_type = rel.parts[0], rel.parts[1]
        source_id = str(rel.with_suffix("")).replace("/", "__")
        metadata = json.loads(json_path.read_text())
        controller = metadata.get("env_info", {}).get("env_kwargs", {}).get("control_mode")
        if not controller and metadata.get("episodes"):
            controller = metadata["episodes"][0].get("control_mode")
        source_commit = metadata.get("commit_info", {}).get("commit_id")
        hz, hz_source = _control_hz(metadata, overrides, source_id)
        if hz is None and source_commit:
            try:
                hz = replay_profile(source_commit).control_hz
                hz_source = "verified_commit_sim_config"
            except ValueError:
                pass
        recommended = source_commit is not None and task_id not in excluded and not (task_id == "PickCube-v1" and source_type == "teleop")
        records.append(SourceRecord(source_id, task_id, source_type, str(controller or "unknown"), str(h5_path), str(json_path), source_commit, hz, hz_source, len(metadata.get("episodes", [])), recommended))
    return records


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_datasets(group: h5py.Group, prefix: str = "") -> Iterator[tuple[str, h5py.Dataset]]:
    for name, value in group.items():
        path = f"{prefix}/{name}"
        if isinstance(value, h5py.Dataset):
            yield path, value
        else:
            yield from iter_datasets(value, path)


def audit_source(record: SourceRecord, *, checksum: bool = False) -> dict:
    issues, datasets_read = [], 0
    adapter = SourceAdapter(record)
    try:
        with h5py.File(record.h5_path, "r") as handle:
            h5_ids = adapter.episode_ids(handle)
            json_ids = sorted(adapter.episodes)
            if h5_ids != json_ids:
                issues.append("json_hdf5_episode_mismatch")
            for episode_id in h5_ids:
                group = handle[f"traj_{episode_id}"]
                if "actions" not in group:
                    issues.append(f"traj_{episode_id}:missing_actions")
                    continue
                n = len(group["actions"])
                for path, dataset in iter_datasets(group):
                    # Slice every first-axis chunk, forcing decompression of all chunks.
                    chunk = dataset.chunks[0] if dataset.chunks else max(1, min(len(dataset), 4096)) if dataset.ndim else 1
                    if dataset.ndim:
                        for start in range(0, len(dataset), chunk):
                            value = dataset[start : start + chunk]
                            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                                issues.append(f"traj_{episode_id}:{path}:nonfinite")
                    else:
                        dataset[()]
                    datasets_read += 1
                env = group.get("env_states")
                if env:
                    for path, dataset in iter_datasets(env):
                        if dataset.ndim and len(dataset) != n + 1:
                            issues.append(f"traj_{episode_id}:{path}:state_action_length")
    except Exception as exc:
        issues.append(f"read_error:{type(exc).__name__}:{exc}")
    try:
        profile = replay_profile(record.source_commit)
        replay_commit_valid = True
    except ValueError as exc:
        profile, replay_commit_valid = None, False
        issues.append(f"replay_quarantined:{exc}")
    effective_hz = record.control_hz if record.control_hz is not None else (profile.control_hz if profile else None)
    result = {**asdict(record), "control_hz": effective_hz, "control_hz_source": record.control_hz_source or ("verified_commit_sim_config" if profile else None), "replay_commit_valid": replay_commit_valid, "valid": not issues and effective_hz is not None, "issues": issues + ([] if effective_hz is not None else ["unresolved_control_hz"]), "datasets_read": datasets_read}
    if checksum:
        result["h5_sha256"] = sha256_file(Path(record.h5_path))
        result["json_sha256"] = sha256_file(Path(record.json_path))
    return result
