"""Discovery and structural audit for LIBERO official HDF5 demonstrations."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


DEFAULT_SUITES = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")
REQUIRED_OBS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "ee_pos",
    "ee_ori",
    "ee_states",
    "gripper_states",
    "joint_states",
)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def discover_task_files(
    source_root: str | Path,
    *,
    suites: Sequence[str] | None = None,
    max_task_files: int | None = None,
) -> list[Path]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO source root does not exist: {root}")
    selected = tuple(suites or DEFAULT_SUITES)
    unknown = sorted(set(selected).difference(DEFAULT_SUITES))
    if unknown:
        raise ValueError(f"unknown LIBERO suites: {unknown}; expected a subset of {DEFAULT_SUITES}")
    files = sorted(path for suite in selected for path in (root / suite).glob("*.hdf5"))
    if max_task_files is not None:
        if max_task_files < 1:
            raise ValueError("max_task_files must be positive")
        files = files[:max_task_files]
    if not files:
        raise FileNotFoundError(f"no LIBERO HDF5 task files found under {root} for suites={selected}")
    return files


def task_metadata(path: Path, source_root: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"{path} has no /data group")
        data = handle["data"]
        env_args = json.loads(_decode(data.attrs.get("env_args", "{}")))
        problem_info = json.loads(_decode(data.attrs.get("problem_info", "{}")))
        relative = path.resolve().relative_to(source_root.resolve())
        suite = relative.parts[0]
        task_slug = path.stem.removesuffix("_demo")
        control_hz = float(env_args.get("env_kwargs", {}).get("control_freq", 0.0))
        return {
            "path": str(path.resolve()),
            "relative_path": relative.as_posix(),
            "suite": suite,
            "source_id": f"libero/{suite}/{task_slug}",
            "task_id": f"libero/{suite}/{task_slug}",
            "task_slug": task_slug,
            "language_instruction": str(problem_info.get("language_instruction", task_slug.replace("_", " "))),
            "control_hz": control_hz,
            "image_convention": _decode(data.attrs.get("macros_image_convention", "")),
            "env_name": _decode(data.attrs.get("env_name", "")),
            "env_args": env_args,
            "demo_names": sorted(data.keys(), key=demo_sort_key),
        }


def iter_selected_demos(
    files: Iterable[Path],
    *,
    max_episodes: int | None = None,
) -> Iterable[tuple[Path, str]]:
    remaining = None if max_episodes is None else int(max_episodes)
    if remaining is not None and remaining < 1:
        raise ValueError("max_episodes must be positive")
    for path in files:
        with h5py.File(path, "r") as handle:
            names = sorted(handle["data"].keys(), key=demo_sort_key)
        for name in names:
            if remaining == 0:
                return
            yield path, name
            if remaining is not None:
                remaining -= 1


def audit_source(
    source_root: str | Path,
    *,
    suites: Sequence[str] | None = None,
    max_task_files: int | None = None,
    max_episodes: int | None = None,
    horizon: int = 32,
    window_stride: int = 8,
    include_checksums: bool = False,
) -> dict[str, Any]:
    if horizon < 1 or window_stride < 1:
        raise ValueError("horizon and window_stride must be positive")
    root = Path(source_root).expanduser().resolve()
    files = discover_task_files(root, suites=suites, max_task_files=max_task_files)
    selected_by_file: dict[Path, list[str]] = {}
    for path, demo_name in iter_selected_demos(files, max_episodes=max_episodes):
        selected_by_file.setdefault(path, []).append(demo_name)

    issues: list[str] = []
    lengths: list[int] = []
    dense_windows = 0
    nonoverlap_windows = 0
    field_missing = Counter()
    controls = Counter()
    controller_types = Counter()
    conventions = Counter()
    suite_counts = Counter()
    action_abs_max = 0.0
    eef_state_error_max = 0.0
    task_reports: list[dict[str, Any]] = []
    for path, names in selected_by_file.items():
        metadata = task_metadata(path, root)
        if not np.isclose(float(metadata["control_hz"]), 20.0, atol=1e-6, rtol=0.0):
            issues.append(f"{path.name}:control_hz={metadata['control_hz']}")
        if str(metadata["image_convention"]).lower() != "opengl":
            issues.append(f"{path.name}:image_convention={metadata['image_convention']}")
        controller = metadata["env_args"].get("env_kwargs", {}).get("controller_configs", {})
        controller_type = str(controller.get("type", ""))
        controller_types[controller_type] += len(names)
        if controller_type != "OSC_POSE" or controller.get("control_delta") is not True:
            issues.append(
                f"{path.name}:controller={controller_type},control_delta={controller.get('control_delta')}"
            )
        controls[str(metadata["control_hz"])] += len(names)
        conventions[str(metadata["image_convention"])] += len(names)
        suite_counts[str(metadata["suite"])] += len(names)
        task_frames = 0
        task_windows = 0
        with h5py.File(path, "r") as handle:
            for demo_name in names:
                demo = handle[f"data/{demo_name}"]
                obs = demo.get("obs")
                required = ("actions", "states", "obs")
                for field in required:
                    if field not in demo:
                        field_missing[field] += 1
                if obs is None:
                    issues.append(f"{path.name}:{demo_name}:missing_obs")
                    continue
                for field in REQUIRED_OBS:
                    if field not in obs:
                        field_missing[f"obs/{field}"] += 1
                if "actions" not in demo or any(field not in obs for field in REQUIRED_OBS):
                    issues.append(f"{path.name}:{demo_name}:missing_required_field")
                    continue
                actions = demo["actions"]
                n = int(actions.shape[0])
                lengths.append(n)
                task_frames += n
                if actions.shape != (n, 7):
                    issues.append(f"{path.name}:{demo_name}:actions_shape={actions.shape}")
                expected_shapes = {
                    "agentview_rgb": (n, 128, 128, 3),
                    "eye_in_hand_rgb": (n, 128, 128, 3),
                    "ee_pos": (n, 3),
                    "ee_ori": (n, 3),
                    "ee_states": (n, 6),
                    "gripper_states": (n, 2),
                    "joint_states": (n, 7),
                }
                for field, expected in expected_shapes.items():
                    if tuple(obs[field].shape) != expected:
                        issues.append(f"{path.name}:{demo_name}:obs/{field}_shape={obs[field].shape}")
                numeric_fields = {
                    "actions": np.asarray(actions),
                    "ee_pos": np.asarray(obs["ee_pos"]),
                    "ee_ori": np.asarray(obs["ee_ori"]),
                    "ee_states": np.asarray(obs["ee_states"]),
                    "gripper_states": np.asarray(obs["gripper_states"]),
                }
                for field, values in numeric_fields.items():
                    if not np.isfinite(values).all():
                        issues.append(f"{path.name}:{demo_name}:{field}_nonfinite")
                action_max = float(np.max(np.abs(numeric_fields["actions"]), initial=0.0))
                action_abs_max = max(action_abs_max, action_max)
                if action_max > 1.0 + 1e-6:
                    issues.append(f"{path.name}:{demo_name}:action_abs_max={action_max}")
                expected_eef = np.concatenate(
                    (numeric_fields["ee_pos"], numeric_fields["ee_ori"]), axis=1
                )
                state_error = float(
                    np.max(np.abs(numeric_fields["ee_states"] - expected_eef), initial=0.0)
                )
                eef_state_error_max = max(eef_state_error_max, state_error)
                if state_error > 1e-8:
                    issues.append(f"{path.name}:{demo_name}:ee_state_error={state_error}")
                transitions = max(0, n - 1)
                window_count = max(0, 1 + (transitions - horizon) // window_stride)
                dense_windows += window_count
                task_windows += window_count
                nonoverlap_windows += max(0, 1 + (transitions - horizon) // horizon)
        task_reports.append(
            {
                "relative_path": metadata["relative_path"],
                "suite": metadata["suite"],
                "episodes": len(names),
                "frames": task_frames,
                "windows": task_windows,
                "language_instruction": metadata["language_instruction"],
                **(
                    {"sha256": _sha256_file(path)}
                    if include_checksums
                    else {}
                ),
            }
        )

    length_array = np.asarray(lengths, dtype=np.int64)
    percentiles = {}
    if len(length_array):
        percentiles = {
            "min": int(length_array.min()),
            "median": float(np.median(length_array)),
            "p90": float(np.percentile(length_array, 90)),
            "max": int(length_array.max()),
        }
    return {
        "schema": "libero-actionvae-source-audit-v1",
        "source_root": str(root),
        "valid": not issues and not field_missing,
        "issues": issues,
        "task_files": len(selected_by_file),
        "episodes": len(lengths),
        "frames": int(length_array.sum()) if len(length_array) else 0,
        "available_transitions": int(np.maximum(length_array - 1, 0).sum()) if len(length_array) else 0,
        "horizon": horizon,
        "window_stride": window_stride,
        "windows": dense_windows,
        "nonoverlap_windows": nonoverlap_windows,
        "episode_length": percentiles,
        "missing_fields": dict(sorted(field_missing.items())),
        "control_hz_episode_counts": dict(sorted(controls.items())),
        "controller_type_episode_counts": dict(sorted(controller_types.items())),
        "image_convention_episode_counts": dict(sorted(conventions.items())),
        "action_abs_max": action_abs_max,
        "eef_state_error_max": eef_state_error_max,
        "suite_episode_counts": dict(sorted(suite_counts.items())),
        "tasks": task_reports,
        "source_checksums_included": bool(include_checksums),
    }


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
