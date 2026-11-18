"""Exact-row StarVLA backend for the immutable official LIBERO Stage 2 release."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from latent_action_model.action_ae.data import ActionSidecarDataset
from latent_action_model.libero_stage2.action_transform import (
    FLOW_ACTION_DIM,
    LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256,
    LiberoActionNormalizer,
)
from latent_action_model.libero_stage2.audit_join import IDENTITY_COLUMNS, sha256_file
from latent_action_model.libero_stage2.teacher_bundle import (
    LIBERO_12S_ACTION_CHECKPOINT_SHA256,
    LIBERO_12S_ACTION_NORMALIZATION_SHA256,
    LIBERO_12S_CONTROL_HZ,
    LIBERO_12S_DURATION_SECONDS,
    LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
    LIBERO_12S_HORIZON_ACTIONS,
    LIBERO_12S_MANIFEST_SHA256,
    LIBERO_12S_ROW_COUNT,
    LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION,
)
from latent_action_model.libero_stage2.state_transform import (
    ActionVAE14Padded32StateAdapter as _SharedActionVAE14Padded32StateAdapter,
)
from starVLA.model.framework.latent_world.batch_utils import prepare_frame_spatial_uint8


LIBERO_OFFICIAL_STAGE2_BACKEND = "libero_official_stage2_v1"
LIBERO_ACTION_SIDECAR_SUCCESS_MARKER = "libero-actionvae-action-a1-release-v1"
LIBERO_ACTION_SIDECAR_PROVENANCE_SCHEMA = "libero-actionvae-provenance-v1"
LIBERO_ACTION_SIDECAR_SCHEMA = "libero-z-action-sidecar-v1-12s"
LIBERO_LATENT_CACHE_SCHEMA = "libero-stage2-latent-cache-v2"
LIBERO_EMBODIMENT_ID = 25
LIBERO_IMAGE_SIZE = 256
LIBERO_VIDEO_FRAMES = 2
ACTION_STATE_SOURCE_ACTIONVAE14_PADDED32 = "actionvae14_padded32"
ACTION_STATE_SOURCE_STARVLA32 = "starvla32"
ACTION_STATE_SOURCES = frozenset(
    {ACTION_STATE_SOURCE_ACTIONVAE14_PADDED32, ACTION_STATE_SOURCE_STARVLA32}
)

_SPLITS = ("train", "validation", "test")
_SPLIT_COUNTS = {"train": 87441, "validation": 10927, "test": 10901}
_BASE_COLUMNS = {
    *IDENTITY_COLUMNS,
    "group_id",
    "source_id",
    "source_commit",
    "controller_id",
    "episode_seed",
    "control_hz",
    "language_instruction",
    "suite",
    "shard_path",
    "shard_row",
    "e39_valid",
}
_LATENT_COLUMNS = {
    *IDENTITY_COLUMNS,
    "latent_row",
    "z_effect_sha256",
    "z_action_sha256",
    "z_action_logvar_sha256",
}
_LATENT_ARRAYS = (
    "z_effect",
    "z_effect_raw",
    "z_action",
    "z_action_raw",
    "z_action_mean",
    "z_action_logvar",
    "z_action_mean_raw",
    "z_action_logvar_raw",
)


@dataclass(frozen=True)
class _ReleaseContract:
    row_count: int = LIBERO_12S_ROW_COUNT
    source_count: int = 130
    episodes_per_source: int = 50
    split_counts: Mapping[str, int] = None  # type: ignore[assignment]
    horizon: int = LIBERO_12S_HORIZON_ACTIONS
    control_hz: float = LIBERO_12S_CONTROL_HZ
    duration_seconds: float = LIBERO_12S_DURATION_SECONDS
    manifest_sha256: str | None = LIBERO_12S_MANIFEST_SHA256
    action_vae_normalization_sha256: str | None = LIBERO_12S_ACTION_NORMALIZATION_SHA256
    flow_action_normalization_sha256: str | None = LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256
    sidecar_success_marker: str = LIBERO_ACTION_SIDECAR_SUCCESS_MARKER

    def __post_init__(self) -> None:
        if self.split_counts is None:
            object.__setattr__(self, "split_counts", dict(_SPLIT_COUNTS))
        if set(self.split_counts) != set(_SPLITS):
            raise ValueError("release split counts must cover train/validation/test")
        if sum(int(value) for value in self.split_counts.values()) != self.row_count:
            raise ValueError("release split counts do not sum to row_count")
        if self.source_count * self.episodes_per_source <= 0:
            raise ValueError("release source/episode counts must be positive")


_DEFAULT_CONTRACT = _ReleaseContract()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_flow_action_normalizer(path: Path) -> LiberoActionNormalizer:
    value = _read_json(path)
    if "stage2_train_only_normalizer" in value:
        value = value["stage2_train_only_normalizer"]
    if not isinstance(value, Mapping):
        raise ValueError("flow-action normalization JSON has no mapping payload")
    return LiberoActionNormalizer.from_dict(value)


class ActionVAE14Padded32StateAdapter(_SharedActionVAE14Padded32StateAdapter):
    """Backward-compatible backend export bound to the selected release SHA."""

    def __init__(self, normalization_path: str | Path) -> None:
        super().__init__(
            normalization_path,
            expected_sha256=LIBERO_12S_ACTION_NORMALIZATION_SHA256,
        )


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return digest


def _read_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not raw.strip():
            continue
        digest, separator, name = raw.partition("  ")
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate checksum entry at {path}:{line_number}")
        result[name] = _require_sha256(digest, name=f"checksum for {name}")
    return result


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _source_relative_path(source_id: str) -> Path:
    parts = str(source_id).split("/")
    if len(parts) != 3 or parts[0] != "libero" or not parts[1] or not parts[2]:
        raise ValueError(f"invalid official LIBERO source_id: {source_id!r}")
    return Path(parts[1]) / f"{parts[2]}_demo.hdf5"


def _normalize_identity_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} manifest is missing identity columns: {missing}")
    result = frame.copy()
    for column in ("sample_id", "split", "task_id", "window_sha256"):
        result[column] = result[column].astype(str)
    for column in ("episode_id", "start_step", "end_step"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    if result.sample_id.duplicated().any():
        raise ValueError(f"{name} manifest contains duplicate sample_id values")
    return result


def _validate_sidecar_release(
    sidecar_root: Path,
    source_root: Path,
    *,
    contract: _ReleaseContract,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any]]:
    required = (
        "_SUCCESS",
        "checksums.sha256",
        "coverage_report.json",
        "schema.json",
        "provenance.json",
        "manifests/all.parquet",
        "manifests/train.parquet",
        "manifests/validation.parquet",
        "manifests/test.parquet",
        "stats/normalization.json",
    )
    for relative in required:
        if not (sidecar_root / relative).is_file():
            raise FileNotFoundError(sidecar_root / relative)
    marker = (sidecar_root / "_SUCCESS").read_text(encoding="utf-8").strip()
    if marker != contract.sidecar_success_marker:
        raise ValueError(
            f"official ActionVAE sidecar marker mismatch: {marker!r} != "
            f"{contract.sidecar_success_marker!r}"
        )

    schema = _read_json(sidecar_root / "schema.json")
    expected_schema = {
        "schema_version": LIBERO_ACTION_SIDECAR_SCHEMA,
        "horizon_actions": contract.horizon,
        "control_hz": contract.control_hz,
        "duration_seconds": contract.duration_seconds,
    }
    for key, expected in expected_schema.items():
        if schema.get(key) != expected:
            raise ValueError(f"official ActionVAE sidecar schema mismatch for {key}")
    if schema.get("waypoint", {}).get("batch_shape") != ["B", contract.horizon, 7]:
        raise ValueError("official ActionVAE sidecar has the wrong waypoint shape")

    checksums = _read_checksums(sidecar_root / "checksums.sha256")
    manifest_path = sidecar_root / "manifests/all.parquet"
    normalization_path = sidecar_root / "stats/normalization.json"
    manifest_sha = sha256_file(manifest_path)
    normalization_sha = sha256_file(normalization_path)
    if checksums.get("manifests/all.parquet") != manifest_sha:
        raise ValueError("ActionVAE all-manifest checksum binding mismatch")
    if checksums.get("stats/normalization.json") != normalization_sha:
        raise ValueError("ActionVAE normalization checksum binding mismatch")
    if contract.manifest_sha256 is not None and manifest_sha != contract.manifest_sha256:
        raise ValueError("selected LIBERO 1.2s ActionVAE manifest SHA256 mismatch")
    if (
        contract.action_vae_normalization_sha256 is not None
        and normalization_sha != contract.action_vae_normalization_sha256
    ):
        raise ValueError("selected LIBERO ActionVAE normalization SHA256 mismatch")

    frame = pd.read_parquet(manifest_path)
    missing = sorted(_BASE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"official ActionVAE manifest is missing columns: {missing}")
    frame = _normalize_identity_frame(frame, name="official ActionVAE")
    if len(frame) != contract.row_count:
        raise ValueError(
            f"official ActionVAE manifest must contain {contract.row_count} rows, found {len(frame)}"
        )
    split_counts = frame.groupby("split", observed=True).size().to_dict()
    if split_counts != dict(contract.split_counts):
        raise ValueError(
            f"official ActionVAE split counts mismatch: {split_counts} != {dict(contract.split_counts)}"
        )
    if not np.all(
        frame.end_step.to_numpy(dtype=np.int64)
        - frame.start_step.to_numpy(dtype=np.int64)
        == contract.horizon
    ):
        raise ValueError("official ActionVAE manifest does not preserve the 24-action horizon")
    if not np.all(frame.control_hz.to_numpy(dtype=np.float64) == contract.control_hz):
        raise ValueError("official ActionVAE manifest does not use exact 20Hz control")
    if not frame.e39_valid.map(lambda value: value is True or value == 1).all():
        raise ValueError("official A1 sidecar contains a row without e39")
    if set(frame.source_commit.astype(str)) != {"libero-v1"}:
        raise ValueError("official sidecar source commit mismatch")
    if set(frame.controller_id.astype(str)) != {"libero_osc_pose_delta_normalized"}:
        raise ValueError("official sidecar controller contract mismatch")
    if (frame.start_step < 0).any() or frame.language_instruction.astype(str).str.len().eq(0).any():
        raise ValueError("official sidecar contains an invalid start step or instruction")

    frame["source_id"] = frame.source_id.astype(str)
    frame["group_id"] = frame.group_id.astype(str)
    frame["suite"] = frame.suite.astype(str)
    if frame.source_id.nunique() != contract.source_count:
        raise ValueError("official sidecar source count mismatch")
    episode_pairs = frame.loc[:, ["source_id", "episode_id"]].drop_duplicates()
    expected_episodes = contract.source_count * contract.episodes_per_source
    if len(episode_pairs) != expected_episodes:
        raise ValueError(
            f"official sidecar must cover {expected_episodes} local episodes, found {len(episode_pairs)}"
        )
    expected_local_ids = tuple(range(contract.episodes_per_source))
    local_id_sets = frame.groupby("source_id", observed=True).episode_id.apply(
        lambda values: tuple(sorted(set(int(value) for value in values)))
    )
    if any(value != expected_local_ids for value in local_id_sets):
        raise ValueError("every official source must retain local episode IDs 0..49")
    if frame.groupby(["source_id", "episode_id"], observed=True).split.nunique().max() != 1:
        raise ValueError("official episode split ownership is not disjoint")
    if frame.groupby(["source_id", "episode_id"], observed=True).group_id.nunique().max() != 1:
        raise ValueError("official source/local episode identity maps to multiple group IDs")
    if frame.groupby("group_id", observed=True).ngroups != expected_episodes:
        raise ValueError("official group IDs do not provide one identity per local episode")

    source_paths: dict[str, Path] = {}
    for source_id in frame.source_id.unique():
        relative = _source_relative_path(source_id)
        if str(frame.loc[frame.source_id == source_id, "suite"].iloc[0]) != relative.parts[0]:
            raise ValueError(f"suite/source mismatch for {source_id}")
        if set(frame.loc[frame.source_id == source_id, "task_id"].astype(str)) != {source_id}:
            raise ValueError(f"task/source mismatch for {source_id}")
        source_paths[source_id] = (source_root / relative).resolve()

    provenance = _read_json(sidecar_root / "provenance.json")
    if provenance.get("schema") != LIBERO_ACTION_SIDECAR_PROVENANCE_SCHEMA:
        raise ValueError("official ActionVAE sidecar provenance schema mismatch")
    provenance_source_root = Path(str(provenance.get("source_root", ""))).expanduser().resolve()
    if provenance_source_root != source_root:
        raise ValueError(
            f"configured official source root {source_root} is not provenance-bound "
            f"to {provenance_source_root}"
        )
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != contract.source_count:
        raise ValueError("official sidecar provenance must list every source file")
    provenance_paths: set[Path] = set()
    for entry in source_files:
        if not isinstance(entry, Mapping):
            raise ValueError("official source provenance entry must be an object")
        relative = Path(str(entry.get("relative_path", "")))
        _require_sha256(entry.get("sha256"), name=f"official source {relative}")
        if int(entry.get("episodes", -1)) != contract.episodes_per_source:
            raise ValueError(f"official source episode count mismatch: {relative}")
        provenance_paths.add(relative)
    expected_paths = {_source_relative_path(source_id) for source_id in source_paths}
    if provenance_paths != expected_paths:
        raise ValueError("official sidecar provenance/source_id file sets differ")
    actual_paths = {
        path.relative_to(source_root) for path in source_root.glob("*/*_demo.hdf5") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("configured official source root does not contain the exact 130-file set")
    for source_id, path in source_paths.items():
        if not path.is_file() or not path.is_relative_to(source_root):
            raise FileNotFoundError(f"official source for {source_id} does not exist below {source_root}")

    for split in _SPLITS:
        split_relative = f"manifests/{split}.parquet"
        split_path = sidecar_root / split_relative
        if checksums.get(split_relative) != sha256_file(split_path):
            raise ValueError(f"ActionVAE {split} manifest checksum mismatch")
        split_frame = _normalize_identity_frame(
            pd.read_parquet(split_path), name=f"official ActionVAE {split}"
        )
        expected_ids = set(frame.loc[frame.split == split, "sample_id"])
        if set(split_frame.sample_id) != expected_ids or len(split_frame) != len(expected_ids):
            raise ValueError(f"ActionVAE {split} manifest is not an exact partition of all.parquet")
    referenced_shards = set(frame.shard_path.astype(str))
    if not referenced_shards.issubset(checksums):
        raise ValueError("ActionVAE manifest references a shard absent from checksums.sha256")

    coverage = _read_json(sidecar_root / "coverage_report.json")
    expected_coverage = {
        "valid": True,
        "rows": contract.row_count,
        "episodes": expected_episodes,
        "task_files": contract.source_count,
        "horizon_actions": contract.horizon,
        "control_hz": int(contract.control_hz),
        "e39_valid_rows": contract.row_count,
        "split_counts": dict(contract.split_counts),
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise ValueError(f"official sidecar coverage report mismatch for {key}")
    return frame, source_paths, provenance


def _validate_latent_release(
    latent_root: Path,
    *,
    contract: _ReleaseContract,
    base_manifest_sha256: str,
) -> pd.DataFrame:
    required = (
        "_SUCCESS",
        "checksums.sha256",
        "latents.h5",
        "manifest.parquet",
        "normalization.json",
        "provenance.json",
    )
    for name in required:
        if not (latent_root / name).is_file():
            raise FileNotFoundError(latent_root / name)
    success = _read_json(latent_root / "_SUCCESS")
    expected_teachers = {
        "effect": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "action": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
    }
    if (
        success.get("schema_version") != LIBERO_LATENT_CACHE_SCHEMA
        or success.get("release_level") != "full"
        or success.get("all_gates_passed") is not True
        or int(success.get("row_count", -1)) != contract.row_count
    ):
        raise ValueError("Stage 2 latent cache is not a complete full Gaussian v2 release")
    if success.get("teacher_checkpoint_sha256") != expected_teachers:
        raise ValueError("Stage 2 latent cache _SUCCESS selected the wrong teachers")
    checksums_path = latent_root / "checksums.sha256"
    if sha256_file(checksums_path) != success.get("checksums_sha256"):
        raise ValueError("Stage 2 latent cache checksum-index SHA256 mismatch")
    checksums = _read_checksums(checksums_path)
    expected_artifacts = {"latents.h5", "manifest.parquet", "normalization.json", "provenance.json"}
    if set(checksums) != expected_artifacts:
        raise ValueError("Stage 2 latent cache checksum index has an unexpected file set")
    for name in ("manifest.parquet", "normalization.json", "provenance.json"):
        if sha256_file(latent_root / name) != checksums[name]:
            raise ValueError(f"Stage 2 latent cache {name} checksum mismatch")

    provenance = _read_json(latent_root / "provenance.json")
    if (
        provenance.get("schema_version") != LIBERO_LATENT_CACHE_SCHEMA
        or provenance.get("release_level") != "full"
        or provenance.get("ready") is not True
        or int(provenance.get("rows", -1)) != contract.row_count
        or provenance.get("teacher_checkpoint_sha256") != expected_teachers
    ):
        raise ValueError("Stage 2 latent cache provenance is incomplete or incompatible")
    if provenance.get("split_counts") != dict(contract.split_counts):
        raise ValueError("Stage 2 latent cache split counts mismatch")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage 2 latent cache provenance lacks artifact bindings")
    for role, name in (
        ("h5", "latents.h5"),
        ("manifest", "manifest.parquet"),
        ("normalization", "normalization.json"),
    ):
        binding = artifacts.get(role)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != name
            or binding.get("sha256") != checksums[name]
        ):
            raise ValueError(f"Stage 2 latent cache provenance binding mismatch for {role}")
    source = provenance.get("source")
    teacher_contract = source.get("teacher_contract") if isinstance(source, Mapping) else None
    expected_contract = {
        "schema_version": LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION,
        "effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
        "horizon_actions": contract.horizon,
        "control_hz": contract.control_hz,
        "duration_seconds": contract.duration_seconds,
        "row_count": contract.row_count,
    }
    if not isinstance(teacher_contract, Mapping) or any(
        teacher_contract.get(key) != expected for key, expected in expected_contract.items()
    ):
        raise ValueError("Stage 2 latent cache lacks the exact selected 1.2s teacher contract")
    if source.get("base_manifest_sha256") != base_manifest_sha256:
        raise ValueError("Stage 2 latent cache is not bound to the selected official manifest")
    action_distribution = provenance.get("action_distribution")
    if (
        provenance.get("default_teacher_value") != "posterior_mean"
        or not isinstance(action_distribution, Mapping)
        or action_distribution.get("latent_dim") != 32
        or action_distribution.get("parameter_dim") != 64
        or not isinstance(action_distribution.get("teacher_statistics"), Mapping)
    ):
        raise ValueError("Stage 2 latent cache lacks Gaussian posterior statistics")

    frame = pd.read_parquet(latent_root / "manifest.parquet")
    missing = sorted(_LATENT_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Stage 2 latent manifest is missing columns: {missing}")
    frame = _normalize_identity_frame(frame, name="Stage 2 latent")
    if len(frame) != contract.row_count:
        raise ValueError("Stage 2 latent manifest row count mismatch")
    latent_rows = pd.to_numeric(frame.latent_row, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(latent_rows, np.arange(contract.row_count, dtype=np.int64)):
        raise ValueError("Stage 2 latent rows must be a complete ordered permutation")
    for column in ("z_effect_sha256", "z_action_sha256", "z_action_logvar_sha256"):
        for value in frame[column]:
            _require_sha256(value, name=column)

    with h5py.File(latent_root / "latents.h5", "r") as handle:
        if (
            _decode(handle.attrs.get("schema_version", "")) != LIBERO_LATENT_CACHE_SCHEMA
            or _decode(handle.attrs.get("release_level", "")) != "full"
            or not bool(handle.attrs.get("complete", False))
        ):
            raise ValueError("Stage 2 latent HDF5 is incomplete or has the wrong schema")
        missing_arrays = sorted(set(_LATENT_ARRAYS).difference(handle.keys()))
        if missing_arrays:
            raise ValueError(f"Stage 2 latent HDF5 is missing arrays: {missing_arrays}")
        for name in _LATENT_ARRAYS:
            if handle[name].shape != (contract.row_count, 32) or handle[name].dtype != np.float32:
                raise ValueError(
                    f"Stage 2 latent array {name} must be float32[{contract.row_count},32]"
                )
    return frame


def _align_latent_manifest(base: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
    base_ids = set(base.sample_id)
    latent_ids = set(latent.sample_id)
    if base_ids != latent_ids:
        raise ValueError(
            "official/latent exact-set mismatch: "
            f"missing={len(base_ids - latent_ids)}, extra={len(latent_ids - base_ids)}"
        )
    aligned = latent.set_index("sample_id", drop=False).loc[base.sample_id]
    for column in IDENTITY_COLUMNS[1:]:
        if not np.array_equal(base[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"official/latent identity mismatch: {column}")
    result = base.copy()
    for column in (
        "latent_row",
        "z_effect_sha256",
        "z_action_sha256",
        "z_action_logvar_sha256",
    ):
        result[column] = aligned[column].to_numpy()
    return result


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).lower()
    mapping = {
        "train": "train",
        "all": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported official Stage 2 dataset mode: {mode!r}")
    return mapping[normalized]


class LiberoOfficialStage2Dataset(Dataset[dict[str, Any]]):
    """Join official images, ActionVAE supervision, and fixed Gaussian teachers by exact row."""

    def __init__(
        self,
        *,
        action_sidecar_root: str | Path,
        source_root: str | Path,
        latent_root: str | Path,
        action_normalization_path: str | Path,
        action_state_source: str,
        split: str,
        allow_test: bool = False,
        num_frames: int = LIBERO_VIDEO_FRAMES,
        image_size: int = LIBERO_IMAGE_SIZE,
        max_rows: int | None = None,
        max_open_source_files: int = 8,
        _contract: _ReleaseContract = _DEFAULT_CONTRACT,
    ) -> None:
        super().__init__()
        self.split = _normalize_mode(split)
        if self.split == "test" and not allow_test:
            raise PermissionError("official Stage 2 test data requires explicit evaluator authorization")
        if num_frames != LIBERO_VIDEO_FRAMES:
            raise ValueError("official Stage 2 requires exactly two primary frames at t and t+23")
        if image_size != LIBERO_IMAGE_SIZE:
            raise ValueError("official Stage 2 image preprocessing is locked to 256x256")
        if max_rows is not None and max_rows < 1:
            raise ValueError("official Stage 2 max_rows must be positive")
        if max_open_source_files < 1:
            raise ValueError("max_open_source_files must be positive")
        self.action_state_source = str(action_state_source)
        if self.action_state_source not in ACTION_STATE_SOURCES:
            raise ValueError(
                "stage2_action_state_source must be one of "
                f"{sorted(ACTION_STATE_SOURCES)}, got {self.action_state_source!r}"
            )
        if self.action_state_source == ACTION_STATE_SOURCE_STARVLA32:
            raise ValueError(
                "starvla32 is intentionally unavailable for the official exact-row backend: "
                "provide an official-HDF5 raw-state layout and a full-6500-demo train-only "
                "StarVLA normalization artifact; merged LeRobot statistics are forbidden"
            )

        self.action_sidecar_root = Path(action_sidecar_root).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        self.latent_root = Path(latent_root).expanduser().resolve()
        self.action_normalization_path = Path(action_normalization_path).expanduser().resolve()
        self.max_open_source_files = int(max_open_source_files)
        self._contract = _contract
        base, self._source_paths, self.sidecar_provenance = _validate_sidecar_release(
            self.action_sidecar_root,
            self.source_root,
            contract=self._contract,
        )
        base_manifest_sha = sha256_file(self.action_sidecar_root / "manifests/all.parquet")
        latent = _validate_latent_release(
            self.latent_root,
            contract=self._contract,
            base_manifest_sha256=base_manifest_sha,
        )
        joined = _align_latent_manifest(base, latent)

        if not self.action_normalization_path.is_file():
            raise FileNotFoundError(self.action_normalization_path)
        self.flow_action_normalization_sha256 = sha256_file(self.action_normalization_path)
        if (
            self._contract.flow_action_normalization_sha256 is not None
            and self.flow_action_normalization_sha256
            != self._contract.flow_action_normalization_sha256
        ):
            raise ValueError("official Stage 2 flow-action normalization SHA256 mismatch")
        self.action_normalizer = _load_flow_action_normalizer(self.action_normalization_path)

        self.action_sidecar = ActionSidecarDataset(
            self.action_sidecar_root,
            self.split,
            verify_row_identity=True,
            require_e39=True,
        )
        sidecar_positions = {
            sample_id: index for index, sample_id in enumerate(self.action_sidecar.sample_ids)
        }
        split_frame = joined.loc[joined.split == self.split].copy()
        if set(split_frame.sample_id) != set(sidecar_positions):
            raise ValueError("official all/split ActionVAE manifests have different sample sets")
        split_frame = split_frame.set_index("sample_id", drop=False).loc[
            list(self.action_sidecar.sample_ids)
        ].reset_index(drop=True)
        if max_rows is not None:
            split_frame = split_frame.iloc[: int(max_rows)].reset_index(drop=True)
        if split_frame.empty:
            raise ValueError(f"official Stage 2 manifest has no {self.split} rows")
        self.frame = split_frame
        self._sidecar_indices = [sidecar_positions[sample_id] for sample_id in self.frame.sample_id]

        self._source_handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._latent_handle: h5py.File | None = None
        self._handle_owner_pid: int | None = None
        self._source_languages: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def handle_owner_pid(self) -> int | None:
        return self._handle_owner_pid

    @property
    def open_source_count(self) -> int:
        return len(self._source_handles)

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"official Stage 2 index {index} out of range for length {len(self)}")
        return int(index)

    def close(self) -> None:
        for handle in getattr(self, "_source_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        if hasattr(self, "_source_handles"):
            self._source_handles.clear()
        latent = getattr(self, "_latent_handle", None)
        if latent is not None:
            try:
                latent.close()
            except Exception:
                pass
        self._latent_handle = None
        self.action_sidecar.close()
        self._handle_owner_pid = None

    def reset_worker_state(self) -> None:
        self.close()
        self._source_handles = OrderedDict()
        self._latent_handle = None
        self._handle_owner_pid = os.getpid()
        self.action_sidecar.reset_worker_state()

    def _ensure_process_local_handles(self) -> None:
        if self._handle_owner_pid != os.getpid():
            self.reset_worker_state()

    def _source_handle(self, source_id: str) -> h5py.File:
        self._ensure_process_local_handles()
        path = self._source_paths[source_id]
        key = str(path)
        cached = self._source_handles.pop(key, None)
        if cached is not None:
            self._source_handles[key] = cached
            return cached
        handle = h5py.File(path, "r")
        self._source_handles[key] = handle
        while len(self._source_handles) > self.max_open_source_files:
            _, evicted = self._source_handles.popitem(last=False)
            evicted.close()
        return handle

    def _latents(self) -> h5py.File:
        self._ensure_process_local_handles()
        if self._latent_handle is None:
            self._latent_handle = h5py.File(self.latent_root / "latents.h5", "r")
        return self._latent_handle

    def _validate_source_language(self, source_id: str, handle: h5py.File, expected: str) -> None:
        cached = self._source_languages.get(source_id)
        if cached is None:
            if "data" not in handle:
                raise ValueError(f"official source has no data group: {self._source_paths[source_id]}")
            data = handle["data"]
            problem = json.loads(_decode(data.attrs.get("problem_info", "")))
            cached = str(problem.get("language_instruction", ""))
            if int(data.attrs.get("num_demos", -1)) != self._contract.episodes_per_source:
                raise ValueError(f"official source demo count mismatch: {source_id}")
            self._source_languages[source_id] = cached
        if cached != expected:
            raise ValueError(f"official source/manifest language mismatch: {source_id}")

    def _read_teacher_latents(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        handle = self._latents()
        latent_row = int(row.latent_row)
        effect = np.asarray(handle["z_effect"][latent_row], dtype=np.float32)
        action = np.asarray(handle["z_action_mean"][latent_row], dtype=np.float32)
        action_alias = np.asarray(handle["z_action"][latent_row], dtype=np.float32)
        action_logvar = np.asarray(handle["z_action_logvar"][latent_row], dtype=np.float32)
        if not np.array_equal(action, action_alias):
            raise ValueError(f"z_action alias/mean mismatch for {row.sample_id}")
        for value, column in (
            (effect, "z_effect_sha256"),
            (action, "z_action_sha256"),
            (action_logvar, "z_action_logvar_sha256"),
        ):
            if value.shape != (32,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {column} latent for {row.sample_id}")
            if _sha256_array(value) != str(row[column]):
                raise ValueError(f"latent row checksum mismatch for {row.sample_id}: {column}")
        return (
            torch.from_numpy(effect.copy()),
            torch.from_numpy(action.copy()),
            torch.from_numpy(action_logvar.copy()),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        row = self.frame.iloc[index]
        sidecar = self.action_sidecar[self._sidecar_indices[index]]
        metadata = sidecar["metadata"]
        for key in (
            "sample_id",
            "split",
            "task_id",
            "source_id",
            "episode_id",
            "start_step",
            "end_step",
            "window_sha256",
        ):
            if str(metadata[key]) != str(row[key]):
                raise ValueError(f"official sidecar row identity mismatch for {key}: {row.sample_id}")

        source_id = str(row.source_id)
        source = self._source_handle(source_id)
        self._validate_source_language(source_id, source, str(row.language_instruction))
        demo_name = f"data/demo_{int(row.episode_id)}"
        if demo_name not in source:
            raise ValueError(f"official source is missing local episode {demo_name}: {source_id}")
        demo = source[demo_name]
        start = int(row.start_step)
        end = int(row.end_step)
        if end - start != self._contract.horizon:
            raise ValueError("official row lost its fixed 24-action teacher endpoint")
        release_future = start + self._contract.horizon - 1
        primary = demo["obs/agentview_rgb"]
        wrist = demo["obs/eye_in_hand_rgb"]
        if end >= len(primary) or release_future >= len(primary) or start >= len(wrist):
            raise IndexError(f"official source cannot satisfy t/t+23/t+24 identity: {row.sample_id}")

        official_actions = np.asarray(demo["actions"][start:end], dtype=np.float32)
        native_actions = sidecar["native_actions"][: self._contract.horizon, :7].numpy()
        if official_actions.shape != (self._contract.horizon, 7) or not np.array_equal(
            official_actions, native_actions
        ):
            raise ValueError(
                f"official source/local episode/start_step action fingerprint mismatch: {row.sample_id}"
            )
        native_mask = sidecar["native_action_mask"][: self._contract.horizon, :7].numpy()
        normalized_actions, normalized_mask, _ = self.action_normalizer.transform(
            native_actions,
            native_mask,
            clip=True,
        )
        if (
            normalized_actions.shape != (self._contract.horizon, FLOW_ACTION_DIM)
            or not normalized_mask[:, :7].all()
            or normalized_mask[:, 7:].any()
        ):
            raise ValueError(f"official flow-action mask is incomplete: {row.sample_id}")

        primary_video = torch.stack(
            [
                prepare_frame_spatial_uint8(np.asarray(primary[step]), target_hw=(256, 256))
                for step in (start, release_future)
            ],
            dim=0,
        ).unsqueeze(0)
        wrist_images = prepare_frame_spatial_uint8(
            np.asarray(wrist[start]), target_hw=(256, 256)
        ).unsqueeze(0)
        z_effect, z_action, z_action_logvar = self._read_teacher_latents(row)
        teacher_current_state = sidecar["current_state"].to(dtype=torch.float32).contiguous()
        teacher_current_state_mask = sidecar["current_state_mask"].to(dtype=torch.bool).contiguous()
        if torch.any(~teacher_current_state_mask & (teacher_current_state != 0)):
            raise ValueError(f"invalid ActionVAE state dimensions must be zero: {row.sample_id}")
        action_predictor_state = ActionVAE14Padded32StateAdapter.pad(teacher_current_state)

        identity = {
            "sample_id": str(row.sample_id),
            "split": str(row.split),
            "task_id": str(row.task_id),
            "source_id": source_id,
            "suite": str(row.suite),
            "group_id": str(row.group_id),
            "episode_id": int(row.episode_id),
            "start_step": start,
            "end_step": end,
            "release_future_step": release_future,
            "release_future_delta_steps": self._contract.horizon - 1,
            "teacher_future_step": end,
            "teacher_future_delta_steps": self._contract.horizon,
            "window_sha256": str(row.window_sha256),
        }
        flow_state = torch.zeros((1, FLOW_ACTION_DIM), dtype=torch.float32)
        flow_state_mask = torch.zeros((1, FLOW_ACTION_DIM), dtype=torch.bool)
        return {
            "primary_videos": primary_video,
            "wrist_images": wrist_images,
            "lang": str(row.language_instruction),
            "state": flow_state,
            "state_mask": flow_state_mask,
            "action_predictor_state": action_predictor_state,
            "action": torch.from_numpy(normalized_actions),
            "action_mask": torch.from_numpy(normalized_mask),
            "embodiment_id": LIBERO_EMBODIMENT_ID,
            "action_hz": float(self._contract.control_hz),
            "sample_id": str(row.sample_id),
            "z_effect": z_effect,
            "z_action": z_action,
            "z_action_logvar": z_action_logvar,
            "teacher_trajectory": sidecar["trajectory"].to(dtype=torch.float32).contiguous(),
            "teacher_current_state": teacher_current_state,
            "teacher_time_mask": sidecar["time_mask"].to(dtype=torch.bool).contiguous(),
            "teacher_pose_mask": sidecar["pose_mask"].to(dtype=torch.bool).contiguous(),
            "teacher_gripper_mask": sidecar["gripper_mask"].to(dtype=torch.bool).contiguous(),
            "teacher_current_state_mask": teacher_current_state_mask,
            "stage2_evaluation_metadata": identity,
        }

    def build_dataset_statistics(self) -> dict[str, Any]:
        episodes = self.frame.loc[:, ["source_id", "episode_id"]].drop_duplicates()
        return {
            "franka": {
                "num_transitions": len(self.frame),
                "num_trajectories": len(episodes),
                "stage2_backend": LIBERO_OFFICIAL_STAGE2_BACKEND,
                "split": self.split,
                "official_manifest_sha256": sha256_file(
                    self.action_sidecar_root / "manifests/all.parquet"
                ),
                "latent_manifest_sha256": sha256_file(self.latent_root / "manifest.parquet"),
                "action_vae_normalization_sha256": sha256_file(
                    self.action_sidecar_root / "stats/normalization.json"
                ),
                "flow_action_normalization_sha256": self.flow_action_normalization_sha256,
                "flow_action": {
                    "normalization_schema": "libero-stage2-action-normalization-v1",
                    "normalization_sha256": self.flow_action_normalization_sha256,
                    "generic_dataset_statistics": False,
                },
                "flow_state": {
                    "available": False,
                    "representation": "zero_with_all_invalid_mask",
                    "requires_flow_cfg_use_state": False,
                    "generic_dataset_statistics": False,
                },
                "action_predictor_state": {
                    "source": ACTION_STATE_SOURCE_ACTIONVAE14_PADDED32,
                    "raw_semantics": "world_xyz_principal_rotvec_physical_width_reserved7",
                    "raw_dim": 14,
                    "valid_dim": 7,
                    "output_dim": 32,
                    "normalization": "median_and_1.4826_mad",
                    "normalization_sha256": sha256_file(
                        self.action_sidecar_root / "stats/normalization.json"
                    ),
                    "generic_dataset_statistics": False,
                },
                "source_root": str(self.source_root),
                "source_count": self._contract.source_count,
                "official_episode_count": self._contract.source_count
                * self._contract.episodes_per_source,
                "official_row_count": self._contract.row_count,
            }
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_source_handles"] = OrderedDict()
        state["_latent_handle"] = None
        state["_handle_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._source_handles = OrderedDict()
        self._latent_handle = None
        self._handle_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _cfg_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def _required_cfg_path(data_cfg: Any, key: str) -> Path:
    value = _cfg_get(data_cfg, key, None)
    if value is None or not str(value).strip():
        raise ValueError(f"{LIBERO_OFFICIAL_STAGE2_BACKEND} requires datasets.vla_data.{key}")
    return Path(str(value))


def _required_cfg_string(data_cfg: Any, key: str) -> str:
    value = _cfg_get(data_cfg, key, None)
    if value is None or not str(value).strip():
        raise ValueError(f"{LIBERO_OFFICIAL_STAGE2_BACKEND} requires datasets.vla_data.{key}")
    return str(value)


def build_libero_official_stage2_dataset(
    data_cfg: Any,
    *,
    mode: str,
) -> LiberoOfficialStage2Dataset:
    return LiberoOfficialStage2Dataset(
        action_sidecar_root=_required_cfg_path(data_cfg, "stage2_action_sidecar_root"),
        source_root=_required_cfg_path(data_cfg, "stage2_source_root"),
        latent_root=_required_cfg_path(data_cfg, "stage2_paired_cache"),
        action_normalization_path=_required_cfg_path(data_cfg, "stage2_action_normalization"),
        action_state_source=_required_cfg_string(data_cfg, "stage2_action_state_source"),
        split=_normalize_mode(mode),
        allow_test=bool(_cfg_get(data_cfg, "stage2_allow_test", False)),
        num_frames=int(_cfg_get(data_cfg, "num_frames", LIBERO_VIDEO_FRAMES)),
        image_size=int(_cfg_get(data_cfg, "image_resolution", LIBERO_IMAGE_SIZE)),
        max_rows=_cfg_get(data_cfg, "stage2_max_rows", None),
        max_open_source_files=int(_cfg_get(data_cfg, "stage2_max_open_source_files", 8)),
    )
