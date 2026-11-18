"""Freeze the validation-selected Stage 1 teachers used by LIBERO Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


PRIMARY_SELECTION_SCHEMA_VERSION = "libero-stage2-primary-action-selection-v1"
ABLATION_SELECTION_SCHEMA_VERSION = "libero-stage2-ablation-action-selection-v1"
BUNDLE_SCHEMA_VERSION = "libero-stage2-teacher-bundle-v1"
HANDOFF_SUCCESS_SCHEMA_VERSION = "libero-stage2-handoff-success-v1"
STAGE1_SELECTION_SCHEMA_VERSION = "lawam-stage1-checkpoint-selection-v1"
SELECTION_MONITOR = "val/loss"
SELECTION_POLICY = "minimum_actionae_validation_loss_then_seed"
ABLATION_SELECTION_POLICY = "explicit_ablation_validation_best_actionae"
LIBERO_12S_SELECTION_SCHEMA_VERSION = "libero-stage2-libero-12s-action-selection-v1"
LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION = "libero-stage2-libero-12s-teachers-v1"

LIBERO_12S_EFFECT_CHECKPOINT_SHA256 = (
    "75db2595c6ae37b636f8d6115051fe998aad973b7cbbda2ce18e4265cc86fe4a"
)
LIBERO_12S_ACTION_CHECKPOINT_SHA256 = (
    "9c7aeb75e226d4d9f0af2632a95117a166dcdb6f4681e70529b623a7489008f5"
)
LIBERO_12S_EFFECT_CONFIG_SHA256 = (
    "8ed6a8a6fa5927ed2f4711fafa596b63213bc432a62e93d45a5c397ee24c9853"
)
LIBERO_12S_ACTION_CONFIG_SHA256 = (
    "47a919903a76057322a42607721f8742d33ed40af07a97547905e96b1db3d5f4"
)
LIBERO_12S_EFFECT_POSTERIOR_SHA256 = (
    "2f6911529152913e8945991e1a1b019eee51d8defd371e209d58a7b6b51277f0"
)
LIBERO_12S_EFFECT_SELECTION_SHA256 = (
    "a1b6891593e4568732900c0a67e3141329a1bee5c6cac212db38dca8620c0d39"
)
LIBERO_12S_MANIFEST_SHA256 = (
    "2b1400cb4caf1ee201e12650f5d9bc8541c0293b65f6ea1abbdfbc821ca20618"
)
LIBERO_12S_ACTION_NORMALIZATION_SHA256 = (
    "bc8b3fd2d814ced88c707c60cd22cc43151912b5834426e60e41b193dbe0a0a4"
)
LIBERO_12S_ROW_COUNT = 109269
LIBERO_12S_HORIZON_ACTIONS = 24
LIBERO_12S_CONTROL_HZ = 20.0
LIBERO_12S_DURATION_SECONDS = 1.2
LIBERO_12S_ACTION_SCHEMA_VERSION = "action-tcn-beta-vae-v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_12S_EFFECT_CHECKPOINT = Path(
    os.environ.get("CLEAR_LIBERO_EFFECT_CHECKPOINT", str(_REPO_ROOT / "checkpoints/clear/libero/effect/epoch-39.ckpt"))
)
LIBERO_12S_ACTION_CHECKPOINT = Path(
    os.environ.get("CLEAR_LIBERO_ACTION_CHECKPOINT", str(_REPO_ROOT / "checkpoints/clear/libero/action/step-0009889.ckpt"))
)
LIBERO_12S_EFFECT_CONFIG = (
    _REPO_ROOT / "configs/reference/effect_stage1_libero_v1_12s_dino_residual.yaml"
)
LIBERO_12S_ACTION_CONFIG = (
    _REPO_ROOT / "configs/reference/action_beta_vae_libero_v1_12s_a1.yaml"
)
LIBERO_12S_ACTION_DATA_ROOT = Path(
    os.environ.get("CLEAR_LIBERO_ACTION_DATA", str(_REPO_ROOT / "data/libero/action_a1"))
)
LIBERO_12S_EFFECT_POSTERIOR = Path(
    os.environ.get("CLEAR_LIBERO_EFFECT_POSTERIOR", str(_REPO_ROOT / "data/libero/effect_artifacts/effect_posterior_means.npz"))
)
LIBERO_12S_EFFECT_SELECTION = Path(
    os.environ.get("CLEAR_LIBERO_EFFECT_SELECTION", str(_REPO_ROOT / "checkpoints/clear/libero/effect/downstream_selection.json"))
)


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return digest


def _manifest_digest(checksums_path: Path, relative_path: str) -> str:
    matches: list[str] = []
    for line_number, raw in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, separator, path = raw.partition("  ")
        if not separator:
            raise ValueError(f"invalid checksum line {checksums_path}:{line_number}")
        if path == relative_path:
            matches.append(_require_sha256(digest, name=f"checksum {relative_path}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one checksum for {relative_path!r} in {checksums_path}, found {len(matches)}"
        )
    return matches[0]


def _checkpoint_validation_loss(checkpoint_path: Path) -> float:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {checkpoint_path}")
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, Mapping):
        raise ValueError(f"checkpoint has no callback state: {checkpoint_path}")
    eligible: list[Mapping[str, Any]] = []
    for state in callbacks.values():
        if isinstance(state, Mapping) and state.get("monitor") == SELECTION_MONITOR:
            eligible.append(state)
    if len(eligible) != 1:
        raise ValueError(
            f"checkpoint must have exactly one {SELECTION_MONITOR!r} callback, found {len(eligible)}: "
            f"{checkpoint_path}"
        )
    state = eligible[0]
    best_path = _require_file(Path(str(state.get("best_model_path", ""))))
    if best_path != checkpoint_path:
        raise ValueError(
            f"candidate is not the callback-selected validation checkpoint: {checkpoint_path} != {best_path}"
        )
    score_value = state.get("best_model_score")
    if isinstance(score_value, torch.Tensor):
        if score_value.numel() != 1:
            raise ValueError(f"best_model_score must be scalar: {checkpoint_path}")
        score_value = score_value.detach().cpu().item()
    score = float(score_value)
    if not math.isfinite(score):
        raise ValueError(f"best validation score is not finite: {checkpoint_path}")
    return score


def _single_resolved_config(run_root: Path) -> Path:
    matches = sorted(run_root.glob("version_*/config.yaml"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one resolved config below {run_root}, found {len(matches)}")
    return _require_file(matches[0])


def select_primary_action_teacher(
    release_provenance: Mapping[str, Any],
    *,
    score_reader: Callable[[Path], float] = _checkpoint_validation_loss,
    verify_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Select one compatible A1 seed without consulting any test metric."""

    if release_provenance.get("status") != "complete":
        raise ValueError("Stage 1 release status is not complete")
    test_policy = release_provenance.get("test_policy")
    if not isinstance(test_policy, Mapping) or test_policy.get(
        "actionae_test_used_for_training_or_checkpoint_selection"
    ) is not False:
        raise ValueError("Stage 1 release does not prove test-independent ActionAE selection")
    effect = release_provenance.get("effect_encoder")
    if not isinstance(effect, Mapping):
        raise ValueError("Stage 1 release is missing effect_encoder provenance")
    canonical_effect_sha = _require_sha256(
        effect.get("canonical_checkpoint_sha256"), name="canonical effect checkpoint"
    )
    action = release_provenance.get("action_ae")
    runs = action.get("runs") if isinstance(action, Mapping) else None
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("Stage 1 release is missing ActionAE runs")

    candidates: list[dict[str, Any]] = []
    for raw_seed, variants in runs.items():
        if not isinstance(variants, Mapping) or not isinstance(variants.get("a1"), Mapping):
            raise ValueError(f"ActionAE seed {raw_seed} has no A1 candidate")
        seed = int(raw_seed)
        candidate = variants["a1"]
        dependency = candidate.get("effect_dependency")
        if not isinstance(dependency, Mapping):
            raise ValueError(f"A1 seed {seed} is missing effect dependency")
        dependency_sha = _require_sha256(
            dependency.get("checkpoint_sha256"), name=f"A1 seed {seed} effect checkpoint"
        )
        if dependency_sha != canonical_effect_sha:
            raise ValueError(f"A1 seed {seed} uses an incompatible effect checkpoint")
        checkpoint_path = _require_file(Path(str(candidate.get("checkpoint", ""))))
        checkpoint_sha = _require_sha256(
            candidate.get("checkpoint_sha256"), name=f"A1 seed {seed} checkpoint"
        )
        if verify_checkpoint_hashes and _sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"A1 seed {seed} checkpoint hash mismatch")
        validation_loss = float(score_reader(checkpoint_path))
        if not math.isfinite(validation_loss):
            raise ValueError(f"A1 seed {seed} validation loss is not finite")
        config_path = _single_resolved_config(checkpoint_path.parents[1])
        candidates.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "config": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "epoch": int(candidate["epoch"]),
                "selection_monitor": SELECTION_MONITOR,
                "validation_loss": validation_loss,
                "effect_checkpoint_sha256_used_for_training": dependency_sha,
            }
        )
    candidates.sort(key=lambda item: (item["validation_loss"], item["seed"]))
    selected = candidates[0]
    return {
        "schema_version": PRIMARY_SELECTION_SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "selection_split": "validation",
        "selection_monitor": SELECTION_MONITOR,
        "mode": "min",
        "tie_break": "lowest_numeric_seed",
        "test_metrics_read": False,
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda item: item["seed"]),
        "selected": selected,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_small_artifact(path: Path, expected_sha256: str, *, name: str) -> Path:
    path = _require_file(path)
    if _sha256_file(path) != _require_sha256(expected_sha256, name=name):
        raise ValueError(f"{name} hash mismatch: {path}")
    return path


def _checkpoint_epoch(checkpoint_path: Path) -> int:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping) or "epoch" not in checkpoint:
        raise ValueError(f"checkpoint does not declare its epoch: {checkpoint_path}")
    epoch = int(checkpoint["epoch"])
    if epoch < 0:
        raise ValueError(f"checkpoint epoch must be non-negative: {checkpoint_path}")
    return epoch


def _require_artifact_sha256(path: Path, expected: str, *, name: str) -> Path:
    resolved = _require_file(path)
    if _sha256_file(resolved) != _require_sha256(expected, name=name):
        raise ValueError(f"{name} hash mismatch: {resolved}")
    return resolved


def _require_bound_path(value: Any, expected: Path, *, name: str) -> None:
    actual = _require_file(Path(str(value)))
    if actual != _require_file(expected):
        raise ValueError(f"{name} path mismatch: {actual} != {expected}")


def preflight_libero_12s_teachers(
    *,
    verify_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Verify the immutable 1.2-second Effect/Action teacher pair."""

    effect_checkpoint = _require_file(LIBERO_12S_EFFECT_CHECKPOINT)
    action_checkpoint = _require_file(LIBERO_12S_ACTION_CHECKPOINT)
    if verify_checkpoint_hashes:
        _require_artifact_sha256(
            effect_checkpoint,
            LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            name="LIBERO 1.2s effect checkpoint",
        )
        _require_artifact_sha256(
            action_checkpoint,
            LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            name="LIBERO 1.2s action checkpoint",
        )
    _require_artifact_sha256(
        LIBERO_12S_EFFECT_CONFIG,
        LIBERO_12S_EFFECT_CONFIG_SHA256,
        name="LIBERO 1.2s effect config",
    )
    _require_artifact_sha256(
        LIBERO_12S_ACTION_CONFIG,
        LIBERO_12S_ACTION_CONFIG_SHA256,
        name="LIBERO 1.2s action config",
    )
    effect_posterior = _require_artifact_sha256(
        LIBERO_12S_EFFECT_POSTERIOR,
        LIBERO_12S_EFFECT_POSTERIOR_SHA256,
        name="LIBERO 1.2s effect posterior",
    )
    effect_selection = _require_artifact_sha256(
        LIBERO_12S_EFFECT_SELECTION,
        LIBERO_12S_EFFECT_SELECTION_SHA256,
        name="LIBERO 1.2s effect selection",
    )

    effect_metadata = _read_json(effect_posterior.with_suffix(".json"))
    expected_effect_metadata = {
        "checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "model_config_sha256": LIBERO_12S_EFFECT_CONFIG_SHA256,
        "deterministic": True,
        "posterior_statistic": "mean",
        "latent_shape": [1, 32],
        "sample_count": LIBERO_12S_ROW_COUNT,
    }
    for key, expected in expected_effect_metadata.items():
        if effect_metadata.get(key) != expected:
            raise ValueError(f"LIBERO 1.2s effect metadata mismatch for {key}")
    _require_bound_path(
        effect_metadata.get("checkpoint"), effect_checkpoint, name="effect metadata checkpoint"
    )
    _require_bound_path(
        effect_metadata.get("model_config"),
        LIBERO_12S_EFFECT_CONFIG,
        name="effect metadata config",
    )
    selection_metadata = effect_metadata.get("checkpoint_selection")
    if not isinstance(selection_metadata, Mapping):
        raise ValueError("LIBERO 1.2s effect metadata lacks checkpoint selection")
    if selection_metadata.get("status") != "explicit_ablation":
        raise ValueError("LIBERO 1.2s effect teacher must retain explicit-ablation provenance")
    if selection_metadata.get("report_sha256") != LIBERO_12S_EFFECT_SELECTION_SHA256:
        raise ValueError("LIBERO 1.2s effect selection SHA256 mismatch")
    _require_bound_path(
        selection_metadata.get("report"), effect_selection, name="effect selection report"
    )

    selection = _read_json(effect_selection)
    if (
        selection.get("schema") != "lawam-stage1-downstream-selection-v1"
        or selection.get("selected_checkpoint_id") != effect_checkpoint.name
        or selection.get("selection_status") != "explicit_ablation"
        or selection.get("eligible_under_locked_gates") is not False
        or selection.get("test_consulted") is not False
    ):
        raise ValueError("LIBERO 1.2s effect selection provenance is invalid")

    data_root = LIBERO_12S_ACTION_DATA_ROOT.expanduser().resolve()
    sidecar_schema_path = _require_file(data_root / "schema.json")
    sidecar_provenance_path = _require_file(data_root / "provenance.json")
    sidecar_manifest_path = _require_artifact_sha256(
        data_root / "manifests/all.parquet",
        LIBERO_12S_MANIFEST_SHA256,
        name="LIBERO 1.2s action manifest",
    )
    sidecar_normalization_path = _require_artifact_sha256(
        data_root / "stats/normalization.json",
        LIBERO_12S_ACTION_NORMALIZATION_SHA256,
        name="LIBERO 1.2s action normalization",
    )
    checksums_path = _require_file(data_root / "checksums.sha256")
    if _manifest_digest(checksums_path, "manifests/all.parquet") != LIBERO_12S_MANIFEST_SHA256:
        raise ValueError("LIBERO 1.2s manifest checksum binding mismatch")

    sidecar_schema = _read_json(sidecar_schema_path)
    expected_temporal = {
        "schema_version": "libero-z-action-sidecar-v1-12s",
        "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
        "control_hz": LIBERO_12S_CONTROL_HZ,
        "duration_seconds": LIBERO_12S_DURATION_SECONDS,
    }
    if any(sidecar_schema.get(key) != value for key, value in expected_temporal.items()):
        raise ValueError("LIBERO action sidecar is not the exact 24@20Hz/1.2s release")
    if sidecar_schema.get("e39") != {
        "sampling": False,
        "semantics": "deterministic posterior mean from an explicit LIBERO Stage-1 ablation checkpoint",
        "shape": [32],
    }:
        raise ValueError("LIBERO action sidecar has incompatible effect conditioning")

    sidecar_provenance = _read_json(sidecar_provenance_path)
    e39 = sidecar_provenance.get("e39")
    if not isinstance(e39, Mapping):
        raise ValueError("LIBERO 1.2s sidecar provenance lacks e39")
    expected_e39 = {
        "checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "sha256": LIBERO_12S_EFFECT_POSTERIOR_SHA256,
        "selection_report_sha256": LIBERO_12S_EFFECT_SELECTION_SHA256,
        "selection_status": "explicit_ablation",
        "teacher": "dino_residual_v1",
    }
    if any(e39.get(key) != value for key, value in expected_e39.items()):
        raise ValueError("LIBERO 1.2s sidecar/effect-teacher provenance mismatch")
    _require_bound_path(e39.get("checkpoint"), effect_checkpoint, name="sidecar effect checkpoint")
    _require_bound_path(e39.get("path"), effect_posterior, name="sidecar effect posterior")
    _require_bound_path(e39.get("selection_report"), effect_selection, name="sidecar effect selection")

    action = torch.load(action_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(action, Mapping):
        raise TypeError("LIBERO 1.2s Action VAE checkpoint must contain a mapping")
    action_metadata = action.get("action_ae_metadata")
    if not isinstance(action_metadata, Mapping):
        raise ValueError("LIBERO 1.2s Action VAE checkpoint lacks metadata")
    if action_metadata.get("schema_version") != LIBERO_12S_ACTION_SCHEMA_VERSION:
        raise ValueError("LIBERO Action teacher is not the standard-normal beta-VAE")
    action_model_config = action_metadata.get("model_config")
    expected_action_model = {
        "variant": "a1",
        "horizon": LIBERO_12S_HORIZON_ACTIONS,
        "state_dim": 14,
        "trajectory_dim": 7,
        "latent_dim": 32,
    }
    if not isinstance(action_model_config, Mapping) or any(
        action_model_config.get(key) != value for key, value in expected_action_model.items()
    ):
        raise ValueError("LIBERO Action teacher architecture/temporal contract mismatch")
    if action_metadata.get("normalization_sha256") != LIBERO_12S_ACTION_NORMALIZATION_SHA256:
        raise ValueError("LIBERO Action teacher normalization binding mismatch")
    validation_loss = _checkpoint_validation_loss(action_checkpoint)

    return {
        "schema_version": LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION,
        "effect_checkpoint": str(effect_checkpoint),
        "effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
        "action_checkpoint": str(action_checkpoint),
        "action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
        "action_checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
        "action_prior": "standard_normal",
        "action_validation_loss": validation_loss,
        "effect_posterior": str(effect_posterior),
        "effect_posterior_sha256": LIBERO_12S_EFFECT_POSTERIOR_SHA256,
        "manifest": str(sidecar_manifest_path),
        "manifest_sha256": LIBERO_12S_MANIFEST_SHA256,
        "normalization": str(sidecar_normalization_path),
        "normalization_sha256": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
        "row_count": LIBERO_12S_ROW_COUNT,
        "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
        "control_hz": LIBERO_12S_CONTROL_HZ,
        "duration_seconds": LIBERO_12S_DURATION_SECONDS,
        "all_gates_passed": True,
    }


def _validate_explicit_ablation_selection(
    selection_path: Path,
    *,
    effect_checkpoint_path: Path,
) -> dict[str, Any]:
    selection = _read_json(selection_path)
    if selection.get("schema") != STAGE1_SELECTION_SCHEMA_VERSION:
        raise ValueError("effect ablation selection schema mismatch")
    if selection.get("selected_checkpoint_id") is not None:
        raise ValueError("effect ablation report must not contain a canonical selected checkpoint")
    if selection.get("eligible_count") != 0:
        raise ValueError("effect ablation report must have zero eligible checkpoints")
    assessments = selection.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise ValueError("effect ablation report has no checkpoint assessments")
    if selection.get("evaluated_count") != len(assessments):
        raise ValueError("effect ablation evaluated_count does not match its assessments")
    if any(not isinstance(item, Mapping) or item.get("eligible") is not False for item in assessments):
        raise ValueError("effect ablation report contains an eligible or malformed assessment")
    checkpoint_id = effect_checkpoint_path.name
    matches = [item for item in assessments if item.get("checkpoint_id") == checkpoint_id]
    if len(matches) != 1:
        raise ValueError(f"effect ablation checkpoint is not uniquely assessed: {checkpoint_id}")
    scores = []
    for assessment in assessments:
        score = float(assessment.get("geometry_score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("effect ablation assessment has a non-finite geometry score")
        scores.append(score)
    selected_score = float(matches[0]["geometry_score"])
    if selected_score != max(scores):
        raise ValueError("effect ablation checkpoint is not the highest-geometry assessed checkpoint")
    return {
        "schema": selection["schema"],
        "selected_checkpoint_id": None,
        "eligible_count": 0,
        "evaluated_count": len(assessments),
        "ablation_checkpoint_id": checkpoint_id,
        "ablation_geometry_score": selected_score,
        "highest_geometry_checkpoint_ids": sorted(
            str(item["checkpoint_id"])
            for item in assessments
            if float(item["geometry_score"]) == selected_score
        ),
    }


def build_ablation_handoff(
    *,
    output: Path,
    effect_checkpoint_path: Path,
    effect_posterior_path: Path,
    effect_selection_path: Path,
    a1_root: Path,
    action_checkpoint_path: Path,
    effect_seed: int,
    action_seed: int,
    teacher_name: str,
    ablation_rationale: str,
) -> dict[str, Any]:
    """Publish an explicit non-canonical Stage 2 teacher ablation bundle."""

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage 2 handoff: {output}")
    teacher_name = teacher_name.strip()
    rationale = ablation_rationale.strip()
    if not teacher_name or not rationale:
        raise ValueError("explicit ablation teacher name and rationale must be non-empty")

    effect_checkpoint_path = _require_file(effect_checkpoint_path)
    effect_checkpoint_sha = _sha256_file(effect_checkpoint_path)
    effect_posterior_path = _require_file(effect_posterior_path)
    effect_posterior_sha = _sha256_file(effect_posterior_path)
    effect_selection_path = _require_file(effect_selection_path)
    effect_selection_sha = _sha256_file(effect_selection_path)
    effect_selection = _validate_explicit_ablation_selection(
        effect_selection_path,
        effect_checkpoint_path=effect_checkpoint_path,
    )

    a1_root = a1_root.expanduser().resolve()
    a1_provenance_path = _require_file(a1_root / "provenance.json")
    a1_provenance = _read_json(a1_provenance_path)
    sidecar_effect = a1_provenance.get("e39")
    if not isinstance(sidecar_effect, Mapping):
        raise ValueError("A1 ablation sidecar provenance is missing e39")
    expected_sidecar = {
        "selection_status": "explicit_ablation",
        "teacher": teacher_name,
        "checkpoint_sha256": effect_checkpoint_sha,
        "sha256": effect_posterior_sha,
        "selection_report_sha256": effect_selection_sha,
        "selection_rationale": rationale,
    }
    for key, expected in expected_sidecar.items():
        if sidecar_effect.get(key) != expected:
            raise ValueError(f"A1 ablation sidecar mismatch for e39.{key}")
    path_bindings = {
        "checkpoint": effect_checkpoint_path,
        "path": effect_posterior_path,
        "selection_report": effect_selection_path,
    }
    for key, expected in path_bindings.items():
        actual = _require_file(Path(str(sidecar_effect.get(key, ""))))
        if actual != expected:
            raise ValueError(f"A1 ablation sidecar path mismatch for e39.{key}")

    normalization_path = _require_file(a1_root / "stats/normalization.json")
    manifest_path = _require_file(a1_root / "manifests/all.parquet")
    checksums_path = _require_file(a1_root / "checksums.sha256")
    manifest_sha = _manifest_digest(checksums_path, "manifests/all.parquet")
    if _sha256_file(manifest_path) != manifest_sha:
        raise ValueError("A1 ablation all-manifest hash mismatch")

    action_checkpoint_path = _require_file(action_checkpoint_path)
    action_checkpoint_sha = _sha256_file(action_checkpoint_path)
    validation_loss = _checkpoint_validation_loss(action_checkpoint_path)
    action_epoch = _checkpoint_epoch(action_checkpoint_path)
    action_config_path = _single_resolved_config(action_checkpoint_path.parents[1])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary Stage 2 handoff exists: {temporary}")
    temporary.mkdir()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        selection = {
            "schema_version": ABLATION_SELECTION_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "selection_policy": ABLATION_SELECTION_POLICY,
            "selection_split": "validation",
            "selection_monitor": SELECTION_MONITOR,
            "mode": "min",
            "test_metrics_read": False,
            "candidate_count": 1,
            "effect_ablation": {
                "teacher": teacher_name,
                "selection_status": "explicit_ablation",
                "rationale": rationale,
                "checkpoint": str(effect_checkpoint_path),
                "checkpoint_sha256": effect_checkpoint_sha,
                "selection_report": str(effect_selection_path),
                "selection_report_sha256": effect_selection_sha,
                **effect_selection,
            },
            "selected": {
                "seed": int(action_seed),
                "checkpoint": str(action_checkpoint_path),
                "checkpoint_sha256": action_checkpoint_sha,
                "config": str(action_config_path),
                "config_sha256": _sha256_file(action_config_path),
                "epoch": action_epoch,
                "selection_monitor": SELECTION_MONITOR,
                "validation_loss": validation_loss,
                "effect_checkpoint_sha256_used_for_training": effect_checkpoint_sha,
            },
        }
        selection_path = temporary / "ablation_action_selection.json"
        _write_json(selection_path, selection)
        selection_sha = _sha256_file(selection_path)
        final_selection_path = output / selection_path.name

        ablation_provenance = {
            "schema_version": "libero-stage2-explicit-effect-ablation-v1",
            "created_at_utc": created_at,
            "teacher": teacher_name,
            "selection_status": "explicit_ablation",
            "rationale": rationale,
            "canonical_stage1_selection_passed": False,
            "effect_selection": {
                "path": str(effect_selection_path),
                "sha256": effect_selection_sha,
                **effect_selection,
            },
            "sidecar_provenance": {
                "path": str(a1_provenance_path),
                "sha256": _sha256_file(a1_provenance_path),
            },
        }
        ablation_provenance_path = temporary / "explicit_ablation_provenance.json"
        _write_json(ablation_provenance_path, ablation_provenance)
        ablation_provenance_sha = _sha256_file(ablation_provenance_path)
        final_ablation_provenance_path = output / ablation_provenance_path.name

        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "handoff_mode": "explicit_ablation",
            "selection_policy": ABLATION_SELECTION_POLICY,
            "effect": {
                "seed": int(effect_seed),
                "teacher": teacher_name,
                "selection_status": "explicit_ablation",
                "selection_rationale": rationale,
                "checkpoint": str(effect_checkpoint_path),
                "checkpoint_sha256": effect_checkpoint_sha,
                "selection_json": str(effect_selection_path),
                "selection_json_sha256": effect_selection_sha,
                "selected_provenance": str(final_ablation_provenance_path),
                "selected_provenance_sha256": ablation_provenance_sha,
                "posterior_cache": str(effect_posterior_path),
                "posterior_cache_sha256": effect_posterior_sha,
            },
            "action": {
                "variant": "A1",
                "seed": int(action_seed),
                "checkpoint": str(action_checkpoint_path),
                "checkpoint_sha256": action_checkpoint_sha,
                "config": str(action_config_path),
                "config_sha256": _sha256_file(action_config_path),
                "normalization": str(normalization_path),
                "normalization_sha256": _sha256_file(normalization_path),
                "selection_json": str(final_selection_path),
                "selection_json_sha256": selection_sha,
                "selection_monitor": SELECTION_MONITOR,
                "validation_loss": validation_loss,
                "effect_checkpoint_sha256_used_for_training": effect_checkpoint_sha,
            },
            "data": {
                "a1_root": str(a1_root),
                "manifest": str(manifest_path),
                "data_manifest_sha256": manifest_sha,
                "sidecar_provenance": str(a1_provenance_path),
                "sidecar_provenance_sha256": _sha256_file(a1_provenance_path),
            },
            "stage1_ablation": {
                "provenance": str(final_ablation_provenance_path),
                "provenance_sha256": ablation_provenance_sha,
            },
        }
        bundle_path = temporary / "teacher_bundle.json"
        _write_json(bundle_path, bundle)
        bundle_sha = _sha256_file(bundle_path)
        checksum_names = (
            selection_path.name,
            ablation_provenance_path.name,
            bundle_path.name,
        )
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text(
            "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in checksum_names),
            encoding="ascii",
        )
        success = {
            "schema_version": HANDOFF_SUCCESS_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "handoff_mode": "explicit_ablation",
            "primary_action_seed": int(action_seed),
            "primary_action_checkpoint_sha256": action_checkpoint_sha,
            "canonical_effect_checkpoint_sha256": effect_checkpoint_sha,
            "effect_checkpoint_sha256": effect_checkpoint_sha,
            "selection_sha256": selection_sha,
            "teacher_bundle_sha256": bundle_sha,
            "checksums_sha256": _sha256_file(checksums_path),
            "all_gates_passed": True,
        }
        _write_json(temporary / "_SUCCESS", success)
        temporary.rename(output)
        validate_handoff(output)
        return {"output": str(output), **success}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_libero_12s_handoff(*, output: Path) -> dict[str, Any]:
    """Publish the exact selected 1.2s E39 + standard-normal Action VAE pair."""

    facts = preflight_libero_12s_teachers(verify_checkpoint_hashes=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage 2 handoff: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary Stage 2 handoff exists: {temporary}")
    temporary.mkdir()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        selection = {
            "schema_version": LIBERO_12S_SELECTION_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "selection_policy": "checkpoint_callback_minimum_validation_loss",
            "selection_split": "validation",
            "selection_monitor": SELECTION_MONITOR,
            "mode": "min",
            "test_metrics_read": False,
            "selected": {
                "checkpoint": str(LIBERO_12S_ACTION_CHECKPOINT.resolve()),
                "checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
                "checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
                "config": str(LIBERO_12S_ACTION_CONFIG.resolve()),
                "config_sha256": LIBERO_12S_ACTION_CONFIG_SHA256,
                "epoch": _checkpoint_epoch(LIBERO_12S_ACTION_CHECKPOINT),
                "validation_loss": facts["action_validation_loss"],
                "prior": "standard_normal",
                "variant": "a1",
            },
        }
        selection_path = temporary / "libero_12s_action_selection.json"
        _write_json(selection_path, selection)
        selection_sha = _sha256_file(selection_path)
        final_selection_path = output / selection_path.name

        contract = {
            "schema_version": LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION,
            "effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            "action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            "action_checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
            "action_prior": "standard_normal",
            "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
            "control_hz": LIBERO_12S_CONTROL_HZ,
            "duration_seconds": LIBERO_12S_DURATION_SECONDS,
            "row_count": LIBERO_12S_ROW_COUNT,
        }
        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "handoff_mode": "libero_12s_selected",
            "selection_policy": selection["selection_policy"],
            "teacher_contract": contract,
            "effect": {
                "teacher": "dino_residual_v1",
                "selection_status": "explicit_ablation",
                "checkpoint": str(LIBERO_12S_EFFECT_CHECKPOINT.resolve()),
                "checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
                "config": str(LIBERO_12S_EFFECT_CONFIG.resolve()),
                "config_sha256": LIBERO_12S_EFFECT_CONFIG_SHA256,
                "selection_json": str(LIBERO_12S_EFFECT_SELECTION.resolve()),
                "selection_json_sha256": LIBERO_12S_EFFECT_SELECTION_SHA256,
                "posterior_cache": str(LIBERO_12S_EFFECT_POSTERIOR.resolve()),
                "posterior_cache_sha256": LIBERO_12S_EFFECT_POSTERIOR_SHA256,
                "posterior_statistic": "mean",
            },
            "action": {
                "variant": "A1",
                "seed": 20260825,
                "prior": "standard_normal",
                "checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
                "checkpoint": str(LIBERO_12S_ACTION_CHECKPOINT.resolve()),
                "checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
                "config": str(LIBERO_12S_ACTION_CONFIG.resolve()),
                "config_sha256": LIBERO_12S_ACTION_CONFIG_SHA256,
                "normalization": str(
                    (LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json").resolve()
                ),
                "normalization_sha256": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
                "selection_json": str(final_selection_path),
                "selection_json_sha256": selection_sha,
                "selection_monitor": SELECTION_MONITOR,
                "validation_loss": facts["action_validation_loss"],
                "effect_checkpoint_sha256_used_for_training": (
                    LIBERO_12S_EFFECT_CHECKPOINT_SHA256
                ),
            },
            "data": {
                "a1_root": str(LIBERO_12S_ACTION_DATA_ROOT.resolve()),
                "manifest": str(
                    (LIBERO_12S_ACTION_DATA_ROOT / "manifests/all.parquet").resolve()
                ),
                "data_manifest_sha256": LIBERO_12S_MANIFEST_SHA256,
                "normalization": str(
                    (LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json").resolve()
                ),
                "normalization_sha256": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
                "schema": str((LIBERO_12S_ACTION_DATA_ROOT / "schema.json").resolve()),
                "provenance": str(
                    (LIBERO_12S_ACTION_DATA_ROOT / "provenance.json").resolve()
                ),
                "row_count": LIBERO_12S_ROW_COUNT,
                "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
                "control_hz": LIBERO_12S_CONTROL_HZ,
                "duration_seconds": LIBERO_12S_DURATION_SECONDS,
            },
        }
        bundle_path = temporary / "teacher_bundle.json"
        _write_json(bundle_path, bundle)
        bundle_sha = _sha256_file(bundle_path)
        checksum_names = (selection_path.name, bundle_path.name)
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text(
            "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in checksum_names),
            encoding="ascii",
        )
        success = {
            "schema_version": HANDOFF_SUCCESS_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "handoff_mode": "libero_12s_selected",
            "primary_action_seed": 20260825,
            "primary_action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            "canonical_effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            "effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            "selection_sha256": selection_sha,
            "teacher_bundle_sha256": bundle_sha,
            "checksums_sha256": _sha256_file(checksums_path),
            "all_gates_passed": True,
        }
        _write_json(temporary / "_SUCCESS", success)
        temporary.rename(output)
        validate_handoff(output)
        return {"output": str(output), **success}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_handoff(
    *,
    output: Path,
    release_provenance_path: Path,
    a1_root: Path,
    verify_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Atomically publish primary selection plus the immutable teacher bundle."""

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage 2 handoff: {output}")
    release_provenance_path = _require_file(release_provenance_path)
    release = _read_json(release_provenance_path)
    release_sha = _sha256_file(release_provenance_path)
    a1_root = a1_root.expanduser().resolve()
    a1_provenance_path = _require_file(a1_root / "provenance.json")
    a1_provenance = _read_json(a1_provenance_path)
    normalization_path = _require_file(a1_root / "stats/normalization.json")
    manifest_path = _require_file(a1_root / "manifests/all.parquet")
    checksums_path = _require_file(a1_root / "checksums.sha256")
    manifest_sha = _manifest_digest(checksums_path, "manifests/all.parquet")
    if _sha256_file(manifest_path) != manifest_sha:
        raise ValueError("A1 all-manifest hash mismatch")

    selection = select_primary_action_teacher(
        release,
        verify_checkpoint_hashes=verify_checkpoint_hashes,
    )
    selected = selection["selected"]
    effect = release["effect_encoder"]
    effect_checkpoint_path = _require_file(Path(effect["canonical_checkpoint"]))
    effect_checkpoint_sha = _require_sha256(
        effect["canonical_checkpoint_sha256"], name="canonical effect checkpoint"
    )
    posterior_path = _verify_small_artifact(
        Path(effect["posterior_cache"]),
        effect["posterior_cache_sha256"],
        name="canonical effect posterior",
    )
    effect_selection_path = _require_file(effect_checkpoint_path.parents[1] / "checkpoint_selection.json")
    selected_effect_provenance_path = _require_file(
        effect_checkpoint_path.parents[1] / "selected_effect_provenance.json"
    )

    sidecar_effect = a1_provenance.get("e39")
    if not isinstance(sidecar_effect, Mapping):
        raise ValueError("A1 sidecar provenance is missing e39")
    if _require_sha256(
        sidecar_effect.get("checkpoint_sha256"), name="A1 sidecar effect checkpoint"
    ) != effect_checkpoint_sha:
        raise ValueError("A1 sidecar and selected Effect Encoder are incompatible")
    if _require_sha256(
        sidecar_effect.get("sha256"), name="A1 sidecar effect posterior"
    ) != _sha256_file(posterior_path):
        raise ValueError("A1 sidecar and selected effect posterior are incompatible")
    if selected["effect_checkpoint_sha256_used_for_training"] != effect_checkpoint_sha:
        raise ValueError("primary A1 candidate and selected Effect Encoder are incompatible")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary Stage 2 handoff exists: {temporary}")
    temporary.mkdir()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        selection = {**selection, "created_at_utc": created_at}
        selection_path = temporary / "primary_action_selection.json"
        _write_json(selection_path, selection)
        selection_sha = _sha256_file(selection_path)
        final_selection_path = output / selection_path.name
        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "selection_policy": "validation_only",
            "effect": {
                "seed": int(effect["canonical_seed"]),
                "checkpoint": str(effect_checkpoint_path),
                "checkpoint_sha256": effect_checkpoint_sha,
                "selection_json": str(effect_selection_path),
                "selection_json_sha256": _sha256_file(effect_selection_path),
                "selected_provenance": str(selected_effect_provenance_path),
                "selected_provenance_sha256": _sha256_file(selected_effect_provenance_path),
                "posterior_cache": str(posterior_path),
                "posterior_cache_sha256": _sha256_file(posterior_path),
            },
            "action": {
                "variant": "A1",
                "seed": int(selected["seed"]),
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "config": selected["config"],
                "config_sha256": selected["config_sha256"],
                "normalization": str(normalization_path),
                "normalization_sha256": _sha256_file(normalization_path),
                "selection_json": str(final_selection_path),
                "selection_json_sha256": selection_sha,
                "selection_monitor": SELECTION_MONITOR,
                "validation_loss": selected["validation_loss"],
                "effect_checkpoint_sha256_used_for_training": selected[
                    "effect_checkpoint_sha256_used_for_training"
                ],
            },
            "data": {
                "a1_root": str(a1_root),
                "manifest": str(manifest_path),
                "data_manifest_sha256": manifest_sha,
                "sidecar_provenance": str(a1_provenance_path),
                "sidecar_provenance_sha256": _sha256_file(a1_provenance_path),
            },
            "stage1_release": {
                "provenance": str(release_provenance_path),
                "provenance_sha256": release_sha,
            },
        }
        bundle_path = temporary / "teacher_bundle.json"
        _write_json(bundle_path, bundle)
        bundle_sha = _sha256_file(bundle_path)
        checksums = (
            f"{selection_sha}  {selection_path.name}\n"
            f"{bundle_sha}  {bundle_path.name}\n"
        )
        (temporary / "checksums.sha256").write_text(checksums, encoding="utf-8")
        success = {
            "schema_version": HANDOFF_SUCCESS_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "primary_action_seed": int(selected["seed"]),
            "primary_action_checkpoint_sha256": selected["checkpoint_sha256"],
            "canonical_effect_checkpoint_sha256": effect_checkpoint_sha,
            "selection_sha256": selection_sha,
            "teacher_bundle_sha256": bundle_sha,
            "checksums_sha256": _sha256_file(temporary / "checksums.sha256"),
            "all_gates_passed": True,
        }
        _write_json(temporary / "_SUCCESS", success)
        temporary.rename(output)
        validate_handoff(output)
        return {"output": str(output), **success}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_handoff(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    success = _read_json(_require_file(root / "_SUCCESS"))
    if success.get("schema_version") != HANDOFF_SUCCESS_SCHEMA_VERSION:
        raise ValueError("Stage 2 handoff success schema mismatch")
    if success.get("all_gates_passed") is not True:
        raise ValueError("Stage 2 handoff did not pass all gates")
    checksums_path = _require_file(root / "checksums.sha256")
    if _sha256_file(checksums_path) != success.get("checksums_sha256"):
        raise ValueError("Stage 2 handoff checksum manifest hash mismatch")
    checked = 0
    for line_number, raw in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative = raw.partition("  ")
        if not separator or Path(relative).is_absolute():
            raise ValueError(f"invalid handoff checksum line {line_number}")
        path = _require_file(root / relative)
        if _sha256_file(path) != _require_sha256(digest, name=f"handoff line {line_number}"):
            raise ValueError(f"Stage 2 handoff artifact hash mismatch: {path}")
        checked += 1
    bundle_path = _require_file(root / "teacher_bundle.json")
    if _sha256_file(bundle_path) != success.get("teacher_bundle_sha256"):
        raise ValueError("Stage 2 handoff teacher bundle hash mismatch")
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("teacher bundle schema mismatch")
    handoff_mode = str(bundle.get("handoff_mode", "canonical_release"))
    if handoff_mode not in {
        "canonical_release",
        "explicit_ablation",
        "libero_12s_selected",
    }:
        raise ValueError(f"unsupported teacher handoff mode: {handoff_mode}")
    if bundle["effect"]["checkpoint_sha256"] != bundle["action"][
        "effect_checkpoint_sha256_used_for_training"
    ]:
        raise ValueError("teacher bundle mixes incompatible effect and action teachers")
    if bundle["action"]["selection_json_sha256"] != success["selection_sha256"]:
        raise ValueError("teacher bundle selection hash mismatch")
    if handoff_mode == "explicit_ablation":
        if success.get("handoff_mode") != "explicit_ablation":
            raise ValueError("explicit ablation handoff success marker mismatch")
        effect = bundle["effect"]
        if effect.get("selection_status") != "explicit_ablation":
            raise ValueError("explicit ablation bundle lacks an explicit effect status")
        if bundle.get("selection_policy") != ABLATION_SELECTION_POLICY:
            raise ValueError("explicit ablation bundle selection policy mismatch")
        if not str(effect.get("teacher", "")).strip() or not str(
            effect.get("selection_rationale", "")
        ).strip():
            raise ValueError("explicit ablation bundle lacks teacher semantics or rationale")
        action_selection = _read_json(_require_file(Path(bundle["action"]["selection_json"])))
        if action_selection.get("schema_version") != ABLATION_SELECTION_SCHEMA_VERSION:
            raise ValueError("explicit ablation action selection schema mismatch")
        effect_selection = action_selection.get("effect_ablation")
        if not isinstance(effect_selection, Mapping):
            raise ValueError("explicit ablation action selection lacks effect evidence")
        if effect_selection.get("selected_checkpoint_id") is not None:
            raise ValueError("explicit ablation action selection claims a canonical effect")
        if effect_selection.get("eligible_count") != 0:
            raise ValueError("explicit ablation action selection has eligible effect checkpoints")
        if effect_selection.get("checkpoint_sha256") != effect["checkpoint_sha256"]:
            raise ValueError("explicit ablation effect selection hash mismatch")
    elif handoff_mode == "libero_12s_selected":
        if success.get("handoff_mode") != "libero_12s_selected":
            raise ValueError("LIBERO 1.2s handoff success marker mismatch")
        expected_contract = {
            "schema_version": LIBERO_12S_TEACHER_CONTRACT_SCHEMA_VERSION,
            "effect_checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            "action_checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            "action_checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
            "action_prior": "standard_normal",
            "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
            "control_hz": LIBERO_12S_CONTROL_HZ,
            "duration_seconds": LIBERO_12S_DURATION_SECONDS,
            "row_count": LIBERO_12S_ROW_COUNT,
        }
        if bundle.get("teacher_contract") != expected_contract:
            raise ValueError("LIBERO 1.2s teacher contract mismatch")
        effect = bundle.get("effect")
        action = bundle.get("action")
        data = bundle.get("data")
        if not all(isinstance(value, Mapping) for value in (effect, action, data)):
            raise ValueError("LIBERO 1.2s bundle is incomplete")
        expected_effect = {
            "checkpoint_sha256": LIBERO_12S_EFFECT_CHECKPOINT_SHA256,
            "config_sha256": LIBERO_12S_EFFECT_CONFIG_SHA256,
            "posterior_cache_sha256": LIBERO_12S_EFFECT_POSTERIOR_SHA256,
            "selection_json_sha256": LIBERO_12S_EFFECT_SELECTION_SHA256,
            "posterior_statistic": "mean",
            "selection_status": "explicit_ablation",
        }
        if any(effect.get(key) != value for key, value in expected_effect.items()):
            raise ValueError("LIBERO 1.2s effect bundle mismatch")
        expected_action = {
            "checkpoint_sha256": LIBERO_12S_ACTION_CHECKPOINT_SHA256,
            "config_sha256": LIBERO_12S_ACTION_CONFIG_SHA256,
            "normalization_sha256": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
            "checkpoint_schema_version": LIBERO_12S_ACTION_SCHEMA_VERSION,
            "prior": "standard_normal",
            "variant": "A1",
            "effect_checkpoint_sha256_used_for_training": (
                LIBERO_12S_EFFECT_CHECKPOINT_SHA256
            ),
        }
        if any(action.get(key) != value for key, value in expected_action.items()):
            raise ValueError("LIBERO 1.2s action bundle mismatch")
        expected_data = {
            "data_manifest_sha256": LIBERO_12S_MANIFEST_SHA256,
            "normalization_sha256": LIBERO_12S_ACTION_NORMALIZATION_SHA256,
            "row_count": LIBERO_12S_ROW_COUNT,
            "horizon_actions": LIBERO_12S_HORIZON_ACTIONS,
            "control_hz": LIBERO_12S_CONTROL_HZ,
            "duration_seconds": LIBERO_12S_DURATION_SECONDS,
        }
        if any(data.get(key) != value for key, value in expected_data.items()):
            raise ValueError("LIBERO 1.2s data bundle mismatch")
        action_selection = _read_json(_require_file(Path(str(action["selection_json"]))))
        if (
            action_selection.get("schema_version") != LIBERO_12S_SELECTION_SCHEMA_VERSION
            or action_selection.get("test_metrics_read") is not False
            or action_selection.get("selected", {}).get("checkpoint_sha256")
            != LIBERO_12S_ACTION_CHECKPOINT_SHA256
        ):
            raise ValueError("LIBERO 1.2s action selection mismatch")
        preflight_libero_12s_teachers(verify_checkpoint_hashes=True)
    return {
        "root": str(root),
        "checked_artifacts": checked,
        "handoff_mode": handoff_mode,
        "primary_action_seed": int(bundle["action"]["seed"]),
        "all_gates_passed": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--release-provenance", required=True, type=Path)
    build_parser.add_argument("--a1-root", required=True, type=Path)
    ablation_parser = subparsers.add_parser("build-ablation")
    ablation_parser.add_argument("--output", required=True, type=Path)
    ablation_parser.add_argument("--effect-checkpoint", required=True, type=Path)
    ablation_parser.add_argument("--effect-posterior", required=True, type=Path)
    ablation_parser.add_argument("--effect-selection", required=True, type=Path)
    ablation_parser.add_argument("--a1-root", required=True, type=Path)
    ablation_parser.add_argument("--action-checkpoint", required=True, type=Path)
    ablation_parser.add_argument("--effect-seed", required=True, type=int)
    ablation_parser.add_argument("--action-seed", required=True, type=int)
    ablation_parser.add_argument("--teacher-name", required=True)
    ablation_parser.add_argument("--ablation-rationale", required=True)
    selected_12s_parser = subparsers.add_parser("build-libero-12s")
    selected_12s_parser.add_argument("--output", required=True, type=Path)
    subparsers.add_parser("preflight-libero-12s")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        result = build_handoff(
            output=args.output,
            release_provenance_path=args.release_provenance,
            a1_root=args.a1_root,
        )
    elif args.command == "build-ablation":
        result = build_ablation_handoff(
            output=args.output,
            effect_checkpoint_path=args.effect_checkpoint,
            effect_posterior_path=args.effect_posterior,
            effect_selection_path=args.effect_selection,
            a1_root=args.a1_root,
            action_checkpoint_path=args.action_checkpoint,
            effect_seed=args.effect_seed,
            action_seed=args.action_seed,
            teacher_name=args.teacher_name,
            ablation_rationale=args.ablation_rationale,
        )
    elif args.command == "build-libero-12s":
        result = build_libero_12s_handoff(output=args.output)
    elif args.command == "preflight-libero-12s":
        result = preflight_libero_12s_teachers(verify_checkpoint_hashes=True)
    else:
        result = validate_handoff(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
