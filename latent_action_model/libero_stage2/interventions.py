"""Deterministic latent normalization and strict intervention routing."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


LATENT_NORMALIZATION_SCHEMA_VERSION = "libero-stage2-latent-standardization-v1"
STRICT_SHUFFLE_SCHEMA_VERSION = "libero-stage2-strict-shuffle-v1"
CONDITION_KEYS = {
    "future": "h_t1_star",
    "effect": "z_effect",
    "action": "z_action",
}


@dataclass(frozen=True)
class LatentStandardizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fit_split: str = "train"
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape:
            raise ValueError("latent mean/std must be non-empty vectors with identical shape")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0.0):
            raise ValueError("latent mean/std must be finite and std must be non-negative")
        if self.fit_split != "train":
            raise ValueError("latent standardization must be fit on train")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("latent epsilon must be finite and positive")

    @classmethod
    def fit(
        cls,
        values: Any,
        *,
        split: str,
        epsilon: float = 1e-6,
    ) -> "LatentStandardizer":
        if split != "train":
            raise ValueError("latent standardizer fitting is restricted to train")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
            raise ValueError("latent fit values must be finite [N,D] with N>=2")
        return cls(
            mean=tuple(array.mean(axis=0).tolist()),
            std=tuple(array.std(axis=0, ddof=0).tolist()),
            fit_split=split,
            epsilon=float(epsilon),
        )

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        if array.shape[-1] != mean.size or not np.isfinite(array).all():
            raise ValueError(f"latent values must be finite with final dimension {mean.size}")
        return ((array - mean) / np.maximum(std, self.epsilon)).astype(np.float32)

    def inverse(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        if array.shape[-1] != mean.size or not np.isfinite(array).all():
            raise ValueError(f"normalized latent values must be finite with final dimension {mean.size}")
        return (array * np.maximum(std, self.epsilon) + mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        std = np.asarray(self.std)
        return {
            "schema_version": LATENT_NORMALIZATION_SCHEMA_VERSION,
            **asdict(self),
            "dimension": len(self.mean),
            "near_zero_std_dimensions": np.flatnonzero(std < self.epsilon).astype(int).tolist(),
            "zero_normalized_semantics": "train_mean",
        }


def _candidate_key(seed: int, sample_id: str, partner_id: str) -> bytes:
    return hashlib.blake2b(
        f"{int(seed)}\0{sample_id}\0{partner_id}".encode("utf-8"),
        digest_size=16,
    ).digest()


def build_strict_shuffle_mapping(rows: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    required = {
        "sample_id",
        "split",
        "task_id",
        "episode_id",
        "start_step",
        "end_step",
        "progress_bin",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"strict shuffle input is missing columns: {missing}")
    frame = rows.copy().reset_index(drop=True)
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError("strict shuffle input contains duplicate sample IDs")
    frame["sample_id"] = frame.sample_id.astype(str)
    frame["split"] = frame.split.astype(str)
    frame["task_id"] = frame.task_id.astype(str)
    frame["episode_id"] = frame.episode_id.astype(int)
    frame["start_step"] = frame.start_step.astype(int)
    frame["end_step"] = frame.end_step.astype(int)
    frame["progress_bin"] = frame.progress_bin.astype(int)
    frame["horizon"] = frame.end_step - frame.start_step
    if bool((frame.horizon <= 0).any()):
        raise ValueError("strict shuffle input contains a non-positive horizon")

    mappings: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        candidates = frame[
            (frame.split == row.split)
            & (frame.task_id == row.task_id)
            & (frame.episode_id != row.episode_id)
            & (frame.horizon == row.horizon)
        ]
        if "valid_signature" in frame.columns:
            candidates = candidates[candidates.valid_signature.astype(str) == str(row.valid_signature)]
        if candidates.empty:
            raise ValueError(
                f"sample {row.sample_id} has no same-split/task, different-episode, compatible-horizon partner"
            )
        distance = (candidates.progress_bin - int(row.progress_bin)).abs()
        minimum_distance = int(distance.min())
        nearest = candidates[distance == minimum_distance]
        partner = min(
            nearest.itertuples(index=False),
            key=lambda item: (_candidate_key(seed, row.sample_id, item.sample_id), item.sample_id),
        )
        mappings.append(
            {
                "sample_id": row.sample_id,
                "partner_sample_id": partner.sample_id,
                "split": row.split,
                "task_id": row.task_id,
                "episode_id": int(row.episode_id),
                "partner_episode_id": int(partner.episode_id),
                "progress_bin": int(row.progress_bin),
                "partner_progress_bin": int(partner.progress_bin),
                "horizon": int(row.horizon),
                "relaxation_level": (
                    "exact_progress_bin" if minimum_distance == 0 else f"nearest_progress_bin_distance_{minimum_distance}"
                ),
                "seed": int(seed),
                "schema_version": STRICT_SHUFFLE_SCHEMA_VERSION,
            }
        )
    mapping = pd.DataFrame(mappings)
    validate_strict_shuffle_mapping(frame, mapping)
    return mapping


def validate_strict_shuffle_mapping(rows: pd.DataFrame, mapping: pd.DataFrame) -> None:
    if mapping.sample_id.astype(str).duplicated().any():
        raise ValueError("strict shuffle mapping contains duplicate source IDs")
    if set(mapping.sample_id.astype(str)) != set(rows.sample_id.astype(str)):
        raise ValueError("strict shuffle mapping source IDs differ from the input manifest")
    indexed = rows.copy()
    indexed["sample_id"] = indexed.sample_id.astype(str)
    indexed = indexed.set_index("sample_id", drop=False)
    for item in mapping.itertuples(index=False):
        source = indexed.loc[str(item.sample_id)]
        partner = indexed.loc[str(item.partner_sample_id)]
        if str(source.sample_id) == str(partner.sample_id):
            raise ValueError(f"strict shuffle self-partner: {source.sample_id}")
        if str(source.split) != str(partner.split) or str(source.task_id) != str(partner.task_id):
            raise ValueError(f"strict shuffle split/task mismatch: {source.sample_id}")
        if int(source.episode_id) == int(partner.episode_id):
            raise ValueError(f"strict shuffle same-episode partner: {source.sample_id}")
        if int(source.end_step) - int(source.start_step) != int(partner.end_step) - int(partner.start_step):
            raise ValueError(f"strict shuffle horizon mismatch: {source.sample_id}")
        if "valid_signature" in indexed.columns and str(source.valid_signature) != str(partner.valid_signature):
            raise ValueError(f"strict shuffle valid-mask mismatch: {source.sample_id}")


def apply_condition_intervention(
    view: Mapping[str, Any],
    *,
    condition: str,
    replacement: Any,
    intervention: str,
    source_sample_id: str | None = None,
) -> dict[str, Any]:
    try:
        key = CONDITION_KEYS[str(condition)]
    except KeyError as exc:
        raise ValueError(f"unknown condition {condition!r}; expected one of {sorted(CONDITION_KEYS)}") from exc
    if key not in view:
        raise KeyError(f"condition {condition!r} is not visible in this arm view")
    result = dict(view)
    result[key] = replacement
    result["intervention"] = str(intervention)
    result["intervention_condition"] = str(condition)
    if source_sample_id is not None:
        result["intervention_source_sample_id"] = str(source_sample_id)
    changed_model_keys = [
        candidate
        for candidate in CONDITION_KEYS.values()
        if candidate in view and result[candidate] is not view[candidate]
    ]
    if changed_model_keys != [key]:
        raise AssertionError(
            f"intervention route changed {changed_model_keys}; expected only {key}"
        )
    return result
