"""Release-relative gates and deterministic Stage-1 checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evaluation import is_posterior_collapsed


LWM_RELEASE_FACTOR = 1.05
STATE_RELEASE_FACTOR = 1.10
EFFECTIVE_RANK_RELEASE_FACTOR = 0.70
EFFECTIVE_RANK_ABSOLUTE_FLOOR = 4.0
SHUFFLE_GAIN_FLOOR = 0.10
TRACE_RATIO_FLOOR = 0.50
MEDIAN_DISTANCE_RATIO_FLOOR = 0.70


def _finite(value: float, *, name: str, non_negative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if non_negative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class ReleaseBaseline:
    """Metrics computed from the release checkpoint on the identical val set."""

    lwm_loss: float
    state_loss: float
    effective_rank: float
    covariance_trace: float
    pairwise_median_distance: float


@dataclass(frozen=True)
class CheckpointMetrics:
    """All scalar inputs needed by the locked two-stage selection rule."""

    checkpoint_id: str
    lwm_loss: float
    state_loss: float
    effective_rank: float
    covariance_trace: float
    pairwise_median_distance: float
    active_dimensions: int
    mean_kl_per_dim: float
    shuffle_gain: float
    overlap_at_16: float
    trustworthiness_at_16: float
    rho_local: float
    epoch: int | None = None
    shuffle_ci_low: float | None = None
    shuffle_ci_high: float | None = None


@dataclass(frozen=True)
class ReleaseComparison:
    lwm_limit: float
    state_limit: float
    effective_rank_floor: float
    covariance_trace_ratio: float
    pairwise_median_distance_ratio: float


@dataclass(frozen=True)
class CheckpointAssessment:
    metrics: CheckpointMetrics
    release_comparison: ReleaseComparison
    geometry_score: float
    posterior_collapsed: bool
    eligible: bool
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointSelection:
    """Selection result plus the complete audit trail for rejected checkpoints."""

    selected: CheckpointAssessment | None
    ranked_eligible: tuple[CheckpointAssessment, ...]
    assessments: tuple[CheckpointAssessment, ...]


def _validate_release(release: ReleaseBaseline) -> ReleaseBaseline:
    lwm_loss = _finite(release.lwm_loss, name="release.lwm_loss", non_negative=True)
    state_loss = _finite(release.state_loss, name="release.state_loss", non_negative=True)
    effective_rank = _finite(release.effective_rank, name="release.effective_rank", non_negative=True)
    covariance_trace = _finite(
        release.covariance_trace, name="release.covariance_trace", non_negative=True
    )
    median_distance = _finite(
        release.pairwise_median_distance,
        name="release.pairwise_median_distance",
        non_negative=True,
    )
    if effective_rank <= 0:
        raise ValueError("release.effective_rank must be positive")
    # These values are ratio denominators.  Silently adding epsilon would make
    # a collapsed release baseline look meaningful, so reject it explicitly.
    if covariance_trace <= 0:
        raise ValueError("release.covariance_trace must be positive")
    if median_distance <= 0:
        raise ValueError("release.pairwise_median_distance must be positive")
    return ReleaseBaseline(
        lwm_loss=lwm_loss,
        state_loss=state_loss,
        effective_rank=effective_rank,
        covariance_trace=covariance_trace,
        pairwise_median_distance=median_distance,
    )


def _validate_metrics(metrics: CheckpointMetrics) -> CheckpointMetrics:
    if not isinstance(metrics.checkpoint_id, str) or not metrics.checkpoint_id:
        raise ValueError("checkpoint_id must be a non-empty string")
    if not isinstance(metrics.active_dimensions, (int, np.integer)) or metrics.active_dimensions < 0:
        raise ValueError("active_dimensions must be a non-negative integer")
    if metrics.epoch is not None and (not isinstance(metrics.epoch, (int, np.integer)) or metrics.epoch < 0):
        raise ValueError("epoch must be a non-negative integer or None")

    numeric_non_negative = {
        "lwm_loss": metrics.lwm_loss,
        "state_loss": metrics.state_loss,
        "effective_rank": metrics.effective_rank,
        "covariance_trace": metrics.covariance_trace,
        "pairwise_median_distance": metrics.pairwise_median_distance,
        "mean_kl_per_dim": metrics.mean_kl_per_dim,
    }
    validated = {
        name: _finite(value, name=f"checkpoint.{name}", non_negative=True)
        for name, value in numeric_non_negative.items()
    }
    shuffle_gain = _finite(metrics.shuffle_gain, name="checkpoint.shuffle_gain")
    overlap = _finite(metrics.overlap_at_16, name="checkpoint.overlap_at_16")
    trustworthiness = _finite(
        metrics.trustworthiness_at_16, name="checkpoint.trustworthiness_at_16"
    )
    rho_local = _finite(metrics.rho_local, name="checkpoint.rho_local")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap_at_16 must be in [0, 1]")
    if not 0.0 <= trustworthiness <= 1.0:
        raise ValueError("trustworthiness_at_16 must be in [0, 1]")
    if not -1.0 <= rho_local <= 1.0:
        raise ValueError("rho_local must be in [-1, 1]")

    ci_low = None
    ci_high = None
    if (metrics.shuffle_ci_low is None) != (metrics.shuffle_ci_high is None):
        raise ValueError("shuffle CI bounds must either both be set or both be None")
    if metrics.shuffle_ci_low is not None:
        ci_low = _finite(metrics.shuffle_ci_low, name="checkpoint.shuffle_ci_low")
        ci_high = _finite(metrics.shuffle_ci_high, name="checkpoint.shuffle_ci_high")
        if ci_low > ci_high:
            raise ValueError("shuffle_ci_low cannot exceed shuffle_ci_high")

    return CheckpointMetrics(
        checkpoint_id=metrics.checkpoint_id,
        lwm_loss=validated["lwm_loss"],
        state_loss=validated["state_loss"],
        effective_rank=validated["effective_rank"],
        covariance_trace=validated["covariance_trace"],
        pairwise_median_distance=validated["pairwise_median_distance"],
        active_dimensions=int(metrics.active_dimensions),
        mean_kl_per_dim=validated["mean_kl_per_dim"],
        shuffle_gain=shuffle_gain,
        overlap_at_16=overlap,
        trustworthiness_at_16=trustworthiness,
        rho_local=rho_local,
        epoch=None if metrics.epoch is None else int(metrics.epoch),
        shuffle_ci_low=ci_low,
        shuffle_ci_high=ci_high,
    )


def geometry_score(*, overlap_at_16: float, trustworthiness_at_16: float, rho_local: float) -> float:
    """Compute the locked macro geometry objective."""

    overlap = _finite(overlap_at_16, name="overlap_at_16")
    trustworthiness = _finite(trustworthiness_at_16, name="trustworthiness_at_16")
    rho = _finite(rho_local, name="rho_local")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap_at_16 must be in [0, 1]")
    if not 0.0 <= trustworthiness <= 1.0:
        raise ValueError("trustworthiness_at_16 must be in [0, 1]")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho_local must be in [-1, 1]")
    return float(0.5 * overlap + 0.25 * trustworthiness + 0.25 * ((rho + 1.0) / 2.0))


def compare_to_release(
    metrics: CheckpointMetrics,
    release: ReleaseBaseline,
) -> ReleaseComparison:
    """Return every release-relative threshold/ratio used by hard gates."""

    validated_metrics = _validate_metrics(metrics)
    validated_release = _validate_release(release)
    return ReleaseComparison(
        lwm_limit=LWM_RELEASE_FACTOR * validated_release.lwm_loss,
        state_limit=STATE_RELEASE_FACTOR * validated_release.state_loss,
        effective_rank_floor=max(
            EFFECTIVE_RANK_ABSOLUTE_FLOOR,
            EFFECTIVE_RANK_RELEASE_FACTOR * validated_release.effective_rank,
        ),
        covariance_trace_ratio=(
            validated_metrics.covariance_trace / validated_release.covariance_trace
        ),
        pairwise_median_distance_ratio=(
            validated_metrics.pairwise_median_distance
            / validated_release.pairwise_median_distance
        ),
    )


def assess_checkpoint(
    metrics: CheckpointMetrics,
    release: ReleaseBaseline,
) -> CheckpointAssessment:
    """Apply all hard feasibility gates before geometry ranking."""

    metrics = _validate_metrics(metrics)
    release = _validate_release(release)
    comparison = compare_to_release(metrics, release)
    posterior_collapsed = is_posterior_collapsed(
        active_dimensions=metrics.active_dimensions,
        mean_kl_per_dim=metrics.mean_kl_per_dim,
        shuffle_gain=metrics.shuffle_gain,
    )

    failed: list[str] = []
    if metrics.lwm_loss > comparison.lwm_limit:
        failed.append("lwm_vs_release")
    if metrics.state_loss > comparison.state_limit:
        failed.append("state_vs_release")
    if metrics.effective_rank < comparison.effective_rank_floor:
        failed.append("effective_rank")
    if metrics.shuffle_gain < SHUFFLE_GAIN_FLOOR:
        failed.append("shuffle_gain")
    if comparison.covariance_trace_ratio < TRACE_RATIO_FLOOR:
        failed.append("mean_contraction_trace")
    if comparison.pairwise_median_distance_ratio < MEDIAN_DISTANCE_RATIO_FLOOR:
        failed.append("mean_contraction_distance")
    if posterior_collapsed:
        failed.append("posterior_collapse")

    score = geometry_score(
        overlap_at_16=metrics.overlap_at_16,
        trustworthiness_at_16=metrics.trustworthiness_at_16,
        rho_local=metrics.rho_local,
    )
    return CheckpointAssessment(
        metrics=metrics,
        release_comparison=comparison,
        geometry_score=score,
        posterior_collapsed=posterior_collapsed,
        eligible=not failed,
        failed_gates=tuple(failed),
    )


def select_best_checkpoint(
    checkpoints: Sequence[CheckpointMetrics],
    release: ReleaseBaseline,
) -> CheckpointSelection:
    """Select max ``S_geo``; exact score ties use lower validation ``L_wm``."""

    if not checkpoints:
        raise ValueError("checkpoints cannot be empty")
    checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in checkpoints]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint_id values must be unique")

    assessments = tuple(assess_checkpoint(checkpoint, release) for checkpoint in checkpoints)
    eligible = [assessment for assessment in assessments if assessment.eligible]
    eligible.sort(
        key=lambda assessment: (
            -assessment.geometry_score,
            assessment.metrics.lwm_loss,
            assessment.metrics.checkpoint_id,
        )
    )
    ranked = tuple(eligible)
    return CheckpointSelection(
        selected=ranked[0] if ranked else None,
        ranked_eligible=ranked,
        assessments=assessments,
    )
