"""Apply release-relative hard gates and select the best Stage-1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from latent_action_model.stage1.checkpoint_selection import (
    CheckpointMetrics,
    ReleaseBaseline,
    geometry_score,
    select_best_checkpoint,
)


LIBERO_TASK_COVERAGE_FLOOR = 0.70
LIBERO_BOOTSTRAP_SEED = 20260803
LIBERO_BOOTSTRAP_SAMPLES = 10_000


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lawam-stage1-checkpoint-evaluation-v2":
        raise ValueError(f"evaluation schema mismatch: {path}")
    return payload


def _release(payload: dict[str, Any]) -> ReleaseBaseline:
    values = payload["release_baseline"]
    return ReleaseBaseline(
        lwm_loss=values["lwm_loss"],
        state_loss=values["state_loss"],
        effective_rank=values["effective_rank"],
        covariance_trace=values["covariance_trace"],
        pairwise_median_distance=values["pairwise_median_distance"],
    )


def _checkpoint(path: Path, payload: dict[str, Any]) -> CheckpointMetrics:
    natural = payload["natural"]
    latent = payload["latent"]
    shuffle = payload["shuffle"]
    geometry = payload["geometry"]
    return CheckpointMetrics(
        checkpoint_id=str(payload.get("checkpoint_id") or Path(payload["checkpoint"]).name),
        lwm_loss=natural["lwm_loss"],
        state_loss=natural["state_loss"],
        effective_rank=latent["effective_rank"],
        covariance_trace=latent["covariance_trace"],
        pairwise_median_distance=latent["pairwise_median_distance"],
        active_dimensions=latent["active_dimensions"],
        mean_kl_per_dim=latent["mean_kl_per_dim"],
        shuffle_gain=shuffle["gain"],
        overlap_at_16=geometry["overlap_at_k"],
        trustworthiness_at_16=geometry["trustworthiness_at_k"],
        rho_local=geometry["rho_local"],
        epoch=payload.get("epoch"),
        shuffle_ci_low=shuffle["bootstrap_ci_low"],
        shuffle_ci_high=shuffle["bootstrap_ci_high"],
    )


def _family_scores(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for family, metrics in payload["geometry"]["per_family"].items():
        result[str(family)] = geometry_score(
            overlap_at_16=metrics["overlap_at_k"],
            trustworthiness_at_16=metrics["trustworthiness_at_k"],
            rho_local=metrics["rho_local"],
        )
    return result


def _paired_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int = LIBERO_BOOTSTRAP_SEED,
    samples: int = LIBERO_BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    if deltas.ndim != 1 or len(deltas) < 2 or not np.isfinite(deltas).all():
        raise ValueError("task geometry deltas must be a finite vector with at least two rows")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start : start + count] = np.mean(deltas[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _libero_geometry_comparison(
    release: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if release.get("split", "validation") != "validation" or candidate.get("split") != "validation":
        raise ValueError("LIBERO checkpoint selection accepts validation evaluations only")
    release_k = int(release["geometry"].get("geometry_k", -1))
    candidate_k = int(candidate["geometry"].get("geometry_k", -1))
    if release_k != 8 or candidate_k != release_k:
        raise ValueError("release and candidate must both use LIBERO geometry_k=8")
    release_ids = release["geometry"].get("anchor_sample_ids")
    candidate_ids = candidate["geometry"].get("anchor_sample_ids")
    if not release_ids or release_ids != candidate_ids:
        raise ValueError("release and candidate anchor_sample_ids must be identical and non-empty")

    release_scores = _family_scores(release)
    candidate_scores = _family_scores(candidate)
    if set(release_scores) != set(candidate_scores):
        raise ValueError("release and candidate geometry families differ")
    families = sorted(release_scores)
    deltas = np.asarray(
        [candidate_scores[family] - release_scores[family] for family in families],
        dtype=np.float64,
    )
    ci_low, ci_high = _paired_bootstrap(deltas)
    release_geometry = release["geometry"]
    candidate_geometry = candidate["geometry"]
    checks = {
        "score_improved": float(candidate_geometry["score"]) > float(release_geometry["score"]),
        "overlap_not_degraded": (
            float(candidate_geometry["overlap_at_k"])
            >= float(release_geometry["overlap_at_k"])
        ),
        "trustworthiness_not_degraded": (
            float(candidate_geometry["trustworthiness_at_k"])
            >= float(release_geometry["trustworthiness_at_k"])
        ),
        "task_coverage_at_least_70pct": float(np.mean(deltas >= 0.0)) >= LIBERO_TASK_COVERAGE_FLOOR,
        "paired_bootstrap_ci_low_positive": ci_low > 0.0,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "family_count": len(families),
        "nonnegative_task_fraction": float(np.mean(deltas >= 0.0)),
        "mean_task_delta": float(np.mean(deltas)),
        "median_task_delta": float(np.median(deltas)),
        "paired_bootstrap_ci_low": ci_low,
        "paired_bootstrap_ci_high": ci_high,
        "bootstrap_seed": LIBERO_BOOTSTRAP_SEED,
        "bootstrap_samples": LIBERO_BOOTSTRAP_SAMPLES,
        "task_deltas": dict(zip(families, deltas.tolist(), strict=True)),
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    release_payload = _read(args.release)
    evaluation_payloads = [(path, _read(path)) for path in args.evaluations]
    release = _release(release_payload)
    metrics = [_checkpoint(path, payload) for path, payload in evaluation_payloads]
    selection = select_best_checkpoint(metrics, release)
    payload_by_id = {
        metric.checkpoint_id: payload
        for metric, (_, payload) in zip(metrics, evaluation_payloads, strict=True)
    }
    libero_by_id = (
        {
            metric.checkpoint_id: _libero_geometry_comparison(
                release_payload,
                payload_by_id[metric.checkpoint_id],
            )
            for metric in metrics
        }
        if args.require_libero_improvement
        else {}
    )
    assessments = []
    for assessment in selection.assessments:
        libero = libero_by_id.get(assessment.metrics.checkpoint_id)
        eligible = assessment.eligible and (libero is None or bool(libero["passes"]))
        failed_gates = list(assessment.failed_gates)
        if libero is not None:
            failed_gates.extend(
                f"libero_{name}" for name, passed in libero["checks"].items() if not passed
            )
        assessments.append(
            {
                "checkpoint_id": assessment.metrics.checkpoint_id,
                "eligible": eligible,
                "failed_gates": failed_gates,
                "geometry_score": assessment.geometry_score,
                "lwm_loss": assessment.metrics.lwm_loss,
                "state_loss": assessment.metrics.state_loss,
                "effective_rank": assessment.metrics.effective_rank,
                "shuffle_gain": assessment.metrics.shuffle_gain,
                "covariance_trace_ratio": assessment.release_comparison.covariance_trace_ratio,
                "pairwise_median_distance_ratio": assessment.release_comparison.pairwise_median_distance_ratio,
                "libero_geometry_improvement": libero,
            }
        )
    eligible_assessments = [item for item in assessments if item["eligible"]]
    eligible_assessments.sort(
        key=lambda item: (-item["geometry_score"], item["lwm_loss"], item["checkpoint_id"]),
    )
    selected_checkpoint_id = (
        eligible_assessments[0]["checkpoint_id"] if eligible_assessments else None
    )
    return {
        "schema": "lawam-stage1-checkpoint-selection-v1",
        "release_evaluation": str(args.release.resolve()),
        "require_libero_improvement": bool(args.require_libero_improvement),
        "selected_checkpoint_id": selected_checkpoint_id,
        "eligible_count": len(eligible_assessments),
        "evaluated_count": len(selection.assessments),
        "assessments": assessments,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--evaluations", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-libero-improvement", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = select(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
