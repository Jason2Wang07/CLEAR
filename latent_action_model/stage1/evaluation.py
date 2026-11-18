"""Deterministic Stage-1 validation and latent-space diagnostics.

This module implements the locked V1 evaluation protocol.  It intentionally
depends only on NumPy (PyTorch tensors are accepted and copied to CPU) so that
checkpoint evaluation does not depend on scikit-learn or GPU state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


EVALUATION_SEED = 20260803
VAL_SAMPLES_PER_FAMILY = 64
VAL_SAMPLES_PER_BIN = 16
GEOMETRY_K = 16
LOCAL_TEACHER_K = 32
SHUFFLES_PER_ANCHOR = 5


def _as_numpy(value: Any, *, name: str, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert a NumPy-like or torch tensor input without retaining gradients."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be converted to an array") from exc
    return array


def _numeric_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = _as_numpy(value, name=name, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _label_array(value: Any, *, name: str, length: int) -> np.ndarray:
    array = _as_numpy(value, name=name)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    for item in array.tolist():
        if item is None:
            raise ValueError(f"{name} contains None")
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            raise ValueError(f"{name} contains NaN or infinite values")
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable values") from exc
    return array


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _sorted_unique(values: Sequence[Hashable]) -> list[Hashable]:
    return sorted(set(values), key=_stable_key)


def _magnitude_bins(magnitudes: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    # side="left" gives the locked intervals: <=Q25, (Q25,Q50],
    # (Q50,Q75], and >Q75.
    return np.searchsorted(quantiles, magnitudes, side="left").astype(np.int64)


def _round_robin_select(
    candidate_indices: Sequence[int],
    count: int,
    *,
    tasks: np.ndarray,
    sample_ids: np.ndarray,
    rng: np.random.Generator,
    initial_indices: Sequence[int] = (),
) -> list[int]:
    """Seeded task-round-robin selection, invariant to input row ordering."""

    if count <= 0 or not candidate_indices:
        return []
    grouped: dict[Hashable, list[int]] = {}
    for index in candidate_indices:
        grouped.setdefault(tasks[index], []).append(int(index))

    task_order = _sorted_unique(list(grouped))
    if len(task_order) > 1:
        task_order = [task_order[i] for i in rng.permutation(len(task_order))]

    queues: dict[Hashable, list[int]] = {}
    for task in task_order:
        candidates = sorted(grouped[task], key=lambda i: _stable_key(sample_ids[i]))
        if len(candidates) > 1:
            candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        queues[task] = candidates

    cursors = {task: 0 for task in task_order}
    task_counts = {
        task: sum(1 for index in initial_indices if tasks[index] == task)
        for task in task_order
    }
    selected: list[int] = []
    while len(selected) < count:
        made_progress = False
        minimum_count = min(
            (task_counts[task] for task in task_order if cursors[task] < len(queues[task])),
            default=None,
        )
        if minimum_count is None:
            break
        for task in task_order:
            cursor = cursors[task]
            if cursor < len(queues[task]) and task_counts[task] == minimum_count:
                selected.append(queues[task][cursor])
                cursors[task] += 1
                task_counts[task] += 1
                made_progress = True
                break
        if not made_progress:
            break
    return selected


@dataclass(frozen=True)
class ValBalancedSelection:
    """The fixed validation anchors and audit metadata used by every run."""

    indices: np.ndarray
    sample_ids: np.ndarray
    families: np.ndarray
    source_bins: np.ndarray
    target_bins: np.ndarray
    family_indices: Mapping[Hashable, np.ndarray]
    train_quantiles: Mapping[Hashable, np.ndarray]
    excluded_families: tuple[Hashable, ...]
    seed: int = EVALUATION_SEED


def select_val_balanced(
    *,
    train_magnitudes: Any,
    train_families: Any,
    val_sample_ids: Any,
    val_magnitudes: Any,
    val_families: Any,
    val_tasks: Any,
    seed: int = EVALUATION_SEED,
    samples_per_family: int = VAL_SAMPLES_PER_FAMILY,
    samples_per_bin: int = VAL_SAMPLES_PER_BIN,
    min_family_size: int = VAL_SAMPLES_PER_FAMILY,
) -> ValBalancedSelection:
    """Build the locked, deterministic magnitude- and task-balanced val set.

    Quantiles are fitted on train samples separately for each family.  Four
    validation magnitude bins contribute the configured anchor quotas.  An underfull bin is
    first exhausted and then filled from the nearest source-bin shell; bins at
    the same distance are pooled and sampled by the same seeded task
    round-robin.  Samples are never repeated.
    """

    remainder = samples_per_family - 4 * samples_per_bin
    if remainder not in range(4):
        raise ValueError(
            "samples_per_family must be between four times samples_per_bin "
            "and that value plus three"
        )
    if min_family_size < samples_per_family:
        raise ValueError("min_family_size cannot be smaller than samples_per_family")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    train_magnitudes_np = _numeric_array(train_magnitudes, name="train_magnitudes", ndim=1)
    train_families_np = _label_array(
        train_families, name="train_families", length=len(train_magnitudes_np)
    )
    val_magnitudes_np = _numeric_array(val_magnitudes, name="val_magnitudes", ndim=1)
    n_val = len(val_magnitudes_np)
    val_sample_ids_np = _label_array(val_sample_ids, name="val_sample_ids", length=n_val)
    val_families_np = _label_array(val_families, name="val_families", length=n_val)
    val_tasks_np = _label_array(val_tasks, name="val_tasks", length=n_val)

    sample_id_list = val_sample_ids_np.tolist()
    if len(set(sample_id_list)) != len(sample_id_list):
        raise ValueError("val_sample_ids must be unique")

    rng = np.random.default_rng(int(seed))
    selected_indices: list[int] = []
    selected_source_bins: list[int] = []
    selected_target_bins: list[int] = []
    family_indices: dict[Hashable, np.ndarray] = {}
    train_quantiles: dict[Hashable, np.ndarray] = {}
    excluded_families: list[Hashable] = []

    for family in _sorted_unique(val_families_np.tolist()):
        val_family_indices = np.flatnonzero(val_families_np == family)
        if len(val_family_indices) < min_family_size:
            excluded_families.append(family)
            continue

        train_family_magnitudes = train_magnitudes_np[train_families_np == family]
        if len(train_family_magnitudes) == 0:
            raise ValueError(f"eligible family {family!r} has no train magnitudes")
        quantiles = np.quantile(train_family_magnitudes, [0.25, 0.50, 0.75]).astype(np.float64)
        train_quantiles[family] = quantiles.copy()

        family_bins = _magnitude_bins(val_magnitudes_np[val_family_indices], quantiles)
        source_bin_by_index = {
            int(index): int(source_bin)
            for index, source_bin in zip(val_family_indices.tolist(), family_bins.tolist(), strict=True)
        }
        available_by_bin: dict[int, list[int]] = {
            source_bin: [
                int(index)
                for index in val_family_indices
                if source_bin_by_index[int(index)] == source_bin
            ]
            for source_bin in range(4)
        }

        target_selections: dict[int, list[int]] = {}
        used: set[int] = set()
        # Reserve each bin's own quota first so one underfull bin cannot consume
        # another bin's required 16 samples.
        target_counts = [
            samples_per_bin + int(target_bin < remainder)
            for target_bin in range(4)
        ]
        for target_bin in range(4):
            own = _round_robin_select(
                available_by_bin[target_bin],
                min(target_counts[target_bin], len(available_by_bin[target_bin])),
                tasks=val_tasks_np,
                sample_ids=val_sample_ids_np,
                rng=rng,
            )
            target_selections[target_bin] = own
            used.update(own)

        # Fill deficits from increasingly distant adjacent-bin shells.  Equal
        # distance left/right bins are pooled before task round-robin.
        for target_bin in range(4):
            missing = target_counts[target_bin] - len(target_selections[target_bin])
            for distance in range(1, 4):
                if missing == 0:
                    break
                donor_bins = [
                    source_bin
                    for source_bin in range(4)
                    if abs(source_bin - target_bin) == distance
                ]
                donor_candidates = [
                    index
                    for source_bin in donor_bins
                    for index in available_by_bin[source_bin]
                    if index not in used
                ]
                additions = _round_robin_select(
                    donor_candidates,
                    min(missing, len(donor_candidates)),
                    tasks=val_tasks_np,
                    sample_ids=val_sample_ids_np,
                    rng=rng,
                    initial_indices=target_selections[target_bin],
                )
                target_selections[target_bin].extend(additions)
                used.update(additions)
                missing -= len(additions)
            if missing:
                raise RuntimeError(
                    f"could not select {samples_per_family} unique samples for family {family!r}"
                )

        family_selected: list[int] = []
        family_target_bins: list[int] = []
        for target_bin in range(4):
            chosen = target_selections[target_bin]
            family_selected.extend(chosen)
            family_target_bins.extend([target_bin] * len(chosen))
        if len(family_selected) != samples_per_family or len(set(family_selected)) != samples_per_family:
            raise RuntimeError(f"invalid balanced selection for family {family!r}")

        family_indices[family] = np.asarray(family_selected, dtype=np.int64)
        selected_indices.extend(family_selected)
        selected_target_bins.extend(family_target_bins)
        selected_source_bins.extend(source_bin_by_index[index] for index in family_selected)

    indices_np = np.asarray(selected_indices, dtype=np.int64)
    return ValBalancedSelection(
        indices=indices_np,
        sample_ids=val_sample_ids_np[indices_np].copy(),
        families=val_families_np[indices_np].copy(),
        source_bins=np.asarray(selected_source_bins, dtype=np.int64),
        target_bins=np.asarray(selected_target_bins, dtype=np.int64),
        family_indices=family_indices,
        train_quantiles=train_quantiles,
        excluded_families=tuple(excluded_families),
        seed=int(seed),
    )


def pairwise_euclidean(values: Any) -> np.ndarray:
    """Return an exact float64 Euclidean distance matrix."""

    values_np = _numeric_array(values, name="values", ndim=2)
    if len(values_np) < 2 or values_np.shape[1] < 1:
        raise ValueError("values must contain at least two rows and one feature")
    squared_norm = np.einsum("ij,ij->i", values_np, values_np)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (values_np @ values_np.T)
    distances = np.sqrt(np.maximum(squared_distance, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def pairwise_mean_huber(values: Any, *, delta: float = 1.0) -> np.ndarray:
    """Pairwise mean Huber dissimilarity used by the PCA DINO teacher."""

    values_np = _numeric_array(values, name="values", ndim=2)
    if len(values_np) < 2 or values_np.shape[1] < 1:
        raise ValueError("values must contain at least two rows and one feature")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite number")
    absolute = np.abs(values_np[:, None, :] - values_np[None, :, :])
    huber = np.where(absolute <= delta, 0.5 * absolute**2, delta * (absolute - 0.5 * delta))
    distances = np.mean(huber, axis=-1, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    return distances


def _distance_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = _numeric_array(value, name=name, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-8 * scale
    if np.min(matrix) < -tolerance:
        raise ValueError(f"{name} contains negative distances")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    if np.max(np.abs(np.diag(matrix))) > tolerance:
        raise ValueError(f"{name} diagonal must be zero")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _neighbor_order(distances: np.ndarray) -> np.ndarray:
    ranked = distances.copy()
    np.fill_diagonal(ranked, np.inf)
    # Stable sorting makes exact-distance ties reproducible in sample order.
    return np.argsort(ranked, axis=1, kind="mergesort")[:, : distances.shape[0] - 1]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _spearman_with_ties(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_centered = first_rank - np.mean(first_rank)
    second_centered = second_rank - np.mean(second_rank)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


@dataclass(frozen=True)
class FamilyGeometryMetrics:
    family: Hashable | None
    sample_count: int
    overlap_at_16: float
    trustworthiness_at_16: float
    rho_local: float
    rho_valid_anchors: int
    rho_skipped_ties: int


def evaluate_family_geometry(
    teacher_distances: Any,
    latent_distances: Any,
    *,
    family: Hashable | None = None,
    expected_size: int = VAL_SAMPLES_PER_FAMILY,
    k: int = GEOMETRY_K,
    teacher_local_k: int = LOCAL_TEACHER_K,
) -> FamilyGeometryMetrics:
    """Evaluate one family's fixed 64x64 teacher and latent distance matrices."""

    teacher = _distance_matrix(teacher_distances, name="teacher_distances")
    latent = _distance_matrix(latent_distances, name="latent_distances")
    if teacher.shape != latent.shape:
        raise ValueError("teacher_distances and latent_distances must have identical shapes")
    n = teacher.shape[0]
    if expected_size is not None and n != expected_size:
        raise ValueError(f"family must contain exactly {expected_size} samples, got {n}")
    if not isinstance(k, int) or k <= 0 or k >= n:
        raise ValueError("k must be an integer in [1, n-1]")
    if k >= n / 2:
        raise ValueError("standard trustworthiness requires k < n/2")
    if not isinstance(teacher_local_k, int) or teacher_local_k < k or teacher_local_k >= n:
        raise ValueError("teacher_local_k must be an integer in [k, n-1]")
    trust_denominator = n * k * (2 * n - 3 * k - 1)
    if trust_denominator <= 0:
        raise ValueError("n and k make the standard trustworthiness denominator non-positive")

    teacher_order = _neighbor_order(teacher)
    latent_order = _neighbor_order(latent)
    teacher_knn = teacher_order[:, :k]
    latent_knn = latent_order[:, :k]

    overlap_sum = 0.0
    trust_penalty = 0.0
    teacher_ranks = np.zeros((n, n), dtype=np.int64)
    row_indices = np.arange(n)[:, None]
    teacher_ranks[row_indices, teacher_order] = np.arange(1, n, dtype=np.int64)[None, :]
    for anchor in range(n):
        teacher_set = set(teacher_knn[anchor].tolist())
        latent_set = set(latent_knn[anchor].tolist())
        overlap_sum += len(teacher_set.intersection(latent_set)) / k
        for neighbor in latent_set.difference(teacher_set):
            trust_penalty += int(teacher_ranks[anchor, neighbor]) - k

    overlap = overlap_sum / n
    trustworthiness = 1.0 - 2.0 * trust_penalty / trust_denominator

    rho_values: list[float] = []
    skipped = 0
    for anchor in range(n):
        local_indices = np.asarray(
            sorted(set(teacher_order[anchor, :teacher_local_k]).union(latent_knn[anchor])),
            dtype=np.int64,
        )
        rho = _spearman_with_ties(teacher[anchor, local_indices], latent[anchor, local_indices])
        if rho is None:
            skipped += 1
        else:
            rho_values.append(rho)
    rho_local = float(np.mean(rho_values)) if rho_values else float("nan")

    return FamilyGeometryMetrics(
        family=family,
        sample_count=n,
        overlap_at_16=float(overlap),
        trustworthiness_at_16=float(trustworthiness),
        rho_local=rho_local,
        rho_valid_anchors=len(rho_values),
        rho_skipped_ties=skipped,
    )


@dataclass(frozen=True)
class GeometryReport:
    per_family: Mapping[Hashable, FamilyGeometryMetrics]
    overlap_at_16: float
    trustworthiness_at_16: float
    rho_local: float
    rho_valid_anchors: int
    rho_skipped_ties: int
    geometry_k: int
    teacher_local_k: int


def macro_average_geometry(
    per_family: Mapping[Hashable, FamilyGeometryMetrics],
    *,
    expected_family_size: int = VAL_SAMPLES_PER_FAMILY,
    geometry_k: int = GEOMETRY_K,
    teacher_local_k: int = LOCAL_TEACHER_K,
) -> GeometryReport:
    """Macro-average family geometry without pooling cross-family pairs."""

    if not per_family:
        raise ValueError("per_family cannot be empty")
    if not isinstance(expected_family_size, int) or expected_family_size < 2:
        raise ValueError("expected_family_size must be an integer >= 2")
    metrics = list(per_family.values())
    for metric in metrics:
        if metric.sample_count != expected_family_size:
            raise ValueError(
                "every family metric must come from exactly "
                f"{expected_family_size} samples"
            )
        if not np.isfinite(metric.overlap_at_16) or not np.isfinite(metric.trustworthiness_at_16):
            raise ValueError("family overlap/trustworthiness must be finite")
    rho_values = [metric.rho_local for metric in metrics]
    rho_macro = float(np.mean(rho_values)) if np.all(np.isfinite(rho_values)) else float("nan")
    return GeometryReport(
        per_family=dict(per_family),
        overlap_at_16=float(np.mean([metric.overlap_at_16 for metric in metrics])),
        trustworthiness_at_16=float(np.mean([metric.trustworthiness_at_16 for metric in metrics])),
        rho_local=rho_macro,
        rho_valid_anchors=sum(metric.rho_valid_anchors for metric in metrics),
        rho_skipped_ties=sum(metric.rho_skipped_ties for metric in metrics),
        geometry_k=int(geometry_k),
        teacher_local_k=int(teacher_local_k),
    )


def evaluate_balanced_geometry(
    *,
    teacher_descriptors: Any,
    latent_means: Any,
    families: Any,
    expected_family_size: int = VAL_SAMPLES_PER_FAMILY,
    k: int = GEOMETRY_K,
    teacher_local_k: int = LOCAL_TEACHER_K,
) -> GeometryReport:
    """Compute V1 geometry directly from scaled PCA descriptors and posterior means."""

    teacher_np = _numeric_array(teacher_descriptors, name="teacher_descriptors", ndim=2)
    latent_np = _numeric_array(latent_means, name="latent_means", ndim=2)
    if teacher_np.shape[0] != latent_np.shape[0]:
        raise ValueError("teacher_descriptors and latent_means must have the same number of rows")
    families_np = _label_array(families, name="families", length=len(teacher_np))
    if teacher_np.shape[1] < 1 or latent_np.shape[1] < 1:
        raise ValueError("teacher_descriptors and latent_means need at least one feature")

    per_family: dict[Hashable, FamilyGeometryMetrics] = {}
    for family in _sorted_unique(families_np.tolist()):
        mask = families_np == family
        count = int(np.sum(mask))
        if count != expected_family_size:
            raise ValueError(
                f"balanced family {family!r} must contain exactly {expected_family_size} rows, got {count}"
            )
        per_family[family] = evaluate_family_geometry(
            pairwise_mean_huber(teacher_np[mask], delta=1.0),
            pairwise_euclidean(latent_np[mask]),
            family=family,
            expected_size=expected_family_size,
            k=int(k),
            teacher_local_k=int(teacher_local_k),
        )
    return macro_average_geometry(
        per_family,
        expected_family_size=expected_family_size,
        geometry_k=int(k),
        teacher_local_k=int(teacher_local_k),
    )


def covariance_and_effective_rank(mu: Any, *, epsilon: float = 1e-12) -> tuple[np.ndarray, float, float]:
    """Return sample covariance, trace, and entropy effective rank in FP64."""

    mu_np = _numeric_array(mu, name="mu", ndim=2)
    if len(mu_np) < 2 or mu_np.shape[1] < 1:
        raise ValueError("mu must contain at least two samples and one dimension")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    centered = mu_np - np.mean(mu_np, axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(mu_np) - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / (float(np.sum(eigenvalues)) + epsilon)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + epsilon))))
    return covariance, float(np.trace(covariance)), effective_rank


def pairwise_median_distance(mu: Any) -> float:
    """Exact median Euclidean distance over all unordered validation pairs."""

    mu_np = _numeric_array(mu, name="mu", ndim=2)
    n = len(mu_np)
    if n < 2 or mu_np.shape[1] < 1:
        raise ValueError("mu must contain at least two samples and one dimension")
    pair_count = n * (n - 1) // 2
    distances = np.empty(pair_count, dtype=np.float64)
    cursor = 0
    for anchor in range(n - 1):
        row = np.linalg.norm(mu_np[anchor + 1 :] - mu_np[anchor], axis=1)
        distances[cursor : cursor + len(row)] = row
        cursor += len(row)
    return float(np.median(distances))


def posterior_kl_by_dimension(mu: Any, logvar: Any) -> np.ndarray:
    """Compute E[KL(q(z_d|x)||N(0,1))] independently for every dimension."""

    mu_np = _numeric_array(mu, name="mu", ndim=2)
    logvar_np = _numeric_array(logvar, name="logvar", ndim=2)
    if mu_np.shape != logvar_np.shape:
        raise ValueError("mu and logvar must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        kl = 0.5 * (mu_np**2 + np.exp(logvar_np) - logvar_np - 1.0)
    if not np.all(np.isfinite(kl)):
        raise ValueError("mu/logvar produce non-finite KL values")
    return np.mean(kl, axis=0, dtype=np.float64)


@dataclass(frozen=True)
class LatentDiagnostics:
    sample_count: int
    latent_dim: int
    covariance: np.ndarray
    effective_rank: float
    covariance_trace: float
    pairwise_median_distance: float
    kl_per_dim: np.ndarray
    active_dimensions: int
    mean_kl_per_dim: float
    mean_sigma_per_dim: np.ndarray
    mean_mu_squared_per_dim: np.ndarray


def evaluate_latent_diagnostics(
    mu: Any,
    logvar: Any,
    *,
    expected_latent_dim: int | None = 32,
    active_kl_threshold: float = 0.01,
) -> LatentDiagnostics:
    """Compute all locked V1 ``val-natural`` latent diagnostics."""

    mu_np = _numeric_array(mu, name="mu", ndim=2)
    logvar_np = _numeric_array(logvar, name="logvar", ndim=2)
    if mu_np.shape != logvar_np.shape:
        raise ValueError("mu and logvar must have identical shapes")
    if expected_latent_dim is not None and mu_np.shape[1] != expected_latent_dim:
        raise ValueError(f"expected latent dimension {expected_latent_dim}, got {mu_np.shape[1]}")
    if not np.isfinite(active_kl_threshold) or active_kl_threshold < 0:
        raise ValueError("active_kl_threshold must be finite and non-negative")

    covariance, trace, effective_rank = covariance_and_effective_rank(mu_np)
    kl_per_dim = posterior_kl_by_dimension(mu_np, logvar_np)
    sigma = np.exp(0.5 * logvar_np)
    if not np.all(np.isfinite(sigma)):
        raise ValueError("logvar produces non-finite posterior standard deviations")
    return LatentDiagnostics(
        sample_count=len(mu_np),
        latent_dim=mu_np.shape[1],
        covariance=covariance,
        effective_rank=effective_rank,
        covariance_trace=trace,
        pairwise_median_distance=pairwise_median_distance(mu_np),
        kl_per_dim=kl_per_dim,
        active_dimensions=int(np.sum(kl_per_dim >= active_kl_threshold)),
        mean_kl_per_dim=float(np.mean(kl_per_dim)),
        mean_sigma_per_dim=np.mean(sigma, axis=0, dtype=np.float64),
        mean_mu_squared_per_dim=np.mean(mu_np**2, axis=0, dtype=np.float64),
    )


@dataclass(frozen=True)
class ContextMatchedShuffles:
    indices: np.ndarray
    context_cosine_distance: np.ndarray
    context_rank_percentile: np.ndarray
    far_quantile: tuple[float, float]


def select_context_matched_far_shuffles(
    teacher_distances: Any,
    start_context: Any,
    *,
    eligible_mask: Any | None = None,
    count: int = SHUFFLES_PER_ANCHOR,
    far_quantile: tuple[float, float] = (0.70, 0.95),
) -> ContextMatchedShuffles:
    """Choose teacher-far latents with the most similar initial DINO context.

    Callers pass one compatible family/view/horizon pool at a time. Context is
    the L2-normalized mean of per-patch-normalized final post-LN DINO start
    tokens. The teacher-far band is formed first, then the five candidates with
    smallest context cosine distance are selected.
    """

    teacher = _distance_matrix(teacher_distances, name="teacher_distances")
    context = _numeric_array(start_context, name="start_context", ndim=2)
    n = len(context)
    if teacher.shape != (n, n):
        raise ValueError("teacher_distances and start_context row counts differ")
    if context.shape[1] < 1:
        raise ValueError("start_context needs at least one feature")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if tuple(far_quantile) != (0.70, 0.95):
        raise ValueError("locked V1 requires far_quantile=(0.70, 0.95)")

    norms = np.linalg.norm(context, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("start_context rows must have positive norm")
    context_unit = context / norms
    context_distance = np.maximum(1.0 - context_unit @ context_unit.T, 0.0)
    np.fill_diagonal(context_distance, np.inf)

    if eligible_mask is None:
        eligible = np.ones((n, n), dtype=bool)
        np.fill_diagonal(eligible, False)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != (n, n):
            raise ValueError(f"eligible_mask must have shape {(n, n)}, got {eligible.shape}")
        eligible = eligible.copy()
        np.fill_diagonal(eligible, False)

    selected = np.empty((n, count), dtype=np.int64)
    selected_context_distance = np.empty((n, count), dtype=np.float64)
    selected_context_percentile = np.empty((n, count), dtype=np.float64)
    for anchor in range(n):
        candidates = np.flatnonzero(eligible[anchor]).astype(np.int64)
        if len(candidates) < count:
            raise ValueError(f"anchor {anchor} has only {len(candidates)} eligible candidates")
        teacher_order = candidates[np.lexsort((candidates, teacher[anchor, candidates]))]
        start = int(math.ceil(far_quantile[0] * len(teacher_order)))
        stop = int(math.ceil(far_quantile[1] * len(teacher_order)))
        far = teacher_order[start:stop]
        if len(far) < count:
            raise ValueError(f"anchor {anchor} has only {len(far)} candidates in the far rank band")

        context_order = candidates[np.lexsort((candidates, context_distance[anchor, candidates]))]
        context_rank = np.full(n, -1, dtype=np.int64)
        context_rank[context_order] = np.arange(1, len(context_order) + 1)
        chosen = far[np.lexsort((far, context_distance[anchor, far]))[:count]]
        selected[anchor] = chosen
        selected_context_distance[anchor] = context_distance[anchor, chosen]
        selected_context_percentile[anchor] = context_rank[chosen] / float(len(context_order))

    return ContextMatchedShuffles(
        indices=selected,
        context_cosine_distance=selected_context_distance,
        context_rank_percentile=selected_context_percentile,
        far_quantile=far_quantile,
    )


@dataclass(frozen=True)
class ShuffleDiagnostics:
    sample_count: int
    shuffle_count: int
    true_loss: float
    shuffle_loss: float
    gain: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    mean_loss_per_shuffle: np.ndarray
    passes_gain_gate: bool
    ci_excludes_zero: bool
    seed: int
    bootstrap_samples: int


def evaluate_shuffle_gate(
    true_losses: Any,
    far_latent_losses: Any,
    *,
    seed: int = EVALUATION_SEED,
    bootstrap_samples: int = 2000,
    epsilon: float = 1e-12,
    gain_threshold: float = 0.10,
) -> ShuffleDiagnostics:
    """Evaluate five far-latent shuffles and an anchor-bootstrap 95% CI.

    Each bootstrap replicate resamples anchors with replacement; an anchor's
    five far-latent losses are averaged before aggregation.  The CI is a
    diagnostic and does not replace the locked hard gate on the point estimate.
    """

    true_np = _numeric_array(true_losses, name="true_losses", ndim=1)
    shuffled_np = _numeric_array(far_latent_losses, name="far_latent_losses", ndim=2)
    if len(true_np) == 0:
        raise ValueError("true_losses cannot be empty")
    if shuffled_np.shape != (len(true_np), SHUFFLES_PER_ANCHOR):
        raise ValueError(
            f"far_latent_losses must have shape ({len(true_np)}, {SHUFFLES_PER_ANCHOR}), "
            f"got {shuffled_np.shape}"
        )
    if np.any(true_np < 0) or np.any(shuffled_np < 0):
        raise ValueError("losses must be non-negative")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(gain_threshold):
        raise ValueError("gain_threshold must be finite")

    anchor_shuffle_loss = np.mean(shuffled_np, axis=1, dtype=np.float64)
    true_loss = float(np.mean(true_np))
    shuffle_loss = float(np.mean(anchor_shuffle_loss))
    gain = (shuffle_loss - true_loss) / (true_loss + epsilon)

    rng = np.random.default_rng(int(seed))
    bootstrap_gains = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, bootstrap_samples, chunk_size):
        count = min(chunk_size, bootstrap_samples - start)
        bootstrap_indices = rng.integers(0, len(true_np), size=(count, len(true_np)))
        sampled_true = np.mean(true_np[bootstrap_indices], axis=1)
        sampled_shuffle = np.mean(anchor_shuffle_loss[bootstrap_indices], axis=1)
        bootstrap_gains[start : start + count] = (sampled_shuffle - sampled_true) / (
            sampled_true + epsilon
        )
    ci_low, ci_high = np.quantile(bootstrap_gains, [0.025, 0.975])

    return ShuffleDiagnostics(
        sample_count=len(true_np),
        shuffle_count=SHUFFLES_PER_ANCHOR,
        true_loss=true_loss,
        shuffle_loss=shuffle_loss,
        gain=float(gain),
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        mean_loss_per_shuffle=np.mean(shuffled_np, axis=0, dtype=np.float64),
        passes_gain_gate=bool(gain >= gain_threshold),
        ci_excludes_zero=bool(ci_low > 0.0),
        seed=int(seed),
        bootstrap_samples=bootstrap_samples,
    )


def is_posterior_collapsed(
    *,
    active_dimensions: int,
    mean_kl_per_dim: float,
    shuffle_gain: float,
) -> bool:
    """Locked joint posterior-collapse criterion (all three clauses required)."""

    if not isinstance(active_dimensions, (int, np.integer)) or active_dimensions < 0:
        raise ValueError("active_dimensions must be a non-negative integer")
    if not np.isfinite(mean_kl_per_dim) or not np.isfinite(shuffle_gain):
        raise ValueError("mean_kl_per_dim and shuffle_gain must be finite")
    return bool(active_dimensions < 4 and mean_kl_per_dim < 0.01 and shuffle_gain < 0.05)
