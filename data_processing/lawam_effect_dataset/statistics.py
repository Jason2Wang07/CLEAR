"""Train-only robust normalization and manifest-level validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def robust_normalization(
    rows: Iterable[Mapping],
    field: str,
    mask_field: str,
    *,
    split: str = "train",
    rms_floor: bool = False,
) -> dict:
    selected = [row for row in rows if row["split"] == split and bool(row.get("effect_valid", True))]
    if not selected:
        raise ValueError(f"no valid {split} rows")
    values = np.stack([np.asarray(row[field], dtype=np.float64) for row in selected])
    masks = np.stack([np.asarray(row[mask_field], dtype=bool) for row in selected])
    median = np.zeros(values.shape[1], dtype=np.float64)
    mad = np.ones(values.shape[1], dtype=np.float64)
    mad_only = np.ones(values.shape[1], dtype=np.float64)
    rms = np.ones(values.shape[1], dtype=np.float64)
    count = masks.sum(axis=0)
    for index in range(values.shape[1]):
        active = values[masks[:, index], index]
        if active.size:
            median[index] = np.median(active)
            raw_mad = np.median(np.abs(active - median[index]))
            mad_only[index] = max(1.4826 * raw_mad, 1e-8)
            rms[index] = max(float(np.sqrt(np.mean(np.square(active - median[index])))), 1e-8)
            mad[index] = max(mad_only[index], rms[index]) if rms_floor else mad_only[index]
    result = {
        "fit_split": split,
        "field": field,
        "count": count.tolist(),
        "median": median.tolist(),
        "scale_mad": mad.tolist(),
    }
    if rms_floor:
        result.update(
            {
                "scale_policy": "max(1.4826*MAD,RMS_about_median,1e-8)",
                "scale_mad_only": mad_only.tolist(),
                "scale_rms_floor": rms.tolist(),
            }
        )
    return result


def audit_splits(rows: Iterable[Mapping]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    samples, issues = set(), []
    count = defaultdict(int)
    for row in rows:
        sample = str(row["sample_id"])
        if sample in samples:
            issues.append(f"duplicate_sample_id:{sample}")
        samples.add(sample)
        group = str(row["group_id"])
        split = str(row["split"])
        groups[group].add(split)
        count[split] += 1
        expected = round(float(row["tau"]) * float(row["control_hz"]))
        if int(row["end_step"]) - int(row["start_step"]) != expected:
            issues.append(f"physical_duration:{sample}")
    leaking = sorted(group for group, splits in groups.items() if len(splits) != 1)
    issues.extend(f"group_leakage:{group}" for group in leaking)
    return {"valid": not issues, "rows": len(samples), "split_counts": dict(count), "leaking_groups": leaking, "issues": issues}
