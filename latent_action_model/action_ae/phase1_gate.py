"""Publish the validation intervention gate required by frozen ActionVAE Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASE1_GATE_SCHEMA_VERSION = "action-vae-phase1-intervention-gate-v1"
EVALUATION_SCHEMA_VERSION = "action-tcn-ae-v0-held-out-evaluation-v1"
LOCKED_MINIMUM_RELATIVE_DEGRADATION = 0.10
INTERVENTIONS = ("z_zero_decoder", "z_shuffled_decoder")
METRICS = {
    "position_l2_mae_cm": ("all", "position", "l2_mae_cm"),
    "rotation_geodesic_mae_deg": ("all", "rotation", "geodesic_mae_deg"),
}


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _metric(reconstruction: Mapping[str, Any], mode: str, path: Sequence[str]) -> float:
    value: Any = reconstruction[mode]
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {mode} reconstruction metric {'.'.join(path)}={result}")
    return result


def build_phase1_gate(
    *,
    evaluation_path: str | Path,
    checkpoint_path: str | Path,
    minimum_relative_degradation: float = 0.10,
    expected_validation_samples: int | None = None,
) -> dict[str, Any]:
    """Validate full held-out evidence and bind it to one deterministic checkpoint."""

    if (
        not math.isfinite(minimum_relative_degradation)
        or minimum_relative_degradation < LOCKED_MINIMUM_RELATIVE_DEGRADATION
    ):
        raise ValueError(
            "minimum_relative_degradation must be finite and at least "
            f"{LOCKED_MINIMUM_RELATIVE_DEGRADATION}"
        )
    evaluation_file = Path(evaluation_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not evaluation_file.is_file():
        raise FileNotFoundError(evaluation_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    report = _read_json(evaluation_file)
    if report.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Phase 1 evaluation schema mismatch")

    checkpoint_sha256 = sha256_file(checkpoint_file)
    evaluated_checkpoint = report.get("checkpoint")
    if not isinstance(evaluated_checkpoint, Mapping):
        raise ValueError("Phase 1 evaluation is missing checkpoint provenance")
    if Path(str(evaluated_checkpoint.get("path", ""))).expanduser().resolve() != checkpoint_file:
        raise ValueError("Phase 1 evaluation/checkpoint path mismatch")
    if str(evaluated_checkpoint.get("sha256", "")).lower() != checkpoint_sha256:
        raise ValueError("Phase 1 evaluation/checkpoint SHA256 mismatch")
    if evaluated_checkpoint.get("variant") != "a1":
        raise ValueError("frozen LIBERO ActionVAE Phase 2 requires an A1 ActionAE checkpoint")

    arguments = report.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Phase 1 evaluation is missing arguments")
    if list(arguments.get("reconstruction_splits", [])) != ["validation"]:
        raise ValueError("Phase 1 gate accepts validation-only reconstruction evidence")
    if arguments.get("terminal_probe") is not False:
        raise ValueError("Phase 1 validation gate must disable the train/test terminal probe")
    subset_sizes = arguments.get("subset_sizes")
    if not isinstance(subset_sizes, Mapping) or subset_sizes.get("validation") is not None:
        raise ValueError("Phase 1 gate requires the full validation split, not a subset")

    splits = report.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"validation"}:
        raise ValueError("Phase 1 gate requires exactly one validation split report")
    validation = splits["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("Phase 1 validation report must be an object")
    sample_count = int(validation.get("sample_count", 0))
    if sample_count < 2:
        raise ValueError("Phase 1 validation report is empty")
    if expected_validation_samples is not None and sample_count != int(expected_validation_samples):
        raise ValueError(
            "Phase 1 validation sample-count mismatch: "
            f"expected {int(expected_validation_samples)}, found {sample_count}"
        )
    collected = report.get("data", {}).get("collected", {}).get("validation", {})
    if int(collected.get("sample_count", -1)) != sample_count:
        raise ValueError("Phase 1 evaluation collected/sample split counts differ")

    reconstruction = validation.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError("Phase 1 evaluation is missing reconstruction metrics")
    checks: dict[str, dict[str, Any]] = {}
    for intervention in INTERVENTIONS:
        checks[intervention] = {}
        for metric_name, metric_path in METRICS.items():
            normal = _metric(reconstruction, "normal", metric_path)
            intervened = _metric(reconstruction, intervention, metric_path)
            if normal <= 0:
                raise ValueError(f"normal {metric_name} must be positive")
            relative_degradation = (intervened - normal) / normal
            checks[intervention][metric_name] = {
                "normal": normal,
                "intervened": intervened,
                "relative_degradation": relative_degradation,
                "minimum_required": float(minimum_relative_degradation),
                "passed": relative_degradation >= minimum_relative_degradation,
            }

    passed = all(
        metric["passed"]
        for intervention in checks.values()
        for metric in intervention.values()
    )
    return {
        "schema_version": PHASE1_GATE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "criterion": (
            "Both zero and globally shuffled decoder-only z-action interventions must increase "
            "full-validation position and rotation error by at least the locked relative threshold."
        ),
        "minimum_relative_degradation": float(minimum_relative_degradation),
        "validation_sample_count": sample_count,
        "validation_sample_ids_sha256": validation.get("sample_ids_sha256"),
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": checkpoint_sha256,
            "epoch": evaluated_checkpoint.get("epoch"),
            "global_step": evaluated_checkpoint.get("global_step"),
            "schema_version": "action-tcn-ae-v0",
        },
        "evaluation": {
            "path": str(evaluation_file),
            "sha256": sha256_file(evaluation_file),
            "schema_version": EVALUATION_SCHEMA_VERSION,
        },
    }


def validate_phase1_gate(
    gate_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Reject missing, failed, stale, or checkpoint-mismatched Phase 1 gates."""

    path = Path(gate_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ActionVAE Phase 1 gate does not exist: {path}")
    gate = _read_json(path)
    if gate.get("schema_version") != PHASE1_GATE_SCHEMA_VERSION:
        raise ValueError("ActionVAE Phase 1 gate schema mismatch")
    if gate.get("passed") is not True:
        raise ValueError("ActionVAE Phase 1 intervention gate did not pass")
    bound = gate.get("checkpoint")
    if not isinstance(bound, Mapping):
        raise ValueError("ActionVAE Phase 1 gate is missing checkpoint binding")
    if Path(str(bound.get("path", ""))).expanduser().resolve() != checkpoint:
        raise ValueError("ActionVAE Phase 1 gate/checkpoint path mismatch")
    actual_checkpoint_sha256 = sha256_file(checkpoint)
    if str(bound.get("sha256", "")).lower() != actual_checkpoint_sha256:
        raise ValueError("ActionVAE Phase 1 gate/checkpoint SHA256 mismatch")
    evaluation = gate.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("ActionVAE Phase 1 gate is missing evaluation binding")
    evaluation_path = Path(str(evaluation.get("path", ""))).expanduser().resolve()
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"bound Phase 1 evaluation does not exist: {evaluation_path}")
    if str(evaluation.get("sha256", "")).lower() != sha256_file(evaluation_path):
        raise ValueError("ActionVAE Phase 1 gate/evaluation SHA256 mismatch")
    threshold = float(gate.get("minimum_relative_degradation", 0.0))
    recomputed = build_phase1_gate(
        evaluation_path=evaluation_path,
        checkpoint_path=checkpoint,
        minimum_relative_degradation=threshold,
        expected_validation_samples=int(gate.get("validation_sample_count", 0)),
    )
    if recomputed["passed"] is not True or gate.get("checks") != recomputed["checks"]:
        raise ValueError("ActionVAE Phase 1 gate checks do not match bound evaluation evidence")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": PHASE1_GATE_SCHEMA_VERSION,
        "passed": True,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": str(evaluation["sha256"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-relative-degradation", type=float, default=0.10)
    parser.add_argument("--expected-validation-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite Phase 1 gate: {output}")
    gate = build_phase1_gate(
        evaluation_path=args.evaluation,
        checkpoint_path=args.checkpoint,
        minimum_relative_degradation=args.minimum_relative_degradation,
        expected_validation_samples=args.expected_validation_samples,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "passed": gate["passed"]}, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
