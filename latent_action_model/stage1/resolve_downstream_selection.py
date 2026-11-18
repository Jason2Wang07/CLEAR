"""Resolve a validation-only Stage-1 checkpoint for downstream A1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(selection_path: Path) -> dict[str, Any]:
    report = json.loads(selection_path.read_text(encoding="utf-8"))
    if report.get("schema") != "lawam-stage1-checkpoint-selection-v1":
        raise ValueError("incompatible Stage-1 checkpoint-selection report")
    assessments = report.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise ValueError("checkpoint-selection report has no assessments")
    selected = report.get("selected_checkpoint_id")
    if selected:
        status = "validation_selected"
        rationale = None
    else:
        ranked = sorted(
            assessments,
            key=lambda item: (
                -float(item["geometry_score"]),
                float(item["lwm_loss"]),
                str(item["checkpoint_id"]),
            ),
        )
        selected = str(ranked[0]["checkpoint_id"])
        status = "explicit_ablation"
        rationale = (
            "No validation candidate passed all locked release-relative gates; use the "
            "highest validation geometry score (tie-break: lower validation LWM loss) "
            "for the requested controlled A1 downstream experiment. Test metrics were "
            "not consulted."
        )
    assessment = next(
        (item for item in assessments if str(item.get("checkpoint_id")) == selected),
        None,
    )
    if assessment is None:
        raise ValueError(f"selected checkpoint {selected!r} has no assessment")
    return {
        "schema": "lawam-stage1-downstream-selection-v1",
        "source_selection_report": str(selection_path.resolve()),
        "source_selection_sha256": _sha256(selection_path),
        "selected_checkpoint_id": selected,
        "selection_status": status,
        "selection_rationale": rationale,
        "validation_geometry_score": float(assessment["geometry_score"]),
        "validation_lwm_loss": float(assessment["lwm_loss"]),
        "eligible_under_locked_gates": bool(assessment.get("eligible")),
        "failed_gates": list(assessment.get("failed_gates", [])),
        "test_consulted": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = resolve(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
