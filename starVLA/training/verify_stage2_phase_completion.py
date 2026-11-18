"""Verify and receipt a completed Stage-2 phase before chaining its child."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from starVLA.training.stage2_phase import (
    STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
    STAGE2_SELECTION_SCHEMA,
    _config_sha256,
    _sha256_file,
    resolve_stage2_module_structure,
    stage2_checkpoint_contract_path,
    validate_stage2_launch_preflight_receipt,
    validate_stage2_parent_checkpoint,
    validate_stage2_training_config,
)
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


COMPLETION_SCHEMA = "lawam-stage2-phase-completion-v1"


def _load_object(path: Path, *, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object: {path}")
    return value


def _verify_checkpoint(
    path: Path,
    *,
    phase: str,
    recipe: str,
    module_structure: str,
    expected_step: int | None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint does not exist: {path}")
    sidecar_path = stage2_checkpoint_contract_path(path)
    sidecar = _load_object(sidecar_path, name="Stage-2 checkpoint sidecar")
    expected = {
        "schema_version": STAGE2_CHECKPOINT_CONTRACT_SCHEMA,
        "module_structure": module_structure,
        "stage2_phase": phase,
        "checkpoint_filename": path.name,
        "checkpoint_bytes": int(path.stat().st_size),
    }
    if recipe:
        expected["stage2_training_recipe"] = recipe
    if expected_step is not None:
        expected["completed_steps"] = int(expected_step)
    mismatches = {
        key: (sidecar.get(key), value)
        for key, value in expected.items()
        if sidecar.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage-2 checkpoint sidecar mismatch: {mismatches}")
    actual_sha = _sha256_file(path)
    if sidecar.get("checkpoint_sha256") != actual_sha:
        raise ValueError(
            "Stage-2 checkpoint SHA256 mismatch: "
            f"sidecar={sidecar.get('checkpoint_sha256')!r}, actual={actual_sha!r}"
        )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": actual_sha,
        "completed_steps": int(sidecar.get("completed_steps", -1)),
        "sidecar": str(sidecar_path),
    }


def _verify_run(
    *,
    cfg: Any,
    output_dir: Path,
    preflight_receipt: Path,
) -> dict[str, Any]:
    contract = validate_stage2_training_config(cfg)
    if contract is None:
        raise ValueError("phase completion requires a phased Stage-2 config")
    validate_stage2_parent_checkpoint(cfg, contract)
    validate_stage2_launch_preflight_receipt(cfg, preflight_receipt, contract)

    output_dir = output_dir.expanduser().resolve()
    max_steps = int(cfg.trainer.max_train_steps)
    phase = contract.phase.value
    recipe = contract.training_recipe
    module_structure = resolve_stage2_module_structure(contract)

    final_checkpoint = output_dir / "final_model" / "pytorch_model.pt"
    final_identity = _verify_checkpoint(
        final_checkpoint,
        phase=phase,
        recipe=recipe,
        module_structure=module_structure,
        expected_step=max_steps,
    )

    selection_path = output_dir / "checkpoints" / "best_validation_selection.json"
    selection = _load_object(selection_path, name="Stage-2 validation selection")
    selection_expected = {
        "schema_version": STAGE2_SELECTION_SCHEMA,
        "stage2_phase": phase,
        "selection_split": "validation",
        "selection_metric": "val_loss_total",
        "selection_mode": "min",
        "test_metrics_read": False,
        "last_evaluated_step": max_steps,
    }
    if recipe:
        selection_expected["stage2_training_recipe"] = recipe
    selection_mismatches = {
        key: (selection.get(key), value)
        for key, value in selection_expected.items()
        if selection.get(key) != value
    }
    if selection_mismatches:
        raise ValueError(
            f"Stage-2 validation selection mismatch: {selection_mismatches}"
        )
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Stage-2 validation selection has no selected checkpoint")
    selected_step = int(selected.get("step", -1))
    if selected_step <= 0 or selected_step > max_steps or selected_step % 500 != 0:
        raise ValueError(f"invalid selected validation step: {selected_step}")
    selected_identity = _verify_checkpoint(
        Path(str(selected.get("checkpoint", ""))),
        phase=phase,
        recipe=recipe,
        module_structure=module_structure,
        expected_step=selected_step,
    )
    for name in ("checkpoint_bytes", "checkpoint_sha256"):
        identity_name = "bytes" if name == "checkpoint_bytes" else "sha256"
        if selected.get(name) != selected_identity[identity_name]:
            raise ValueError(
                f"selection {name} mismatch: selected={selected.get(name)!r}, "
                f"actual={selected_identity[identity_name]!r}"
            )

    preserved = [
        _verify_checkpoint(
            output_dir / "checkpoints" / f"steps_{int(step)}_pytorch_model.pt",
            phase=phase,
            recipe=recipe,
            module_structure=module_structure,
            expected_step=int(step),
        )
        for step in (getattr(cfg.trainer, "preserve_checkpoint_steps", []) or [])
    ]

    return {
        "schema_version": COMPLETION_SCHEMA,
        "complete": True,
        "stage2_phase": phase,
        "stage2_training_recipe": recipe,
        "module_structure": module_structure,
        "config_sha256": _config_sha256(cfg),
        "max_train_steps": max_steps,
        "output_dir": str(output_dir),
        "preflight_receipt": str(preflight_receipt.expanduser().resolve()),
        "final": final_identity,
        "selection": str(selection_path),
        "selected": selected_identity,
        "preserved": preserved,
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preflight_receipt", required=True)
    parser.add_argument("--receipt", required=True)
    args, overrides = parser.parse_known_args(argv)

    cfg = OmegaConf.load(args.config_yaml)
    override_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(overrides))
    cfg = OmegaConf.merge(cfg, override_cfg)
    payload = _verify_run(
        cfg=cfg,
        output_dir=Path(args.output_dir),
        preflight_receipt=Path(args.preflight_receipt),
    )
    receipt = Path(args.receipt)
    if receipt.is_file():
        existing = _load_object(receipt, name="Stage-2 completion receipt")
        if dict(existing) != payload:
            raise ValueError(f"existing completion receipt mismatch: {receipt}")
    else:
        _write_receipt(receipt, payload)
    print(f"Stage-2 phase completion verified: {receipt.expanduser().resolve()}")
    print(f"selected_checkpoint={payload['selected']['path']}")
    print(f"selection={payload['selection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
