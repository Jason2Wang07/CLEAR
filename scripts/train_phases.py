#!/usr/bin/env python3
"""Train W/P/E in order, retaining the original preflight and selection checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--run-name", required=True, help="Stable identifier; reuse it to resume.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = args.configs.resolve()
    os.chdir(ROOT)
    env = os.environ.copy()
    env["LAWAM_RUN_TIMESTAMP"] = args.run_name
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    parent = selection = None
    for phase in "wpe":
        config = configs / f"phase_{phase}.yaml"
        cfg = yaml.safe_load(config.read_text())
        batch = cfg["datasets"]["vla_data"]["per_device_batch_size"]
        accumulation = cfg["trainer"]["gradient_accumulation_steps"]
        expected = 256 if cfg["datasets"]["vla_data"]["data_mix"] == "libero" else 1024
        if args.gpus * batch * accumulation != expected:
            raise ValueError(f"Global batch must be {expected}; got {args.gpus * batch * accumulation}")
        run = Path(cfg["run_root_dir"]) / f"{args.run_name}+{cfg['run_id']}"
        receipt_dir = configs / "receipts" / args.run_name
        preflight = receipt_dir / f"{phase}_preflight.json"
        receipt = receipt_dir / f"{phase}_complete.json"
        overrides = [] if phase == "w" else [f"--trainer.pretrained_checkpoint={parent}", f"--trainer.stage2_parent_selection={selection}"]
        print(f"{phase.upper()}: {cfg['trainer']['max_train_steps']} updates, global batch {expected}", flush=True)
        if args.dry_run:
            continue
        receipt_dir.mkdir(parents=True, exist_ok=True)
        verify = [sys.executable, "-m", "starVLA.training.verify_stage2_phase_completion",
                  "--config_yaml", str(config), "--output_dir", str(run),
                  "--preflight_receipt", str(preflight), "--receipt", str(receipt), *overrides]
        if not receipt.exists() and not (run / "final_model/pytorch_model.pt").exists():
            subprocess.run([sys.executable, "-m", "starVLA.training.preflight_stage2",
                            "--config_yaml", str(config), "--output", str(preflight), *overrides], env=env, check=True)
            env["LAWAM_STAGE2_PREFLIGHT_RECEIPT"] = str(preflight)
            train_overrides = list(overrides)
            states = sorted((run / "checkpoints").glob("steps_*_training_state"),
                            key=lambda p: int(p.name.split("_")[1]))
            if states:
                train_overrides.append(f"--trainer.resume_training_state={states[-1]}")
            elif list((run / "checkpoints").glob("steps_*_pytorch_model.pt")):
                raise RuntimeError(f"Model-only checkpoints cannot exactly resume training: {run}")
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                            f"--nproc_per_node={args.gpus}", "starVLA/training/train_starvla.py",
                            "--config_yaml", str(config), *train_overrides], env=env, check=True)
        subprocess.run(verify, env=env, check=True)
        completed = json.loads(receipt.read_text())
        parent, selection = completed["selected"]["path"], completed["selection"]
    if not args.dry_run:
        print(f"Selected CLEAR policy: {parent}")


if __name__ == "__main__":
    main()
