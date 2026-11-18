"""Single-process launch preflight for formal phased Stage-2 runs."""

from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from starVLA.training.stage2_phase import write_stage2_launch_preflight_receipt
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--output", required=True)
    args, overrides = parser.parse_known_args(argv)

    cfg = OmegaConf.load(args.config_yaml)
    override_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(overrides))
    cfg = OmegaConf.merge(cfg, override_cfg)
    try:
        receipt = write_stage2_launch_preflight_receipt(cfg=cfg, output=args.output)
    except Exception as exc:
        print(f"Stage-2 launch preflight failed: {exc}", file=sys.stderr)
        return 1
    print(f"Stage-2 launch preflight passed: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
