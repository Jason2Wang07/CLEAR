#!/usr/bin/env python3
"""Audit or convert LIBERO official HDF5 data for Stage-1 and ActionAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from libero_dataset.converter import build_release
from libero_dataset.source import DEFAULT_SUITES, audit_source


DEFAULT_SOURCE = Path("/data/share/libero_official")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--suites", nargs="+", choices=DEFAULT_SUITES)
    parser.add_argument("--max-task-files", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--horizon-actions", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=8)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="inspect source coverage without writing training shards")
    _common(audit)
    audit.add_argument("--output", type=Path)
    audit.add_argument(
        "--include-checksums",
        action="store_true",
        help="include SHA256 for every selected source HDF5 file",
    )
    build = subparsers.add_parser("build", help="write aligned effect and ActionAE releases")
    _common(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--shard-rows", type=int, default=256)
    build.add_argument("--split-seed", type=int, default=20260801)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "audit":
        report = audit_source(
            args.source_root,
            suites=args.suites,
            max_task_files=args.max_task_files,
            max_episodes=args.max_episodes,
            horizon=args.horizon_actions,
            window_stride=args.window_stride,
            include_checksums=args.include_checksums,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        report = build_release(
            args.source_root,
            args.output,
            suites=args.suites,
            max_task_files=args.max_task_files,
            max_episodes=args.max_episodes,
            horizon_actions=args.horizon_actions,
            window_stride=args.window_stride,
            shard_rows=args.shard_rows,
            split_seed=args.split_seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
