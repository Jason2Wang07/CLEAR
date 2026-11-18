#!/usr/bin/env python3
"""Download a CLEAR model bundle and verify the published file checksums."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--kind", choices=["policy", "teachers", "all"], default="policy")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/clear"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    kinds = ["policy"] if args.kind == "policy" else ["effect", "action"]
    if args.kind == "all":
        kinds = ["policy", "effect", "action"]
    prefixes = [f"{args.benchmark}/{kind}/" for kind in kinds]
    snapshot_download("Jason2Jason/CLEAR", revision=args.revision, local_dir=args.output,
                      allow_patterns=["manifest.json", "README.md", "LICENSE*", *[prefix + "*" for prefix in prefixes]])
    manifest = json.loads((args.output / "manifest.json").read_text())
    checked = 0
    for entry in manifest["files"]:
        if not entry["path"].startswith(tuple(prefixes)):
            continue
        path = args.output / entry["path"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 << 20), b""):
                digest.update(chunk)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Model checksum mismatch: {entry['path']}")
        checked += 1
    if not checked:
        raise RuntimeError("Manifest contains no files for the requested model")
    print(f"Verified {checked} files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
