"""Resolve release paths in both policy and lightweight simulator clients."""
from pathlib import Path


def resolve_release_paths(value, *, run_dir):
    roots = {
        "repo://": Path(__file__).resolve().parents[2],
        "bundle://": Path(run_dir).resolve(),
    }
    if isinstance(value, dict):
        return {key: resolve_release_paths(item, run_dir=run_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_release_paths(item, run_dir=run_dir) for item in value]
    if isinstance(value, str):
        for prefix, root in roots.items():
            if value.startswith(prefix):
                return str(root / value[len(prefix):])
    return value
