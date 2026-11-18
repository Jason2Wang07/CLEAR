"""Single-writer immutable HDF5 shards with atomic publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .schema import FIELDS, SCHEMA_VERSION


def _encode(values: Sequence, dtype: str) -> np.ndarray:
    if dtype == "string":
        return np.asarray(values, dtype=h5py.string_dtype("utf-8"))
    return np.asarray(values, dtype=dtype)


def write_shard_atomic(
    path: Path,
    rows: Sequence[Mapping],
    *,
    compression: str = "gzip",
    schema_version: str = SCHEMA_VERSION,
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable shard already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = schema_version
            handle.attrs["complete"] = False
            for name, (dtype, shape) in FIELDS.items():
                if not rows or name not in rows[0]:
                    continue
                values = _encode([row[name] for row in rows], dtype)
                expected = (len(rows), *shape)
                if values.shape != expected:
                    raise ValueError(f"{name}: got {values.shape}, expected {expected}")
                kwargs = {} if dtype == "string" or values.ndim == 1 else {"compression": compression, "shuffle": True}
                handle.create_dataset(name, data=values, **kwargs)
            handle.attrs["complete"] = True
            handle.flush()
        with h5py.File(temporary, "r") as handle:
            if not handle.attrs.get("complete"):
                raise RuntimeError("incomplete shard")
            for dataset in handle.values():
                if isinstance(dataset, h5py.Dataset):
                    dataset[...]
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
