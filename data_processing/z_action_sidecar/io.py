"""Atomic HDF5 I/O for z-action sidecar shards."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .schema import FIELDS, SCHEMA_VERSION


def _array(values: Sequence, dtype: str) -> np.ndarray:
    if dtype == "string":
        return np.asarray(values, dtype=h5py.string_dtype("utf-8"))
    return np.asarray(values, dtype=dtype)


def write_shard_atomic(
    path: Path,
    rows: Sequence[Mapping],
    *,
    schema_version: str = SCHEMA_VERSION,
    fields: Mapping[str, tuple[str, tuple[int, ...]]] | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"immutable sidecar shard already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(schema_version=schema_version, complete=False)
            field_contract = FIELDS if fields is None else fields
            for name, (dtype, shape) in field_contract.items():
                if rows and name not in rows[0]:
                    raise KeyError(f"sidecar row is missing required field {name!r}")
                values = _array([row[name] for row in rows], dtype)
                expected = (len(rows), *shape)
                if values.shape != expected:
                    raise ValueError(f"{name}: got {values.shape}, expected {expected}")
                kwargs = {}
                if dtype != "string" and values.ndim > 1 and len(rows):
                    kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                handle.create_dataset(name, data=values, **kwargs)
            handle.attrs["complete"] = True
            handle.flush()
        with h5py.File(temporary, "r") as handle:
            if handle.attrs.get("schema_version") != schema_version or not handle.attrs.get("complete"):
                raise RuntimeError(f"failed to publish complete {schema_version} shard")
            for value in handle.values():
                value[...]
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
