"""Exact-row compact conditioning cache for ActionVAE priors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor


PRIOR_CONTEXT_SCHEMA_VERSION = "action-vae-prior-context-v1"
PRIOR_CONTEXT_KEYS = ("current_context", "current_visual", "future_visual")


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode(value.item())
    return str(value)


class ActionPriorContextCache:
    """Lazy process-local reader with an exact sample-ID/split join."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_ids: Sequence[str],
        split: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ActionVAE prior context cache does not exist: {self.path}")
        with h5py.File(self.path, "r") as handle:
            if handle.attrs.get("schema_version") != PRIOR_CONTEXT_SCHEMA_VERSION:
                raise ValueError("ActionVAE prior context cache schema mismatch")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError("ActionVAE prior context cache is incomplete")
            required = {"sample_id", "split", *PRIOR_CONTEXT_KEYS}
            missing = sorted(required.difference(handle.keys()))
            if missing:
                raise ValueError(f"ActionVAE prior context cache is missing arrays: {missing}")
            cache_ids = [_decode(value) for value in handle["sample_id"][:]]
            cache_splits = [_decode(value) for value in handle["split"][:]]
            if len(cache_ids) != len(cache_splits) or len(set(cache_ids)) != len(cache_ids):
                raise ValueError("ActionVAE prior context cache IDs must be a unique vector")
            row_count = len(cache_ids)
            widths = {}
            for key in PRIOR_CONTEXT_KEYS:
                dataset = handle[key]
                if dataset.ndim != 2 or dataset.shape[0] != row_count or dataset.shape[1] < 1:
                    raise ValueError(f"prior context field {key!r} must have shape [N,D]")
                if dataset.dtype.kind != "f":
                    raise TypeError(f"prior context field {key!r} must be floating point")
                widths[key] = int(dataset.shape[1])
        positions = {sample_id: index for index, sample_id in enumerate(cache_ids)}
        required_ids = [str(value) for value in sample_ids]
        missing_ids = sorted(set(required_ids).difference(positions))
        if missing_ids:
            raise ValueError(f"prior context cache is missing sidecar IDs: {missing_ids[:20]}")
        selected_rows = [positions[sample_id] for sample_id in required_ids]
        mismatched = [
            required_ids[index]
            for index, row in enumerate(selected_rows)
            if cache_splits[row] != str(split)
        ]
        if mismatched:
            raise ValueError(f"prior context cache split mismatch: {mismatched[:20]}")
        self.widths = widths
        self._positions = positions
        self._handle: h5py.File | None = None
        self._owner_pid: int | None = None

    def _get_handle(self) -> h5py.File:
        if self._owner_pid != os.getpid() or self._handle is None:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._owner_pid = os.getpid()
        return self._handle

    def get(self, sample_id: str) -> dict[str, Tensor]:
        try:
            row = self._positions[str(sample_id)]
        except KeyError as exc:
            raise KeyError(f"sample ID is absent from prior context cache: {sample_id}") from exc
        handle = self._get_handle()
        result = {
            key: torch.from_numpy(np.asarray(handle[key][row], dtype=np.float32).copy())
            for key in PRIOR_CONTEXT_KEYS
        }
        invalid = [key for key, value in result.items() if not bool(torch.isfinite(value).all())]
        if invalid:
            raise ValueError(f"prior context row {sample_id!r} contains NaN/Inf: {invalid}")
        return result

    def reset_worker_state(self) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._owner_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
