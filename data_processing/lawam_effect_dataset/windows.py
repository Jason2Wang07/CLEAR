"""Fixed-physical-time candidate generation and deterministic identifiers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Window:
    start_step: int
    end_step: int
    anchor_step: int
    anchor_event: str


def fixed_horizon(control_hz: float, tau: float = 1.6, *, tolerance: float = 1e-6) -> int:
    if control_hz <= 0 or tau <= 0:
        raise ValueError("control_hz and tau must be positive")
    exact = control_hz * tau
    rounded = round(exact)
    if abs(exact - rounded) > tolerance:
        raise ValueError(f"tau={tau} is not an integral number of steps at {control_hz} Hz")
    return int(rounded)


def generate_fixed_windows(n_steps: int, control_hz: float, anchors: Iterable[tuple[int, str]], *, tau: float = 1.6, alignments=(0.25, 0.5, 0.75)) -> list[Window]:
    horizon = fixed_horizon(control_hz, tau)
    if n_steps < horizon:
        return []
    unique: dict[tuple[int, int], Window] = {}
    for anchor, event in anchors:
        for alignment in alignments:
            start = int(round(anchor - alignment * horizon))
            start = min(max(start, 0), n_steps - horizon)
            end = start + horizon
            unique.setdefault((start, end), Window(start, end, int(anchor), str(event)))
    return sorted(unique.values(), key=lambda w: (w.start_step, w.end_step, w.anchor_event))


def stable_id(source_id: str, episode_id: int, start: int, end: int, *, version: str = "v1") -> str:
    payload = f"{version}\0{source_id}\0{episode_id}\0{start}\0{end}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def group_id(task_id: str, episode_seed: int) -> str:
    return hashlib.blake2b(f"{task_id}\0{episode_seed}".encode(), digest_size=12).hexdigest()


def split_group(group: str, seed: int = 20260801) -> str:
    value = int.from_bytes(hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=8).digest(), "big") / 2**64
    return "train" if value < .8 else "validation" if value < .9 else "test"
