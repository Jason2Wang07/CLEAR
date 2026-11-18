"""Task-specific effect adapters. All outputs use a fixed storage width plus a mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .geometry import pose_in_frame, relative_pose, rotation, validate_pose

EFFECT_WIDTH = 12


@dataclass(frozen=True)
class EffectResult:
    family: str
    raw: np.ndarray
    mask: np.ndarray
    valid: bool
    direction: str
    magnitude: float
    quality: float = 1.0


def _pack(family: str, values: np.ndarray, mask: np.ndarray | None = None, *, quality: float = 1.0) -> EffectResult:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size > EFFECT_WIDTH or not np.isfinite(values).all():
        raise ValueError("effect values must be finite and fit storage width")
    raw = np.zeros(EFFECT_WIDTH, dtype=np.float32)
    raw[: values.size] = values
    out_mask = np.zeros(EFFECT_WIDTH, dtype=np.uint8)
    out_mask[: values.size] = 1 if mask is None else np.asarray(mask, dtype=np.uint8)
    active = raw[out_mask.astype(bool)]
    translation = raw[:3]
    axis = int(np.argmax(np.abs(translation))) if np.any(translation) else 0
    direction = "zero" if np.linalg.norm(active) < 1e-8 else ("+" if translation[axis] >= 0 else "-") + "xyz"[axis]
    return EffectResult(family, raw, out_mask, True, direction, float(np.linalg.norm(active)), quality)


def _pose(states: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    aliases = {"tee": ("tee", "Tee"), "peg": ("peg", "peg_0"), "box_with_hole": ("box_with_hole", "box_with_hole_0")}
    for candidate in aliases.get(name, (name,)):
        if candidate in states:
            return states[candidate]
    raise KeyError(name)


def rigid(start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray], target: str, *, rotation_enabled: bool) -> EffectResult:
    effect = relative_pose(_pose(start, target), _pose(end, target))
    mask = np.asarray([1, 1, 1, int(rotation_enabled), int(rotation_enabled), int(rotation_enabled)], dtype=np.uint8)
    return _pack("single_rigid", effect, mask)


def planar(start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray], target: str) -> EffectResult:
    delta = relative_pose(_pose(start, target), _pose(end, target))
    # Retain the common pose6 coordinate convention: dx, dy and yaw at index 5.
    values = np.asarray([delta[0], delta[1], 0, 0, 0, delta[5]], dtype=np.float32)
    return _pack("planar_se2", values, np.asarray([1, 1, 0, 0, 0, 1], dtype=np.uint8))


def relational(start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray], target: str, reference: str, family: str) -> EffectResult:
    before = pose_in_frame(_pose(start, target), _pose(start, reference))
    after = pose_in_frame(_pose(end, target), _pose(end, reference))
    return _pack(family, after - before)


def pyramid(start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray]) -> EffectResult:
    # Select the most displaced cube deterministically, then store its displacement
    # and endpoint relation to each other cube (translations only in v1).
    names = ("cubeA", "cubeB", "cubeC")
    displacements = [np.linalg.norm(validate_pose(end[n])[:3] - validate_pose(start[n])[:3]) for n in names]
    active = names[int(np.argmax(displacements))]
    others = [n for n in names if n != active]
    local = relative_pose(start[active], end[active])[:3]
    relations = np.concatenate([pose_in_frame(end[active], end[n])[:3] for n in others])
    return _pack("multi_object_relational", np.concatenate((local, relations)))


def tool(start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray]) -> EffectResult:
    target = relative_pose(start["cube"], end["cube"])
    tool_motion = relative_pose(start["l_shape_tool"], end["l_shape_tool"])
    return _pack("tool_use", np.concatenate((target, tool_motion)))


def _single(target: str, rotation_enabled: bool = False) -> Callable:
    return lambda s, e: rigid(s, e, target, rotation_enabled=rotation_enabled)


EFFECT_ADAPTERS: dict[str, Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], EffectResult]] = {
    "PushCube-v1": _single("cube"), "PullCube-v1": _single("cube"),
    "PickCube-v1": _single("cube"), "PokeCube-v1": _single("cube"),
    "LiftPegUpright-v1": _single("peg", True),
    "RollBall-v1": _single("ball"),
    "PushT-v1": lambda s, e: planar(s, e, "tee"),
    "StackCube-v1": lambda s, e: relational(s, e, "cubeA", "cubeB", "relative_placement"),
    "PegInsertionSide-v1": lambda s, e: relational(s, e, "peg", "box_with_hole", "precision_insertion"),
    "PlugCharger-v1": lambda s, e: relational(s, e, "charger", "receptacle", "precision_insertion"),
    "PullCubeTool-v1": tool,
    "StackPyramid-v1": pyramid,
}


def compute_effect(task_id: str, start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray]) -> EffectResult:
    try:
        return EFFECT_ADAPTERS[task_id](start, end)
    except KeyError as exc:
        if task_id not in EFFECT_ADAPTERS:
            raise ValueError(f"unsupported task: {task_id}") from exc
        raise ValueError(f"{task_id}: missing actor {exc.args[0]!r}") from exc
