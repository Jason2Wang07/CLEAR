"""Versioned task-level instructions for the ManiSkill z-action probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping


INSTRUCTION_SCHEMA_VERSION = "maniskill-z-action-canonical-instructions-v1"
INSTRUCTION_MAPPING_PATH = Path(__file__).with_name("canonical_instructions_v1.json")


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_instruction_document(path: str | Path = INSTRUCTION_MAPPING_PATH) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != INSTRUCTION_SCHEMA_VERSION:
        raise ValueError(
            f"instruction schema must be {INSTRUCTION_SCHEMA_VERSION!r}, "
            f"got {document.get('schema_version')!r}"
        )
    if document.get("language_level") != "task":
        raise ValueError("canonical instruction language_level must be 'task'")
    mapping = document.get("task_instructions")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("instruction document must contain a non-empty task_instructions object")
    invalid = [
        task_id
        for task_id, instruction in mapping.items()
        if not isinstance(task_id, str)
        or not task_id
        or not isinstance(instruction, str)
        or not instruction.strip()
    ]
    if invalid:
        raise ValueError(f"invalid canonical instruction entries: {sorted(map(str, invalid))}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("canonical instructions must be unique across tasks")
    return document


_DOCUMENT = load_instruction_document()
CANONICAL_INSTRUCTIONS: Mapping[str, str] = MappingProxyType(
    dict(_DOCUMENT["task_instructions"])
)
INSTRUCTION_MAPPING_SHA256 = _sha256_file(INSTRUCTION_MAPPING_PATH)


def canonical_instruction(task_id: str) -> str:
    """Return the frozen instruction for one exact ManiSkill task ID."""

    try:
        return CANONICAL_INSTRUCTIONS[str(task_id)]
    except KeyError as exc:
        raise KeyError(f"no canonical z-action instruction for task_id={task_id!r}") from exc


def validate_instruction_coverage(
    task_ids: Iterable[str],
    *,
    require_all_canonical_tasks: bool = False,
) -> None:
    """Fail if task IDs cannot be joined unambiguously to the frozen mapping."""

    observed = {str(task_id) for task_id in task_ids}
    expected = set(CANONICAL_INSTRUCTIONS)
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed) if require_all_canonical_tasks else []
    if unknown or missing:
        raise ValueError(f"instruction coverage mismatch: unknown={unknown}, missing={missing}")
