"""Validated source-commit routing for replay workers.

Recorded commits were verified against the official mani-skill/ManiSkill Git
history on 2026-08-01. Unknown provenance is quarantined, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommitProfile:
    commit: str
    maniskill_version: str
    sapien_spec: str
    control_hz: float


PROFILES = {
    "baab60ede2e89167c1b7aaed41a9aa8e690a9d1e": CommitProfile("baab60ede2e89167c1b7aaed41a9aa8e690a9d1e", "3.0.0b14", "==3.0.0b1", 20),
    "6762a5bd9d9aebd510cafd385dba402f4a974b38": CommitProfile("6762a5bd9d9aebd510cafd385dba402f4a974b38", "3.0.0b18", "==3.0.0b1", 20),
    "9d5e0e01c3b13796dd1c9fc83213b315ce8a1946": CommitProfile("9d5e0e01c3b13796dd1c9fc83213b315ce8a1946", "3.0.0b14", "==3.0.0b1", 20),
    "652ad9353c0223507a938f0e8d990dd6f1c771ad": CommitProfile("652ad9353c0223507a938f0e8d990dd6f1c771ad", "3.0.0b4", "==3.0.0b1", 20),
    "ecc579b7567c31bb3d7176539d90e86ae49296be": CommitProfile("ecc579b7567c31bb3d7176539d90e86ae49296be", "0.0.2", "==3.0.0.dev2", 20),
    "95ea99d4eaad36477823d97077ab935caeb73359": CommitProfile("95ea99d4eaad36477823d97077ab935caeb73359", "3.0.0b12", "==3.0.0b1", 20),
    "e77e4ff3da2ee84216ec06a8b6b4d8b90d2eed79": CommitProfile("e77e4ff3da2ee84216ec06a8b6b4d8b90d2eed79", "3.0.0b9", "==3.0.0b1", 20),
    "ee5f8826afa693e0b1c602fae759a92fc462e05b": CommitProfile("ee5f8826afa693e0b1c602fae759a92fc462e05b", "3.0.0b21", ">=3.0.0b1", 20),
}


def replay_profile(source_commit: str | None) -> CommitProfile:
    """Return an exact profile or refuse replay when provenance is unknown."""
    if not source_commit:
        raise ValueError("source has no recorded ManiSkill commit; exact replay is quarantined")
    try:
        return PROFILES[source_commit]
    except KeyError as exc:
        raise ValueError(f"unverified ManiSkill commit: {source_commit}") from exc
