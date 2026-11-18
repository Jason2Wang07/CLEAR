"""Production-oriented ManiSkill effect-window data processing primitives."""

from .effects import EFFECT_ADAPTERS, EffectResult, compute_effect
from .schema import SCHEMA_VERSION, schema_document
from .sources import SourceAdapter, SourceRecord, discover_sources
from .windows import Window, fixed_horizon, generate_fixed_windows

__all__ = [
    "EFFECT_ADAPTERS",
    "EffectResult",
    "SCHEMA_VERSION",
    "SourceAdapter",
    "SourceRecord",
    "Window",
    "compute_effect",
    "discover_sources",
    "fixed_horizon",
    "generate_fixed_windows",
    "schema_document",
]
