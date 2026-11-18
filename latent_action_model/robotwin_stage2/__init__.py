"""RoboTwin Stage-2 paired z-effect/z-action support."""

from .temporal import (
    DEFAULT_TEMPORAL_CONTRACT,
    DENSE_S1_TEMPORAL_CONTRACT,
    RoboTwinStage2TemporalContract,
)


def __getattr__(name: str):
    if name in {
        "PairedExpertAdapters",
        "PairedLaWMAdapter",
        "PairedLatentPredictor",
    }:
        from . import paired

        return getattr(paired, name)
    raise AttributeError(name)

__all__ = [
    "DEFAULT_TEMPORAL_CONTRACT",
    "DENSE_S1_TEMPORAL_CONTRACT",
    "PairedExpertAdapters",
    "PairedLaWMAdapter",
    "PairedLatentPredictor",
    "RoboTwinStage2TemporalContract",
]
