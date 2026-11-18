"""Release-relative calibration used by the locked Stage-1 objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StudentGammaCalibration:
    gamma: float
    anchor_count: int
    latent_dim: int
    neighbor_rank: int
    rms_distance_p10: float
    rms_distance_p50: float
    rms_distance_p90: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "median_compatible_kth_neighbor_rms_distance",
            "gamma": self.gamma,
            "anchor_count": self.anchor_count,
            "latent_dim": self.latent_dim,
            "neighbor_rank": self.neighbor_rank,
            "rms_distance_p10": self.rms_distance_p10,
            "rms_distance_p50": self.rms_distance_p50,
            "rms_distance_p90": self.rms_distance_p90,
        }


def calibrate_student_gamma(
    bank: Any,
    *,
    neighbor_rank: int = 8,
    batch_size: int = 512,
) -> StudentGammaCalibration:
    """Set gamma to the median release-latent compatible k-NN RMS distance.

    The Stage-1 student distance is ``||mu_i-mu_j||^2 / D``. Its square root
    is an RMS coordinate distance. Calibrating at the same k used by latent
    neighbour mining gives the Student kernel a fixed local scale while
    remaining independent of all fine-tuning checkpoints.
    """

    latents = np.asarray(bank.latents, dtype=np.float32)
    if latents.ndim != 2 or len(latents) < 2:
        raise ValueError("bank.latents must be a non-empty [N,D] matrix")
    if not np.all(np.isfinite(latents)):
        raise ValueError("bank.latents contains NaN or infinity")
    if not isinstance(neighbor_rank, int) or neighbor_rank <= 0:
        raise ValueError("neighbor_rank must be a positive integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    kth_squared_l2 = np.empty(len(latents), dtype=np.float32)
    all_indices = np.arange(len(latents), dtype=np.int64)
    for start in range(0, len(latents), batch_size):
        stop = min(len(latents), start + batch_size)
        distances, _ = bank.query(
            all_indices[start:stop],
            latents[start:stop],
            k=neighbor_rank,
        )
        kth_squared_l2[start:stop] = distances[:, neighbor_rank - 1]

    rms = np.sqrt(np.maximum(kth_squared_l2, 0.0) / float(latents.shape[1])).astype(np.float64)
    if not np.all(np.isfinite(rms)) or np.any(rms <= 0):
        raise ValueError("release latent k-NN distances must be finite and strictly positive")
    p10, p50, p90 = np.quantile(rms, [0.10, 0.50, 0.90])
    return StudentGammaCalibration(
        gamma=float(p50),
        anchor_count=len(latents),
        latent_dim=int(latents.shape[1]),
        neighbor_rank=neighbor_rank,
        rms_distance_p10=float(p10),
        rms_distance_p50=float(p50),
        rms_distance_p90=float(p90),
    )
