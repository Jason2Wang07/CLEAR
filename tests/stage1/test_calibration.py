import numpy as np
import pytest

from latent_action_model.stage1.calibration import calibrate_student_gamma
from latent_action_model.stage1.memory_bank import LatentMemoryBank


def _bank() -> LatentMemoryBank:
    latents = np.arange(12, dtype=np.float32)[:, None]
    return LatentMemoryBank(
        [f"s{i}" for i in range(12)],
        latents,
        ["pool"] * 12,
        [f"episode-{i}" for i in range(12)],
        np.zeros(12, dtype=np.int64),
        np.ones(12, dtype=np.int64),
        backend="exact",
        allow_exact_fallback=True,
    )


def test_gamma_is_median_kth_compatible_neighbor_rms_distance() -> None:
    result = calibrate_student_gamma(_bank(), neighbor_rank=2, batch_size=5)
    assert result.gamma == pytest.approx(1.0)
    assert result.anchor_count == 12
    assert result.latent_dim == 1
    assert result.neighbor_rank == 2
    assert result.as_dict()["method"] == "median_compatible_kth_neighbor_rms_distance"


def test_gamma_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calibrate_student_gamma(_bank(), neighbor_rank=0)
