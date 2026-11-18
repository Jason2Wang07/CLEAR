from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from .state_transform import (
    RoboTwinActionVAE14Padded32StateAdapter,
    build_robotwin_actionvae_raw_state,
)


def test_raw16_wxyz_bridge_matches_stage1_geometry() -> None:
    left_rotation = Rotation.from_rotvec([0.2, -0.3, 0.4])
    right_rotation = Rotation.from_rotvec([-0.5, 0.1, 0.2])
    left_xyzw = left_rotation.as_quat()
    right_xyzw = right_rotation.as_quat()
    raw = np.asarray(
        [
            0.1,
            0.2,
            0.3,
            left_xyzw[3],
            *left_xyzw[:3],
            0.75,
            -0.1,
            -0.2,
            0.8,
            right_xyzw[3],
            *right_xyzw[:3],
            0.25,
        ],
        dtype=np.float32,
    )

    state, mask = build_robotwin_actionvae_raw_state(raw)

    np.testing.assert_allclose(state[0:3], raw[0:3])
    np.testing.assert_allclose(state[3:6], left_rotation.as_rotvec(), atol=1.0e-6)
    assert state[6] == pytest.approx(0.75)
    np.testing.assert_allclose(state[7:10], raw[8:11])
    np.testing.assert_allclose(state[10:13], right_rotation.as_rotvec(), atol=1.0e-6)
    assert state[13] == pytest.approx(0.25)
    assert mask.shape == (14,) and mask.all()


def test_state_adapter_uses_robust_scale_and_pads_to_32(tmp_path) -> None:
    metadata = {
        "schema_version": "robotwin-z-action-normalization-v1-12s",
        "fit_split": "train",
        "estimator": "median_and_max_1.4826_mad_rms_about_median",
        "sample_count": 8,
        "current_state": {
            "median": list(range(14)),
            "scale": [2.0] * 14,
            "count": [8] * 14,
        },
    }
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = RoboTwinActionVAE14Padded32StateAdapter(path, expected_sha256=digest)

    raw = torch.arange(14, dtype=torch.float32) + 2.0
    result = adapter(raw, torch.ones(14, dtype=torch.bool))

    torch.testing.assert_close(result[:14], torch.ones(14))
    torch.testing.assert_close(result[14:], torch.zeros(18))


def test_state_adapter_rejects_partial_dual_arm_mask(tmp_path) -> None:
    metadata = {
        "schema_version": "robotwin-z-action-normalization-v1-12s",
        "fit_split": "train",
        "estimator": "median_and_max_1.4826_mad_rms_about_median",
        "sample_count": 1,
        "current_state": {
            "median": [0.0] * 14,
            "scale": [1.0] * 14,
            "count": [1] * 14,
        },
    }
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = RoboTwinActionVAE14Padded32StateAdapter(path, expected_sha256=digest)
    mask = torch.ones(14, dtype=torch.bool)
    mask[-1] = False

    with pytest.raises(ValueError, match="all valid"):
        adapter(torch.zeros(14), mask)
