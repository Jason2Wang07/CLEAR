"""Fixed, non-learned RoboTwin ActionVAE adapter for the LIBERO Full pipeline.

The source ActionVAE and its normalization stay frozen. LIBERO's single arm
occupies the left-arm slot; missing right-arm inputs are zero in normalized
source coordinates. Temporal interpolation is linear in xyz/principal-rotvec
coordinates and zero-order hold for gripper opening, not a new physical horizon.
The 2026-09-09 source is conditioned on the official LaWAM IDM. Its frozen
condition encoder must also be used when generating LIBERO posterior targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from latent_action_model.action_ae.beta_vae import ActionBetaVAEConfig, StandardNormalActionVAE
from .teacher_bundle import LIBERO_12S_ACTION_DATA_ROOT, LIBERO_12S_ACTION_NORMALIZATION_SHA256


TRANSFER_ID = "robotwin_lawam09_to_libero_left_chartlinear_v1"
REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "results/account-b/robotwin_lawam_idm_20260909/data/robotwin_action_v1_12s_dense_stride1_lawam_idm"
CHECKPOINT = REPO / "results/account-b/robotwin_lawam_idm_20260909/action-beta-vae/checkpoints/step-0005000.ckpt"
CHECKPOINT_SHA256 = "48a3fd7984a8b9edec0290318b19c3e8023e51ce0e720f8ecfd55299d739c2ef"
CONFIG = REPO / "latent_action_model/config/action_beta_vae_robotwin_v1_12s_lawam_idm.yaml"
CONFIG_SHA256 = "0dde8fcd740a3192ae8d4845c9a9f564e8e4bb203155250ce4bbefb398b43788"
NORMALIZATION_SHA256 = "255a793877156dc668fdb69b1e74f1bf2415a109ae287821f8f4b5f50e605896"
CONDITION_CHECKPOINT = REPO / "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
CONDITION_CONFIG = REPO / "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
CONDITION_SHA256 = "57c31dcba13c2998c4e866cdcfc427f14ae3c1b762a586778276e2100b48af88"
CONDITION_CONFIG_SHA256 = "e5f98738d1c6dc8d8643d5a73d44ebd46385fbb2fc4031d46a02ad384a353449"
SCHEMA = "robotwin-action-tcn-beta-vae-v1"
SOURCE_STEPS = 36
TARGET_STEPS = 24
GRIPPER_WIDTH_METERS = 0.08


def checked_transfer(value: str | None) -> bool:
    if value not in (None, "", TRANSFER_ID):
        raise ValueError(f"unsupported ActionVAE transfer: {value!r}")
    return value == TRANSFER_ID


def check_marker(payload: Mapping, transfer: str | None) -> None:
    expected = TRANSFER_ID if checked_transfer(transfer) else None
    if payload.get("action_teacher_transfer") != expected:
        raise ValueError("ActionVAE transfer marker mismatch")
    if expected and payload.get("condition_source") != "lawam_idm":
        raise ValueError("9.9 ActionVAE requires LaWAM IDM cache conditions")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source() -> tuple[StandardNormalActionVAE, dict, dict]:
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256), (CONFIG, CONFIG_SHA256),
        (SOURCE_ROOT / "stats/normalization.json", NORMALIZATION_SHA256),
        (LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json", LIBERO_12S_ACTION_NORMALIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"transfer asset hash mismatch: {path}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    metadata = payload["action_ae_metadata"]
    if (metadata["schema_version"] != SCHEMA
            or metadata["normalization_sha256"] != NORMALIZATION_SHA256
            or metadata.get("condition_checkpoint_sha256") != CONDITION_SHA256
            or metadata.get("condition_source") != "lawam_idm"):
        raise ValueError("source checkpoint is not the selected 9.9 LaWAM-conditioned RoboTwin VAE")
    config = ActionBetaVAEConfig(**metadata["model_config"])
    if (config.state_dim, config.trajectory_dim, config.horizon, config.latent_dim) != (14, 14, 36, 32):
        raise ValueError("source ActionVAE shape mismatch")
    model = StandardNormalActionVAE(config)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    target_norm = json.loads((LIBERO_12S_ACTION_DATA_ROOT / "stats/normalization.json").read_text())
    return model, metadata["normalization"], target_norm


def retime(trajectory: Tensor, current_gripper: Tensor, steps: int) -> Tensor:
    """Resample post-phase waypoints over the SAME interval, including t=0."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != 7:
        raise ValueError("retime expects [B,T,7] single-arm waypoints")
    batch, source_steps, _ = trajectory.shape
    if current_gripper.shape != (batch,) or steps < 1:
        raise ValueError("invalid gripper batch or target step count")
    anchor = torch.zeros_like(trajectory[:, :1])
    anchor[:, 0, 6] = current_gripper
    source = torch.cat((anchor, trajectory), dim=1)
    pose = F.interpolate(source[:, :, :6].transpose(1, 2), size=steps + 1,
                         mode="linear", align_corners=True).transpose(1, 2)[:, 1:]
    indices = torch.div(torch.arange(1, steps + 1, device=trajectory.device) * source_steps,
                        steps, rounding_mode="floor")
    gripper = source[:, indices, 6:7]
    return torch.cat((pose, gripper), dim=-1)


class FixedRobotwinLiberoAdapter(nn.Module):
    def __init__(self, source_norm: Mapping, target_norm: Mapping):
        super().__init__()
        for prefix, norm, groups, scale_key in (
            ("source", source_norm, ("left_position", "left_rotation_rotvec", "left_gripper"), "scale"),
            ("target", target_norm, ("position", "rotation_rotvec", "gripper"), "scale_mad"),
        ):
            for kind, key in (("center", "median"), ("scale", scale_key)):
                trajectory = torch.tensor([v for group in groups for v in norm[group][key]], dtype=torch.float32)
                state = torch.tensor(norm["current_state"][key], dtype=torch.float32)
                if trajectory.shape != (7,) or state.shape != (14,):
                    raise ValueError("transfer normalization dimensions are invalid")
                if not torch.isfinite(trajectory).all() or not torch.isfinite(state).all():
                    raise ValueError("nonfinite transfer normalization")
                if kind == "scale" and ((trajectory <= 0).any() or (state <= 0).any()):
                    raise ValueError("nonpositive transfer scales")
                self.register_buffer(f"{prefix}_{kind}", trajectory)
                self.register_buffer(f"{prefix}_state_{kind}", state)

    def raw_state(self, current_state: Tensor) -> Tensor:
        if current_state.ndim != 2 or current_state.shape[1] != 14:
            raise ValueError("LIBERO current state must be [B,14]")
        return current_state.float() * self.target_state_scale + self.target_state_center

    def source_state(self, current_state: Tensor) -> Tensor:
        raw = self.raw_state(current_state)[:, :7].clone()
        raw[:, 6] = raw[:, 6] / GRIPPER_WIDTH_METERS
        left = (raw - self.source_state_center[:7]) / self.source_state_scale[:7]
        return torch.cat((left, torch.zeros_like(left)), dim=-1)

    def source_trajectory(self, trajectory: Tensor, current_state: Tensor) -> Tensor:
        if tuple(trajectory.shape[1:]) != (24, 7):
            raise ValueError("LIBERO trajectory must be [B,24,7]")
        raw = trajectory.float() * self.target_scale + self.target_center
        state = self.raw_state(current_state)
        raw = retime(raw, state[:, 6], SOURCE_STEPS)
        raw = torch.cat((raw[:, :, :6], raw[:, :, 6:] / GRIPPER_WIDTH_METERS), dim=-1)
        left = (raw - self.source_center) / self.source_scale
        return torch.cat((left, torch.zeros_like(left)), dim=-1)

    def target_trajectory(self, trajectory: Tensor, current_state: Tensor) -> Tensor:
        if tuple(trajectory.shape[1:]) != (36, 14):
            raise ValueError("RoboTwin trajectory must be [B,36,14]")
        raw = trajectory[:, :, :7].float() * self.source_scale + self.source_center
        raw = torch.cat((raw[:, :, :6], raw[:, :, 6:] * GRIPPER_WIDTH_METERS), dim=-1)
        raw = retime(raw, self.raw_state(current_state)[:, 6], TARGET_STEPS)
        return (raw - self.target_center) / self.target_scale


class FrozenTransferredActionDecoder(nn.Module):
    def __init__(self, model: StandardNormalActionVAE, adapter: FixedRobotwinLiberoAdapter):
        super().__init__()
        self.condition_mlp = model.condition_mlp
        self.decoder = model.decoder
        self.transfer_adapter = adapter
        self.requires_grad_(False).train(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, z_action_raw: Tensor, current_state: Tensor, z_effect_raw: Tensor) -> Tensor:
        state = self.transfer_adapter.source_state(current_state)
        condition = self.condition_mlp(state, z_effect_raw.detach())
        decoded = self.decoder(z_action_raw.unsqueeze(1), condition)
        return self.transfer_adapter.target_trajectory(decoded, current_state)


def load_decoder(checkpoint_path: str | Path, action_data_root: str | Path) -> FrozenTransferredActionDecoder:
    if Path(checkpoint_path).resolve() != CHECKPOINT.resolve() or Path(action_data_root).resolve() != LIBERO_12S_ACTION_DATA_ROOT.resolve():
        raise ValueError("wrong RoboTwin transfer checkpoint or target normalization root")
    model, source_norm, target_norm = load_source()
    return FrozenTransferredActionDecoder(model, FixedRobotwinLiberoAdapter(source_norm, target_norm))
