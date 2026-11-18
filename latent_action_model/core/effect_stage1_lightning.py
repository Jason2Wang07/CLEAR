"""Effect-aware Stage-1 fine-tuning for the released LaWAM latent model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from .lam_lightinng import VJEPA_LAM
from .lora import LoRAInjection, inject_lora_into_encoder, is_lora_parameter_name
from latent_action_model.stage1.losses import (
    linear_neighbor_weight,
    neighborhood_distribution_loss,
    warmup_cosine_multiplier,
)
from latent_action_model.stage1.calibration import calibrate_student_gamma
from latent_action_model.stage1.memory_bank import LatentMemoryBank, Stage1NeighborProvider
from latent_action_model.stage1.teacher import TeacherGraph


def _release_state_dict(path: str | Path) -> Dict[str, Tensor]:
    """Load a release LAM state dict and normalize it to Lightning ``lam.*`` keys."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"release checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:  # PyTorch versions without mmap/weights_only.
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("release checkpoint must contain a mapping at top-level key `state_dict`")
    raw = payload["state_dict"]
    normalized: Dict[str, Tensor] = {}
    for key, value in raw.items():
        name = str(key)
        if name.startswith("lam."):
            normalized[name] = value
        else:
            normalized[f"lam.{name}"] = value
    return normalized


class EffectStage1LAM(VJEPA_LAM):
    """Fine-tune IDM/VAE/LaWM with an optional effect-geometry constraint.

    The release path remains untouched: this subclass only adds optimization,
    memory-bank, and neighborhood-loss behavior around ``LatentLAMModel``.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 4,
        enc_layers: int = 24,
        codebook_size: int = 32,
        code_dim: int = 32,
        max_state_dim: int = 14,
        num_frames: int = 2,
        num_queries: int = 1,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 12,
        dropout: float = 0.0,
        lambda_aux: float = 1.0,
        loss_type: str = "smooth_l1",
        state_loss_type: str = "l2",
        project: str = "lawam-stage1-effect-v1",
        task_name: str = "stage1_effect_v1",
        wandb_offline: bool = True,
        vision_model_id: str = "weights/dinov3-vitb16-pretrain-lvd1689m",
        norm_latents: bool = True,
        norm_latents_type: str = "ln",
        disable_vq: bool = False,
        vq_type: str = "vae",
        enc_add_state: bool = False,
        enc_modal_mask: bool = True,
        latent_layer_to_use: Any = -2,
        multi_input: bool = False,
        num_embodiments: int = 32,
        image_hw: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        image_aug: bool = True,
        dual_view_aug: bool = True,
        decoder_last_ln: bool = True,
        init_checkpoint: str = "",
        use_neighbor_constraint: bool = True,
        teacher_graph_path: str = "",
        latent_bank_path: str = "",
        neighbor_backend: str = "faiss",
        student_gamma: Optional[float] = None,
        teacher_bandwidth_scale: float = 0.30,
        teacher_bandwidth_floor: float = 0.0,
        neighbor_weight_target: float = 0.05,
        neighbor_zero_epochs: int = 1,
        neighbor_ramp_epochs: int = 4,
        neighbor_zero_steps: Optional[int] = None,
        neighbor_ramp_steps: Optional[int] = None,
        steps_per_epoch: int = 92,
        total_epochs: int = 40,
        total_training_steps: Optional[int] = None,
        full_latent_bank_refresh_epochs: Sequence[int] = (5, 10, 15, 20, 25, 30, 35),
        lr_idm: float = 3.0e-5,
        lr_vae: float = 3.0e-5,
        lr_lawm_decoder: float = 1.0e-5,
        lr_state_decoder: float = 3.0e-5,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1.0e-8,
        weight_decay: float = 1.0e-2,
        exclude_bias_norm_from_wd: bool = True,
        warmup_epochs: int = 1,
        warmup_steps: Optional[int] = None,
        min_lr_ratio: float = 0.1,
        kl_beta: float = 1.0e-5,
        lora_rank: int = 0,
        lora_alpha: float = 32.0,
        lora_train_vae: bool = True,
        **kwargs: Any,
    ) -> None:
        if vq_type not in {"vae", "beta_vae"}:
            raise ValueError("Effect Stage 1 requires a VAE bottleneck so the loss can use posterior mean mu.")
        merged_vq_kwargs = dict(vq_kwargs or {})
        merged_vq_kwargs["beta"] = float(kl_beta)
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            enc_layers=enc_layers,
            codebook_size=codebook_size,
            code_dim=code_dim,
            max_state_dim=max_state_dim,
            num_frames=num_frames,
            num_queries=num_queries,
            vq_kwargs=merged_vq_kwargs,
            dec_layers=dec_layers,
            dropout=dropout,
            lambda_aux=lambda_aux,
            loss_type=loss_type,
            state_loss_type=state_loss_type,
            project=project,
            task_name=task_name,
            wandb_offline=wandb_offline,
            optimizer=torch.optim.AdamW,
            weight_decay=weight_decay,
            exclude_bias_norm_from_wd=exclude_bias_norm_from_wd,
            vision_model_id=vision_model_id,
            warmup_steps=0,
            lambda_diversity=0.0,
            norm_latents=norm_latents,
            norm_latents_type=norm_latents_type,
            disable_vq=disable_vq,
            vq_type=vq_type,
            enc_add_state=enc_add_state,
            enc_modal_mask=enc_modal_mask,
            latent_layer_to_use=latent_layer_to_use,
            multi_input=multi_input,
            num_embodiments=num_embodiments,
            image_hw=image_hw,
            patch_size=patch_size,
            image_aug=image_aug,
            dual_view_aug=dual_view_aug,
            decoder_last_ln=decoder_last_ln,
            **kwargs,
        )

        self.init_checkpoint = str(init_checkpoint)
        self.use_neighbor_constraint = bool(use_neighbor_constraint)
        self.teacher_graph_path = str(teacher_graph_path)
        self.latent_bank_path = str(latent_bank_path)
        self.neighbor_backend = str(neighbor_backend)
        self.student_gamma = None if student_gamma is None else float(student_gamma)
        self.teacher_bandwidth_scale = float(teacher_bandwidth_scale)
        self.teacher_bandwidth_floor = float(teacher_bandwidth_floor)
        self.neighbor_weight_target = float(neighbor_weight_target)
        self.neighbor_zero_epochs = int(neighbor_zero_epochs)
        self.neighbor_ramp_epochs = int(neighbor_ramp_epochs)
        self.neighbor_zero_steps = (
            None if neighbor_zero_steps is None else int(neighbor_zero_steps)
        )
        self.neighbor_ramp_steps = (
            None if neighbor_ramp_steps is None else int(neighbor_ramp_steps)
        )
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)
        self.total_training_steps = (
            None if total_training_steps is None else int(total_training_steps)
        )
        self.full_latent_bank_refresh_epochs = tuple(int(x) for x in full_latent_bank_refresh_epochs)
        self.lr_idm = float(lr_idm)
        self.lr_vae = float(lr_vae)
        self.lr_lawm_decoder = float(lr_lawm_decoder)
        self.lr_state_decoder = float(lr_state_decoder)
        self.adamw_betas = tuple(float(x) for x in adamw_betas)
        self.adamw_eps = float(adamw_eps)
        self.stage1_weight_decay = float(weight_decay)
        self.stage1_exclude_bias_norm_from_wd = bool(exclude_bias_norm_from_wd)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_steps = None if warmup_steps is None else int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.kl_beta = float(kl_beta)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_train_vae = bool(lora_train_vae)
        self.lora_injection: Optional[LoRAInjection] = None
        self._neighbor_provider: Any = None
        self._restored_bank_state: Optional[Dict[str, Any]] = None

        if self.steps_per_epoch <= 0 or self.total_epochs <= 0:
            raise ValueError("steps_per_epoch and total_epochs must be positive")
        if self.total_training_steps is not None and self.total_training_steps <= 0:
            raise ValueError("total_training_steps must be positive when provided")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.total_epochs:
            raise ValueError("warmup_epochs must be in [0,total_epochs)")
        effective_total_steps = self.total_training_steps or self.total_epochs * self.steps_per_epoch
        if self.warmup_steps is not None and not 0 <= self.warmup_steps < effective_total_steps:
            raise ValueError("warmup_steps must be in [0,total_training_steps)")
        if self.neighbor_zero_steps is not None and self.neighbor_zero_steps < 0:
            raise ValueError("neighbor_zero_steps must be non-negative when provided")
        if self.neighbor_ramp_steps is not None and self.neighbor_ramp_steps <= 0:
            raise ValueError("neighbor_ramp_steps must be positive when provided")
        if len(self.adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly two values")
        if int(code_dim) != 32:
            raise ValueError(f"Locked V1 requires code_dim=32, got {code_dim}")
        if not math.isclose(self.teacher_bandwidth_scale, 0.30, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Locked V1 requires teacher_bandwidth_scale=0.30")
        if not math.isfinite(self.teacher_bandwidth_floor) or self.teacher_bandwidth_floor < 0:
            raise ValueError("teacher_bandwidth_floor must be finite and non-negative")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be non-negative")
        if self.lora_rank > 0 and (not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0):
            raise ValueError("lora_alpha must be finite and positive when LoRA is enabled")
        if not self.use_neighbor_constraint:
            if self.teacher_graph_path or self.latent_bank_path:
                raise ValueError(
                    "No-manifold Stage-1 requires empty teacher_graph_path and latent_bank_path"
                )
            if self.student_gamma is not None:
                raise ValueError("No-manifold Stage-1 requires student_gamma=None")

        if self.init_checkpoint:
            state = _release_state_dict(self.init_checkpoint)
            incompatible = self.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"strict release load failed: {incompatible}")

        # The beta is a Python scalar, not checkpoint state; enforce locked V1
        # after release initialization.
        self.lam.vq.beta = self.kl_beta
        if self.lora_rank > 0:
            self.lora_injection = inject_lora_into_encoder(
                self.lam.encoder,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
            )
        self._configure_trainable_modules()
        self.save_hyperparameters(
            {
                "init_checkpoint": self.init_checkpoint,
                "use_neighbor_constraint": self.use_neighbor_constraint,
                "teacher_graph_path": self.teacher_graph_path,
                "latent_bank_path": self.latent_bank_path,
                "neighbor_backend": self.neighbor_backend,
                "student_gamma": self.student_gamma,
                "teacher_bandwidth_scale": self.teacher_bandwidth_scale,
                "teacher_bandwidth_floor": self.teacher_bandwidth_floor,
                "neighbor_weight_target": self.neighbor_weight_target,
                "neighbor_zero_epochs": self.neighbor_zero_epochs,
                "neighbor_ramp_epochs": self.neighbor_ramp_epochs,
                "neighbor_zero_steps": self.neighbor_zero_steps,
                "neighbor_ramp_steps": self.neighbor_ramp_steps,
                "steps_per_epoch": self.steps_per_epoch,
                "total_epochs": self.total_epochs,
                "total_training_steps": self.total_training_steps,
                "full_latent_bank_refresh_epochs": self.full_latent_bank_refresh_epochs,
                "lr_idm": self.lr_idm,
                "lr_vae": self.lr_vae,
                "lr_lawm_decoder": self.lr_lawm_decoder,
                "lr_state_decoder": self.lr_state_decoder,
                "adamw_betas": self.adamw_betas,
                "adamw_eps": self.adamw_eps,
                "weight_decay": self.stage1_weight_decay,
                "exclude_bias_norm_from_wd": self.stage1_exclude_bias_norm_from_wd,
                "warmup_epochs": self.warmup_epochs,
                "warmup_steps": self.warmup_steps,
                "min_lr_ratio": self.min_lr_ratio,
                "kl_beta": self.kl_beta,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_train_vae": self.lora_train_vae,
            }
        )

    def _configure_trainable_modules(self) -> None:
        for parameter in self.lam.parameters():
            parameter.requires_grad = False
        if self.lora_rank > 0:
            for name, parameter in self.lam.encoder.named_parameters():
                if is_lora_parameter_name(name):
                    parameter.requires_grad = True
            if self.lora_train_vae:
                for parameter in self.lam.vq.parameters():
                    parameter.requires_grad = True
            self._validate_lora_trainability()
        else:
            for module in (self.lam.encoder, self.lam.vq, self.lam.decoder, self.lam.state_decoder):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        for parameter in self.lam.vision_encoder.parameters():
            parameter.requires_grad = False
        self.lam.vision_encoder.eval()

    def _validate_lora_trainability(self) -> None:
        if self.lora_injection is None or not self.lora_injection.target_names:
            raise RuntimeError("LoRA is enabled but no effect-encoder targets were registered")
        trainable = {name for name, parameter in self.lam.named_parameters() if parameter.requires_grad}
        expected_lora = {f"encoder.{name}" for name, parameter in self.lam.encoder.named_parameters() if is_lora_parameter_name(name)}
        expected_vae = (
            {f"vq.{name}" for name, parameter in self.lam.vq.named_parameters()}
            if self.lora_train_vae
            else set()
        )
        if not expected_lora:
            raise RuntimeError("LoRA is enabled but no trainable adapter parameters were found")
        if trainable != expected_lora | expected_vae:
            unexpected = sorted(trainable.difference(expected_lora | expected_vae))
            missing = sorted((expected_lora | expected_vae).difference(trainable))
            raise RuntimeError(f"LoRA trainability boundary mismatch: unexpected={unexpected}, missing={missing}")

    def lora_summary(self) -> Dict[str, Any]:
        trainable = sum(parameter.numel() for parameter in self.lam.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.lam.parameters())
        adapter = 0 if self.lora_injection is None else self.lora_injection.parameter_count
        return {
            "enabled": self.lora_rank > 0,
            "rank": self.lora_rank,
            "alpha": self.lora_alpha,
            "target": "effect_encoder",
            "target_weight_count": 0 if self.lora_injection is None else len(self.lora_injection.target_names),
            "adapter_parameter_count": adapter,
            "trainable_parameter_count": trainable,
            "total_parameter_count": total,
            "trainable_fraction": trainable / total if total else 0.0,
            "vae_head_trainable": self.lora_train_vae,
        }

    def train(self, mode: bool = True) -> "EffectStage1LAM":
        super().train(mode)
        # Lightning recursively enters train mode at every epoch.  The frozen
        # feature target must remain in eval mode for the entire Stage-1 run.
        self.lam.vision_encoder.eval()
        return self

    @property
    def neighbor_provider(self) -> Any:
        return self._neighbor_provider

    def attach_neighbor_provider(self, provider: Any) -> None:
        if not self.use_neighbor_constraint:
            raise RuntimeError("No-manifold Stage-1 cannot attach a neighbor provider")
        required = ("get", "update_bank", "rebuild_index")
        missing = [name for name in required if not hasattr(provider, name)]
        if missing:
            raise TypeError(f"neighbor provider is missing methods: {missing}")
        self._neighbor_provider = provider

    def on_fit_start(self) -> None:
        super_hook = getattr(super(), "on_fit_start", None)
        if callable(super_hook):
            super_hook()
        if self.lora_rank > 0 and self.trainer.is_global_zero:
            self.print(f"STAGE1_LORA {json.dumps(self.lora_summary(), sort_keys=True)}")
        if not self.use_neighbor_constraint:
            if self._neighbor_provider is not None:
                raise RuntimeError("No-manifold Stage-1 cannot use a neighbor provider")
            if self.trainer.is_global_zero:
                self.print("STAGE1_NEIGHBOR_CONSTRAINT_DISABLED")
            return
        if self._neighbor_provider is not None:
            return
        if not self.teacher_graph_path or not self.latent_bank_path:
            raise RuntimeError("Stage-1 fit requires teacher_graph_path and latent_bank_path artifacts.")
        datamodule = getattr(self.trainer, "datamodule", None)
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is None:
            raise RuntimeError("Stage-1 DataModule must initialize train_dataset before model.on_fit_start.")
        sample_ids = train_dataset.sample_ids
        graph = TeacherGraph.load(self.teacher_graph_path, expected_sample_ids=sample_ids)
        bank = LatentMemoryBank.load(
            self.latent_bank_path,
            expected_sample_ids=sample_ids,
            backend=self.neighbor_backend,
            full_refresh_epochs=self.full_latent_bank_refresh_epochs,
        )
        if self._restored_bank_state is not None:
            restored = self._restored_bank_state["latents"]
            if isinstance(restored, Tensor):
                restored = restored.detach().float().cpu().numpy()
            bank.replace_all(restored, epoch=int(self._restored_bank_state.get("epoch", 0)))
            bank.rebuild_index(epoch=int(self._restored_bank_state.get("epoch", 0)))
        provider = Stage1NeighborProvider(
            graph,
            bank,
            k_z=8,
            middle_count=8,
            far_count=8,
            teacher_bandwidth_scale=self.teacher_bandwidth_scale,
            teacher_bandwidth_floor=self.teacher_bandwidth_floor,
        )
        self.attach_neighbor_provider(provider)
        if self.student_gamma is None:
            calibration = calibrate_student_gamma(bank, neighbor_rank=8)
            self.student_gamma = calibration.gamma
            if self.trainer.is_global_zero:
                self.print(f"STAGE1_STUDENT_GAMMA {calibration.as_dict()}")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))

    def current_neighbor_weight(self) -> float:
        if not self.use_neighbor_constraint:
            return 0.0
        neighbor_zero_steps = getattr(self, "neighbor_zero_steps", None)
        neighbor_ramp_steps = getattr(self, "neighbor_ramp_steps", None)
        if neighbor_zero_steps is not None or neighbor_ramp_steps is not None:
            zero_steps = (
                self.neighbor_zero_epochs * self.steps_per_epoch
                if neighbor_zero_steps is None
                else neighbor_zero_steps
            )
            ramp_steps = (
                self.neighbor_ramp_epochs * self.steps_per_epoch
                if neighbor_ramp_steps is None
                else neighbor_ramp_steps
            )
            return linear_neighbor_weight(
                int(self.global_step),
                steps_per_epoch=1,
                target=self.neighbor_weight_target,
                zero_epochs=zero_steps,
                ramp_epochs=ramp_steps,
            )
        return linear_neighbor_weight(
            int(self.global_step),
            steps_per_epoch=self.steps_per_epoch,
            target=self.neighbor_weight_target,
            zero_epochs=self.neighbor_zero_epochs,
            ramp_epochs=self.neighbor_ramp_epochs,
        )

    @staticmethod
    def _batch_indices(batch: Dict[str, Any], *, device: torch.device) -> Tensor:
        value = batch.get("dataset_indices", batch.get("sample_indices"))
        if value is None:
            raise KeyError("Stage-1 batch requires `dataset_indices`.")
        indices = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if indices.ndim != 1:
            raise ValueError(f"dataset_indices must be [B], got {tuple(indices.shape)}")
        return indices.to(device=device, dtype=torch.long)

    def _distributed_bank_update(self, indices: Tensor, mu: Tensor) -> None:
        if self._neighbor_provider is None:
            return
        local_indices = indices.detach().to(device=self.device, dtype=torch.long)
        local_mu = mu.detach().to(device=self.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
            gathered_mu = [torch.empty_like(local_mu) for _ in range(world_size)]
            dist.all_gather(gathered_indices, local_indices)
            dist.all_gather(gathered_mu, local_mu)
            local_indices = torch.cat(gathered_indices, dim=0)
            local_mu = torch.cat(gathered_mu, dim=0)
        self._neighbor_provider.update_bank(
            local_indices.cpu().numpy(),
            local_mu.cpu().numpy(),
        )

    def _compute_neighborhood_loss(self, batch: Dict[str, Any], mu: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        weight = self.current_neighbor_weight()
        zero = mu.sum() * 0.0
        if self._neighbor_provider is None:
            if weight > 0:
                raise RuntimeError(
                    "Neighborhood weight is positive but no provider is attached. "
                    "Set teacher_graph_path/latent bank artifacts before Epoch 2."
                )
            return zero, {"neighbor_loss": zero.detach(), "neighbor_weight": mu.new_tensor(weight)}
        if self.student_gamma is None or self.student_gamma <= 0:
            if weight > 0:
                raise RuntimeError("A positive fixed student_gamma is required before neighborhood training starts.")
            return zero, {"neighbor_loss": zero.detach(), "neighbor_weight": mu.new_tensor(weight)}

        indices = self._batch_indices(batch, device=mu.device)
        live_mu = mu[:, 0] if mu.ndim == 3 and mu.shape[1] == 1 else mu
        neighbors = self._neighbor_provider.get(
            indices.detach().cpu().numpy(),
            live_mu.detach().float().cpu().numpy(),
            epoch=int(self.current_epoch) + 1,
        )
        candidate_mu = torch.as_tensor(neighbors.candidate_bank_mu, device=mu.device)
        teacher_distances = torch.as_tensor(neighbors.teacher_distances, device=mu.device)
        bandwidth = torch.as_tensor(neighbors.bandwidth, device=mu.device)
        candidate_mask = torch.as_tensor(neighbors.candidate_mask, device=mu.device)
        result = neighborhood_distribution_loss(
            live_mu,
            candidate_mu,
            teacher_distances,
            bandwidth,
            student_gamma=float(self.student_gamma),
            candidate_mask=candidate_mask,
            expected_latent_dim=32,
        )
        weighted = result.loss * float(weight)
        logs = {
            "neighbor_loss": result.loss.detach(),
            "neighbor_weight": mu.new_tensor(weight),
            "neighbor_weighted": weighted.detach(),
            "teacher_perplexity": result.teacher_perplexity.detach(),
            "neighbor_candidate_count": result.mean_candidate_count.detach(),
            "teacher_bandwidth": result.mean_teacher_bandwidth.detach(),
        }
        return weighted, logs

    def _compute_step(self, batch: Dict, vq_training: bool) -> Tuple[Tensor, Dict]:
        base_loss, logs = super()._compute_step(batch=batch, vq_training=vq_training)
        mu, logvar = self.lam.vq.get_last_posterior_stats()
        with torch.amp.autocast(device_type="cuda" if mu.is_cuda else "cpu", enabled=False):
            mu_fp32 = mu.float()
            logvar_fp32 = logvar.float()
            kl_raw = 0.5 * (mu_fp32.square() + logvar_fp32.exp() - 1.0 - logvar_fp32).sum(dim=-1).mean()
        logs = {**logs, "kl_raw": kl_raw.detach(), "kl_weight": mu.new_tensor(self.kl_beta)}

        if not vq_training:
            return base_loss, logs
        if not self.use_neighbor_constraint:
            zero = mu_fp32.sum() * 0.0
            return base_loss, {
                **logs,
                "neighbor_loss": zero.detach(),
                "neighbor_weight": zero.detach(),
                "neighbor_weighted": zero.detach(),
            }
        neighbor_term, neighbor_logs = self._compute_neighborhood_loss(batch, mu_fp32)
        indices = self._batch_indices(batch, device=mu.device)
        self._distributed_bank_update(indices, mu_fp32[:, 0] if mu_fp32.ndim == 3 else mu_fp32)
        return base_loss + neighbor_term, {**logs, **neighbor_logs}

    @torch.inference_mode()
    def _full_refresh_bank(self, epoch: int) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "full_train_dataloader"):
            raise RuntimeError("Full latent-bank refresh requires EffectWindowDataModule.full_train_dataloader().")
        was_training = self.training
        self.eval()
        local_indices: list[np.ndarray] = []
        local_mu: list[np.ndarray] = []
        try:
            for batch in datamodule.full_train_dataloader():
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                output = self.lam.get_latent_action(
                    videos=batch["videos"],
                    states=batch["states"],
                    dec_videos=batch["dec_videos"],
                    state_mask=batch.get("state_mask"),
                    embodiment_ids=batch.get("embodiment_ids"),
                    predict_future_frame=False,
                )
                mu = output.get("latent_mu")
                if mu is None:
                    raise RuntimeError("VAE did not expose latent_mu during full-bank refresh.")
                local_indices.append(batch["dataset_indices"].detach().cpu().numpy().astype(np.int64, copy=False))
                local_mu.append(mu.squeeze(1).float().cpu().numpy())
        finally:
            if was_training:
                self.train()

        payload = (
            np.concatenate(local_indices) if local_indices else np.empty(0, dtype=np.int64),
            np.concatenate(local_mu) if local_mu else np.empty((0, 32), dtype=np.float32),
        )
        if dist.is_available() and dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        bank = self._neighbor_provider.bank
        refreshed = np.empty_like(bank.latents)
        seen = np.zeros(len(refreshed), dtype=bool)
        for indices, values in gathered:
            if np.any(seen[indices]):
                raise RuntimeError("Full latent-bank refresh produced duplicate dataset indices.")
            refreshed[indices] = values
            seen[indices] = True
        if not np.all(seen):
            missing = np.flatnonzero(~seen)
            raise RuntimeError(f"Full latent-bank refresh missed {len(missing)} rows; first={missing[:8].tolist()}")
        bank.replace_all(refreshed, epoch=epoch)

    def on_train_epoch_end(self) -> None:
        if self._neighbor_provider is None:
            return
        epoch = int(self.current_epoch) + 1
        if self._neighbor_provider.bank.should_full_refresh(epoch):
            self._full_refresh_bank(epoch)
        self._neighbor_provider.rebuild_index(epoch)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.use_neighbor_constraint:
            if self._neighbor_provider is not None or self.student_gamma is not None:
                raise RuntimeError("No-manifold Stage-1 contains forbidden neighbor state")
            return
        if self._neighbor_provider is not None:
            checkpoint["stage1_latent_bank"] = {
                "latents": torch.from_numpy(self._neighbor_provider.bank.latents.copy()),
                "epoch": int(self.current_epoch) + 1,
            }
        checkpoint["stage1_student_gamma"] = self.student_gamma

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        restored = checkpoint.get("stage1_latent_bank")
        if not self.use_neighbor_constraint and restored is not None:
            raise RuntimeError("No-manifold Stage-1 cannot restore a latent memory bank")
        if restored is not None:
            self._restored_bank_state = restored
        if not self.use_neighbor_constraint and "stage1_student_gamma" in checkpoint:
            raise RuntimeError("No-manifold Stage-1 cannot restore student_gamma state")
        restored_gamma = checkpoint.get("stage1_student_gamma")
        if restored_gamma is not None:
            self.student_gamma = float(restored_gamma)

    @staticmethod
    def _split_decay_params(module: nn.Module) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
        norm_types = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        seen: set[int] = set()
        for submodule in module.modules():
            is_norm = isinstance(submodule, norm_types)
            for name, parameter in submodule.named_parameters(recurse=False):
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if is_norm or name == "bias" or name.endswith("bias"):
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
        expected = {id(p) for p in module.parameters() if p.requires_grad}
        if seen != expected:
            raise RuntimeError("optimizer grouping did not cover every trainable parameter exactly once")
        return decay, no_decay

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.lora_rank > 0:
            module_specs = (
                ("idm_lora", self.lam.encoder, self.lr_idm),
                ("vae", self.lam.vq, self.lr_vae),
            )
        else:
            module_specs = (
                ("idm", self.lam.encoder, self.lr_idm),
                ("vae", self.lam.vq, self.lr_vae),
                ("lawm_decoder", self.lam.decoder, self.lr_lawm_decoder),
                ("state_decoder", self.lam.state_decoder, self.lr_state_decoder),
            )
        groups: list[Dict[str, Any]] = []
        assigned: set[int] = set()
        for name, module, lr in module_specs:
            if module is None:
                continue
            decay, no_decay = self._split_decay_params(module)
            overlap = assigned.intersection({id(p) for p in decay + no_decay})
            if overlap:
                raise RuntimeError(f"parameters assigned to multiple optimizer groups in {name}")
            assigned.update(id(p) for p in decay + no_decay)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": self.stage1_weight_decay, "name": name})
            if no_decay:
                groups.append(
                    {
                        "params": no_decay,
                        "lr": lr,
                        "weight_decay": 0.0 if self.stage1_exclude_bias_norm_from_wd else self.stage1_weight_decay,
                        "name": f"{name}_no_decay",
                    }
                )
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if assigned != expected:
            raise RuntimeError("stage1 optimizer groups do not match the full trainable parameter set")

        optimizer = torch.optim.AdamW(
            groups,
            betas=self.adamw_betas,
            eps=self.adamw_eps,
            weight_decay=self.stage1_weight_decay,
        )
        total_steps = self.total_training_steps or self.total_epochs * self.steps_per_epoch
        warmup_steps = (
            self.warmup_epochs * self.steps_per_epoch
            if self.warmup_steps is None
            else self.warmup_steps
        )

        def lr_lambda(step: int) -> float:
            return warmup_cosine_multiplier(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=self.min_lr_ratio,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
