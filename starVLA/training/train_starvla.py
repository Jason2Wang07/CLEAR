# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate DDP, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.trainer_tools import build_per_group_scheduler
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from starVLA.training.stage2_phase import (
    STAGE2_PREFLIGHT_RECEIPT_ENV,
    STAGE2_SELECTION_SCHEMA,
    _sha256_file,
    resolve_stage2_phase_contract,
    validate_stage2_launch_preflight_receipt,
    validate_stage2_parent_checkpoint,
    validate_stage2_runtime_assets,
    validate_stage2_training_config,
    write_stage2_checkpoint_contract,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)

LEGACY_METRIC_KEY_ALIASES = {
    "train_total_loss": "train_loss_total",
    "train_flow_loss": "train_loss_flow",
    "train_perceptual_loss": "train_loss_perceptual",
    "train_distill_loss": "train_loss_distill",
    "train_mse_loss": "train_loss_mse",
    "train_lpips_loss": "train_loss_lpips",
    "val_total_loss": "val_loss_total",
    "val_flow_loss": "val_loss_flow",
    "val_perceptual_loss": "val_loss_perceptual",
    "val_distill_loss": "val_loss_distill",
    "val_mse_loss": "val_loss_mse",
    "val_lpips_loss": "val_loss_lpips",
}

TRAIN_COMPONENT_METRIC_ALIASES = {
    "train_loss_total": ("total_loss", "loss_total"),
    "train_loss_flow": ("loss_flow", "flow_loss"),
    "train_loss_perceptual": ("loss_perceptual", "perceptual_loss"),
    "train_loss_distill": ("loss_distill", "distill_loss"),
    "train_loss_lawam_distill": ("loss_lawam_distill",),
    "train_loss_mse": ("loss_mse", "mse_loss"),
    "train_loss_lpips": ("loss_lpips", "lpips_loss"),
    "train_loss_effect_distill": ("loss_effect_distill",),
    "train_loss_action_distill": ("loss_action_distill",),
    "train_loss_plan": ("loss_plan",),
    "train_loss_action_kl": ("loss_action_kl",),
    "train_loss_action_mean": ("loss_action_mean",),
    "train_loss_action_trajectory": ("loss_action_trajectory",),
    "train_loss_action_trajectory_position": ("loss_action_trajectory_position",),
    "train_loss_action_trajectory_rotation": ("loss_action_trajectory_rotation",),
    "train_loss_action_trajectory_gripper": ("loss_action_trajectory_gripper",),
    "train_stage2_residual_enabled": ("stage2_residual_enabled",),
    "train_stage2_residual_teacher_probability": (
        "stage2_residual_teacher_probability",
    ),
    "train_stage2_residual_teacher_fraction": ("stage2_residual_teacher_fraction",),
    "train_stage2_residual_keep_fraction": ("stage2_residual_keep_fraction",),
    "train_stage2_residual_predictor_detached": (
        "stage2_residual_predictor_detached",
    ),
}

def _accumulate_eval_scalar(metric_numerator: torch.Tensor, metric_denominator: torch.Tensor, value) -> None:
    """Accumulate an optional scalar metric as float64 sum/count."""
    if not torch.is_tensor(value):
        return
    scalar = value.detach()
    if scalar.numel() != 1:
        scalar = scalar.mean()
    metric_numerator += scalar.to(dtype=torch.float64)
    metric_denominator += 1


def _coerce_config_bool(value, *, default: bool, field_name: str) -> bool:
    """Parse config booleans explicitly to avoid `bool("false") == True` surprises."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for `{field_name}`: {value!r}")


def prune_process_checkpoints(
    checkpoint_dir: str | Path,
    *,
    keep_limit: int,
    preserve_steps: tuple[int, ...] = (),
) -> list[Path]:
    """Keep the newest rolling checkpoints plus explicit milestone steps."""

    if int(keep_limit) < 1:
        raise ValueError("process checkpoint keep_limit must be >= 1")
    protected_steps = {int(step) for step in preserve_steps}
    if any(step <= 0 for step in protected_steps):
        raise ValueError("process checkpoint preserve_steps must be positive")
    root = Path(checkpoint_dir)
    pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
    candidates: list[tuple[int, Path]] = []
    for candidate in root.glob("steps_*_pytorch_model.pt"):
        match = pattern.fullmatch(candidate.name)
        if match is not None and candidate.is_file():
            candidates.append((int(match.group(1)), candidate))
    candidates.sort(key=lambda item: item[0])
    rolling_candidates = [
        item for item in candidates if item[0] not in protected_steps
    ]
    deleted: list[Path] = []
    for _, checkpoint in rolling_candidates[:-int(keep_limit)]:
        sidecar = checkpoint.with_name(f"{checkpoint.name}.stage2.json")
        checkpoint.unlink()
        sidecar.unlink(missing_ok=True)
        deleted.append(checkpoint)
    return deleted


def should_save_process_checkpoint(
    completed_steps: int,
    *,
    save_interval: int,
    preserve_steps: tuple[int, ...] = (),
) -> bool:
    """Return whether this optimizer step is a periodic or named milestone."""

    step = int(completed_steps)
    interval = int(save_interval)
    if interval <= 0:
        raise ValueError("trainer.save_interval must be > 0")
    milestones = {int(value) for value in preserve_steps}
    if any(value <= 0 for value in milestones):
        raise ValueError("trainer.preserve_checkpoint_steps must contain positive steps")
    return step > 0 and (step % interval == 0 or step in milestones)


def build_accelerator(cfg) -> Accelerator:
    """build accelerator in DDP mode"""
    trainer_cfg = getattr(cfg, "trainer", None)
    gradient_accumulation_steps = int(getattr(trainer_cfg, "gradient_accumulation_steps", 1))
    ddp_find_unused_parameters = bool(getattr(trainer_cfg, "ddp_find_unused_parameters", True))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
    dataloader_config = DataLoaderConfiguration(non_blocking=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        dataloader_config=dataloader_config,
    )
    accelerator.print(accelerator.state)
    return accelerator


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    base_run_id = str(cfg.run_id)
    run_timestamp = os.environ.get("LAWAM_RUN_TIMESTAMP")
    if dist.is_initialized():
        timestamp_list = [run_timestamp if dist.get_rank() == 0 else None]
        if dist.get_rank() == 0 and not timestamp_list[0]:
            timestamp_list[0] = time.strftime("%m%d_%H%M%S")
        dist.broadcast_object_list(timestamp_list, src=0)
        timestamp = timestamp_list[0]
    else:
        timestamp = run_timestamp or time.strftime("%m%d_%H%M%S")
    run_folder_name = f"{timestamp}+{base_run_id}"

    cfg.output_dir = os.path.join(cfg.run_root_dir, run_folder_name)
    cfg.log_dir = os.path.join(cfg.output_dir, "logs")
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "logs", exist_ok=True)

        # Save the full merged config so `.pt` checkpoints remain reloadable for inference.
        serialized_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(serialized_cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r", encoding="utf-8") as f_yaml, open(
            output_dir / "config.json", "w", encoding="utf-8"
        ) as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloaders


def prepare_data(cfg, accelerator) -> tuple[DataLoader, Optional[DataLoader]]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader, vla_val_dataloader = build_dataloaders(cfg=cfg)

    if accelerator.dataloader_config is not None:
        accelerator.dataloader_config.dispatch_batches = False
    # if dist.is_initialized():
    #     dist.barrier()

    return vla_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    base_warmup = int(cfg.trainer.num_warmup_steps)
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            warmup = group.get("num_warmup_steps", base_warmup)
            logger.info(
                f"LR Group {group['name']}: lr={group['lr']}, "
                f"warmup_steps={warmup}, num_params={len(group['params'])}"
            )

    # initialize learning rate scheduler (supports per-group warmup steps)
    has_per_group_warmup = any("num_warmup_steps" in g for g in optimizer.param_groups)
    if has_per_group_warmup:
        lr_scheduler = build_per_group_scheduler(optimizer=optimizer, cfg=cfg)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
        )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        trackers = list(getattr(self.config, "trackers", [])) if hasattr(self.config, "trackers") else []
        self.use_wandb = "wandb" in trackers if trackers else True

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Strict finetune initialization from a full model checkpoint, if configured.
        self._init_checkpointing()

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        if self.vla_val_dataloader is None:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,  # must be the first param
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_val_dataloader = (
                self.setup_distributed_training(
                    self.accelerator,  # must be the first param
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.vla_val_dataloader,
                )
            )

        save_training_state = _coerce_config_bool(
            getattr(self.config.trainer, "save_training_state", False),
            default=False,
            field_name="trainer.save_training_state",
        )
        resume_training_state = getattr(self.config.trainer, "resume_training_state", None)
        if save_training_state or resume_training_state:
            # The scheduler is not passed to Accelerator.prepare(), so register it explicitly.
            self.accelerator.register_for_checkpointing(self.lr_scheduler)
        if resume_training_state:
            self._restore_training_state(str(resume_training_state))

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if not self.use_wandb:
            return
        if self.accelerator.is_main_process:
            mode = str(getattr(self.config, "wandb_mode", "offline")).strip().lower()
            required = _coerce_config_bool(
                getattr(self.config, "wandb_required", False),
                default=False,
                field_name="wandb_required",
            )
            if mode not in {"online", "offline"}:
                raise ValueError("wandb_mode must be `online` or `offline`")
            try:
                os.environ["WANDB_MODE"] = mode
                wandb.init(
                    name=Path(self.config.output_dir).name,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                    mode=mode,
                )
            except Exception as e:
                if required:
                    raise RuntimeError(
                        f"Required W&B {mode} initialization failed"
                    ) from e
                self.use_wandb = False
                logger.warning(f"W&B init failed, disable wandb logging for this run: {e}")

    def _init_checkpointing(self):
        """Initialize the checkpoint directory and optionally strict-load finetune init weights."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        resume_training_state = getattr(self.config.trainer, "resume_training_state", None)
        load_pretrained_policy_flow = _coerce_config_bool(
            getattr(self.config.trainer, "load_pretrained_policy_flow", True),
            default=True,
            field_name="trainer.load_pretrained_policy_flow",
        )
        load_pretrained_lam = _coerce_config_bool(
            getattr(self.config.trainer, "load_pretrained_lam", True),
            default=True,
            field_name="trainer.load_pretrained_lam",
        )
        strict_finetune_init = _coerce_config_bool(
            getattr(self.config.trainer, "strict_finetune_init", False),
            default=False,
            field_name="trainer.strict_finetune_init",
        )

        if resume_training_state:
            resume_path = Path(str(resume_training_state))
            if not resume_path.is_dir():
                raise FileNotFoundError(f"Training-state checkpoint does not exist: {resume_path}")
            self.completed_steps = 0
            logger.info(
                "Deferring model/optimizer/scheduler restore to prepared training state `%s`.",
                resume_path,
            )
        elif pretrained_checkpoint:
            self.model = self.load_finetune_init_weights(
                self.model,
                checkpoint_path=pretrained_checkpoint,
                load_pretrained_policy_flow=load_pretrained_policy_flow,
                load_pretrained_lam=load_pretrained_lam,
                strict_structure=strict_finetune_init,
            )
            self.completed_steps = 0
            logger.info(
                "Initialized model weights for finetune from `%s`; load_pretrained_policy_flow=%s; "
                "load_pretrained_lam=%s; strict_finetune_init=%s; training starts from step 0.",
                pretrained_checkpoint,
                load_pretrained_policy_flow,
                load_pretrained_lam,
                strict_finetune_init,
            )
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

        self._restored_training_state = False

    def _restore_training_state(self, training_state_dir: str) -> None:
        """Restore a complete Accelerate state saved at an optimizer boundary."""
        state_dir = Path(training_state_dir).resolve()
        match = re.fullmatch(r"steps_(\d+)_training_state", state_dir.name)
        if match is None:
            raise ValueError(
                "Training-state directory must be named `steps_<N>_training_state`, "
                f"got `{state_dir.name}`."
            )
        completed_steps = int(match.group(1))
        metadata_path = state_dir / "resume_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing resume metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != "lawam-accelerate-training-state-v2":
            raise ValueError(
                f"Unsupported training-state schema: {metadata.get('schema')!r}"
            )
        if int(metadata.get("completed_steps", -1)) != completed_steps:
            raise ValueError(
                f"Resume step mismatch: directory={completed_steps}, metadata={metadata.get('completed_steps')}"
            )
        if str(metadata.get("run_id")) != str(self.config.run_id):
            raise ValueError(
                f"Resume run_id mismatch: state={metadata.get('run_id')!r}, config={self.config.run_id!r}"
            )
        expected_contract = self._training_state_contract()
        saved_contract = metadata.get("contract")
        if saved_contract != expected_contract:
            raise ValueError(
                "Resume training contract mismatch: "
                f"state={saved_contract!r}, current={expected_contract!r}"
            )
        if completed_steps >= int(self.config.trainer.max_train_steps):
            raise ValueError(
                f"Resume step {completed_steps} must be below max_train_steps "
                f"{self.config.trainer.max_train_steps}."
            )

        self.accelerator.load_state(str(state_dir))
        self.completed_steps = completed_steps
        self._resume_metadata = metadata
        self._restored_training_state = True
        logger.info(
            "Restored complete training state from `%s` at optimizer step %d.",
            state_dir,
            completed_steps,
        )

    def _training_state_contract(self) -> dict[str, object]:
        """Return the immutable fields required for an exact optimizer-step resume."""
        pretrained_checkpoint = Path(
            str(self.config.trainer.pretrained_checkpoint)
        ).expanduser().resolve()
        action_model_cfg = self.config.framework.action_model
        contract = {
            "run_id": str(self.config.run_id),
            "stage2_arm": str(self.config.framework.action_model.stage2_arm),
            "seed": int(self.config.seed),
            "pretrained_checkpoint": str(pretrained_checkpoint),
            "pretrained_checkpoint_bytes": int(pretrained_checkpoint.stat().st_size),
            "max_train_steps": int(self.config.trainer.max_train_steps),
            "gradient_accumulation_steps": int(self.accelerator.gradient_accumulation_steps),
            "world_size": int(self.accelerator.num_processes),
            "per_device_batch_size": int(
                self.config.datasets.vla_data.per_device_batch_size
            ),
            "train_batches_per_rank": int(len(self.vla_train_dataloader)),
        }
        stage2_phase = getattr(action_model_cfg, "stage2_phase", None)
        if not bool(getattr(action_model_cfg, "stage2_action_injection_enabled", True)):
            contract["stage2_action_injection_enabled"] = False
        if stage2_phase is not None and str(stage2_phase).strip():
            contract["stage2_phase"] = str(stage2_phase)
            contract["stage2_parent_phase"] = str(
                getattr(self.config.trainer, "stage2_parent_phase", "")
            )
            training_recipe = str(
                getattr(action_model_cfg, "stage2_training_recipe", "") or ""
            )
            if training_recipe:
                contract["stage2_training_recipe"] = training_recipe
                phase_contract = resolve_stage2_phase_contract(action_model_cfg)
                contract["stage2_losses"] = phase_contract.losses.to_dict()
                contract["stage2_residual_curriculum"] = {
                    name: getattr(action_model_cfg, name)
                    for name in (
                        "stage2_residual_warmup_start_step",
                        "stage2_residual_teacher_prob_start",
                        "stage2_residual_teacher_prob_end",
                        "stage2_residual_teacher_ramp_steps",
                        "stage2_residual_predictor_grad_start_step",
                        "stage2_residual_condition_dropout",
                    )
                }
        return contract

    def _save_training_state(self) -> None:
        """Atomically save model, optimizer, scheduler, and per-rank RNG state."""
        # Validate metadata on every rank before writing the expensive state bundle.
        training_contract = self._training_state_contract()
        final_dir = Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_training_state"
        temp_dir = Path(self.checkpoint_dir) / f".steps_{self.completed_steps}_training_state.tmp"

        if self.accelerator.is_main_process:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if final_dir.exists():
                raise FileExistsError(f"Refusing to overwrite training state: {final_dir}")
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(str(temp_dir), safe_serialization=False)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            metadata = {
                "schema": "lawam-accelerate-training-state-v2",
                "run_id": str(self.config.run_id),
                "completed_steps": int(self.completed_steps),
                "contract": training_contract,
            }
            (temp_dir / "resume_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.rename(temp_dir, final_dir)

            # Only the newest complete state is needed for recovery. Model-only 10k
            # checkpoints remain untouched for matched checkpoint selection.
            for candidate in Path(self.checkpoint_dir).glob("steps_*_training_state"):
                if candidate != final_dir and candidate.is_dir():
                    shutil.rmtree(candidate)
            self.accelerator.print(f"Complete training state saved at {final_dir}")
        self.accelerator.wait_for_everyone()

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            model_checkpoint_path = Path(checkpoint_path + "_pytorch_model.pt")
            temporary_checkpoint = model_checkpoint_path.with_name(
                f".{model_checkpoint_path.name}.{os.getpid()}.tmp"
            )
            try:
                torch.save(state_dict, temporary_checkpoint)
                os.replace(temporary_checkpoint, model_checkpoint_path)
            finally:
                temporary_checkpoint.unlink(missing_ok=True)
            write_stage2_checkpoint_contract(
                checkpoint=model_checkpoint_path,
                cfg=self.config,
                completed_steps=self.completed_steps,
            )

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # Preserve the full merged config written by setup_directories(); it is
            # required to reload model-only checkpoints after migration. Keep the
            # access-tracker diagnostic as a separate file.
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("Saving accessed configuration diagnostic...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(
                    output_dir / "accessed_config.yaml",
                    use_original_values=False,
                )
                logger.info("Accessed configuration diagnostic saved")

        self.accelerator.wait_for_everyone()

        if _coerce_config_bool(
            getattr(self.config.trainer, "save_training_state", False),
            default=False,
            field_name="trainer.save_training_state",
        ):
            self._save_training_state()

        raw_keep_limit = getattr(
            self.config.trainer, "process_checkpoint_keep_limit", None
        )
        if raw_keep_limit is not None:
            keep_limit = int(raw_keep_limit)
            if self.accelerator.is_main_process:
                deleted = prune_process_checkpoints(
                    self.checkpoint_dir,
                    keep_limit=keep_limit,
                    preserve_steps=tuple(
                        int(step)
                        for step in (
                            getattr(
                                self.config.trainer,
                                "preserve_checkpoint_steps",
                                [],
                            )
                            or []
                        )
                    ),
                )
                if deleted:
                    self.accelerator.print(
                        "Pruned old process checkpoints: "
                        + ", ".join(path.name for path in deleted)
                    )
            self.accelerator.wait_for_everyone()

    def _maybe_save_phase_best(self, step_metrics: dict[str, float]) -> None:
        """Select a checkpoint using validation total loss only.

        Phased Stage-2 runs always enable this path. A non-phased baseline must
        opt in explicitly so existing training behavior remains unchanged.
        """

        phase_contract = resolve_stage2_phase_contract(
            self.config.framework.action_model
        )
        baseline_selection_enabled = _coerce_config_bool(
            getattr(
                self.config.trainer,
                "enable_validation_checkpoint_selection",
                False,
            ),
            default=False,
            field_name="trainer.enable_validation_checkpoint_selection",
        )
        if phase_contract is None and not baseline_selection_enabled:
            return
        stage2_arm = str(
            getattr(self.config.framework.action_model, "stage2_arm", "")
        ).lower()
        selection_tag = str(
            getattr(self.config.trainer, "validation_selection_tag", "")
        )
        if phase_contract is None:
            if stage2_arm != "b0":
                raise ValueError(
                    "Non-phased validation checkpoint selection is restricted to stage2_arm=b0"
                )
            if not selection_tag:
                raise ValueError(
                    "Baseline validation checkpoint selection requires "
                    "trainer.validation_selection_tag"
                )
        value = step_metrics.get("val_loss_total")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                "Stage-2 validation must produce finite `val_loss_total` for selection"
            )
        selection_path = Path(self.checkpoint_dir) / "best_validation_selection.json"
        checkpoint_path = Path(self.checkpoint_dir) / "best_validation_pytorch_model.pt"

        if self.accelerator.is_main_process:
            previous = None
            if selection_path.is_file():
                previous = json.loads(selection_path.read_text(encoding="utf-8"))
                contract_matches = previous.get("schema_version") == STAGE2_SELECTION_SCHEMA
                if phase_contract is not None:
                    contract_matches = contract_matches and (
                        previous.get("stage2_phase") == phase_contract.phase.value
                        and previous.get("stage2_training_recipe", "")
                        == phase_contract.training_recipe
                    )
                else:
                    contract_matches = contract_matches and (
                        previous.get("stage2_arm") == stage2_arm
                        and previous.get("selection_tag") == selection_tag
                    )
                if not contract_matches:
                    raise ValueError("existing Stage-2 best-selection contract mismatch")
            candidates = list(previous.get("candidates", [])) if previous else []
            candidate = {
                "step": int(self.completed_steps),
                "val_loss_total": float(value),
            }
            candidates.append(candidate)
            previous_value = (
                float(previous["selected"]["val_loss_total"])
                if previous is not None
                else math.inf
            )
            if float(value) < previous_value:
                temporary_checkpoint = checkpoint_path.with_name(
                    f".{checkpoint_path.name}.{os.getpid()}.tmp"
                )
                state_dict = self.accelerator.get_state_dict(self.model)
                try:
                    torch.save(state_dict, temporary_checkpoint)
                    os.replace(temporary_checkpoint, checkpoint_path)
                finally:
                    temporary_checkpoint.unlink(missing_ok=True)
                sidecar_path = write_stage2_checkpoint_contract(
                    checkpoint=checkpoint_path,
                    cfg=self.config,
                    completed_steps=self.completed_steps,
                )
                selected = {
                    **candidate,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_bytes": int(checkpoint_path.stat().st_size),
                }
                if phase_contract is not None:
                    if sidecar_path is None:
                        raise AssertionError("phase checkpoint did not produce a phase sidecar")
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    selected.update(
                        {
                            "checkpoint_sha256": sidecar["checkpoint_sha256"],
                            "phase_contract": str(sidecar_path.resolve()),
                        }
                    )
                else:
                    if sidecar_path is not None:
                        raise AssertionError("B0 checkpoint unexpectedly produced a phase sidecar")
                    selected["checkpoint_sha256"] = _sha256_file(checkpoint_path)
            else:
                selected = dict(previous["selected"])
            selection = {
                "schema_version": STAGE2_SELECTION_SCHEMA,
                "selection_split": "validation",
                "selection_metric": "val_loss_total",
                "selection_mode": "min",
                "test_metrics_read": False,
                "selected": selected,
                "candidates": candidates,
                "last_evaluated_step": int(self.completed_steps),
            }
            if phase_contract is not None:
                selection["stage2_phase"] = phase_contract.phase.value
                if phase_contract.training_recipe:
                    selection["stage2_training_recipe"] = phase_contract.training_recipe
            else:
                selection.update(
                    {
                        "stage2_arm": stage2_arm,
                        "selection_tag": selection_tag,
                    }
                )
            temporary_selection = selection_path.with_name(
                f".{selection_path.name}.{os.getpid()}.tmp"
            )
            try:
                temporary_selection.write_text(
                    json.dumps(selection, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_selection, selection_path)
            finally:
                temporary_selection.unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()

    def _get_learning_rate_metrics(self):
        """Collect learning-rate metrics for all optimizer parameter groups."""
        lr_metrics = {}
        for idx, group in enumerate(self.optimizer.param_groups):
            group_name = str(group.get("name") or f"group_{idx}")
            lr_metrics[f"learning_rate/{group_name}"] = float(group["lr"])

        if self.optimizer.param_groups:
            lr_metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])

        return lr_metrics

    def _normalize_metric_aliases(self, metrics):
        """Collapse legacy metric spellings onto the canonical W&B keys."""
        normalized_metrics = dict(metrics)
        for alias_key, canonical_key in LEGACY_METRIC_KEY_ALIASES.items():
            if alias_key not in normalized_metrics:
                continue
            if canonical_key not in normalized_metrics:
                normalized_metrics[canonical_key] = normalized_metrics[alias_key]
            normalized_metrics.pop(alias_key, None)
        return normalized_metrics

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                metrics = self._normalize_metric_aliases(metrics)
                # Log both the legacy scalar learning rate and named lr groups.
                metrics.update(self._get_learning_rate_metrics())

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                if self.use_wandb:
                    wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        if not self._restored_training_state or self.completed_steps == 0:
            self.vla_iter = iter(self.vla_train_dataloader)
            return

        batches_per_epoch = len(self.vla_train_dataloader)
        if batches_per_epoch <= 0:
            raise ValueError("Cannot restore data position from an empty training dataloader.")
        consumed_batches = self.completed_steps * int(self.accelerator.gradient_accumulation_steps)
        saved_batches_per_epoch = int(
            self._resume_metadata["contract"]["train_batches_per_rank"]
        )
        if batches_per_epoch != saved_batches_per_epoch:
            raise ValueError(
                "Resume dataloader length mismatch: "
                f"state={saved_batches_per_epoch}, current={batches_per_epoch}"
            )
        epoch_count, batches_into_epoch = divmod(consumed_batches, batches_per_epoch)
        if hasattr(self.vla_train_dataloader, "sampler") and callable(
            getattr(self.vla_train_dataloader.sampler, "set_epoch", None)
        ):
            self.vla_train_dataloader.sampler.set_epoch(epoch_count)
        if hasattr(self.vla_train_dataloader, "dataset"):
            TrainerUtils._set_epoch_on_dataset_tree(self.vla_train_dataloader.dataset, epoch_count)
        self.vla_epoch_count = epoch_count

        resume_dataloader = self.vla_train_dataloader
        if batches_into_epoch:
            resume_dataloader = self.accelerator.skip_first_batches(
                self.vla_train_dataloader,
                batches_into_epoch,
            )
        self.vla_iter = iter(resume_dataloader)
        logger.info(
            "Restored train data cursor: epoch=%d batch=%d/%d (optimizer_step=%d).",
            epoch_count,
            batches_into_epoch,
            batches_per_epoch,
            self.completed_steps,
        )

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            desc=str(getattr(self.config, "run_id", "train")),
            disable=not self.accelerator.is_local_main_process,
            initial=self.completed_steps,
        )

        perf_smoke_enabled = os.environ.get("LAWAM_PERF_SMOKE", "0") == "1"
        perf_step_started_at = time.perf_counter()
        if perf_smoke_enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # Wait for the next batch from the prepared dataloader.
            # Under Accelerate device placement, host-to-device copies happen before the batch is yielded,
            # so the displayed `data_times` includes any H2D transfer done by the dataloader wrapper.
            t_start_data_wait = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data_wait = time.perf_counter()

            # Pure training-step compute time for the current process.
            t_start_compute = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_compute = time.perf_counter()

            # update progress
            step_advanced = False
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                step_advanced = True

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data_wait - t_start_data_wait:.3f}",
                            "model_times": f"{t_end_compute - t_start_compute:.3f}",
                        }
                    )

            if step_advanced:
                if perf_smoke_enabled:
                    perf_values = torch.tensor(
                        [
                            time.perf_counter() - perf_step_started_at,
                            torch.cuda.max_memory_allocated(self.accelerator.device)
                            / (1024**3),
                            torch.cuda.max_memory_reserved(self.accelerator.device)
                            / (1024**3),
                        ],
                        dtype=torch.float64,
                        device=self.accelerator.device,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(perf_values, op=dist.ReduceOp.MAX)
                    if self.accelerator.is_main_process:
                        logger.info(
                            "LAWAM_PERF_SMOKE %s",
                            json.dumps(
                                {
                                    "optimizer_step": self.completed_steps,
                                    "wall_sec_max_rank": perf_values[0].item(),
                                    "peak_allocated_gib_max_rank": perf_values[1].item(),
                                    "peak_reserved_gib_max_rank": perf_values[2].item(),
                                },
                                sort_keys=True,
                            ),
                        )
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(self.accelerator.device)
                    perf_step_started_at = time.perf_counter()

                # evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)
                    self._maybe_save_phase_best(step_metrics)

                # record metrics
                step_metrics["data_time"] = t_end_data_wait - t_start_data_wait
                step_metrics["model_time"] = t_end_compute - t_start_compute
                self._log_metrics(step_metrics)

                # save checkpoint
                preserve_checkpoint_steps = tuple(
                    int(step)
                    for step in (
                        getattr(
                            self.config.trainer,
                            "preserve_checkpoint_steps",
                            [],
                        )
                        or []
                    )
                )
                if should_save_process_checkpoint(
                    self.completed_steps,
                    save_interval=int(self.config.trainer.save_interval),
                    preserve_steps=preserve_checkpoint_steps,
                ):
                    self._save_checkpoint()
                elif _coerce_config_bool(
                    getattr(self.config.trainer, "save_training_state", False),
                    default=False,
                    field_name="trainer.save_training_state",
                ):
                    state_interval = int(
                        getattr(
                            self.config.trainer,
                            "training_state_interval",
                            self.config.trainer.save_interval,
                        )
                    )
                    if state_interval <= 0:
                        raise ValueError("trainer.training_state_interval must be > 0.")
                    if self.completed_steps % state_interval == 0 and self.completed_steps > 0:
                        self._save_training_state()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """
        if step_metrics is None:
            step_metrics = {}

        if self.vla_val_dataloader is None:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.warning("Validation dataloader is None, skip eval_action_model for this step.")
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        eval_batches = int(getattr(self.config.trainer, "eval_batches", 50))
        if eval_batches <= 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.warning(f"`trainer.eval_batches` must be > 0, got {eval_batches}; skip eval.")
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        eval_model = self.accelerator.unwrap_model(self.model)
        model_was_training = bool(getattr(eval_model, "training", True))
        eval_model.eval()

        if not hasattr(eval_model, "policy_runner"):
            if self.accelerator.is_main_process:
                logger.warning(
                    "eval_action_model requires `model.policy_runner.infer_step_with_aligned_targets_from_train_batch`; "
                    "skip eval for framework `%s`.",
                    str(getattr(getattr(self.config, "framework", None), "name", "")),
                )
            if model_was_training:
                eval_model.train()
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        metric_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        metric_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_total_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_total_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_perceptual_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_perceptual_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_distill_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_distill_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_lawam_distill_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_lawam_distill_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_effect_distill_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_effect_distill_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_action_distill_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_action_distill_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_iter = iter(self.vla_val_dataloader)
        policy_backend = getattr(eval_model, "policy_backend", None)
        if policy_backend is not None and hasattr(policy_backend, "set_flow_train_step"):
            policy_backend.set_flow_train_step(self.completed_steps)

        try:
            with torch.no_grad():
                for _ in range(eval_batches):
                    try:
                        batch = next(val_iter)
                    except StopIteration:
                        break

                    try:
                        eval_output_dict = eval_model.forward(batch)
                    except Exception as exc:
                        if self.accelerator.is_main_process:
                            logger.warning(f"LatentWorld eval skip forward loss metrics due to runtime error: {exc}")
                    else:
                        _accumulate_eval_scalar(
                            val_loss_total_numerator,
                            val_loss_total_denominator,
                            eval_output_dict.get("total_loss"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_perceptual_numerator,
                            val_loss_perceptual_denominator,
                            eval_output_dict.get("loss_perceptual"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_distill_numerator,
                            val_loss_distill_denominator,
                            eval_output_dict.get("loss_distill"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_lawam_distill_numerator,
                            val_loss_lawam_distill_denominator,
                            eval_output_dict.get("loss_lawam_distill"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_effect_distill_numerator,
                            val_loss_effect_distill_denominator,
                            eval_output_dict.get("loss_effect_distill"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_action_distill_numerator,
                            val_loss_action_distill_denominator,
                            eval_output_dict.get("loss_action_distill"),
                        )

                    try:
                        pred_actions, gt_actions, action_mask = (
                            eval_model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
                        )
                    except Exception as exc:
                        if self.accelerator.is_main_process:
                            logger.warning(f"LatentWorld eval skip action MSE metrics due to runtime error: {exc}")
                        continue

                    if pred_actions.shape != gt_actions.shape or pred_actions.shape != action_mask.shape:
                        if self.accelerator.is_main_process:
                            logger.warning(
                                "LatentWorld eval skip batch due to shape mismatch: "
                                f"pred={tuple(pred_actions.shape)}, "
                                f"gt={tuple(gt_actions.shape)}, "
                                f"mask={tuple(action_mask.shape)}"
                            )
                        continue

                    valid_count = action_mask.sum()
                    if int(valid_count.item()) <= 0:
                        if self.accelerator.is_main_process:
                            logger.warning("LatentWorld eval skip batch: empty valid action mask.")
                        continue

                    sq_error_sum = ((pred_actions - gt_actions) ** 2).masked_select(action_mask).sum()
                    metric_numerator += sq_error_sum.to(dtype=torch.float64)
                    metric_denominator += valid_count.to(dtype=torch.float64)
        finally:
            if model_was_training:
                eval_model.train()

        if dist.is_initialized():
            dist.all_reduce(metric_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(metric_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_total_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_total_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_perceptual_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_perceptual_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_distill_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_distill_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_lawam_distill_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_lawam_distill_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_effect_distill_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_effect_distill_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_action_distill_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_action_distill_denominator, op=dist.ReduceOp.SUM)

        if float(val_loss_total_denominator.item()) > 0:
            step_metrics["val_loss_total"] = float(
                (val_loss_total_numerator / val_loss_total_denominator).item()
            )
        elif resolve_stage2_phase_contract(self.config.framework.action_model) is not None:
            raise RuntimeError(
                "Phased Stage-2 validation produced no `total_loss`; best selection is undefined"
            )

        if self.accelerator.is_main_process:
            if float(metric_denominator.item()) > 0:
                val_mse_score = float((metric_numerator / metric_denominator).item())
                step_metrics["val_mse_score"] = val_mse_score
            else:
                logger.warning(
                    "Validation eval produced no valid batches/elements; skip metric logging for this eval step."
                )

            if float(val_loss_perceptual_denominator.item()) > 0:
                step_metrics["val_loss_perceptual"] = float(
                    (val_loss_perceptual_numerator / val_loss_perceptual_denominator).item()
                )
            else:
                logger.warning("Validation eval produced no valid batches for `val_loss_perceptual`.")

            if float(val_loss_distill_denominator.item()) > 0:
                step_metrics["val_loss_distill"] = float(
                    (val_loss_distill_numerator / val_loss_distill_denominator).item()
                )
            else:
                logger.warning("Validation eval produced no valid batches for `val_loss_distill`.")

            if float(val_loss_lawam_distill_denominator.item()) > 0:
                step_metrics["val_loss_lawam_distill"] = float(
                    (val_loss_lawam_distill_numerator / val_loss_lawam_distill_denominator).item()
                )

            if float(val_loss_effect_distill_denominator.item()) > 0:
                step_metrics["val_loss_effect_distill"] = float(
                    (val_loss_effect_distill_numerator / val_loss_effect_distill_denominator).item()
                )
            if float(val_loss_action_distill_denominator.item()) > 0:
                step_metrics["val_loss_action_distill"] = float(
                    (val_loss_action_distill_numerator / val_loss_action_distill_denominator).item()
                )

        if dist.is_initialized():
            dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla):
        """execute single training step"""
        # Guard against mode leakage from eval_action_model().
        self.model.train()
        base_model = self.accelerator.unwrap_model(self.model)
        policy_backend = getattr(base_model, "policy_backend", None)
        if policy_backend is not None and hasattr(policy_backend, "set_flow_train_step"):
            policy_backend.set_flow_train_step(self.completed_steps)
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            output_dict = self.model.forward(batch_vla)
            if "total_loss" not in output_dict:
                raise KeyError("Model forward must return `total_loss`.")
            total_loss = output_dict["total_loss"]

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # Clip gradients only on the accumulation boundary right before the real optimizer update.
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()

        metrics = {"train_loss": float(total_loss.detach().item())}
        # Optional component losses with stable logging keys.
        for log_key, raw_keys in TRAIN_COMPONENT_METRIC_ALIASES.items():
            for raw_key in raw_keys:
                if raw_key in output_dict and torch.is_tensor(output_dict[raw_key]):
                    metrics[log_key] = float(output_dict[raw_key].detach().item())
                    break
        return self._normalize_metric_aliases(metrics)

    def _finalize_training(self):
        """training end processing"""
        # save final model
        save_final_model = _coerce_config_bool(
            getattr(self.config.trainer, "save_final_model", True),
            default=True,
            field_name="trainer.save_final_model",
        )
        if self.accelerator.is_main_process and save_final_model:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            model_checkpoint_path = os.path.join(final_checkpoint, "pytorch_model.pt")
            torch.save(state_dict, model_checkpoint_path)
            write_stage2_checkpoint_contract(
                checkpoint=model_checkpoint_path,
                cfg=self.config,
                completed_steps=self.completed_steps,
            )
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        elif self.accelerator.is_main_process:
            logger.info("Training complete. Final model save disabled by trainer.save_final_model=false.")


        # close W&B
        if self.accelerator.is_main_process and self.use_wandb:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    # Validate phase/loss/freeze/checkpoint contracts before constructing the
    # multi-billion-parameter model or allocating optimizer state.
    phase_contract = validate_stage2_training_config(cfg)
    preflight_receipt = os.environ.get(STAGE2_PREFLIGHT_RECEIPT_ENV)
    if phase_contract is not None and preflight_receipt:
        validate_stage2_launch_preflight_receipt(cfg, preflight_receipt, phase_contract)
    else:
        distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if phase_contract is not None and distributed_world_size > 1:
            raise RuntimeError(
                "distributed phased Stage-2 requires a single-process launch preflight; "
                f"set {STAGE2_PREFLIGHT_RECEIPT_ENV} via "
                "scripts/train_libero_stage2_phased.sh"
            )
        validate_stage2_runtime_assets(cfg, phase_contract)
        validate_stage2_parent_checkpoint(cfg, phase_contract)
    accelerator = build_accelerator(cfg)
    logger.info("VLA Training :: Warming Up")

    # New Stage-2 modules must initialize identically across matched arms.
    seed = int(getattr(cfg, "seed", 3047)) + int(accelerator.process_index)
    set_seed(seed)

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    vla = apply_training_freeze_policy(vla, cfg)
    # prepare data
    vla_train_dataloader, vla_val_dataloader = prepare_data(cfg=cfg, accelerator=accelerator)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_train_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
