from __future__ import annotations

from typing import Any, Sequence

import torch

from latent_action_model.bridge_stage2.temporal import (
    is_bridge_stage2_temporal_contract,
)
from latent_action_model.libero_stage2.runtime_contract import (
    is_libero_04s_adapter_temporal_contract,
)
from latent_action_model.robotwin_stage2.temporal import (
    is_robotwin_stage2_temporal_contract,
    resolve_stage2_temporal_contract,
)
from starVLA.model.framework.latent_world.batch_utils import (
    align_feature_dim_with_mask,
    apply_shared_color_jitter_uint8,
    apply_shared_random_resized_crop_uint8,
    build_placeholder_masks,
    imagenet_normalize_video_,
    sample_or_pad_sequence_with_mask,
)
from starVLA.model.framework.latent_world.processor_utils import (
    LatentWorldProcessorSpec,
    load_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyTrainBatch, LatentWorldPolicyTrainRawSample
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages
from starVLA.model.framework.latent_world.vlm_adapter import (
    DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
)


_TEACHER_ACTION_VAE_FIELDS = (
    "teacher_trajectory",
    "teacher_current_state",
    "teacher_time_mask",
    "teacher_pose_mask",
    "teacher_gripper_mask",
    "teacher_current_state_mask",
)


class LatentWorldTrainCollator:
    def __init__(
        self,
        *,
        policy_cfg: Any,
        processor_spec: LatentWorldProcessorSpec,
        act_queries: int,
        flow_queries: int,
        enable_primary_video_aug: bool,
        enable_primary_random_resized_crop: bool,
        cot_prompt_before_wrist: str | None = None,
        cot_prompt_after_wrist: str | None = None,
    ) -> None:
        self.policy_cfg = policy_cfg
        self.stage2_temporal_contract = resolve_stage2_temporal_contract(
            getattr(policy_cfg, "stage2_temporal_contract", None)
        )
        self.processor_spec = processor_spec
        self.placeholder_token = str(processor_spec.placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.enable_primary_video_aug = bool(enable_primary_video_aug)
        self.enable_primary_random_resized_crop = bool(enable_primary_random_resized_crop)
        self.cot_prompt_before_wrist = (
            str(cot_prompt_before_wrist)
            if cot_prompt_before_wrist is not None
            else DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT
        )
        self.cot_prompt_after_wrist = (
            str(cot_prompt_after_wrist)
            if cot_prompt_after_wrist is not None
            else DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT
        )
        self.training = True
        self._processor: Any | None = None
        self._placeholder_token_id: int | None = None

    def train(self) -> "LatentWorldTrainCollator":
        self.training = True
        return self

    def eval(self) -> "LatentWorldTrainCollator":
        self.training = False
        return self

    def _ensure_processor(self) -> tuple[Any, int]:
        if self._processor is None:
            processor, _, placeholder_token_id = load_latent_world_processor(self.processor_spec)
            self._processor = processor
            self._placeholder_token_id = placeholder_token_id
        assert self._processor is not None
        assert self._placeholder_token_id is not None
        return self._processor, self._placeholder_token_id

    @staticmethod
    def _to_wrist_view_list(wrist_images: torch.Tensor) -> list[torch.Tensor]:
        # Match inference semantics: missing wrist views are represented as an empty view list.
        if int(wrist_images.shape[0]) == 0:
            return []
        return [img.contiguous() for img in wrist_images]

    @staticmethod
    def _to_primary_view_list(primary_videos: torch.Tensor) -> list[torch.Tensor]:
        if primary_videos.ndim != 5 or int(primary_videos.shape[1]) < 1:
            raise ValueError(
                "Expected primary_videos with shape [V, T, C, H, W] and T>=1, "
                f"got {tuple(primary_videos.shape)}."
            )
        return [video[0].contiguous() for video in primary_videos]

    @staticmethod
    def _augment_primary_and_wrist_views(
        *,
        primary_videos: torch.Tensor,
        wrist_images: torch.Tensor,
        enable_random_resized_crop: bool,
        enable_color_jitter: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary_videos.ndim != 5:
            raise ValueError(f"Expected primary_videos with shape [V, T, C, H, W], got {tuple(primary_videos.shape)}.")
        primary_view_count = int(primary_videos.shape[0])
        primary_frame_count = int(primary_videos.shape[1])
        flat_primary = primary_videos.reshape(primary_view_count * primary_frame_count, *primary_videos.shape[-3:])

        stacked_views = [flat_primary]
        if int(wrist_images.shape[0]) > 0:
            stacked_views.append(wrist_images)
        augmented_views = torch.cat(stacked_views, dim=0)
        if enable_random_resized_crop:
            augmented_views = apply_shared_random_resized_crop_uint8(augmented_views)
        if enable_color_jitter:
            augmented_views = apply_shared_color_jitter_uint8(augmented_views)

        primary_flat_count = int(flat_primary.shape[0])
        processed_primary_videos = augmented_views[:primary_flat_count].reshape_as(primary_videos).contiguous()
        processed_wrist_images = augmented_views[primary_flat_count:].contiguous()
        return processed_primary_videos, processed_wrist_images

    def __call__(self, features: Sequence[LatentWorldPolicyTrainRawSample]) -> LatentWorldPolicyTrainBatch:
        if len(features) == 0:
            raise ValueError("LatentWorldTrainCollator received an empty feature list.")

        condition_fields = [
            "z_lawam" if "z_lawam" in sample else "z_effect" if "z_effect" in sample else None
            for sample in features
        ]
        if any("z_lawam" in sample and "z_effect" in sample for sample in features):
            raise ValueError("a sample cannot provide both z_lawam and z_effect")
        present_condition_fields = {field for field in condition_fields if field is not None}
        if len(present_condition_fields) > 1:
            raise ValueError("all paired samples must use the same condition latent field")
        condition_field = next(iter(present_condition_fields), None)
        pair_presence = [
            (field is not None, "z_action" in sample)
            for field, sample in zip(condition_fields, features)
        ]
        if any(condition != action for condition, action in pair_presence):
            raise ValueError("condition latent and z_action must be provided together")
        paired_enabled = all(condition and action for condition, action in pair_presence)
        if not paired_enabled and any(condition or action for condition, action in pair_presence):
            raise ValueError("paired teacher tokens must be present for every sample in a batch")
        raw_condition_field = (
            f"{condition_field}_raw" if condition_field is not None else None
        )
        raw_condition_presence = [
            raw_condition_field is not None and raw_condition_field in sample
            for sample in features
        ]
        if any(raw_condition_presence) and not all(raw_condition_presence):
            raise ValueError("raw condition latent must be present for every sample or none")
        if any(raw_condition_presence) and not paired_enabled:
            raise ValueError("raw condition latent requires paired teacher tokens")
        action_logvar_presence = ["z_action_logvar" in sample for sample in features]
        if any(action_logvar_presence) and not all(action_logvar_presence):
            raise ValueError("z_action_logvar must be present for every sample or none")
        if any(action_logvar_presence) and not paired_enabled:
            raise ValueError("z_action_logvar requires paired teacher tokens")
        explicit_action_masks = ["action_mask" in sample for sample in features]
        if any(explicit_action_masks) and not all(explicit_action_masks):
            raise ValueError("action_mask must be present for every sample or none")
        explicit_state_masks = ["state_mask" in sample for sample in features]
        if any(explicit_state_masks) and not all(explicit_state_masks):
            raise ValueError("state_mask must be present for every sample or none")
        teacher_field_presence = [
            tuple(name in sample for name in _TEACHER_ACTION_VAE_FIELDS)
            for sample in features
        ]
        teacher_fields_enabled = any(any(presence) for presence in teacher_field_presence)
        if teacher_fields_enabled and not all(all(presence) for presence in teacher_field_presence):
            raise ValueError(
                "all ActionVAE teacher fields must be present for every sample: "
                f"{list(_TEACHER_ACTION_VAE_FIELDS)}"
            )
        action_predictor_state_presence = [
            "action_predictor_state" in sample for sample in features
        ]
        if any(action_predictor_state_presence) and not all(action_predictor_state_presence):
            raise ValueError("action_predictor_state must be present for every sample or none")
        action_predictor_state_enabled = all(action_predictor_state_presence)
        if action_predictor_state_enabled != teacher_fields_enabled:
            raise ValueError(
                "action_predictor_state and the ActionVAE teacher fields must be provided together"
            )
        physical_count_presence = ["physical_valid_count" in sample for sample in features]
        if any(physical_count_presence) and not all(physical_count_presence):
            raise ValueError("physical_valid_count must be present for every sample or none")
        physical_count_enabled = all(physical_count_presence)
        if physical_count_enabled and not teacher_fields_enabled:
            raise ValueError("physical_valid_count requires the ActionVAE teacher fields")
        if (
            (
                is_libero_04s_adapter_temporal_contract(self.stage2_temporal_contract)
                or is_bridge_stage2_temporal_contract(self.stage2_temporal_contract)
            )
            and not physical_count_enabled
        ):
            raise ValueError("resampled Stage-2 samples require physical_valid_count")

        image_views_batch: list[list[Any]] = []
        wrist_image_views_batch: list[list[Any]] = []
        primary_video_uint8_seqs: list[torch.Tensor] = []
        instructions: list[str] = []
        actions_list: list[torch.Tensor] = []
        explicit_action_mask_list: list[torch.Tensor] = []
        states_list: list[torch.Tensor] = []
        explicit_state_mask_list: list[torch.Tensor] = []
        embodiment_ids: list[int] = []
        action_hz_list: list[float] = []
        paired_conditions: list[torch.Tensor] = []
        paired_raw_conditions: list[torch.Tensor] = []
        paired_actions: list[torch.Tensor] = []
        paired_action_logvars: list[torch.Tensor] = []
        teacher_trajectories: list[torch.Tensor] = []
        teacher_current_states: list[torch.Tensor] = []
        teacher_time_masks: list[torch.Tensor] = []
        teacher_pose_masks: list[torch.Tensor] = []
        teacher_gripper_masks: list[torch.Tensor] = []
        teacher_current_state_masks: list[torch.Tensor] = []
        action_predictor_states: list[torch.Tensor] = []
        physical_valid_counts: list[int] = []

        for sample in features:
            primary_videos = sample["primary_videos"].contiguous()
            wrist_images = sample["wrist_images"].contiguous()
            state_tensor = sample["state"].to(dtype=torch.float32).contiguous()
            action_tensor = sample["action"].to(dtype=torch.float32).contiguous()

            instruction = str(sample["lang"])
            embodiment_id = int(sample["embodiment_id"])
            action_hz = float(sample["action_hz"])

            processed_primary_videos = primary_videos
            processed_wrist_images = wrist_images
            if self.training and (self.enable_primary_random_resized_crop or self.enable_primary_video_aug):
                processed_primary_videos, processed_wrist_images = self._augment_primary_and_wrist_views(
                    primary_videos=primary_videos,
                    wrist_images=wrist_images,
                    enable_random_resized_crop=self.enable_primary_random_resized_crop,
                    enable_color_jitter=self.enable_primary_video_aug,
                )

            image_views_batch.append(self._to_primary_view_list(processed_primary_videos))
            wrist_image_views_batch.append(self._to_wrist_view_list(processed_wrist_images))
            primary_video_uint8_seqs.append(processed_primary_videos[0].contiguous())
            instructions.append(instruction)
            actions_list.append(action_tensor)
            if explicit_action_masks[0]:
                action_mask = sample["action_mask"].to(dtype=torch.bool).contiguous()
                if action_mask.shape != action_tensor.shape:
                    raise ValueError("explicit action_mask must match the raw action shape")
                explicit_action_mask_list.append(action_mask)
            states_list.append(state_tensor)
            if explicit_state_masks[0]:
                state_mask = sample["state_mask"]
                if state_mask.ndim == 1:
                    state_mask = state_mask.unsqueeze(0)
                if state_mask.shape != state_tensor.shape:
                    raise ValueError(
                        "explicit state_mask must match the raw state shape, "
                        f"got mask={tuple(state_mask.shape)} state={tuple(state_tensor.shape)}"
                    )
                if state_mask.dtype != torch.bool and not torch.all(
                    (state_mask == 0) | (state_mask == 1)
                ):
                    raise ValueError("explicit state_mask must be binary")
                explicit_state_mask_list.append(
                    state_mask.to(dtype=torch.bool).contiguous()
                )
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)
            if paired_enabled:
                assert condition_field is not None
                condition = sample[condition_field].to(dtype=torch.float32).reshape(-1)
                action_latent = sample["z_action"].to(dtype=torch.float32).reshape(-1)
                if condition.shape != (32,) or action_latent.shape != (32,):
                    raise ValueError("paired teacher tokens must both be 32D")
                paired_conditions.append(condition.contiguous())
                if raw_condition_presence[0]:
                    assert raw_condition_field is not None
                    raw_condition = sample[raw_condition_field].to(
                        dtype=torch.float32
                    ).reshape(-1)
                    if raw_condition.shape != (32,) or not torch.isfinite(
                        raw_condition
                    ).all():
                        raise ValueError("raw teacher condition must contain 32 finite values")
                    paired_raw_conditions.append(raw_condition.contiguous())
                paired_actions.append(action_latent.contiguous())
                if action_logvar_presence[0]:
                    action_logvar = sample["z_action_logvar"].to(dtype=torch.float32).reshape(-1)
                    if action_logvar.shape != (32,):
                        raise ValueError("paired teacher action logvar must be 32D")
                    paired_action_logvars.append(action_logvar.contiguous())
            if teacher_fields_enabled:
                action_predictor_state = sample["action_predictor_state"].to(
                    dtype=torch.float32
                )
                if action_predictor_state.shape != (32,) or not torch.isfinite(
                    action_predictor_state
                ).all():
                    raise ValueError(
                        "action_predictor_state must contain finite float values with exact shape [32]"
                    )
                trajectory = sample["teacher_trajectory"].to(dtype=torch.float32)
                current_state = sample["teacher_current_state"].to(dtype=torch.float32)
                expected_trajectory_shape = (
                    self.stage2_temporal_contract.representation_steps,
                    self.stage2_temporal_contract.trajectory_dim,
                )
                if trajectory.shape != expected_trajectory_shape:
                    raise ValueError(
                        f"teacher_trajectory must have exact shape {list(expected_trajectory_shape)}, "
                        f"got {tuple(trajectory.shape)}"
                    )
                expected_state_shape = (self.stage2_temporal_contract.state_dim,)
                if current_state.shape != expected_state_shape:
                    raise ValueError(
                        f"teacher_current_state must have exact shape {list(expected_state_shape)}, "
                        f"got {tuple(current_state.shape)}"
                    )
                if not torch.isfinite(trajectory).all() or not torch.isfinite(current_state).all():
                    raise ValueError("ActionVAE teacher trajectory/state must contain only finite values")

                masks: dict[str, torch.Tensor] = {}
                steps = self.stage2_temporal_contract.representation_steps
                arm_count = self.stage2_temporal_contract.arm_count
                arm_mask_shape = (steps,) if arm_count == 1 else (steps, arm_count)
                for name, expected_shape in (
                    ("teacher_time_mask", (steps,)),
                    ("teacher_pose_mask", arm_mask_shape),
                    ("teacher_gripper_mask", arm_mask_shape),
                    (
                        "teacher_current_state_mask",
                        (self.stage2_temporal_contract.state_dim,),
                    ),
                ):
                    raw_mask = sample[name]
                    if raw_mask.shape != expected_shape:
                        raise ValueError(
                            f"{name} must have exact shape {list(expected_shape)}, "
                            f"got {tuple(raw_mask.shape)}"
                        )
                    if raw_mask.dtype != torch.bool and not torch.all(
                        (raw_mask == 0) | (raw_mask == 1)
                    ):
                        raise ValueError(f"{name} must be a binary mask")
                    masks[name] = raw_mask.to(dtype=torch.bool).contiguous()
                if (
                    is_libero_04s_adapter_temporal_contract(
                        self.stage2_temporal_contract
                    )
                    and not bool(masks["teacher_time_mask"].all())
                ):
                    raise ValueError(
                        "0.4s adapter teacher_time_mask must keep all representation slots"
                    )
                if (
                    is_robotwin_stage2_temporal_contract(
                        self.stage2_temporal_contract
                    )
                    and not bool(masks["teacher_time_mask"].all())
                ):
                    raise ValueError(
                        "RoboTwin 1.2s teacher_time_mask must keep all 36 slots"
                    )
                if (
                    is_bridge_stage2_temporal_contract(self.stage2_temporal_contract)
                    and not bool(masks["teacher_time_mask"].all())
                ):
                    raise ValueError(
                        "Bridge 1.2s teacher_time_mask must keep all 24 resampled slots"
                    )

                teacher_trajectories.append(trajectory.contiguous())
                teacher_current_states.append(current_state.contiguous())
                teacher_time_masks.append(masks["teacher_time_mask"])
                teacher_pose_masks.append(masks["teacher_pose_mask"])
                teacher_gripper_masks.append(masks["teacher_gripper_mask"])
                teacher_current_state_masks.append(masks["teacher_current_state_mask"])
                action_predictor_states.append(action_predictor_state.contiguous())
                if physical_count_enabled:
                    raw_count = sample["physical_valid_count"]
                    if torch.is_tensor(raw_count):
                        if raw_count.numel() != 1 or raw_count.dtype == torch.bool:
                            raise ValueError("physical_valid_count must be one integer scalar")
                        raw_count = raw_count.item()
                    if isinstance(raw_count, bool):
                        raise ValueError("physical_valid_count must be one integer scalar")
                    count = int(raw_count)
                    if float(raw_count) != float(count):
                        raise ValueError("physical_valid_count must be one integer scalar")
                    if count != self.stage2_temporal_contract.physical_valid_count:
                        raise ValueError(
                            "physical_valid_count does not match stage2_temporal_contract: "
                            f"got={count}, expected={self.stage2_temporal_contract.physical_valid_count}"
                        )
                    physical_valid_counts.append(count)

        processor, placeholder_token_id = self._ensure_processor()
        qwen_messages = build_qwenvl_messages(
            images=image_views_batch,
            wrist_images=wrist_image_views_batch,
            instructions=instructions,
            placeholder_token=self.placeholder_token,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            cot_prompt_before_wrist=self.cot_prompt_before_wrist,
            cot_prompt_after_wrist=self.cot_prompt_after_wrist,
        )
        qwen_inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            placeholder_id=placeholder_token_id,
        )

        action_dim = int(self.policy_cfg.flow_cfg.action_dim)
        window_size = int(self.policy_cfg.action_horizon)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []
        for idx, state_tensor in enumerate(states_list):
            action_tensor = actions_list[idx]
            action_tensor, action_dim_mask = align_feature_dim_with_mask(action_tensor, action_dim)
            action_tensor, action_time_mask = sample_or_pad_sequence_with_mask(action_tensor, window_size)
            action_tensors.append(action_tensor)
            if explicit_action_masks[0]:
                explicit_mask, _ = align_feature_dim_with_mask(
                    explicit_action_mask_list[idx], action_dim
                )
                explicit_mask, _ = sample_or_pad_sequence_with_mask(
                    explicit_mask, window_size
                )
                action_masks.append(
                    explicit_mask.to(dtype=torch.bool) & action_time_mask.unsqueeze(1)
                )
            else:
                action_masks.append(
                    action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0)
                )

            aligned_state, state_dim_mask = align_feature_dim_with_mask(state_tensor, state_dim)
            if explicit_state_masks[0]:
                explicit_mask, _ = align_feature_dim_with_mask(
                    explicit_state_mask_list[idx].to(dtype=torch.float32), state_dim
                )
                state_dim_mask = state_dim_mask & explicit_mask[-1].to(dtype=torch.bool)
            if bool(getattr(self.policy_cfg.flow_cfg, "use_state", False)) and not bool(
                state_dim_mask.any()
            ):
                raise ValueError(
                    "flow_cfg.use_state=true requires at least one valid flow-state dimension; "
                    "the official LIBERO Stage-2 backend has no compatible StarVLA state statistics"
                )
            state_tensors.append(aligned_state[-1].contiguous())
            state_masks.append(state_dim_mask.contiguous())

        primary_video_uint8_batch = torch.stack(primary_video_uint8_seqs, dim=0)
        primary_video_batch = primary_video_uint8_batch.to(dtype=torch.float32).div_(255.0)
        imagenet_normalize_video_(primary_video_batch)

        batch: LatentWorldPolicyTrainBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_video": primary_video_batch,
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "actions": torch.stack(action_tensors, dim=0),
            "actions_mask": torch.stack(action_masks, dim=0),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        if paired_enabled:
            assert condition_field is not None
            batch[condition_field] = torch.stack(paired_conditions, dim=0)
            if paired_raw_conditions:
                assert raw_condition_field is not None
                batch[raw_condition_field] = torch.stack(
                    paired_raw_conditions, dim=0
                )
            batch["z_action"] = torch.stack(paired_actions, dim=0)
            if paired_action_logvars:
                batch["z_action_logvar"] = torch.stack(paired_action_logvars, dim=0)
        if teacher_fields_enabled:
            batch["action_predictor_state"] = torch.stack(action_predictor_states, dim=0)
            batch["teacher_trajectory"] = torch.stack(teacher_trajectories, dim=0)
            batch["teacher_current_state"] = torch.stack(teacher_current_states, dim=0)
            batch["teacher_time_mask"] = torch.stack(teacher_time_masks, dim=0)
            batch["teacher_pose_mask"] = torch.stack(teacher_pose_masks, dim=0)
            batch["teacher_gripper_mask"] = torch.stack(teacher_gripper_masks, dim=0)
            batch["teacher_current_state_mask"] = torch.stack(
                teacher_current_state_masks, dim=0
            )
            if physical_valid_counts:
                batch["physical_valid_count"] = torch.tensor(
                    physical_valid_counts, dtype=torch.long
                )
        return batch
