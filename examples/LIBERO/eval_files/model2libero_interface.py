from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import math
import os
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from latent_action_model.libero_stage2.action_transform import (
    FLOW_ACTION_DIM,
    LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256,
    LiberoActionNormalizer,
)
from latent_action_model.libero_stage2.teacher_bundle import (
    LIBERO_12S_ACTION_NORMALIZATION_SHA256,
)
from latent_action_model.libero_stage2.state_transform import ACTIONVAE_STATE_SOURCE
from starVLA.model.tools import read_mode_config


LIBERO_OFFICIAL_STAGE2_BACKEND = "libero_official_stage2_v1"


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        image_size: list[int] = [256, 256],
        action_hz: float = 20.0,
        embodiment_id: int = 25,
        host: str = "127.0.0.1",
        port: int = 10095,
        eval_action_chunk_len: Optional[int] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if policy_ckpt_path is None or str(policy_ckpt_path).strip() == "":
            raise ValueError("`policy_ckpt_path` must be a non-empty checkpoint path.")
        self.policy_ckpt_path = Path(policy_ckpt_path).expanduser().resolve()

        self.model_config, self.norm_stats = read_mode_config(self.policy_ckpt_path)
        framework_name = self.model_config.get("framework", {}).get("name", "")
        self.framework_name = framework_name
        self.framework_cfg = self.model_config.get("framework", {})
        action_cfg = self.framework_cfg.get("action_model", {})
        data_cfg = self.model_config.get("datasets", {}).get("vla_data", {})
        self.flow_uses_state = bool(action_cfg.get("flow_cfg", {}).get("use_state", True))
        self.stage2_action_state_source = str(
            action_cfg.get("stage2_action_state_source", "") or ""
        )
        self.requires_action_predictor_raw_state = (
            self.stage2_action_state_source == ACTIONVAE_STATE_SOURCE
        )
        self.policy_setup = policy_setup
        self.unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        self.image_size = image_size  # Retained for backward-compatible CLI/API surface; resizing happens on server.
        self.action_hz = float(action_hz)
        if self.action_hz <= 0.0:
            raise ValueError(f"`action_hz` must be > 0, got {self.action_hz}.")
        self.embodiment_id = int(embodiment_id)
        flow_cfg = action_cfg.get("flow_cfg", {})
        self.horizon_sec = float(flow_cfg.get("horizon_sec", 1.0))
        if self._is_latent_world_framework(framework_name) and self.horizon_sec <= 0.0:
            raise ValueError(f"`flow_cfg.horizon_sec` must be > 0, got {self.horizon_sec}.")
        self.fixed_action_horizon = self._resolve_fixed_action_horizon(self.framework_cfg)
        self.eval_action_chunk_len = (
            None if eval_action_chunk_len is None else int(eval_action_chunk_len)
        )
        if self.eval_action_chunk_len is not None and self.eval_action_chunk_len <= 0:
            raise ValueError(
                f"`eval_action_chunk_len` must be > 0 when set, got {self.eval_action_chunk_len}."
            )

        self.dataset_stats = self.norm_stats[self.unnorm_key]
        self.is_official_stage2 = (
            self.dataset_stats.get("stage2_backend") == LIBERO_OFFICIAL_STAGE2_BACKEND
        )
        self.official_action_normalizer: LiberoActionNormalizer | None = None
        if self.is_official_stage2:
            self._configure_official_stage2(
                action_cfg=action_cfg,
                data_cfg=data_cfg,
            )
            self.action_norm_stats = None
            self.state_norm_stats = None
        else:
            self.action_norm_stats = self.dataset_stats["action"]
            self.state_norm_stats = self.dataset_stats["state"]

        self.client = WebsocketClientPolicy(host, port)
        self._validate_server_metadata_or_raise(framework_name=framework_name)

        self._slot_states: dict[int, _SlotState] = {}

        print(
            "*** "
            f"policy_setup: {policy_setup}, "
            f"unnorm_key: {self.unnorm_key}, "
            f"action_hz: {self.action_hz}, "
            f"horizon_sec: {self.horizon_sec}, "
            f"action_horizon: {self.fixed_action_horizon}, "
            f"eval_action_chunk_len: {self.eval_action_chunk_len} "
            "***"
        )

    @staticmethod
    def _normalize_framework_name(name: str) -> str:
        return str(name).replace("_", "").lower()

    @classmethod
    def _is_latent_world_framework(cls, framework_name: str) -> bool:
        return cls._normalize_framework_name(framework_name) == "lawam"

    @staticmethod
    def _resolve_fixed_action_horizon(framework_cfg: dict) -> Optional[int]:
        action_cfg = framework_cfg.get("action_model", {})
        if "action_horizon" in action_cfg and action_cfg["action_horizon"] is not None:
            return int(action_cfg["action_horizon"])
        return None

    def _validate_server_metadata_or_raise(self, *, framework_name: str) -> None:
        metadata = self.client.get_server_metadata() or {}
        server_ckpt_raw = metadata.get("ckpt_path", None)
        if server_ckpt_raw is None:
            raise ValueError(
                "Server metadata does not contain `ckpt_path`; "
                "refuse to run because checkpoint consistency cannot be verified."
            )

        server_ckpt = Path(str(server_ckpt_raw)).expanduser().resolve()
        if server_ckpt != self.policy_ckpt_path:
            raise ValueError(
                "Checkpoint mismatch between eval client and server: "
                f"client_ckpt={self.policy_ckpt_path}, server_ckpt={server_ckpt}."
            )

        local_fw = self._normalize_framework_name(framework_name)
        server_fw_raw = metadata.get("framework_name", "")
        if server_fw_raw and local_fw:
            server_fw = self._normalize_framework_name(server_fw_raw)
            if server_fw != local_fw:
                raise ValueError(
                    "Server framework does not match local framework. "
                    f"server_framework={server_fw_raw!r}."
                )
        action_cfg = getattr(self, "framework_cfg", {}).get("action_model", {})
        if action_cfg.get("stage2_action_injection_enabled") is False:
            if metadata.get("stage2_action_injection_enabled") is not False:
                raise ValueError("Aux-only policy server has not confirmed disabled action injection")
            expected_future = os.environ.get("LAWAM_ACTION_PREDICTOR_FUTURE_INPUT", "predicted")
            if metadata.get("action_predictor_future_input") != expected_future:
                raise ValueError("Aux-only policy server future-input mode mismatch")
            print(f"[AUX_ONLY_SERVER_VERIFIED] action_injection=disabled future_input={expected_future}")

    def close(self) -> None:
        self.client.close()

    def reset(
        self,
        task_description: str,
        slot_key: int = 0,
        **kwargs,
    ) -> None:
        slot_key = self._resolve_slot_key(slot_key=slot_key, kwargs=kwargs)
        self._get_slot_state(slot_key).reset(task_description=task_description)

    def step(
        self,
        example: dict,
        step: int = 0,
        return_intermediates: bool = False,
        action_seed: Optional[int] = None,
        **kwargs,
    ) -> dict[str, object]:
        del step
        slot_key = self._resolve_slot_key(slot_key=0, kwargs=kwargs)
        return self.step_batch(
            [example],
            cache_keys=[slot_key],
            return_intermediates=return_intermediates,
            action_seeds=None if action_seed is None else [int(action_seed)],
        )[0]

    def step_batch(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        cache_keys: Optional[Sequence[int]] = None,
        return_intermediates: bool = False,
        action_seeds: Optional[Sequence[int]] = None,
    ) -> list[dict[str, object]]:
        if cache_keys is None:
            cache_keys = list(range(len(examples)))
        if len(examples) != len(cache_keys):
            raise ValueError(
                f"`examples` and `cache_keys` must have the same length, got {len(examples)} vs {len(cache_keys)}."
            )
        if action_seeds is not None and len(action_seeds) != len(examples):
            raise ValueError(
                "`action_seeds` must have the same length as `examples`, got "
                f"{len(action_seeds)} vs {len(examples)}."
            )

        query_positions: list[int] = []
        query_examples: list[dict[str, Any]] = []
        outputs: list[dict[str, object] | None] = [None] * len(examples)

        for idx, (example, cache_key) in enumerate(zip(examples, cache_keys)):
            cache_key = int(cache_key)
            slot_state = self._get_slot_state(cache_key)
            prepared = self._prepare_example(example)
            task_description = str(prepared["lang"])
            if task_description != slot_state.task_description:
                slot_state.reset(task_description=task_description)
            if slot_state.needs_query():
                query_positions.append(idx)
                query_examples.append(prepared)

        split_intermediates: list[Any | None] = [None] * len(query_examples)
        query_result_index_by_example: dict[int, int] = {}
        if query_examples:
            query: dict[str, Any] = {
                "examples": query_examples,
                "return_intermediates": bool(return_intermediates),
            }
            if action_seeds is not None:
                query["action_seeds"] = [int(action_seeds[index]) for index in query_positions]
            response = self.client.predict_action(query)
            if not bool(response.get("ok", False)):
                error = response.get("error", {})
                message = error.get("message", "Unknown inference server error.")
                raise RuntimeError(
                    "Inference server returned an error response: "
                    f"{message}. Full response: {response}"
                )

            data = response.get("data", {})
            try:
                normalized_actions = np.asarray(data["normalized_actions"], dtype=np.float32)
            except KeyError as exc:
                raise KeyError(f"Key 'normalized_actions' not found in response: {response}") from exc

            if normalized_actions.ndim == 2:
                normalized_actions = normalized_actions[None, :, :]
            if normalized_actions.ndim != 3:
                raise ValueError(
                    f"`normalized_actions` must have shape [B, T, D] or [T, D], got {normalized_actions.shape}."
                )
            if int(normalized_actions.shape[0]) != len(query_examples):
                raise ValueError(
                    f"Expected {len(query_examples)} batched predictions, got shape {normalized_actions.shape}."
                )

            split_intermediates = self._split_intermediates(
                data.get("intermediates", None),
                batch_size=len(query_examples),
                return_intermediates=bool(return_intermediates),
            )

            for batch_idx, example_idx in enumerate(query_positions):
                query_result_index_by_example[int(example_idx)] = int(batch_idx)
                cache_key = int(cache_keys[example_idx])
                slot_state = self._get_slot_state(cache_key)
                inference_example = query_examples[batch_idx]
                chunk = normalized_actions[batch_idx]
                expected_len = self._expected_action_chunk_length(
                    action_hz=float(inference_example["action_hz"]),
                )
                if expected_len is not None and int(chunk.shape[0]) != expected_len:
                    raise ValueError(
                        "Inference action chunk length mismatch: "
                        f"returned={int(chunk.shape[0])}, expected={expected_len}, "
                        f"horizon_sec={self.horizon_sec}, action_hz={float(inference_example['action_hz'])}, "
                        f"action_horizon={self.fixed_action_horizon}."
                    )
                if self.eval_action_chunk_len is not None:
                    chunk = chunk[: int(self.eval_action_chunk_len)]
                if self.is_official_stage2:
                    slot_state.raw_actions = self._unnormalize_official_stage2_actions(chunk)
                else:
                    assert self.action_norm_stats is not None
                    slot_state.raw_actions = self.unnormalize_actions(
                        normalized_actions=chunk,
                        action_norm_stats=self.action_norm_stats,
                    )
                if int(slot_state.raw_actions.shape[0]) <= 0:
                    raise RuntimeError("Inference returned empty action chunk.")
                slot_state.action_cursor = 0

        for idx, cache_key in enumerate(cache_keys):
            cache_key = int(cache_key)
            slot_state = self._get_slot_state(cache_key)
            if slot_state.raw_actions is None or slot_state.action_cursor >= int(slot_state.raw_actions.shape[0]):
                raise RuntimeError(f"Slot {cache_key} does not have any cached actions after inference.")
            raw_actions = np.asarray(slot_state.raw_actions[slot_state.action_cursor], dtype=np.float32)[None, :]
            slot_state.action_cursor += 1
            raw_action = {
                "world_vector": np.array(raw_actions[0, :3]),
                "rotation_delta": np.array(raw_actions[0, 3:6]),
                "open_gripper": np.array(raw_actions[0, -1:]),
            }
            output: dict[str, object] = {"raw_action": raw_action}
            query_idx = query_result_index_by_example.get(int(idx), -1)
            if bool(return_intermediates) and query_idx >= 0:
                output["intermediates"] = split_intermediates[query_idx]
            outputs[idx] = output

        return [output for output in outputs if output is not None]

    def _get_slot_state(self, slot_key: int) -> "_SlotState":
        slot_key = int(slot_key)
        if slot_key not in self._slot_states:
            self._slot_states[slot_key] = _SlotState()
        return self._slot_states[slot_key]

    def _resolve_slot_key(self, slot_key: int = 0, kwargs: Optional[dict[str, Any]] = None) -> int:
        extra = dict(kwargs or {})
        if "slot_id" in extra:
            slot_key = int(extra.pop("slot_id"))
        if "cache_key" in extra:
            slot_key = int(extra.pop("cache_key"))
        if "slot_key" in extra:
            slot_key = int(extra.pop("slot_key"))
        if extra:
            raise TypeError(f"Unexpected keyword arguments: {sorted(extra.keys())}")
        return int(slot_key)

    def _prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        task_description = example.get("lang", None)
        if task_description is None:
            raise KeyError("LIBERO eval example must contain key `lang`.")
        primary_images = example.get("primary_image", None)
        if not isinstance(primary_images, (list, tuple)) or len(primary_images) == 0:
            raise ValueError("LIBERO online infer requires `example['primary_image']` as a non-empty list.")
        if not all(isinstance(img, np.ndarray) for img in primary_images):
            bad = next(i for i, img in enumerate(primary_images) if not isinstance(img, np.ndarray))
            raise TypeError(
                f"LIBERO online infer requires all `example['primary_image']` entries to be `np.ndarray`, "
                f"got {type(primary_images[bad]).__name__} at index {bad}."
            )

        inference_example = {
            "primary_image": [np.asarray(img) for img in primary_images],
            "lang": str(task_description),
            "embodiment_id": int(example.get("embodiment_id", self.embodiment_id)),
            "action_hz": float(example.get("action_hz", self.action_hz)),
        }
        if inference_example["action_hz"] <= 0.0:
            raise ValueError(f"`action_hz` must be > 0, got {inference_example['action_hz']}.")

        if self.requires_action_predictor_raw_state:
            raw_state = np.asarray(
                example.get("action_predictor_raw_state", None), dtype=np.float32
            ).reshape(-1)
            raw_mask = np.asarray(
                example.get("action_predictor_raw_state_mask", None)
            ).reshape(-1)
            if raw_state.shape != (14,) or raw_mask.shape != (14,):
                raise ValueError(
                    "LIBERO Stage-2 action predictor requires canonical raw state/mask "
                    "with shape [14]"
                )
            if not np.isin(raw_mask, (0, 1)).all():
                raise ValueError("action_predictor_raw_state_mask must be binary")
            expected_mask = np.zeros(14, dtype=np.bool_)
            expected_mask[:7] = True
            if not np.array_equal(raw_mask.astype(np.bool_), expected_mask):
                raise ValueError(
                    "canonical LIBERO ActionVAE state mask must be true7/false7"
                )
            if not np.isfinite(raw_state[:7]).all() or not np.array_equal(
                raw_state[7:], np.zeros(7, dtype=np.float32)
            ):
                raise ValueError(
                    "canonical LIBERO ActionVAE raw state must be finite in the first 7 "
                    "dimensions and zero in reserved dimensions"
                )
            if float(raw_state[6]) < 0.0:
                raise ValueError("canonical LIBERO physical gripper width must be non-negative")
            inference_example["action_predictor_raw_state"] = raw_state
            inference_example["action_predictor_raw_state_mask"] = raw_mask.astype(np.bool_)

        if not self.is_official_stage2:
            state = example.get("state", None)
            if state is None:
                raise KeyError("LIBERO eval example must contain key `state`.")
            state_arr = np.asarray(state, dtype=np.float32)
            if state_arr.ndim == 1:
                state_arr = state_arr[None, :]
            if state_arr.ndim != 2:
                raise ValueError(f"`state` must have shape [D] or [T,D], got {state_arr.shape}.")
            assert self.state_norm_stats is not None
            inference_example["state"] = self.normalize_state(
                state=state_arr,
                state_norm_stats=self.state_norm_stats,
            )

        wrist_raw = example.get("wrist_image", None)
        if wrist_raw is not None:
            if not isinstance(wrist_raw, (list, tuple)):
                raise ValueError("`wrist_image` must be a list/tuple when provided.")
            if not all(isinstance(image, np.ndarray) for image in wrist_raw):
                raise TypeError("`wrist_image` entries must all be `np.ndarray`.")
            inference_example["wrist_image"] = [np.asarray(image) for image in wrist_raw]

        return inference_example

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _configure_official_stage2(self, *, action_cfg: dict, data_cfg: dict) -> None:
        if data_cfg.get("dataset_backend") != LIBERO_OFFICIAL_STAGE2_BACKEND:
            raise ValueError(
                "official Stage-2 dataset statistics require "
                f"datasets.vla_data.dataset_backend={LIBERO_OFFICIAL_STAGE2_BACKEND}"
            )
        if self.flow_uses_state:
            raise ValueError(
                "official LIBERO Stage-2 has no compatible StarVLA flow-state statistics; "
                "flow_cfg.use_state must be false"
            )
        if self.stage2_action_state_source != ACTIONVAE_STATE_SOURCE:
            raise ValueError(
                "official LIBERO Stage-2 online inference requires "
                "stage2_action_state_source=actionvae14_padded32"
            )
        if "action" in self.dataset_stats or "state" in self.dataset_stats:
            raise ValueError(
                "official Stage-2 dataset statistics must not masquerade the SHA-bound "
                "flow action or robust ActionVAE state transforms as generic action/state stats"
            )

        expected_action_sha = str(
            self.dataset_stats.get("flow_action_normalization_sha256", "")
        )
        if expected_action_sha != LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256:
            raise ValueError("official Stage-2 flow-action normalization SHA256 mismatch")
        action_path_raw = data_cfg.get("stage2_action_normalization")
        if action_path_raw is None:
            raise ValueError(
                "official Stage-2 config is missing datasets.vla_data.stage2_action_normalization"
            )
        action_path = Path(str(action_path_raw)).expanduser().resolve()
        if self._sha256_file(action_path) != LIBERO_STAGE2_FLOW_ACTION_NORMALIZATION_SHA256:
            raise ValueError("configured official Stage-2 flow-action artifact has the wrong SHA256")
        action_metadata = json.loads(action_path.read_text(encoding="utf-8"))
        if "stage2_train_only_normalizer" in action_metadata:
            action_metadata = action_metadata["stage2_train_only_normalizer"]
        self.official_action_normalizer = LiberoActionNormalizer.from_dict(action_metadata)

        expected_state_sha = str(
            self.dataset_stats.get("action_vae_normalization_sha256", "")
        )
        if expected_state_sha != LIBERO_12S_ACTION_NORMALIZATION_SHA256:
            raise ValueError("official Stage-2 ActionVAE state normalization SHA256 mismatch")
        action_data_root = action_cfg.get("stage2_action_decoder_data_root")
        if action_data_root is None:
            raise ValueError("official Stage-2 config is missing stage2_action_decoder_data_root")
        state_path = Path(str(action_data_root)).expanduser().resolve() / "stats/normalization.json"
        if self._sha256_file(state_path) != LIBERO_12S_ACTION_NORMALIZATION_SHA256:
            raise ValueError("configured ActionVAE state normalization artifact has the wrong SHA256")

    def _unnormalize_official_stage2_actions(
        self, normalized_actions: np.ndarray
    ) -> np.ndarray:
        if self.official_action_normalizer is None:
            raise RuntimeError("official Stage-2 action normalizer was not initialized")
        values = np.asarray(normalized_actions, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[-1] != FLOW_ACTION_DIM:
            raise ValueError(
                "official Stage-2 normalized actions must have shape [T,32], "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("official Stage-2 normalized actions contain NaN/Inf")
        clipped = values.copy()
        clipped[:, :6] = np.clip(clipped[:, :6], -1.0, 1.0)
        mask = np.zeros_like(clipped, dtype=np.bool_)
        mask[:, :7] = True
        native, native_mask = self.official_action_normalizer.inverse(clipped, mask)
        if not native_mask.all() or native.shape != (values.shape[0], 7):
            raise RuntimeError("official Stage-2 action inverse returned an invalid native mask")
        return native

    def _expected_action_chunk_length(self, action_hz: float) -> Optional[int]:
        expected_len = None
        if self._is_latent_world_framework(self.framework_name):
            expected_len = int(math.floor(self.horizon_sec * float(action_hz)))
            if expected_len < 1:
                raise ValueError(
                    "Invalid expected action length from inference settings: "
                    f"floor(horizon_sec * action_hz)={expected_len}, "
                    f"horizon_sec={self.horizon_sec}, action_hz={float(action_hz)}."
                )
        elif self.fixed_action_horizon is not None:
            expected_len = int(self.fixed_action_horizon)
        return expected_len

    @staticmethod
    def _split_intermediates(
        intermediates: Any,
        *,
        batch_size: int,
        return_intermediates: bool,
    ) -> list[Any | None]:
        if not return_intermediates:
            return [None] * batch_size
        if intermediates is None:
            return [None] * batch_size
        if batch_size == 1 and not isinstance(intermediates, list):
            return [intermediates]
        if isinstance(intermediates, list):
            if len(intermediates) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} intermediate entries, got {len(intermediates)}."
                )
            return list(intermediates)
        if isinstance(intermediates, dict):
            split: list[dict[str, Any]] = []
            for batch_idx in range(batch_size):
                item: dict[str, Any] = {}
                for key, value in intermediates.items():
                    value_arr = np.asarray(value)
                    if value_arr.ndim == 0 or int(value_arr.shape[0]) != batch_size:
                        raise ValueError(
                            f"Cannot split intermediate `{key}` with shape {value_arr.shape} for batch size {batch_size}."
                        )
                    item[key] = value_arr[batch_idx]
                split.append(item)
            return split
        raise TypeError(f"Unsupported intermediates container type: {type(intermediates).__name__}")

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Denormalize actions using per-dimension statistics.

        `action_norm_stats` is computed on the LIBERO control space. If the
        policy returns extra trailing dimensions, the LIBERO client treats the
        first `len(action_stats)` dimensions as the environment action and
        discards the rest.
        """
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"

        action_high = np.asarray(action_norm_stats[high_key], dtype=np.float32)
        action_low = np.asarray(action_norm_stats[low_key], dtype=np.float32)
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))

        # Ensure array and at least 2D [T, D].
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]

        stats_dim = int(action_high.shape[0])
        if normalized_actions.shape[-1] != stats_dim:
            if normalized_actions.shape[-1] < stats_dim:
                raise ValueError(
                    "Policy action dimension is smaller than action statistics dimension: "
                    f"action_dim={normalized_actions.shape[-1]}, stats_dim={stats_dim}."
                )
            normalized_actions = normalized_actions[..., :stats_dim]

        if not np.all(np.isfinite(normalized_actions)):
            raise ValueError(
                "Policy returned non-finite normalized actions: "
                f"min={np.nanmin(normalized_actions)}, max={np.nanmax(normalized_actions)}"
            )

        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)

        # Binarize gripper on the final action dimension.
        if stats_dim > 0:
            normalized_actions[:, -1] = np.where(normalized_actions[:, -1] < 0.5, 0.0, 1.0)

        denorm = 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low
        raw_actions = np.where(mask, denorm, normalized_actions)
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError(
                "Action denormalization produced non-finite actions: "
                f"min={np.nanmin(raw_actions)}, max={np.nanmax(raw_actions)}"
            )
        return raw_actions

    @staticmethod
    def normalize_state(
        state: np.ndarray,
        state_norm_stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2:
            raise ValueError(f"`state` must have shape [D] or [T,D], got {state.shape}.")

        high_key = "max" if "max" in state_norm_stats else "q99"
        low_key = "min" if "min" in state_norm_stats else "q01"
        state_high = np.asarray(state_norm_stats[high_key], dtype=np.float32)
        state_low = np.asarray(state_norm_stats[low_key], dtype=np.float32)

        stats_dim = int(state_high.shape[0])
        if state.shape[-1] != stats_dim:
            if state.shape[-1] < stats_dim:
                raise ValueError(
                    "Policy state dimension is smaller than state statistics dimension: "
                    f"state_dim={state.shape[-1]}, stats_dim={stats_dim}."
                )
            state = state[..., :stats_dim]

        normalized = state.copy()
        denom = state_high - state_low
        valid = np.abs(denom) > 1e-12
        if np.any(valid):
            normalized[..., valid] = (
                (normalized[..., valid] - state_low[valid]) / denom[valid] * 2.0 - 1.0
            )
        if np.any(~valid):
            normalized[..., ~valid] = 0.0

        return np.clip(normalized, -1.0, 1.0)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint contains multiple dataset statistics; please pass `unnorm_key`. "
                    f"Available: {list(norm_stats.keys())}"
                )
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"Invalid `unnorm_key`={unnorm_key}. Available: {list(norm_stats.keys())}"
            )
        return unnorm_key


@dataclass
class _SlotState:
    task_description: Optional[str] = None
    raw_actions: Optional[np.ndarray] = None
    action_cursor: int = 0

    def reset(self, task_description: Optional[str] = None) -> None:
        self.task_description = None if task_description is None else str(task_description)
        self.raw_actions = None
        self.action_cursor = 0

    def needs_query(self) -> bool:
        if self.raw_actions is None:
            return True
        return int(self.action_cursor) >= int(self.raw_actions.shape[0])
