# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, compute_opd_loss, kl_penalty
from verl.trainer.ppo.env_aux_loss_utils import (
    create_inverse_dynamics_messages,
    create_search_inverse_dynamics_messages,
    create_state_prediction_messages,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "compose_actor_objective"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

JUNE24_OUTCOME_CLASSES = ("fixed", "both_wrong", "harmed", "both_correct")


def compose_actor_objective(
    *,
    policy_objective: torch.Tensor,
    opd_loss: torch.Tensor,
    opd_loss_coef: float,
    opd_only: bool,
) -> torch.Tensor:
    """Combine objectives while making the OPD-only gradient boundary explicit."""

    if opd_only:
        return opd_loss * float(opd_loss_coef)
    return policy_objective + opd_loss * float(opd_loss_coef)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        self.tokenizer = None
        self.global_step = 0

        self.sp_coef_initial = self._config_float("sp_coef", 0.0)
        self.id_coef_initial = self._config_float("id_coef", 0.0)
        self.sp_coef_min = self._config_float("sp_coef_min", 0.0)
        self.id_coef_min = self._config_float("id_coef_min", 0.0)
        self.aux_coef_decay = self.config.get("aux_coef_decay", self.config.get("coef_decay", "cos"))
        self.aux_history_length = self.config.get("aux_history_length", None)
        self.aux_max_length = self.config.get("aux_max_length", None)
        self.aux_data_ratio = float(self.config.get("aux_data_ratio", 1.0) or 1.0)
        self.sp_coef = self.sp_coef_initial
        self.id_coef = self.id_coef_initial

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _config_float(self, key: str, default: float = 0.0) -> float:
        value = self.config.get(key, default)
        if value is None:
            return default
        return float(value)

    def _trainable_parameter_sha256(self) -> str:
        digest = hashlib.sha256()
        count = 0
        for name, parameter in sorted(self.actor_module.named_parameters(), key=lambda item: item[0]):
            if not parameter.requires_grad:
                continue
            tensor = parameter.detach()
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            tensor = tensor.float().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            count += tensor.numel()
        if count == 0:
            raise RuntimeError("actor.opd_only found no trainable parameters")
        return digest.hexdigest()

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().contiguous().cpu()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _get_batch_item(batch_value: Any, index: int, default: Any = None) -> Any:
        if batch_value is None:
            return default
        try:
            return batch_value[index]
        except Exception:
            return default

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_action(self, response_text: str) -> Tuple[str, bool]:
        search_action = self._extract_tag(response_text, "search")
        if search_action:
            return search_action, True

        action = self._extract_tag(response_text, "action")
        if action:
            return action, False

        return response_text.strip(), False

    def _normalize_history_pairs(self, history: Any) -> List[Tuple[str, str]]:
        if history is None:
            return []
        if hasattr(history, "tolist"):
            try:
                history = history.tolist()
            except Exception:
                pass
        if isinstance(history, dict):
            history_items = [history]
        elif isinstance(history, (list, tuple)):
            history_items = list(history)
        else:
            return []

        pairs = []
        for item in history_items:
            obs = ""
            action = ""
            if isinstance(item, dict):
                obs = (
                    item.get("text_obs")
                    or item.get("observation")
                    or item.get("obs")
                    or item.get("information")
                    or item.get("anchor_obs")
                    or ""
                )
                action = item.get("action") or item.get("search") or item.get("response") or ""
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                obs, action = item[0], item[1]
            obs_text = self._to_text(obs).strip()
            action_text = self._to_text(action).strip()
            if obs_text or action_text:
                pairs.append((obs_text, action_text))
        return pairs

    def _normalize_admissibles(self, admissibles: Any) -> List[str]:
        if admissibles is None:
            return []
        if hasattr(admissibles, "tolist"):
            try:
                admissibles = admissibles.tolist()
            except Exception:
                pass
        if isinstance(admissibles, dict):
            actions = []
            if admissibles.get("has_search_bar"):
                actions.append("search[<your query>]")
            for clickable in admissibles.get("clickables", []) or []:
                actions.append(f"click[{clickable}]")
            return actions
        if isinstance(admissibles, str):
            return [line.strip(" '-,") for line in admissibles.splitlines() if line.strip()]
        if isinstance(admissibles, (list, tuple, set)):
            return [self._to_text(action).strip() for action in admissibles if self._to_text(action).strip()]
        return [self._to_text(admissibles).strip()]

    def _compute_decayed_coef(self, initial_value: float, min_value: float, current_step: int, total_steps: int) -> float:
        if self.aux_coef_decay == "none":
            return initial_value
        if initial_value < min_value or total_steps <= 0:
            return initial_value

        progress = min(1.0, max(0.0, current_step / total_steps))
        if self.aux_coef_decay == "cos":
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_value + (initial_value - min_value) * cosine_decay
        if self.aux_coef_decay == "linear":
            return min_value + (initial_value - min_value) * (1 - progress)
        raise ValueError(f"Unsupported aux_coef_decay '{self.aux_coef_decay}'. Expected one of: cos, linear, none.")

    def _update_aux_coefs(self):
        total_training_steps = int(self.config.get("optim", {}).get("total_training_steps", 0) or 0)
        self.sp_coef = self._compute_decayed_coef(
            self.sp_coef_initial,
            self.sp_coef_min,
            self.global_step,
            total_training_steps,
        )
        self.id_coef = self._compute_decayed_coef(
            self.id_coef_initial,
            self.id_coef_min,
            self.global_step,
            total_training_steps,
        )

    def _env_aux_loss_enabled(self) -> bool:
        self._update_aux_coefs()
        return self.sp_coef > 0 or self.id_coef > 0

    def _sample_aux_records(self, records: List[Dict[str, Any]], task_type: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        original_count = len(records)
        if original_count == 0 or self.aux_data_ratio >= 1.0:
            return records

        num_samples = max(1, int(original_count * self.aux_data_ratio))
        if num_samples >= original_count:
            return records

        sampled_indices = sorted(random.sample(range(original_count), num_samples))
        metrics[f"actor/{task_type}_samples_used"] = float(num_samples)
        metrics[f"actor/{task_type}_samples_total"] = float(original_count)
        return [records[idx] for idx in sampled_indices]

    def _build_env_auxiliary_records(self, data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
        stats = {
            "actor/env_aux_candidate_steps": 0.0,
            "actor/env_aux_built_sp_records": 0.0,
            "actor/env_aux_built_id_records": 0.0,
            "actor/env_aux_skipped_tokenizer_missing": 0.0,
            "actor/env_aux_skipped_metadata_missing": 0.0,
            "actor/env_aux_skipped_next_obs_empty": 0.0,
        }
        if self.tokenizer is None:
            stats["actor/env_aux_skipped_tokenizer_missing"] = 1.0
            return [], [], stats

        history_batch = data.get("history")
        next_obs_batch = data.get("next_obs")
        current_obs_batch = data.get("anchor_obs", data.get("obs_text"))
        if history_batch is None or next_obs_batch is None or current_obs_batch is None:
            stats["actor/env_aux_skipped_metadata_missing"] = 1.0
            return [], [], stats

        responses = data["responses"]
        action_texts = self.tokenizer.batch_decode(responses.detach().cpu().tolist(), skip_special_tokens=True)
        admissibles_batch = data.get("admissibles")
        active_masks = data.get("active_masks")
        batch_size = len(action_texts)
        stats["actor/env_aux_candidate_steps"] = float(batch_size)

        sp_records = []
        id_records = []
        for sample_idx in range(batch_size):
            if active_masks is not None and not bool(self._get_batch_item(active_masks, sample_idx, True)):
                continue

            current_obs = self._to_text(self._get_batch_item(current_obs_batch, sample_idx)).strip()
            next_obs = self._to_text(self._get_batch_item(next_obs_batch, sample_idx)).strip()
            if not next_obs:
                stats["actor/env_aux_skipped_next_obs_empty"] += 1.0
                continue

            action, is_search_action = self._extract_action(action_texts[sample_idx])
            history_pairs_all = self._normalize_history_pairs(self._get_batch_item(history_batch, sample_idx, []))
            total_history_steps = len(history_pairs_all)
            if self.aux_history_length is not None and total_history_steps > int(self.aux_history_length):
                history_pairs = history_pairs_all[-int(self.aux_history_length) :]
            else:
                history_pairs = history_pairs_all
            step_number = total_history_steps + 1
            history_start_step = total_history_steps - len(history_pairs) + 1
            admissibles = self._normalize_admissibles(self._get_batch_item(admissibles_batch, sample_idx, []))

            if self.sp_coef > 0:
                sp_msgs = create_state_prediction_messages(
                    history_pairs=history_pairs,
                    current_obs=current_obs,
                    action=action,
                    next_obs=next_obs,
                    step_number=step_number,
                    history_start_step=history_start_step,
                )
                sp_records.append(
                    {
                        "task_type": "sp",
                        "formatted_text": self.tokenizer.apply_chat_template(
                            sp_msgs,
                            tokenize=False,
                            add_generation_prompt=False,
                        ),
                    }
                )
                stats["actor/env_aux_built_sp_records"] += 1.0

            if self.id_coef > 0:
                if is_search_action:
                    id_msgs = create_search_inverse_dynamics_messages(
                        history_pairs=history_pairs,
                        current_obs=current_obs,
                        next_obs=next_obs,
                        action=action,
                        step_number=step_number,
                        history_start_step=history_start_step,
                    )
                else:
                    id_msgs = create_inverse_dynamics_messages(
                        history_pairs=history_pairs,
                        current_obs=current_obs,
                        next_obs=next_obs,
                        action=action,
                        admissible_actions=admissibles,
                        step_number=step_number,
                        history_start_step=history_start_step,
                    )
                id_records.append(
                    {
                        "task_type": "id",
                        "formatted_text": self.tokenizer.apply_chat_template(
                            id_msgs,
                            tokenize=False,
                            add_generation_prompt=False,
                        ),
                    }
                )
                stats["actor/env_aux_built_id_records"] += 1.0

        return sp_records, id_records, stats

    def compute_env_auxiliary_loss(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        sp_records, id_records, metrics = self._build_env_auxiliary_records(data)
        sp_records = self._sample_aux_records(sp_records, "sp", metrics)
        id_records = self._sample_aux_records(id_records, "id", metrics)

        device = get_torch_device().current_device()
        zero = torch.tensor(0.0, device=device)
        if not sp_records and not id_records:
            metrics["actor/env_aux_loss"] = 0.0
            return zero, metrics

        def compute_lm_loss(texts: List[str]) -> torch.Tensor:
            max_len = int(self.aux_max_length or 2560)
            aux_micro_batch_size = int(self.config.get("aux_micro_batch_size", 8) or 8)
            total_loss = None
            total_samples = 0

            for start in range(0, len(texts), aux_micro_batch_size):
                chunk = texts[start : start + aux_micro_batch_size]
                if not chunk:
                    continue
                encodings = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                if input_ids.size(1) == 0:
                    continue

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    outputs = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                weighted_loss = outputs.loss * len(chunk)
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                total_samples += len(chunk)

            if total_samples == 0 or total_loss is None:
                return zero
            return total_loss / total_samples

        total_aux_loss = zero
        if self.sp_coef > 0 and sp_records:
            sp_loss = compute_lm_loss([record["formatted_text"] for record in sp_records])
            total_aux_loss = total_aux_loss + self.sp_coef * sp_loss
            metrics["actor/sp_loss"] = sp_loss.detach().item()
        if self.id_coef > 0 and id_records:
            id_loss = compute_lm_loss([record["formatted_text"] for record in id_records])
            total_aux_loss = total_aux_loss + self.id_coef * id_loss
            metrics["actor/id_loss"] = id_loss.detach().item()

        metrics["actor/env_aux_loss"] = total_aux_loss.detach().item()
        return total_aux_loss, metrics

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
            selected_data = data.select(select_keys, non_tensor_select_keys)
            micro_batches = [
                selected_data[start : start + micro_batch_size]
                for start in range(0, len(selected_data), micro_batch_size)
            ]
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @staticmethod
    def _skill_gen_payload_tensor(payload: Dict[str, Any], key: str, device) -> torch.Tensor:
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.to(device)

    def backward_skill_gen_loss(
        self,
        payload: Dict[str, Any],
        temperature: float,
        loss_coef: float,
    ) -> Dict[str, float]:
        device = get_torch_device().current_device()
        tensors = {
            "responses": self._skill_gen_payload_tensor(payload, "responses", device),
            "input_ids": self._skill_gen_payload_tensor(payload, "input_ids", device),
            "attention_mask": self._skill_gen_payload_tensor(payload, "attention_mask", device),
            "position_ids": self._skill_gen_payload_tensor(payload, "position_ids", device),
            "rewards": self._skill_gen_payload_tensor(payload, "rewards", device).float(),
        }
        global_batch_size = int(tensors["responses"].size(0))
        if global_batch_size == 0:
            return {
                "actor/skill_gen_loss": 0.0,
                "actor/skill_gen_loss_coef": float(loss_coef),
                "actor/skill_gen_samples": 0.0,
                "actor/skill_gen_local_samples": 0.0,
            }

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        local_indices = torch.arange(rank, global_batch_size, world_size, device=device)
        local_batch_size = int(local_indices.numel())
        if local_batch_size == 0:
            return {
                "actor/skill_gen_loss": 0.0,
                "actor/skill_gen_loss_coef": float(loss_coef),
                "actor/skill_gen_samples": float(global_batch_size),
                "actor/skill_gen_local_samples": 0.0,
            }

        tensors = {
            key: value.index_select(0, local_indices)
            for key, value in tensors.items()
        }
        micro_batch_size_config = self.config.get("skill_gen_micro_batch_size_per_gpu", None)
        if micro_batch_size_config is None:
            micro_batch_size_config = self.config.get("skill_gen_micro_batch_size", 1)
        micro_batch_size = max(int(micro_batch_size_config or 1), 1)
        loss_sum = 0.0
        seq_log_prob_sum = 0.0
        seq_log_prob_count = 0
        for start in range(0, local_batch_size, micro_batch_size):
            end = min(start + micro_batch_size, local_batch_size)
            micro_batch = {
                "responses": tensors["responses"][start:end],
                "input_ids": tensors["input_ids"][start:end],
                "attention_mask": tensors["attention_mask"][start:end],
                "position_ids": tensors["position_ids"][start:end],
            }
            _, log_probs = self._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=False,
            )
            response_length = micro_batch["responses"].size(-1)
            response_mask = micro_batch["attention_mask"][:, -response_length:].to(dtype=log_probs.dtype)
            token_counts = response_mask.sum(dim=-1).clamp_min(1.0)
            seq_log_prob = (log_probs * response_mask).sum(dim=-1) / token_counts
            rewards = tensors["rewards"][start:end].to(dtype=log_probs.dtype).detach()
            loss_terms = -(rewards * seq_log_prob)
            weighted_loss = loss_coef * (float(world_size) / float(global_batch_size)) * loss_terms.sum()
            weighted_loss.backward()
            loss_sum += float(loss_terms.detach().float().sum().item())
            seq_log_prob_sum += float(seq_log_prob.detach().float().sum().item())
            seq_log_prob_count += int(seq_log_prob.numel())

        rewards = tensors["rewards"].detach().float()
        metrics = {
            "actor/skill_gen_loss": loss_sum / max(local_batch_size, 1),
            "actor/skill_gen_loss_coef": float(loss_coef),
            "actor/skill_gen_samples": float(global_batch_size),
            "actor/skill_gen_local_samples": float(local_batch_size),
            "actor/skill_gen_reward_mean": rewards.mean().item(),
            "actor/skill_gen_reward_min": rewards.min().item(),
            "actor/skill_gen_reward_max": rewards.max().item(),
            "actor/skill_gen_seq_log_prob_mean": seq_log_prob_sum / max(seq_log_prob_count, 1),
        }
        for key, value in (payload.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"actor/{key}"] = float(value)
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        self.global_step = int(data.meta_info.get("global_step", self.global_step) or 0)
        opd_only = bool(self.config.get("opd_only", False))
        task_ids_for_evidence = []
        outcome_classes_for_evidence = []
        for task_id, outcome_class in zip(
            data.non_tensor_batch.get("task_id", []),
            data.non_tensor_batch.get("june24_outcome_class", []),
        ):
            task_id = str(task_id)
            outcome_class = str(outcome_class)
            if task_id not in task_ids_for_evidence:
                task_ids_for_evidence.append(task_id)
                outcome_classes_for_evidence.append(outcome_class)
        trainable_sha_before = self._trainable_parameter_sha256() if opd_only else None
        response_token_sha256 = self._tensor_sha256(data.batch["responses"]) if opd_only else None
        task_token_metrics_for_evidence = []
        if opd_only:
            response_length_for_evidence = data.batch["responses"].size(-1)
            if multi_turn:
                generated_mask_for_evidence = data.batch["loss_mask"][:, -response_length_for_evidence:].bool()
            else:
                generated_mask_for_evidence = data.batch["attention_mask"][:, -response_length_for_evidence:].bool()
            generated_mask_sha256 = self._tensor_sha256(generated_mask_for_evidence)
            teacher_step_mask_for_evidence = data.batch.get("teacher_signal_mask")
            active_token_count_for_evidence = (
                int((generated_mask_for_evidence & teacher_step_mask_for_evidence.bool().unsqueeze(-1)).sum().item())
                if teacher_step_mask_for_evidence is not None
                else 0
            )
            if teacher_step_mask_for_evidence is not None:
                active_mask_for_task_evidence = (
                    generated_mask_for_evidence
                    & teacher_step_mask_for_evidence.bool().unsqueeze(-1)
                )
                response_tokens_per_row = (
                    generated_mask_for_evidence.sum(dim=-1).detach().cpu().tolist()
                )
                active_tokens_per_row = (
                    active_mask_for_task_evidence.sum(dim=-1).detach().cpu().tolist()
                )
                teacher_steps_per_row = (
                    teacher_step_mask_for_evidence.bool().detach().cpu().tolist()
                )
                compact_by_task = {}
                for row_index, (task_id, outcome_class) in enumerate(
                    zip(
                        data.non_tensor_batch.get("task_id", []),
                        data.non_tensor_batch.get("june24_outcome_class", []),
                    )
                ):
                    task_id = str(task_id)
                    entry = compact_by_task.setdefault(
                        task_id,
                        {
                            "task_id": task_id,
                            "outcome_class": str(outcome_class),
                            "actor_row_count": 0,
                            "teacher_step_count": 0,
                            "response_token_count": 0,
                            "active_token_count": 0,
                        },
                    )
                    entry["actor_row_count"] += 1
                    entry["teacher_step_count"] += int(bool(teacher_steps_per_row[row_index]))
                    entry["response_token_count"] += int(response_tokens_per_row[row_index])
                    entry["active_token_count"] += int(active_tokens_per_row[row_index])
                task_token_metrics_for_evidence = list(compact_by_task.values())
            active_tokens_by_class_for_evidence = {}
            outcome_class_ids_for_evidence = data.batch.get("june24_outcome_class_id")
            if outcome_class_ids_for_evidence is not None and teacher_step_mask_for_evidence is not None:
                active_mask_for_evidence = (
                    generated_mask_for_evidence
                    & teacher_step_mask_for_evidence.bool().unsqueeze(-1)
                )
                for class_id, class_name in enumerate(JUNE24_OUTCOME_CLASSES):
                    active_tokens_by_class_for_evidence[class_name] = int(
                        (
                            active_mask_for_evidence
                            & outcome_class_ids_for_evidence.eq(class_id).unsqueeze(-1)
                        ).sum().item()
                    )
        use_env_aux_loss = False if opd_only else self._env_aux_loss_enabled()
        if use_env_aux_loss and self.config.use_dynamic_bsz:
            raise RuntimeError("SP/ID environment auxiliary loss is not supported with actor.use_dynamic_bsz=True.")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if not opd_only:
            select_keys.extend(["old_log_probs", "advantages"])
        opd_loss_coef = float(self.config.get("opd_loss_coef", 0.0) or 0.0)
        skill_gen_loss_coef = float(self.config.get("skill_gen_loss_coef", 0.0) or 0.0)
        seed_skill_gen_payload = data.meta_info.get("seed_skill_gen")
        use_skill_gen_loss = (
            not opd_only
            and
            skill_gen_loss_coef > 0
            and isinstance(seed_skill_gen_payload, dict)
            and all(key in seed_skill_gen_payload for key in ("responses", "input_ids", "attention_mask", "position_ids", "rewards"))
        )
        teacher_mask_key = "teacher_signal_mask" if "teacher_signal_mask" in data.batch.keys() else "critical_step_mask"
        has_teacher_signal = "teacher_log_prob" in data.batch.keys() and teacher_mask_key in data.batch.keys()
        has_control_teacher_signal = "control_teacher_log_prob" in data.batch.keys()
        use_opd_loss = (
            opd_loss_coef > 0
            and has_teacher_signal
        )
        if opd_only:
            forbidden = {
                "entropy_coeff": float(self.config.get("entropy_coeff", 0.0) or 0.0),
                "kl_loss_coef": float(self.config.get("kl_loss_coef", 0.0) or 0.0)
                if self.config.get("use_kl_loss", False)
                else 0.0,
                "skill_gen_loss_coef": skill_gen_loss_coef,
                "sp_coef": float(self.config.get("sp_coef", 0.0) or 0.0),
                "id_coef": float(self.config.get("id_coef", 0.0) or 0.0),
            }
            active_forbidden = {key: value for key, value in forbidden.items() if value != 0.0}
            if active_forbidden:
                raise ValueError(f"actor.opd_only forbids auxiliary/RL loss coefficients: {active_forbidden}")
            if not use_opd_loss:
                raise RuntimeError("actor.opd_only requires aligned teacher_log_prob and teacher_signal_mask")
        if use_opd_loss:
            select_keys.extend(["teacher_log_prob", teacher_mask_key])
            if has_control_teacher_signal:
                select_keys.append("control_teacher_log_prob")
        if "june24_outcome_class_id" in data.batch.keys():
            select_keys.append("june24_outcome_class_id")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if use_env_aux_loss:
            non_tensor_select_keys.extend(
                [
                    "history",
                    "next_obs",
                    "anchor_obs",
                    "obs_text",
                    "admissibles",
                    "active_masks",
                    "traj_uid",
                    "data_source",
                    "is_action_valid",
                ]
            )
            non_tensor_select_keys = [key for key in non_tensor_select_keys if key in data.non_tensor_batch]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if opd_only:
            # A BFCL rollout batch contains four tasks but may flatten to more
            # than four active tool turns.  OPD accumulates every active turn
            # and performs exactly one optimizer step for the four-task batch.
            # Splitting here by ppo_mini_batch_size would make the number of
            # optimizer steps depend on trajectory length and break the frozen
            # 200-update schedule.
            dataloader = [batch]
        elif has_multi_modal_inputs or use_env_aux_loss:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if isinstance(mini_batch, DataProto):
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    effective_mini_batch_size = int(mini_batch.batch_size[0]) if opd_only else int(self.config.ppo_mini_batch_size)
                    if effective_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu != 0:
                        raise ValueError(
                            "effective OPD turn batch must be divisible by "
                            "ppo_micro_batch_size_per_gpu"
                        )
                    self.gradient_accumulation = effective_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data.get("old_log_probs")
                    advantages = data.get("advantages")

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = 0.0 if opd_only else self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    if opd_only:
                        pg_loss = log_prob.new_tensor(0.0)
                        pg_clipfrac = log_prob.new_tensor(0.0)
                        ppo_kl = log_prob.new_tensor(0.0)
                        pg_clipfrac_lower = log_prob.new_tensor(0.0)
                        policy_loss = log_prob.new_tensor(0.0)
                    else:
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        if loss_mode == "vanilla":
                            policy_loss_fn = compute_policy_loss
                        elif loss_mode == "gspo":
                            policy_loss_fn = compute_policy_loss_gspo
                        else:
                            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                    opd_loss = log_prob.new_tensor(0.0)
                    opd_active_token_ratio = log_prob.new_tensor(0.0)
                    opd_gate_mean = log_prob.new_tensor(0.0)
                    opd_gate_active_ratio = log_prob.new_tensor(0.0)
                    opd_teacher_gap_mean = log_prob.new_tensor(0.0)
                    control_opd_loss = log_prob.new_tensor(0.0)
                    control_opd_active_token_ratio = log_prob.new_tensor(0.0)
                    control_opd_gate_mean = log_prob.new_tensor(0.0)
                    control_opd_gate_active_ratio = log_prob.new_tensor(0.0)
                    control_opd_teacher_gap_mean = log_prob.new_tensor(0.0)
                    if use_opd_loss and "teacher_log_prob" in data and teacher_mask_key in data:
                        (
                            opd_loss,
                            opd_active_token_ratio,
                            opd_gate_mean,
                            opd_gate_active_ratio,
                            opd_teacher_gap_mean,
                        ) = compute_opd_loss(
                            log_prob=log_prob,
                            teacher_log_prob=data["teacher_log_prob"],
                            response_mask=response_mask,
                            opd_step_mask=data[teacher_mask_key],
                            gate_beta=self.config.get("opd_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = compose_actor_objective(
                            policy_objective=policy_loss,
                            opd_loss=opd_loss,
                            opd_loss_coef=opd_loss_coef,
                            opd_only=opd_only,
                        )
                        if "control_teacher_log_prob" in data:
                            (
                                control_opd_loss,
                                control_opd_active_token_ratio,
                                control_opd_gate_mean,
                                control_opd_gate_active_ratio,
                                control_opd_teacher_gap_mean,
                            ) = compute_opd_loss(
                                log_prob=log_prob,
                                teacher_log_prob=data["control_teacher_log_prob"],
                                response_mask=response_mask,
                                opd_step_mask=data[teacher_mask_key],
                                gate_beta=self.config.get("opd_gate_beta", 5.0),
                                loss_agg_mode=loss_agg_mode,
                            )

                        if "june24_outcome_class_id" in data:
                            for class_id, class_name in enumerate(JUNE24_OUTCOME_CLASSES):
                                class_rows = data["june24_outcome_class_id"].eq(class_id)
                                if not bool(class_rows.any()):
                                    continue
                                (
                                    class_opd_loss,
                                    _,
                                    class_gate_mean,
                                    _,
                                    class_gap_mean,
                                ) = compute_opd_loss(
                                    log_prob=log_prob[class_rows],
                                    teacher_log_prob=data["teacher_log_prob"][class_rows],
                                    response_mask=response_mask[class_rows],
                                    opd_step_mask=data[teacher_mask_key][class_rows],
                                    gate_beta=self.config.get("opd_gate_beta", 5.0),
                                    loss_agg_mode=loss_agg_mode,
                                )
                                append_to_dict(
                                    metrics,
                                    {
                                        f"actor/opd_by_class/{class_name}/loss": class_opd_loss.detach().item(),
                                        f"actor/opd_by_class/{class_name}/gate_mean": class_gate_mean.detach().item(),
                                        f"actor/opd_by_class/{class_name}/teacher_gap_mean": class_gap_mean.detach().item(),
                                    },
                                )
                                if "control_teacher_log_prob" in data:
                                    (
                                        class_control_loss,
                                        _,
                                        class_control_gate_mean,
                                        _,
                                        class_control_gap_mean,
                                    ) = compute_opd_loss(
                                        log_prob=log_prob[class_rows],
                                        teacher_log_prob=data["control_teacher_log_prob"][class_rows],
                                        response_mask=response_mask[class_rows],
                                        opd_step_mask=data[teacher_mask_key][class_rows],
                                        gate_beta=self.config.get("opd_gate_beta", 5.0),
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                    append_to_dict(
                                        metrics,
                                        {
                                            f"actor/control_by_class/{class_name}/loss": class_control_loss.detach().item(),
                                            f"actor/control_by_class/{class_name}/gate_mean": class_control_gate_mean.detach().item(),
                                            f"actor/control_by_class/{class_name}/teacher_gap_mean": class_control_gap_mean.detach().item(),
                                        },
                                    )

                    if self.config.use_kl_loss and not opd_only:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    env_aux_loss = log_prob.new_tensor(0.0)
                    env_aux_metrics = {
                        "actor/env_aux_loss": 0.0,
                        "actor/sp_coef": self.sp_coef,
                        "actor/id_coef": self.id_coef,
                    }
                    if use_env_aux_loss:
                        env_aux_loss, env_aux_metrics = self.compute_env_auxiliary_loss(data)
                        policy_loss = policy_loss + env_aux_loss

                    if opd_only and loss_agg_mode == "token-mean":
                        micro_opd_mask = response_mask * data[teacher_mask_key].to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        ).unsqueeze(-1)
                        micro_active_tokens = micro_opd_mask.sum()
                        if active_token_count_for_evidence <= 0:
                            raise RuntimeError("OPD-only update has zero active tokens")
                        # Match one full-batch compute_opd_loss(token-mean)
                        # exactly while keeping one-turn microbatches for memory.
                        loss = policy_loss * (
                            micro_active_tokens / float(active_token_count_for_evidence)
                        )
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/opd_loss": opd_loss.detach().item(),
                        "actor/opd_loss_coef": opd_loss_coef,
                        "actor/opd_active_token_ratio": opd_active_token_ratio.detach().item(),
                        "actor/opd_gate_mean": opd_gate_mean.detach().item(),
                        "actor/opd_gate_active_ratio": opd_gate_active_ratio.detach().item(),
                        "actor/opd_teacher_gap_mean": opd_teacher_gap_mean.detach().item(),
                        "actor/control_opd_loss": control_opd_loss.detach().item(),
                        "actor/control_opd_active_token_ratio": control_opd_active_token_ratio.detach().item(),
                        "actor/control_opd_gate_mean": control_opd_gate_mean.detach().item(),
                        "actor/control_opd_gate_active_ratio": control_opd_gate_active_ratio.detach().item(),
                        "actor/control_opd_teacher_gap_mean": control_opd_teacher_gap_mean.detach().item(),
                        "actor/opd_only": 1.0 if opd_only else 0.0,
                        "actor/rl_gradient_contribution": 0.0 if opd_only else 1.0,
                        "actor/env_aux_loss": env_aux_loss.detach().item(),
                        "actor/sp_coef": self.sp_coef,
                        "actor/id_coef": self.id_coef,
                    }
                    data.update(env_aux_metrics)
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)

        if use_skill_gen_loss:
            self.actor_optimizer.zero_grad()
            skill_gen_metrics = self.backward_skill_gen_loss(
                seed_skill_gen_payload,
                temperature=temperature,
                loss_coef=skill_gen_loss_coef,
            )
            skill_gen_grad_norm = self._optimizer_step()
            skill_gen_metrics["actor/skill_gen_grad_norm"] = skill_gen_grad_norm.detach().item()
            append_to_dict(metrics, skill_gen_metrics)
        if opd_only:
            trainable_sha_after = self._trainable_parameter_sha256()
            checksum_changed = trainable_sha_before != trainable_sha_after
            append_to_dict(metrics, {"actor/trainable_checksum_changed": float(checksum_changed)})
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                evidence = {
                    "global_step": self.global_step,
                    "trainable_sha256_before": trainable_sha_before,
                    "trainable_sha256_after": trainable_sha_after,
                    "changed": checksum_changed,
                }
                evidence_path = self.config.get("opd_evidence_path")
                if evidence_path:
                    path = Path(str(evidence_path)).expanduser()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(
                    "SEED_OPD_TRAINABLE_CHECKSUM "
                    f"before={trainable_sha_before} after={trainable_sha_after} changed={checksum_changed}",
                    flush=True,
                )
                diagnostics_root = self.config.get("opd_diagnostics_path")
                if diagnostics_root:
                    updates_per_iteration = int(
                        self.config.get("opd_updates_per_iteration") or 40
                    )
                    if updates_per_iteration <= 0:
                        raise RuntimeError("OPD evidence updates-per-iteration must be positive")

                    def _metric_mean(key: str) -> float:
                        values = metrics.get(key, [])
                        if not isinstance(values, (list, tuple)):
                            values = [values]
                        return float(sum(float(value) for value in values) / len(values)) if values else 0.0

                    diagnostics = {
                        "global_step": self.global_step,
                        "iteration": ((self.global_step - 1) // updates_per_iteration) + 1,
                        "batch_in_iteration": ((self.global_step - 1) % updates_per_iteration) + 1,
                        "updates_per_iteration": updates_per_iteration,
                        "task_ids": task_ids_for_evidence,
                        "outcome_classes": outcome_classes_for_evidence,
                        "outcome_class_counts": {
                            class_name: outcome_classes_for_evidence.count(class_name)
                            for class_name in JUNE24_OUTCOME_CLASSES
                        },
                        "response_token_sha256": response_token_sha256,
                        "generated_token_mask_sha256": generated_mask_sha256,
                        "generated_token_only_mask": True,
                        "teacher_student_token_alignment": True,
                        "control_teacher_student_token_alignment": has_control_teacher_signal,
                        "active_token_count": active_token_count_for_evidence,
                        "active_tokens_by_outcome_class": active_tokens_by_class_for_evidence,
                        "task_token_metrics": task_token_metrics_for_evidence,
                        "learning_rate_used": float(self.actor_optimizer.param_groups[0]["lr"]),
                        "optimizer": type(self.actor_optimizer).__name__,
                        "optimizer_steps_in_batch": 1,
                        "trainable_sha256_before": trainable_sha_before,
                        "trainable_sha256_after": trainable_sha_after,
                        "trainable_changed": checksum_changed,
                        "metrics": {
                            key: _metric_mean(key)
                            for key in (
                                "actor/opd_loss",
                                "actor/opd_gate_mean",
                                "actor/opd_teacher_gap_mean",
                                "actor/control_opd_loss",
                                "actor/control_opd_gate_mean",
                                "actor/control_opd_teacher_gap_mean",
                                "actor/grad_norm",
                                "actor/rl_gradient_contribution",
                            )
                        },
                        "outcome_class_metrics": {
                            class_name: {
                                "opd_loss": _metric_mean(f"actor/opd_by_class/{class_name}/loss"),
                                "gate_mean": _metric_mean(f"actor/opd_by_class/{class_name}/gate_mean"),
                                "teacher_gap_mean": _metric_mean(
                                    f"actor/opd_by_class/{class_name}/teacher_gap_mean"
                                ),
                                "control_opd_loss": _metric_mean(
                                    f"actor/control_by_class/{class_name}/loss"
                                ),
                                "control_gate_mean": _metric_mean(
                                    f"actor/control_by_class/{class_name}/gate_mean"
                                ),
                                "control_teacher_gap_mean": _metric_mean(
                                    f"actor/control_by_class/{class_name}/teacher_gap_mean"
                                ),
                            }
                            for class_name in JUNE24_OUTCOME_CLASSES
                            if active_tokens_by_class_for_evidence.get(class_name, 0) > 0
                        },
                    }
                    diagnostics["privileged_minus_control"] = {
                        "teacher_gap_mean": (
                            diagnostics["metrics"]["actor/opd_teacher_gap_mean"]
                            - diagnostics["metrics"]["actor/control_opd_teacher_gap_mean"]
                        ),
                        "gate_mean": (
                            diagnostics["metrics"]["actor/opd_gate_mean"]
                            - diagnostics["metrics"]["actor/control_opd_gate_mean"]
                        ),
                        "opd_loss": (
                            diagnostics["metrics"]["actor/opd_loss"]
                            - diagnostics["metrics"]["actor/control_opd_loss"]
                        ),
                    }
                    diagnostics_path = Path(str(diagnostics_root)).expanduser() / f"step_{self.global_step:06d}.json"
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostics_path.write_text(
                        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
        self.actor_optimizer.zero_grad()
        return metrics
