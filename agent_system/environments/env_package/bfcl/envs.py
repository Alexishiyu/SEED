"""BFCL simulator routing for SEED.

Each SEED step is one exact model generation.  The official BFCL handler owns
the conversation and simulator state; its formatted prompt is passed through
unchanged on every step.  Tool observations therefore enter only the next
prompt and are never copied into a sampled response tensor.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase


class BFCLEnvironmentError(RuntimeError):
    """Raised when the pinned official BFCL environment cannot be used."""


def _add_bfcl_root(bfcl_root: Path) -> None:
    root = bfcl_root.expanduser().resolve()
    if not (root / "bfcl_eval").is_dir():
        raise BFCLEnvironmentError(f"BFCL root does not contain bfcl_eval: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_bfcl_components(bfcl_root: Path) -> dict[str, Any]:
    _add_bfcl_root(bfcl_root)
    try:
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        )
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
            is_empty_execute_response,
        )
        from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    except Exception as exc:  # pragma: no cover - exercised in the pinned Colab runtime.
        raise BFCLEnvironmentError(
            "Could not import the pinned official bfcl_eval package from " f"{bfcl_root}"
        ) from exc
    return {
        "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        "MODEL_CONFIG_MAPPING": MODEL_CONFIG_MAPPING,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
        "is_empty_execute_response": is_empty_execute_response,
        "load_dataset_entry": load_dataset_entry,
        "load_ground_truth_entry": load_ground_truth_entry,
        "multi_turn_checker": multi_turn_checker,
    }


def _make_handler(components: Mapping[str, Any], model_id: str, temperature: float) -> Any:
    mapping = components["MODEL_CONFIG_MAPPING"]
    if model_id not in mapping:
        raise BFCLEnvironmentError(f"unknown BFCL model id: {model_id}")
    config = mapping[model_id]
    return config.model_handler(
        model_name=config.model_name,
        temperature=temperature,
        registry_name=model_id,
        is_fc_model=config.is_fc_model,
    )


def _find_by_id(rows: Sequence[Mapping[str, Any]], task_id: str, label: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("id")) == task_id]
    if len(matches) != 1:
        raise BFCLEnvironmentError(f"{task_id}: expected one official {label}, found {len(matches)}")
    return copy.deepcopy(dict(matches[0]))


def hydrate_task(components: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("id") or task.get("task_id") or "").strip()
    if not task_id:
        raise BFCLEnvironmentError("BFCL task has no id")
    category = str(task.get("test_category") or task_id.rsplit("_", 1)[0])
    if task.get("question") and task.get("function"):
        hydrated = copy.deepcopy(dict(task))
    else:
        hydrated = _find_by_id(components["load_dataset_entry"](category), task_id, "task")
    hydrated["id"] = task_id
    hydrated.setdefault("test_category", category)
    if "possible_answer" not in hydrated:
        ground_truth = _find_by_id(
            components["load_ground_truth_entry"](category), task_id, "ground truth"
        )
        hydrated["possible_answer"] = ground_truth.get(
            "possible_answer", ground_truth.get("ground_truth")
        )
    possible_answer = hydrated.get("possible_answer")
    if not isinstance(possible_answer, list):
        raise BFCLEnvironmentError(f"{task_id}: official possible_answer is unavailable")
    return hydrated


class BFCLWorker:
    def __init__(
        self,
        *,
        components: Mapping[str, Any],
        model_id: str,
        temperature: float,
        max_steps_per_turn: int,
    ) -> None:
        self.components = components
        self.model_id = model_id
        self.temperature = temperature
        self.max_steps_per_turn = max_steps_per_turn
        self.done = True
        self.last_observation = ""
        self.last_info: dict[str, Any] = {}

    def _format_prompt(self) -> str:
        prompt = self.handler._format_prompt(
            self.inference_data["message"], self.inference_data["function"]
        )
        if not isinstance(prompt, str) or not prompt:
            raise BFCLEnvironmentError(f"{self.task_id}: official handler returned an empty prompt")
        return prompt

    def _turn_message(self, turn_id: int) -> list[dict[str, Any]]:
        missed = self.task.get("missed_function", {}) or {}
        if str(turn_id) in missed:
            return [
                {
                    "role": "user",
                    "content": self.components[
                        "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING"
                    ].format(functions=missed[str(turn_id)]),
                }
            ]
        return copy.deepcopy(self.task["question"][turn_id])

    def _add_turn(self, turn_id: int) -> None:
        message = self._turn_message(turn_id)
        if turn_id == 0:
            self.inference_data = self.handler.add_first_turn_message_prompting(
                self.inference_data, message
            )
        else:
            self.inference_data = self.handler._add_next_turn_user_message_prompting(
                self.inference_data, message
            )

    def _base_info(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "step_in_turn": self.step_in_turn,
            "won": False,
            "is_action_valid": False,
            "tool_calling": False,
        }

    def reset(self, kwargs: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        raw_task = kwargs.get("bfcl_task_json", kwargs.get("bfcl_task"))
        if isinstance(raw_task, str):
            try:
                raw_task = json.loads(raw_task)
            except json.JSONDecodeError as exc:
                raise BFCLEnvironmentError("bfcl_task_json is invalid JSON") from exc
        if not isinstance(raw_task, Mapping):
            raise BFCLEnvironmentError("env_kwargs must contain bfcl_task_json or bfcl_task")
        self.task = hydrate_task(self.components, raw_task)
        self.task_id = str(self.task["id"])
        self.category = str(self.task.get("test_category") or self.task_id.rsplit("_", 1)[0])
        self.handler = _make_handler(self.components, self.model_id, self.temperature)
        self.initial_config = copy.deepcopy(self.task.get("initial_config", {}))
        self.involved_classes = copy.deepcopy(self.task.get("involved_classes") or [])
        self.components["execute_multi_turn_func_call"](
            [],
            self.initial_config,
            self.involved_classes,
            self.handler.model_name_underline_replaced,
            self.task_id,
            long_context=("long_context" in self.category or "composite" in self.category),
            is_evaL_run=False,
        )
        self.inference_data = self.handler._pre_query_processing_prompting(self.task)
        self.turn_id = 0
        self.step_in_turn = 0
        self.decoded_turns: list[list[list[str]]] = [
            [] for _ in range(len(self.task["question"]))
        ]
        self.raw_turns: list[list[str]] = [[] for _ in range(len(self.task["question"]))]
        self.done = False
        self._add_turn(0)
        self.last_observation = self._format_prompt()
        self.last_info = self._base_info()
        return self.last_observation, dict(self.last_info)

    def _verify(self) -> dict[str, Any]:
        try:
            result = self.components["multi_turn_checker"](
                self.decoded_turns,
                self.task["possible_answer"],
                self.task,
                self.category,
                self.handler.model_name_underline_replaced,
            )
        except Exception as exc:
            return {
                "valid": False,
                "error_type": "multi_turn:verification_exception",
                "error_message": str(exc),
            }
        return dict(result)

    def _finish(self, reason: str, *, action_valid: bool) -> tuple[str, float, bool, dict[str, Any]]:
        self.done = True
        verification = self._verify()
        won = bool(verification.get("valid"))
        info = self._base_info()
        info.update(
            {
                "won": won,
                "is_action_valid": action_valid,
                "termination": reason,
                "verifier_result": verification,
                "raw_turns": copy.deepcopy(self.raw_turns),
                "decoded_turns": copy.deepcopy(self.decoded_turns),
            }
        )
        self.last_info = info
        return self.last_observation, float(won), True, dict(info)

    def step(self, text_action: str) -> tuple[str, float, bool, dict[str, Any]]:
        if self.done:
            return self.last_observation, 0.0, True, dict(self.last_info)

        text_action = str(text_action)
        self.raw_turns[self.turn_id].append(text_action)
        model_response_data = {
            "model_responses": text_action,
            "reasoning_content": "",
            "input_token": 0,
            "output_token": 0,
        }
        self.inference_data = self.handler._add_assistant_message_prompting(
            self.inference_data, model_response_data
        )

        decoded: list[str] | None = None
        decode_error: str | None = None
        try:
            decoded = self.handler.decode_execute(text_action, has_tool_call_tag=False)
        except Exception as exc:
            decode_error = str(exc)

        is_empty = decoded is None or self.components["is_empty_execute_response"](decoded)
        if decode_error is None and not is_empty:
            decoded_list = list(decoded)
            self.decoded_turns[self.turn_id].append(decoded_list)
            model_response_data["model_responses_decoded"] = decoded_list
            execution_results, _ = self.components["execute_multi_turn_func_call"](
                decoded_list,
                self.initial_config,
                self.involved_classes,
                self.handler.model_name_underline_replaced,
                self.task_id,
                long_context=("long_context" in self.category or "composite" in self.category),
                is_evaL_run=False,
            )
            self.inference_data = self.handler._add_execution_results_prompting(
                self.inference_data, execution_results, model_response_data
            )
            self.step_in_turn += 1
            self.last_observation = self._format_prompt()
            if self.step_in_turn > self.max_steps_per_turn:
                return self._finish("maximum_step_limit", action_valid=True)
            info = self._base_info()
            info.update(
                {
                    "is_action_valid": True,
                    "tool_calling": True,
                    "decoded_calls": copy.deepcopy(decoded_list),
                    "tool_results": copy.deepcopy(execution_results),
                }
            )
            self.last_info = info
            return self.last_observation, 0.0, False, dict(info)

        completed_turn = self.turn_id
        self.turn_id += 1
        self.step_in_turn = 0
        if self.turn_id >= len(self.task["question"]):
            reason = "completed" if decode_error is None else "final_decode_error"
            return self._finish(reason, action_valid=False)

        self._add_turn(self.turn_id)
        self.last_observation = self._format_prompt()
        info = self._base_info()
        info.update(
            {
                "is_action_valid": False,
                "termination": "turn_complete" if decode_error is None else "decode_error",
                "completed_turn_id": completed_turn,
                "decode_error": decode_error,
            }
        )
        self.last_info = info
        return self.last_observation, 0.0, False, dict(info)


class BFCLEnvs:
    """Synchronous vector wrapper around independent official BFCL workers."""

    def __init__(
        self,
        *,
        bfcl_root: Path,
        model_id: str,
        temperature: float,
        env_num: int,
        group_n: int,
        max_steps_per_turn: int,
    ) -> None:
        self.batch_size = int(env_num) * int(group_n)
        components = load_bfcl_components(bfcl_root)
        self.workers = [
            BFCLWorker(
                components=components,
                model_id=model_id,
                temperature=temperature,
                max_steps_per_turn=max_steps_per_turn,
            )
            for _ in range(self.batch_size)
        ]

    def reset(self, kwargs: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        if kwargs is None or len(kwargs) != self.batch_size:
            got = 0 if kwargs is None else len(kwargs)
            raise BFCLEnvironmentError(
                f"BFCL reset requires exactly {self.batch_size} task rows, got {got}"
            )
        results = [worker.reset(row) for worker, row in zip(self.workers, kwargs)]
        observations, infos = zip(*results)
        return list(observations), list(infos)

    def step(self, actions: Sequence[str]):
        if len(actions) != self.batch_size:
            raise BFCLEnvironmentError(
                f"BFCL step requires exactly {self.batch_size} actions, got {len(actions)}"
            )
        results = [worker.step(action) for worker, action in zip(self.workers, actions)]
        observations, rewards, dones, infos = zip(*results)
        return list(observations), np.asarray(rewards), np.asarray(dones), list(infos)

    def close(self) -> None:
        return None


class BFCLEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs: BFCLEnvs, config: Any) -> None:
        super().__init__(envs=envs, projection_f=lambda value: value, config=config)

    @staticmethod
    def _observations(texts: Sequence[str], infos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "text": list(texts),
            "text_base": list(texts),
            "image": None,
            "anchor": list(texts),
            "preformatted": [True] * len(texts),
            "task_id": [str(info.get("task_id") or "") for info in infos],
            "bfcl_turn_id": [int(info.get("turn_id") or 0) for info in infos],
            "bfcl_step_in_turn": [int(info.get("step_in_turn") or 0) for info in infos],
        }

    def reset(self, kwargs):
        texts, infos = self.envs.reset(kwargs=kwargs)
        return self._observations(texts, infos), infos

    def step(self, text_actions):
        texts, rewards, dones, infos = self.envs.step(text_actions)
        return self._observations(texts, infos), rewards, dones, infos


def build_bfcl_envs(
    *,
    bfcl_root: Path,
    model_id: str,
    temperature: float,
    env_num: int,
    group_n: int,
    max_steps_per_turn: int,
) -> BFCLEnvs:
    return BFCLEnvs(
        bfcl_root=bfcl_root,
        model_id=model_id,
        temperature=temperature,
        env_num=env_num,
        group_n=group_n,
        max_steps_per_turn=max_steps_per_turn,
    )
