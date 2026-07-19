from types import SimpleNamespace

from agent_system.environments.env_package.bfcl.envs import BFCLWorker, hydrate_task


class FakeHandler:
    model_name_underline_replaced = "fake_model"

    def __init__(self, **kwargs):
        pass

    def _pre_query_processing_prompting(self, task):
        return {"message": [{"role": "system", "content": "ordinary BFCL"}], "function": task["function"]}

    def add_first_turn_message_prompting(self, data, message):
        data["message"].extend(message)
        return data

    def _add_next_turn_user_message_prompting(self, data, message):
        data["message"].extend(message)
        return data

    def _add_assistant_message_prompting(self, data, response):
        data["message"].append({"role": "assistant", "content": response["model_responses"]})
        return data

    def _add_execution_results_prompting(self, data, results, response):
        for result in results:
            data["message"].append({"role": "tool", "content": result})
        return data

    def _format_prompt(self, messages, functions):
        body = "\n".join(f"{item['role']}:{item['content']}" for item in messages)
        return f"<|im_start|>system\nordinary BFCL<|im_end|>\n{body}\nassistant:"

    def decode_execute(self, text, has_tool_call_tag=False):
        if text == "bad":
            raise ValueError("parse error")
        if text == "done":
            return []
        if text == "multi":
            return ["one()", "two()"]
        return [text]


def _components(executions):
    def execute(calls, initial_config, involved_classes, model_name, task_id, **kwargs):
        executions.append(
            {
                "calls": list(calls),
                "initial_config": initial_config,
                "task_id": task_id,
            }
        )
        return [f"result:{call}" for call in calls], {}

    def checker(decoded_turns, possible_answer, task, category, model_name):
        flattened = [call for turn in decoded_turns for step in turn for call in step]
        return {"valid": flattened == ["one()", "two()"]}

    return {
        "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING": "functions={functions}",
        "MODEL_CONFIG_MAPPING": {
            "fake": SimpleNamespace(
                model_handler=FakeHandler,
                model_name="fake",
                is_fc_model=False,
            )
        },
        "execute_multi_turn_func_call": execute,
        "is_empty_execute_response": lambda value: not value,
        "load_dataset_entry": lambda category: [
            {key: value for key, value in _task().items() if key != "possible_answer"}
        ],
        "load_ground_truth_entry": lambda category: [
            {
                "id": "multi_turn_base_0",
                "ground_truth": [["one()", "two()"]],
            }
        ],
        "multi_turn_checker": checker,
    }


def _task(turns=1):
    return {
        "id": "multi_turn_base_0",
        "test_category": "multi_turn_base",
        "question": [[{"role": "user", "content": f"turn {index}"}] for index in range(turns)],
        "function": [{"name": "one"}, {"name": "two"}],
        "initial_config": {},
        "involved_classes": [],
        "possible_answer": [["one()", "two()"]] + [[] for _ in range(turns - 1)],
    }


def test_valid_multicall_completion_and_empty_state_are_preserved():
    executions = []
    worker = BFCLWorker(
        components=_components(executions),
        model_id="fake",
        temperature=0.0,
        max_steps_per_turn=3,
    )
    prompt, info = worker.reset({"bfcl_task": _task()})
    assert prompt.startswith("<|im_start|>system")
    assert info["task_id"] == "multi_turn_base_0"
    _, reward, done, info = worker.step("multi")
    assert not done and reward == 0.0
    assert info["decoded_calls"] == ["one()", "two()"]
    _, reward, done, info = worker.step("done")
    assert done and reward == 1.0
    assert info["verifier_result"]["valid"] is True
    assert executions[0]["initial_config"] == {}


def test_official_ground_truth_field_hydrates_verifier_answers():
    task = hydrate_task(_components([]), {"id": "multi_turn_base_0"})
    assert task["possible_answer"] == [["one()", "two()"]]


def test_invalid_action_advances_turn_and_preserves_turn_ids():
    worker = BFCLWorker(
        components=_components([]),
        model_id="fake",
        temperature=0.0,
        max_steps_per_turn=3,
    )
    worker.reset({"bfcl_task": _task(turns=2)})
    _, reward, done, info = worker.step("bad")
    assert not done and reward == 0.0
    assert info["is_action_valid"] is False
    assert info["turn_id"] == 1
    assert info["completed_turn_id"] == 0


def test_tool_observation_enters_next_prompt_not_the_response():
    worker = BFCLWorker(
        components=_components([]),
        model_id="fake",
        temperature=0.0,
        max_steps_per_turn=3,
    )
    worker.reset({"bfcl_task": _task()})
    prompt, _, _, _ = worker.step("multi")
    assert "tool:result:one()" in prompt
    assert worker.raw_turns == [["multi"]]
