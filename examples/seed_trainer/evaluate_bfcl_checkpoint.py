"""Run an official plain-prompt BFCL evaluation for selected task IDs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _install_capped_vllm_server(max_model_len: int) -> None:
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    original = OSSHandler.spin_up_local_server

    def spin_up(self, num_gpus, gpu_memory_utilization, backend, skip_server_setup, local_model_path):
        if backend != "vllm" or skip_server_setup:
            return original(
                self,
                num_gpus,
                gpu_memory_utilization,
                backend,
                skip_server_setup,
                local_model_path,
            )
        model_source = local_model_path or self.model_name_huggingface
        served_name = self.model_name_huggingface
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(model_source),
                "--served-model-name",
                str(served_name),
                "--port",
                str(self.local_server_port),
                "--dtype",
                str(self.dtype),
                "--tensor-parallel-size",
                str(num_gpus),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--max-model-len",
                str(max_model_len),
                "--trust-remote-code",
            ]
        )
        try:
            result = original(
                self,
                num_gpus,
                gpu_memory_utilization,
                backend,
                True,
                local_model_path,
            )
        except Exception:
            process.terminate()
            raise
        self._server_process = process
        self.model_path_or_id = served_name
        return result

    OSSHandler.spin_up_local_server = spin_up


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--local-model-path", type=Path, default=None)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bfcl_root = args.bfcl_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    if not task_ids:
        raise ValueError("--task-ids is empty")
    if str(bfcl_root) not in sys.path:
        sys.path.insert(0, str(bfcl_root))
    project_root.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["LOCAL_SERVER_PORT"] = str(args.port)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    from bfcl_eval import _llm_response_generation as generation_module
    from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH, TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.utils import get_directory_structure_by_category, get_file_name_by_category

    Path(TEST_IDS_TO_GENERATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TEST_IDS_TO_GENERATE_PATH).write_text(
        json.dumps({args.category: task_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    _install_capped_vllm_server(args.max_model_len)
    generation_module.main(
        SimpleNamespace(
            model=args.model,
            test_category=args.category,
            temperature=0.001,
            include_input_log=False,
            exclude_state_log=False,
            num_threads=1,
            num_gpus=1,
            backend="vllm",
            gpu_memory_utilization=args.gpu_memory_utilization,
            result_dir=None,
            run_ids=True,
            allow_overwrite=True,
            skip_server_setup=False,
            local_model_path=str(args.local_model_path.expanduser().resolve())
            if args.local_model_path
            else None,
        )
    )
    evaluation_main([args.model], [args.category], None, None, partial_eval=True)

    model_dir = args.model.replace("/", "_")
    result_path = (
        RESULT_PATH
        / model_dir
        / get_directory_structure_by_category(args.category)
        / get_file_name_by_category(args.category, is_result_file=True)
    )
    score_path = (
        SCORE_PATH
        / model_dir
        / get_directory_structure_by_category(args.category)
        / get_file_name_by_category(args.category, is_score_file=True)
    )
    score_rows = _records(score_path)
    aggregate = next((row for row in score_rows if isinstance(row, dict) and "accuracy" in row), None)
    by_id = {str(row.get("id")): row for row in score_rows if isinstance(row, dict) and row.get("id")}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    # BFCL can omit valid per-task score rows; aggregate plus result coverage is
    # still official evidence, but every selected result must exist.
    result_ids = {
        str(row.get("id")) for row in _records(result_path) if isinstance(row, dict) and row.get("id")
    }
    missing_results = [task_id for task_id in task_ids if task_id not in result_ids]
    if aggregate is None or missing_results:
        raise RuntimeError(
            f"official BFCL evaluation is incomplete: aggregate={aggregate is not None}, "
            f"missing_results={missing_results}"
        )
    summary = {
        "task_ids": task_ids,
        "model": args.model,
        "local_model_path": str(args.local_model_path.expanduser().resolve())
        if args.local_model_path
        else None,
        "result_path": str(result_path),
        "score_path": str(score_path),
        "aggregate": aggregate,
        "score_rows_present": sorted(set(task_ids) - set(missing)),
        "score_rows_omitted_by_bfcl": missing,
    }
    summary_path = project_root / "official_bfcl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SEED_BFCL_OFFICIAL_EVAL_OK summary={summary_path}")


if __name__ == "__main__":
    main()
