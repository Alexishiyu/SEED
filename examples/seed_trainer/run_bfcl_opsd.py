"""Plan and launch June 24 Skill-SD OPD-only training on official BFCL tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from agent_system.environments.env_package.bfcl.envs import (
    BFCLWorker,
    hydrate_task,
    load_bfcl_components,
)
from seed.june24_skill_summary import (
    load_skill_bank,
    sha256_file,
    validate_context_lengths,
    validate_fixed_manifest,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_task_ids(value: str | None, fixed_ids: list[str]) -> list[str]:
    if value is None:
        return fixed_ids[:4]
    task_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("--task-ids must contain unique comma-separated ids")
    return task_ids


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _a100_guard() -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if count != 1 or "A100" not in names[0].upper():
        raise RuntimeError(f"training requires exactly one NVIDIA A100, found count={count}, names={names}")
    return {"device_count": count, "device_names": names}


def _official_tasks_and_prompts(
    *,
    bfcl_root: Path,
    model: str,
    task_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    components = load_bfcl_components(bfcl_root)
    tasks = []
    prompts: dict[str, str] = {}
    for task_id in task_ids:
        task = hydrate_task(components, {"id": task_id})
        worker = BFCLWorker(
            components=components,
            model_id=model,
            temperature=0.001,
            max_steps_per_turn=20,
        )
        prompt, info = worker.reset({"bfcl_task": task})
        if info["task_id"] != task_id:
            raise RuntimeError(f"official BFCL reset changed task id {task_id} to {info['task_id']}")
        tasks.append(task)
        prompts[task_id] = prompt
    return tasks, prompts


def _write_dataset(path: Path, tasks: list[dict[str, Any]]) -> None:
    import pandas as pd

    rows = []
    for index, task in enumerate(tasks):
        task_id = str(task["id"])
        rows.append(
            {
                "data_source": "bfcl",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "official_bfcl_multi_turn",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": index, "task_id": task_id},
                # JSON encoding is intentional: PyArrow must preserve empty
                # nested state such as initial_config={} without inferring an
                # unsupported empty struct.
                "env_kwargs": {
                    "bfcl_task_json": json.dumps(task, ensure_ascii=True, separators=(",", ":")),
                    "task_id": task_id,
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _hydra_command(args: argparse.Namespace, task_ids: list[str], train_path: Path) -> list[str]:
    total_model_len = args.max_prompt_length + args.max_response_length
    checkpoint_root = args.run_root / "checkpoints" / args.arm
    evidence_root = args.run_root / "evidence" / args.arm
    task_csv = ",".join(task_ids)
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=seed",
        "algorithm.use_kl_in_reward=False",
        "algorithm.seed.step_advantage_w=0.0",
        "algorithm.seed.episode_skill_teacher_advantage_w=0.0",
        "algorithm.seed.step_skill_teacher_advantage_w=0.0",
        "algorithm.seed.skill_mode=episode_only",
        "algorithm.seed.skill_teacher_mode=step_priority",
        "algorithm.seed.enable_analysis=True",
        "algorithm.seed.selector=llm",
        "algorithm.seed.failed_only=False",
        "algorithm.seed.analysis_backend=june24_skill_summary",
        f"algorithm.seed.june24_skill_bank={args.june24_skill_bank}",
        f"algorithm.seed.june24_fixed_manifest={args.fixed_manifest}",
        f"algorithm.seed.june24_task_ids='{task_csv}'",
        f"algorithm.seed.june24_same_prompt_control={str(args.same_prompt_control)}",
        "algorithm.seed.june24_verify_source_files=True",
        "algorithm.seed.skill_gen.enable=False",
        f"data.train_files={train_path}",
        f"data.val_files={train_path}",
        f"data.train_batch_size={args.batch_size}",
        f"data.val_batch_size={args.batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "data.shuffle=False",
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.use_invalid_action_penalty=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.skill_gen_loss_coef=0.0",
        "actor_rollout_ref.actor.sp_coef=0.0",
        "actor_rollout_ref.actor.id_coef=0.0",
        "actor_rollout_ref.actor.opd_only=True",
        "actor_rollout_ref.actor.opd_loss_coef=1.0",
        "actor_rollout_ref.actor.opd_gate_beta=5.0",
        f"actor_rollout_ref.actor.opd_evidence_path={evidence_root / 'trainable_checksum.json'}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={total_model_len}",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        f"actor_rollout_ref.rollout.max_model_len={total_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={total_model_len}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "reward_model.enable=False",
        "env.env_name=bfcl",
        f"env.bfcl.root={args.bfcl_root}",
        f"env.bfcl.model_id={args.model}",
        "env.bfcl.temperature=0.001",
        "env.bfcl.max_steps_per_turn=20",
        "env.max_steps=64",
        "env.rollout.n=1",
        "trainer.logger=['console']",
        "trainer.project_name=seed_bfcl_opsd",
        f"trainer.experiment_name={args.arm}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.updates}",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={checkpoint_root}",
        f"trainer.rollout_data_dir={evidence_root / 'rollouts'}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--june24-skill-bank", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-prompt-length", type=int, default=16384)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--rlpaper-sha", default=os.environ.get("RLPAPER_SHA"))
    parser.add_argument("--same-prompt-control", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.arm = "same_prompt_control" if args.same_prompt_control else "privileged_june24"
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    args.june24_skill_bank = args.june24_skill_bank.expanduser().resolve()
    args.fixed_manifest = args.fixed_manifest.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    args.bfcl_root = args.bfcl_root.expanduser().resolve()
    if not args.rlpaper_sha:
        raise ValueError("--rlpaper-sha is required for provenance")
    if args.updates != 1:
        raise ValueError("the initial smoke contract requires exactly one optimizer update")

    fixed_value = json.loads(args.fixed_manifest.read_text(encoding="utf-8"))
    fixed_ids = validate_fixed_manifest(fixed_value)
    task_ids = _parse_task_ids(args.task_ids, fixed_ids)
    if args.batch_size != len(task_ids):
        raise ValueError(
            f"one rollout per task requires --batch-size={len(task_ids)}, got {args.batch_size}"
        )
    bank = load_skill_bank(
        args.june24_skill_bank,
        fixed_manifest=args.fixed_manifest,
        required_task_ids=task_ids,
        verify_source_files=True,
    )
    tasks, student_prompts = _official_tasks_and_prompts(
        bfcl_root=args.bfcl_root,
        model=args.model,
        task_ids=task_ids,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt_lengths = validate_context_lengths(
        tokenizer=tokenizer,
        bank=bank,
        student_prompts=student_prompts,
        max_prompt_length=args.max_prompt_length,
    )

    args.run_root.mkdir(parents=True, exist_ok=True)
    train_path = args.run_root / "inputs" / "bfcl_fixed4.parquet"
    _write_dataset(train_path, tasks)
    command = _hydra_command(args, task_ids, train_path)
    provenance = {
        "status": "preflight-only",
        "arm": args.arm,
        "seed_sha": _git_sha(repo_root),
        "rlpaper_sha": args.rlpaper_sha,
        "bfcl_sha": _git_sha(args.bfcl_root.parent),
        "model": args.model,
        "task_ids": task_ids,
        "skill_bank_path": str(args.june24_skill_bank),
        "skill_bank_sha256": bank.sha256,
        "fixed_manifest_path": str(args.fixed_manifest),
        "fixed_manifest_sha256": bank.fixed_manifest_sha256,
        "source_calls": [
            {
                "task_id": task_id,
                "path": bank.summary(task_id)["source_call_path"],
                "sha256": bank.summary(task_id)["source_call_sha256"],
            }
            for task_id in task_ids
        ],
        "dataset_path": str(train_path),
        "dataset_sha256": sha256_file(train_path),
        "prompt_lengths": prompt_lengths,
        "opd": {"only": True, "loss_coef": 1.0, "gate_beta": 5.0},
        "external_analysis_calls": False,
        "dynamic_hindsight_analysis": False,
        "same_prompt_control": args.same_prompt_control,
    }
    metadata_path = args.run_root / "metadata" / f"{args.arm}_provenance.json"
    plan_path = args.run_root / "metadata" / f"{args.arm}_plan.json"
    _write_json(metadata_path, provenance)
    _write_json(plan_path, {"command": command, "provenance": provenance})
    print(
        f"SEED_BFCL_PREFLIGHT_OK arm={args.arm} tasks={','.join(task_ids)} "
        f"skill_bank_sha256={bank.sha256}"
    )
    if args.execute:
        gpu = _a100_guard()
        _write_json(args.run_root / "metadata" / "gpu.json", gpu)
        log_path = args.run_root / "logs" / f"{args.arm}_train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(f"SEED training failed with exit code {result.returncode}; see {log_path}")
        provenance["status"] = "training-finished"
        provenance["training_log"] = str(log_path)
        _write_json(metadata_path, provenance)
        print(f"SEED_BFCL_TRAIN_DONE arm={args.arm} log={log_path}")
    return provenance


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
