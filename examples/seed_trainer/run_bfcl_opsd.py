"""Plan and launch frozen June 24 OPD-only training on official BFCL tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from examples.seed_trainer._common.bfcl_checkpoint_supervisor import (
    archive_uncheckpointed_evidence,
    checkpoint_step,
    run_checkpoint_segments,
)
from agent_system.environments.env_package.bfcl.envs import (
    BFCLWorker,
    hydrate_task,
    load_bfcl_components,
)
from seed.june24_skill_summary import (
    exact_divisor_batch_size,
    load_skill_bank,
    sha256_file,
    validate_all200_cohort_manifest,
    validate_context_lengths,
    validate_fixed_manifest,
    validate_stratified_split_manifest,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _bfcl_memory_profile(*, all200_mode: bool, batch_size: int) -> dict[str, object]:
    """Keep the batch-64 A100 update below the 40 GB memory ceiling."""

    batch64_mode = all200_mode and batch_size >= 64
    return {
        "rollout_gpu_memory_utilization": (
            0.35 if batch64_mode else (0.41 if all200_mode else 0.45)
        ),
        "actor_activation_offload": batch64_mode,
        "actor_param_offload": batch64_mode,
        "actor_optimizer_offload": batch64_mode,
    }


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    components = load_bfcl_components(bfcl_root)
    tasks: dict[str, dict[str, Any]] = {}
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
        tasks[task_id] = task
        prompts[task_id] = prompt
    return tasks, prompts


def _dataset_rows(
    *,
    tasks: Mapping[str, dict[str, Any]],
    scheduled_task_ids: Iterable[str],
    split: str,
    classification_by_id: Mapping[str, str],
    iteration: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, task_id in enumerate(scheduled_task_ids):
        task = tasks[task_id]
        rows.append(
            {
                "data_source": "bfcl",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "official_bfcl_multi_turn",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": index,
                    "task_id": task_id,
                    "split": split,
                    "outcome_class": classification_by_id.get(task_id),
                    "iteration": iteration,
                },
                # JSON encoding preserves empty nested state without asking
                # PyArrow to infer an unsupported empty struct.
                "env_kwargs": {
                    "bfcl_task_json": json.dumps(task, ensure_ascii=True, separators=(",", ":")),
                    "task_id": task_id,
                },
            }
        )
    return rows


def _write_or_validate_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = pd.read_parquet(path)
        expected_ids = [row["extra_info"]["task_id"] for row in rows]
        observed_ids = [str(value.get("task_id")) for value in existing["extra_info"].tolist()]
        if observed_ids != expected_ids:
            raise ValueError(f"existing dataset does not match frozen schedule: {path}")
        return
    pd.DataFrame(rows).to_parquet(path, index=False)


def _classification_map(cohort_value: Mapping[str, Any]) -> dict[str, str]:
    records = validate_all200_cohort_manifest(cohort_value)
    return {record["task_id"]: record["classification"] for record in records}


def _hydra_command(
    args: argparse.Namespace,
    *,
    task_ids: list[str],
    train_path: Path,
    validation_path: Path,
    validation_batch_size: int,
    total_updates: int,
    updates_per_iteration: int,
) -> list[str]:
    total_model_len = args.max_prompt_length + args.max_response_length
    all200_mode = args.cohort_manifest is not None
    checkpoint_updates = args.checkpoint_updates if all200_mode else updates_per_iteration
    memory_profile = _bfcl_memory_profile(
        all200_mode=all200_mode,
        batch_size=args.batch_size,
    )
    checkpoint_root = args.run_root / "checkpoints" / args.arm
    evidence_root = args.run_root / "evidence" / args.arm
    task_csv = ",".join(task_ids)
    command = [
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
        f"algorithm.seed.june24_task_ids='{task_csv}'",
        f"algorithm.seed.june24_same_prompt_control={str(args.same_prompt_control)}",
        f"algorithm.seed.june24_inline_same_prompt_diagnostics={str(args.inline_same_prompt_diagnostics)}",
        "algorithm.seed.june24_verify_source_files=True",
        "algorithm.seed.skill_gen.enable=False",
        f"data.train_files={train_path}",
        f"data.val_files={validation_path}",
        f"data.train_batch_size={args.batch_size}",
        f"data.val_batch_size={validation_batch_size}",
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
        "actor_rollout_ref.model.enable_activation_offload="
        f"{str(memory_profile['actor_activation_offload'])}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.optim.name={args.optimizer}",
        f"actor_rollout_ref.actor.optim.lr={args.final_lr}",
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]",
        "actor_rollout_ref.actor.optim.eps=1e-8",
        f"actor_rollout_ref.actor.optim.weight_decay={args.weight_decay}",
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
        f"actor_rollout_ref.actor.opd_diagnostics_path={evidence_root / 'updates'}",
        f"actor_rollout_ref.actor.lora_only_resume={str(all200_mode)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={total_model_len}",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload="
        f"{str(memory_profile['actor_param_offload'])}",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload="
        f"{str(memory_profile['actor_optimizer_offload'])}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization="
        f"{memory_profile['rollout_gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
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
        "trainer.balance_batch=False",
        f"trainer.save_freq={checkpoint_updates}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_updates}",
        f"trainer.resume_mode={args.resume}",
        f"trainer.restart_after_checkpoint={str(all200_mode)}",
        f"trainer.skip_val_before_train_on_resume={str(all200_mode)}",
        f"trainer.shutdown_local_ray_on_exit={str(all200_mode)}",
        f"trainer.default_local_dir={checkpoint_root}",
        f"trainer.rollout_data_dir={evidence_root / 'rollouts'}",
        f"trainer.validation_data_dir={evidence_root / 'validation'}",
    ]
    if args.fixed_manifest:
        command.extend(
            [
                f"algorithm.seed.june24_fixed_manifest={args.fixed_manifest}",
                "trainer.val_before_train=False",
                "trainer.test_freq=-1",
            ]
        )
    else:
        command.extend(
            [
                f"algorithm.seed.june24_cohort_manifest={args.cohort_manifest}",
                f"algorithm.seed.june24_split_manifest={args.split_manifest}",
                "trainer.val_before_train=True",
                f"trainer.test_freq={updates_per_iteration}",
                f"trainer.max_actor_ckpt_to_keep={total_updates // checkpoint_updates}",
            ]
        )
        if args.lr_schedule == "two_stage_linear":
            command.extend(
                [
                    f"actor_rollout_ref.actor.optim.lr_warmup_steps={args.warmup_updates}",
                    "actor_rollout_ref.actor.optim.warmup_style=two_stage_linear",
                    f"actor_rollout_ref.actor.optim.warmup_target_lr={args.warmup_target_lr}",
                    f"actor_rollout_ref.actor.optim.final_lr={args.final_lr}",
                ]
            )
        else:
            command.extend(
                [
                    "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                    "actor_rollout_ref.actor.optim.warmup_style=constant",
                ]
            )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--june24-skill-bank", type=Path, required=True)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--fixed-manifest", type=Path)
    authority.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--task-ids", default=None, help="Legacy fixed-40 smoke selection only")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1, help="Legacy fixed-40 smoke updates")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=None)
    parser.add_argument("--warmup-updates", type=int, default=67)
    parser.add_argument("--warmup-target-lr", type=float, default=1e-7)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=("constant", "two_stage_linear"), default=None)
    parser.add_argument("--learning-rate", type=float, default=None, help="Legacy fixed-40 alias for --final-lr")
    parser.add_argument("--max-prompt-length", type=int, default=16384)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        default=10,
        help="All-200 recovery checkpoint/process interval; must divide 40",
    )
    parser.add_argument("--resume", choices=("disable", "auto"), default=None)
    parser.add_argument("--rlpaper-sha", default=os.environ.get("RLPAPER_SHA"))
    parser.add_argument("--same-prompt-control", action="store_true")
    parser.add_argument("--inline-same-prompt-diagnostics", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-after-update",
        type=int,
        default=None,
        help="Execution-only gate: stop after this committed checkpoint, then resume the same run later",
    )
    args = parser.parse_args()
    args.arm = "same_prompt_control" if args.same_prompt_control else "privileged_june24"
    return args


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "june24_skill_bank",
        "fixed_manifest",
        "cohort_manifest",
        "split_manifest",
        "run_root",
        "bfcl_root",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _resolve_paths(args)
    if not args.rlpaper_sha:
        raise ValueError("--rlpaper-sha is required for provenance")

    all200_mode = args.cohort_manifest is not None
    if all200_mode:
        if args.split_manifest is None:
            raise ValueError("--cohort-manifest requires --split-manifest")
        if args.task_ids is not None or args.same_prompt_control:
            raise ValueError("all-200 mode forbids legacy --task-ids and separate same-prompt-control arms")
        split_header = _read_json(args.split_manifest)
        split_profile = str(split_header.get("split_profile") or "160x40_b4")
        profile_contracts = {
            "160x40_b4": {
                "batch_size": 4,
                "updates_per_iteration": 40,
                "total_updates": 200,
                "lr_schedule": "two_stage_linear",
            },
            "128x72_b64": {
                "batch_size": 64,
                "updates_per_iteration": 2,
                "total_updates": 10,
                "lr_schedule": "constant",
            },
        }
        profile_contract = profile_contracts.get(split_profile)
        if profile_contract is None:
            raise ValueError(f"unsupported all-200 split profile: {split_profile}")
        if args.iterations != 5 or args.batch_size != profile_contract["batch_size"]:
            raise ValueError(
                f"split profile {split_profile} requires --iterations=5 and "
                f"--batch-size={profile_contract['batch_size']}"
            )
        if args.checkpoint_updates <= 0 or profile_contract["updates_per_iteration"] % args.checkpoint_updates:
            raise ValueError(
                "--checkpoint-updates must be a positive divisor of updates per iteration"
            )
        args.lr_schedule = args.lr_schedule or profile_contract["lr_schedule"]
        if args.lr_schedule != profile_contract["lr_schedule"]:
            raise ValueError(
                f"split profile {split_profile} requires --lr-schedule={profile_contract['lr_schedule']}"
            )
        args.resume = args.resume or "auto"
        if args.resume != "auto":
            raise ValueError("the canonical all-200 segmented run requires --resume=auto")
        if args.updates != 1:
            raise ValueError("--updates is a legacy fixed-40 option and must remain 1 in all-200 mode")
        if args.learning_rate is not None:
            raise ValueError("all-200 mode uses --warmup-target-lr and --final-lr, not --learning-rate")
        args.optimizer = args.optimizer or "adam"
        args.weight_decay = 0.0
        if args.optimizer != "adam" or args.weight_decay != 0.0:
            raise ValueError("the canonical all-200 run requires strict Adam with zero weight decay")
        if not args.inline_same_prompt_diagnostics:
            raise ValueError("the canonical all-200 run requires --inline-same-prompt-diagnostics")

        cohort_value = _read_json(args.cohort_manifest)
        split_value = split_header
        classification_by_id = _classification_map(cohort_value)
        train_ids, validation_ids, schedules = validate_stratified_split_manifest(
            split_value,
            cohort_manifest=args.cohort_manifest,
        )
        schedule_value = split_value["training_schedule"]
        total_updates = int(schedule_value["total_updates"])
        updates_per_iteration = int(schedule_value["updates_per_iteration"])
        if (
            total_updates != profile_contract["total_updates"]
            or updates_per_iteration != profile_contract["updates_per_iteration"]
        ):
            raise ValueError(f"training schedule does not match split profile {split_profile}")
        bank = load_skill_bank(
            args.june24_skill_bank,
            cohort_manifest=args.cohort_manifest,
            split_manifest=args.split_manifest,
            required_task_ids=train_ids,
            verify_source_files=True,
        )
        unique_task_ids = train_ids + validation_ids
        tasks, student_prompts = _official_tasks_and_prompts(
            bfcl_root=args.bfcl_root,
            model=args.model,
            task_ids=unique_task_ids,
        )
        scheduled_train_ids = [
            task_id
            for schedule in schedules
            for task_id in schedule["task_ids"]
        ]
        train_rows = []
        offset = 0
        for schedule in schedules:
            iteration_rows = _dataset_rows(
                tasks=tasks,
                scheduled_task_ids=schedule["task_ids"],
                split="train",
                classification_by_id=classification_by_id,
                iteration=int(schedule["iteration"]),
            )
            for row_index, row in enumerate(iteration_rows, start=offset):
                row["extra_info"]["index"] = row_index
            offset += len(iteration_rows)
            train_rows.extend(iteration_rows)
        validation_rows = _dataset_rows(
            tasks=tasks,
            scheduled_task_ids=validation_ids,
            split="validation",
            classification_by_id=classification_by_id,
        )
        prompt_task_ids = train_ids
        run_mode = (
            "june24_all200_train128_val72_b64"
            if split_profile == "128x72_b64"
            else "june24_all200_train160_val40"
        )
        manifest_provenance = {
            "cohort_manifest_path": str(args.cohort_manifest),
            "cohort_manifest_sha256": sha256_file(args.cohort_manifest),
            "split_manifest_path": str(args.split_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
        }
    else:
        if args.split_manifest is not None:
            raise ValueError("--split-manifest is valid only with --cohort-manifest")
        if args.updates != 1:
            raise ValueError("the fixed-40 smoke contract requires exactly one optimizer update")
        args.optimizer = args.optimizer or "adamw"
        args.weight_decay = 0.01
        args.resume = args.resume or "disable"
        if args.learning_rate is not None:
            args.final_lr = args.learning_rate
        fixed_value = _read_json(args.fixed_manifest)
        fixed_ids = validate_fixed_manifest(fixed_value)
        train_ids = _parse_task_ids(args.task_ids, fixed_ids)
        validation_ids = list(train_ids)
        if args.batch_size != len(train_ids):
            raise ValueError(
                f"one rollout per task requires --batch-size={len(train_ids)}, got {args.batch_size}"
            )
        bank = load_skill_bank(
            args.june24_skill_bank,
            fixed_manifest=args.fixed_manifest,
            required_task_ids=train_ids,
            verify_source_files=True,
        )
        tasks, student_prompts = _official_tasks_and_prompts(
            bfcl_root=args.bfcl_root,
            model=args.model,
            task_ids=train_ids,
        )
        classification_by_id = {task_id: "fixed" for task_id in train_ids}
        train_rows = _dataset_rows(
            tasks=tasks,
            scheduled_task_ids=train_ids,
            split="train",
            classification_by_id=classification_by_id,
            iteration=1,
        )
        validation_rows = list(train_rows)
        scheduled_train_ids = list(train_ids)
        total_updates = 1
        updates_per_iteration = 1
        prompt_task_ids = train_ids
        split_profile = None
        run_mode = "june24_fixed40_smoke"
        manifest_provenance = {
            "fixed_manifest_path": str(args.fixed_manifest),
            "fixed_manifest_sha256": bank.fixed_manifest_sha256,
        }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt_lengths = validate_context_lengths(
        tokenizer=tokenizer,
        bank=bank,
        student_prompts={task_id: student_prompts[task_id] for task_id in prompt_task_ids},
        max_prompt_length=args.max_prompt_length,
    )

    args.run_root.mkdir(parents=True, exist_ok=True)
    train_path = args.run_root / "inputs" / (
        f"bfcl_train{len(train_ids)}_x{args.iterations}.parquet" if all200_mode else "bfcl_fixed4.parquet"
    )
    validation_path = args.run_root / "inputs" / (
        f"bfcl_validation{len(validation_ids)}.parquet" if all200_mode else "bfcl_fixed4.parquet"
    )
    _write_or_validate_dataset(train_path, train_rows)
    if validation_path != train_path:
        _write_or_validate_dataset(validation_path, validation_rows)
    validation_batch_size = exact_divisor_batch_size(
        requested_batch_size=args.batch_size,
        task_count=len(validation_ids),
    )
    command = _hydra_command(
        args,
        task_ids=train_ids,
        train_path=train_path,
        validation_path=validation_path,
        validation_batch_size=validation_batch_size,
        total_updates=total_updates,
        updates_per_iteration=updates_per_iteration,
    )
    memory_profile = _bfcl_memory_profile(
        all200_mode=all200_mode,
        batch_size=args.batch_size,
    )
    bank_value = _read_json(args.june24_skill_bank)
    source_calls = bank_value.get("source_calls") or [
        {
            "task_id": task_id,
            "path": bank.summary(task_id)["source_call_path"],
            "sha256": bank.summary(task_id)["source_call_sha256"],
        }
        for task_id in train_ids
    ]
    seed_sha = _git_sha(repo_root)
    provenance = {
        "status": "preflight-only",
        "mode": run_mode,
        "split_profile": split_profile,
        "arm": args.arm,
        "seed_sha": seed_sha,
        "rlpaper_sha": args.rlpaper_sha,
        "bfcl_sha": _git_sha(args.bfcl_root.parent),
        "model": args.model,
        "task_ids": train_ids,
        "train_task_ids": train_ids,
        "validation_task_ids": validation_ids,
        "scheduled_train_task_ids": scheduled_train_ids,
        "iterations": args.iterations if all200_mode else 1,
        "batch_size": args.batch_size,
        "validation_batch_size": validation_batch_size,
        "updates_per_iteration": updates_per_iteration,
        "total_updates": total_updates,
        "training_rollouts": len(train_rows),
        "validation_rollouts": (args.iterations + 1) * len(validation_ids) if all200_mode else 0,
        "skill_bank_path": str(args.june24_skill_bank),
        "skill_bank_sha256": bank.sha256,
        **manifest_provenance,
        "source_calls": source_calls,
        "repaired_source_policy": bank_value.get("repaired_source_policy"),
        "repaired_source_overrides": bank_value.get("repaired_source_overrides") or [],
        "train_dataset_path": str(train_path),
        "train_dataset_sha256": sha256_file(train_path),
        "dataset_path": str(train_path),
        "dataset_sha256": sha256_file(train_path),
        "validation_dataset_path": str(validation_path),
        "validation_dataset_sha256": sha256_file(validation_path),
        "prompt_lengths": prompt_lengths,
        "optimizer": {
            "name": args.optimizer,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
        },
        "learning_rate": {
            "style": args.lr_schedule if all200_mode else "constant",
            "warmup_updates": args.warmup_updates if all200_mode and args.lr_schedule == "two_stage_linear" else 0,
            "warmup_target_lr": args.warmup_target_lr if all200_mode and args.lr_schedule == "two_stage_linear" else None,
            "final_lr": args.final_lr,
        },
        "checkpoints": [
            updates_per_iteration * iteration
            for iteration in range(1, (args.iterations if all200_mode else 1) + 1)
        ],
        "checkpoint_updates": args.checkpoint_updates if all200_mode else updates_per_iteration,
        "checkpoint_schedule": (
            list(range(args.checkpoint_updates, total_updates + 1, args.checkpoint_updates))
            if all200_mode
            else [total_updates]
        ),
        "validation_steps": (
            [0, *[updates_per_iteration * index for index in range(1, args.iterations + 1)]]
            if all200_mode
            else []
        ),
        "opd": {"only": True, "loss_coef": 1.0, "gate_beta": 5.0},
        "inline_same_prompt_diagnostics": args.inline_same_prompt_diagnostics,
        "external_analysis_calls": False,
        "dynamic_hindsight_analysis": False,
        "same_prompt_control": args.same_prompt_control,
        "resume_mode": args.resume,
        "checkpoint_process_mode": (
            "fresh_process_per_recovery_checkpoint" if all200_mode else "single_process"
        ),
        "memory_profile": memory_profile,
        "rollout_gpu_memory_utilization": memory_profile["rollout_gpu_memory_utilization"],
        "execution_stop_after_update": args.stop_after_update,
    }
    metadata_path = args.run_root / "metadata" / f"{args.arm}_provenance.json"
    plan_path = args.run_root / "metadata" / f"{args.arm}_plan.json"
    previous_provenance = _read_json(metadata_path) if metadata_path.is_file() else {}
    previous_seed_history = previous_provenance.get("seed_sha_history") or [
        previous_provenance.get("seed_sha")
    ]
    seed_sha_history = []
    for value in [*previous_seed_history, seed_sha]:
        if value and value not in seed_sha_history:
            seed_sha_history.append(value)
    provenance["seed_sha_history"] = seed_sha_history
    provenance["uncheckpointed_evidence_archives"] = list(
        previous_provenance.get("uncheckpointed_evidence_archives") or []
    )
    if plan_path.is_file():
        previous_plan = _read_json(plan_path)
        if previous_plan.get("command") != command:
            preserved_plan = (
                args.run_root / "metadata" / f"{args.arm}_plan_before_{seed_sha[:12]}.json"
            )
            if not preserved_plan.exists():
                _write_json(preserved_plan, previous_plan)
    _write_json(metadata_path, provenance)
    _write_json(plan_path, {"command": command, "provenance": provenance})
    print(
        f"SEED_BFCL_PREFLIGHT_OK mode={provenance['mode']} arm={args.arm} "
        f"train={len(train_ids)} validation={len(validation_ids)} updates={total_updates} "
        f"skill_bank_sha256={bank.sha256}"
    )
    if args.execute:
        gpu = _a100_guard()
        _write_json(args.run_root / "metadata" / "gpu.json", gpu)
        log_path = args.run_root / "logs" / f"{args.arm}_train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "a" if args.resume == "auto" and log_path.is_file() else "w"
        if log_mode == "w":
            log_path.write_text("", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nSEED_BFCL_LAUNCH resume={args.resume} total_updates={total_updates}\n")
        if all200_mode:
            checkpoint_root = args.run_root / "checkpoints" / args.arm
            resumable_step = checkpoint_step(checkpoint_root)
            archive_record = archive_uncheckpointed_evidence(
                evidence_root=args.run_root / "evidence" / args.arm,
                checkpoint=resumable_step,
            )
            if archive_record is not None:
                archives = list(provenance.get("uncheckpointed_evidence_archives") or [])
                archives.append(archive_record)
                provenance["uncheckpointed_evidence_archives"] = archives
                _write_json(metadata_path, provenance)
                print(
                    "SEED_BFCL_UNCHECKPOINTED_EVIDENCE_ARCHIVED "
                    f"checkpoint={resumable_step} archive={archive_record['archive_root']} "
                    f"files={len(archive_record['moved'])}",
                    flush=True,
                )
            execution_end_update = args.stop_after_update or total_updates
            if (
                execution_end_update <= 0
                or execution_end_update > total_updates
                or execution_end_update % args.checkpoint_updates
            ):
                raise ValueError("--stop-after-update must be a committed checkpoint within the run")
            run_checkpoint_segments(
                command=command,
                cwd=repo_root,
                log_path=log_path,
                checkpoint_root=checkpoint_root,
                history_path=args.run_root / "metadata" / f"{args.arm}_segment_history.json",
                total_updates=execution_end_update,
                updates_per_segment=args.checkpoint_updates,
                seed_sha=seed_sha,
                existing_checkpoint_seed_sha=provenance["seed_sha_history"][0],
            )
        else:
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode != 0:
                raise RuntimeError(f"SEED training failed with exit code {result.returncode}; see {log_path}")
        completed_update = checkpoint_step(args.run_root / "checkpoints" / args.arm) if all200_mode else total_updates
        provenance["status"] = "training-finished" if completed_update == total_updates else "drive-backed-smoke"
        provenance["completed_update"] = completed_update
        provenance["training_log"] = str(log_path)
        _write_json(metadata_path, provenance)
        print(f"SEED_BFCL_TRAIN_DONE arm={args.arm} log={log_path}")
    return provenance


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
