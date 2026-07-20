"""Validate the June 24 all-200 five-pass run and write one completion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from seed.june24_skill_summary import (
    EXPECTED_TRAIN_OUTCOME_COUNTS,
    EXPECTED_VALIDATION_OUTCOME_COUNTS,
    KNOWN_JUNE24_REPAIRED_TASK_IDS,
    sha256_file,
    validate_all200_cohort_manifest,
    validate_stratified_split_manifest,
)


CHECKPOINT_STEPS = (40, 80, 120, 160, 200)
VALIDATION_STEPS = (0, 40, 80, 120, 160, 200)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {result}")
    return result


def _checkpoint_evidence(
    checkpoint_root: Path,
    *,
    checkpoint_updates: int = 40,
) -> list[dict[str, Any]]:
    observed_steps = sorted(
        int(path.name.rsplit("_", 1)[-1])
        for path in checkpoint_root.glob("global_step_*")
        if path.is_dir()
    )
    if checkpoint_updates <= 0 or 40 % checkpoint_updates:
        raise RuntimeError(f"invalid recovery checkpoint interval: {checkpoint_updates}")
    if not set(CHECKPOINT_STEPS).issubset(observed_steps):
        raise RuntimeError(f"missing iteration checkpoints {CHECKPOINT_STEPS}; got {observed_steps}")
    invalid_steps = [
        step
        for step in observed_steps
        if step <= 0 or step > CHECKPOINT_STEPS[-1] or step % checkpoint_updates
    ]
    if invalid_steps:
        raise RuntimeError(f"noncanonical recovery checkpoints: {invalid_steps}")
    evidence = []
    for step in observed_steps:
        root = checkpoint_root / f"global_step_{step}"
        actor = root / "actor"
        adapter = actor / "lora_adapter"
        required_patterns = {
            "model": "model_world_size_*_rank_*.pt",
            "optimizer": "optim_world_size_*_rank_*.pt",
            "extra": "extra_state_world_size_*_rank_*.pt",
        }
        missing = [label for label, pattern in required_patterns.items() if not list(actor.glob(pattern))]
        if missing or not (root / "data.pt").is_file() or not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"checkpoint {step} is not resumable or exportable; missing={missing}")
        evidence.append(
            {
                "global_step": step,
                "path": str(root),
                "adapter_sha256": _tree_sha256(adapter),
                "resumable": True,
            }
        )
    adapter_hashes = [item["adapter_sha256"] for item in evidence]
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise RuntimeError("iteration-level LoRA checksums did not change at every checkpoint")
    return evidence


def _segment_evidence(
    path: Path,
    *,
    seed_sha_history: list[str],
    checkpoint_updates: int = 40,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing checkpoint-process segment history: {path}")
    payload = _json(path)
    if payload.get("schema_version") != "seed.bfcl.segment_history.v1":
        raise RuntimeError("checkpoint-process segment history has an invalid schema")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("checkpoint-process segment history is empty")
    if checkpoint_updates <= 0 or 40 % checkpoint_updates:
        raise RuntimeError(f"invalid recovery checkpoint interval: {checkpoint_updates}")
    allowed_steps = set(range(checkpoint_updates, CHECKPOINT_STEPS[-1] + 1, checkpoint_updates))

    covered = set()
    failed_attempt_count = 0
    for item in segments:
        kind = item.get("kind")
        after = int(item.get("checkpoint_after", -1))
        before = int(item.get("checkpoint_before", -1))
        if item.get("seed_sha") not in seed_sha_history:
            raise RuntimeError(f"unrecorded SEED SHA in segment history: {item.get('seed_sha')}")
        if kind == "training_segment":
            expected = int(item.get("expected_checkpoint", -1))
            if int(item.get("returncode", -1)) != 0:
                failed_attempt_count += 1
                if after not in ({0} | allowed_steps):
                    raise RuntimeError(f"failed segment left a noncanonical checkpoint: {item}")
                continue
            delta = after - before
            if expected != after or delta not in {checkpoint_updates, 40}:
                raise RuntimeError(
                    "checkpoint-process segment did not advance by the configured recovery "
                    f"interval or the legacy 40-step interval: {item}"
                )
        elif kind == "adopted_existing_checkpoint":
            if before != after:
                raise RuntimeError(f"adopted checkpoint history is inconsistent: {item}")
        else:
            raise RuntimeError(f"unknown checkpoint-process segment kind: {kind}")
        if after not in allowed_steps:
            raise RuntimeError(f"noncanonical checkpoint in segment history: {after}")
        covered.add(after)

    if not set(CHECKPOINT_STEPS).issubset(covered):
        raise RuntimeError(
            f"segment history does not cover every checkpoint: expected={CHECKPOINT_STEPS}, got={sorted(covered)}"
        )
    return {
        **payload,
        "validation": {
            "successful_checkpoint_coverage": sorted(covered),
            "iteration_checkpoint_coverage": list(CHECKPOINT_STEPS),
            "checkpoint_updates": checkpoint_updates,
            "failed_attempt_count": failed_attempt_count,
        },
    }


def _update_evidence(
    root: Path,
    *,
    schedules: list[dict[str, Any]],
    classification_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    paths = sorted(root.glob("step_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"expected 200 optimizer diagnostic files, found {len(paths)}")
    updates = [_json(path) for path in paths]
    if [int(value.get("global_step", -1)) for value in updates] != list(range(1, 201)):
        raise RuntimeError("optimizer diagnostics do not cover ordered steps 1 through 200")
    expected_batches = [batch for schedule in schedules for batch in schedule["batches"]]
    if len(expected_batches) != 200:
        raise RuntimeError("split manifest does not define exactly 200 four-task batches")
    for index, value in enumerate(updates, start=1):
        if value.get("optimizer") != "Adam":
            raise RuntimeError(f"step {index} did not use strict Adam")
        if value.get("generated_token_only_mask") is not True:
            raise RuntimeError(f"step {index} did not use generated-token-only masking")
        if value.get("teacher_student_token_alignment") is not True:
            raise RuntimeError(f"step {index} lost privileged response-token alignment")
        if value.get("control_teacher_student_token_alignment") is not True:
            raise RuntimeError(f"step {index} lost ordinary-prompt response-token alignment")
        if int(value.get("active_token_count") or 0) <= 0:
            raise RuntimeError(f"step {index} has no active OPD tokens")
        if int(value.get("optimizer_steps_in_batch") or 0) != 1:
            raise RuntimeError(f"step {index} did not perform exactly one accumulated Adam update")
        if int(value.get("iteration") or 0) != ((index - 1) // 40) + 1:
            raise RuntimeError(f"step {index} has an invalid iteration coordinate")
        if int(value.get("batch_in_iteration") or 0) != ((index - 1) % 40) + 1:
            raise RuntimeError(f"step {index} has an invalid within-iteration batch coordinate")
        if value.get("task_ids") != expected_batches[index - 1]:
            raise RuntimeError(f"step {index} task ids differ from the frozen schedule")
        expected_classes = [classification_by_id[task_id] for task_id in expected_batches[index - 1]]
        if value.get("outcome_classes") != expected_classes:
            raise RuntimeError(f"step {index} outcome classes differ from the cohort authority")
        active_by_class = value.get("active_tokens_by_outcome_class") or {}
        class_metrics = value.get("outcome_class_metrics") or {}
        for classification in set(expected_classes):
            if int(active_by_class.get(classification) or 0) <= 0:
                raise RuntimeError(f"step {index} class {classification} has no active OPD tokens")
            selected_class_metrics = class_metrics.get(classification) or {}
            if set(selected_class_metrics) != {
                "opd_loss",
                "gate_mean",
                "teacher_gap_mean",
                "control_opd_loss",
                "control_gate_mean",
                "control_teacher_gap_mean",
            }:
                raise RuntimeError(f"step {index} class {classification} diagnostics are incomplete")
            for metric_name, metric_value in selected_class_metrics.items():
                _require_finite(metric_value, f"step {index} {classification} {metric_name}")
        for hash_key in ("response_token_sha256", "generated_token_mask_sha256"):
            if len(str(value.get(hash_key) or "")) != 64:
                raise RuntimeError(f"step {index} has invalid {hash_key}")
        metrics = value.get("metrics") or {}
        for key in (
            "actor/opd_loss",
            "actor/opd_gate_mean",
            "actor/opd_teacher_gap_mean",
            "actor/control_opd_loss",
            "actor/control_opd_gate_mean",
            "actor/control_opd_teacher_gap_mean",
            "actor/grad_norm",
        ):
            _require_finite(metrics.get(key), f"step {index} {key}")
        comparison = value.get("privileged_minus_control") or {}
        if set(comparison) != {"teacher_gap_mean", "gate_mean", "opd_loss"}:
            raise RuntimeError(f"step {index} lacks privileged-versus-control diagnostics")
        for key, metric_value in comparison.items():
            _require_finite(metric_value, f"step {index} privileged-minus-control {key}")
        if _require_finite(metrics.get("actor/rl_gradient_contribution"), "RL contribution") != 0.0:
            raise RuntimeError(f"step {index} has an RL gradient contribution")
        if index > 1 and updates[index - 2]["trainable_sha256_after"] != value["trainable_sha256_before"]:
            raise RuntimeError(f"step {index} did not start from the preceding updated actor")
    expected_lr = {1: 0.0, 67: 1e-7, 200: 1e-6}
    for step, expected in expected_lr.items():
        observed = _require_finite(updates[step - 1].get("learning_rate_used"), f"step {step} LR")
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"step {step} LR mismatch: expected={expected}, observed={observed}")
    observed_lrs = [_require_finite(value.get("learning_rate_used"), "learning rate") for value in updates]
    if any(left > right for left, right in zip(observed_lrs, observed_lrs[1:])):
        raise RuntimeError("two-stage learning-rate schedule is not monotonic")
    return updates


def _validation_evidence(
    root: Path,
    *,
    validation_ids: list[str],
    classification_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    curve = []
    for step in VALIDATION_STEPS:
        path = root / f"global_step_{step:06d}.json"
        if not path.is_file():
            raise RuntimeError(f"missing held-out validation evidence for step {step}: {path}")
        value = _json(path)
        records = value.get("records") or []
        task_ids = [str(record.get("task_id") or "") for record in records]
        if len(records) != 40 or set(task_ids) != set(validation_ids) or len(set(task_ids)) != 40:
            raise RuntimeError(f"step {step} validation does not exactly cover the held-out 40")
        if value.get("prompt_mode") != "ordinary_bfcl" or value.get("privileged_context") is not False:
            raise RuntimeError(f"step {step} validation used privileged context")
        by_class = {}
        for classification in EXPECTED_VALIDATION_OUTCOME_COUNTS:
            selected = [record for record in records if classification_by_id[record["task_id"]] == classification]
            expected_count = EXPECTED_VALIDATION_OUTCOME_COUNTS[classification]
            if len(selected) != expected_count:
                raise RuntimeError(f"step {step} validation class {classification} count mismatch")
            success_count = sum(bool(record.get("success")) for record in selected)
            by_class[classification] = {
                "task_count": len(selected),
                "success_count": success_count,
                "accuracy": success_count / len(selected),
            }
        success_count = sum(bool(record.get("success")) for record in records)
        curve.append(
            {
                "global_step": step,
                "task_count": 40,
                "success_count": success_count,
                "accuracy": success_count / 40,
                "by_class": by_class,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    initial = curve[0]
    for point in curve:
        point["accuracy_change_from_initial"] = point["accuracy"] - initial["accuracy"]
        for classification in EXPECTED_VALIDATION_OUTCOME_COUNTS:
            point["by_class"][classification]["accuracy_change_from_initial"] = (
                point["by_class"][classification]["accuracy"]
                - initial["by_class"][classification]["accuracy"]
            )
    return curve


def _privileged_control_comparison(updates: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("teacher_gap_mean", "gate_mean", "opd_loss")
    aggregates = {}
    for metric_name in metric_names:
        values = [
            _require_finite(
                (update.get("privileged_minus_control") or {}).get(metric_name),
                f"privileged-minus-control {metric_name}",
            )
            for update in updates
        ]
        aggregates[metric_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "positive_update_count": sum(value > 0 for value in values),
            "negative_update_count": sum(value < 0 for value in values),
            "zero_update_count": sum(value == 0 for value in values),
        }
    return {
        "scope": "same_sampled_tokens_ordinary_prompt_diagnostic_control",
        "update_count": len(updates),
        "metrics": aggregates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--export-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    provenance_path = run_root / "metadata" / "privileged_june24_provenance.json"
    provenance = _json(provenance_path)
    if provenance.get("mode") != "june24_all200_train160_val40":
        raise RuntimeError("provenance is not the June 24 all-200 five-pass run")
    if provenance.get("status") != "training-finished":
        raise RuntimeError("training has not finished")
    if (
        len(provenance.get("train_task_ids") or []) != 160
        or len(provenance.get("validation_task_ids") or []) != 40
        or int(provenance.get("training_rollouts") or 0) != 800
        or int(provenance.get("validation_rollouts") or 0) != 240
        or int(provenance.get("total_updates") or 0) != 200
    ):
        raise RuntimeError("provenance does not describe the frozen 160/40 five-pass schedule")
    if provenance.get("optimizer") != {
        "name": "adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }:
        raise RuntimeError("provenance optimizer is not strict Adam")

    cohort_path = Path(provenance["cohort_manifest_path"])
    split_path = Path(provenance["split_manifest_path"])
    if sha256_file(cohort_path) != provenance["cohort_manifest_sha256"]:
        raise RuntimeError("cohort manifest hash drifted")
    if sha256_file(split_path) != provenance["split_manifest_sha256"]:
        raise RuntimeError("split manifest hash drifted")
    cohort_value = _json(cohort_path)
    split_value = _json(split_path)
    cohort_records = validate_all200_cohort_manifest(cohort_value)
    train_ids, validation_ids, schedules = validate_stratified_split_manifest(
        split_value,
        cohort_manifest=cohort_path,
    )
    if train_ids != provenance["train_task_ids"] or validation_ids != provenance["validation_task_ids"]:
        raise RuntimeError("runtime task ids differ from the frozen split")
    classification_by_id = {record["task_id"]: record["classification"] for record in cohort_records}
    train_counts = {
        name: sum(classification_by_id[task_id] == name for task_id in train_ids)
        for name in EXPECTED_TRAIN_OUTCOME_COUNTS
    }
    if train_counts != EXPECTED_TRAIN_OUTCOME_COUNTS:
        raise RuntimeError("training classification counts drifted")

    source_calls = provenance.get("source_calls") or []
    if len(source_calls) != 200:
        raise RuntimeError("provenance does not contain all 200 source-call hashes")
    for source in source_calls:
        path = Path(source["source_call_path"])
        if not path.is_file() or sha256_file(path) != source["source_call_sha256"]:
            raise RuntimeError(f"source-call provenance drifted for {source.get('task_id')}")
    overrides = provenance.get("repaired_source_overrides") or []
    override_ids = [str(item.get("task_id") or "") for item in overrides]
    if (
        provenance.get("repaired_source_policy") != "explicit_historical_task_allowlist"
        or override_ids != list(KNOWN_JUNE24_REPAIRED_TASK_IDS)
    ):
        raise RuntimeError("historical repaired-source exceptions differ from the authorized three-task allowlist")
    flagged_ids = [
        str(item.get("task_id") or "")
        for item in source_calls
        if item.get("repaired_source_override") is True
    ]
    if flagged_ids != override_ids:
        raise RuntimeError("source-call repair flags differ from the authorized override evidence")

    updates = _update_evidence(
        run_root / "evidence" / "privileged_june24" / "updates",
        schedules=schedules,
        classification_by_id=classification_by_id,
    )
    checkpoint_updates = int(provenance.get("checkpoint_updates") or 40)
    checkpoints = _checkpoint_evidence(
        run_root / "checkpoints" / "privileged_june24",
        checkpoint_updates=checkpoint_updates,
    )
    segment_history = _segment_evidence(
        run_root / "metadata" / "privileged_june24_segment_history.json",
        seed_sha_history=[str(value) for value in provenance.get("seed_sha_history") or [provenance["seed_sha"]]],
        checkpoint_updates=checkpoint_updates,
    )
    for checkpoint in checkpoints[:-1]:
        next_step = checkpoint["global_step"] + 1
        if updates[next_step - 1]["trainable_sha256_before"] != updates[checkpoint["global_step"] - 1]["trainable_sha256_after"]:
            raise RuntimeError(f"iteration after checkpoint {checkpoint['global_step']} did not use its updated actor")
    validation_curve = _validation_evidence(
        run_root / "evidence" / "privileged_june24" / "validation",
        validation_ids=validation_ids,
        classification_by_id=classification_by_id,
    )
    curve_path = run_root / "evidence" / "privileged_june24" / "validation_curve.json"
    curve_path.write_text(json.dumps(validation_curve, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    export = _json(args.export_evidence.expanduser().resolve())
    if not export.get("checkpoint_resumable") or not (export.get("vllm_reload") or {}).get("ok"):
        raise RuntimeError("final merged export is not resumable and vLLM-reloadable")
    if not str(export.get("checkpoint") or "").endswith("global_step_200"):
        raise RuntimeError("final export did not use checkpoint 200")

    report = {
        "status": "complete",
        "run_root": str(run_root),
        "evidence_scope": "inline_same_token_diagnostics_not_separate_trained_control",
        "provenance": provenance,
        "source_call_count": len(source_calls),
        "checkpoints": checkpoints,
        "checkpoint_process_segments": segment_history,
        "optimizer_updates": {
            "count": len(updates),
            "first": updates[0],
            "warmup_boundary": updates[66],
            "last": updates[-1],
        },
        "privileged_versus_control": _privileged_control_comparison(updates),
        "validation_curve": validation_curve,
        "official_plain_prompt_bfcl_before_after": {
            "before": validation_curve[0],
            "after": validation_curve[-1],
        },
        "validation_curve_path": str(curve_path),
        "checkpoint_and_export": export,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SEED_BFCL_FIVE_PASS_COMPLETE report={args.output}")


if __name__ == "__main__":
    main()
