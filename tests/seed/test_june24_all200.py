import ast
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from seed.june24_skill_summary import (
    EXPECTED_OUTCOME_COUNTS,
    EXPECTED_TRAIN_OUTCOME_COUNTS,
    EXPECTED_TRAIN_128_OUTCOME_COUNTS,
    EXPECTED_VALIDATION_OUTCOME_COUNTS,
    EXPECTED_VALIDATION_72_OUTCOME_COUNTS,
    KNOWN_JUNE24_REPAIRED_TASK_IDS,
    LARGE_BATCH_SPLIT_PROFILE,
    build_all200_cohort_manifest,
    build_stratified_split_manifest,
    exact_divisor_batch_size,
    load_skill_bank,
    materialize_all200_training_skill_bank,
    validate_all200_opd_update_task_count,
    validate_stratified_split_manifest,
)


def _load_launcher_function(function_name):
    launcher = Path(__file__).resolve().parents[2] / "examples/seed_trainer/run_bfcl_opsd.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), launcher, "exec"), namespace)
    return namespace[function_name]


def test_bfcl_batch64_uses_the_a100_memory_safety_profile():
    profile = _load_launcher_function("_bfcl_memory_profile")

    assert profile(all200_mode=True, batch_size=64) == {
        "rollout_gpu_memory_utilization": 0.41,
        "actor_activation_offload": True,
        "actor_param_offload": True,
        "actor_optimizer_offload": True,
    }
    assert profile(all200_mode=True, batch_size=4) == {
        "rollout_gpu_memory_utilization": 0.41,
        "actor_activation_offload": False,
        "actor_param_offload": False,
        "actor_optimizer_offload": False,
    }
    assert (
        profile(all200_mode=False, batch_size=4)["rollout_gpu_memory_utilization"]
        == 0.45
    )


def test_checkpoint20_extension_is_accepted_only_for_all200_mode(tmp_path):
    launcher = Path(__file__).resolve().parents[2] / "examples/seed_trainer/run_bfcl_opsd.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    function_names = {"_bfcl_memory_profile", "_hydra_command"}
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {"argparse": argparse, "Path": Path, "sys": sys}
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), launcher, "exec"), namespace)
    hydra_command = namespace["_hydra_command"]

    args = SimpleNamespace(
        max_prompt_length=1024,
        max_response_length=256,
        cohort_manifest=tmp_path / "cohort.json",
        checkpoint_updates=1,
        batch_size=64,
        run_root=tmp_path / "run",
        arm="privileged_june24",
        june24_skill_bank=tmp_path / "bank.json",
        same_prompt_control=False,
        inline_same_prompt_diagnostics=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=16,
        lora_alpha=32,
        optimizer="adam",
        final_lr=1e-6,
        weight_decay=0.0,
        bfcl_root=tmp_path / "bfcl",
        resume="auto",
        fixed_manifest=None,
        split_manifest=tmp_path / "split.json",
        lr_schedule="constant",
        warmup_updates=0,
        warmup_target_lr=None,
        extend_from_update=10,
    )
    command = hydra_command(
        args,
        task_ids=["multi_turn_base_0"],
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "validation.parquet",
        validation_batch_size=8,
        total_updates=20,
        updates_per_iteration=2,
    )
    assert f"algorithm.seed.june24_cohort_manifest={args.cohort_manifest}" in command
    assert "trainer.total_training_steps=20" in command

    args.fixed_manifest = tmp_path / "fixed.json"
    args.cohort_manifest = None
    with pytest.raises(ValueError, match="all-200 batch-64 continuation"):
        hydra_command(
            args,
            task_ids=["multi_turn_base_0"],
            train_path=tmp_path / "train.parquet",
            validation_path=tmp_path / "train.parquet",
            validation_batch_size=1,
            total_updates=1,
            updates_per_iteration=1,
        )


def test_checkpoint20_extension_reuses_checkpoint10_snapshots_after_preflight(tmp_path):
    launcher = Path(__file__).resolve().parents[2] / "examples/seed_trainer/run_bfcl_opsd.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    function_names = {
        "_read_json",
        "_write_json",
        "_write_json_immutable",
        "_validate_extension_parent",
    }
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]

    schedules = [
        {"iteration": iteration, "task_ids": ["multi_turn_base_1"]}
        for iteration in range(1, 11)
    ]

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    namespace = {
        "Any": object,
        "Mapping": dict,
        "Path": Path,
        "argparse": argparse,
        "json": json,
        "sha256_file": sha256_file,
        "checkpoint_step": lambda _path: 11,
        "validate_stratified_split_manifest": (
            lambda _payload, cohort_manifest: ([], [], schedules[:5])
        ),
    }
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), launcher, "exec"), namespace)

    run_root = tmp_path / "run"
    metadata_root = run_root / "metadata"
    evidence_root = run_root / "evidence" / "privileged_june24"
    metadata_root.mkdir(parents=True)
    (evidence_root / "updates").mkdir(parents=True)
    (evidence_root / "validation").mkdir(parents=True)
    parent_split = tmp_path / "parent_split.json"
    parent_cohort = tmp_path / "parent_cohort.json"
    parent_split.write_text("{}\n", encoding="utf-8")
    parent_cohort.write_text("{}\n", encoding="utf-8")
    (evidence_root / "updates" / "step_000010.json").write_text("{}\n", encoding="utf-8")
    (evidence_root / "validation" / "global_step_000010.json").write_text(
        "{}\n", encoding="utf-8"
    )

    train_ids = ["multi_turn_base_1"]
    validation_ids = ["multi_turn_base_2"]
    parent = {
        "mode": "june24_all200_train128_val72_b64",
        "split_profile": "128x72_b64",
        "iterations": 5,
        "batch_size": 64,
        "updates_per_iteration": 2,
        "total_updates": 10,
        "completed_update": 10,
        "status": "training-finished",
        "train_task_ids": train_ids,
        "validation_task_ids": validation_ids,
        "optimizer": {
            "name": "adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "learning_rate": {
            "style": "constant",
            "warmup_updates": 0,
            "warmup_target_lr": None,
            "final_lr": 1e-6,
        },
        "opd": {"only": True, "loss_coef": 1.0, "gate_beta": 5.0},
        "split_manifest_path": str(parent_split),
        "cohort_manifest_path": str(parent_cohort),
        "seed_sha": "parent-sha",
        "seed_sha_history": ["parent-sha"],
        "train_dataset_path": str(tmp_path / "train.parquet"),
        "train_dataset_sha256": "train-sha",
    }
    parent_plan = {"command": ["original"]}
    (metadata_root / "privileged_june24_provenance_at_checkpoint10.json").write_text(
        json.dumps(parent), encoding="utf-8"
    )
    (metadata_root / "privileged_june24_plan_at_checkpoint10.json").write_text(
        json.dumps(parent_plan), encoding="utf-8"
    )
    # A prior extension preflight has already replaced the mutable live metadata.
    (metadata_root / "privileged_june24_provenance.json").write_text(
        json.dumps({"status": "preflight-only", "iterations": 10, "total_updates": 20}),
        encoding="utf-8",
    )
    (metadata_root / "privileged_june24_plan.json").write_text(
        json.dumps({"command": ["extension"]}), encoding="utf-8"
    )

    args = SimpleNamespace(
        extend_from_update=10,
        iterations=10,
        run_root=run_root,
        arm="privileged_june24",
    )
    snapshot = namespace["_validate_extension_parent"](
        args=args,
        split_profile="128x72_b64",
        train_ids=train_ids,
        validation_ids=validation_ids,
        schedules=schedules,
        total_updates=20,
    )

    assert snapshot["extension_from_update"] == 10
    assert snapshot["target_update"] == 20
    assert json.loads(
        (metadata_root / "privileged_june24_provenance_at_checkpoint10.json").read_text(
            encoding="utf-8"
        )
    ) == parent


def test_validation_batch_size_exactly_partitions_72_without_changing_train_batch():
    assert exact_divisor_batch_size(requested_batch_size=64, task_count=72) == 8
    assert exact_divisor_batch_size(requested_batch_size=4, task_count=40) == 4


@pytest.mark.parametrize(
    ("requested_batch_size", "validation_task_count"),
    [(0, 72), (64, 0), (-1, 72), (64, -1)],
)
def test_validation_batch_size_rejects_nonpositive_inputs(
    requested_batch_size,
    validation_task_count,
):
    with pytest.raises(ValueError, match="must be positive"):
        exact_divisor_batch_size(
            requested_batch_size=requested_batch_size,
            task_count=validation_task_count,
        )


@pytest.mark.parametrize("batch_size", [4, 64])
def test_all200_opd_update_accepts_configured_task_trajectory_count(batch_size):
    validate_all200_opd_update_task_count(
        [f"multi_turn_base_{index}" for index in range(batch_size)],
        configured_batch_size=batch_size,
        split_manifest_path="split.json",
    )


def test_all200_opd_update_rejects_legacy_four_task_assumption_for_batch64():
    with pytest.raises(RuntimeError, match="exactly 64 task trajectories, got 4"):
        validate_all200_opd_update_task_count(
            [f"multi_turn_base_{index}" for index in range(4)],
            configured_batch_size=64,
            split_manifest_path="split.json",
        )


def test_fixed40_path_does_not_apply_all200_batch_count_contract():
    validate_all200_opd_update_task_count(
        ["multi_turn_base_0"],
        configured_batch_size=4,
        split_manifest_path=None,
    )


def _classification(index: int) -> tuple[str, bool, bool]:
    if index < 40:
        return "fixed", False, True
    if index < 152:
        return "both_wrong", False, False
    if index < 165:
        return "harmed", True, False
    return "both_correct", True, True


def _all200_evidence(tmp_path: Path):
    pairwise = tmp_path / "pairwise_tasks.csv"
    with pairwise.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "classification", "baseline_valid", "skill_valid"],
        )
        writer.writeheader()
        for index in range(200):
            classification, baseline_valid, skill_valid = _classification(index)
            writer.writerow(
                {
                    "task_id": f"multi_turn_base_{index}",
                    "classification": classification,
                    "baseline_valid": baseline_valid,
                    "skill_valid": skill_valid,
                }
            )
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(
        json.dumps(build_all200_cohort_manifest(pairwise), indent=2) + "\n",
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(build_stratified_split_manifest(cohort_path), indent=2) + "\n",
        encoding="utf-8",
    )
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    for index in range(200):
        task_id = f"multi_turn_base_{index}"
        target = (source_a if index < 50 else source_b) / f"skill_summary_call_{task_id}.json"
        target.write_text(
            json.dumps(
                {
                    "parsed_skill": {
                        "success_analysis": f"success {task_id}",
                        "mistake_analysis": f"mistake {task_id}",
                        "golden_workflow": f"workflow {task_id}",
                    }
                }
            ),
            encoding="utf-8",
        )
    return cohort_path, split_path, source_a, source_b


def test_builds_exact_stratified_160_40_and_five_schedules(tmp_path):
    cohort_path, split_path, _, _ = _all200_evidence(tmp_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids, validation_ids, schedules = validate_stratified_split_manifest(
        split,
        cohort_manifest=cohort_path,
    )
    assert len(train_ids) == 160
    assert len(validation_ids) == 40
    assert set(train_ids).isdisjoint(validation_ids)
    assert split["train"]["classification_counts"] == EXPECTED_TRAIN_OUTCOME_COUNTS
    assert split["validation"]["classification_counts"] == EXPECTED_VALIDATION_OUTCOME_COUNTS
    assert split["training_schedule"]["total_updates"] == 200
    assert split["training_schedule"]["total_rollouts"] == 800
    assert len(schedules) == 5
    expected_validation = []
    class_ranges = {
        "fixed": range(0, 40),
        "both_wrong": range(40, 152),
        "harmed": range(152, 165),
        "both_correct": range(165, 200),
    }
    for classification, indices in class_ranges.items():
        ordered = sorted(
            (f"multi_turn_base_{index}" for index in indices),
            key=lambda task_id: hashlib.sha256(f"20260624:{task_id}".encode()).hexdigest(),
        )
        expected_validation.extend(
            ordered[: EXPECTED_VALIDATION_OUTCOME_COUNTS[classification]]
        )
    assert set(validation_ids) == set(expected_validation)
    for schedule in schedules:
        assert len(schedule["task_ids"]) == 160
        assert set(schedule["task_ids"]) == set(train_ids)
        assert len(schedule["batches"]) == 40
        assert all(len(batch) == 4 for batch in schedule["batches"])
        assert schedule["task_ids"] == sorted(
            train_ids,
            key=lambda task_id: hashlib.sha256(
                f"20260624:{schedule['iteration']}:{task_id}".encode()
            ).hexdigest(),
        )


def test_builds_stratified_128_72_with_two_batches_per_iteration(tmp_path):
    cohort_path, old_split_path, source_a, source_b = _all200_evidence(tmp_path)
    old_split = json.loads(old_split_path.read_text(encoding="utf-8"))
    split = build_stratified_split_manifest(
        cohort_path,
        batch_size=64,
        split_profile=LARGE_BATCH_SPLIT_PROFILE,
    )
    train_ids, validation_ids, schedules = validate_stratified_split_manifest(
        split,
        cohort_manifest=cohort_path,
    )

    assert len(train_ids) == 128
    assert len(validation_ids) == 72
    assert set(old_split["validation"]["task_ids"]).issubset(validation_ids)
    assert split["train"]["classification_counts"] == EXPECTED_TRAIN_128_OUTCOME_COUNTS
    assert split["validation"]["classification_counts"] == EXPECTED_VALIDATION_72_OUTCOME_COUNTS
    assert split["training_schedule"]["total_updates"] == 10
    assert split["training_schedule"]["total_rollouts"] == 640
    assert len(schedules) == 5
    for schedule in schedules:
        assert len(schedule["task_ids"]) == 128
        assert set(schedule["task_ids"]) == set(train_ids)
        assert [len(batch) for batch in schedule["batches"]] == [64, 64]

    split_path = tmp_path / "split_128x72.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    bank = materialize_all200_training_skill_bank(
        source_dirs=[source_a, source_b],
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        output_path=tmp_path / "train128_bank.json",
    )
    assert bank["task_count"] == 128
    assert bank["task_ids"] == train_ids
    assert bank["source_task_count"] == 200


def test_batch64_checkpoint20_extension_preserves_first_five_schedules(tmp_path):
    cohort_path, _, _, _ = _all200_evidence(tmp_path)
    original = build_stratified_split_manifest(
        cohort_path,
        batch_size=64,
        split_profile=LARGE_BATCH_SPLIT_PROFILE,
    )
    extension = build_stratified_split_manifest(
        cohort_path,
        iterations=10,
        batch_size=64,
        split_profile=LARGE_BATCH_SPLIT_PROFILE,
    )
    train_ids, validation_ids, schedules = validate_stratified_split_manifest(
        extension,
        cohort_manifest=cohort_path,
    )

    assert extension["train"] == original["train"]
    assert extension["validation"] == original["validation"]
    assert extension["training_schedule"]["iteration_schedules"][:5] == original[
        "training_schedule"
    ]["iteration_schedules"]
    assert extension["training_schedule"]["total_updates"] == 20
    assert extension["training_schedule"]["total_rollouts"] == 1280
    assert len(schedules) == 10
    assert len(train_ids) == 128
    assert len(validation_ids) == 72
    for expected_iteration, schedule in enumerate(schedules, start=1):
        assert schedule["iteration"] == expected_iteration
        assert set(schedule["task_ids"]) == set(train_ids)
        assert [len(batch) for batch in schedule["batches"]] == [64, 64]


def test_160x40_profile_rejects_ten_iterations(tmp_path):
    cohort_path, _, _, _ = _all200_evidence(tmp_path)
    with pytest.raises(ValueError, match="iterations in"):
        build_stratified_split_manifest(cohort_path, iterations=10, batch_size=4)


def test_materializes_160_summaries_but_audits_all_200_sources(tmp_path):
    cohort_path, split_path, source_a, source_b = _all200_evidence(tmp_path)
    bank_path = tmp_path / "train_bank.json"
    bank = materialize_all200_training_skill_bank(
        source_dirs=[source_a, source_b],
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        output_path=bank_path,
    )
    assert bank["task_count"] == 160
    assert bank["source_task_count"] == 200
    assert len(bank["source_calls"]) == 200
    loaded = load_skill_bank(
        bank_path,
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        required_task_ids=bank["task_ids"],
        verify_source_files=True,
    )
    assert len(loaded.records) == 160
    assert len(loaded.validation_task_ids) == 40
    with pytest.raises(ValueError, match="has no record"):
        loaded.summary(loaded.validation_task_ids[0])


def test_all200_contract_rejects_source_hash_drift_and_forbidden_fields(tmp_path):
    cohort_path, split_path, source_a, source_b = _all200_evidence(tmp_path)
    bank_path = tmp_path / "train_bank.json"
    materialize_all200_training_skill_bank(
        source_dirs=[source_a, source_b],
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        output_path=bank_path,
    )
    source = source_a / "skill_summary_call_multi_turn_base_0.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source call SHA-256 mismatch"):
        load_skill_bank(
            bank_path,
            cohort_manifest=cohort_path,
            split_manifest=split_path,
            verify_source_files=True,
        )

    value = json.loads((source_b / "skill_summary_call_multi_turn_base_199.json").read_text(encoding="utf-8"))
    value["parsed_skill"]["possible_answer"] = "forbidden"
    (source_b / "skill_summary_call_multi_turn_base_199.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden fields"):
        materialize_all200_training_skill_bank(
            source_dirs=[source_a, source_b],
            cohort_manifest=cohort_path,
            split_manifest=split_path,
            output_path=tmp_path / "rejected.json",
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"fallback": True}, "fallback"),
        ({"repair_attempts": [{"attempt": 1}]}, "repaired"),
        ({"reference_mode": "generated_ground_truth_with_possible_answer_regenerated"}, "structured/V2R"),
    ],
)
def test_all200_rejects_fallback_repair_or_structured_source_metadata(
    tmp_path,
    metadata,
    message,
):
    cohort_path, split_path, source_a, source_b = _all200_evidence(tmp_path)
    path = source_a / "skill_summary_call_multi_turn_base_0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(metadata)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        materialize_all200_training_skill_bank(
            source_dirs=[source_a, source_b],
            cohort_manifest=cohort_path,
            split_manifest=split_path,
            output_path=tmp_path / "rejected_metadata.json",
        )


def test_all200_allows_only_the_explicit_three_repaired_historical_records(tmp_path):
    cohort_path, split_path, source_a, source_b = _all200_evidence(tmp_path)
    for task_id in KNOWN_JUNE24_REPAIRED_TASK_IDS:
        index = int(task_id.rsplit("_", 1)[-1])
        source_dir = source_a if index < 50 else source_b
        path = source_dir / f"skill_summary_call_{task_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["repair_request"] = {"redacted": True}
        value["repair_response"] = {"redacted": True}
        path.write_text(json.dumps(value), encoding="utf-8")

    bank_path = tmp_path / "allowed_repaired.json"
    bank = materialize_all200_training_skill_bank(
        source_dirs=[source_a, source_b],
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        output_path=bank_path,
        allowed_repaired_task_ids=KNOWN_JUNE24_REPAIRED_TASK_IDS,
    )
    assert [item["task_id"] for item in bank["repaired_source_overrides"]] == list(
        KNOWN_JUNE24_REPAIRED_TASK_IDS
    )
    assert all(item["repair_metadata_fields"] == ["repair_request", "repair_response"] for item in bank["repaired_source_overrides"])
    assert sum(item["repaired_source_override"] is True for item in bank["source_calls"]) == 3
    loaded = load_skill_bank(
        bank_path,
        cohort_manifest=cohort_path,
        split_manifest=split_path,
        verify_source_files=True,
    )
    assert len(loaded.records) == 160


def test_all200_repaired_override_remains_fail_closed(tmp_path):
    cohort_path, split_path, source_a, source_b = _all200_evidence(tmp_path)
    with pytest.raises(ValueError, match="unrecognized repaired-record overrides"):
        materialize_all200_training_skill_bank(
            source_dirs=[source_a, source_b],
            cohort_manifest=cohort_path,
            split_manifest=split_path,
            output_path=tmp_path / "unknown_override.json",
            allowed_repaired_task_ids=["multi_turn_base_1"],
        )

    with pytest.raises(ValueError, match="no repair metadata exists"):
        materialize_all200_training_skill_bank(
            source_dirs=[source_a, source_b],
            cohort_manifest=cohort_path,
            split_manifest=split_path,
            output_path=tmp_path / "stale_override.json",
            allowed_repaired_task_ids=KNOWN_JUNE24_REPAIRED_TASK_IDS,
        )


def test_all200_expected_counts_are_frozen():
    assert EXPECTED_OUTCOME_COUNTS == {
        "fixed": 40,
        "both_wrong": 112,
        "harmed": 13,
        "both_correct": 35,
    }


def test_all200_cohort_rejects_validity_or_pairwise_hash_drift(tmp_path):
    cohort_path, _, _, _ = _all200_evidence(tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["records"][0]["baseline_valid"] = True
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with validity flags"):
        build_stratified_split_manifest(cohort_path)

    second = tmp_path / "hash_drift"
    second.mkdir()
    cohort_path, _, _, _ = _all200_evidence(second)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    pairwise = Path(cohort["pairwise_tasks_csv"]["path"])
    pairwise.write_text(pairwise.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pairwise CSV SHA-256 mismatch"):
        build_stratified_split_manifest(cohort_path)
