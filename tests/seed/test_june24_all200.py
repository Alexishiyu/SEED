import csv
import hashlib
import json
from pathlib import Path

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
    load_skill_bank,
    materialize_all200_training_skill_bank,
    validate_stratified_split_manifest,
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
