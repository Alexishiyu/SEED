"""Fail-closed June 24 Skill-SD inputs for BFCL OPD training.

This module deliberately supports only the historical three-field Skill-SD
summaries.  It rejects V2R, possible-answer, predicted-call, predicted-state,
and structured-ground-truth payloads instead of attempting to sanitize them at
training time.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BACKEND_NAME = "june24_skill_summary"
SCHEMA_VERSION = "seed.june24_skill_summary.v1"
ALL200_BANK_SCHEMA_VERSION = "seed.june24_skill_summary.all200.v2"
ALL200_COHORT_SCHEMA_VERSION = "seed.june24_all200.v1"
ALL200_SPLIT_SCHEMA_VERSION = "seed.june24_all200_split.v1"
FIXED_SELECTION = "teacher_success_student_failed/fixed"
EXPECTED_FIXED_COUNT = 40
EXPECTED_REJECTED_COUNT = 75
EXPECTED_ALL200_COUNT = 200
DEFAULT_SPLIT_SEED = 20260624
DEFAULT_TRAIN_ITERATIONS = 5
DEFAULT_TRAIN_BATCH_SIZE = 4
OUTCOME_CLASSES = ("fixed", "both_wrong", "harmed", "both_correct")
EXPECTED_OUTCOME_COUNTS = {
    "fixed": 40,
    "both_wrong": 112,
    "harmed": 13,
    "both_correct": 35,
}
EXPECTED_TRAIN_OUTCOME_COUNTS = {
    "fixed": 32,
    "both_wrong": 90,
    "harmed": 10,
    "both_correct": 28,
}
EXPECTED_VALIDATION_OUTCOME_COUNTS = {
    "fixed": 8,
    "both_wrong": 22,
    "harmed": 3,
    "both_correct": 7,
}
REQUIRED_FIELDS = ("success_analysis", "mistake_analysis", "golden_workflow")
SOURCE_FIELDS = ("source_call_path", "source_call_sha256")
ALLOWED_RECORD_FIELDS = {"task_id", *REQUIRED_FIELDS, *SOURCE_FIELDS}
FORBIDDEN_FIELD_NAMES = {
    "possible_answer",
    "possible_answers",
    "predicted_tool_calls",
    "predicted_final_state",
    "generated_reference_answer",
    "structured_reference",
    "structured_ground_truth",
    "reference_target",
    "ground_truth_target",
    "v2r",
}

# User-authorized exception for the historical June 24 all-200 experiment.
# These exact source calls contain a second, repaired summarizer response.  The
# exception is task-local, remains forbidden in the fixed-40 path, and is
# recorded in the normalized bank without copying repair prompts/responses.
KNOWN_JUNE24_REPAIRED_TASK_IDS = (
    "multi_turn_base_56",
    "multi_turn_base_154",
    "multi_turn_base_169",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _task_id(record: Mapping[str, Any]) -> str:
    return str(record.get("task_id") or record.get("id") or "").strip()


def _bfcl_task_sort_key(task_id: str) -> tuple[int, str]:
    prefix = "multi_turn_base_"
    if task_id.startswith(prefix):
        suffix = task_id[len(prefix) :]
        if suffix.isdigit():
            return int(suffix), task_id
    return EXPECTED_ALL200_COUNT + 1, task_id


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_csv_bool(value: Any, *, field: str, task_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{task_id}: {field} must be True or False, got {value!r}")


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            normalized = key_text.strip().lower()
            if normalized in FORBIDDEN_FIELD_NAMES or "possible_answer" in normalized:
                found.append(location)
            found.extend(_forbidden_keys(child, location))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def read_fixed_ids_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task_id", "classification", "baseline_valid", "skill_valid"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"fixed-task CSV lacks required columns: {sorted(required)}")
    invalid = [
        str(row.get("task_id") or "")
        for row in rows
        if str(row.get("classification")) != "fixed"
        or str(row.get("baseline_valid")) != "False"
        or str(row.get("skill_valid")) != "True"
    ]
    if invalid:
        raise ValueError(f"fixed-task CSV contains non-fixed rows: {invalid[:10]}")
    task_ids = [str(row["task_id"]).strip() for row in rows]
    if len(task_ids) != EXPECTED_FIXED_COUNT or len(set(task_ids)) != EXPECTED_FIXED_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_FIXED_COUNT} unique fixed tasks, got "
            f"{len(task_ids)} rows and {len(set(task_ids))} unique ids"
        )
    return task_ids


def read_rejected_success_ids(path: Path) -> list[str]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("rejected teacher-success manifest must be a JSON object")
    task_ids = value.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ValueError("rejected teacher-success manifest lacks a string task_ids list")
    if len(task_ids) != EXPECTED_REJECTED_COUNT or len(set(task_ids)) != EXPECTED_REJECTED_COUNT:
        raise ValueError("input is not the rejected 75-task all-teacher-success cohort")
    return list(task_ids)


def build_fixed_manifest(fixed_csv: Path, rejected_manifest: Path) -> dict[str, Any]:
    fixed_ids = read_fixed_ids_csv(fixed_csv)
    rejected_ids = read_rejected_success_ids(rejected_manifest)
    missing = sorted(set(fixed_ids) - set(rejected_ids))
    extras = sorted(set(rejected_ids) - set(fixed_ids))
    if missing or len(extras) != 35:
        raise ValueError(
            "fixed cohort must be a strict 40-task subset of the rejected 75-task cohort; "
            f"missing={missing}, extras={len(extras)}"
        )
    return {
        "schema_version": "seed.june24_fixed40.v1",
        "selection": FIXED_SELECTION,
        "definition": "June 24 historical baseline/student wrong and skilled teacher correct",
        "task_count": EXPECTED_FIXED_COUNT,
        "task_ids": fixed_ids,
        "fixed_tasks_csv": {
            "path": str(fixed_csv),
            "sha256": sha256_file(fixed_csv),
        },
        "rejected_cohort": {
            "selection": "all June 24 skilled-teacher successes",
            "task_count": EXPECTED_REJECTED_COUNT,
            "extra_already_correct_task_count": len(extras),
            "extra_already_correct_task_ids": extras,
            "path": str(rejected_manifest),
            "sha256": sha256_file(rejected_manifest),
            "reuse_forbidden": True,
        },
        "fail_closed_checks": {
            "all_rows_classification_fixed": True,
            "all_rows_baseline_student_wrong": True,
            "all_rows_skilled_teacher_correct": True,
            "exact_unique_count_40": True,
            "strict_subset_of_rejected_75": True,
        },
    }


def validate_fixed_manifest(value: Mapping[str, Any]) -> list[str]:
    if value.get("selection") != FIXED_SELECTION:
        raise ValueError("fixed manifest is not the corrected fixed-40 cohort")
    task_ids = value.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ValueError("fixed manifest lacks a string task_ids list")
    if len(task_ids) != EXPECTED_FIXED_COUNT or len(set(task_ids)) != EXPECTED_FIXED_COUNT:
        raise ValueError("fixed manifest must contain exactly 40 unique task ids")
    rejected = value.get("rejected_cohort")
    if not isinstance(rejected, Mapping):
        raise ValueError("fixed manifest lacks rejected_cohort provenance")
    if int(rejected.get("task_count") or 0) != EXPECTED_REJECTED_COUNT:
        raise ValueError("fixed manifest does not identify the rejected 75-task cohort")
    if rejected.get("reuse_forbidden") is not True:
        raise ValueError("fixed manifest must explicitly forbid reuse of the 75-task cohort")
    checks = value.get("fail_closed_checks")
    required_checks = {
        "all_rows_classification_fixed",
        "all_rows_baseline_student_wrong",
        "all_rows_skilled_teacher_correct",
        "exact_unique_count_40",
        "strict_subset_of_rejected_75",
    }
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise ValueError("fixed manifest fail-closed checks are incomplete")
    return list(task_ids)


def build_all200_cohort_manifest(pairwise_tasks_csv: Path) -> dict[str, Any]:
    """Build the independent, audited June 24 all-200 cohort authority."""

    with pairwise_tasks_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task_id", "classification", "baseline_valid", "skill_valid"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"pairwise task CSV lacks required columns: {sorted(required)}")

    expected_ids = {f"multi_turn_base_{index}" for index in range(EXPECTED_ALL200_COUNT)}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_validity = {
        "fixed": (False, True),
        "both_wrong": (False, False),
        "harmed": (True, False),
        "both_correct": (True, True),
    }
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        classification = str(row.get("classification") or "").strip()
        if not task_id or task_id in seen:
            raise ValueError(f"pairwise task CSV has missing or duplicate task id: {task_id!r}")
        if classification not in expected_validity:
            raise ValueError(f"{task_id}: unsupported pairwise classification {classification!r}")
        baseline_valid = _parse_csv_bool(row.get("baseline_valid"), field="baseline_valid", task_id=task_id)
        skill_valid = _parse_csv_bool(row.get("skill_valid"), field="skill_valid", task_id=task_id)
        if (baseline_valid, skill_valid) != expected_validity[classification]:
            raise ValueError(
                f"{task_id}: classification {classification} disagrees with "
                f"baseline_valid={baseline_valid}, skill_valid={skill_valid}"
            )
        seen.add(task_id)
        records.append(
            {
                "task_id": task_id,
                "classification": classification,
                "baseline_valid": baseline_valid,
                "skill_valid": skill_valid,
            }
        )

    if seen != expected_ids:
        missing = sorted(expected_ids - seen, key=_bfcl_task_sort_key)
        extras = sorted(seen - expected_ids, key=_bfcl_task_sort_key)
        raise ValueError(f"all-200 cohort identity mismatch: missing={missing[:10]}, extras={extras[:10]}")
    records.sort(key=lambda item: _bfcl_task_sort_key(item["task_id"]))
    counts = {name: sum(record["classification"] == name for record in records) for name in OUTCOME_CLASSES}
    if counts != EXPECTED_OUTCOME_COUNTS:
        raise ValueError(f"all-200 outcome counts mismatch: expected={EXPECTED_OUTCOME_COUNTS}, got={counts}")

    return {
        "schema_version": ALL200_COHORT_SCHEMA_VERSION,
        "selection": "june24_all200/pairwise_outcomes",
        "definition": "All 200 June 24 BFCL tasks classified by baseline versus skilled-teacher validity",
        "task_count": EXPECTED_ALL200_COUNT,
        "task_ids": [record["task_id"] for record in records],
        "classification_counts": counts,
        "records": records,
        "pairwise_tasks_csv": {
            "path": str(pairwise_tasks_csv.resolve()),
            "sha256": sha256_file(pairwise_tasks_csv),
        },
        "fail_closed_checks": {
            "exact_multi_turn_base_0_through_199": True,
            "exact_unique_count_200": True,
            "classification_matches_validity": True,
            "expected_outcome_counts": True,
        },
    }


def validate_all200_cohort_manifest(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != ALL200_COHORT_SCHEMA_VERSION:
        raise ValueError("cohort manifest is not the audited June 24 all-200 cohort")
    if value.get("selection") != "june24_all200/pairwise_outcomes":
        raise ValueError("all-200 cohort selection is invalid")
    if int(value.get("task_count") or 0) != EXPECTED_ALL200_COUNT:
        raise ValueError("all-200 cohort task_count is invalid")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_ALL200_COUNT:
        raise ValueError("all-200 cohort manifest must contain exactly 200 records")
    expected_ids = [f"multi_turn_base_{index}" for index in range(EXPECTED_ALL200_COUNT)]
    task_ids = value.get("task_ids")
    if task_ids != expected_ids:
        raise ValueError("all-200 cohort task ids must be ordered multi_turn_base_0 through 199")
    record_ids = [str(record.get("task_id") or "") for record in records if isinstance(record, Mapping)]
    if record_ids != expected_ids:
        raise ValueError("all-200 cohort records do not match the canonical task ordering")
    expected_validity = {
        "fixed": (False, True),
        "both_wrong": (False, False),
        "harmed": (True, False),
        "both_correct": (True, True),
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("all-200 cohort record is not an object")
        classification = record.get("classification")
        if classification not in expected_validity:
            raise ValueError(f"{record.get('task_id')}: invalid outcome classification")
        observed = (record.get("baseline_valid"), record.get("skill_valid"))
        if observed != expected_validity[classification]:
            raise ValueError(
                f"{record.get('task_id')}: outcome classification disagrees with validity flags"
            )
    counts = {name: sum(record.get("classification") == name for record in records) for name in OUTCOME_CLASSES}
    if counts != EXPECTED_OUTCOME_COUNTS or value.get("classification_counts") != counts:
        raise ValueError("all-200 cohort classification counts are invalid")
    checks = value.get("fail_closed_checks")
    required_checks = {
        "exact_multi_turn_base_0_through_199",
        "exact_unique_count_200",
        "classification_matches_validity",
        "expected_outcome_counts",
    }
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise ValueError("all-200 cohort fail-closed checks are incomplete")
    source = value.get("pairwise_tasks_csv")
    if not isinstance(source, Mapping):
        raise ValueError("all-200 cohort lacks pairwise CSV provenance")
    source_path = Path(str(source.get("path") or "")).expanduser()
    source_hash = str(source.get("sha256") or "")
    if not source_path.is_file() or sha256_file(source_path) != source_hash:
        raise ValueError("all-200 cohort pairwise CSV SHA-256 mismatch")
    return [dict(record) for record in records]


def build_stratified_split_manifest(
    cohort_manifest: Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    iterations: int = DEFAULT_TRAIN_ITERATIONS,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
) -> dict[str, Any]:
    cohort_value = _read_json(cohort_manifest)
    if not isinstance(cohort_value, Mapping):
        raise ValueError("all-200 cohort manifest must be a JSON object")
    records = validate_all200_cohort_manifest(cohort_value)
    if seed != DEFAULT_SPLIT_SEED or iterations != DEFAULT_TRAIN_ITERATIONS or batch_size != DEFAULT_TRAIN_BATCH_SIZE:
        raise ValueError("the canonical all-200 split requires seed=20260624, iterations=5, batch_size=4")

    by_class = {
        name: [record["task_id"] for record in records if record["classification"] == name]
        for name in OUTCOME_CLASSES
    }
    train_ids: list[str] = []
    validation_ids: list[str] = []
    for name in OUTCOME_CLASSES:
        ordered = sorted(by_class[name], key=lambda task_id: (_sha256_text(f"{seed}:{task_id}"), task_id))
        validation_count = EXPECTED_VALIDATION_OUTCOME_COUNTS[name]
        validation_ids.extend(ordered[:validation_count])
        train_ids.extend(ordered[validation_count:])
    train_ids.sort(key=_bfcl_task_sort_key)
    validation_ids.sort(key=_bfcl_task_sort_key)

    iteration_schedules = []
    for iteration in range(1, iterations + 1):
        ordered = sorted(
            train_ids,
            key=lambda task_id: (_sha256_text(f"{seed}:{iteration}:{task_id}"), task_id),
        )
        batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
        iteration_schedules.append(
            {
                "iteration": iteration,
                "task_ids": ordered,
                "batches": batches,
            }
        )

    return {
        "schema_version": ALL200_SPLIT_SCHEMA_VERSION,
        "selection": "june24_all200/stratified_160_40",
        "seed": seed,
        "selection_key": "SHA256('20260624:<task_id>') within outcome class",
        "schedule_key": "SHA256('20260624:<iteration>:<task_id>')",
        "cohort_manifest": {
            "path": str(cohort_manifest.resolve()),
            "sha256": sha256_file(cohort_manifest),
        },
        "train": {
            "task_count": len(train_ids),
            "task_ids": train_ids,
            "classification_counts": dict(EXPECTED_TRAIN_OUTCOME_COUNTS),
        },
        "validation": {
            "task_count": len(validation_ids),
            "task_ids": validation_ids,
            "classification_counts": dict(EXPECTED_VALIDATION_OUTCOME_COUNTS),
        },
        "training_schedule": {
            "iterations": iterations,
            "batch_size": batch_size,
            "updates_per_iteration": len(train_ids) // batch_size,
            "total_updates": iterations * len(train_ids) // batch_size,
            "total_rollouts": iterations * len(train_ids),
            "iteration_schedules": iteration_schedules,
        },
    }


def validate_stratified_split_manifest(
    value: Mapping[str, Any],
    *,
    cohort_manifest: Path,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    if value.get("schema_version") != ALL200_SPLIT_SCHEMA_VERSION:
        raise ValueError("split manifest is not the canonical June 24 all-200 160/40 split")
    cohort_value = _read_json(cohort_manifest)
    if not isinstance(cohort_value, Mapping):
        raise ValueError("all-200 cohort manifest must be a JSON object")
    records = validate_all200_cohort_manifest(cohort_value)
    cohort_ref = value.get("cohort_manifest")
    if not isinstance(cohort_ref, Mapping) or cohort_ref.get("sha256") != sha256_file(cohort_manifest):
        raise ValueError("split manifest cohort hash mismatch")
    expected = build_stratified_split_manifest(cohort_manifest)
    if value != expected:
        raise ValueError("split manifest differs from the deterministic canonical 160/40 split")
    train_ids = list(value["train"]["task_ids"])
    validation_ids = list(value["validation"]["task_ids"])
    all_ids = {record["task_id"] for record in records}
    if len(train_ids) != 160 or len(validation_ids) != 40 or set(train_ids) & set(validation_ids):
        raise ValueError("split manifest does not contain disjoint 160/40 task sets")
    if set(train_ids) | set(validation_ids) != all_ids:
        raise ValueError("split manifest train/validation union does not cover all 200 tasks")
    schedules = list(value["training_schedule"]["iteration_schedules"])
    return train_ids, validation_ids, schedules


def _repair_metadata_fields(source: Mapping[str, Any]) -> tuple[str, ...]:
    fields = []
    repair_attempts = source.get("repair_attempts")
    if repair_attempts not in (None, []):
        fields.append("repair_attempts")
    if source.get("repair_request") is not None:
        fields.append("repair_request")
    if source.get("repair_response") is not None:
        fields.append("repair_response")
    return tuple(fields)


def _extract_three_fields(
    source: Mapping[str, Any],
    task_id: str,
    *,
    allow_repaired: bool = False,
) -> dict[str, str]:
    fallback = source.get("fallback")
    if fallback not in (None, False):
        raise ValueError(f"{task_id}: fallback June 24 skill records are forbidden")
    repair_fields = _repair_metadata_fields(source)
    if repair_fields and not allow_repaired:
        raise ValueError(f"{task_id}: repaired June 24 skill records are forbidden")
    if allow_repaired and not repair_fields:
        raise ValueError(f"{task_id}: repaired-record override was requested but no repair metadata exists")
    reference_mode = str(source.get("reference_mode") or "").strip().lower()
    if any(
        marker in reference_mode
        for marker in ("possible_answer", "direct_ground_truth", "regenerated", "v2r")
    ):
        raise ValueError(f"{task_id}: structured/V2R June 24 skill source is forbidden")
    forbidden = _forbidden_keys(source)
    if forbidden:
        raise ValueError(f"{task_id}: forbidden fields in selected skill payload: {forbidden[:8]}")
    skill = source.get("parsed_skill")
    if not isinstance(skill, Mapping):
        skill = source.get("skill")
    if not isinstance(skill, Mapping):
        skill = source
    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = skill.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{task_id}: missing non-empty {field}")
        values[field] = value.strip()
    return values


def materialize_skill_bank(
    *,
    source_dirs: Sequence[Path],
    fixed_manifest: Path,
    output_path: Path,
    selected_task_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_value = _read_json(fixed_manifest)
    if not isinstance(manifest_value, Mapping):
        raise ValueError("fixed manifest must be a JSON object")
    fixed_ids = validate_fixed_manifest(manifest_value)
    selected = list(selected_task_ids) if selected_task_ids is not None else fixed_ids
    if not selected:
        raise ValueError("no task ids selected")
    if len(selected) != len(set(selected)):
        raise ValueError("selected task ids contain duplicates")
    outside = [task_id for task_id in selected if task_id not in set(fixed_ids)]
    if outside:
        raise ValueError(f"selected task ids are outside fixed-40: {outside[:10]}")

    records = []
    for task_id in selected:
        candidates = [directory / f"skill_summary_call_{task_id}.json" for directory in source_dirs]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"{task_id}: expected one June 24 source call, found {len(matches)}")
        source_path = matches[0].resolve()
        source = _read_json(source_path)
        if not isinstance(source, Mapping):
            raise ValueError(f"{task_id}: source call must be a JSON object")
        fields = _extract_three_fields(source, task_id)
        records.append(
            {
                "task_id": task_id,
                **fields,
                "source_call_path": str(source_path),
                "source_call_sha256": sha256_file(source_path),
            }
        )

    bank = {
        "schema_version": SCHEMA_VERSION,
        "analysis_backend": BACKEND_NAME,
        "fixed_manifest_path": str(fixed_manifest.resolve()),
        "fixed_manifest_sha256": sha256_file(fixed_manifest),
        "task_count": len(records),
        "task_ids": selected,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bank, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return bank


def materialize_all200_training_skill_bank(
    *,
    source_dirs: Sequence[Path],
    cohort_manifest: Path,
    split_manifest: Path,
    output_path: Path,
    allowed_repaired_task_ids: Iterable[str] = (),
) -> dict[str, Any]:
    cohort_value = _read_json(cohort_manifest)
    split_value = _read_json(split_manifest)
    if not isinstance(cohort_value, Mapping) or not isinstance(split_value, Mapping):
        raise ValueError("cohort and split manifests must be JSON objects")
    records = validate_all200_cohort_manifest(cohort_value)
    train_ids, _, _ = validate_stratified_split_manifest(
        split_value,
        cohort_manifest=cohort_manifest,
    )
    train_set = set(train_ids)
    allowed_repaired = tuple(allowed_repaired_task_ids)
    if len(allowed_repaired) != len(set(allowed_repaired)):
        raise ValueError("repaired-record override task ids contain duplicates")
    unknown_repaired = sorted(set(allowed_repaired) - set(KNOWN_JUNE24_REPAIRED_TASK_IDS))
    if unknown_repaired:
        raise ValueError(f"unrecognized repaired-record overrides: {unknown_repaired}")
    allowed_repaired_set = set(allowed_repaired)
    normalized_records = []
    source_calls = []
    repaired_source_overrides = []
    for cohort_record in records:
        task_id = cohort_record["task_id"]
        candidates = [directory / f"skill_summary_call_{task_id}.json" for directory in source_dirs]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"{task_id}: expected one June 24 source call, found {len(matches)}")
        source_path = matches[0].resolve()
        source = _read_json(source_path)
        if not isinstance(source, Mapping):
            raise ValueError(f"{task_id}: source call must be a JSON object")
        repair_fields = _repair_metadata_fields(source)
        fields = _extract_three_fields(
            source,
            task_id,
            allow_repaired=task_id in allowed_repaired_set,
        )
        source_hash = sha256_file(source_path)
        source_calls.append(
            {
                "task_id": task_id,
                "source_call_path": str(source_path),
                "source_call_sha256": source_hash,
                "repaired_source_override": task_id in allowed_repaired_set,
            }
        )
        if task_id in allowed_repaired_set:
            repaired_source_overrides.append(
                {
                    "task_id": task_id,
                    "source_call_path": str(source_path),
                    "source_call_sha256": source_hash,
                    "repair_metadata_fields": list(repair_fields),
                }
            )
        if task_id in train_set:
            normalized_records.append(
                {
                    "task_id": task_id,
                    **fields,
                    "source_call_path": str(source_path),
                    "source_call_sha256": source_hash,
                }
            )
    normalized_records.sort(key=lambda item: _bfcl_task_sort_key(item["task_id"]))
    if [record["task_id"] for record in normalized_records] != train_ids:
        raise ValueError("training skill records do not exactly match the frozen 160-task split")
    if {item["task_id"] for item in repaired_source_overrides} != allowed_repaired_set:
        raise ValueError("repaired-record overrides were not fully audited")

    bank = {
        "schema_version": ALL200_BANK_SCHEMA_VERSION,
        "analysis_backend": BACKEND_NAME,
        "cohort_manifest_path": str(cohort_manifest.resolve()),
        "cohort_manifest_sha256": sha256_file(cohort_manifest),
        "split_manifest_path": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
        "task_count": len(normalized_records),
        "task_ids": train_ids,
        "source_task_count": len(source_calls),
        "source_calls": source_calls,
        "repaired_source_policy": "explicit_historical_task_allowlist",
        "repaired_source_overrides": repaired_source_overrides,
        "records": normalized_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bank, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return bank


@dataclass(frozen=True)
class June24SkillSummaryBank:
    path: Path
    sha256: str
    fixed_manifest_path: Path | None
    fixed_manifest_sha256: str | None
    fixed_task_ids: tuple[str, ...]
    records: Mapping[str, Mapping[str, str]]
    cohort_manifest_path: Path | None = None
    cohort_manifest_sha256: str | None = None
    split_manifest_path: Path | None = None
    split_manifest_sha256: str | None = None
    validation_task_ids: tuple[str, ...] = ()
    outcome_class_by_task_id: Mapping[str, str] | None = None

    analysis_prompt_version: str = "seed"
    skill_mode: str = "episode_only"

    def summary(self, task_id: str) -> Mapping[str, str]:
        try:
            return self.records[task_id]
        except KeyError as exc:
            raise ValueError(f"June 24 skill bank has no record for {task_id}") from exc


def load_skill_bank(
    path: Path,
    *,
    fixed_manifest: Path | None = None,
    cohort_manifest: Path | None = None,
    split_manifest: Path | None = None,
    required_task_ids: Iterable[str] | None = None,
    verify_source_files: bool = True,
) -> June24SkillSummaryBank:
    path = path.expanduser().resolve()
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("skill bank must be a JSON object")
    if _forbidden_keys(value):
        raise ValueError(f"skill bank contains forbidden fields: {_forbidden_keys(value)[:8]}")

    fixed_mode = fixed_manifest is not None
    all200_mode = cohort_manifest is not None or split_manifest is not None
    if fixed_mode == all200_mode:
        raise ValueError("select exactly one skill-bank authority: fixed manifest or all-200 cohort/split manifests")
    validation_ids: list[str] = []
    outcome_class_by_task_id: dict[str, str] = {}
    if fixed_mode:
        fixed_manifest = fixed_manifest.expanduser().resolve()
        manifest_value = _read_json(fixed_manifest)
        if not isinstance(manifest_value, Mapping):
            raise ValueError("fixed manifest must be a JSON object")
        if value.get("schema_version") != SCHEMA_VERSION or value.get("analysis_backend") != BACKEND_NAME:
            raise ValueError("skill bank is not a June 24 fixed-40 three-field SEED bank")
        allowed_ids = validate_fixed_manifest(manifest_value)
        outcome_class_by_task_id = {task_id: "fixed" for task_id in allowed_ids}
        manifest_hash = sha256_file(fixed_manifest)
        if value.get("fixed_manifest_sha256") != manifest_hash:
            raise ValueError("skill bank fixed-manifest hash mismatch")
        cohort_hash = None
        split_hash = None
    else:
        if cohort_manifest is None or split_manifest is None:
            raise ValueError("all-200 skill bank requires both cohort_manifest and split_manifest")
        cohort_manifest = cohort_manifest.expanduser().resolve()
        split_manifest = split_manifest.expanduser().resolve()
        cohort_value = _read_json(cohort_manifest)
        split_value = _read_json(split_manifest)
        if not isinstance(cohort_value, Mapping) or not isinstance(split_value, Mapping):
            raise ValueError("cohort and split manifests must be JSON objects")
        if value.get("schema_version") != ALL200_BANK_SCHEMA_VERSION or value.get("analysis_backend") != BACKEND_NAME:
            raise ValueError("skill bank is not a June 24 all-200 three-field SEED bank")
        allowed_ids, validation_ids, _ = validate_stratified_split_manifest(
            split_value,
            cohort_manifest=cohort_manifest,
        )
        outcome_class_by_task_id = {
            str(record["task_id"]): str(record["classification"])
            for record in validate_all200_cohort_manifest(cohort_value)
        }
        cohort_hash = sha256_file(cohort_manifest)
        split_hash = sha256_file(split_manifest)
        if value.get("cohort_manifest_sha256") != cohort_hash:
            raise ValueError("skill bank cohort-manifest hash mismatch")
        if value.get("split_manifest_sha256") != split_hash:
            raise ValueError("skill bank split-manifest hash mismatch")
        manifest_hash = None
        source_calls = value.get("source_calls")
        if not isinstance(source_calls, list) or len(source_calls) != EXPECTED_ALL200_COUNT:
            raise ValueError("all-200 skill bank must record exactly 200 source calls")
        if [str(item.get("task_id") or "") for item in source_calls if isinstance(item, Mapping)] != [
            f"multi_turn_base_{index}" for index in range(EXPECTED_ALL200_COUNT)
        ]:
            raise ValueError("all-200 source-call manifest does not cover canonical task ids 0 through 199")
        overrides = value.get("repaired_source_overrides")
        if value.get("repaired_source_policy") != "explicit_historical_task_allowlist" or not isinstance(
            overrides, list
        ):
            raise ValueError("all-200 skill bank lacks an explicit repaired-source policy")
        override_ids = [str(item.get("task_id") or "") for item in overrides if isinstance(item, Mapping)]
        if len(override_ids) != len(overrides) or len(override_ids) != len(set(override_ids)):
            raise ValueError("all-200 repaired-source overrides are malformed or duplicated")
        if set(override_ids) - set(KNOWN_JUNE24_REPAIRED_TASK_IDS):
            raise ValueError("all-200 skill bank contains an unauthorized repaired-source override")
        flagged_ids = [
            str(item.get("task_id") or "")
            for item in source_calls
            if isinstance(item, Mapping) and item.get("repaired_source_override") is True
        ]
        if flagged_ids != override_ids:
            raise ValueError("all-200 repaired-source override audit does not match source calls")
        for override in overrides:
            task_id = str(override.get("task_id") or "")
            source = next(item for item in source_calls if item.get("task_id") == task_id)
            if (
                override.get("source_call_path") != source.get("source_call_path")
                or override.get("source_call_sha256") != source.get("source_call_sha256")
                or not override.get("repair_metadata_fields")
            ):
                raise ValueError(f"{task_id}: repaired-source override provenance is incomplete")
        if verify_source_files:
            for item in source_calls:
                source_path = Path(str(item.get("source_call_path") or "")).expanduser()
                source_hash = str(item.get("source_call_sha256") or "")
                if not source_path.is_file() or sha256_file(source_path) != source_hash:
                    raise ValueError(f"{item.get('task_id')}: all-200 source call SHA-256 mismatch")

    raw_records = value.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("skill bank records must be a non-empty list")
    records: dict[str, Mapping[str, str]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("skill bank record is not an object")
        unknown = set(raw) - ALLOWED_RECORD_FIELDS
        if unknown:
            raise ValueError(f"skill bank record contains unsupported fields: {sorted(unknown)}")
        task_id = _task_id(raw)
        if not task_id or task_id in records:
            raise ValueError(f"missing or duplicate task_id: {task_id!r}")
        if task_id not in set(allowed_ids):
            scope = "corrected fixed-40 cohort" if fixed_mode else "frozen 160-task training split"
            raise ValueError(f"{task_id}: outside {scope}")
        for field in REQUIRED_FIELDS:
            if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
                raise ValueError(f"{task_id}: missing non-empty {field}")
        source_path = Path(str(raw.get("source_call_path") or "")).expanduser()
        source_hash = str(raw.get("source_call_sha256") or "")
        if len(source_hash) != 64:
            raise ValueError(f"{task_id}: invalid source SHA-256")
        if verify_source_files:
            if not source_path.is_file():
                raise ValueError(f"{task_id}: source call file is unavailable: {source_path}")
            if sha256_file(source_path) != source_hash:
                raise ValueError(f"{task_id}: source call SHA-256 mismatch")
        records[task_id] = {key: str(raw[key]) for key in ALLOWED_RECORD_FIELDS}

    if int(value.get("task_count") or 0) != len(records):
        raise ValueError("skill bank task_count does not match records")
    requested = list(required_task_ids) if required_task_ids is not None else list(records)
    if len(requested) != len(set(requested)):
        raise ValueError("required task ids contain duplicates")
    missing = [task_id for task_id in requested if task_id not in records]
    outside = [task_id for task_id in requested if task_id not in set(allowed_ids)]
    if missing or outside:
        raise ValueError(f"June 24 coverage failure: missing={missing}, outside_authority={outside}")

    return June24SkillSummaryBank(
        path=path,
        sha256=sha256_file(path),
        fixed_manifest_path=fixed_manifest if fixed_mode else None,
        fixed_manifest_sha256=manifest_hash,
        fixed_task_ids=tuple(allowed_ids) if fixed_mode else (),
        cohort_manifest_path=cohort_manifest if all200_mode else None,
        cohort_manifest_sha256=cohort_hash,
        split_manifest_path=split_manifest if all200_mode else None,
        split_manifest_sha256=split_hash,
        validation_task_ids=tuple(validation_ids),
        outcome_class_by_task_id=outcome_class_by_task_id,
        records=records,
    )


def guidance_block(task_id: str, summary: Mapping[str, str]) -> str:
    return (
        "\n\n# Skill-SD Teacher Guidance\n"
        "The following task-local skill is training-time privileged guidance. "
        "Use it as diagnostic workflow guidance, not as ground truth. Verify "
        "all task facts with tool results before acting.\n\n"
        f"Task id: {task_id}\n"
        f"success_analysis: {summary['success_analysis']}\n"
        f"mistake_analysis: {summary['mistake_analysis']}\n"
        f"golden_workflow: {summary['golden_workflow']}\n"
    )


def inject_guidance_into_system_prompt(student_prompt: str, task_id: str, summary: Mapping[str, str]) -> str:
    prefix = "<|im_start|>system\n"
    end_marker = "<|im_end|>"
    if not student_prompt.startswith(prefix):
        raise ValueError(f"{task_id}: canonical BFCL Qwen prompt has no leading system block")
    end_index = student_prompt.find(end_marker, len(prefix))
    if end_index < 0:
        raise ValueError(f"{task_id}: canonical BFCL Qwen system block is unterminated")
    return student_prompt[:end_index].rstrip() + guidance_block(task_id, summary) + student_prompt[end_index:]


def validate_context_lengths(
    *,
    tokenizer: Any,
    bank: June24SkillSummaryBank,
    student_prompts: Mapping[str, str],
    max_prompt_length: int,
) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for task_id, student_prompt in student_prompts.items():
        teacher_prompt = inject_guidance_into_system_prompt(student_prompt, task_id, bank.summary(task_id))
        student_tokens = len(tokenizer.encode(student_prompt, add_special_tokens=False))
        teacher_tokens = len(tokenizer.encode(teacher_prompt, add_special_tokens=False))
        if student_tokens > max_prompt_length or teacher_tokens > max_prompt_length:
            raise ValueError(
                f"{task_id}: prompt overflow student={student_tokens}, teacher={teacher_tokens}, "
                f"limit={max_prompt_length}; prompts are never truncated"
            )
        report[task_id] = {"student_tokens": student_tokens, "teacher_tokens": teacher_tokens}
    return report
