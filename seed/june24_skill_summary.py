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
FIXED_SELECTION = "teacher_success_student_failed/fixed"
EXPECTED_FIXED_COUNT = 40
EXPECTED_REJECTED_COUNT = 75
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


def _extract_three_fields(source: Mapping[str, Any], task_id: str) -> dict[str, str]:
    forbidden = _forbidden_keys(source.get("parsed_skill", source))
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


@dataclass(frozen=True)
class June24SkillSummaryBank:
    path: Path
    sha256: str
    fixed_manifest_path: Path
    fixed_manifest_sha256: str
    fixed_task_ids: tuple[str, ...]
    records: Mapping[str, Mapping[str, str]]

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
    fixed_manifest: Path,
    required_task_ids: Iterable[str] | None = None,
    verify_source_files: bool = True,
) -> June24SkillSummaryBank:
    path = path.expanduser().resolve()
    fixed_manifest = fixed_manifest.expanduser().resolve()
    value = _read_json(path)
    manifest_value = _read_json(fixed_manifest)
    if not isinstance(value, Mapping) or not isinstance(manifest_value, Mapping):
        raise ValueError("skill bank and fixed manifest must be JSON objects")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("analysis_backend") != BACKEND_NAME:
        raise ValueError("skill bank is not a June 24 three-field SEED bank")
    if _forbidden_keys(value):
        raise ValueError(f"skill bank contains forbidden fields: {_forbidden_keys(value)[:8]}")
    fixed_ids = validate_fixed_manifest(manifest_value)
    manifest_hash = sha256_file(fixed_manifest)
    if value.get("fixed_manifest_sha256") != manifest_hash:
        raise ValueError("skill bank fixed-manifest hash mismatch")

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
        if task_id not in set(fixed_ids):
            raise ValueError(f"{task_id}: outside corrected fixed-40 cohort")
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
    outside = [task_id for task_id in requested if task_id not in set(fixed_ids)]
    if missing or outside:
        raise ValueError(f"June 24 coverage failure: missing={missing}, outside_fixed40={outside}")

    return June24SkillSummaryBank(
        path=path,
        sha256=sha256_file(path),
        fixed_manifest_path=fixed_manifest,
        fixed_manifest_sha256=manifest_hash,
        fixed_task_ids=tuple(fixed_ids),
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
