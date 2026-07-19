import csv
import json
from pathlib import Path

import pytest

from seed.june24_skill_summary import (
    build_fixed_manifest,
    load_skill_bank,
    materialize_skill_bank,
    validate_fixed_manifest,
)


def _evidence(tmp_path: Path):
    fixed_ids = [f"multi_turn_base_{index}" for index in range(40)]
    rejected_ids = fixed_ids + [f"multi_turn_base_{index}" for index in range(40, 75)]
    fixed_csv = tmp_path / "fixed_tasks.csv"
    with fixed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "classification", "baseline_valid", "skill_valid"],
        )
        writer.writeheader()
        for task_id in fixed_ids:
            writer.writerow(
                {
                    "task_id": task_id,
                    "classification": "fixed",
                    "baseline_valid": "False",
                    "skill_valid": "True",
                }
            )
    rejected = tmp_path / "teacher_success_75.json"
    rejected.write_text(json.dumps({"task_ids": rejected_ids}), encoding="utf-8")
    manifest = tmp_path / "fixed40.json"
    manifest.write_text(
        json.dumps(build_fixed_manifest(fixed_csv, rejected), indent=2) + "\n",
        encoding="utf-8",
    )
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    for index, task_id in enumerate(fixed_ids[:4]):
        target = (source_a if index < 2 else source_b) / f"skill_summary_call_{task_id}.json"
        target.write_text(
            json.dumps(
                {
                    "parsed_skill": {
                        "success_analysis": f"success {task_id}",
                        "mistake_analysis": f"mistake {task_id}",
                        "golden_workflow": f"workflow {task_id}",
                    },
                    "ignored_call_metadata": {"request_id": task_id},
                }
            ),
            encoding="utf-8",
        )
    return fixed_ids, rejected, manifest, source_a, source_b


def test_materializes_only_ordered_three_field_fixed4_with_source_hashes(tmp_path):
    fixed_ids, _, manifest, source_a, source_b = _evidence(tmp_path)
    bank_path = tmp_path / "bank.json"
    value = materialize_skill_bank(
        source_dirs=[source_a, source_b],
        fixed_manifest=manifest,
        output_path=bank_path,
        selected_task_ids=fixed_ids[:4],
    )
    assert value["task_ids"] == fixed_ids[:4]
    assert all(
        set(record)
        == {
            "task_id",
            "success_analysis",
            "mistake_analysis",
            "golden_workflow",
            "source_call_path",
            "source_call_sha256",
        }
        for record in value["records"]
    )
    loaded = load_skill_bank(
        bank_path,
        fixed_manifest=manifest,
        required_task_ids=fixed_ids[:4],
        verify_source_files=True,
    )
    assert list(loaded.records) == fixed_ids[:4]


def test_rejects_v2r_or_structured_fields_in_selected_payload(tmp_path):
    fixed_ids, _, manifest, source_a, source_b = _evidence(tmp_path)
    source = source_a / f"skill_summary_call_{fixed_ids[0]}.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    value["parsed_skill"]["predicted_tool_calls"] = [{"name": "forbidden"}]
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden fields"):
        materialize_skill_bank(
            source_dirs=[source_a, source_b],
            fixed_manifest=manifest,
            output_path=tmp_path / "bank.json",
            selected_task_ids=fixed_ids[:4],
        )


def test_rejects_75_task_manifest_as_cohort_authority(tmp_path):
    _, rejected, _, _, _ = _evidence(tmp_path)
    with pytest.raises(ValueError, match="fixed-40"):
        validate_fixed_manifest(json.loads(rejected.read_text(encoding="utf-8")))


def test_recomputes_source_hashes_fail_closed(tmp_path):
    fixed_ids, _, manifest, source_a, source_b = _evidence(tmp_path)
    bank_path = tmp_path / "bank.json"
    materialize_skill_bank(
        source_dirs=[source_a, source_b],
        fixed_manifest=manifest,
        output_path=bank_path,
        selected_task_ids=fixed_ids[:4],
    )
    source = source_a / f"skill_summary_call_{fixed_ids[0]}.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_skill_bank(
            bank_path,
            fixed_manifest=manifest,
            required_task_ids=fixed_ids[:4],
            verify_source_files=True,
        )
