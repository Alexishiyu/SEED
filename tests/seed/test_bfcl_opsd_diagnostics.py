import json
from pathlib import Path

from examples.seed_trainer.build_bfcl_opsd_diagnostics import build_diagnostics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_partial_run_diagnostics_write_compact_tables_and_plots(tmp_path: Path):
    run_root = tmp_path / "run"
    cohort = run_root / "inputs" / "cohort.json"
    _write_json(
        cohort,
        {
            "records": [
                {"task_id": "multi_turn_base_1", "classification": "fixed"},
                {"task_id": "multi_turn_base_2", "classification": "both_wrong"},
            ]
        },
    )
    _write_json(
        run_root / "metadata" / "privileged_june24_provenance.json",
        {
            "cohort_manifest_path": str(cohort),
            "validation_task_ids": ["multi_turn_base_2"],
            "updates_per_iteration": 2,
            "learning_rate": {
                "style": "constant",
                "warmup_updates": 0,
                "warmup_target_lr": None,
                "final_lr": 1e-6,
            },
        },
    )
    update = {
        "global_step": 1,
        "iteration": 1,
        "batch_in_iteration": 1,
        "task_ids": ["multi_turn_base_1"],
        "active_token_count": 12,
        "learning_rate_used": 1e-6,
        "trainable_changed": True,
        "trainable_sha256_before": "a" * 64,
        "trainable_sha256_after": "b" * 64,
        "generated_token_only_mask": True,
        "teacher_student_token_alignment": True,
        "control_teacher_student_token_alignment": True,
        "metrics": {
            "actor/opd_loss": 0.01,
            "actor/opd_gate_mean": 0.48,
            "actor/opd_teacher_gap_mean": -0.3,
            "actor/control_opd_loss": 0.0,
            "actor/control_opd_gate_mean": 0.5,
            "actor/control_opd_teacher_gap_mean": 0.0,
            "actor/grad_norm": 0.1,
            "actor/rl_gradient_contribution": 0.0,
        },
        "outcome_class_metrics": {
            "fixed": {
                "opd_loss": 0.01,
                "gate_mean": 0.48,
                "teacher_gap_mean": -0.3,
                "control_opd_loss": 0.0,
                "control_gate_mean": 0.5,
                "control_teacher_gap_mean": 0.0,
            }
        },
    }
    _write_json(
        run_root / "evidence" / "privileged_june24" / "updates" / "step_000001.json",
        update,
    )
    _write_json(
        run_root
        / "evidence"
        / "privileged_june24"
        / "validation"
        / "global_step_000000.json",
        {
            "global_step": 0,
            "prompt_mode": "ordinary_bfcl",
            "privileged_context": False,
            "records": [
                {
                    "task_id": "multi_turn_base_2",
                    "success": False,
                    "tool_call_count": 4,
                }
            ],
        },
    )
    marker = (
        run_root
        / "checkpoints"
        / "privileged_june24"
        / "latest_checkpointed_iteration.txt"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("1", encoding="utf-8")

    summary = build_diagnostics(
        run_root=run_root,
        arm="privileged_june24",
        require_latest_checkpoint=1,
        fail_on_anomaly=True,
    )

    output = run_root / "evidence" / "privileged_june24" / "diagnostics"
    assert summary["status"] == "healthy"
    assert summary["latest_checkpoint"] == 1
    assert (output / "training_metrics.csv").is_file()
    assert (output / "validation_metrics.csv").is_file()
    assert (output / "validation_task_transitions.csv").is_file()
    assert (output / "training_diagnostics.png").is_file()
    assert (output / "validation_diagnostics.png").is_file()
    assert (output / "validation_task_transitions.png").is_file()
