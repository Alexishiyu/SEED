"""Validate the paired June 24/control smoke and write one completion report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "actor/opd_loss",
    "actor/opd_gate_mean",
    "actor/opd_teacher_gap_mean",
    "actor/opd_active_token_ratio",
    "actor/grad_norm",
    "actor/rl_gradient_contribution",
    "seed/teacher_active_tokens",
    "seed/teacher_student_token_alignment",
    "seed/generated_token_only_mask",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float] = {}
    for key in METRIC_KEYS:
        pattern = re.compile(rf"['\"]?{re.escape(key)}['\"]?\s*[:=]\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
        matches = pattern.findall(text)
        if matches:
            values[key] = float(matches[-1])
    missing = [key for key in METRIC_KEYS if key not in values]
    if missing:
        raise RuntimeError(f"training log {path} lacks required metrics: {missing}")
    nonfinite = {key: value for key, value in values.items() if not math.isfinite(value)}
    if nonfinite:
        raise RuntimeError(f"non-finite training metrics in {path}: {nonfinite}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--before-eval", type=Path, required=True)
    parser.add_argument("--after-eval", type=Path, required=True)
    parser.add_argument("--export-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control-tolerance",
        type=float,
        default=1e-2,
        help=(
            "Maximum same-prompt teacher/student mean log-prob gap. The SEED "
            "vLLM-rollout/FSDP-rescore pair is not bitwise identical; 1e-2 is "
            "the recorded backend tolerance for this smoke."
        ),
    )
    parser.add_argument(
        "--minimum-privileged-separation",
        type=float,
        default=5e-2,
        help="Minimum absolute privileged-versus-control teacher-gap separation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    privileged_log = run_root / "logs" / "privileged_june24_train.log"
    control_log = run_root / "logs" / "same_prompt_control_train.log"
    privileged_metrics = _metrics(privileged_log)
    control_metrics = _metrics(control_log)
    privileged_checksum = _json(
        run_root / "evidence" / "privileged_june24" / "trainable_checksum.json"
    )
    control_checksum = _json(
        run_root / "evidence" / "same_prompt_control" / "trainable_checksum.json"
    )
    if privileged_checksum.get("changed") is not True:
        raise RuntimeError("privileged June 24 arm did not change the LoRA checksum")
    for label, metrics in (("privileged", privileged_metrics), ("control", control_metrics)):
        if metrics["actor/rl_gradient_contribution"] != 0.0:
            raise RuntimeError(f"{label} arm has an RL gradient contribution")
        if metrics["seed/teacher_student_token_alignment"] != 1.0:
            raise RuntimeError(f"{label} arm failed response-token alignment")
        if metrics["seed/generated_token_only_mask"] != 1.0:
            raise RuntimeError(f"{label} arm did not use generated-token-only masking")
        if metrics["seed/teacher_active_tokens"] <= 0:
            raise RuntimeError(f"{label} arm has no active teacher tokens")
    privileged_gap = privileged_metrics["actor/opd_teacher_gap_mean"]
    control_gap = control_metrics["actor/opd_teacher_gap_mean"]
    if abs(control_gap) > args.control_tolerance:
        raise RuntimeError(
            f"same-prompt control gap {control_gap} exceeds tolerance {args.control_tolerance}"
        )
    privileged_control_separation = abs(privileged_gap - control_gap)
    if privileged_control_separation < args.minimum_privileged_separation:
        raise RuntimeError(
            "privileged June 24 context is not measurably separated from the "
            f"same-prompt control: separation={privileged_control_separation}, "
            f"minimum={args.minimum_privileged_separation}"
        )

    before = _json(args.before_eval.expanduser().resolve())
    after = _json(args.after_eval.expanduser().resolve())
    export = _json(args.export_evidence.expanduser().resolve())
    if not export.get("checkpoint_resumable"):
        raise RuntimeError("SEED checkpoint is not resumable")
    if not (export.get("vllm_reload") or {}).get("ok"):
        raise RuntimeError("merged Hugging Face export did not reload in vLLM")
    for label, evaluation in (("before", before), ("after", after)):
        if not isinstance(evaluation.get("aggregate"), dict):
            raise RuntimeError(f"official {label} BFCL aggregate is missing")

    privileged_provenance = _json(
        run_root / "metadata" / "privileged_june24_provenance.json"
    )
    control_provenance = _json(
        run_root / "metadata" / "same_prompt_control_provenance.json"
    )
    report = {
        "status": "complete",
        "run_root": str(run_root),
        "task_ids": privileged_provenance["task_ids"],
        "provenance": {
            "seed_sha": privileged_provenance["seed_sha"],
            "rlpaper_sha": privileged_provenance["rlpaper_sha"],
            "bfcl_sha": privileged_provenance["bfcl_sha"],
            "skill_bank_sha256": privileged_provenance["skill_bank_sha256"],
            "fixed_manifest_sha256": privileged_provenance["fixed_manifest_sha256"],
            "source_calls": privileged_provenance["source_calls"],
        },
        "privileged": {
            "metrics": privileged_metrics,
            "trainable_checksum": privileged_checksum,
        },
        "same_prompt_control": {
            "metrics": control_metrics,
            "trainable_checksum": control_checksum,
            "gap_tolerance": args.control_tolerance,
        },
        "comparison": {
            "privileged_teacher_gap_mean": privileged_gap,
            "control_teacher_gap_mean": control_gap,
            "privileged_minus_control": privileged_gap - control_gap,
            "absolute_teacher_gap_separation": privileged_control_separation,
            "minimum_privileged_separation": args.minimum_privileged_separation,
            "privileged_gap_direction": (
                "higher_sampled_token_log_prob"
                if privileged_gap > control_gap
                else "lower_sampled_token_log_prob"
            ),
            "privileged_opd_loss_exceeds_control": (
                privileged_metrics["actor/opd_loss"] > control_metrics["actor/opd_loss"]
            ),
        },
        "checkpoint_and_export": export,
        "official_bfcl_before": before,
        "official_bfcl_after": after,
        "control_provenance_matches": {
            key: control_provenance.get(key) == privileged_provenance.get(key)
            for key in ("seed_sha", "rlpaper_sha", "bfcl_sha", "skill_bank_sha256", "fixed_manifest_sha256", "task_ids")
        },
    }
    if not all(report["control_provenance_matches"].values()):
        raise RuntimeError("privileged and same-prompt control provenance do not match")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SEED_BFCL_SMOKE_COMPLETE report={args.output}")


if __name__ == "__main__":
    main()
