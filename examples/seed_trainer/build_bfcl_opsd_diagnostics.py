"""Build compact BFCL OPD debugging tables, health checks, and plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


_UPDATE_RE = re.compile(r"step_(\d+)\.json$")
_VALIDATION_RE = re.compile(r"global_step_(\d+)\.json$")
_TELEMETRY_NUMERIC_FIELDS = (
    "elapsed_seconds",
    "gpu_memory_used_mib",
    "gpu_memory_free_mib",
    "gpu_utilization_percent",
    "gpu_temperature_c",
    "gpu_power_w",
    "host_memory_available_bytes",
    "drive_disk_free_bytes",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ordered_json_files(root: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    matches = []
    if root.is_dir():
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if match and path.is_file():
                matches.append((int(match.group(1)), path))
    return sorted(matches)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _classification_map(provenance: Mapping[str, Any]) -> dict[str, str]:
    cohort_path = Path(str(provenance["cohort_manifest_path"]))
    records = _json(cohort_path).get("records") or []
    return {
        str(record["task_id"]): str(record["classification"])
        for record in records
        if isinstance(record, Mapping)
    }


def _trainer_scalar_metrics(root: Path) -> dict[int, dict[str, float]]:
    result = {}
    for step, path in _ordered_json_files(root, _UPDATE_RE):
        value = _json(path)
        result[step] = {
            str(key): float(metric)
            for key, metric in (value.get("scalar_metrics") or {}).items()
            if isinstance(metric, (int, float)) and math.isfinite(float(metric))
        }
    return result


def _training_rows(
    evidence_root: Path,
    *,
    provenance: Mapping[str, Any],
    anomalies: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    updates_per_iteration = int(provenance["updates_per_iteration"])
    learning_rate = provenance.get("learning_rate") or {}
    expected_lr = float(learning_rate.get("final_lr") or 0.0)
    trainer_metrics = _trainer_scalar_metrics(evidence_root / "trainer_metrics")
    rows = []
    parsed = []
    for step, path in _ordered_json_files(evidence_root / "updates", _UPDATE_RE):
        value = _json(path)
        parsed.append(value)
        metrics = value.get("metrics") or {}
        row: dict[str, Any] = {
            "global_step": step,
            "iteration": ((step - 1) // updates_per_iteration) + 1,
            "batch_in_iteration": ((step - 1) % updates_per_iteration) + 1,
            "observed_iteration": value.get("iteration"),
            "observed_batch_in_iteration": value.get("batch_in_iteration"),
            "task_count": len(value.get("task_ids") or []),
            "active_token_count": value.get("active_token_count"),
            "learning_rate": value.get("learning_rate_used"),
            "opd_loss": metrics.get("actor/opd_loss"),
            "opd_gate_mean": metrics.get("actor/opd_gate_mean"),
            "teacher_gap_mean": metrics.get("actor/opd_teacher_gap_mean"),
            "control_opd_loss": metrics.get("actor/control_opd_loss"),
            "control_gate_mean": metrics.get("actor/control_opd_gate_mean"),
            "control_teacher_gap_mean": metrics.get("actor/control_opd_teacher_gap_mean"),
            "grad_norm": metrics.get("actor/grad_norm"),
            "rl_gradient_contribution": metrics.get("actor/rl_gradient_contribution"),
            "trainable_changed": value.get("trainable_changed"),
            "trainable_sha256_before": value.get("trainable_sha256_before"),
            "trainable_sha256_after": value.get("trainable_sha256_after"),
        }
        for class_name, class_metrics in (value.get("outcome_class_metrics") or {}).items():
            for metric_name, metric_value in class_metrics.items():
                row[f"{class_name}_{metric_name}"] = metric_value
        task_metrics = value.get("task_token_metrics") or []
        if task_metrics:
            row["task_token_metric_count"] = len(task_metrics)
            row["response_token_count"] = sum(int(item["response_token_count"]) for item in task_metrics)
            row["teacher_step_count"] = sum(int(item["teacher_step_count"]) for item in task_metrics)
        for metric_name, metric_value in trainer_metrics.get(step, {}).items():
            if metric_name.startswith(("timing_s/", "perf/", "training/")):
                row[metric_name.replace("/", "_")] = metric_value
        rows.append(row)

        required = {
            "opd_loss": row["opd_loss"],
            "opd_gate_mean": row["opd_gate_mean"],
            "teacher_gap_mean": row["teacher_gap_mean"],
            "grad_norm": row["grad_norm"],
            "active_token_count": row["active_token_count"],
        }
        for name, metric in required.items():
            if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                anomalies.append(f"step {step}: {name} is not finite")
        if value.get("trainable_changed") is not True:
            anomalies.append(f"step {step}: trainable checksum did not change")
        if value.get("generated_token_only_mask") is not True:
            anomalies.append(f"step {step}: generated-token-only mask is false")
        if value.get("teacher_student_token_alignment") is not True:
            anomalies.append(f"step {step}: privileged token alignment is false")
        if value.get("control_teacher_student_token_alignment") is not True:
            anomalies.append(f"step {step}: control token alignment is false")
        if float(row.get("rl_gradient_contribution") or 0.0) != 0.0:
            anomalies.append(f"step {step}: nonzero RL gradient contribution")
        if learning_rate.get("style") == "constant" and not math.isclose(
            float(row["learning_rate"]), expected_lr, rel_tol=1e-9, abs_tol=1e-12
        ):
            anomalies.append(f"step {step}: learning rate differs from constant {expected_lr}")
        expected_coordinate = (row["iteration"], row["batch_in_iteration"])
        observed_coordinate = (row["observed_iteration"], row["observed_batch_in_iteration"])
        if observed_coordinate != expected_coordinate:
            warnings.append(
                f"step {step}: legacy raw coordinate {observed_coordinate} normalized to {expected_coordinate}"
            )

    observed_steps = [int(row["global_step"]) for row in rows]
    if observed_steps and observed_steps != list(range(1, max(observed_steps) + 1)):
        anomalies.append(f"non-contiguous update evidence: {observed_steps}")
    for previous, current in zip(parsed, parsed[1:]):
        if previous.get("trainable_sha256_after") != current.get("trainable_sha256_before"):
            anomalies.append(
                f"checksum discontinuity between steps {previous.get('global_step')} and {current.get('global_step')}"
            )
    return rows


def _validation_rows(
    evidence_root: Path,
    *,
    classification_by_id: Mapping[str, str],
    expected_task_count: int,
    anomalies: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = []
    parsed = []
    initial_success: set[str] = set()
    previous_success: set[str] = set()
    for index, (step, path) in enumerate(
        _ordered_json_files(evidence_root / "validation", _VALIDATION_RE)
    ):
        value = _json(path)
        records = value.get("records") or []
        parsed.append((step, records))
        task_ids = [str(record.get("task_id") or "") for record in records]
        success = {str(record["task_id"]) for record in records if record.get("success")}
        if index == 0:
            initial_success = set(success)
            previous_success = set(success)
        tool_calls = [int(record.get("tool_call_count") or 0) for record in records]
        row: dict[str, Any] = {
            "global_step": step,
            "task_count": len(records),
            "success_count": len(success),
            "accuracy": len(success) / len(records) if records else 0.0,
            "tool_calls_mean": statistics.fmean(tool_calls) if tool_calls else 0.0,
            "tool_calls_median": statistics.median(tool_calls) if tool_calls else 0.0,
            "tool_calls_p90": _percentile(tool_calls, 0.9),
            "tool_calls_max": max(tool_calls) if tool_calls else 0,
            "gained_from_initial": len(success - initial_success),
            "lost_from_initial": len(initial_success - success),
            "gained_from_previous": len(success - previous_success),
            "lost_from_previous": len(previous_success - success),
        }
        for class_name in sorted(set(classification_by_id.values())):
            selected = [
                record
                for record in records
                if classification_by_id.get(str(record.get("task_id"))) == class_name
            ]
            row[f"{class_name}_task_count"] = len(selected)
            row[f"{class_name}_success_count"] = sum(bool(record.get("success")) for record in selected)
            row[f"{class_name}_accuracy"] = (
                row[f"{class_name}_success_count"] / len(selected) if selected else 0.0
            )
        points.append(row)
        previous_success = set(success)

        if len(records) != expected_task_count or len(set(task_ids)) != expected_task_count:
            anomalies.append(f"validation step {step}: expected {expected_task_count} unique tasks")
        if value.get("prompt_mode") != "ordinary_bfcl" or value.get("privileged_context") is not False:
            anomalies.append(f"validation step {step}: prompt contract is not ordinary BFCL")

    all_task_ids = sorted(
        {str(record.get("task_id")) for _, records in parsed for record in records}
    )
    transitions = []
    for task_id in all_task_ids:
        row: dict[str, Any] = {
            "task_id": task_id,
            "outcome_class": classification_by_id.get(task_id),
        }
        sequence = []
        for step, records in parsed:
            record = next((item for item in records if str(item.get("task_id")) == task_id), None)
            if record is None:
                row[f"success_step_{step}"] = ""
                row[f"tool_calls_step_{step}"] = ""
                continue
            value = bool(record.get("success"))
            sequence.append((step, value))
            row[f"success_step_{step}"] = int(value)
            row[f"tool_calls_step_{step}"] = int(record.get("tool_call_count") or 0)
        row["success_flip_count"] = sum(
            left[1] != right[1] for left, right in zip(sequence, sequence[1:])
        )
        row["first_changed_step"] = next(
            (step for step, value in sequence[1:] if value != sequence[0][1]),
            None,
        ) if sequence else None
        transitions.append(row)
    return points, transitions


def _segment_rows(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "metadata" / "privileged_june24_segment_history.json"
    if not path.is_file():
        return []
    rows = []
    for item in _json(path).get("segments") or []:
        if item.get("kind") == "training_segment":
            rows.append(dict(item))
    return rows


def _telemetry_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"telemetry line {line_number} is not an object")
        rows.append(value)
    return rows


def _plot_training(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["global_step"] for row in rows]
    fields = [
        ("opd_loss", "OPD loss"),
        ("opd_gate_mean", "OPD gate mean"),
        ("teacher_gap_mean", "Teacher gap mean"),
        ("grad_norm", "Gradient norm"),
        ("active_token_count", "Active tokens"),
        ("response_token_count", "Response tokens (new diagnostics)"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, fields):
        values = [row.get(field) for row in rows]
        axis.plot(steps, values, marker="o", linewidth=1.6)
        axis.set_title(title)
        axis.set_xlabel("Checkpoint")
        axis.grid(alpha=0.25)
    figure.suptitle("BFCL OPD training diagnostics")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_validation(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["global_step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [row["accuracy"] for row in rows], marker="o", label="overall")
    class_fields = sorted(
        key for key in rows[0] if key.endswith("_accuracy") and key != "accuracy"
    ) if rows else []
    for field in class_fields:
        axes[0, 1].plot(steps, [row[field] for row in rows], marker="o", label=field.removesuffix("_accuracy"))
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(steps, [row["tool_calls_mean"] for row in rows], marker="o", label="mean")
    axes[1, 0].plot(steps, [row["tool_calls_p90"] for row in rows], marker="o", label="p90")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, [row["gained_from_initial"] for row in rows], marker="o", label="gained")
    axes[1, 1].plot(steps, [row["lost_from_initial"] for row in rows], marker="o", label="lost")
    axes[1, 1].legend()
    for axis, title in zip(
        axes.flat,
        ("Overall accuracy", "Accuracy by outcome class", "Tool calls", "Success transitions vs step 0"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Checkpoint")
        axis.grid(alpha=0.25)
    figure.suptitle("Ordinary-prompt held-out validation")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_transition_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    step_fields = sorted(
        {key for row in rows for key in row if key.startswith("success_step_")},
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    matrix = np.asarray(
        [
            [float(row[field]) if row.get(field) != "" else np.nan for field in step_fields]
            for row in rows
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(max(8, len(step_fields) * 0.9), 12), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(step_fields)), [field.rsplit("_", 1)[1] for field in step_fields])
    axis.set_xlabel("Checkpoint")
    axis.set_ylabel("Validation task (stable task-id order)")
    axis.set_title("Held-out task success transitions")
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="Success")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_runtime(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(x, [row.get("gpu_memory_used_mib") for row in rows], label="used MiB")
    axes[0].plot(x, [row.get("gpu_memory_free_mib") for row in rows], label="free MiB")
    axes[1].plot(x, [row.get("gpu_utilization_percent") for row in rows], label="GPU utilization %")
    axes[1].plot(x, [row.get("gpu_temperature_c") for row in rows], label="temperature C")
    axes[2].plot(x, [row.get("gpu_power_w") for row in rows], label="power W")
    for axis, title in zip(axes, ("GPU memory", "GPU utilization and temperature", "GPU power")):
        axis.set_title(title)
        axis.set_xlabel("Telemetry sample")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Checkpoint-segment runtime telemetry")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def build_diagnostics(
    *,
    run_root: Path,
    arm: str,
    output_dir: Path | None = None,
    require_latest_checkpoint: int | None = None,
    fail_on_anomaly: bool = False,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    evidence_root = run_root / "evidence" / arm
    output_dir = (output_dir or evidence_root / "diagnostics").expanduser().resolve()
    provenance = _json(run_root / "metadata" / f"{arm}_provenance.json")
    marker_path = run_root / "checkpoints" / arm / "latest_checkpointed_iteration.txt"
    latest_checkpoint = int(marker_path.read_text(encoding="utf-8").strip())
    if require_latest_checkpoint is not None and latest_checkpoint != require_latest_checkpoint:
        raise RuntimeError(
            f"expected checkpoint {require_latest_checkpoint}, observed {latest_checkpoint}"
        )

    anomalies: list[str] = []
    warnings: list[str] = []
    classification_by_id = _classification_map(provenance)
    training = _training_rows(
        evidence_root,
        provenance=provenance,
        anomalies=anomalies,
        warnings=warnings,
    )
    validation, transitions = _validation_rows(
        evidence_root,
        classification_by_id=classification_by_id,
        expected_task_count=len(provenance["validation_task_ids"]),
        anomalies=anomalies,
    )
    segments = _segment_rows(run_root)
    telemetry_path = evidence_root / "runtime_telemetry.jsonl"
    telemetry = _telemetry_rows(telemetry_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "training_metrics_csv": output_dir / "training_metrics.csv",
        "validation_metrics_csv": output_dir / "validation_metrics.csv",
        "validation_task_transitions_csv": output_dir / "validation_task_transitions.csv",
        "segment_metrics_csv": output_dir / "segment_metrics.csv",
        "runtime_telemetry_csv": output_dir / "runtime_telemetry.csv",
        "training_plot": output_dir / "training_diagnostics.png",
        "validation_plot": output_dir / "validation_diagnostics.png",
        "transition_plot": output_dir / "validation_task_transitions.png",
        "runtime_plot": output_dir / "runtime_resources.png",
    }
    _write_csv(artifacts["training_metrics_csv"], training)
    _write_csv(artifacts["validation_metrics_csv"], validation)
    _write_csv(artifacts["validation_task_transitions_csv"], transitions)
    _write_csv(artifacts["segment_metrics_csv"], segments)
    _write_csv(artifacts["runtime_telemetry_csv"], telemetry)
    if training:
        _plot_training(training, artifacts["training_plot"])
    if validation:
        _plot_validation(validation, artifacts["validation_plot"])
    if transitions and validation:
        _plot_transition_heatmap(transitions, artifacts["transition_plot"])
    if telemetry:
        _plot_runtime(telemetry, artifacts["runtime_plot"])

    existing_artifacts = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in artifacts.items()
        if path.is_file()
    }
    summary = {
        "schema_version": "seed.bfcl.opsd_diagnostics.v1",
        "status": "healthy" if not anomalies else "anomalies-detected",
        "run_root": str(run_root),
        "arm": arm,
        "latest_checkpoint": latest_checkpoint,
        "update_evidence_count": len(training),
        "validation_point_count": len(validation),
        "validation_steps": [row["global_step"] for row in validation],
        "latest_validation": validation[-1] if validation else None,
        "anomalies": sorted(set(anomalies)),
        "warnings": sorted(set(warnings)),
        "artifacts": existing_artifacts,
    }
    summary_path = output_dir / "diagnostic_summary.json"
    _write_json(summary_path, summary)
    if fail_on_anomaly and anomalies:
        raise RuntimeError(f"BFCL diagnostics found anomalies: {sorted(set(anomalies))}")
    print(
        "SEED_BFCL_DIAGNOSTICS_OK "
        f"checkpoint={latest_checkpoint} updates={len(training)} "
        f"validation_points={len(validation)} summary={summary_path}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", default="privileged_june24")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-latest-checkpoint", type=int)
    parser.add_argument("--fail-on-anomaly", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_diagnostics(
        run_root=args.run_root,
        arm=args.arm,
        output_dir=args.output_dir,
        require_latest_checkpoint=args.require_latest_checkpoint,
        fail_on_anomaly=args.fail_on_anomaly,
    )


if __name__ == "__main__":
    main()
