"""Fail-closed process segmentation for long single-GPU BFCL runs."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


_CHECKPOINT_DIR_RE = re.compile(r"global_step_(\d+)$")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def checkpoint_step(checkpoint_root: Path) -> int:
    """Return the latest complete resumable checkpoint, or fail on partial state."""

    checkpoint_root = checkpoint_root.expanduser().resolve()
    marker = checkpoint_root / "latest_checkpointed_iteration.txt"
    observed_dirs: dict[int, Path] = {}
    if checkpoint_root.is_dir():
        for path in checkpoint_root.glob("global_step_*"):
            match = _CHECKPOINT_DIR_RE.fullmatch(path.name)
            if match and path.is_dir():
                observed_dirs[int(match.group(1))] = path

    if not marker.is_file():
        if observed_dirs:
            raise RuntimeError(
                f"checkpoint directories exist without an atomic latest marker: {sorted(observed_dirs)}"
            )
        return 0

    raw = marker.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"(?:global_step_)?(\d+)", raw)
    if match is None:
        raise RuntimeError(f"invalid checkpoint marker {marker}: {raw!r}")
    step = int(match.group(1))
    if step <= 0:
        raise RuntimeError(f"checkpoint marker must be positive: {marker}={step}")
    if step not in observed_dirs:
        raise RuntimeError(f"checkpoint marker points to a missing directory: global_step_{step}")
    if observed_dirs and max(observed_dirs) != step:
        raise RuntimeError(
            "checkpoint directories extend beyond the atomic latest marker: "
            f"marker={step}, directories={sorted(observed_dirs)}"
        )

    root = observed_dirs[step]
    actor = root / "actor"
    required_files = [
        root / "data.pt",
        actor / "lora_adapter" / "adapter_model.safetensors",
        actor / "lora_adapter" / "adapter_config.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    for pattern in (
        "model_world_size_*_rank_*.pt",
        "optim_world_size_*_rank_*.pt",
        "extra_state_world_size_*_rank_*.pt",
    ):
        if not any(actor.glob(pattern)):
            missing.append(str(actor / pattern))
    if missing:
        raise RuntimeError(f"checkpoint global_step_{step} is incomplete; missing={missing}")
    return step


def run_checkpoint_segments(
    *,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    checkpoint_root: Path,
    history_path: Path,
    total_updates: int,
    updates_per_segment: int,
    seed_sha: str,
    existing_checkpoint_seed_sha: str | None = None,
    run_process: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one checkpoint-bounded trainer process at a time until completion."""

    if total_updates <= 0 or updates_per_segment <= 0:
        raise ValueError("training update counts must be positive")
    if total_updates % updates_per_segment:
        raise ValueError("total_updates must be divisible by updates_per_segment")

    run_process = run_process or subprocess.run
    allowed_steps = set(range(0, total_updates + 1, updates_per_segment))
    current = checkpoint_step(checkpoint_root)
    if current not in allowed_steps:
        raise RuntimeError(
            f"checkpoint {current} is not a canonical segment boundary; expected {sorted(allowed_steps)}"
        )

    if history_path.is_file():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "seed.bfcl.segment_history.v1":
            raise RuntimeError(f"invalid segment history: {history_path}")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError(f"invalid segment list: {history_path}")
    else:
        payload = {"schema_version": "seed.bfcl.segment_history.v1", "segments": []}
        segments = payload["segments"]

    if current and not any(item.get("checkpoint_after") == current for item in segments):
        segments.append(
            {
                "kind": "adopted_existing_checkpoint",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_before": current,
                "checkpoint_after": current,
                "seed_sha": existing_checkpoint_seed_sha or seed_sha,
            }
        )
        _write_json(history_path, payload)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    while current < total_updates:
        expected = current + updates_per_segment
        started_at = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                "\nSEED_BFCL_SEGMENT_START "
                f"checkpoint_before={current} expected_checkpoint={expected} seed_sha={seed_sha}\n"
            )
            log.flush()
            result = run_process(
                list(command),
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        after = checkpoint_step(checkpoint_root)
        record = {
            "kind": "training_segment",
            "timestamp_utc": started_at,
            "checkpoint_before": current,
            "expected_checkpoint": expected,
            "checkpoint_after": after,
            "returncode": int(result.returncode),
            "seed_sha": seed_sha,
        }
        segments.append(record)
        _write_json(history_path, payload)
        if result.returncode != 0:
            raise RuntimeError(
                f"SEED training segment {current}->{expected} failed with exit code "
                f"{result.returncode}; checkpoint_after={after}; see {log_path}"
            )
        if after != expected:
            raise RuntimeError(
                f"SEED training segment exited without its exact checkpoint: "
                f"before={current}, expected={expected}, after={after}"
            )
        print(
            f"SEED_BFCL_SEGMENT_DONE checkpoint_before={current} "
            f"checkpoint_after={after} log={log_path}",
            flush=True,
        )
        current = after

    return segments
