import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.seed_trainer._common.bfcl_checkpoint_supervisor import (
    _last_training_progress,
    archive_uncheckpointed_evidence,
    checkpoint_step,
    run_checkpoint_segments,
)
from examples.seed_trainer.collect_bfcl_opsd_five_pass_evidence import (
    _checkpoint_evidence,
    _segment_evidence,
)


def test_live_progress_reader_extracts_latest_tqdm_record(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_bytes(
        b"setup\n\x1b[36m(TaskRunner pid=1)\x1b[0m \rTraining Progress:   1%| 1/10\r"
        b"Training Progress:  20%| 2/10 [01:23<05:00]\n"
    )

    assert _last_training_progress(log) == "Training Progress:  20%| 2/10 [01:23<05:00]"


def test_live_progress_reader_returns_none_before_training(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("initializing validation\n", encoding="utf-8")

    assert _last_training_progress(log) is None


def _write_checkpoint(root: Path, step: int, *, include_full_model: bool = True) -> None:
    actor = root / f"global_step_{step}" / "actor"
    adapter = actor / "lora_adapter"
    adapter.mkdir(parents=True)
    (root / f"global_step_{step}" / "data.pt").write_bytes(b"data")
    if include_full_model:
        (actor / "model_world_size_1_rank_0.pt").write_bytes(b"model")
    (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"optim")
    (actor / "extra_state_world_size_1_rank_0.pt").write_bytes(b"extra")
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")


def test_checkpoint_step_rejects_partial_atomic_state(tmp_path: Path):
    root = tmp_path / "checkpoints"
    (root / "global_step_40").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without an atomic latest marker"):
        checkpoint_step(root)


def test_checkpoint_step_accepts_lora_only_resumable_state(tmp_path: Path):
    root = tmp_path / "checkpoints"
    _write_checkpoint(root, 40, include_full_model=False)

    assert checkpoint_step(root) == 40


def test_checkpoint_step_rejects_missing_optimizer_state(tmp_path: Path):
    root = tmp_path / "checkpoints"
    _write_checkpoint(root, 40, include_full_model=False)
    (root / "global_step_40" / "actor" / "optim_world_size_1_rank_0.pt").unlink()

    with pytest.raises(RuntimeError, match="optim_world_size"):
        checkpoint_step(root)


def test_completion_evidence_accepts_lora_only_checkpoints(tmp_path: Path):
    root = tmp_path / "checkpoints"
    for step in (40, 80, 120, 160, 200):
        _write_checkpoint(root, step, include_full_model=False)
        adapter = root / f"global_step_{step}" / "actor" / "lora_adapter"
        (adapter / "adapter_model.safetensors").write_bytes(f"lora-{step}".encode())

    evidence = _checkpoint_evidence(root)

    assert [item["global_step"] for item in evidence] == [40, 80, 120, 160, 200]
    assert all(item["resumable"] for item in evidence)
    assert all(item["resume_model_source"] == "pinned_base_plus_lora_adapter" for item in evidence)
    assert not any(item["full_model_shard_present"] for item in evidence)


def test_segment_supervisor_resumes_at_40_and_reaches_200(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root, 40)
    calls = []

    def fake_run(command, **kwargs):
        before = checkpoint_step(checkpoint_root)
        calls.append((list(command), before, kwargs))
        _write_checkpoint(checkpoint_root, before + 40)
        return SimpleNamespace(returncode=0)

    history = run_checkpoint_segments(
        command=["python", "trainer.py"],
        cwd=tmp_path,
        log_path=tmp_path / "train.log",
        checkpoint_root=checkpoint_root,
        history_path=tmp_path / "segment_history.json",
        total_updates=200,
        updates_per_segment=40,
        seed_sha="abc123",
        existing_checkpoint_seed_sha="original456",
        run_process=fake_run,
    )

    assert [before for _, before, _ in calls] == [40, 80, 120, 160]
    assert checkpoint_step(checkpoint_root) == 200
    assert history[0]["kind"] == "adopted_existing_checkpoint"
    assert history[0]["seed_sha"] == "original456"
    assert [item["checkpoint_after"] for item in history[1:]] == [80, 120, 160, 200]


def test_segment_supervisor_fails_when_process_does_not_advance(tmp_path: Path):
    with pytest.raises(RuntimeError, match="without its exact checkpoint"):
        run_checkpoint_segments(
            command=["python", "trainer.py"],
            cwd=tmp_path,
            log_path=tmp_path / "train.log",
            checkpoint_root=tmp_path / "checkpoints",
            history_path=tmp_path / "segment_history.json",
            total_updates=40,
            updates_per_segment=40,
            seed_sha="abc123",
            run_process=lambda *args, **kwargs: SimpleNamespace(returncode=0),
        )


def test_archive_uncheckpointed_evidence_preserves_checkpointed_rows(tmp_path: Path):
    evidence = tmp_path / "evidence"
    for directory in ("updates", "rollouts", "validation"):
        (evidence / directory).mkdir(parents=True)
    (evidence / "updates" / "step_000080.json").write_text("{}\n", encoding="utf-8")
    (evidence / "updates" / "step_000081.json").write_text("{}\n", encoding="utf-8")
    (evidence / "rollouts" / "80.jsonl").write_text("{}\n", encoding="utf-8")
    (evidence / "rollouts" / "115.jsonl").write_text("{}\n", encoding="utf-8")
    (evidence / "validation" / "global_step_000080.json").write_text("{}\n", encoding="utf-8")
    (evidence / "validation" / "global_step_000120.json").write_text("{}\n", encoding="utf-8")
    (evidence / "trainable_checksum.json").write_text(
        json.dumps({"global_step": 115}), encoding="utf-8"
    )

    record = archive_uncheckpointed_evidence(
        evidence_root=evidence,
        checkpoint=80,
        timestamp_utc="20260720T120000Z",
    )

    assert record is not None
    assert len(record["moved"]) == 4
    assert (evidence / "updates" / "step_000080.json").is_file()
    assert (evidence / "rollouts" / "80.jsonl").is_file()
    assert (evidence / "validation" / "global_step_000080.json").is_file()
    archive = evidence / "recovery_archive" / "uncheckpointed_after_step_80_20260720T120000Z"
    assert (archive / "updates" / "step_000081.json").is_file()
    assert (archive / "rollouts" / "115.jsonl").is_file()
    assert (archive / "validation" / "global_step_000120.json").is_file()
    assert (archive / "trainable_checksum.json").is_file()
    assert (archive / "manifest.json").is_file()


def test_segment_supervisor_accepts_ten_update_recovery_boundaries(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root, 80)
    calls = []

    def fake_run(command, **kwargs):
        before = checkpoint_step(checkpoint_root)
        calls.append(before)
        _write_checkpoint(checkpoint_root, before + 10)
        return SimpleNamespace(returncode=0)

    history = run_checkpoint_segments(
        command=["python", "trainer.py"],
        cwd=tmp_path,
        log_path=tmp_path / "train.log",
        checkpoint_root=checkpoint_root,
        history_path=tmp_path / "segment_history.json",
        total_updates=120,
        updates_per_segment=10,
        seed_sha="recovery",
        run_process=fake_run,
    )

    assert calls == [80, 90, 100, 110]
    assert checkpoint_step(checkpoint_root) == 120
    assert [item["checkpoint_after"] for item in history if item["kind"] == "training_segment"] == [
        90,
        100,
        110,
        120,
    ]


def test_segment_supervisor_extends_checkpoint10_through_checkpoint20(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root, 10, include_full_model=False)
    calls = []

    def fake_run(command, **kwargs):
        before = checkpoint_step(checkpoint_root)
        calls.append(before)
        _write_checkpoint(checkpoint_root, before + 1, include_full_model=False)
        return SimpleNamespace(returncode=0)

    history = run_checkpoint_segments(
        command=["python", "trainer.py"],
        cwd=tmp_path,
        log_path=tmp_path / "train.log",
        checkpoint_root=checkpoint_root,
        history_path=tmp_path / "segment_history.json",
        total_updates=20,
        updates_per_segment=1,
        seed_sha="extension",
        existing_checkpoint_seed_sha="checkpoint10",
        run_process=fake_run,
        post_segment_command=[sys.executable, "-c", "pass"],
    )

    assert calls == list(range(10, 20))
    assert checkpoint_step(checkpoint_root) == 20
    completed = [item for item in history if item["kind"] == "training_segment"]
    assert [item["checkpoint_after"] for item in completed] == list(range(11, 21))
    assert all(item["postprocess_returncode"] == 0 for item in completed)
    assert all(item["duration_seconds"] >= 0 for item in completed)


def test_segment_evidence_accepts_recovered_initial_checkpoint(tmp_path: Path):
    path = tmp_path / "segment_history.json"
    segments = [
        {
            "kind": "adopted_existing_checkpoint",
            "checkpoint_before": 40,
            "checkpoint_after": 40,
            "seed_sha": "original",
        }
    ]
    segments.append(
        {
            "kind": "training_segment",
            "checkpoint_before": 40,
            "expected_checkpoint": 80,
            "checkpoint_after": 40,
            "returncode": 1,
            "seed_sha": "recovery",
        }
    )
    for before in (40, 80, 120, 160):
        segments.append(
            {
                "kind": "training_segment",
                "checkpoint_before": before,
                "expected_checkpoint": before + 40,
                "checkpoint_after": before + 40,
                "returncode": 0,
                "seed_sha": "recovery",
            }
        )
    path.write_text(
        json.dumps({"schema_version": "seed.bfcl.segment_history.v1", "segments": segments}),
        encoding="utf-8",
    )

    evidence = _segment_evidence(path, seed_sha_history=["original", "recovery"])
    assert len(evidence["segments"]) == 6
    assert evidence["validation"]["failed_attempt_count"] == 1


def test_segment_evidence_accepts_mixed_legacy_and_recovery_intervals(tmp_path: Path):
    path = tmp_path / "segment_history.json"
    boundaries = [(0, 40), (40, 80), (80, 90), (90, 100), (100, 110), (110, 120), (120, 160), (160, 200)]
    segments = [
        {
            "kind": "training_segment",
            "checkpoint_before": before,
            "expected_checkpoint": after,
            "checkpoint_after": after,
            "returncode": 0,
            "seed_sha": "recovery",
        }
        for before, after in boundaries
    ]
    path.write_text(
        json.dumps({"schema_version": "seed.bfcl.segment_history.v1", "segments": segments}),
        encoding="utf-8",
    )

    evidence = _segment_evidence(
        path,
        seed_sha_history=["recovery"],
        checkpoint_updates=10,
    )

    assert evidence["validation"]["checkpoint_updates"] == 10
    assert evidence["validation"]["iteration_checkpoint_coverage"] == [40, 80, 120, 160, 200]
