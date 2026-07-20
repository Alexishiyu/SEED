import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.seed_trainer._common.bfcl_checkpoint_supervisor import (
    checkpoint_step,
    run_checkpoint_segments,
)
from examples.seed_trainer.collect_bfcl_opsd_five_pass_evidence import _segment_evidence


def _write_checkpoint(root: Path, step: int) -> None:
    actor = root / f"global_step_{step}" / "actor"
    adapter = actor / "lora_adapter"
    adapter.mkdir(parents=True)
    (root / f"global_step_{step}" / "data.pt").write_bytes(b"data")
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
