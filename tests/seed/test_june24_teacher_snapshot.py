import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_async_teacher_snapshot_preserves_exact_bfcl_identity():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "step_rewards": torch.tensor([0.0, 1.0]),
        },
        non_tensors={
            "obs_text": np.asarray(["first", "second"], dtype=object),
            "traj_uid": np.asarray(["traj-a", "traj-b"], dtype=object),
            "task_id": np.asarray(["multi_turn_base_16", "multi_turn_base_19"], dtype=object),
            "bfcl_turn_id": np.asarray([0, 2], dtype=np.int64),
            "bfcl_step_in_turn": np.asarray([1, 3], dtype=np.int64),
        },
    )

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    snapshot = trainer._build_seed_teacher_signal_snapshot(batch)

    assert snapshot.non_tensor_batch["task_id"].tolist() == [
        "multi_turn_base_16",
        "multi_turn_base_19",
    ]
    assert snapshot.non_tensor_batch["bfcl_turn_id"].tolist() == [0, 2]
    assert snapshot.non_tensor_batch["bfcl_step_in_turn"].tolist() == [1, 3]

    batch.non_tensor_batch["task_id"][0] = "mutated"
    assert snapshot.non_tensor_batch["task_id"][0] == "multi_turn_base_16"
