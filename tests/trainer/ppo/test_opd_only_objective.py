import ast
from pathlib import Path

import torch

from verl.workers.actor.dp_actor import compose_actor_objective


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_standalone_function(relative_path, function_name):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), relative_path, "exec"),
        namespace,
    )
    return namespace[function_name]


def test_opd_only_has_zero_policy_gradient_contribution():
    policy_parameter = torch.tensor(2.0, requires_grad=True)
    opd_parameter = torch.tensor(3.0, requires_grad=True)
    policy_objective = policy_parameter.square()
    opd_loss = opd_parameter.square()
    objective = compose_actor_objective(
        policy_objective=policy_objective,
        opd_loss=opd_loss,
        opd_loss_coef=1.0,
        opd_only=True,
    )
    policy_grad, opd_grad = torch.autograd.grad(
        objective,
        (policy_parameter, opd_parameter),
        allow_unused=True,
    )
    assert policy_grad is None
    assert torch.equal(opd_grad, torch.tensor(6.0))


def test_joint_mode_retains_policy_gradient_for_upstream_compatibility():
    policy_parameter = torch.tensor(2.0, requires_grad=True)
    opd_parameter = torch.tensor(3.0, requires_grad=True)
    objective = compose_actor_objective(
        policy_objective=policy_parameter.square(),
        opd_loss=opd_parameter.square(),
        opd_loss_coef=0.5,
        opd_only=False,
    )
    policy_grad, opd_grad = torch.autograd.grad(objective, (policy_parameter, opd_parameter))
    assert torch.equal(policy_grad, torch.tensor(4.0))
    assert torch.equal(opd_grad, torch.tensor(3.0))


def test_old_log_prob_entropy_is_skipped_only_for_zero_entropy_opd_only():
    should_calculate = _load_standalone_function(
        "verl/workers/fsdp_workers.py",
        "_should_calculate_actor_entropy",
    )

    assert should_calculate(opd_only=True, entropy_coeff=0.0) is False
    assert should_calculate(opd_only=True, entropy_coeff=0.001) is True
    assert should_calculate(opd_only=False, entropy_coeff=0.0) is True
    assert should_calculate(opd_only=False, entropy_coeff=0.001) is True


def test_trainer_accepts_missing_entropy_only_for_zero_entropy_opd_only():
    allows_missing = _load_standalone_function(
        "verl/trainer/ppo/ray_trainer.py",
        "_allows_missing_actor_entropy",
    )

    assert allows_missing(opd_only=True, entropy_coeff=0.0) is True
    assert allows_missing(opd_only=True, entropy_coeff=0.001) is False
    assert allows_missing(opd_only=False, entropy_coeff=0.0) is False
    assert allows_missing(opd_only=False, entropy_coeff=0.001) is False


def test_bfcl_launcher_releases_rollout_cache_before_actor_rescoring():
    source = (REPO_ROOT / "examples/seed_trainer/run_bfcl_opsd.py").read_text(
        encoding="utf-8"
    )

    assert '"actor_rollout_ref.rollout.free_cache_engine=True"' in source
    assert '"actor_rollout_ref.rollout.free_cache_engine=False"' not in source


def test_bfcl_all200_resume_uses_lora_only_model_restore():
    launcher = (REPO_ROOT / "examples/seed_trainer/run_bfcl_opsd.py").read_text(
        encoding="utf-8"
    )
    manager = (
        REPO_ROOT / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
    ).read_text(encoding="utf-8")

    assert (
        'f"actor_rollout_ref.actor.lora_only_resume={str(all200_mode)}"'
        in launcher
    )
    assert "SEED_LORA_ONLY_CHECKPOINT_LOAD" in manager
    assert "FSDP.summon_full_params(self.model, recurse=True, writeback=True)" in manager
    assert "self.optimizer.load_state_dict(optimizer_state_dict)" in manager
    assert "self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)" in manager
