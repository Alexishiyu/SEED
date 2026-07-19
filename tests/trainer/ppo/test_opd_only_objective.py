import torch

from verl.workers.actor.dp_actor import compose_actor_objective


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
