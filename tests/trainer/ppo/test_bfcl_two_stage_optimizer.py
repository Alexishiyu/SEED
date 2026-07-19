import torch

from verl.trainer.ppo.core_algos import compute_opd_loss
from verl.utils.torch_functional import build_torch_optimizer, get_two_stage_linear_schedule


def test_strict_adam_uses_only_trainable_parameters_and_no_weight_decay():
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    optimizer = build_torch_optimizer(
        [frozen, trainable],
        name="adam",
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.0
    assert [parameter for group in optimizer.param_groups for parameter in group["params"]] == [trainable]


def test_two_stage_lr_hits_zero_warmup_target_and_final_value():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-6)
    scheduler = get_two_stage_linear_schedule(
        optimizer,
        num_warmup_steps=67,
        num_training_steps=200,
        warmup_target_lr=1e-7,
        final_lr=1e-6,
    )
    used = []
    for _ in range(200):
        used.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert used[0] == 0.0
    assert used[66] == 1e-7
    assert used[199] == 1e-6
    assert all(left <= right for left, right in zip(used, used[1:]))


def test_two_stage_lr_state_resumes_exactly():
    first_parameter = torch.nn.Parameter(torch.ones(1))
    first_optimizer = torch.optim.Adam([first_parameter], lr=1e-6)
    first_scheduler = get_two_stage_linear_schedule(
        first_optimizer,
        num_warmup_steps=67,
        num_training_steps=200,
        warmup_target_lr=1e-7,
        final_lr=1e-6,
    )
    for _ in range(80):
        first_optimizer.step()
        first_scheduler.step()
    optimizer_state = first_optimizer.state_dict()
    scheduler_state = first_scheduler.state_dict()

    resumed_parameter = torch.nn.Parameter(torch.ones(1))
    resumed_optimizer = torch.optim.Adam([resumed_parameter], lr=1e-6)
    resumed_scheduler = get_two_stage_linear_schedule(
        resumed_optimizer,
        num_warmup_steps=67,
        num_training_steps=200,
        warmup_target_lr=1e-7,
        final_lr=1e-6,
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    assert resumed_scheduler.get_last_lr() == first_scheduler.get_last_lr()
    first_optimizer.step()
    first_scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_scheduler.get_last_lr() == first_scheduler.get_last_lr()


def test_turn_microbatch_accumulation_matches_full_token_mean_opd():
    student_full = torch.tensor(
        [[-2.0, -1.0, -3.0], [-1.5, -2.5, -4.0], [-3.0, -2.0, -1.0], [-2.2, -2.1, -2.0]],
        requires_grad=True,
    )
    teacher = student_full.detach() + 0.25
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
        dtype=torch.float32,
    )
    step_mask = torch.ones(4, dtype=torch.bool)
    full_loss = compute_opd_loss(
        student_full,
        teacher,
        response_mask,
        step_mask,
        gate_beta=5.0,
        loss_agg_mode="token-mean",
    )[0]
    full_loss.backward()
    full_gradient = student_full.grad.detach().clone()

    student_micro = student_full.detach().clone().requires_grad_(True)
    total_active = response_mask.sum()
    for index in range(4):
        micro_loss = compute_opd_loss(
            student_micro[index : index + 1],
            teacher[index : index + 1],
            response_mask[index : index + 1],
            step_mask[index : index + 1],
            gate_beta=5.0,
            loss_agg_mode="token-mean",
        )[0]
        (micro_loss * response_mask[index].sum() / total_active).backward()
    torch.testing.assert_close(student_micro.grad, full_gradient)
