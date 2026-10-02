"""SAM invariants: update equivalence, perturbation radius, and safe restoration."""

import copy

import pytest
import torch
from torch import nn

from beautyrec.optimizers import SAM


def test_rho_zero_matches_multiple_adamw_updates():
    """Two deterministic passes at rho=0 must be exactly an ordinary AdamW step."""
    torch.manual_seed(11)
    model = nn.Linear(3, 2).double()
    reference = copy.deepcopy(model)
    options = {"lr": 0.01, "weight_decay": 0.1}
    optimizer = SAM(torch.optim.AdamW(model.parameters(), **options), rho=0)
    adamw = torch.optim.AdamW(reference.parameters(), **options)
    x, y = torch.randn(4, 3, dtype=torch.double), torch.randn(4, 2, dtype=torch.double)
    for _ in range(3):
        adamw.zero_grad()
        (reference(x) - y).square().mean().backward()
        adamw.step()
        optimizer.zero_grad()
        (model(x) - y).square().mean().backward()
        optimizer.first_step()
        (model(x) - y).square().mean().backward()
        optimizer.second_step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not optimizer.pending


def test_global_perturbation_radius_and_exact_restoration():
    first = nn.Parameter(torch.tensor([1e-8, -1e-8], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
    unused = nn.Parameter(torch.tensor([2.0]))
    original = [parameter.detach().clone() for parameter in (first, second, unused)]
    base = torch.optim.AdamW([{"params": [first]}, {"params": [second, unused]}])
    optimizer = SAM(base, rho=0.05)
    first.grad, second.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
    assert optimizer.first_step() == pytest.approx(5.0)
    delta = torch.cat(
        [(parameter - before).detach() for parameter, before in zip((first, second), original)]
    )
    assert float(delta.norm()) == pytest.approx(0.05, abs=1e-7)
    assert torch.equal(unused, original[2])
    optimizer.restore()
    optimizer.restore()  # Safe even after cleanup has already happened.
    assert not optimizer.pending
    assert not optimizer.base_optimizer.state
    for parameter, before in zip((first, second, unused), original):
        assert torch.equal(parameter, before)
        assert parameter.grad is None


def test_base_optimizer_uses_original_parameters_and_second_gradient():
    parameter = nn.Parameter(torch.tensor([2.0]))
    unused = nn.Parameter(torch.tensor([7.0]))
    base = torch.optim.AdamW([parameter, unused], lr=0.01, weight_decay=0.1)
    optimizer = SAM(base)
    assert optimizer.param_groups is base.param_groups
    assert optimizer.param_groups[0]["params"][0] is parameter
    expected = nn.Parameter(parameter.detach().clone())
    reference = torch.optim.AdamW([expected], lr=0.01, weight_decay=0.1)
    parameter.grad = torch.tensor([1.0])
    optimizer.first_step()
    parameter.grad = torch.tensor([-3.0])
    expected.grad = parameter.grad.clone()
    reference.step()
    optimizer.second_step()
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert torch.equal(unused, torch.tensor([7.0]))
    assert parameter in base.state
    assert not optimizer.pending


@pytest.mark.parametrize("gradient", [None, torch.zeros(2)])
def test_missing_or_zero_first_gradient_leaves_no_perturbation(gradient):
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    before = parameter.detach().clone()
    optimizer = SAM(torch.optim.AdamW([parameter], weight_decay=0))
    parameter.grad = gradient
    assert optimizer.first_step() == 0
    assert optimizer.pending
    assert torch.equal(parameter, before)
    optimizer.second_step()
    assert torch.equal(parameter, before)
    assert not optimizer.pending


def test_no_second_gradient_still_restores_original_parameter():
    parameter = nn.Parameter(torch.tensor([1e-8]))
    original = parameter.detach().clone()
    optimizer = SAM(torch.optim.AdamW([parameter]))
    parameter.grad = torch.ones_like(parameter)
    optimizer.first_step()
    optimizer.second_step()  # grad=None after first_step's zero_grad.
    assert torch.equal(parameter, original)
    assert not optimizer.pending
    assert not optimizer.base_optimizer.state


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("pass_number", [1, 2])
def test_nonfinite_gradients_raise_and_restore(value, pass_number):
    parameter = nn.Parameter(torch.tensor([1e-8]))
    original = parameter.detach().clone()
    optimizer = SAM(torch.optim.AdamW([parameter]))
    if pass_number == 2:
        parameter.grad = torch.ones_like(parameter)
        optimizer.first_step()
    parameter.grad = torch.tensor([value])
    with pytest.raises(FloatingPointError, match="non-finite"):
        if pass_number == 1:
            optimizer.first_step()
        else:
            optimizer.second_step()
    assert torch.equal(parameter, original)
    assert parameter.grad is None
    assert not optimizer.pending
    assert not optimizer.base_optimizer.state


def test_perturbation_overflow_restores_exact_weights():
    maximum = torch.finfo(torch.float32).max
    parameter = nn.Parameter(torch.tensor([maximum]))
    original = parameter.detach().clone()
    optimizer = SAM(torch.optim.AdamW([parameter]), rho=maximum)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(FloatingPointError, match="parameter non-finite"):
        optimizer.first_step()
    assert torch.equal(parameter, original)
    assert not optimizer.pending
    assert parameter.grad is None


def test_caller_exception_can_restore_without_updating_state():
    parameter = nn.Parameter(torch.tensor([1e-8]))
    original = parameter.detach().clone()
    optimizer = SAM(torch.optim.AdamW([parameter]))
    parameter.grad = torch.ones_like(parameter)
    optimizer.first_step()
    try:
        raise RuntimeError("second forward failed")
    except RuntimeError:
        optimizer.restore()
    assert torch.equal(parameter, original)
    assert not optimizer.pending
    assert not optimizer.base_optimizer.state


def test_base_exception_sees_restored_weights_and_clears_pending_state(monkeypatch):
    parameter = nn.Parameter(torch.tensor([1e-8]))
    original = parameter.detach().clone()
    base = torch.optim.AdamW([parameter])
    optimizer = SAM(base)
    parameter.grad = torch.ones_like(parameter)
    optimizer.first_step()
    parameter.grad = torch.ones_like(parameter)

    def fail_step():
        assert torch.equal(parameter, original)
        raise RuntimeError("base step failed")

    monkeypatch.setattr(base, "step", fail_step)
    with pytest.raises(RuntimeError, match="base step failed"):
        optimizer.second_step()
    assert torch.equal(parameter, original)
    assert parameter.grad is None
    assert not optimizer.pending


def test_completed_update_cannot_restore_stale_weights():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = SAM(torch.optim.AdamW([parameter], lr=0.1, weight_decay=0))
    parameter.grad = torch.ones_like(parameter)
    optimizer.first_step()
    parameter.grad = torch.ones_like(parameter)
    optimizer.second_step()
    updated = parameter.detach().clone()
    optimizer.restore()
    assert torch.equal(parameter, updated)
    optimizer.first_step()  # No new gradient; an old perturbation must not be reused.
    optimizer.second_step()
    assert torch.equal(parameter, updated)


def test_optimizer_state_roundtrip_and_pending_save_guard():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = SAM(torch.optim.AdamW([parameter], lr=0.1), rho=0.02)
    parameter.grad = torch.ones_like(parameter)
    optimizer.first_step()
    with pytest.raises(RuntimeError, match="before saving"):
        optimizer.state_dict()
    parameter.grad = torch.ones_like(parameter)
    optimizer.second_step()
    state = copy.deepcopy(optimizer.state_dict())
    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = SAM(torch.optim.AdamW([restored_parameter], lr=0.2))
    restored.load_state_dict(state)
    assert restored.rho == 0.02
    assert restored.param_groups[0]["lr"] == 0.1
    assert restored.param_groups[0]["params"][0] is restored_parameter
    for name in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            optimizer.base_optimizer.state[parameter][name],
            restored.base_optimizer.state[restored_parameter][name],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("rho", [-0.01, float("nan"), float("inf")])
def test_invalid_rho_is_rejected(rho):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SAM(torch.optim.AdamW([nn.Parameter(torch.ones(1))]), rho=rho)
