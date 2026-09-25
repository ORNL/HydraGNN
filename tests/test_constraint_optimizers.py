import pytest
import torch

from hydragnn.domain_losses import create_constraint_optimizer


def test_fixed_penalty_is_stateless_and_scaled():
    optimizer = create_constraint_optimizer(["balance"], {"type": "fixed_penalty"})

    penalty = optimizer.penalty(
        "balance", torch.tensor(3.0, requires_grad=True), scale=2.0, update_state=True
    )

    assert penalty.item() == pytest.approx(18.0)
    assert optimizer.state_dict() == {}


def test_augmented_lagrangian_updates_named_dual_and_rho():
    optimizer = create_constraint_optimizer(
        ["balance"],
        {
            "type": "augmented_lagrangian",
            "rho": 2.0,
            "rho_growth": 3.0,
            "rho_max": 10.0,
            "required_reduction": 0.5,
            "update_every": 1,
        },
    )

    first = optimizer.penalty("balance", torch.tensor(0.5), update_state=True)
    assert first.item() == pytest.approx(0.25)
    optimizer.update(0)
    assert optimizer.dual_balance.item() == pytest.approx(1.0)

    optimizer.penalty("balance", torch.tensor(0.5), update_state=True)
    optimizer.update(1)
    assert optimizer.dual_balance.item() == pytest.approx(2.0)
    assert optimizer.rho.item() == pytest.approx(6.0)


def test_unknown_constraint_optimizer_is_rejected():
    with pytest.raises(ValueError, match="Unknown constraint_optimizer"):
        create_constraint_optimizer(["x"], {"type": "mystery"})
