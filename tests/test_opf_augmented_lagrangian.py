import importlib.util
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

_MODULE_PATH = Path(__file__).parents[1] / "examples" / "opf" / "opf_solution_utils.py"
_SPEC = importlib.util.spec_from_file_location("opf_solution_utils", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
OPFDomainLoss = _MODULE.OPFDomainLoss
OPFEnhancedModelWrapper = _MODULE.OPFEnhancedModelWrapper


def _domain_loss(**overrides):
    config = {
        "enabled": True,
        "rho": 2.0,
        "rho_growth": 3.0,
        "rho_max": 20.0,
        "constraint_reduction": 0.5,
    }
    config.update(overrides)
    return OPFDomainLoss(config)


def test_augmented_term_has_linear_dual_and_quadratic_parts():
    loss = _domain_loss()
    residual = torch.tensor(0.5, requires_grad=True)

    first = loss._augmented_term("voltage_bound", residual, 1.0, 1.0, True, {})
    assert first.item() == pytest.approx(0.25)
    first.backward()
    assert residual.grad.item() == pytest.approx(1.0)

    loss.update_multipliers(completed_epoch=0)
    assert loss.dual_voltage_bound.item() == pytest.approx(1.0)

    second = loss._augmented_term(
        "voltage_bound", torch.tensor(0.5), 1.0, 1.0, True, {}
    )
    assert second.item() == pytest.approx(0.75)


def test_multiplier_is_projected_and_rho_grows_when_residual_stalls():
    loss = _domain_loss()
    loss._augmented_term("voltage_bound", torch.tensor(0.4), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=0)
    assert loss.dual_voltage_bound.item() == pytest.approx(0.8)
    assert loss.rho.item() == pytest.approx(2.0)

    loss._augmented_term("voltage_bound", torch.tensor(0.4), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=1)
    assert loss.dual_voltage_bound.item() == pytest.approx(1.6)
    assert loss.rho.item() == pytest.approx(6.0)
    assert loss.dual_voltage_bound.item() >= 0.0


def test_evaluation_residuals_do_not_update_dual_state():
    loss = _domain_loss()
    loss._augmented_term("voltage_bound", torch.tensor(0.7), 1.0, 1.0, False, {})
    loss.update_multipliers(completed_epoch=0)

    assert loss.dual_voltage_bound.item() == 0.0
    assert torch.isinf(loss.previous_constraint_norm)


def test_voltage_violation_flows_through_domain_loss_and_updates_dual():
    loss = _domain_loss(
        voltage_bound_weight=1.0,
        voltage_bound_feature_indices=[0, 1],
        voltage_output_index=1,
    )
    data = HeteroData()
    data["bus"].x = torch.tensor([[0.9, 1.1]])
    prediction = [torch.tensor([[0.0, 1.2]], requires_grad=True)]
    target = [torch.zeros(1, 2)]

    augmented, metrics = loss(prediction, target, [0], data, update_state=True)

    assert metrics["opf_voltage_bound"].item() == pytest.approx(0.1)
    assert augmented.item() == pytest.approx(0.01)
    augmented.backward()
    assert prediction[0].grad[0, 1].item() == pytest.approx(0.2)

    loss.update_multipliers(completed_epoch=0)
    assert loss.dual_voltage_bound.item() == pytest.approx(0.2)


def test_augmented_lagrangian_state_round_trips_through_state_dict():
    loss = _domain_loss()
    loss._augmented_term("ac_angle_diff", torch.tensor(0.25), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=0)

    restored = _domain_loss(rho=1.0)
    restored.load_state_dict(loss.state_dict())

    assert restored.rho.item() == pytest.approx(loss.rho.item())
    assert restored.dual_ac_angle_diff.item() == pytest.approx(
        loss.dual_ac_angle_diff.item()
    )
    assert restored.previous_constraint_norm.item() == pytest.approx(
        loss.previous_constraint_norm.item()
    )


def test_wrapper_finalizes_duals_when_training_switches_to_validation():
    class StubModel(torch.nn.Module):
        def forward(self, data):
            return data

    domain_loss = _domain_loss()
    wrapper = OPFEnhancedModelWrapper(StubModel(), domain_loss)
    wrapper._last_seen_epoch = 0
    domain_loss._augmented_term("voltage_bound", torch.tensor(0.25), 1.0, 1.0, True, {})

    wrapper.eval()

    assert domain_loss.dual_voltage_bound.item() == pytest.approx(0.5)
    assert domain_loss._constraint_counts["voltage_bound"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"rho": 0.0},
        {"rho": 2.0, "rho_max": 1.0},
        {"rho_growth": 0.5},
        {"constraint_reduction": 0.0},
        {"constraint_reduction": 1.1},
        {"dual_update_interval": 0},
    ],
)
def test_invalid_augmented_lagrangian_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _domain_loss(**overrides)
