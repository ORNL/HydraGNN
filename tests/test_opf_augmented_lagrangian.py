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
    optimizer = {
        "type": "augmented_lagrangian",
        "rho": 2.0,
        "rho_growth": 3.0,
        "rho_max": 20.0,
        "required_reduction": 0.5,
        "update_every": 1,
    }
    optimizer.update(overrides)
    config = {"enabled": True, "constraint_optimizer": optimizer}
    return OPFDomainLoss(config)


def test_augmented_term_has_linear_dual_and_quadratic_parts():
    loss = _domain_loss()
    residual = torch.tensor(0.5, requires_grad=True)

    first = loss._augmented_term("voltage_bound", residual, 1.0, 1.0, True, {})
    assert first.item() == pytest.approx(0.25)
    first.backward()
    assert residual.grad.item() == pytest.approx(1.0)

    loss.update_multipliers(completed_epoch=0)
    assert loss.constraint_optimizer.dual_voltage_bound.item() == pytest.approx(1.0)

    second = loss._augmented_term(
        "voltage_bound", torch.tensor(0.5), 1.0, 1.0, True, {}
    )
    assert second.item() == pytest.approx(0.75)


def test_multiplier_is_projected_and_rho_grows_when_residual_stalls():
    loss = _domain_loss()
    loss._augmented_term("voltage_bound", torch.tensor(0.4), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=0)
    assert loss.constraint_optimizer.dual_voltage_bound.item() == pytest.approx(0.8)
    assert loss.constraint_optimizer.rho.item() == pytest.approx(2.0)

    loss._augmented_term("voltage_bound", torch.tensor(0.4), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=1)
    assert loss.constraint_optimizer.dual_voltage_bound.item() == pytest.approx(1.6)
    assert loss.constraint_optimizer.rho.item() == pytest.approx(6.0)


def test_evaluation_residuals_do_not_update_dual_state():
    loss = _domain_loss()
    loss._augmented_term("voltage_bound", torch.tensor(0.7), 1.0, 1.0, False, {})
    loss.update_multipliers(completed_epoch=0)

    assert loss.constraint_optimizer.dual_voltage_bound.item() == 0.0
    assert torch.isinf(loss.constraint_optimizer.previous_residual_norm)


def test_voltage_violation_flows_through_domain_loss_and_updates_dual():
    variables = {
        "inputs": [
            {"name": "vmin", "level": "node", "node_type": "bus", "dim": 1},
            {"name": "vmax", "level": "node", "node_type": "bus", "dim": 1},
        ],
        "outputs": [
            {
                "name": "voltage",
                "node_type": "bus",
                "components": ["angle", "magnitude"],
            }
        ],
    }
    config = {
        "enabled": True,
        "constraints": [
            {
                "name": "voltage_limits",
                "operator": "bounded",
                "scale": 1.0,
                "value": "magnitude",
                "lower": "vmin",
                "upper": "vmax",
            }
        ],
        "constraint_optimizer": {"type": "augmented_lagrangian", "rho": 2.0},
    }
    loss = OPFDomainLoss(config, variables=variables)
    data = HeteroData()
    data["bus"].x = torch.tensor([[0.9, 1.1]])
    prediction = [torch.tensor([[0.0, 1.2]], requires_grad=True)]
    target = [torch.zeros(1, 2)]

    augmented, metrics = loss(prediction, target, [0], data, update_state=True)

    assert metrics["opf_voltage_bound"].item() == pytest.approx(0.1)
    assert augmented.item() == pytest.approx(0.01)
    component = loss.last_loss_components["constraints.voltage_limits"]
    assert component["raw"].item() == pytest.approx(0.1)
    assert component["weight"] == pytest.approx(1.0)
    assert component["weighted"].item() == pytest.approx(0.01)
    augmented.backward()
    assert prediction[0].grad[0, 1].item() == pytest.approx(0.2)

    loss.update_multipliers(completed_epoch=0)
    assert loss.constraint_optimizer.dual_voltage_bound.item() == pytest.approx(0.2)


def test_augmented_lagrangian_state_round_trips_through_state_dict():
    loss = _domain_loss()
    loss._augmented_term("ac_angle_diff", torch.tensor(0.25), 1.0, 1.0, True, {})
    loss.update_multipliers(completed_epoch=0)

    restored = _domain_loss(rho=1.0)
    restored.load_state_dict(loss.state_dict())

    assert restored.constraint_optimizer.rho.item() == pytest.approx(
        loss.constraint_optimizer.rho.item()
    )
    assert restored.constraint_optimizer.dual_ac_angle_diff.item() == pytest.approx(
        loss.constraint_optimizer.dual_ac_angle_diff.item()
    )
    assert restored.constraint_optimizer.previous_residual_norm.item() == pytest.approx(
        loss.constraint_optimizer.previous_residual_norm.item()
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

    assert domain_loss.constraint_optimizer.dual_voltage_bound.item() == pytest.approx(
        0.5
    )


def test_wrapper_uses_declarative_supervised_metric_and_weight():
    class StubModel(torch.nn.Module):
        def loss(self, pred, value, head_index):
            raise AssertionError(
                "The base loss must not override declarative OPF loss."
            )

    config = {
        "enabled": True,
        "supervised": {
            "default_metric": "mae",
            "terms": [{"variable": "bus_va_vm", "weight": 2.0}],
        },
        "constraints": [],
        "constraint_optimizer": {"type": "fixed_penalty"},
    }
    variables = {"outputs": [{"name": "bus_va_vm", "level": "node", "dim": 1}]}
    domain_loss = OPFDomainLoss(config, variables=variables)
    wrapper = OPFEnhancedModelWrapper(StubModel(), domain_loss)
    prediction = [torch.tensor([[3.0], [1.0]])]
    target = torch.tensor([[1.0], [0.0]])
    wrapper._last_batch = HeteroData()

    total, tasks = wrapper.loss(
        prediction, target, [torch.tensor([0, 1], dtype=torch.long)]
    )

    assert tasks[0].item() == pytest.approx(1.5)
    assert total.item() == pytest.approx(3.0)
    component = wrapper.last_loss_components["supervised.bus_va_vm"]
    assert component["raw"].item() == pytest.approx(1.5)
    assert component["weight"] == pytest.approx(2.0)
    assert component["weighted"].item() == pytest.approx(3.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rho": 0.0},
        {"rho": 2.0, "rho_max": 1.0},
        {"rho_growth": 0.5},
        {"required_reduction": 0.0},
        {"required_reduction": 1.1},
        {"update_every": 0},
    ],
)
def test_invalid_augmented_lagrangian_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _domain_loss(**overrides)
