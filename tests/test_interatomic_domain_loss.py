import pytest
import torch
from torch_geometric.data import Data

from hydragnn.domain_losses import InteratomicPotentialDomainLoss, create_domain_loss


class EnergyModel(torch.nn.Module):
    num_heads = 1
    head_type = ["node"]
    loss_function_type = "mse"

    def forward(self, data):
        return [data.pos.square().sum(dim=1, keepdim=True)]


def _config(**term_overrides):
    terms = {
        "energy": {
            "variable": "energy",
            "weight": 2.0,
            "normalization": "per_structure",
        },
        "energy_per_atom": {
            "variable": "energy_per_atom",
            "weight": 3.0,
            "normalization": "per_atom",
        },
        "forces": {
            "variable": "forces",
            "weight": 4.0,
            "prediction": {"operator": "negative_gradient"},
        },
    }
    terms.update(term_overrides)
    return {
        "enabled": True,
        "provider": "interatomic_potential",
        "supervised": {"default_metric": "mse", "terms": list(terms.values())},
        "constraints": [],
        "constraint_optimizer": {"type": "fixed_penalty"},
    }


def test_declarative_interatomic_terms_compute_expected_weighted_loss():
    model = create_domain_loss(EnergyModel(), _config())
    data = Data(
        pos=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True),
        batch=torch.tensor([0, 0]),
        energy=torch.tensor([4.0]),
        forces=torch.tensor([[-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
    )

    total, components = model.energy_force_loss(model(data), data)

    assert [value.item() for value in components] == pytest.approx([1.0, 0.25, 0.0])
    assert total.item() == pytest.approx(2.75)
    assert model.last_loss_components["supervised.energy"]["raw"].item() == 1.0
    assert model.last_loss_components["supervised.energy"]["weight"] == 2.0
    assert model.last_loss_components["supervised.energy"]["weighted"].item() == 2.0
    assert model.last_loss_components["supervised.forces"]["raw"].item() == 0.0
    total.backward()
    assert data.pos.grad is not None


def test_disabled_domain_loss_returns_original_model():
    model = EnergyModel()
    assert create_domain_loss(model, {"enabled": False}) is model


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"enabled": True, "provider": "unknown", "supervised": {"terms": []}},
            "Unknown",
        ),
        (_config(unknown={"variable": "unknown", "weight": 1.0}), "Unsupported"),
        (
            _config(
                forces={
                    "variable": "forces",
                    "weight": 1.0,
                    "prediction": {"operator": "direct"},
                }
            ),
            "negative_gradient",
        ),
    ],
)
def test_invalid_interatomic_domain_loss_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        create_domain_loss(EnergyModel(), config)


def test_wrapper_exposes_active_term_metadata_to_training_loop():
    config = _config(
        energy_per_atom={"variable": "energy_per_atom", "weight": 0.0},
        forces={"variable": "forces", "weight": 0.0},
    )
    model = InteratomicPotentialDomainLoss(EnergyModel(), config)

    assert model.task_names == ["energy"]
    assert model.task_weights == [2.0]
    assert model.atomistic_mode_enabled
