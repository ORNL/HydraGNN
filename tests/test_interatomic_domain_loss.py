import importlib

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


def test_duplicate_interatomic_terms_are_rejected():
    config = _config()
    config["supervised"]["terms"].append(
        {"variable": "energy", "weight": 1.0, "normalization": "per_structure"}
    )

    with pytest.raises(ValueError, match="Duplicate.*energy"):
        create_domain_loss(EnergyModel(), config)


@pytest.mark.parametrize(
    "disabled, expected",
    [
        ({"energy_per_atom", "forces"}, ["energy"]),
        ({"energy"}, ["energy_per_atom", "forces"]),
        (set(), ["energy", "energy_per_atom", "forces"]),
    ],
)
def test_prediction_target_pairs_follow_active_task_names(disabled, expected):
    overrides = {
        name: {
            **term,
            "weight": 0.0 if name in disabled else term["weight"],
        }
        for name, term in {
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
        }.items()
    }
    model = create_domain_loss(EnergyModel(), _config(**overrides))
    data = Data(
        pos=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True),
        batch=torch.tensor([0, 0]),
        energy=torch.tensor([5.0]),
        forces=torch.tensor([[-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
        num_graphs=1,
    )

    values = model.prediction_target_pairs(model(data), data, create_graph=False)

    assert model.task_names == expected
    assert list(values) == expected


def test_test_sampling_uses_active_interatomic_tasks(monkeypatch):
    workflow = importlib.import_module("hydragnn.train.train_validate_test")
    monkeypatch.setattr(
        workflow, "iterate_tqdm", lambda iterator, *args, **kwargs: iterator
    )
    config = _config(
        energy={"variable": "energy", "weight": 0.0},
    )
    provider = create_domain_loss(EnergyModel(), config)

    class Wrapped(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, data):
            return self.module(data)

    data = Data(
        pos=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        energy=torch.tensor([5.0]),
        forces=torch.tensor([[-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
        num_graphs=1,
    )

    class Loader:
        dataset = [data]

        def __len__(self):
            return 1

        def __iter__(self):
            yield data

    _, task_errors, true_values, predicted_values = workflow.test(
        Loader(),
        Wrapped(provider),
        verbosity=0,
        reduce_ranks=False,
        return_samples=True,
        compute_grad_energy=True,
    )

    assert provider.task_names == ["energy_per_atom", "forces"]
    assert task_errors.shape == (2,)
    assert [values.shape for values in true_values] == [(1, 1), (6, 1)]
    assert [values.shape for values in predicted_values] == [(1, 1), (6, 1)]


def test_wrapper_exposes_active_term_metadata_to_training_loop():
    config = _config(
        energy_per_atom={"variable": "energy_per_atom", "weight": 0.0},
        forces={"variable": "forces", "weight": 0.0},
    )
    model = InteratomicPotentialDomainLoss(EnergyModel(), config)

    assert model.task_names == ["energy"]
    assert model.task_weights == [2.0]
    assert model.atomistic_mode_enabled
    assert model.requires_find_unused_parameters


def test_interatomic_provider_enables_unused_parameter_detection(monkeypatch):
    distributed = importlib.import_module("hydragnn.utils.distributed.distributed")
    model = InteratomicPotentialDomainLoss(EnergyModel(), _config())
    captured = {}

    def capture_wrapper(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(distributed, "get_distributed_model", capture_wrapper)
    monkeypatch.setattr(
        distributed,
        "configure_local_sgd",
        lambda model, optimizer, *args, **kwargs: (model, optimizer),
    )

    wrapped, _ = distributed.distributed_model_wrapper(
        model,
        object(),
        config={"NeuralNetwork": {"Training": {}}},
    )

    assert wrapped is model
    assert captured["find_unused_parameters"] is True
    assert captured["enhanced_model"] is True
