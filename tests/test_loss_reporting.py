from types import SimpleNamespace

import pytest
import torch

from hydragnn.loss_reporting import (
    accumulate_loss_report,
    finalize_loss_report,
    format_loss_report,
    reset_loss_report,
    supervised_loss_report,
)
from hydragnn.models.Base import Base
from hydragnn.utils.input_config_parsing.variable_schema import parse_variable_schema


def test_variable_schema_accepts_named_components_and_edge_attributes():
    schema = parse_variable_schema(
        {
            "graph_type": "homogeneous",
            "inputs": [{"name": "features", "level": "node", "dim": 1}],
            "outputs": [
                {
                    "name": "state",
                    "level": "graph",
                    "dim": 2,
                    "components": ["energy", "volume"],
                }
            ],
            "edge_attributes": {"bond": ["distance", "order"]},
        }
    )

    assert schema.outputs[0].components == ("energy", "volume")
    assert schema.edge_attributes["bond"] == ("distance", "order")


def test_multi_property_head_reports_raw_values_before_head_weight():
    model = SimpleNamespace(
        loss_weights=[0.25],
        loss_component_names=[["energy", "volume"]],
        loss_function=torch.nn.functional.mse_loss,
        loss_function_type="mse",
    )
    prediction = [torch.tensor([[2.0, 5.0], [4.0, 9.0]])]
    target = torch.tensor([[1.0], [3.0], [3.0], [7.0]])
    report = supervised_loss_report(
        model,
        prediction,
        target,
        [torch.tensor([0, 1, 2, 3])],
        [torch.nn.functional.mse_loss(prediction[0], target.reshape(2, 2))],
    )

    assert report["supervised.energy"]["raw"].item() == pytest.approx(1.0)
    assert report["supervised.energy"]["weight"] == pytest.approx(0.25)
    assert report["supervised.energy"]["weighted"].item() == pytest.approx(0.25)
    assert report["supervised.volume"]["raw"].item() == pytest.approx(4.0)
    assert report["supervised.volume"]["weighted"].item() == pytest.approx(1.0)


def test_base_loss_publishes_multi_property_report_without_changing_total():
    model = SimpleNamespace(
        num_heads=1,
        loss_weights=[0.25],
        loss_component_names=[["energy", "volume"]],
        loss_function=torch.nn.functional.mse_loss,
        loss_function_type="mse",
    )
    prediction = [torch.tensor([[2.0, 5.0], [4.0, 9.0]])]
    target = torch.tensor([[1.0], [3.0], [3.0], [7.0]])

    total, head_losses = Base.loss_hpweighted(
        model, prediction, target, [torch.tensor([0, 1, 2, 3])]
    )

    assert total.item() == pytest.approx(0.625)
    assert head_losses[0].item() == pytest.approx(2.5)
    assert model.last_loss_components["supervised.energy"]["raw"].item() == 1.0
    assert model.last_loss_components["supervised.volume"]["raw"].item() == 4.0


def test_split_accumulator_averages_raw_and_weighted_values_independently():
    model = SimpleNamespace()
    reset_loss_report(model)
    model.last_loss_components = {
        "constraints.balance": {
            "raw": torch.tensor(2.0),
            "weight": 3.0,
            "weighted": torch.tensor(7.0),
        }
    }
    accumulate_loss_report(model, 2)
    model.last_loss_components["constraints.balance"]["raw"] = torch.tensor(4.0)
    model.last_loss_components["constraints.balance"]["weighted"] = torch.tensor(11.0)
    accumulate_loss_report(model, 1)

    report = finalize_loss_report(model, torch.device("cpu"))

    assert report["constraints.balance"]["raw"] == pytest.approx(8.0 / 3.0)
    assert report["constraints.balance"]["weight"] == pytest.approx(3.0)
    assert report["constraints.balance"]["weighted"] == pytest.approx(25.0 / 3.0)
    line = format_loss_report(2, "validation", 9.0, report)
    assert "split=validation" in line
    assert "constraints.balance.raw=2.66666667" in line
    assert "constraints.balance.weight=3" in line
