import importlib
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
from hydragnn.utils.print.print_utils import setup_log


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
    assert model._loss_report_sums == {}
    assert model._loss_report_count == 0
    line = format_loss_report(2, "validation", 9.0, report)
    assert "split=validation" in line
    assert "constraints.balance.raw=2.66666667" in line
    assert "constraints.balance.weight=3" in line


def test_training_workflow_writes_each_split_report_to_run_log(tmp_path, monkeypatch):
    workflow = importlib.import_module("hydragnn.train.train_validate_test")

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))
            self.num_heads = 1
            self.head_dims = [2]
            self.loss_weights = [0.25]

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Core()

    class Loader:
        batch_sampler = None
        sampler = None

    model = Wrapped()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = SimpleNamespace(step=lambda loss: None)
    reports = {
        "train": (1.0, 0.25),
        "validation": (3.0, 0.75),
        "test": (5.0, 1.25),
    }

    def publish(split):
        raw, weighted = reports[split]
        model.module.last_epoch_loss_report = {
            "supervised.energy": {
                "raw": raw,
                "weight": 0.25,
                "weighted": weighted,
            }
        }

    def fake_train(*args, **kwargs):
        publish("train")
        return torch.tensor(10.0), torch.tensor([1.0])

    def fake_validate(*args, **kwargs):
        publish("validation")
        return torch.tensor(20.0), torch.tensor([2.0])

    def fake_test(*args, **kwargs):
        publish("test")
        return torch.tensor(30.0), torch.tensor([3.0]), [], []

    monkeypatch.setattr(workflow, "train", fake_train)
    monkeypatch.setattr(workflow, "validate", fake_validate)
    monkeypatch.setattr(workflow, "test", fake_test)
    monkeypatch.setattr(workflow.tr, "enable", lambda: None)
    monkeypatch.setattr(workflow.tr, "disable", lambda: None)
    monkeypatch.setattr(workflow.tr, "start", lambda name: None)
    monkeypatch.setattr(workflow.tr, "stop", lambda name: None)
    monkeypatch.setattr(workflow.tr, "reset", lambda: None)
    monkeypatch.chdir(tmp_path)
    setup_log("loss-report-ci")

    workflow.train_validate_test(
        model,
        optimizer,
        Loader(),
        Loader(),
        Loader(),
        None,
        scheduler,
        {
            "Variables": {
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
            },
            "NeuralNetwork": {
                "Training": {
                    "num_epoch": 1,
                    "Checkpoint": False,
                    "EarlyStopping": False,
                    "CheckRemainingTime": False,
                }
            },
        },
        "loss-report-ci",
        verbosity=1,
        create_plots=False,
    )

    report_lines = [
        line
        for line in (tmp_path / "logs/loss-report-ci/run.log").read_text().splitlines()
        if "LossComponents" in line
    ]
    assert len(report_lines) == 3
    assert "split=train" in report_lines[0]
    assert "supervised.energy.raw=1.00000000" in report_lines[0]
    assert "split=validation" in report_lines[1]
    assert "supervised.energy.raw=3.00000000" in report_lines[1]
    assert "split=test" in report_lines[2]
    assert "supervised.energy.raw=5.00000000" in report_lines[2]
