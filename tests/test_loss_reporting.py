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
from hydragnn.models.base import Base
from hydragnn.config.variable_schema import parse_variable_schema
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


def _run_minimal_training_workflow(
    tmp_path,
    monkeypatch,
    log_name,
    scheduler_factory=None,
    num_epoch=1,
    optimizer_factory=None,
    writer=None,
    optimizer_config=None,
    train_calls=None,
):
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
    if optimizer_factory is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = optimizer_factory(model)
    if scheduler_factory is None:
        scheduler = SimpleNamespace(step=lambda loss: None)
    else:
        scheduler = scheduler_factory(optimizer)
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
        if train_calls is not None:
            train_calls.append(kwargs)
        sched = kwargs.get("scheduler")
        if sched is not None:
            # Emulate two optimizer steps per epoch.
            for _ in range(2):
                optimizer.step()
                sched.step()
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
    setup_log(log_name)

    workflow.train_validate_test(
        model,
        optimizer,
        Loader(),
        Loader(),
        Loader(),
        writer,
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
                    "num_epoch": num_epoch,
                    "Checkpoint": False,
                    "EarlyStopping": False,
                    "CheckRemainingTime": False,
                    **(
                        {"Optimizer": optimizer_config}
                        if optimizer_config is not None
                        else {}
                    ),
                }
            },
        },
        log_name,
        verbosity=1,
        create_plots=False,
    )
    return optimizer, (tmp_path / f"logs/{log_name}/run.log").read_text()


def test_training_workflow_writes_each_split_report_to_run_log(tmp_path, monkeypatch):
    _, run_log = _run_minimal_training_workflow(tmp_path, monkeypatch, "loss-report-ci")

    report_lines = [line for line in run_log.splitlines() if "LossComponents" in line]
    assert len(report_lines) == 3
    assert "split=train" in report_lines[0]
    assert "supervised.energy.raw=1.00000000" in report_lines[0]
    assert "split=validation" in report_lines[1]
    assert "supervised.energy.raw=3.00000000" in report_lines[1]
    assert "split=test" in report_lines[2]
    assert "supervised.energy.raw=5.00000000" in report_lines[2]


def test_training_workflow_steps_epoch_scheduler_without_metric(tmp_path, monkeypatch):
    optimizer, run_log = _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-steplr-ci",
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        ),
        num_epoch=2,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    # The reported LR is the one used during the epoch, before the scheduler step.
    assert "LR: 0.10000000" in run_log
    assert "LR: 0.05000000" in run_log


def test_training_workflow_steps_plateau_scheduler_with_val_loss(tmp_path, monkeypatch):
    seen = []

    class Plateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metrics):
            seen.append(float(metrics))
            super().step(metrics)

    _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-plateau-ci",
        scheduler_factory=lambda opt: Plateau(opt),
    )

    assert seen == [pytest.approx(20.0)]


def test_training_workflow_steps_onecycle_scheduler_per_batch(tmp_path, monkeypatch):
    train_calls = []
    optimizer, _ = _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-onecycle-ci",
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=4
        ),
        num_epoch=2,
        train_calls=train_calls,
    )

    assert len(train_calls) == 2
    assert all(
        isinstance(c["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
        for c in train_calls
    )
    # 2 epochs x 2 batches = 4 steps, no extra epoch-level step (which would
    # exceed total_steps and raise).
    assert train_calls[0]["scheduler"].last_epoch == 4


def test_training_workflow_scheduler_interval_config(tmp_path, monkeypatch):
    train_calls = []
    optimizer, run_log = _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-interval-ci",
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        ),
        num_epoch=2,
        optimizer_config={"scheduler_interval": "batch"},
        train_calls=train_calls,
    )

    assert all(c["scheduler"] is not None for c in train_calls)
    # Stepped 4 times per batch, never at epoch end.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**4)


def test_training_workflow_epoch_scheduler_not_passed_to_train(tmp_path, monkeypatch):
    train_calls = []
    _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-epoch-default-ci",
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        ),
        train_calls=train_calls,
    )

    assert train_calls[0]["scheduler"] is None


@pytest.mark.parametrize(
    "optimizer_config",
    [{"scheduler_interval": "step"}, {"scheduler_interval": "batch"}],
)
def test_scheduler_interval_rejects_invalid_settings(optimizer_config):
    workflow = importlib.import_module("hydragnn.train.train_validate_test")
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=0.1)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    with pytest.raises(ValueError):
        workflow._scheduler_steps_per_batch(
            plateau, {"Training": {"Optimizer": optimizer_config}}
        )


def test_training_workflow_logs_lr_of_every_param_group(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self):
            self.scalars = []
            self.multi = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def add_scalars(self, tag, values, step):
            self.multi.append((tag, dict(values), step))

    def two_groups(model):
        extra = torch.nn.Parameter(torch.tensor(0.0))
        return torch.optim.SGD(
            [
                {"params": model.parameters(), "lr": 0.1},
                {"params": [extra], "lr": 0.01},
            ]
        )

    writer = FakeWriter()
    _, run_log = _run_minimal_training_workflow(
        tmp_path,
        monkeypatch,
        "lr-groups-ci",
        optimizer_factory=two_groups,
        writer=writer,
    )

    assert writer.multi == [
        ("learning rate", {"group_0": pytest.approx(0.1), "group_1": 0.01}, 0)
    ]
    assert not any(tag == "learning rate" for tag, _, _ in writer.scalars)
    assert "LR: [0.10000000, 0.01000000]" in run_log


def test_plot_history_saves_lr_history(tmp_path, monkeypatch):
    import pickle

    from hydragnn.postprocess.visualizer import Visualizer

    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs/lr-history-ci").mkdir(parents=True)
    visualizer = Visualizer("lr-history-ci", node_feature=[[0.0]], num_nodes_list=[1])
    epochs = 3
    lr_history = [[0.1], [0.05], [0.025]]
    visualizer.plot_history(
        torch.ones(epochs),
        torch.ones(epochs),
        torch.ones(epochs),
        torch.ones(epochs, 2),
        torch.ones(epochs, 2),
        torch.ones(epochs, 2),
        [0.5, 0.5],
        ["a", "b"],
        lr_history=lr_history,
    )

    with open(tmp_path / "logs/lr-history-ci/history_loss.pckl", "rb") as f:
        history = pickle.load(f)
    assert len(history) == 9
    assert history[7] == ["a", "b"]
    assert history[8] == lr_history
