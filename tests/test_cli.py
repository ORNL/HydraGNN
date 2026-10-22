import json
import os

import pytest
import torch
from torch_geometric.data import Data

MPI = pytest.importorskip("mpi4py.MPI")

import hydragnn
from hydragnn import cli

INPUTS = os.path.join(os.path.dirname(__file__), "inputs")


def _samples(count):
    torch.manual_seed(0)
    samples = []
    for i in range(count):
        n = 3 + i % 3
        x = torch.rand(n, 1)
        samples.append(
            Data(
                node_features=x,
                sum_x_x2_x3=(x + x**2 + x**3).sum().reshape(1, 1),
                pos=torch.rand(n, 3),
                edge_index=torch.combinations(torch.arange(n)).t().repeat(1, 2),
            )
        )
    return samples


def test_build_parser_defaults_and_overrides():
    parser = cli.build_parser(default_inputfile="cfg.json")
    args = parser.parse_args([])
    assert (args.inputfile, args.format, args.energy_per_atom) == (
        "cfg.json",
        "adios",
        False,
    )
    args = parser.parse_args(["--pickle", "--energy_per_atom", "--num_epoch", "3"])
    assert (args.format, args.energy_per_atom, args.num_epoch) == ("pickle", True, 3)
    with pytest.raises(SystemExit):
        parser.parse_args(["--adios", "--pickle"])


def test_dataset_path(tmp_path):
    assert cli.dataset_path(str(tmp_path), "qm9", "adios").endswith("dataset/qm9.bp")
    assert cli.dataset_path(str(tmp_path), "qm9", "pickle").endswith(
        "dataset/qm9.pickle"
    )


@pytest.mark.parametrize("fmt", ["pickle", "adios"])
def test_hydragnn_train_end_to_end(tmp_path, monkeypatch, fmt):
    if fmt == "adios":
        pytest.importorskip("adios2")
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _samples(12)
    dataset = str(tmp_path / ("graphs.bp" if fmt == "adios" else "graphs.pickle"))
    cli.write_split_datasets(
        dataset, fmt, samples[:8], samples[8:10], samples[10:], comm=MPI.COMM_SELF
    )

    with open(os.path.join(INPUTS, "ci.json")) as f:
        config = json.load(f)
    config["NeuralNetwork"]["Training"]["Checkpoint"] = False
    inputfile = tmp_path / "cfg.json"
    inputfile.write_text(json.dumps(config))

    cli.main(
        [
            "train",
            "--inputfile",
            str(inputfile),
            "--" + fmt,
            "--dataset",
            dataset,
            "--num_epoch",
            "2",
            "--batch_size",
            "4",
            "--log",
            "cli_test",
        ]
    )
    assert (tmp_path / "logs" / "cli_test" / "cli_test.pk").exists()
    assert (tmp_path / "logs" / "cli_test" / "config.json").exists()
