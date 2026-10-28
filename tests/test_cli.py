import json
import os

import pytest
import torch
from torch_geometric.data import Data, HeteroData

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
    assert cli.dataset_path(str(tmp_path), "qm9", "hdf5").endswith("dataset/qm9.h5")


@pytest.mark.parametrize("fmt", ["pickle", "adios", "hdf5"])
def test_hydragnn_train_end_to_end(tmp_path, monkeypatch, fmt):
    pytest.importorskip({"pickle": "pickle", "adios": "adios2", "hdf5": "h5py"}[fmt])
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _samples(12)
    dataset = cli.dataset_path(str(tmp_path), "graphs", fmt)
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


def _hetero_samples(count):
    torch.manual_seed(0)
    samples = []
    for _ in range(count):
        data = HeteroData()
        data["a"].x = torch.rand(4, 2)
        data["b"].x = torch.rand(3, 3)
        data["a", "to", "b"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]])
        data["b", "to", "a"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        data.y = data["a"].x.sum().reshape(1, 1)
        samples.append(data)
    return samples


def _hetero_config(mpnn_type):
    return {
        "Verbosity": {"level": 0},
        "NeuralNetwork": {
            "Architecture": {
                "mpnn_type": mpnn_type,
                "hidden_dim": 8,
                "num_conv_layers": 1,
                "pe_dim": 0,
                "max_neighbours": 10,
                "edge_types": [
                    {"source_type": s, "relation": "to", "target_type": t, "dim": 0}
                    for s, t in (("a", "b"), ("b", "a"))
                ],
                "output_heads": {
                    "graph": {
                        "num_sharedlayers": 1,
                        "dim_sharedlayers": 8,
                        "num_headlayers": 1,
                        "dim_headlayers": [8],
                    }
                },
                "task_weights": [1.0],
            },
            "Training": {
                "num_epoch": 1,
                "batch_size": 2,
                "Checkpoint": False,
                "Optimizer": {"type": "AdamW", "learning_rate": 1e-3},
            },
        },
        "Variables": {
            "graph_type": "heterogeneous",
            "node_types": ["a", "b"],
            "inputs": [
                {"name": "a_x", "level": "node", "dim": 2, "node_type": "a"},
                {"name": "b_x", "level": "node", "dim": 3, "node_type": "b"},
            ],
            "outputs": [{"name": "target", "level": "graph", "dim": 1}],
        },
    }


def test_hydragnn_train_heterogeneous_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _hetero_samples(8)
    dataset = str(tmp_path / "hetero.pickle")
    cli.write_split_datasets(
        dataset, "pickle", samples[:4], samples[4:6], samples[6:], comm=MPI.COMM_SELF
    )
    inputfile = tmp_path / "cfg.json"
    inputfile.write_text(json.dumps(_hetero_config("HeteroGIN")))

    cli.main(
        [
            "train",
            "--inputfile",
            str(inputfile),
            "--pickle",
            "--dataset",
            dataset,
            "--log",
            "cli_hetero",
        ]
    )
    assert (tmp_path / "logs" / "cli_hetero" / "cli_hetero.pk").exists()


def test_gather_hetero_deg_counts_all_edge_types():
    from hydragnn.preprocess.graph_samples_checks_and_updates import (
        gather_hetero_deg,
    )

    hydragnn.distributed.setup_ddp()
    deg = gather_hetero_deg(_hetero_samples(2), verbosity=0)
    # Per sample: in-degrees a=[0,1,1,1], b=[1,2,1] -> one 0, five 1s, one 2.
    assert list(deg) == [2, 10, 2]


def test_save_config_serializes_tensors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs" / "tensor_cfg").mkdir(parents=True)
    hydragnn.config.save_config({"pna_deg": torch.tensor([1, 2])}, "tensor_cfg")
    with open(tmp_path / "logs" / "tensor_cfg" / "config.json") as f:
        assert json.load(f) == {"pna_deg": [1, 2]}


@pytest.mark.parametrize("mpnn_type", ["HeteroGIN", "HeteroPNA"])
def test_run_training_heterogeneous_with_hooks(tmp_path, monkeypatch, mpnn_type):
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _hetero_samples(8)
    hydragnn.utils.print.setup_log("hetero_" + mpnn_type)
    calls = []

    def wrap_model(model, config):
        calls.append("model_wrapper")
        return model

    def before_train(model, optimizer, loaders, config):
        calls.append("before_train")
        assert len(loaders) == 3

    model, config = cli.run_training(
        _hetero_config(mpnn_type),
        samples[:4],
        samples[4:6],
        samples[6:],
        "hetero_" + mpnn_type,
        None,
        model_wrapper=wrap_model,
        before_train=before_train,
        after_train=lambda model: calls.append("after_train"),
    )
    arch = config["NeuralNetwork"]["Architecture"]
    assert arch["node_input_dims"] == {"a": 2, "b": 3}
    if mpnn_type == "HeteroPNA":
        assert list(arch["pna_deg"]) == [4, 20, 4]
    assert calls == ["model_wrapper", "before_train", "after_train"]
    assert (tmp_path / "logs" / ("hetero_" + mpnn_type) / "config.json").exists()


@pytest.mark.parametrize("node_head", [False, True])
def test_run_training_heterogeneous_multihead(tmp_path, monkeypatch, node_head):
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _hetero_samples(8)
    for data in samples:
        energy = data["a"].x.sum().reshape(1, 1)
        second = data["a"].x[:, :1] if node_head else data["b"].x.sum().reshape(1, 1)
        data.y = torch.cat([energy, second])
        data.y_loc = torch.tensor([[0, 1, 1 + second.shape[0]]])
    config = _hetero_config("HeteroGIN")
    arch = config["NeuralNetwork"]["Architecture"]
    arch["task_weights"] = [1.0, 1.0]
    outputs = config["Variables"]["outputs"]
    if node_head:
        arch["node_target_type"] = "a"
        arch["output_heads"]["node"] = {
            "num_headlayers": 1,
            "dim_headlayers": [8],
            "type": "mlp",
        }
        outputs.append({"name": "a_out", "level": "node", "dim": 1, "node_type": "a"})
    else:
        outputs.append({"name": "target2", "level": "graph", "dim": 1})
    log_name = "hetero_multihead_%s" % node_head
    hydragnn.utils.print.setup_log(log_name)
    cli.run_training(config, samples[:4], samples[4:6], samples[6:], log_name, None)
    assert (tmp_path / "logs" / log_name / (log_name + ".pk")).exists()


def test_write_split_datasets_rejects_heterogeneous_adios(tmp_path):
    samples = _hetero_samples(3)
    with pytest.raises(NotImplementedError, match="pickle"):
        cli.write_split_datasets(
            str(tmp_path / "h.bp"),
            "adios",
            samples[:1],
            samples[1:2],
            samples[2:],
            comm=MPI.COMM_SELF,
        )


def test_write_split_datasets_empty_shard_follows_hetero_ranks(tmp_path):
    class PeerIsHeteroComm:
        def allreduce(self, value, op):
            return True

    with pytest.raises(NotImplementedError, match="pickle"):
        cli.write_split_datasets(
            str(tmp_path / "h.bp"), "adios", [], [], [], comm=PeerIsHeteroComm()
        )


def test_gather_hetero_deg_rank_local(monkeypatch):
    import torch.distributed as dist
    from hydragnn.preprocess.graph_samples_checks_and_updates import (
        gather_hetero_deg,
    )

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op: None)

    class TwoIdenticalRanksComm:
        def allreduce(self, value, op):
            return value if op == MPI.MAX else value * 2

    samples = _hetero_samples(2)
    assert gather_hetero_deg(samples, 0) == [1, 5, 1]
    assert gather_hetero_deg(samples, 0, comm=TwoIdenticalRanksComm()) == [4, 20, 4]


def test_run_training_reuses_cached_hetero_pna_deg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _hetero_samples(8)

    class CachedDegList(list):
        pna_deg = [7, 0, 0, 1]

    hydragnn.utils.print.setup_log("hetero_cached")
    _, config = cli.run_training(
        _hetero_config("HeteroPNA"),
        CachedDegList(samples[:4]),
        samples[4:6],
        samples[6:],
        "hetero_cached",
        None,
        before_train=lambda *args: True,
    )
    arch = config["NeuralNetwork"]["Architecture"]
    assert arch["pna_deg"] == [7, 0, 0, 1]
    assert arch["max_neighbours"] == 3


def test_run_training_before_train_can_skip_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hydragnn.distributed.setup_ddp()
    samples = _hetero_samples(6)
    hydragnn.utils.print.setup_log("hetero_skip")
    after = []
    cli.run_training(
        _hetero_config("HeteroGIN"),
        samples[:2],
        samples[2:4],
        samples[4:],
        "hetero_skip",
        None,
        before_train=lambda *args: True,
        after_train=after.append,
    )
    assert after == []
    assert not (tmp_path / "logs" / "hetero_skip" / "hetero_skip.pk").exists()
