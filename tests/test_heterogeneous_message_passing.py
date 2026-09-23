##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.models.create import create_model
from hydragnn.utils.model.model import update_multibranch_heads
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from examples.pglearn.download_and_uncompress_data import _validate_path_component


def _build_simple_hetero_graph(input_dim: int = 4, edge_dim: int = None):
    data = HeteroData()

    # Node features
    data["a"].x = torch.randn(4, input_dim)
    data["b"].x = torch.randn(3, input_dim)

    # Edges: a -> b
    edge_index_ab = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]], dtype=torch.long)
    data[("a", "to", "b")].edge_index = edge_index_ab
    if edge_dim is not None:
        data[("a", "to", "b")].edge_attr = torch.randn(edge_index_ab.size(1), edge_dim)

    # Edges: b -> a
    edge_index_ba = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data[("b", "to", "a")].edge_index = edge_index_ba
    if edge_dim is not None:
        data[("b", "to", "a")].edge_attr = torch.randn(edge_index_ba.size(1), edge_dim)

    return data


def _graph_head_model_args(mpnn_type, edge_dim, **overrides):
    output_heads = {
        "graph": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [8],
        }
    }
    args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [2],
        "pe_dim": 0,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["graph"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
        "edge_dim": edge_dim,
        "metadata": (["a", "b"], [("a", "to", "b"), ("b", "to", "a")]),
        "node_input_dims": {"a": 4, "b": 4},
    }
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    "value", ["../escape", "nested/case", "..\\escape", "/absolute"]
)
def test_pglearn_identifiers_reject_path_components(value):
    with pytest.raises(ValueError, match="single non-empty path component"):
        _validate_path_component(value, "case_name")


@pytest.mark.parametrize("mpnn_type", ["HeteroGAT", "HeteroRGAT", "HeteroPNA"])
def test_shared_relation_weights_reject_different_edge_widths(mpnn_type):
    edge_dim = {("a", "to", "b"): 3, ("b", "to", "a"): 5}
    args = _graph_head_model_args(
        mpnn_type,
        edge_dim,
        share_relation_weights=True,
    )
    if mpnn_type == "HeteroPNA":
        args["pna_deg"] = [1, 2, 1]

    with pytest.raises(ValueError, match="identical edge feature widths"):
        create_model(**args)


def test_heterogeneous_pooling_pads_missing_node_types():
    model = create_model(**_graph_head_model_args("HeteroSAGE", None))
    x_dict = {"a": torch.ones(2, 16), "b": torch.ones(1, 16)}
    batch_dict = {
        "a": torch.tensor([0, 1]),
        "b": torch.tensor([0]),
    }

    pooled = model._pool_hetero_graph_features(x_dict, batch_dict)

    assert pooled.shape == (2, 16)
    torch.testing.assert_close(pooled[1], torch.ones(16))


@pytest.mark.parametrize(
    ("mode", "module_name"),
    [
        ("film", "graph_conditioner"),
        ("concat_node", "graph_concat_projector"),
        ("fuse_pool", "graph_pool_projector"),
    ],
)
def test_graph_conditioning_parameters_exist_before_optimizer_and_preserve_dtype(
    mode, module_name
):
    args = _graph_head_model_args(
        "HeteroSAGE",
        None,
        use_graph_attr_conditioning=True,
        graph_attr_dim=2,
        graph_attr_conditioning_mode=mode,
    )
    model = create_model(**args).double()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    conditioning_params = list(getattr(model, module_name).parameters())
    assert conditioning_params
    assert all(id(param) in optimizer_param_ids for param in conditioning_params)

    data = _build_simple_hetero_graph()
    data.graph_attr = torch.tensor([1.0, 2.0], dtype=torch.float32)
    model.eval()
    output = model(data)[0]

    assert output.dtype == torch.float64


def test_hetero_heat_with_gps_unpacks_local_output():
    data = _build_simple_hetero_graph(edge_dim=3)
    args = _graph_head_model_args(
        "HeteroHEAT",
        3,
        global_attn_engine="GPS",
        global_attn_type="multihead",
    )
    model = create_model(**args)
    model.eval()

    output = model(data)[0]

    assert output.shape == (1, 2)


def _build_random_hetero_graph(
    rng: torch.Generator,
    input_dim: int = 4,
    edge_dim: int = None,
    num_nodes_a: int = 5,
    num_nodes_b: int = 4,
):
    data = HeteroData()

    # Graph-level latent signals for higher statistical quality.
    z_a = torch.randn(input_dim, generator=rng)
    z_b = torch.randn(input_dim, generator=rng)

    data["a"].x = z_a + 0.2 * torch.randn(num_nodes_a, input_dim, generator=rng)
    data["b"].x = z_b + 0.2 * torch.randn(num_nodes_b, input_dim, generator=rng)

    num_edges_ab = max(2 * num_nodes_a, 4)
    src_ab = torch.randint(0, num_nodes_a, (num_edges_ab,), generator=rng)
    dst_ab = torch.randint(0, num_nodes_b, (num_edges_ab,), generator=rng)
    src_ab = torch.cat([src_ab, torch.arange(num_nodes_a)])
    dst_ab = torch.cat(
        [dst_ab, torch.randint(0, num_nodes_b, (num_nodes_a,), generator=rng)]
    )
    data[("a", "to", "b")].edge_index = torch.stack([src_ab, dst_ab], dim=0)

    num_edges_ba = max(2 * num_nodes_b, 4)
    src_ba = torch.randint(0, num_nodes_b, (num_edges_ba,), generator=rng)
    dst_ba = torch.randint(0, num_nodes_a, (num_edges_ba,), generator=rng)
    src_ba = torch.cat([src_ba, torch.arange(num_nodes_b)])
    dst_ba = torch.cat(
        [dst_ba, torch.randint(0, num_nodes_a, (num_nodes_b,), generator=rng)]
    )
    data[("b", "to", "a")].edge_index = torch.stack([src_ba, dst_ba], dim=0)

    if edge_dim is not None:
        edge_index_ab = data[("a", "to", "b")].edge_index
        edge_index_ba = data[("b", "to", "a")].edge_index

        xa = data["a"].x[edge_index_ab[0]]
        xb = data["b"].x[edge_index_ab[1]]
        base_ab = 0.5 * (xa + xb).mean(dim=1, keepdim=True)
        data[("a", "to", "b")].edge_attr = base_ab.repeat(
            1, edge_dim
        ) + 0.05 * torch.randn(base_ab.size(0), edge_dim, generator=rng)

        xb = data["b"].x[edge_index_ba[0]]
        xa = data["a"].x[edge_index_ba[1]]
        base_ba = 0.5 * (xa + xb).mean(dim=1, keepdim=True)
        data[("b", "to", "a")].edge_attr = base_ba.repeat(
            1, edge_dim
        ) + 0.05 * torch.randn(base_ba.size(0), edge_dim, generator=rng)

    edge_index_ab = data[("a", "to", "b")].edge_index
    edge_index_ba = data[("b", "to", "a")].edge_index

    edge_signal_ab = (
        data["a"].x[edge_index_ab[0]].mean(dim=1)
        * data["b"].x[edge_index_ab[1]].mean(dim=1)
    ).mean()
    edge_signal_ba = (
        data["b"].x[edge_index_ba[0]].mean(dim=1)
        * data["a"].x[edge_index_ba[1]].mean(dim=1)
    ).mean()

    edge_attr_signal = 0.0
    if edge_dim is not None:
        edge_attr_signal = (
            data[("a", "to", "b")].edge_attr.mean()
            + data[("b", "to", "a")].edge_attr.mean()
        )

    graph_value = (
        0.6 * z_a.mean()
        + 0.4 * z_b.mean()
        + 0.25 * edge_signal_ab
        + 0.25 * edge_signal_ba
        + 0.1 * edge_attr_signal
        + 0.02 * torch.randn((), generator=rng)
    )
    data.y = graph_value.view(1, 1)

    return data


def _build_random_hetero_dataset(
    num_graphs: int,
    input_dim: int,
    edge_dim: int,
    seed: int = 0,
):
    rng = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(num_graphs):
        num_nodes_a = int(torch.randint(3, 7, (1,), generator=rng).item())
        num_nodes_b = int(torch.randint(2, 6, (1,), generator=rng).item())
        dataset.append(
            _build_random_hetero_graph(
                rng,
                input_dim=input_dim,
                edge_dim=edge_dim,
                num_nodes_a=num_nodes_a,
                num_nodes_b=num_nodes_b,
            )
        )
    return dataset


class _HeteroBatchAdapter:
    def __init__(self, loader, node_type="a"):
        self.loader = loader
        self.node_type = node_type
        self.dataset = loader.dataset
        self.sampler = loader.sampler
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for data in self.loader:
            if not hasattr(data, "batch"):
                data.batch = data[self.node_type].batch
            yield data


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,edge_dim,pna_deg",
    [
        ("HeteroGIN", 3, None),
        ("HeteroSAGE", 3, None),
        ("HeteroGAT", 3, None),
        ("HeteroRGAT", 3, None),
        ("HeteroHGT", 3, None),
        ("HeteroHEAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def test_hetero_graph_head_forward(mpnn_type, edge_dim, pna_deg):
    data = _build_simple_hetero_graph(edge_dim=edge_dim)

    output_heads = {
        "graph": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [8],
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [2],
        "pe_dim": 1,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["graph"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
    }
    if edge_dim is not None:
        config_args["edge_dim"] = edge_dim
    if pna_deg is not None:
        config_args["pna_deg"] = pna_deg

    model = create_model(**config_args)
    model.eval()

    outputs = model(data)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (1, 2)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,edge_dim,pna_deg",
    [
        ("HeteroGIN", None, None),
        ("HeteroSAGE", None, None),
        ("HeteroGAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def pytest_hetero_node_conv_head_forward(mpnn_type, edge_dim, pna_deg):
    data = _build_simple_hetero_graph(edge_dim=edge_dim)

    output_heads = {
        "node": {
            "num_headlayers": 2,
            "dim_headlayers": [16, 8],
            "type": "conv",
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [1],
        "pe_dim": 1,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
        "node_target_type": "a",
    }
    if edge_dim is not None:
        config_args["edge_dim"] = edge_dim
    if pna_deg is not None:
        config_args["pna_deg"] = pna_deg

    model = create_model(**config_args)
    model.eval()

    outputs = model(data)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (data["a"].num_nodes, 1)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,edge_dim,pna_deg",
    [
        ("HeteroGIN", None, None),
        ("HeteroSAGE", None, None),
        ("HeteroGAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def pytest_hetero_mpnn_training_randomized_dataset(mpnn_type, edge_dim, pna_deg):
    torch.manual_seed(7)

    dataset = _build_random_hetero_dataset(
        num_graphs=30000,
        input_dim=4,
        edge_dim=edge_dim,
        seed=13,
    )
    trainset, valset, testset = split_dataset(
        dataset, perc_train=0.8, stratify_splitting=False
    )
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, batch_size=8
    )
    train_loader = _HeteroBatchAdapter(train_loader)
    val_loader = _HeteroBatchAdapter(val_loader)
    test_loader = _HeteroBatchAdapter(test_loader)

    output_heads = {
        "graph": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 2,
            "dim_headlayers": [16, 8],
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [1],
        "pe_dim": 0,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["graph"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
    }
    if edge_dim is not None:
        config_args["edge_dim"] = edge_dim
    if pna_deg is not None:
        config_args["pna_deg"] = pna_deg

    verbosity = 0
    setup_ddp()

    model = create_model(**config_args)
    model = get_distributed_model(model, verbosity)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1.0e-5
    )

    nn_config = {
        "Training": {
            "num_epoch": 10,
            "conv_checkpointing": False,
        },
        "Variables_of_interest": {"output_names": ["y"]},
    }

    log_name = "hetero_mpnn_randomized"

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        None,
        scheduler,
        nn_config,
        log_name,
        verbosity,
        create_plots=False,
    )

    final_loss, _, true_values, predicted_values = hydragnn.train.test(
        test_loader,
        model,
        verbosity,
        num_tasks=1,
        precision="fp32",
    )

    mae = torch.nn.L1Loss()
    final_mae = mae(true_values[0], predicted_values[0])

    thresholds = {
        "HeteroGIN": [0.09, 0.26],
        "HeteroSAGE": [0.09, 0.27],
        "HeteroGAT": [0.13, 0.30],
        "HeteroPNA": [0.11, 0.28],
    }

    assert torch.isfinite(torch.tensor(final_loss))
    assert torch.isfinite(final_mae)
    assert final_loss < thresholds[mpnn_type][0]
    assert final_mae < thresholds[mpnn_type][1]
