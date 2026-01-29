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


def _build_random_hetero_graph(
    rng: torch.Generator,
    input_dim: int = 4,
    edge_dim: int = None,
    num_nodes_a: int = 5,
    num_nodes_b: int = 4,
):
    data = HeteroData()

    data["a"].x = torch.randn(num_nodes_a, input_dim, generator=rng)
    data["b"].x = torch.randn(num_nodes_b, input_dim, generator=rng)

    num_edges_ab = max(2, num_nodes_a)
    src_ab = torch.randint(0, num_nodes_a, (num_edges_ab,), generator=rng)
    dst_ab = torch.randint(0, num_nodes_b, (num_edges_ab,), generator=rng)
    data[("a", "to", "b")].edge_index = torch.stack([src_ab, dst_ab], dim=0)
    if edge_dim is not None:
        data[("a", "to", "b")].edge_attr = torch.randn(
            num_edges_ab, edge_dim, generator=rng
        )

    num_edges_ba = max(2, num_nodes_b)
    src_ba = torch.randint(0, num_nodes_b, (num_edges_ba,), generator=rng)
    dst_ba = torch.randint(0, num_nodes_a, (num_edges_ba,), generator=rng)
    data[("b", "to", "a")].edge_index = torch.stack([src_ba, dst_ba], dim=0)
    if edge_dim is not None:
        data[("b", "to", "a")].edge_attr = torch.randn(
            num_edges_ba, edge_dim, generator=rng
        )

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
        data["a"].x.mean()
        + 0.5 * data["b"].x.mean()
        + 0.25 * edge_signal_ab
        + 0.25 * edge_signal_ba
        + 0.1 * edge_attr_signal
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
        ("HeteroGIN", None, None),
        ("HeteroSAGE", None, None),
        ("HeteroGAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def pytest_hetero_graph_head_forward(mpnn_type, edge_dim, pna_deg):
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
        num_graphs=10000,
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
        "HeteroGIN": [0.070, 0.230],
        "HeteroSAGE": [0.070, 0.240],
        "HeteroGAT": [0.110, 0.270],
        "HeteroPNA": [0.090, 0.250],
    }

    assert torch.isfinite(torch.tensor(final_loss))
    assert torch.isfinite(final_mae)
    assert final_loss < thresholds[mpnn_type][0]
    assert final_mae < thresholds[mpnn_type][1]
