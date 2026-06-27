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

"""
Parquet-free smoke tests for the TemporalGCN integration.

Regression guard for PR #1 of the T-GCN -> HydraGNN port:
  * C1 — weighted edges: data.edge_weight must reach GCNConv (the geographic
         exp(-d/sigma) similarity that the standalone T-GCN relies on).
  * C2 — temporal_batch_norm: the BatchNorm between GCN layers must be skippable
         so the model can reproduce the standalone's bare-ReLU stack.

These run in-process on tiny synthetic data (no parquet, no DDP, no real PMU
files), so they complete in well under a second on CPU. They build the model
through the real factory path (update_config -> create_model_config), so a
regression in the conv-string wiring, the TemporalBase._embedding override, or
the BN gating will fail here.
"""

import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, Batch

try:
    from torch_geometric.loader import DataLoader
except ImportError:  # older PyG
    from torch_geometric.data import DataLoader

import hydragnn


N_NODES = 5
N_FEAT = 2
LOOKBACK = 6
T_STEPS = 24
SEED = 0


# --------------------------------------------------------------------------- #
# Synthetic data + config helpers
# --------------------------------------------------------------------------- #


def _build_graph(num_nodes=N_NODES, seed=SEED):
    """Undirected ring with non-uniform positive edge weights.

    Weights live in (0.1, 1.0) — deliberately non-uniform so that the weighted
    GCN normalization differs from the binary (unweighted) one, which is what
    the C1 test relies on.
    """
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(
        rng.uniform(0.1, 1.0, size=edge_index.shape[1]), dtype=torch.float32
    )
    return edge_index, edge_weight


def _make_dataset(edge_index, edge_weight=None, seed=SEED):
    """Sliding-window Data objects mirroring the fnet/synthetic x_seq contract.

    Each window: x [N,F] (last step), x_seq [N,lookback,F], y [N,1] (next-step
    feature 0), y_loc [[0,N]], pos/batch, and optionally edge_weight [E].
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T_STEPS, N_NODES, N_FEAT)).astype(np.float32)
    pos = torch.zeros(N_NODES, 3)
    batch = torch.zeros(N_NODES, dtype=torch.long)
    y_loc = torch.tensor([[0, N_NODES]], dtype=torch.int64)

    dataset = []
    for s in range(LOOKBACK - 1, T_STEPS - 1):
        x_seq = (
            torch.from_numpy(X[s - LOOKBACK + 1 : s + 1]).permute(1, 0, 2).contiguous()
        )
        d = Data(
            x=torch.from_numpy(X[s]),
            x_seq=x_seq,
            edge_index=edge_index.clone(),
            y=torch.from_numpy(X[s + 1, :, 0:1]),
            y_loc=y_loc.clone(),
            pos=pos,
            batch=batch,
            num_nodes=N_NODES,
        )
        if edge_weight is not None:
            d.edge_weight = edge_weight.clone()
        dataset.append(d)
    return dataset


def _base_config(temporal_mode="post_gcn", temporal_batch_norm=True):
    """Minimal valid config; update_config fills the rest (input_dim, num_nodes,
    pna_deg=None, GPS defaults, etc.) exactly as for the shipped examples."""
    return {
        "Verbosity": {"level": 0},
        "NeuralNetwork": {
            "Architecture": {
                "mpnn_type": "TemporalGCN",
                "temporal_backbone": "gru",
                "temporal_mode": temporal_mode,
                "temporal_num_layers": 1,
                "temporal_batch_norm": temporal_batch_norm,
                "hidden_dim": 8,
                "num_conv_layers": 2,
                "max_neighbours": 10,
                "output_heads": {
                    "node": {
                        "num_headlayers": 1,
                        "dim_headlayers": [8],
                        "type": "mlp",
                    }
                },
                "task_weights": [1.0],
            },
            "Variables_of_interest": {
                "input_node_features": [0, 1],
                "output_names": ["r_next"],
                "output_index": [0],
                "output_dim": [1],
                "type": ["node"],
                "denormalize_output": False,
            },
            "Training": {
                "num_epoch": 1,
                "perc_train": 0.7,
                "loss_function_type": "mse",
                "conv_checkpointing": False,
                "batch_size": 4,
                "Optimizer": {"type": "AdamW", "learning_rate": 1e-3},
            },
        },
    }


def _build_model(config, dataset):
    loader = DataLoader(
        dataset, batch_size=config["NeuralNetwork"]["Training"]["batch_size"]
    )
    # update_config's default graph-size check (check_if_graph_size_variable_dist)
    # asserts an initialized process group and does an all_reduce. Our graphs are
    # fixed-size, so we tell HydraGNN that explicitly via
    # HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=0, which skips that dist call and keeps the
    # test single-process (no DDP / no libuv on Windows). Restore the var so it
    # cannot leak into the subprocess-based example tests.
    _prev = os.environ.get("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE")
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "0"
    try:
        # update_config infers input_dim/output_dim/num_nodes from a sample batch
        # and populates every remaining Architecture key create_model_config wants.
        config = hydragnn.utils.input_config_parsing.update_config(
            config, loader, loader, loader
        )
    finally:
        if _prev is None:
            os.environ.pop("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", None)
        else:
            os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = _prev
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=0
    )
    return model, loader


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("temporal_mode", ["post_gcn", "pre_gcn", "interleaved"])
@pytest.mark.parametrize("temporal_batch_norm", [True, False])
def pytest_temporalgcn_forward_runs(temporal_mode, temporal_batch_norm):
    """C1+C2: a weighted-edge TemporalGCN runs forward in every temporal mode,
    with BatchNorm on and off, producing finite per-node output of shape
    [n_graphs * N, output_dim]."""
    torch.manual_seed(SEED)
    edge_index, edge_weight = _build_graph()
    dataset = _make_dataset(edge_index, edge_weight)
    model, loader = _build_model(
        _base_config(temporal_mode, temporal_batch_norm), dataset
    )

    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)

    assert isinstance(out, list) and len(out) == 1
    head = out[0]
    n_graphs = int(batch.batch.max().item()) + 1
    assert head.shape == (n_graphs * N_NODES, 1)
    assert torch.isfinite(head).all()


@pytest.mark.mpi_skip()
def pytest_edge_weight_changes_output():
    """C1: edge weights actually reach GCNConv.

    With BatchNorm off and the model in eval mode, the same model run on two
    inputs that differ ONLY in the presence of edge_weight must produce
    different outputs. If edge_weight were dropped (the pre-fix behavior),
    GCNConv would see a binary graph in both cases and the outputs would be
    identical.
    """
    torch.manual_seed(SEED)
    edge_index, edge_weight = _build_graph()

    ds_weighted = _make_dataset(edge_index, edge_weight)
    model, _ = _build_model(
        _base_config("post_gcn", temporal_batch_norm=False), ds_weighted
    )
    model.eval()

    ds_unweighted = _make_dataset(edge_index, edge_weight=None)  # same X + graph
    batch_w = Batch.from_data_list([ds_weighted[0]])
    batch_u = Batch.from_data_list([ds_unweighted[0]])

    with torch.no_grad():
        out_w = model(batch_w)[0]
        out_u = model(batch_u)[0]

    assert torch.isfinite(out_w).all() and torch.isfinite(out_u).all()
    assert not torch.allclose(out_w, out_u, atol=1e-6), (
        "edge_weight had no effect: weighted and unweighted outputs are "
        "identical — the C1 edge_weight path is broken."
    )


@pytest.mark.mpi_skip()
def pytest_batch_norm_flag_changes_training_output():
    """C2: temporal_batch_norm actually gates the BatchNorm.

    In train() mode BatchNorm normalizes by batch statistics, so toggling the
    flag on one model (same weights, same input) must change the output. (In
    eval mode with untrained running stats the two paths are ~identical, so the
    behavioral check is done in train mode.)
    """
    torch.manual_seed(SEED)
    edge_index, edge_weight = _build_graph()
    dataset = _make_dataset(edge_index, edge_weight)
    model, loader = _build_model(
        _base_config("post_gcn", temporal_batch_norm=True), dataset
    )
    batch = next(iter(loader))

    model.train()
    with torch.no_grad():
        model._temporal_batch_norm = True
        out_bn = model(batch)[0]
        model._temporal_batch_norm = False
        out_nobn = model(batch)[0]

    assert torch.isfinite(out_bn).all() and torch.isfinite(out_nobn).all()
    assert not torch.allclose(out_bn, out_nobn, atol=1e-6), (
        "temporal_batch_norm flag had no effect on the forward pass."
    )


@pytest.mark.mpi_skip()
def pytest_batch_norm_off_removes_bn_params():
    """C2/DDP guard: with temporal_batch_norm=False the BatchNorm feature layers
    are replaced by Identity, so there are no unused parameters. (Skipping BN in
    the forward pass but leaving the modules registered makes DDP with
    find_unused_parameters=False raise 'parameters that were not used in producing
    loss'.)"""
    import torch.nn as nn

    edge_index, edge_weight = _build_graph()
    dataset = _make_dataset(edge_index, edge_weight)

    model_off, _ = _build_model(
        _base_config("post_gcn", temporal_batch_norm=False), dataset
    )
    assert all(
        isinstance(m, nn.Identity) for m in model_off.feature_layers
    ), "feature_layers must be Identity when temporal_batch_norm=False"

    model_on, _ = _build_model(
        _base_config("post_gcn", temporal_batch_norm=True), dataset
    )
    assert not any(
        isinstance(m, nn.Identity) for m in model_on.feature_layers
    ), "feature_layers must keep BatchNorm when temporal_batch_norm=True"
