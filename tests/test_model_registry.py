##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Fast per-architecture contracts for every registered ``mpnn_type``.

Each model is built from a tiny config and checked for forward shapes,
finite gradients, state-dict round trips, seeded determinism and node
permutation behaviour. None of these tests train a model.
"""

import copy

import pytest
import torch
from torch_geometric.data import Batch, Data, HeteroData

from hydragnn.models.create import create_model
from hydragnn.models.registry import (
    MODEL_REGISTRY,
    available_models,
    get_model_builder,
    register_model,
)
from hydragnn.utils.model import update_multibranch_heads

pytestmark = pytest.mark.mpi_skip()

HOMOGENEOUS_MODELS = [
    "AllScAIP",
    "CGCNN",
    "DimeNet",
    "EGNN",
    "GAT",
    "GIN",
    "MACE",
    "MFC",
    "PAINN",
    "PNA",
    "PNAEq",
    "PNAPlus",
    "SAGE",
    "SchNet",
    "UMA",
]
HETEROGENEOUS_MODELS = [
    "HeteroGAT",
    "HeteroGIN",
    "HeteroHEAT",
    "HeteroHGT",
    "HeteroPNA",
    "HeteroRGAT",
    "HeteroSAGE",
]

HIDDEN_DIM = 8
NODE_DIM = 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_every_supported_model():
    assert available_models() == sorted(HOMOGENEOUS_MODELS + HETEROGENEOUS_MODELS)


def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown mpnn_type: NotAModel"):
        get_model_builder("NotAModel")


def test_create_model_rejects_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown mpnn_type: NotAModel"):
        create_model(**{**_homogeneous_args("GIN"), "mpnn_type": "NotAModel"})


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_model("GIN")(lambda p: None)
    assert MODEL_REGISTRY["GIN"].__name__ == "_build_gin"


# ---------------------------------------------------------------------------
# Homogeneous fixtures
# ---------------------------------------------------------------------------


def _homogeneous_args(mpnn_type):
    heads = update_multibranch_heads(
        {
            "graph": {
                "num_sharedlayers": 1,
                "dim_sharedlayers": 4,
                "num_headlayers": 1,
                "dim_headlayers": [4],
            },
            "node": {"num_headlayers": 1, "dim_headlayers": [4], "type": "mlp"},
        }
    )
    args = dict(
        mpnn_type=mpnn_type,
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        output_dim=[1, NODE_DIM],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["graph", "node"],
        output_heads=heads,
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0, 1.0],
        num_conv_layers=2,
        max_neighbours=8,
        pna_deg=torch.tensor([0, 2, 3, 1]),
        num_before_skip=1,
        num_after_skip=1,
        num_radial=4,
        radial_type="bessel",
        basis_emb_size=4,
        int_emb_size=4,
        out_emb_size=8,
        envelope_exponent=5,
        num_spherical=2,
        num_gaussians=4,
        num_filters=4,
        radius=3.0,
        correlation=2,
        max_ell=1,
        node_max_ell=1,
        avg_num_neighbors=2.0,
        use_gpu=False,
    )
    if mpnn_type == "CGCNN":
        # Mirrors config_utils: CGCNN keeps hidden_dim == input_dim without GPS.
        args.update(edge_dim=0, hidden_dim=args["input_dim"])
    if mpnn_type == "MACE":
        args.update(num_nodes=5)
    if mpnn_type == "AllScAIP":
        args.update(allscaip_num_heads=2, allscaip_use_freq_mask=False)
    if mpnn_type == "UMA":
        args.update(max_ell=2, uma_edge_channels=8)
    return args


def _build_homogeneous(mpnn_type):
    try:
        return create_model(**_homogeneous_args(mpnn_type))
    except ImportError as exc:
        pytest.skip(f"{mpnn_type} dependencies unavailable: {exc}")


def _molecule(num_nodes, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(num_nodes, 3, generator=g)
    src, dst = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    z = torch.randint(1, 9, (num_nodes, 1), generator=g).float()
    return Data(
        x=z,
        atomic_numbers=z[:, 0].long(),
        pos=pos,
        edge_index=edge_index,
        edge_shifts=torch.zeros(edge_index.size(1), 3),
    )


def _batch():
    return Batch.from_data_list([_molecule(4, seed=1), _molecule(5, seed=2)])


def _permute(data, perm):
    """Relabel nodes so that new node ``i`` is old node ``perm[i]``."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    out = data.clone()
    for key in ("x", "atomic_numbers", "pos", "batch"):
        out[key] = data[key][perm]
    out.edge_index = inverse[data.edge_index]
    return out


# ---------------------------------------------------------------------------
# Homogeneous contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mpnn_type", HOMOGENEOUS_MODELS)
def test_forward_shapes_and_finite_gradients(mpnn_type):
    model = _build_homogeneous(mpnn_type)
    data = _batch()

    graph_out, node_out = model(data)[:2]

    assert graph_out.shape == (data.num_graphs, 1)
    assert node_out.shape == (data.num_nodes, NODE_DIM)
    (graph_out.square().sum() + node_out.square().sum()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{mpnn_type} produced no parameter gradients"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("mpnn_type", HOMOGENEOUS_MODELS)
def test_state_dict_round_trip_reproduces_outputs(mpnn_type, tmp_path):
    model = _build_homogeneous(mpnn_type).eval()
    data = _batch()
    with torch.no_grad():
        expected = [out.clone() for out in model(data)[:2]]
    torch.save(model.state_dict(), tmp_path / "model.pt")

    restored = _build_homogeneous(mpnn_type).eval()
    with torch.no_grad():
        for param in restored.parameters():
            if param.is_floating_point():
                param.add_(0.5)
    restored.load_state_dict(torch.load(tmp_path / "model.pt"), strict=True)

    with torch.no_grad():
        actual = restored(data)[:2]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("mpnn_type", HOMOGENEOUS_MODELS)
def test_construction_is_deterministic(mpnn_type):
    first = _build_homogeneous(mpnn_type).state_dict()
    second = _build_homogeneous(mpnn_type).state_dict()

    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], msg=key)


@pytest.mark.parametrize("mpnn_type", HOMOGENEOUS_MODELS)
def test_node_permutation_equivariance(mpnn_type):
    model = _build_homogeneous(mpnn_type).eval()
    data = _batch()
    # Permute within each graph so graph membership order is unchanged.
    perm = torch.cat([torch.tensor([2, 0, 3, 1]), torch.tensor([8, 4, 6, 5, 7])])

    with torch.no_grad():
        graph_out, node_out = model(data)[:2]
        graph_perm, node_perm = model(_permute(data, perm))[:2]

    torch.testing.assert_close(graph_perm, graph_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(node_perm, node_out[perm], rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Heterogeneous contracts
# ---------------------------------------------------------------------------

EDGE_TYPES = [("a", "to", "b"), ("b", "to", "a")]


def _heterogeneous_args(mpnn_type):
    heads = update_multibranch_heads(
        {
            "graph": {
                "num_sharedlayers": 1,
                "dim_sharedlayers": 4,
                "num_headlayers": 1,
                "dim_headlayers": [4],
            }
        }
    )
    return dict(
        mpnn_type=mpnn_type,
        input_dim=3,
        hidden_dim=HIDDEN_DIM,
        output_dim=[2],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["graph"],
        output_heads=heads,
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        edge_dim=2,
        pna_deg=[1, 2, 1],
        hetero_attention_heads=2,
        metadata=(["a", "b"], EDGE_TYPES),
        node_input_dims={"a": 3, "b": 3},
        use_gpu=False,
    )


def _hetero_graph(seed):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["a"].x = torch.randn(4, 3, generator=g)
    data["b"].x = torch.randn(3, 3, generator=g)
    data[EDGE_TYPES[0]].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]])
    data[EDGE_TYPES[1]].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    for edge_type in EDGE_TYPES:
        num_edges = data[edge_type].edge_index.size(1)
        data[edge_type].edge_attr = torch.randn(num_edges, 2, generator=g)
    return data


@pytest.mark.parametrize("mpnn_type", HETEROGENEOUS_MODELS)
def test_heterogeneous_forward_shapes_and_finite_gradients(mpnn_type):
    model = create_model(**_heterogeneous_args(mpnn_type))
    data = Batch.from_data_list([_hetero_graph(1), _hetero_graph(2)])

    output = model(data)[0]

    assert output.shape == (2, 2)
    output.square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{mpnn_type} produced no parameter gradients"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("mpnn_type", HETEROGENEOUS_MODELS)
def test_heterogeneous_state_dict_round_trip(mpnn_type):
    data = _hetero_graph(3)
    model = create_model(**_heterogeneous_args(mpnn_type)).eval()
    with torch.no_grad():
        expected = model(data)[0].clone()

    restored = create_model(**_heterogeneous_args(mpnn_type)).eval()
    with torch.no_grad():
        restored(data)  # materialize lazily-initialized parameters
        for param in restored.parameters():
            if param.is_floating_point():
                param.add_(0.5)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

    with torch.no_grad():
        torch.testing.assert_close(restored(data)[0], expected)
