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

import os
import json

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.models.create import create_model_config


def _tiny_loader(num_graphs=4, num_nodes=6):
    """Build a tiny in-memory loader mirroring the LennardJones example.

    Each graph carries a single scalar node feature (``atom_type``),
    3D positions, atomic numbers, and optional per-graph charge / spin
    scalars so both the UMA and AllScAIP conditioning paths are
    exercised at model-construction time.
    """
    torch.manual_seed(0)
    data_list = []
    for _ in range(num_graphs):
        pos = torch.randn(num_nodes, 3)
        atom_type = torch.randint(1, 10, (num_nodes, 1)).float()
        d = Data(x=atom_type, pos=pos)
        d.atomic_numbers = atom_type.view(-1).long()
        d.charge = torch.zeros(1, dtype=torch.long)
        d.spin = torch.zeros(1, dtype=torch.long)
        data_list.append(d)
    return DataLoader(data_list, batch_size=2, shuffle=False)


def _vector_head_loader(num_graphs=4, num_nodes=6):
    """Loader carrying an energy graph target and a 3-dim node force target.

    ``y``/``y_loc`` encode two heads (energy dim 1, forces dim 3) so
    ``update_config`` infers the ``[1, 3]`` output dims required by the
    equivariant-vector-head example config.
    """
    torch.manual_seed(0)
    data_list = []
    for _ in range(num_graphs):
        pos = torch.randn(num_nodes, 3)
        atom_type = torch.randint(1, 10, (num_nodes, 1)).float()
        energy = torch.randn(1)
        forces = torch.randn(3 * num_nodes)
        d = Data(x=atom_type, pos=pos)
        d.atomic_numbers = atom_type.view(-1).long()
        d.charge = torch.zeros(1, dtype=torch.long)
        d.spin = torch.zeros(1, dtype=torch.long)
        d.y = torch.cat([energy, forces])
        d.y_loc = torch.tensor([[0, 1, 1 + 3 * num_nodes]], dtype=torch.long)
        data_list.append(d)
    return DataLoader(data_list, batch_size=2, shuffle=False)


def _load_example_config(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "examples", "LennardJones", filename)
    with open(path, "r") as f:
        config = json.load(f)
    # The example driver stamps the feature bookkeeping onto
    # Variables_of_interest from the Dataset block; replicate that here.
    voi = config["NeuralNetwork"]["Variables_of_interest"]
    voi["graph_feature_names"] = config["Dataset"]["graph_features"]["name"]
    voi["graph_feature_dims"] = config["Dataset"]["graph_features"]["dim"]
    voi["node_feature_names"] = config["Dataset"]["node_features"]["name"]
    voi["node_feature_dims"] = config["Dataset"]["node_features"]["dim"]
    return config


@pytest.mark.parametrize(
    "filename, expected_mpnn, expected_str",
    [
        ("LJ_UMA.json", "UMA", "UMAStack"),
        ("LJ_AllScAIP.json", "AllScAIP", "AllScAIPStack"),
    ],
)
def pytest_example_config_builds_model(filename, expected_mpnn, expected_str):
    """The shipped UMA / AllScAIP example configs build a valid model."""
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "0"
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        config = _load_example_config(filename)
        assert config["NeuralNetwork"]["Architecture"]["mpnn_type"] == expected_mpnn

        loader = _tiny_loader()
        update_config(config, loader, loader, loader)

        model = create_model_config(
            config=config["NeuralNetwork"], verbosity=0, use_gpu=False
        )
        # enable_interatomic_potential wraps the backbone in
        # EnhancedModelWrapper; unwrap to inspect the underlying stack.
        inner = getattr(model, "model", model)
        assert str(inner) == expected_str
        if expected_mpnn == "UMA":
            assert inner.uma_periodic is True
        # Model has learnable parameters and a backbone attribute.
        assert sum(p.numel() for p in model.parameters()) > 0
    finally:
        torch.set_default_dtype(prev_dtype)
        os.environ.pop("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", None)


@pytest.mark.parametrize("uma_variant", ["S", "M", "L"])
def pytest_uma_variant_config_builds_model(uma_variant):
    """The UMA example config builds for every S / M / L capacity tier."""
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "0"
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        config = _load_example_config("LJ_UMA.json")
        config["NeuralNetwork"]["Architecture"]["uma_variant"] = uma_variant

        loader = _tiny_loader()
        update_config(config, loader, loader, loader)

        model = create_model_config(
            config=config["NeuralNetwork"], verbosity=0, use_gpu=False
        )
        inner = getattr(model, "model", model)
        assert str(inner) == "UMAStack"
        assert inner.uma_variant == uma_variant
        if uma_variant == "S":
            assert inner.uma_num_experts == 0
        else:
            assert inner.uma_num_experts > 0
    finally:
        torch.set_default_dtype(prev_dtype)
        os.environ.pop("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", None)


def pytest_uma_equivariant_vector_head_config_builds_model():
    """The equivariant-vector-head example config builds and wires the head.

    Mirrors ``LJ_UMA_equivariant_vector.json``: an energy graph head plus a
    3-dim ``forces`` node head with ``uma_equivariant_vector_head`` enabled.
    The dim-3 node head must be auto-detected and backed by the equivariant
    SO(3) vector head rather than a scalar MLP readout.
    """
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "0"
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        config = _load_example_config("LJ_UMA_equivariant_vector.json")
        arch = config["NeuralNetwork"]["Architecture"]
        assert arch["uma_equivariant_vector_head"] is True

        loader = _vector_head_loader()
        update_config(config, loader, loader, loader)

        model = create_model_config(
            config=config["NeuralNetwork"], verbosity=0, use_gpu=False
        )
        inner = getattr(model, "model", model)
        assert str(inner) == "UMAStack"
        assert inner.equivariant_vector_head is not None
        # The dim-3 'forces' node head (index 1) is the equivariant target.
        assert inner._vector_head_index == 1
        assert inner.head_dims[inner._vector_head_index] == 3
    finally:
        torch.set_default_dtype(prev_dtype)
        os.environ.pop("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", None)
