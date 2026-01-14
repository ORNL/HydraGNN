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

import os
import json
import importlib.util

import pytest
import torch
import torch_geometric
from torch_geometric.transforms import AddLaplacianEigenvectorPE

import hydragnn


def _load_example_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("graph_attr_mode", ["concat_node", "film", "fuse_pool"])
@pytest.mark.parametrize("example", ["qm9", "md17"])
@pytest.mark.mpi_skip()
def pytest_examples_graph_attr(tmp_path, example, graph_attr_mode):
    examples_root = os.path.join(os.path.dirname(__file__), "..", "examples", example)

    if example == "qm9":
        qm9_module = _load_example_module(
            os.path.join(examples_root, "qm9.py"), "qm9_module"
        )
        with open(os.path.join(examples_root, "qm9.json")) as f:
            config = json.load(f)
        arch = config["NeuralNetwork"].setdefault("Architecture", {})
        arch.setdefault("use_graph_attr_conditioning", True)
        arch["graph_attr_conditioning_mode"] = graph_attr_mode
        pe_dim = config["NeuralNetwork"]["Architecture"]["pe_dim"]
        transform = AddLaplacianEigenvectorPE(
            k=pe_dim, attr_name="pe", is_undirected=True
        )
        dataset = torch_geometric.datasets.QM9(
            root=os.path.join(tmp_path, "qm9"),
            pre_transform=lambda data: qm9_module.qm9_pre_transform(data, transform),
            pre_filter=qm9_module.qm9_pre_filter,
        )
    else:
        md17_module = _load_example_module(
            os.path.join(examples_root, "md17.py"), "md17_module"
        )
        with open(os.path.join(examples_root, "md17.json")) as f:
            config = json.load(f)
        arch = config["NeuralNetwork"].setdefault("Architecture", {})
        arch.setdefault("use_graph_attr_conditioning", True)
        arch["graph_attr_conditioning_mode"] = graph_attr_mode
        arch_config = config["NeuralNetwork"]["Architecture"]
        transform = AddLaplacianEigenvectorPE(
            k=arch_config["pe_dim"], attr_name="pe", is_undirected=True
        )
        compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)
        torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"
        dataset = torch_geometric.datasets.MD17(
            root=os.path.join(tmp_path, "md17"),
            name="uracil",
            pre_transform=lambda data: md17_module.md17_pre_transform(
                data, compute_edges, transform
            ),
            pre_filter=md17_module.md17_pre_filter,
        )

    sample = dataset[0]
    expected = torch.tensor([0.0, 1.0], dtype=torch.float32)
    assert hasattr(sample, "graph_attr")
    assert torch.allclose(sample.graph_attr, expected)
