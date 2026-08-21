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

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("rdkit")

from hydragnn.utils.descriptors_and_embeddings.smiles_utils import (
    generate_graphdata_from_smilestr,
)


def pytest_csce_smiles_features_are_compiled_from_named_attributes():
    config_path = Path(__file__).parents[1] / "examples" / "csce" / "csce_gap.json"
    with config_path.open() as stream:
        variables = json.load(stream)["Variables"]

    node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}
    target = torch.tensor([[1.25]], dtype=torch.float32)
    data = generate_graphdata_from_smilestr(
        "CO", target, node_types, var_config=variables
    )

    assert data.atom_type.shape == (data.num_nodes, 6)
    assert data.atom_descriptors.shape == (data.num_nodes, 6)
    torch.testing.assert_close(
        data.x,
        torch.cat([data.atom_type, data.atom_descriptors], dim=1),
    )
    assert data.x.shape == (data.num_nodes, 12)
    torch.testing.assert_close(data.GAP, target)
    torch.testing.assert_close(data.y, target)
    assert data.y_loc.tolist() == [[0, 1]]
    assert "edge_attr" not in data
    assert data.bond_attributes.shape[1] == 4


def pytest_smiles_edge_outputs_use_one_row_per_edge():
    variables = {
        "inputs": [
            {"name": "atom_type", "level": "node", "dim": 6},
            {"name": "atom_descriptors", "level": "node", "dim": 6},
        ],
        "outputs": [{"name": "bond_target", "level": "edge", "dim": 1}],
    }
    node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}
    # "CO" has two directed edges, one in each direction.
    target = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    data = generate_graphdata_from_smilestr(
        "CO", target, node_types, var_config=variables
    )

    assert data.bond_target.shape == (data.num_edges, 1)
    torch.testing.assert_close(data.bond_target, target)
    assert data.edge_output.shape == (data.num_edges, 1)
