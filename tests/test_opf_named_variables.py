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
import importlib.util
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

_MODULE_PATH = Path(__file__).parents[1] / "examples" / "opf" / "opf_solution_utils.py"
_SPEC = importlib.util.spec_from_file_location("opf_solution_utils", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _named_sample():
    data = HeteroData()
    data.context = torch.tensor([[1.0]])
    data["bus"].node_features = torch.randn(3, 4)
    data["bus"].bus_solution = torch.randn(3, 2)
    data["load"].node_features = torch.randn(2, 2)
    data["bus", "ac_line", "bus"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    data["bus", "ac_line", "bus"].ac_line_features = torch.randn(2, 9)
    return data


def test_named_hetero_opf_fields_build_internal_tensors():
    data = _MODULE.compile_named_hetero_opf_sample(_named_sample(), "bus")

    assert data["bus"].x is data["bus"].node_features
    assert data["load"].x is data["load"].node_features
    assert data["bus"].y is data["bus"].bus_solution
    assert data.y.shape == (3, 2)
    assert data.graph_attr.shape == (1, 1)
    torch.testing.assert_close(
        data["bus", "ac_line", "bus"].edge_attr,
        data["bus", "ac_line", "bus"].ac_line_features,
    )


def test_prepared_opf_dataset_rejects_legacy_unnamed_target():
    data = _MODULE.compile_named_hetero_opf_sample(_named_sample(), "bus")
    del data["bus"].bus_solution
    data["bus"].y = torch.randn(3, 2)
    adapter = _MODULE.NodeTargetDatasetAdapter([data], "bus", edge_dim={"ac_line": 9})

    with pytest.raises(RuntimeError, match="missing named target"):
        adapter[0]
