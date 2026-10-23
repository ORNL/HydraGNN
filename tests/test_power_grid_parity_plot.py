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
"""Power-grid inference parity plots split flattened node outputs per component."""

import importlib.util
import os

import pytest
import torch

pytest.importorskip("mpi4py.MPI")

from tests._script_runner import REPO_ROOT  # noqa: E402

SCRIPTS = [
    ("pglearn", "infer_pglearn_solution_homogeneous.py"),
    ("opflearn", "infer_opflearn_solution_homogeneous.py"),
]


def _load(subdir, name):
    path = os.path.join(REPO_ROOT, "examples", subdir, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("subdir,name", SCRIPTS)
def test_plot_parity_one_figure_per_component(tmp_path, monkeypatch, subdir, name):
    module = _load(subdir, name)
    plotted = []
    monkeypatch.setattr(
        module.plt.Axes,
        "scatter",
        lambda self, t, p, **kw: plotted.append((t.copy(), p.copy())),
    )
    nodes, dim = 5, 2
    true = torch.arange(nodes * dim, dtype=torch.float32)
    module._plot_parity(true, true + 100, str(tmp_path), dim, name="bus")

    assert len(plotted) == dim
    for component, (t, p) in enumerate(plotted):
        assert t.tolist() == true.reshape(nodes, dim)[:, component].tolist()
        assert (p - t).tolist() == [100.0] * nodes
    assert sorted(os.listdir(tmp_path)) == [
        "parity_bus_dim%d.png" % i for i in range(dim)
    ]
