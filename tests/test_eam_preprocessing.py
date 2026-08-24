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
import numpy as np
import torch

from examples.eam import eam_preprocess


def _config(outputs):
    return {
        "Variables": {
            "inputs": [{"name": "node_features", "level": "node", "dim": 1}],
            "outputs": outputs,
        }
    }


def test_eam_cfg_parser_exposes_named_attributes(monkeypatch, tmp_path):
    cfg = tmp_path / "sample.cfg"
    cfg.touch()
    cfg.with_suffix(".bulk").write_text("42.5\n", encoding="utf-8")

    class FakeAtoms:
        arrays = {
            "numbers": np.array([28, 41]),
            "c_peratom": np.array([-1.0, -2.0]),
            "fx": np.array([0.1, 0.2]),
            "fy": np.array([0.3, 0.4]),
            "fz": np.array([0.5, 0.6]),
        }
        positions = np.zeros((2, 3))

        class Cell:
            array = np.eye(3) * 10

        cell = Cell()

        def __len__(self):
            return len(self.arrays["numbers"])

    atoms = FakeAtoms()
    monkeypatch.setattr(eam_preprocess, "read_cfg", lambda _: atoms)
    sample, schema = eam_preprocess.parse_eam_cfg(
        cfg,
        _config(
            [
                {"name": "bulk_modulus", "level": "graph", "dim": 1},
                {"name": "atomic_energy", "level": "node", "dim": 1},
                {"name": "atomic_forces", "level": "node", "dim": 3},
            ]
        ),
    )

    assert sample.node_features.shape == (2, 1)
    assert sample.atomic_energy.shape == (2, 1)
    assert sample.atomic_forces.shape == (2, 3)
    torch.testing.assert_close(sample.bulk_modulus, torch.tensor([[42.5]]))
    assert len(schema.outputs) == 3
