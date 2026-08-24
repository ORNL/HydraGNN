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

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.utils.materials import normalize_stress, validate_materials_sample
from hydragnn.utils.materials.preprocessing import (
    GPA_PER_EV_PER_ANGSTROM_CUBED,
)


def _sample(**updates):
    values = {
        "pos": torch.zeros(2, 3),
        "atomic_numbers": torch.tensor([[1.0], [8.0]]),
        "forces": torch.zeros(2, 3),
        "cell": torch.eye(3),
        "stress": torch.eye(3),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }
    values.update(updates)
    return Data(**values)


def test_normalize_vasp_kbar_stress_and_expand_voigt():
    stress = normalize_stress(
        [10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
        source_unit="kbar",
        source_sign="compression_positive",
        dtype=torch.float64,
    )
    expected = (
        -0.1
        * torch.tensor(
            [[10.0, 6.0, 5.0], [6.0, 20.0, 4.0], [5.0, 4.0, 30.0]],
            dtype=torch.float64,
        )
        / GPA_PER_EV_PER_ANGSTROM_CUBED
    )
    torch.testing.assert_close(stress, expected)


def test_normalize_stress_preserves_ase_convention():
    stress = torch.tensor([[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 3.0]])
    output = normalize_stress(
        stress,
        source_unit="ev_per_angstrom_cubed",
        source_sign="tension_positive",
    )
    torch.testing.assert_close(output, stress)


@pytest.mark.parametrize(
    ("stress", "message"),
    [
        (torch.zeros(2, 2), "shape"),
        (
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "symmetric",
        ),
        (
            torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "non-finite",
        ),
    ],
)
def test_normalize_stress_rejects_malformed_input(stress, message):
    with pytest.raises(ValueError, match=message):
        normalize_stress(
            stress,
            source_unit="gpa",
            source_sign="tension_positive",
        )


def test_validate_materials_sample_accepts_consistent_data():
    sample = _sample()
    assert validate_materials_sample(sample, require_stress=True) is sample


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"forces": torch.zeros(3, 3)}, "forces"),
        ({"pos": torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]])}, "pos"),
        ({"stress": torch.zeros(6)}, "stress"),
        ({"edge_index": torch.tensor([[0, 1], [0, 0]])}, "self-loops"),
    ],
)
def test_validate_materials_sample_rejects_invalid_data(updates, message):
    with pytest.raises(ValueError, match=message):
        validate_materials_sample(_sample(**updates), require_stress=True)
