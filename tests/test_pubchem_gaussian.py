##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import importlib.util
from pathlib import Path

import pytest
import torch

from hydragnn.models.create import compute_forces_and_hessian


def _load_example_module():
    path = Path(__file__).parents[1] / "examples" / "pubchem_gaussian" / "train.py"
    spec = importlib.util.spec_from_file_location("pubchem_gaussian_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.mpi_skip()
def test_force_and_hessian_autograd_identities_and_backpropagation():
    positions = torch.tensor([[0.2, -0.3, 0.5], [0.7, 0.1, -0.4]], requires_grad=True)
    stiffness = torch.nn.Parameter(torch.tensor(1.5))
    energy = 0.5 * stiffness * positions.square().sum()

    forces, hessian = compute_forces_and_hessian(
        energy, positions, compute_hessian=True, create_graph=True
    )

    dimension = positions.numel()
    expected_hessian = stiffness * torch.eye(dimension)
    assert torch.allclose(forces, -stiffness * positions)
    assert hessian.shape == (dimension, dimension)
    assert torch.allclose(hessian, expected_hessian)
    assert torch.allclose(hessian, hessian.T)

    loss = torch.nn.functional.mse_loss(hessian, 2.0 * torch.eye(dimension))
    loss.backward()
    assert stiffness.grad is not None
    assert stiffness.grad.abs() > 0


@pytest.mark.mpi_skip()
def test_pubchem_parsers_return_matching_energy_force_and_hessian(tmp_path):
    example = _load_example_module()
    structure = tmp_path / "Structure.txt"
    trajectory = tmp_path / "Opt_Trj.txt"
    hessian_file = tmp_path / "Hessian.txt"
    structure.write_text("1 0.0 0.0 0.0\n1 0.0 0.0 0.74\n")
    trajectory.write_text(
        "Optimization step 3 Energy = -1.125000D+00\n"
        "1 0.0 0.0 0.0  0.10 0.20 0.30\n"
        "1 0.0 0.0 0.74 -0.10 -0.20 -0.30\n"
    )
    hessian_file.write_text(
        "1 2 3 4 5\n"
        "1 1.0\n2 0.0 2.0\n3 0.0 0.0 3.0\n"
        "4 0.0 0.0 0.0 4.0\n5 0.0 0.0 0.0 0.0 5.0\n"
        "6 0.0 0.0 0.0 0.0 0.0\n"
        "6\n6 6.0\n"
    )

    atomic_numbers, positions = example.parse_structure(structure)
    energy, forces = example.parse_optimization_forces(
        trajectory, atomic_numbers, positions
    )
    hessian = example.parse_hessian(hessian_file, num_atoms=2)

    assert energy.shape == (1, 1)
    assert energy.item() == pytest.approx(-1.125)
    assert forces.shape == (2, 3)
    assert hessian.shape == (6, 6)
    assert torch.allclose(hessian, hessian.T)
