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
def test_pubchem_trajectory_assigns_hessian_only_to_optimized_geometry(
    tmp_path, monkeypatch
):
    example = _load_example_module()
    molecule = tmp_path / "1"
    molecule.mkdir()
    (molecule / "Structure.txt").write_text("1 0.0 0.0 0.0\n")
    (molecule / "Hessian.txt").write_text(
        "1 2 3\n1 1.0\n2 0.1 2.0\n3 0.2 0.3 3.0\n"
    )
    (molecule / "1.log").write_text(
        " Input orientation:\n"
        " -----\n header\n -----\n"
        " 1 1 0 0.100000 0.000000 0.000000\n -----\n"
        " SCF Done: E(RHF) = -1.000000D+00\n"
        " Forces (Hartrees/Bohr)\n -----\n"
        " 1 1 0.10 0.20 0.30\n -----\n"
        " Input orientation:\n"
        " -----\n header\n -----\n"
        " 1 1 0 0.000000 0.000000 0.000000\n -----\n"
        " SCF Done: E(RHF) = -1.125000D+00\n"
        " Forces (Hartrees/Bohr)\n -----\n"
        " 1 1 -0.10 -0.20 -0.30\n -----\n"
    )
    identity_transform = lambda *args, **kwargs: lambda data: data
    monkeypatch.setattr(example, "RadiusGraph", identity_transform)
    monkeypatch.setattr(example, "Distance", identity_transform)

    config = {
        "NeuralNetwork": {
            "Architecture": {"radius": 5.0, "max_neighbours": 8}
        }
    }
    dataset = example.PubChemGaussianDataset(tmp_path, config)

    assert len(dataset) == 2
    assert dataset[0].energy.shape == (1, 1)
    assert dataset[0].forces.shape == (1, 3)
    assert not dataset[0].hessian_available.item()
    assert torch.isnan(dataset[0].hessian).all()
    assert dataset[1].energy.item() == pytest.approx(-1.125)
    assert dataset[1].hessian_available.item()
    assert torch.isfinite(dataset[1].hessian).all()
    assert torch.allclose(dataset[1].hessian, dataset[1].hessian.T)
