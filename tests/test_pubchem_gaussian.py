##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest
import torch

from hydragnn.models.create import compute_forces_and_hessian

from examples.pubchem_gaussian.pubchem_gaussian_hpo import (
    configure_trial,
    validation_objective,
)


def _load_example_module():
    path = Path(__file__).parents[1] / "examples" / "pubchem_gaussian" / "train.py"
    spec = importlib.util.spec_from_file_location("pubchem_gaussian_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hpo_parameters(**updates):
    parameters = {
        "mpnn_type": "SchNet",
        "use_equivariant_graph_transformer": "on",
        "energy_weight": 0.1,
        "force_weight": 1.0,
        "hessian_weight": 10.0,
        "num_conv_layers": 3,
        "hidden_dim": 128,
        "global_attn_type": "multihead",
        "global_attn_heads": 4,
        "global_attn_num_hidden_layers": 3,
        "global_attn_hidden_dim": 64,
    }
    parameters.update(updates)
    return parameters


def test_pubchem_hpo_configures_architecture_and_conditional_attention():
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "pubchem_gaussian"
        / "pubchem_gaussian.json"
    )
    config = configure_trial(
        json.loads(config_path.read_text()),
        _hpo_parameters(),
    )
    architecture = config["NeuralNetwork"]["Architecture"]

    assert architecture["mpnn_type"] == "SchNet"
    assert architecture["num_conv_layers"] == 3
    assert architecture["hidden_dim"] == 128
    assert architecture["global_attn_engine"] == "GPS"
    assert architecture["global_attn_type"] == "multihead"
    assert architecture["global_attn_heads"] == 4
    assert architecture["energy_weight"] == pytest.approx(0.1)
    assert architecture["force_weight"] == pytest.approx(1.0)
    assert architecture["hessian_weight"] == pytest.approx(10.0)
    assert architecture["global_attn_num_hidden_layers"] == 3
    assert architecture["global_attn_hidden_dim"] == 64

    performer = configure_trial(
        config,
        _hpo_parameters(global_attn_type="performer", global_attn_heads=8),
    )
    assert performer["NeuralNetwork"]["Architecture"]["global_attn_heads"] == 1

    disabled = configure_trial(
        config,
        _hpo_parameters(use_equivariant_graph_transformer="off"),
    )
    disabled_architecture = disabled["NeuralNetwork"]["Architecture"]
    assert disabled_architecture["global_attn_engine"] == ""
    assert disabled_architecture["global_attn_type"] == ""
    assert disabled_architecture["global_attn_heads"] == 1

    for model_type, equivariance in (("AllScAIP", False), ("UMA", True)):
        native = configure_trial(config, _hpo_parameters(mpnn_type=model_type))
        native_architecture = native["NeuralNetwork"]["Architecture"]
        assert native_architecture["global_attn_engine"] == ""
        assert native_architecture["equivariance"] is equivariance


def test_pubchem_hpo_objective_uses_latest_named_validation_losses(tmp_path):
    log_path = tmp_path / "trial.log"
    log_path.write_text(
        "Energy Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Forces Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Hessian Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Energy Train Loss: 1, Val Loss: 2, Test Loss: 3\n"
        "Forces Train Loss: 4, Val Loss: 5, Test Loss: 6\n"
        "Hessian Train Loss: 7, Val Loss: 8, Test Loss: 9\n"
    )

    assert validation_objective(log_path) == pytest.approx(-5.0)


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
def test_atomic_reference_archive_uses_final_scf_energy(tmp_path):
    example = _load_example_module()
    archive_path = tmp_path / "atomization.tar.gz"
    contents = (
        b" SCF Done: E(UB3LYP) = -0.400000D+00 A.U. after 4 cycles\n"
        b" SCF Done: E(UB3LYP) = -0.502257D+00 A.U. after 6 cycles\n"
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("atomization/H/elem.log")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))

    references = example.parse_atomic_reference_energies(archive_path)

    assert references == {1: pytest.approx(-0.502257)}


@pytest.mark.mpi_skip()
def test_pubchem_trajectory_selects_optimized_energy_force_and_hessian(
    tmp_path, monkeypatch
):
    example = _load_example_module()
    molecule = tmp_path / "1"
    molecule.mkdir()
    (molecule / "Structure.txt").write_text("1 0.0 0.0 0.0\n")
    (molecule / "Hessian.txt").write_text("1 2 3\n1 1.0\n2 0.1 2.0\n3 0.2 0.3 3.0\n")
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
    monkeypatch.setattr(
        example,
        "radius_graph",
        lambda positions, **kwargs: torch.empty((2, 0), dtype=torch.int64),
    )
    monkeypatch.setattr(example, "Distance", identity_transform)

    config = {"NeuralNetwork": {"Architecture": {"radius": 5.0, "max_neighbours": 8}}}
    dataset = example.PubChemGaussianDataset(
        tmp_path, config, atomic_reference_energies={1: -0.5}
    )

    assert len(dataset) == 1
    assert dataset[0].energy.shape == (1, 1)
    assert dataset[0].total_energy.item() == pytest.approx(-1.125)
    assert dataset[0].atomic_reference_energy.item() == pytest.approx(-0.5)
    assert dataset[0].formation_energy.item() == pytest.approx(-0.625)
    assert dataset[0].atomization_energy.item() == pytest.approx(0.625)
    assert torch.equal(dataset[0].energy, dataset[0].formation_energy)
    assert dataset[0].forces.shape == (1, 3)
    assert torch.isfinite(dataset[0].hessian).all()
    assert torch.allclose(dataset[0].hessian, dataset[0].hessian.T)
