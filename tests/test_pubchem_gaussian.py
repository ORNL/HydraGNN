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
    validation_losses,
    validation_objective,
)
from examples.pubchem_gaussian.pubchem_gaussian_multistage_hpo import (
    _command,
    normalized_score,
    rank_with_auxiliary_tiebreakers,
    sample_candidates,
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
        "global_attn_heads": 4,
        "equivariant_attn_num_hidden_layers": 3,
        "equivariant_attn_feedforward_multiplier": 4,
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
    assert architecture["global_attn_engine"] == "EquivariantTransformer"
    assert architecture["global_attn_type"] == ""
    assert architecture["global_attn_heads"] == 4
    assert architecture["energy_weight"] == pytest.approx(0.1)
    assert architecture["force_weight"] == pytest.approx(1.0)
    assert architecture["hessian_weight"] == pytest.approx(10.0)
    assert architecture["equivariant_attn_num_hidden_layers"] == 3
    assert architecture["equivariant_attn_feedforward_multiplier"] == 4
    assert architecture["equivariant_attn_allow_scalar_only"] is True
    assert architecture["equivariant_attn_require_tensor_coupling"] is False

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

    uma = configure_trial(config, _hpo_parameters(mpnn_type="UMA"))
    uma_architecture = uma["NeuralNetwork"]["Architecture"]
    assert uma_architecture["max_ell"] == 2
    assert uma_architecture["uma_mmax"] == 2

    mace = configure_trial(config, _hpo_parameters(mpnn_type="MACE"))
    mace_architecture = mace["NeuralNetwork"]["Architecture"]
    assert mace_architecture["max_ell"] == 2
    assert mace_architecture["node_max_ell"] == 2
    assert mace_architecture["equivariant_attn_lmax"] == 2


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


def test_pubchem_hpo_collects_auxiliary_losses_from_latest_epoch(tmp_path):
    log_path = tmp_path / "trial.log"
    log_path.write_text(
        "Energy Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Forces Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Hessian Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "dipole_magnitude Train Loss: 9, Val Loss: 9, Test Loss: 9\n"
        "Energy Train Loss: 1, Val Loss: 2, Test Loss: 3\n"
        "Forces Train Loss: 4, Val Loss: 5, Test Loss: 6\n"
        "Hessian Train Loss: 7, Val Loss: 8, Test Loss: 9\n"
        "dipole_magnitude Train Loss: 1, Val Loss: 0.5, Test Loss: 2\n"
    )

    assert validation_losses(log_path) == {
        "Energy": 2.0,
        "Forces": 5.0,
        "Hessian": 8.0,
        "dipole_magnitude": 0.5,
    }


def test_pubchem_multistage_candidates_are_balanced_and_reproducible():
    first = sample_candidates(16, seed=7)
    second = sample_candidates(16, seed=7)

    assert first == second
    model_counts = {}
    for candidate in first:
        model_type = candidate["parameters"]["mpnn_type"]
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
    assert set(model_counts.values()) == {2}


def test_pubchem_multistage_score_uses_fixed_metric_scales():
    losses = {"Energy": 2.0, "Forces": 10.0, "Hessian": 40.0}
    scales = {"Energy": 1.0, "Forces": 5.0, "Hessian": 20.0}

    assert normalized_score(losses, scales) == pytest.approx(2.0)


def test_pubchem_multistage_uses_auxiliaries_only_inside_primary_tolerance():
    def result(identifier, primary, auxiliary):
        return {
            "id": identifier,
            "losses": {
                "Energy": primary[0],
                "Forces": primary[1],
                "Hessian": primary[2],
                "dipole_magnitude": auxiliary,
            },
        }

    results = [
        result("primary-anchor", (1.0, 1.0, 1.0), 10.0),
        result("comparable-better-aux", (1.01, 1.01, 1.01), 1.0),
        result("outside-great-aux", (1.03, 1.0, 1.0), 0.0),
    ]
    scales = {
        "Energy": 1.0,
        "Forces": 1.0,
        "Hessian": 1.0,
        "dipole_magnitude": 1.0,
    }

    ranked = rank_with_auxiliary_tiebreakers(results, scales, 0.02)

    assert [entry["id"] for entry in ranked] == [
        "comparable-better-aux",
        "primary-anchor",
        "outside-great-aux",
    ]
    assert [entry["primary_comparable"] for entry in ranked] == [True, True, False]


def test_pubchem_multistage_launches_each_slurm_trial_with_multinode_ddp(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    stage = {
        "epochs": 3,
        "train_samples": 50_000,
        "val_samples": 10_000,
        "test_samples": 10_000,
    }

    command = _command(tmp_path / "config.json", "trial", stage, 42, 4, 8)

    assert command[:12] == [
        "srun",
        "--exclusive",
        "--exact",
        "-N",
        "4",
        "-n",
        "32",
        "--ntasks-per-node=8",
        "--gpus-per-node=8",
        "--gpus-per-task=1",
        "--gpu-bind=closest",
        "--kill-on-bad-exit=1",
    ]
    assert command[-1] == "--adios"


def test_pubchem_multistage_propagates_adios_cache_options(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    stage = {
        "epochs": 3,
        "train_samples": 50_000,
        "val_samples": 10_000,
        "test_samples": 10_000,
    }

    command = _command(
        tmp_path / "config.json",
        "trial",
        stage,
        42,
        1,
        1,
        dataset_format="adios",
        ddstore=True,
        ddstore_width=32,
    )

    assert command[-3:] == ["--adios", "--ddstore", "--ddstore-width=32"]


def test_pubchem_deterministic_subset_is_nested():
    example = _load_example_module()
    dataset = list(range(100))
    small = example.deterministic_subset(dataset, 10, seed=11)
    large = example.deterministic_subset(dataset, 30, seed=11)

    assert small.indices == large.indices[:10]


def test_pubchem_archive_limits_preserve_global_order():
    example = _load_example_module()

    limits, unavailable = example._allocate_archive_limits({2: 4, 0: 3, 1: 2}, 6)

    assert limits == {0: 3, 1: 2, 2: 1}
    assert unavailable == 0


def test_pubchem_archive_limits_report_shortfall():
    example = _load_example_module()

    limits, unavailable = example._allocate_archive_limits({0: 2, 1: 1}, 5)

    assert limits == {0: 2, 1: 1}
    assert unavailable == 2


def test_pubchem_mpi_degree_histogram_does_not_require_torch_distributed():
    example = _load_example_module()
    graph = example.Data(
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
        num_nodes=3,
    )

    histogram = example.gather_degree_mpi([graph], example.MPI.COMM_SELF)

    assert histogram.tolist() == [0, 2, 1]


def test_pubchem_dataset_limit_applies_before_rank_partition(tmp_path):
    example = _load_example_module()
    for molecule_id in (5, 1, 4, 2, 3):
        (tmp_path / str(molecule_id)).mkdir()
    (tmp_path / "temporary").mkdir()

    selected = example.select_molecule_dirs(tmp_path, rank=1, world_size=2, limit=3)

    assert [path.name for path in selected] == ["2"]


def test_pubchem_loads_external_adios_path(tmp_path, monkeypatch):
    example = _load_example_module()
    calls = []

    class FakeDataset:
        keys = example.PROCESSED_FIELDS

        def __init__(self, label):
            self.label = label

    def fake_dataset(path, label, comm, **options):
        calls.append((path, label))
        return FakeDataset(label)

    monkeypatch.setattr(example, "AdiosDataset", fake_dataset)
    external_path = tmp_path / "external.bp"

    datasets = example.load_datasets(
        {"inputs": [], "outputs": []},
        "adios",
        example.MPI.COMM_SELF,
        False,
        None,
        False,
        external_path,
    )

    assert tuple(dataset.label for dataset in datasets) == (
        "trainset",
        "valset",
        "testset",
    )
    assert {path for path, _ in calls} == {str(external_path)}


def test_pubchem_adapts_legacy_processed_dataset():
    example = _load_example_module()
    sample = example.Data(
        atomic_numbers=torch.tensor([[1.0], [6.0]]),
        pos=torch.zeros((2, 3)),
        total_charge=torch.zeros((1, 1)),
        spin_multiplicity=torch.ones((1, 1)),
        energy=torch.tensor([[-1.0]]),
        mulliken_charges=torch.zeros((2, 1)),
        dipole_magnitude=torch.zeros((1, 1)),
        quadrupole_eigenvalues=torch.zeros((1, 3)),
        polarizability_eigenvalues=torch.zeros((1, 3)),
        frontier_orbital_energies=torch.zeros((1, 3)),
        rotational_constants=torch.zeros((1, 3)),
        thermochemistry=torch.zeros((1, 8)),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "pubchem_gaussian"
        / "pubchem_gaussian.json"
    )
    variables = json.loads(config_path.read_text())["Variables"]

    dataset = example.model_ready_dataset([sample], variables)
    prepared = dataset[0]

    assert prepared.x.equal(sample.atomic_numbers)
    assert prepared.y is not None
    assert prepared.graph_attr.shape == (1, 2)


@pytest.mark.parametrize(
    ("header", "columns", "rows", "expected"),
    [
        (
            "Mulliken charges:",
            "1",
            "1 C -0.125000\n2 H 0.125000",
            [-0.125, 0.125],
        ),
        (
            "Mulliken charges and spin densities:",
            "1 2",
            "1 C -0.250000 0.750000\n2 H 0.250000 -0.750000",
            [-0.25, 0.25],
        ),
    ],
)
def test_pubchem_parses_closed_and_open_shell_mulliken_charges(
    header, columns, rows, expected
):
    example = _load_example_module()
    text = f"{header}\n {columns}\n{rows}\n Sum of Mulliken charges = 0.0\n"

    charges = example._parse_mulliken_charges(text, num_atoms=2)

    assert charges == pytest.approx(expected)


def test_pubchem_parses_concatenated_fixed_width_polarizability():
    example = _load_example_module()
    text = " Exact polarizability: 593.655 219.358 239.797-134.162 -68.483 247.212\n"

    values = example._last_labeled_floats("Exact polarizability:", text, 6)

    assert values == pytest.approx(
        [593.655, 219.358, 239.797, -134.162, -68.483, 247.212]
    )


def test_pubchem_parsers_accept_preloaded_log_text(tmp_path):
    example = _load_example_module()
    missing_path = tmp_path / "not-read.log"
    text = (
        " Input orientation:\n"
        " -----\n header\n -----\n"
        " 1 1 0 0.000000 0.000000 0.000000\n -----\n"
        " SCF Done: E(RHF) = -1.125000D+00\n"
        " Forces (Hartrees/Bohr)\n -----\n"
        " 1 1 -0.10 -0.20 -0.30\n -----\n"
    )

    records = example.parse_gaussian_log(missing_path, text=text)

    assert records[0]["energy"].item() == pytest.approx(-1.125)


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
def test_force_only_graph_is_retained_for_force_loss_backpropagation():
    positions = torch.tensor([[0.2, -0.3, 0.5]], requires_grad=True)
    stiffness = torch.nn.Parameter(torch.tensor(1.5))
    energy = stiffness * positions.pow(4).sum()

    forces, hessian = compute_forces_and_hessian(
        energy, positions, compute_hessian=False, create_graph=True
    )
    force_loss = forces.square().sum()
    force_loss.backward()

    assert hessian is None
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
