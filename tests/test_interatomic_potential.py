##############################################################################
# Copyright (c) 2025, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

"""
Test script for interatomic potential enhancements in HydraGNN.
This test validates the enhanced forward method functionality for molecular simulations.
"""

import torch
import torch_scatter
from torch_geometric.data import Batch, Data
import pytest


def create_mock_molecular_data(num_atoms=10, num_graphs=2):
    """
    Create mock molecular data for testing interatomic potential functionality.

    Args:
        num_atoms: Number of atoms per molecule
        num_graphs: Number of molecules in the batch

    Returns:
        PyTorch Geometric Data object with molecular structure
    """
    # Create atomic positions (coordinates)
    pos = torch.randn(num_atoms * num_graphs, 3) * 5.0  # Random positions

    # Create node features (atomic numbers)
    x = torch.randint(1, 10, (num_atoms * num_graphs, 1)).float()  # Atomic numbers 1-9

    # Create edge connectivity (simple nearest neighbor graph)
    edge_index_list = []
    for graph_idx in range(num_graphs):
        start_idx = graph_idx * num_atoms
        end_idx = (graph_idx + 1) * num_atoms

        # Connect each atom to its nearest neighbors
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    # Add edge if distance is reasonable (simplified)
                    dist = torch.norm(pos[i] - pos[j])
                    if dist < 4.0:  # Cutoff distance
                        edge_index_list.extend([[i, j]])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list).t().contiguous()
    else:
        # Fallback: create minimal connectivity
        edge_index = torch.tensor([[0, 1], [1, 0]]).t().contiguous()

    # Create batch assignment
    batch = torch.repeat_interleave(torch.arange(num_graphs), num_atoms)

    # Create mock energy and forces for training
    energy = torch.randn(num_graphs, 1)  # Energy per molecule
    forces = torch.randn(num_atoms * num_graphs, 3)  # Forces per atom
    cell = torch.eye(3).repeat(num_graphs, 1, 1) * 10.0
    stress = torch.randn(num_graphs, 3, 3)
    stress = 0.5 * (stress + stress.transpose(-1, -2))

    # Create positional encodings (random for testing)
    pe = torch.randn(num_atoms * num_graphs, 6)  # 6D positional encoding

    # Set positions to require gradients for force computation
    pos.requires_grad_(True)

    data = Data(
        x=x,
        atomic_numbers=x[:, 0].long(),
        pos=pos,
        edge_index=edge_index,
        batch=batch,
        energy=energy,
        forces=forces,
        cell=cell,
        stress=stress,
        pe=pe,
        edge_shifts=torch.zeros(
            edge_index.size(1), 3
        ),  # No periodic boundary conditions
    )

    return data


def create_interatomic_model(*terms):
    from hydragnn.models.create import create_model
    from hydragnn.models.utils.model import update_multibranch_heads

    output_heads = {
        "node": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [1],
            "type": "mlp",
        }
    }
    return create_model(
        mpnn_type="EGNN",
        input_dim=1,
        hidden_dim=16,
        output_dim=[1],
        pe_dim=6,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads=update_multibranch_heads(output_heads),
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=1,
        num_nodes=3,
        domain_loss_config={
            "enabled": True,
            "provider": "interatomic_potential",
            "supervised": {"default_metric": "mse", "terms": list(terms)},
            "constraints": [],
            "constraint_optimizer": {"type": "fixed_penalty"},
        },
        use_gpu=False,
    )


@pytest.mark.mpi_skip()
def test_force_only_loss_does_not_require_energy():
    model = create_interatomic_model(
        {
            "variable": "forces",
            "weight": 1.0,
            "prediction": {
                "operator": "negative_gradient",
                "of": "energy",
                "with_respect_to": "positions",
            },
        }
    )
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    del data.energy
    loss, tasks = model.energy_force_loss(model(data), data)
    assert loss.requires_grad
    assert len(tasks) == 1


@pytest.mark.mpi_skip()
def test_energy_and_force_loss_still_works():
    model = create_interatomic_model(
        {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
        {
            "variable": "forces",
            "weight": 1.0,
            "prediction": {
                "operator": "negative_gradient",
                "of": "energy",
                "with_respect_to": "positions",
            },
        },
    )
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    loss, tasks = model.energy_force_loss(model(data), data)
    assert loss.requires_grad
    assert len(tasks) == 2


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("all_unavailable", [False, True])
def test_nan_force_labels_are_skipped(all_unavailable):
    model = create_interatomic_model(
        {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
        {
            "variable": "forces",
            "weight": 1.0,
            "prediction": {
                "operator": "negative_gradient",
                "of": "energy",
                "with_respect_to": "positions",
            },
        },
    )
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    if all_unavailable:
        data.forces.fill_(torch.nan)
    else:
        data.forces[0] = torch.nan

    loss, tasks = model.energy_force_loss(model(data), data)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert torch.isfinite(tasks[1])
    if all_unavailable:
        assert tasks[1].item() == 0.0


@pytest.mark.mpi_skip()
def test_unavailable_hessian_is_skipped():
    model = create_interatomic_model(
        {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
        {
            "variable": "forces",
            "weight": 1.0,
            "prediction": {
                "operator": "negative_gradient",
                "of": "energy",
                "with_respect_to": "positions",
            },
        },
        {
            "variable": "hessian",
            "weight": 1.0,
            "prediction": {
                "operator": "hessian",
                "of": "energy",
                "with_respect_to": "positions",
            },
        },
    )
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    data.hessian = torch.full((9, 9), torch.nan)
    data = Batch.from_data_list([data])

    loss, tasks = model.energy_force_loss(model(data), data)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert len(tasks) == 3
    assert tasks[-1].item() == 0.0


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "term,missing,message",
    [
        (
            {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
            "energy",
            "energy target",
        ),
        (
            {
                "variable": "forces",
                "weight": 1.0,
                "prediction": {
                    "operator": "negative_gradient",
                    "of": "energy",
                    "with_respect_to": "positions",
                },
            },
            "forces",
            "data.forces",
        ),
        (
            {
                "variable": "hessian",
                "weight": 1.0,
                "prediction": {
                    "operator": "hessian",
                    "of": "energy",
                    "with_respect_to": "positions",
                },
            },
            "hessian",
            "data.hessian",
        ),
    ],
)
def test_enabled_interatomic_loss_requires_its_label(term, missing, message):
    model = create_interatomic_model(term)
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    if missing in data:
        del data[missing]
    with pytest.raises(ValueError, match=message):
        model.energy_force_loss(model(data), data)


FORCES_TERM = {
    "variable": "forces",
    "weight": 1.0,
    "prediction": {
        "operator": "negative_gradient",
        "of": "energy",
        "with_respect_to": "positions",
    },
}
STRESS_TERM = {
    "variable": "stress",
    "weight": 2.0,
    "prediction": {
        "operator": "gradient_per_volume",
        "of": "energy",
        "with_respect_to": "strain",
    },
}


@pytest.mark.mpi_skip()
def test_energy_force_stress_loss():
    """Stress contributes to MLIP loss through an energy/strain derivative."""
    model = create_interatomic_model(
        {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
        FORCES_TERM,
        STRESS_TERM,
    )
    data = create_mock_molecular_data(num_atoms=3, num_graphs=2)
    prediction = model(data)
    loss, task_losses = model.energy_force_loss(prediction, data)

    assert len(task_losses) == 3
    assert all(torch.isfinite(value) for value in task_losses)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("with_integer_shifts", [True, False])
def test_stress_strains_periodic_edge_shifts(with_integer_shifts):
    """Periodic image offsets must deform with the cell under strain."""
    model = create_interatomic_model(STRESS_TERM)
    data = create_mock_molecular_data(num_atoms=4, num_graphs=2)
    num_edges = data.edge_index.size(1)
    integer_shifts = torch.randint(-1, 2, (num_edges, 3), dtype=torch.int8)
    edge_graph = data.batch[data.edge_index[0]]
    original_shifts = torch.bmm(
        integer_shifts.float().unsqueeze(1), data.cell[edge_graph]
    ).squeeze(1)
    data.edge_shifts = original_shifts
    if with_integer_shifts:
        data.integer_edge_shifts = integer_shifts

    captured = {}
    backbone_forward = model.model.forward

    def capture_forward(batch):
        captured["edge_shifts"] = batch.edge_shifts
        captured["cell"] = batch.cell
        return backbone_forward(batch)

    model.model.forward = capture_forward
    model(data)

    expected = torch.bmm(
        integer_shifts.float().unsqueeze(1), captured["cell"][edge_graph]
    ).squeeze(1)
    assert torch.allclose(captured["edge_shifts"], expected)
    (gradient,) = torch.autograd.grad(
        captured["edge_shifts"].square().sum(), model._stress_displacement
    )
    assert torch.any(gradient != 0)
    assert data.edge_shifts is original_shifts


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("with_integer_shifts", [True, False])
def test_periodic_stress_matches_finite_difference(with_integer_shifts):
    """Autograd stress of a periodic pair potential matches strain finite differences."""
    from hydragnn.models.utils.operations import get_edge_vectors_and_lengths
    from hydragnn.preprocess.graph_samples_checks_and_updates import RadiusGraphPBC

    dtype = torch.float64
    generator = torch.Generator().manual_seed(0)
    cell = torch.tensor(
        [[3.1, 0.2, 0.0], [0.3, 2.9, 0.1], [0.0, 0.4, 3.3]], dtype=dtype
    )
    num_atoms = 4
    pos = torch.rand(num_atoms, 3, generator=generator, dtype=dtype) @ cell
    data = Data(
        x=torch.ones(num_atoms, 1, dtype=dtype),
        pos=pos,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    data = RadiusGraphPBC(4.0, loop=False, max_num_neighbors=200)(data)
    assert torch.any(data.integer_edge_shifts != 0)
    integer_shifts = data.integer_edge_shifts.to(dtype)
    if not with_integer_shifts:
        del data.integer_edge_shifts
    data.pe = torch.zeros(num_atoms, 6, dtype=dtype)
    data.energy = torch.zeros(1, 1, dtype=dtype)
    data.stress = torch.zeros(3, 3, dtype=dtype)

    def node_energies(pos, edge_index, shifts):
        _, lengths = get_edge_vectors_and_lengths(pos, edge_index, shifts)
        pair = torch.exp(-lengths) * (1.0 + lengths)
        return torch_scatter.scatter_add(
            0.5 * pair, edge_index[0], dim=0, dim_size=pos.size(0)
        )

    model = create_interatomic_model(STRESS_TERM).to(dtype)
    model.model.forward = lambda batch: [
        node_energies(batch.pos, batch.edge_index, batch.edge_shifts)
    ]
    batch = Batch.from_data_list([data])
    batch.pos.requires_grad_(True)
    pred = model(batch)
    stress = model.prediction_target_pairs(pred, batch, create_graph=False)["stress"][
        0
    ].detach()[0]

    def strained_energy(strain):
        deformation = torch.eye(3, dtype=dtype) + strain
        strained_cell = cell @ deformation
        return node_energies(
            pos @ deformation, data.edge_index, integer_shifts @ strained_cell
        ).sum()

    step = 1e-6
    reference = torch.zeros(3, 3, dtype=dtype)
    for i in range(3):
        for j in range(3):
            strain = torch.zeros(3, 3, dtype=dtype)
            strain[i, j] += 0.5
            strain[j, i] += 0.5
            reference[i, j] = (
                strained_energy(step * strain) - strained_energy(-step * strain)
            ) / (2 * step)
    reference /= torch.det(cell).abs()

    assert reference.abs().max() > 1e-3
    assert torch.allclose(stress, reference, rtol=1e-6, atol=1e-8)


PERIODIC_BACKBONE_ARGS = {
    "SchNet": dict(num_gaussians=10, num_filters=16, max_neighbours=200),
    "EGNN": {},
    "DimeNet": dict(
        num_radial=6,
        num_spherical=7,
        envelope_exponent=5,
        basis_emb_size=8,
        int_emb_size=16,
        out_emb_size=16,
        num_before_skip=1,
        num_after_skip=1,
    ),
    "PAINN": dict(num_radial=8),
    "PNAEq": dict(pna_deg=[1, 2, 3, 4, 5], num_radial=8),
    "PNAPlus": dict(pna_deg=[1, 2, 3, 4, 5], num_radial=8, envelope_exponent=5),
    "MACE": dict(
        max_ell=2,
        node_max_ell=2,
        correlation=2,
        num_radial=8,
        distance_transform="None",
        radial_type="bessel",
        avg_num_neighbors=12,
        envelope_exponent=5,
    ),
    "UMA": dict(
        max_ell=2,
        num_radial=8,
        max_neighbours=200,
        periodic_boundary_conditions=True,
    ),
    "AllScAIP": dict(
        num_radial=8,
        max_neighbours=30,
        periodic_boundary_conditions=True,
        allscaip_use_freq_mask=False,
        allscaip_num_heads=2,
    ),
}


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,equivariance",
    [("SchNet", False)] + [(name, True) for name in PERIODIC_BACKBONE_ARGS],
)
def test_backbone_periodic_stress(mpnn_type, equivariance):
    """Backbone energy is supercell-extensive and its stress matches finite differences."""
    from hydragnn.models.create import create_model
    from hydragnn.models.utils.model import update_multibranch_heads
    from hydragnn.preprocess.graph_samples_checks_and_updates import RadiusGraphPBC

    dtype = torch.float64
    head_dims = [16, 16] if mpnn_type in ("PAINN", "PNAEq") else [16, 1]
    torch.manual_seed(0)
    model = create_model(
        mpnn_type=mpnn_type,
        input_dim=1,
        hidden_dim=16,
        output_dim=[1],
        pe_dim=6,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads=update_multibranch_heads(
            {
                "node": {
                    "num_sharedlayers": 1,
                    "dim_sharedlayers": 16,
                    "num_headlayers": len(head_dims),
                    "dim_headlayers": head_dims,
                    "type": "mlp",
                }
            }
        ),
        activation_function="tanh",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        num_nodes=4,
        radius=3.0,
        equivariance=equivariance,
        domain_loss_config={
            "enabled": True,
            "provider": "interatomic_potential",
            "supervised": {"default_metric": "mse", "terms": [STRESS_TERM]},
            "constraints": [],
            "constraint_optimizer": {"type": "fixed_penalty"},
        },
        use_gpu=False,
        **PERIODIC_BACKBONE_ARGS[mpnn_type],
    ).to(dtype)
    model.eval()
    backbone = model.model

    def build(cell, frac):
        num_atoms = frac.size(0)
        data = Data(
            x=torch.ones(num_atoms, 1, dtype=dtype),
            atomic_numbers=torch.full((num_atoms,), 14),
            pos=frac @ cell,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        data = RadiusGraphPBC(3.0, loop=False, max_num_neighbors=200)(data)
        data.pe = torch.zeros(num_atoms, 6, dtype=dtype)
        data.energy = torch.zeros(1, 1, dtype=dtype)
        data.stress = torch.zeros(3, 3, dtype=dtype)
        return data

    def energy(data):
        return backbone(Batch.from_data_list([data]))[0].sum()

    cell = torch.tensor(
        [[3.1, 0.2, 0.0], [0.3, 2.9, 0.1], [0.0, 0.4, 3.3]], dtype=dtype
    )
    frac = torch.rand(4, 3, generator=torch.Generator().manual_seed(1), dtype=dtype)
    data = build(cell, frac)
    assert torch.any(data.integer_edge_shifts != 0)

    half = torch.tensor([0.5, 1.0, 1.0], dtype=dtype)
    offset = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    supercell = build(
        cell * torch.tensor([[2.0], [1.0], [1.0]], dtype=dtype),
        torch.cat([frac * half, (frac + offset) * half]),
    )
    # AllScAIP's global attention is not strictly local, so allow a small deviation.
    rtol = 1e-4 if mpnn_type == "AllScAIP" else 1e-10
    assert torch.allclose(energy(supercell), 2 * energy(data), rtol=rtol)

    batch = Batch.from_data_list([data])
    batch.pos.requires_grad_(True)
    stress = model.prediction_target_pairs(model(batch), batch, create_graph=False)[
        "stress"
    ][0].detach()[0]

    def strained_energy(strain):
        deformation = torch.eye(3, dtype=dtype) + strain
        strained = data.clone()
        strained.pos = data.pos @ deformation
        strained.cell = cell @ deformation
        strained.edge_shifts = data.integer_edge_shifts.to(dtype) @ strained.cell
        return energy(strained)

    step = 1e-6
    reference = torch.zeros(3, 3, dtype=dtype)
    for i in range(3):
        for j in range(3):
            strain = torch.zeros(3, 3, dtype=dtype)
            strain[i, j] += 0.5
            strain[j, i] += 0.5
            reference[i, j] = (
                strained_energy(step * strain) - strained_energy(-step * strain)
            ) / (2 * step)
    reference /= torch.det(cell).abs()

    assert reference.abs().max() > 1e-5
    assert torch.allclose(stress, reference, rtol=1e-6, atol=1e-10)


@pytest.mark.mpi_skip()
def test_stress_loss_requires_cell():
    """A clear error is raised instead of silently omitting stress."""
    model = create_interatomic_model(STRESS_TERM)
    data = create_mock_molecular_data(num_atoms=3, num_graphs=1)
    del data.cell
    with pytest.raises(ValueError, match="requires data.cell"):
        model(data)

    data = create_mock_molecular_data(num_atoms=5, num_graphs=2)
    data.cell = torch.eye(3).repeat(3, 1, 1)
    with pytest.raises(ValueError, match="Expected one cell or 2 cells"):
        model(data)

    data = create_mock_molecular_data(num_atoms=5, num_graphs=2)
    data.cell = torch.zeros(6, 2)
    with pytest.raises(ValueError, match="Unexpected cell shape"):
        model(data)


@pytest.mark.mpi_skip()
def test_model_creation_with_enhancement():
    """Test creating a model with interatomic potential enhancement."""
    from hydragnn.models.create import create_model
    from hydragnn.models.utils.model import update_multibranch_heads

    # Basic model configuration
    output_heads = {
        "node": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [1],
            "type": "mlp",
        }
    }

    # The model instantiated here must use data.pos in the message passing to make sure that all the intermediate variable used to compute the energy depend on data.pos
    config_args = {
        "mpnn_type": "EGNN",
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": [1],
        "pe_dim": 6,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "num_nodes": 10,
        "domain_loss_config": {
            "enabled": True,
            "provider": "interatomic_potential",
            "supervised": {
                "default_metric": "mse",
                "terms": [
                    {
                        "variable": "energy",
                        "weight": 1.0,
                        "normalization": "per_structure",
                    },
                    {
                        "variable": "forces",
                        "weight": 1.0,
                        "prediction": {
                            "operator": "negative_gradient",
                            "of": "energy",
                            "with_respect_to": "positions",
                        },
                    },
                ],
            },
            "constraints": [],
            "constraint_optimizer": {"type": "fixed_penalty"},
        },
        "use_gpu": False,
    }

    try:
        model = create_model(**config_args)
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.mpi_skip()
def test_forward_pass():
    """Test the enhanced forward pass with molecular data."""
    from hydragnn.models.create import create_model
    from hydragnn.models.utils.model import update_multibranch_heads

    # Create model with interatomic potential enhancement
    output_heads = {
        "node": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [1],
            "type": "mlp",
        }
    }

    # The model instantiated here must use data.pos in the message passing to make sure that all the intermediate variable used to compute the energy depend on data.pos
    config_args = {
        "mpnn_type": "EGNN",
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": [1],
        "pe_dim": 6,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "num_nodes": 10,
        "domain_loss_config": {
            "enabled": True,
            "provider": "interatomic_potential",
            "supervised": {
                "default_metric": "mse",
                "terms": [
                    {
                        "variable": "energy",
                        "weight": 1.0,
                        "normalization": "per_structure",
                    },
                    {
                        "variable": "forces",
                        "weight": 1.0,
                        "prediction": {
                            "operator": "negative_gradient",
                            "of": "energy",
                            "with_respect_to": "positions",
                        },
                    },
                ],
            },
            "constraints": [],
            "constraint_optimizer": {"type": "fixed_penalty"},
        },
        "use_gpu": False,
    }

    model = create_model(**config_args)
    model.eval()

    # Create mock molecular data
    data = create_mock_molecular_data(num_atoms=5, num_graphs=2)

    # Test forward pass
    with torch.no_grad():
        output = model(data)

    # Basic assertions
    assert output is not None
    if isinstance(output, (list, tuple)):
        assert len(output) > 0
        assert output[0].shape[0] > 0  # Should have some outputs


@pytest.mark.mpi_skip()
def test_energy_force_consistency():
    """Test that forces can be computed from energy gradients."""
    from hydragnn.models.create import create_model
    from hydragnn.models.utils.model import update_multibranch_heads

    # Create model for energy prediction
    output_heads = {
        "node": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [1],
            "type": "mlp",
        }
    }

    # The model instantiated here must use data.pos in the message passing to make sure that all the intermediate variable used to compute the energy depend on data.pos
    config_args = {
        "mpnn_type": "EGNN",
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": [1],
        "pe_dim": 6,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "num_nodes": 10,
        "domain_loss_config": {
            "enabled": True,
            "provider": "interatomic_potential",
            "supervised": {
                "default_metric": "mse",
                "terms": [
                    {
                        "variable": "energy",
                        "weight": 1.0,
                        "normalization": "per_structure",
                    },
                    {
                        "variable": "forces",
                        "weight": 1.0,
                        "prediction": {
                            "operator": "negative_gradient",
                            "of": "energy",
                            "with_respect_to": "positions",
                        },
                    },
                ],
            },
            "constraints": [],
            "constraint_optimizer": {"type": "fixed_penalty"},
        },
        "use_gpu": False,
    }

    # Debug the output_heads structure
    print(f"Original output_heads: {output_heads}")
    updated_heads = update_multibranch_heads(output_heads)
    print(f"Updated output_heads: {updated_heads}")
    config_args["output_heads"] = updated_heads

    model = create_model(**config_args)
    model.train()  # Need gradients

    # Create mock molecular data
    data = create_mock_molecular_data(num_atoms=5, num_graphs=1)

    # Ensure positions require gradients
    data.pos.requires_grad_(True)

    # Forward pass to get energy
    energy_output = model(data)

    if isinstance(energy_output, (list, tuple)):
        energy = (
            torch_scatter.scatter_add(energy_output[0], data.batch, dim=0)
            .squeeze()
            .float()
        )
    else:
        energy = (
            torch_scatter.scatter_add(energy_output, data.batch, dim=0)
            .squeeze()
            .float()
        )

    # Compute forces as negative gradients of energy w.r.t. positions
    forces = torch.autograd.grad(
        energy, data.pos, create_graph=True, retain_graph=True
    )[0]
    forces = -forces  # Force = -gradient of energy

    # Basic assertions
    assert energy is not None
    assert forces is not None
    assert forces.shape == data.pos.shape
    assert not torch.isnan(energy).any()
    assert not torch.isnan(forces).any()
