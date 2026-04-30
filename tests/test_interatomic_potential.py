##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
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
from torch_geometric.data import Data
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

    # Create positional encodings (random for testing)
    pe = torch.randn(num_atoms * num_graphs, 6)  # 6D positional encoding

    # Set positions to require gradients for force computation
    pos.requires_grad_(True)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        batch=batch,
        energy=energy,
        forces=forces,
        pe=pe,
        edge_shifts=torch.zeros(
            edge_index.size(1), 3
        ),  # No periodic boundary conditions
    )

    return data


@pytest.mark.mpi_skip()
def pytest_model_creation_with_enhancement():
    """Test creating a model with interatomic potential enhancement."""
    from hydragnn.models.create import create_model
    from hydragnn.utils.model.model import update_multibranch_heads

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
        "enable_interatomic_potential": True,
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
def pytest_forward_pass():
    """Test the enhanced forward pass with molecular data."""
    from hydragnn.models.create import create_model
    from hydragnn.utils.model.model import update_multibranch_heads

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
        "enable_interatomic_potential": True,
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
def pytest_energy_force_consistency():
    """Test that forces can be computed from energy gradients."""
    from hydragnn.models.create import create_model
    from hydragnn.utils.model.model import update_multibranch_heads

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
        "enable_interatomic_potential": True,
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
