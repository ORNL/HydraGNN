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
Test force equivariance properties of different output head architectures.

This test verifies that forces (computed as gradients of energy predictions)
are equivariant under rotations for different head types: convolutional heads,
regular MLP heads, and rotation-invariant MLP heads.

Force equivariance: F(R·x) = R·F(x) for any rotation matrix R
where forces F = -∇E (negative gradient of energy w.r.t. positions)

Tests multiple MPNN architectures (EGNN, SchNet) with various head types across
diverse molecular structures (random, linear, planar, clustered) and rotation
angles to ensure robust equivariance preservation.
"""

import torch
import numpy as np
from torch_geometric.data import Data

from hydragnn.models.create import create_model
from hydragnn.utils.model.model import update_multibranch_heads


def random_rotation_matrix():
    """Generate a random 3D rotation matrix using QR decomposition."""
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def specific_rotation_matrix(axis, angle_deg):
    """Generate rotation matrix around specific axis by specific angle."""
    angle = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return torch.tensor(R, dtype=torch.float32)


def create_molecular_system(num_atoms=10, seed=None, structure_type="random"):
    """
    Create a molecular system for testing.

    Args:
        num_atoms: Number of atoms
        seed: Random seed
        structure_type: "random", "linear", "planar", or "clustered"
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate positions based on structure type
    if structure_type == "linear":
        # Atoms along a line (tests edge case)
        pos = torch.zeros(num_atoms, 3)
        pos[:, 0] = torch.linspace(0, num_atoms - 1, num_atoms)
        pos += torch.randn(num_atoms, 3) * 0.1  # Small perturbations
    elif structure_type == "planar":
        # Atoms in a plane
        pos = torch.randn(num_atoms, 2) * 2.0
        pos = torch.cat([pos, torch.zeros(num_atoms, 1)], dim=1)
        pos += torch.randn(num_atoms, 3) * 0.1  # Small out-of-plane noise
    elif structure_type == "clustered":
        # Two clusters of atoms
        half = num_atoms // 2
        pos = torch.zeros(num_atoms, 3)
        pos[:half] = torch.randn(half, 3) * 0.5  # First cluster
        pos[half:] = torch.randn(num_atoms - half, 3) * 0.5 + torch.tensor(
            [5.0, 0.0, 0.0]
        )  # Second cluster
    else:  # "random"
        pos = torch.randn(num_atoms, 3) * 2.0

    x = torch.randint(1, 10, (num_atoms, 1)).float()

    cutoff = 5.0
    dist_matrix = torch.cdist(pos, pos)
    edge_index = (dist_matrix < cutoff).nonzero(as_tuple=False).t()

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    edge_shifts = torch.zeros(edge_index.size(1), 3)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        batch=torch.zeros(num_atoms, dtype=torch.long),
    )

    return data


def rotate_data(data, rotation_matrix):
    """Apply rotation to positions in a Data object."""
    data_rotated = data.clone()
    data_rotated.pos = torch.matmul(data.pos, rotation_matrix.t())
    return data_rotated


def compute_forces(model, data):
    """Compute forces via automatic differentiation of energy."""
    data.pos.requires_grad_(True)

    pred = model(data)

    if isinstance(pred, dict):
        energy = pred["node"][0].sum()
    elif isinstance(pred, (list, tuple)):
        energy = pred[0].sum()
    else:
        energy = pred.sum()

    forces = -torch.autograd.grad(
        energy,
        data.pos,
        create_graph=False,
        retain_graph=False,
    )[0]

    return forces


def test_head_type_equivariance(
    mpnn_type, head_type, num_structures=5, num_rotations=3
):
    """
    Test equivariance with a specific head type.

    Args:
        mpnn_type: Type of message passing layer ("EGNN", "SchNet", etc.)
        head_type: Type of output head ("conv" or "mlp")
        num_structures: Number of different molecular structures to test
        num_rotations: Number of random rotations per structure

    Returns:
        max_error: Maximum equivariance error across all tests
        relative_error: Maximum relative equivariance error
    """
    torch.manual_seed(42)
    np.random.seed(42)

    output_heads = {
        "node": {
            "num_headlayers": 3,
            "dim_headlayers": [64, 32, 1],
            "type": head_type,  # "conv" or "mlp"
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 1,
        "hidden_dim": 64,
        "output_dim": [1],
        "pe_dim": 0,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 3,
        "equivariance": True,
        "use_gpu": False,
        "num_nodes": 10,  # Required for MLP heads
    }

    # Add model-specific parameters
    if mpnn_type == "SchNet":
        config_args["num_filters"] = 64
        config_args["num_gaussians"] = 50
        config_args["radius"] = 5.0
        config_args["max_neighbours"] = 100
    elif mpnn_type == "EGNN":
        pass  # EGNN doesn't need extra parameters

    model = create_model(**config_args)
    model.eval()

    max_errors = []
    relative_errors = []

    structure_types = ["random", "linear", "planar", "clustered"]

    # Test multiple structure types and rotations for robustness
    for structure_type in structure_types:
        for istructure in range(num_structures // len(structure_types) + 1):
            if len(max_errors) >= num_structures * num_rotations:
                break

            # Create test system with different structure types
            seed_val = 123 + len(max_errors)
            data = create_molecular_system(
                num_atoms=config_args["num_nodes"],
                seed=seed_val,
                structure_type=structure_type,
            )

            # Test with random rotations and specific angles
            rotation_tests = []
            for _ in range(num_rotations // 2):
                rotation_tests.append(random_rotation_matrix())
            # Add specific rotations (90° around different axes)
            rotation_tests.append(specific_rotation_matrix(np.array([1, 0, 0]), 90))
            rotation_tests.append(specific_rotation_matrix(np.array([0, 1, 0]), 180))

            for R in rotation_tests[:num_rotations]:
                forces_original = compute_forces(model, data.clone())
                data_rotated = rotate_data(data, R)
                forces_rotated = compute_forces(model, data_rotated)
                forces_original_rotated = torch.matmul(forces_original, R.t())

                max_error = torch.max(
                    torch.abs(forces_rotated - forces_original_rotated)
                ).item()
                relative_error = max_error / (
                    torch.max(torch.abs(forces_rotated)).item() + 1e-10
                )

                max_errors.append(max_error)
                relative_errors.append(relative_error)

    # Return worst-case errors
    return max(max_errors), max(relative_errors)


def compare_mlp_vs_conv_heads():
    """
    Compare force equivariance properties across different head architectures.

    Tests convolutional, regular MLP, and rotation-invariant MLP heads to determine
    which architectures best preserve the equivariance from message passing layers.

    Tests multiple structure types and rotations for robustness.
    """
    print("=" * 70)
    print("Force Equivariance Comparison Across Head Architectures")
    print("=" * 70)
    print("\nTesting force equivariance: F(R·x) = R·F(x)")
    print("Head types: Convolutional, MLP per-node, Rotation-invariant MLP")
    print("Structure types: random, linear, planar, clustered")
    print("Rotations: random + specific (90°, 180°) around different axes")
    print(f"Total tests per head: {5 * 3} = 15 test cases\n")

    mpnn_types = ["EGNN", "SchNet"]
    head_types = ["conv", "mlp_per_node", "rotation_invariant_mlp_per_node"]

    results = {}

    for mpnn_type in mpnn_types:
        results[mpnn_type] = {}
        print(f"\n{mpnn_type} Architecture:")
        print("-" * 50)

        for head_type in head_types:
            max_error, relative_error = test_head_type_equivariance(
                mpnn_type, head_type
            )
            results[mpnn_type][head_type] = {
                "max_error": max_error,
                "relative_error": relative_error,
            }

            status = "✓ PRESERVED" if max_error < 1e-4 else "✗ BROKEN"

            print(f"  {head_type.upper()} head:")
            print(f"    Max error:      {max_error:.2e}")
            print(f"    Relative error: {relative_error:.2e}")
            print(f"    Status:         {status}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    for mpnn_type in mpnn_types:
        conv_error = results[mpnn_type]["conv"]["max_error"]
        mlp_error = results[mpnn_type]["mlp_per_node"]["max_error"]
        rot_inv_mlp_error = results[mpnn_type]["rotation_invariant_mlp_per_node"][
            "max_error"
        ]

        print(f"\n{mpnn_type}:")
        print(f"  Convolutional head error:           {conv_error:.2e}")
        print(f"  Regular MLP head error:             {mlp_error:.2e}")
        print(f"  Rotation-invariant MLP head error:  {rot_inv_mlp_error:.2e}")

        if mlp_error > 1e-4:
            print(f"  ⚠️  Regular MLP heads BREAK equivariance for {mpnn_type}!")
        else:
            print(f"  ✓  Regular MLP heads preserve equivariance for {mpnn_type}")

        if rot_inv_mlp_error > 1e-4:
            print(
                f"  ⚠️  Rotation-invariant MLP heads BREAK equivariance for {mpnn_type}!"
            )
        else:
            print(
                f"  ✓  Rotation-invariant MLP heads preserve equivariance for {mpnn_type}"
            )

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = compare_mlp_vs_conv_heads()
