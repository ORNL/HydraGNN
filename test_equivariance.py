#!/usr/bin/env python3
"""
Test script to validate E(3) equivariance of the GPSConvEquivariant layer.

This script tests that the layer satisfies the equivariance property:
f(R @ x) = R @ f(x) for any rotation matrix R

Where:
- f is our equivariant function (GPSConvEquivariant)
- R is a rotation matrix
- x are the input positions
- @ denotes matrix multiplication
"""

import torch
import numpy as np
from torch_geometric.data import Data
from hydragnn.globalAtt.gps_equivariant import GPSConvEquivariant


def create_rotation_matrix(axis="z", angle=np.pi / 4):
    """Create a rotation matrix around the specified axis."""
    if axis == "z":
        R = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif axis == "x":
        R = torch.tensor(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ],
            dtype=torch.float32,
        )
    elif axis == "y":
        R = torch.tensor(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return R


def test_equivariance():
    """Test that GPSConvEquivariant maintains E(3) equivariance."""
    torch.manual_seed(42)

    # Create test data
    num_nodes = 5
    channels = 16

    # Create random positions and scalar features
    positions = torch.randn(num_nodes, 3) * 2.0
    scalar_features = torch.randn(num_nodes, channels)

    # Create a simple GPS layer (without conv layer for simplicity)
    gps_layer = GPSConvEquivariant(
        channels=channels,
        conv=None,  # No conv layer for this test
        heads=2,
        dropout=0.0,
        attn_type="multihead",
    )
    gps_layer.eval()  # Set to eval mode to disable dropout

    # Forward pass with original positions
    with torch.no_grad():
        scalar_out_orig, vector_out_orig = gps_layer(
            inv_node_feat=scalar_features, equiv_node_feat=positions, graph_batch=None
        )

    # Test multiple rotations
    test_results = []

    for axis in ["x", "y", "z"]:
        for angle in [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
            R = create_rotation_matrix(axis, angle)

            # Rotate positions
            rotated_positions = positions @ R.T

            # Forward pass with rotated positions
            with torch.no_grad():
                scalar_out_rot, vector_out_rot = gps_layer(
                    inv_node_feat=scalar_features,
                    equiv_node_feat=rotated_positions,
                    graph_batch=None,
                )

            # Check scalar invariance: scalar features should be approximately the same
            scalar_diff = torch.norm(scalar_out_orig - scalar_out_rot)
            scalar_invariant = scalar_diff < 1e-4

            # Check vector equivariance: R @ vector_out_orig ≈ vector_out_rot
            # Apply rotation to each spatial component of the original vectors
            vector_out_orig_rotated = torch.zeros_like(vector_out_orig)
            for i in range(3):
                for j in range(3):
                    vector_out_orig_rotated[:, i, :] += (
                        R[i, j] * vector_out_orig[:, j, :]
                    )

            vector_diff = torch.norm(vector_out_orig_rotated - vector_out_rot)
            vector_equivariant = (
                vector_diff < 1e-3
            )  # Slightly more tolerance for vectors

            test_results.append(
                {
                    "axis": axis,
                    "angle": f"{angle:.3f}",
                    "scalar_invariant": scalar_invariant,
                    "scalar_diff": scalar_diff.item(),
                    "vector_equivariant": vector_equivariant,
                    "vector_diff": vector_diff.item(),
                }
            )

            print(f"Rotation {axis}-axis, {angle:.3f} rad:")
            print(f"  Scalar invariant: {scalar_invariant} (diff: {scalar_diff:.6f})")
            print(
                f"  Vector equivariant: {vector_equivariant} (diff: {vector_diff:.6f})"
            )

    # Summary
    all_scalar_invariant = all(r["scalar_invariant"] for r in test_results)
    all_vector_equivariant = all(r["vector_equivariant"] for r in test_results)

    print("\n" + "=" * 60)
    print("EQUIVARIANCE TEST SUMMARY")
    print("=" * 60)
    print(f"Scalar features invariant: {all_scalar_invariant}")
    print(f"Vector features equivariant: {all_vector_equivariant}")

    if all_scalar_invariant and all_vector_equivariant:
        print("✅ SUCCESS: The layer maintains E(3) equivariance!")
    else:
        print("❌ FAILURE: The layer does not maintain E(3) equivariance!")

        if not all_scalar_invariant:
            print("   - Scalar features are not invariant to rotations")
        if not all_vector_equivariant:
            print("   - Vector features are not equivariant to rotations")

    return all_scalar_invariant and all_vector_equivariant


if __name__ == "__main__":
    print("Testing E(3) equivariance of GPSConvEquivariant...")
    print("=" * 60)
    success = test_equivariance()

    if success:
        print("\n🎉 All tests passed! The implementation is truly equivariant.")
    else:
        print("\n⚠️  Some tests failed. The implementation needs fixes.")
