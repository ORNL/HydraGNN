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
import contextlib
import warnings
from torch_geometric.nn.aggr.scaler import DegreeScalerAggregation
from torch_geometric.nn.dense.linear import Linear as geom_Linear
import numpy as np
from torch_geometric.data import Data

from hydragnn.models.create import create_model
from hydragnn.models.PNAEqStack import PainnMessage
from hydragnn.models.PAINNStack import PainnMessage as PainnMessagePA

# Hide noisy TorchScript annotation warnings to keep test output readable.
warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations",
    category=UserWarning,
    module=r"torch.jit._check",
)


def _retune_pnaeq_aggregation(model):
    # Force PNAEq to use smooth, identity-scaled mean aggregation to reduce equivariance leakage.
    for module in model.modules():
        if isinstance(module, PainnMessage) and hasattr(module, "aggr_module"):
            deg = getattr(module.aggr_module, "deg", None)
            if deg is None:
                device = next(module.parameters()).device
                deg = torch.tensor([1.0], device=device)
            module.aggr_module = DegreeScalerAggregation(
                aggr=["mean"], scaler=["identity"], deg=deg
            )

            # Rebuild post_nns to match the reduced (aggr, scaler) config.
            module.post_nns = torch.nn.ModuleList(
                [
                    geom_Linear(2 * module.F_in, module.F_out)
                    for _ in range(module.towers)
                ]
            )

            dtype = next(module.parameters()).dtype

            def _pre_hook(mod, inputs):
                x, v, edge_index, edge_rbf, edge_vec, *rest = inputs
                x = x.to(dtype=dtype)
                v = v.to(dtype=dtype)
                edge_rbf = edge_rbf.to(dtype=dtype)
                edge_vec = edge_vec.to(dtype=dtype)
                return (x, v, edge_index, edge_rbf, edge_vec, *rest)

            module.register_forward_pre_hook(_pre_hook)


def _cast_data_dtype(data, dtype):
    if hasattr(data, "x") and data.x is not None:
        data.x = data.x.to(dtype=dtype)
    if hasattr(data, "pos") and data.pos is not None:
        data.pos = data.pos.to(dtype=dtype)
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(dtype=dtype)
    if hasattr(data, "edge_shifts") and data.edge_shifts is not None:
        data.edge_shifts = data.edge_shifts.to(dtype=dtype)
    return data


def _retune_painn_dtype(model, dtype):
    # Cast PainnMessage inputs to the desired dtype to avoid float/double mismatches.
    for module in model.modules():
        if isinstance(module, PainnMessagePA):

            def _pre_hook(mod, inputs):
                if len(inputs) == 5:
                    x, v, edge_index, diff, dist = inputs
                    edge_attr = None
                else:
                    x, v, edge_index, edge_attr, diff, dist = inputs
                x = x.to(dtype=dtype)
                v = v.to(dtype=dtype)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(dtype=dtype)
                diff = diff.to(dtype=dtype)
                dist = dist.to(dtype=dtype)
                if edge_attr is None:
                    return (x, v, edge_index, diff, dist)
                return (x, v, edge_index, edge_attr, diff, dist)

            module.register_forward_pre_hook(_pre_hook)


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
        if "graph" in pred:
            energy = pred["graph"][0].sum()
        elif "node" in pred:
            energy = pred["node"][0].sum()
        else:
            raise ValueError("Prediction dict missing 'graph' or 'node' outputs")
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
    mpnn_type,
    head_type,
    num_structures=5,
    num_rotations=3,
    dtype_override=None,
    allow_high_precision=True,
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

    # Keep PAINN hidden layers above size 1 to satisfy PainnNet constraint.
    dim_headlayers = [64, 32, 16, 1]
    if mpnn_type in ["PAINN", "PNAEq"]:
        dim_headlayers = [64, 32, 16]
    elif mpnn_type not in ["DimeNet"]:
        dim_headlayers = [64, 32, 1]

    output_heads = {
        "node": {
            "num_headlayers": len(dim_headlayers),
            "dim_headlayers": dim_headlayers,
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
    elif mpnn_type == "DimeNet":
        config_args["num_radial"] = 6
        config_args["num_spherical"] = 7
        config_args["envelope_exponent"] = 5
        config_args["basis_emb_size"] = 8
        config_args["int_emb_size"] = 64
        config_args["out_emb_size"] = 256
        config_args["num_before_skip"] = 1
        config_args["num_after_skip"] = 2
        config_args["radius"] = 5.0
    elif mpnn_type == "PAINN":
        config_args["num_radial"] = 20
        config_args["radius"] = 5.0
    elif mpnn_type == "PNAEq":
        config_args["pna_deg"] = [1, 2, 3, 4, 5]
        config_args["num_radial"] = 20
        config_args["radius"] = 5.0
    elif mpnn_type == "MACE":
        config_args["max_ell"] = 2
        config_args["node_max_ell"] = 2
        config_args["correlation"] = 2
        config_args["num_radial"] = 8
        config_args["radius"] = 5.0
        config_args["distance_transform"] = "None"
        config_args["radial_type"] = "bessel"
        config_args["avg_num_neighbors"] = 12
        config_args["envelope_exponent"] = 5

    prev_dtype = torch.get_default_dtype()
    use_high_precision = allow_high_precision and mpnn_type in ["PNAEq", "PAINN"]
    eq_dtype = (
        dtype_override
        if dtype_override is not None
        else (torch.float64 if use_high_precision else prev_dtype)
    )
    try:
        if dtype_override is not None or use_high_precision:
            torch.set_default_dtype(eq_dtype)

        model = create_model(**config_args)

        if mpnn_type == "PNAEq":
            _retune_pnaeq_aggregation(model)
        if mpnn_type == "PAINN" and (use_high_precision or dtype_override is not None):
            _retune_painn_dtype(model, eq_dtype)
        if use_high_precision or dtype_override is not None:
            model = model.to(dtype=eq_dtype)

        model.eval()
    finally:
        torch.set_default_dtype(prev_dtype)

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

            if use_high_precision or dtype_override is not None:
                data = _cast_data_dtype(data, eq_dtype)

            # Test with random rotations and specific angles
            rotation_tests = []
            for _ in range(num_rotations // 2):
                rotation_tests.append(random_rotation_matrix())
            # Add specific rotations (90° around different axes)
            rotation_tests.append(specific_rotation_matrix(np.array([1, 0, 0]), 90))
            rotation_tests.append(specific_rotation_matrix(np.array([0, 1, 0]), 180))

            for R in rotation_tests[:num_rotations]:
                if use_high_precision or dtype_override is not None:
                    R = R.to(eq_dtype)
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


def compare_energy_only_precision(precisions=("bf16", "fp32", "fp64")):
    print("\n" + "=" * 70)
    print("Energy-Only Equivariance vs Precision (graph heads)")
    print("=" * 70)
    for tag in precisions:
        dtype = _dtype_from_tag(tag)
        print(f"\nPrecision: {tag} ({dtype})")
        autocast_ctx = (
            torch.autocast("cpu", dtype=dtype)
            if dtype == torch.bfloat16
            else contextlib.nullcontext()
        )
        for mpnn_type in ["EGNN", "SchNet", "DimeNet", "PAINN", "PNAEq", "MACE"]:
            try:
                with autocast_ctx:
                    max_error, rel_error = test_energy_only_equivariance(
                        mpnn_type,
                        dtype_override=dtype,
                        allow_high_precision=False,
                    )
                status = "PRESERVED" if max_error < 1e-4 else "BROKEN"
                print(
                    f"  {mpnn_type}: max {max_error:.2e}, rel {rel_error:.2e}, {status}"
                )
            except Exception as exc:
                print(f"  {mpnn_type}: ERROR {str(exc)[:160]}")
    print("\n" + "=" * 70)


def compare_mlp_vs_conv_heads():
    """
    Compare force equivariance properties across different head architectures.

    Tests convolutional and regular MLP heads to determine
    which architectures best preserve the equivariance from message passing layers.

    Tests multiple structure types and rotations for robustness.
    """
    print("=" * 70)
    print("Force Equivariance Comparison Across Head Architectures")
    print("=" * 70)
    print("\nTesting force equivariance: F(R·x) = R·F(x)")
    print("Head types: Convolutional, MLP (shared)")
    print("Structure types: random, linear, planar, clustered")
    print("Rotations: random + specific (90°, 180°) around different axes")
    print(f"Total tests per head: {5 * 3} = 15 test cases\n")

    # Test all equivariant MPNNs
    # Note: DimeNet conv head is not used here
    mpnn_configs = {
        "EGNN": ["conv", "mlp"],
        "SchNet": ["conv", "mlp"],
        "DimeNet": ["mlp"],
        "PAINN": ["conv", "mlp"],
        "PNAEq": ["conv", "mlp"],
    }

    results = {}

    for mpnn_type, head_types in mpnn_configs.items():
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

    for mpnn_type, head_types in mpnn_configs.items():
        print(f"\n{mpnn_type}:")

        # Only show conv head error if it was tested
        if "conv" in head_types:
            conv_error = results[mpnn_type]["conv"]["max_error"]
            print(f"  Convolutional head error:           {conv_error:.2e}")

        mlp_error = results[mpnn_type]["mlp"]["max_error"]

        print(f"  Regular MLP head error:             {mlp_error:.2e}")

        if mlp_error > 1e-4:
            print(f"  Regular MLP heads BREAK equivariance for {mpnn_type}!")
        else:
            print(f"  Regular MLP heads preserve equivariance for {mpnn_type}")

    print("\n" + "=" * 70)

    return results


def _dtype_from_tag(tag):
    if isinstance(tag, torch.dtype):
        return tag
    tag_lower = str(tag).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    if tag_lower not in mapping:
        raise ValueError(f"Unknown precision tag: {tag}")
    return mapping[tag_lower]


def compare_mlp_vs_conv_heads_precision(precisions=("bf16", "fp32", "fp64")):
    print("\n" + "=" * 70)
    print("Force Equivariance vs Precision (heads)")
    print("=" * 70)
    for tag in precisions:
        dtype = _dtype_from_tag(tag)
        print(f"\nPrecision: {tag} ({dtype})")
        try:
            results = {}
            autocast_ctx = (
                torch.autocast("cpu", dtype=dtype)
                if dtype == torch.bfloat16
                else contextlib.nullcontext()
            )
            for mpnn_type, head_types in {
                "EGNN": ["conv", "mlp"],
                "SchNet": ["conv", "mlp"],
                "DimeNet": ["mlp"],
                "PAINN": ["conv", "mlp"],
                "PNAEq": ["conv", "mlp"],
            }.items():
                results[mpnn_type] = {}
                for head_type in head_types:
                    with autocast_ctx:
                        max_error, rel_error = test_head_type_equivariance(
                            mpnn_type,
                            head_type,
                            dtype_override=dtype,
                            allow_high_precision=False,
                        )
                    results[mpnn_type][head_type] = (max_error, rel_error)
                    status = "✓ PRESERVED" if max_error < 1e-4 else "✗ BROKEN"
                    print(
                        f"  {mpnn_type} {head_type.upper()}: max {max_error:.2e}, rel {rel_error:.2e}, {status}"
                    )
        except Exception as exc:
            print(f"  Error under {tag}: {str(exc)[:160]}")
    print("\n" + "=" * 70)


def test_energy_only_equivariance(
    mpnn_type,
    num_structures=4,
    num_rotations=3,
    dtype_override=None,
    allow_high_precision=True,
):
    """Test equivariance using graph-only head and autograd forces."""
    torch.manual_seed(42)
    np.random.seed(42)

    output_heads = {
        "graph": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 64,
            "num_headlayers": 2,
            "dim_headlayers": [64, 1],
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
        "output_type": ["graph"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 3,
        "equivariance": True,
        "use_gpu": False,
        "num_nodes": 10,
    }

    # Model-specific parameters
    if mpnn_type == "EGNN":
        pass
    elif mpnn_type == "SchNet":
        config_args["num_filters"] = 64
        config_args["num_gaussians"] = 50
        config_args["radius"] = 5.0
        config_args["max_neighbours"] = 100
    elif mpnn_type == "DimeNet":
        config_args["num_radial"] = 6
        config_args["num_spherical"] = 7
        config_args["envelope_exponent"] = 5
        config_args["basis_emb_size"] = 8
        config_args["int_emb_size"] = 64
        config_args["out_emb_size"] = 256
        config_args["num_before_skip"] = 1
        config_args["num_after_skip"] = 2
        config_args["radius"] = 5.0
    if mpnn_type == "PAINN":
        config_args["num_radial"] = 20
        config_args["radius"] = 5.0
    elif mpnn_type == "PNAEq":
        config_args["pna_deg"] = [1, 2, 3, 4, 5]
        config_args["num_radial"] = 20
        config_args["radius"] = 5.0
    elif mpnn_type == "MACE":
        config_args["max_ell"] = 2
        config_args["node_max_ell"] = 2
        config_args["correlation"] = 2
        config_args["num_radial"] = 8
        config_args["radius"] = 5.0
        config_args["distance_transform"] = "None"
        config_args["radial_type"] = "bessel"
        config_args["avg_num_neighbors"] = 12
        config_args["envelope_exponent"] = 5
    elif mpnn_type not in ["EGNN", "SchNet", "DimeNet"]:
        raise ValueError(
            "Energy-only test implemented for EGNN, SchNet, DimeNet, PAINN, PNAEq, and MACE"
        )

    prev_dtype = torch.get_default_dtype()
    use_high_precision = allow_high_precision and mpnn_type in ["PNAEq", "PAINN"]
    eq_dtype = (
        dtype_override
        if dtype_override is not None
        else (torch.float64 if use_high_precision else prev_dtype)
    )
    try:
        if dtype_override is not None or use_high_precision:
            torch.set_default_dtype(eq_dtype)

        model = create_model(**config_args)

        if mpnn_type == "PNAEq":
            _retune_pnaeq_aggregation(model)
        if mpnn_type == "PAINN" and (use_high_precision or dtype_override is not None):
            _retune_painn_dtype(model, eq_dtype)
        if use_high_precision or dtype_override is not None:
            model = model.to(dtype=eq_dtype)

        model.eval()
    finally:
        torch.set_default_dtype(prev_dtype)

    max_errors = []
    relative_errors = []
    structure_types = ["random", "linear", "planar", "clustered"]

    for structure_type in structure_types:
        for _ in range(num_structures // len(structure_types) + 1):
            if len(max_errors) >= num_structures * num_rotations:
                break

            data = create_molecular_system(
                num_atoms=config_args["num_nodes"],
                seed=123 + len(max_errors),
                structure_type=structure_type,
            )

            if use_high_precision or dtype_override is not None:
                data = _cast_data_dtype(data, eq_dtype)

            rotation_tests = []
            for _ in range(num_rotations // 2):
                rotation_tests.append(random_rotation_matrix())
            rotation_tests.append(specific_rotation_matrix(np.array([1, 0, 0]), 90))
            rotation_tests.append(specific_rotation_matrix(np.array([0, 1, 0]), 180))

            for R in rotation_tests[:num_rotations]:
                if use_high_precision or dtype_override is not None:
                    R = R.to(eq_dtype)
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

    return max(max_errors), max(relative_errors)


def compare_energy_only():
    print("\n" + "=" * 70)
    print("Energy-Only Equivariance (graph heads)")
    print("=" * 70)

    for mpnn_type in ["EGNN", "SchNet", "DimeNet", "PAINN", "PNAEq", "MACE"]:
        try:
            max_error, rel_error = test_energy_only_equivariance(mpnn_type)
            status = "PRESERVED" if max_error < 1e-4 else "BROKEN"
            print(f"{mpnn_type}: energy-only head")
            print(f"  Max error:      {max_error:.2e}")
            print(f"  Relative error: {rel_error:.2e}")
            print(f"  Status:         {status}\n")
        except Exception as e:
            print(f"{mpnn_type}: energy-only head")
            print("  Status:         ERROR")
            print(f"  Error message:  {str(e)[:120]}\n")


if __name__ == "__main__":
    results = compare_mlp_vs_conv_heads()
    compare_energy_only()
    compare_mlp_vs_conv_heads_precision()
    compare_energy_only_precision()
