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
Equivariance / invariance tests for the AMP (anisotropic message passing) stack.

The AMP potential is invariant-by-construction: multipoles are built
equivariantly from bond bases scaled by learned scalars, and the anisotropic
features g_0..g_8 are full Cartesian contractions (rotation invariant). The
per-atom energy therefore must be invariant under SO(3) and translation, and the
forces F = -dE/dpos must be equivariant: F(R x + t) = R F(x).

We test on random / linear / planar / clustered geometries under random and
axis-aligned rotations, in float64 for a tight tolerance.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from hydragnn.models.create import create_model, create_model_config
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model.amp import (
    BesselKernel,
    aniso_features,
    build_Rx2,
    build_poles,
)
from hydragnn.utils.model.model import update_multibranch_heads


def random_rotation_matrix():
    A = torch.randn(3, 3, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def axis_rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return torch.tensor(R, dtype=torch.float64)


def make_system(num_atoms=10, seed=0, structure_type="random", cutoff=5.0):
    torch.manual_seed(seed)
    if structure_type == "linear":
        pos = torch.zeros(num_atoms, 3, dtype=torch.float64)
        pos[:, 0] = torch.linspace(0, num_atoms - 1, num_atoms, dtype=torch.float64)
        pos += torch.randn(num_atoms, 3, dtype=torch.float64) * 0.1
    elif structure_type == "planar":
        pos = torch.randn(num_atoms, 2, dtype=torch.float64) * 2.0
        pos = torch.cat([pos, torch.zeros(num_atoms, 1, dtype=torch.float64)], dim=1)
        pos += torch.randn(num_atoms, 3, dtype=torch.float64) * 0.1
    elif structure_type == "clustered":
        half = num_atoms // 2
        pos = torch.zeros(num_atoms, 3, dtype=torch.float64)
        pos[:half] = torch.randn(half, 3, dtype=torch.float64) * 0.5
        pos[half:] = torch.randn(
            num_atoms - half, 3, dtype=torch.float64
        ) * 0.5 + torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
    else:
        pos = torch.randn(num_atoms, 3, dtype=torch.float64) * 2.0

    x = torch.randint(1, 10, (num_atoms, 1)).double()

    dist = torch.cdist(pos, pos)
    edge_index = (dist < cutoff).nonzero(as_tuple=False).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_shifts = torch.zeros(edge_index.size(1), 3, dtype=torch.float64)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        batch=torch.zeros(num_atoms, dtype=torch.long),
    )


def build_amp_model(num_atoms=10):
    output_heads = {
        "node": {
            "num_headlayers": 2,
            "dim_headlayers": [64, 64],
            "type": "mlp",
        }
    }
    config_args = {
        "mpnn_type": "AMP",
        "input_dim": 1,
        "hidden_dim": 64,
        "output_dim": [1],
        "pe_dim": 0,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "silu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": True,
        "use_gpu": False,
        "num_nodes": num_atoms,
        # AMP / Table-4-style hyperparameters
        "num_radial": 8,
        "radius": 6.0,
        "edge_dim": 32,
        "amp_edge_size": 32,
        "amp_num_channels": 8,
        "amp_envelope_p": 6.0,
        "amp_max_z": 20,
    }
    model = create_model(**config_args).to(torch.float64)
    model.eval()
    return model


def energy_and_forces(model, data):
    data = data.clone()
    data.pos.requires_grad_(True)
    pred = model(data)
    node_energy = pred[0] if isinstance(pred, (list, tuple)) else pred
    energy = node_energy.sum()
    forces = -torch.autograd.grad(energy, data.pos, create_graph=False)[0]
    return energy.detach(), forces.detach()


def pytest_amp_layer_dimensions_match_ampv3():
    model = build_amp_model()
    first, second = model.graph_convs
    aniso_size = 9 * model.num_channels

    assert first.eq_message_layer[0].in_features == 2 * model.hidden_dim + 32
    assert (
        second.eq_message_layer[0].in_features == 2 * model.hidden_dim + 32 + aniso_size
    )
    assert first.eq_message_layer[-1].out_features == 2 * model.num_channels
    assert (
        first.in_message_layer[0].in_features == 2 * model.hidden_dim + 32 + aniso_size
    )


def pytest_amp_config_construction(monkeypatch):
    monkeypatch.setenv("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", "0")
    sample = Data(
        x=torch.tensor([[1.0], [6.0], [8.0]]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
        ),
        batch=torch.zeros(3, dtype=torch.long),
    )

    class Loader:
        dataset = [sample]

    config = {
        "NeuralNetwork": {
            "Architecture": {
                "mpnn_type": "AMP",
                "radius": 4.0,
                "max_neighbours": 32,
                "num_radial": 8,
                "hidden_dim": 128,
                "num_conv_layers": 2,
                "activation_function": "silu",
                "amp_edge_size": 32,
                "amp_num_channels": 8,
                "amp_envelope_p": 6.0,
                "amp_max_z": 54,
                "amp_trainable_bessel": False,
                "output_heads": {
                    "graph": {
                        "num_sharedlayers": 1,
                        "dim_sharedlayers": 32,
                        "num_headlayers": 1,
                        "dim_headlayers": [32],
                    }
                },
                "task_weights": [1.0],
            },
            "Variables_of_interest": {
                "input_node_features": [0],
                "output_index": [0],
                "output_dim": [1],
                "output_names": ["energy"],
                "type": ["graph"],
            },
            "Training": {"Optimizer": {"type": "AdamW"}},
        }
    }
    loader = Loader()
    config = update_config(config, loader, loader, loader)

    model = create_model_config(config["NeuralNetwork"], use_gpu=False)
    assert model(sample)[0].shape == (1, 1)


def pytest_amp_forward_matches_reference_residual_loop():
    """HydraGNN must not add an activation after AMP's equation-7 update."""
    torch.manual_seed(11)
    model = build_amp_model(num_atoms=8)
    data = make_system(num_atoms=8, seed=12, cutoff=4.0)

    nodes, aniso, conv_args = model._embedding(data)
    for conv in model.graph_convs:
        nodes, aniso = conv(
            inv_node_feat=nodes,
            equiv_node_feat=aniso,
            **conv_args,
        )
    expected = model.heads_NN[0]["branch-0"](x=nodes, batch=data.batch)
    actual = model(data)[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def pytest_amp_enforces_cutoff_and_prefers_atomic_numbers():
    model = build_amp_model(num_atoms=3)
    model.radius = 2.0
    model.radial_embedding.cutoff = 2.0
    data = Data(
        # Deliberately not valid atomic numbers: atomic_numbers must take priority.
        x=torch.tensor([[0.2], [0.4], [0.6]], dtype=torch.float64),
        atomic_numbers=torch.tensor([[1], [6], [8]]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
        batch=torch.zeros(3, dtype=torch.long),
    )

    nodes, _, conv_args = model._embedding(data)
    assert nodes.shape == (3, model.hidden_dim)
    assert torch.equal(conv_args["edge_index"], torch.tensor([[0, 1], [1, 0]]))


def pytest_amp_cartesian_features_are_rotation_invariant():
    torch.manual_seed(19)
    dtype = torch.float64
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    senders, receivers = edge_index
    rx1 = torch.randn(edge_index.size(1), 3, dtype=dtype)
    rx1 = rx1 / torch.linalg.vector_norm(rx1, dim=-1, keepdim=True)
    coefficients = torch.randn(edge_index.size(1), 6, dtype=dtype)
    envelope = torch.rand(edge_index.size(1), 1, dtype=dtype)

    dipoles, quadrupoles = build_poles(
        coefficients, envelope, rx1, build_Rx2(rx1), receivers, 3
    )
    features = aniso_features(
        dipoles, quadrupoles, rx1, build_Rx2(rx1), senders, receivers
    )

    rotation = random_rotation_matrix()
    rx1_rotated = rx1 @ rotation.t()
    dipoles_rotated, quadrupoles_rotated = build_poles(
        coefficients,
        envelope,
        rx1_rotated,
        build_Rx2(rx1_rotated),
        receivers,
        3,
    )
    features_rotated = aniso_features(
        dipoles_rotated,
        quadrupoles_rotated,
        rx1_rotated,
        build_Rx2(rx1_rotated),
        senders,
        receivers,
    )
    torch.testing.assert_close(features_rotated, features, rtol=1e-12, atol=1e-12)


def pytest_amp_bessel_envelope_vanishes_at_cutoff():
    kernel = BesselKernel(cutoff=4.0, n_bessel=8, p=6.0).to(torch.float64)
    radial, envelope = kernel(torch.tensor([[4.0]], dtype=torch.float64))
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=1e-14, rtol=0)
    torch.testing.assert_close(envelope, torch.zeros_like(envelope), atol=1e-14, rtol=0)


def pytest_amp_convolution_checkpointing_preserves_forces():
    model = build_amp_model(num_atoms=6)
    data = make_system(num_atoms=6, seed=41, cutoff=4.0)
    energy, forces = energy_and_forces(model, data)

    model.enable_conv_checkpointing()
    checkpointed_energy, checkpointed_forces = energy_and_forces(model, data)
    torch.testing.assert_close(checkpointed_energy, energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(checkpointed_forces, forces, rtol=0.0, atol=0.0)


STRUCTURES = ["random", "linear", "planar", "clustered"]


@pytest.mark.parametrize("structure_type", STRUCTURES)
def pytest_amp_rotation_translation_invariance_equivariance(structure_type):
    torch.manual_seed(42)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_amp_model(num_atoms=10)
        data = make_system(num_atoms=10, seed=7, structure_type=structure_type)

        e0, f0 = energy_and_forces(model, data)

        rotations = [
            random_rotation_matrix(),
            axis_rotation_matrix([1, 0, 0], 90),
            axis_rotation_matrix([0, 1, 0], 180),
            axis_rotation_matrix([1, 1, 1], 120),
        ]
        translation = torch.tensor([3.1, -2.4, 0.7], dtype=torch.float64)

        for R in rotations:
            data_t = data.clone()
            # x' = x R^T + t  (rotate then translate)
            data_t.pos = data.pos @ R.t() + translation

            e1, f1 = energy_and_forces(model, data_t)

            # Energy invariance under rotation + translation.
            energy_err = (e1 - e0).abs().item()
            assert (
                energy_err < 1e-8
            ), f"[{structure_type}] energy not invariant: |dE|={energy_err:.2e}"

            # Force equivariance: F(R x + t) = R F(x)  =>  f1 == f0 @ R^T.
            f0_rot = f0 @ R.t()
            force_err = (f1 - f0_rot).norm(dim=1).max().item()
            denom = f1.norm(dim=1).max().item() + 1e-12
            assert force_err / denom < 1e-8, (
                f"[{structure_type}] forces not equivariant: "
                f"max|df|={force_err:.2e} (rel {force_err / denom:.2e})"
            )
    finally:
        torch.set_default_dtype(prev_dtype)


def pytest_amp_pure_translation_invariance():
    """Energy is translation invariant and net force is (near) zero."""
    torch.manual_seed(1)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_amp_model(num_atoms=12)
        data = make_system(num_atoms=12, seed=3, structure_type="random")
        e0, f0 = energy_and_forces(model, data)

        data_t = data.clone()
        data_t.pos = data.pos + torch.tensor([10.0, -5.0, 2.0], dtype=torch.float64)
        e1, f1 = energy_and_forces(model, data_t)

        assert (e1 - e0).abs().item() < 1e-8, "energy not translation invariant"
        assert (f1 - f0).norm().item() < 1e-7, "forces changed under translation"
        # Newton's third law: internal forces sum to (approximately) zero.
        assert f0.sum(dim=0).norm().item() < 1e-6, "net force is nonzero"
    finally:
        torch.set_default_dtype(prev_dtype)


if __name__ == "__main__":
    for s in STRUCTURES:
        pytest_amp_rotation_translation_invariance_equivariance(s)
        print(f"  [{s}] rotation+translation OK")
    pytest_amp_pure_translation_invariance()
    print("  translation invariance OK")
    print("All AMP equivariance tests passed.")
