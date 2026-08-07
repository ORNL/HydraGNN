##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

"""Unit tests for the UMA rotation-equivariant per-node vector head.

These exercise ``_UMAEquivariantVectorHead`` in isolation (no full UMA
backbone) to verify (a) the output contract / gradient flow and (b) the
equivariance property: because ``SO3_Linear`` shares one weight across the
three L=1 components, applying *any* linear map to the component axis of the
input commutes with the head, which is exactly the rotation-equivariance
condition in the spherical-harmonic basis.
"""

import pytest
import torch

from hydragnn.models.UMAStack import _UMAEquivariantVectorHead


def _random_node_embedding(num_nodes, lmax, channels, dtype):
    """(N, (lmax+1)**2, C) SO(3)-irrep-layout node embedding."""
    return torch.randn(num_nodes, (lmax + 1) ** 2, channels, dtype=dtype)


@pytest.mark.parametrize("num_vectors", [1, 3])
def pytest_output_shape_and_gradient(num_vectors):
    torch.manual_seed(0)
    num_nodes, lmax, channels = 5, 2, 8
    head = _UMAEquivariantVectorHead(channels, num_vectors=num_vectors).to(
        torch.float64
    )
    x = _random_node_embedding(num_nodes, lmax, channels, torch.float64)
    x.requires_grad_(True)

    out = head(x)
    assert out.shape == (num_nodes, 3 * num_vectors)

    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def pytest_reads_only_l1_component():
    """The head output must be invariant to changes in L>=2 channels and
    depend only on the L=0/L=1 block that SO3_Linear(lmax=1) consumes."""
    torch.manual_seed(1)
    num_nodes, lmax, channels = 4, 3, 6
    head = _UMAEquivariantVectorHead(channels, num_vectors=1).to(torch.float64)
    x = _random_node_embedding(num_nodes, lmax, channels, torch.float64)

    out_ref = head(x)
    # Perturb only L>=2 harmonics (indices 4 onward); output must not change.
    x_perturbed = x.clone()
    x_perturbed[:, 4:, :] += torch.randn_like(x_perturbed[:, 4:, :])
    out_perturbed = head(x_perturbed)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-10)


def pytest_equivariance_under_component_rotation():
    """Applying an orthogonal map R to the L=1 component axis of the input
    rotates the output vectors by the same R (equivariance)."""
    torch.manual_seed(2)
    num_nodes, lmax, channels, num_vectors = 7, 2, 8, 2
    head = _UMAEquivariantVectorHead(channels, num_vectors=num_vectors).to(
        torch.float64
    )
    x = _random_node_embedding(num_nodes, lmax, channels, torch.float64)

    # A random rotation (proper orthogonal) acting on the 3 L=1 components.
    a = torch.randn(3, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    R = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # orthogonal
    if torch.det(R) < 0:  # make it a proper rotation
        R[:, 0] = -R[:, 0]

    # Rotate the L=1 block (indices 1..3) along the component axis.
    x_rot = x.clone()
    x_rot[:, 1:4, :] = torch.einsum("ij,njc->nic", R, x[:, 1:4, :])

    out = head(x).view(num_nodes, num_vectors, 3)
    out_rot = head(x_rot).view(num_nodes, num_vectors, 3)

    expected = torch.einsum("ij,nvj->nvi", R, out)
    assert torch.allclose(out_rot, expected, atol=1e-9)


def pytest_invalid_num_vectors_channel_shapes():
    """Constructing the head with mismatched channels still yields a valid
    module; forward with wrong channel count must raise."""
    head = _UMAEquivariantVectorHead(8, num_vectors=1).to(torch.float64)
    bad = torch.randn(3, 4, 5, dtype=torch.float64)  # channels=5 != 8
    with pytest.raises(RuntimeError):
        head(bad)


def _build_uma_with_vector_head():
    """Build a small UMAStack with an energy (node) + 3-vector (node) head and
    the equivariant vector head enabled. Skips if the vendored UMA backbone
    dependencies are unavailable."""
    from hydragnn.models.create import create_model
    from hydragnn.utils.model.model import update_multibranch_heads

    output_heads = {
        "node": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [16],
            "type": "mlp",
        }
    }
    config_args = {
        "mpnn_type": "UMA",
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": [1, 3],  # head 0: scalar (node), head 1: 3-vector (node)
        "pe_dim": 6,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node", "node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0, 1.0],
        "num_conv_layers": 2,
        "num_nodes": None,
        "max_neighbours": 20,
        "radius": 5.0,
        "max_ell": 2,
        "num_radial": 8,
        "equivariance": True,
        "uma_equivariant_vector_head": True,  # auto-detect the dim-3 node head
        "enable_interatomic_potential": False,
        "use_gpu": False,
    }
    try:
        return create_model(**config_args)
    except ImportError as exc:  # vendored UMA backbone deps missing
        pytest.skip(f"UMA backbone unavailable: {exc}")


def _fully_connected(n):
    idx = torch.arange(n)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def _make_batch(dtype):
    from torch_geometric.data import Data, Batch

    graphs = []
    for n in (5, 7):
        pos = torch.randn(n, 3, dtype=dtype)
        pos = pos - pos.mean(dim=0, keepdim=True)  # translation-neutral
        x = torch.randint(1, 10, (n, 1)).to(dtype)
        edge_index = _fully_connected(n)
        d = Data(
            pos=pos,
            x=x,
            atomic_numbers=x[:, 0].long(),
            edge_index=edge_index,
            edge_shifts=torch.zeros(edge_index.size(1), 3, dtype=dtype),
        )
        graphs.append(d)
    return Batch.from_data_list(graphs)


def pytest_end_to_end_rotational_equivariance():
    """Through the real UMA backbone: rotating atomic positions rotates the
    equivariant vector-head output by the same rotation, while the scalar node
    head stays invariant."""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = _build_uma_with_vector_head().to(torch.float64)
        model.eval()

        assert getattr(model, "equivariant_vector_head", None) is not None
        # Auto-detect must pick the dim-3 node head (index 1).
        assert model._vector_head_index == 1

        batch = _make_batch(torch.float64)

        # Proper random rotation.
        a = torch.randn(3, 3, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        R = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if torch.det(R) < 0:
            R[:, 0] = -R[:, 0]

        with torch.no_grad():
            out = model(batch)
            scalar, vec = out[0], out[1]

            batch_rot = batch.clone()
            batch_rot.pos = batch.pos @ R.T
            batch_rot.edge_shifts = batch.edge_shifts @ R.T
            out_rot = model(batch_rot)
            scalar_rot, vec_rot = out_rot[0], out_rot[1]

        assert vec.shape == (batch.pos.shape[0], 3)
        # Vector head is equivariant: v_rot == v @ R^T. UMA's grid-based
        # feed-forward and spherical-harmonic norm are only approximately
        # equivariant, so a modest tolerance is used.
        assert torch.allclose(vec_rot, vec @ R.T, atol=1e-3), (
            (vec_rot - vec @ R.T).abs().max().item()
        )
        # Scalar node head is invariant.
        assert torch.allclose(scalar_rot, scalar, atol=1e-3), (
            (scalar_rot - scalar).abs().max().item()
        )
    finally:
        torch.set_default_dtype(prev_dtype)

