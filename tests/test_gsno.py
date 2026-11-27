"""Contracts for the graph Laplacian eigenbasis and the GSNO."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import (
    attach_graph_spectral_basis,
    graph_spectral_basis,
    unit_square_mesh,
)
from hydragnn.data.meshes.graph_spectral import normalized_laplacian
from hydragnn.models.create import create_model
from hydragnn.models.gsno import GSNO, GraphSpectralConv
from hydragnn.models.registry import get_capabilities

pytestmark = pytest.mark.mpi_skip()


def _mesh(n=6, k=8, seed=0, channels=2):
    mesh = unit_square_mesh(n, n)
    g = torch.Generator().manual_seed(seed)
    mesh.x = torch.randn(mesh.num_nodes, channels, generator=g, dtype=torch.float64)
    attach_graph_spectral_basis(mesh, k)
    return mesh


def _ring(n):
    """Cycle graph: every nonzero eigenvalue of its Laplacian is doubly degenerate."""
    angle = 2 * np.pi * torch.arange(n, dtype=torch.float64) / n
    pos = torch.stack([angle.cos(), angle.sin()], dim=1)
    i = torch.arange(n)
    edge_index = torch.stack([i, (i + 1) % n])
    return pos, torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _model(**kwargs):
    torch.manual_seed(0)
    settings = dict(spatial_modes=8, projection_channels=16)
    settings.update(kwargs)
    return GSNO(2, 8, [1], ["node"], 2, **settings).double().eval()


def _args(**overrides):
    args = dict(
        mpnn_type="GSNO",
        input_dim=2,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads={},
        activation_function="gelu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        use_gpu=False,
        gsno_spatial_modes=4,
    )
    args.update(overrides)
    return args


# -- eigenbasis ------------------------------------------------------------------
def test_basis_is_orthonormal_eigenbasis():
    mesh = unit_square_mesh(6, 6)
    values, basis, _, info = graph_spectral_basis(mesh.pos, mesh.edge_index, 10)
    torch.testing.assert_close(basis.T @ basis, torch.eye(10, dtype=torch.float64))
    laplacian, _ = normalized_laplacian(mesh.pos, mesh.edge_index, info["sigma"])
    residual = laplacian @ basis.numpy() - basis.numpy() * values.numpy()
    assert np.abs(residual).max() < 1e-10
    assert values[0].abs() < 1e-10 and bool((values[1:] >= values[:-1]).all())


def test_basis_signs_are_canonical():
    mesh = unit_square_mesh(5, 5)
    _, basis, _, _ = graph_spectral_basis(mesh.pos, mesh.edge_index, 6)
    peak = basis.abs().argmax(dim=0)
    assert bool((basis[peak, torch.arange(6)] > 0).all())


def test_lobpcg_matches_dense_spectrum():
    mesh = unit_square_mesh(10, 10)
    dense, *_ = graph_spectral_basis(mesh.pos, mesh.edge_index, 6, method="dense")
    sparse, basis, _, _ = graph_spectral_basis(
        mesh.pos, mesh.edge_index, 6, method="lobpcg"
    )
    torch.testing.assert_close(sparse, dense, atol=1e-8, rtol=0)
    torch.testing.assert_close(basis.T @ basis, torch.eye(6, dtype=torch.float64))


def test_degenerate_groups_and_split_flag():
    pos, edge_index = _ring(12)
    values, _, groups, info = graph_spectral_basis(pos, edge_index, 4)
    assert groups.tolist() == [0, 1, 1, 3]
    assert info["split_eigenspace"]  # mode 4 pairs with mode 3
    _, _, _, info = graph_spectral_basis(pos, edge_index, 5)
    assert not info["split_eigenspace"]


def test_basis_validation():
    mesh = unit_square_mesh(3, 3)
    with pytest.raises(ValueError, match="num_modes"):
        graph_spectral_basis(mesh.pos, mesh.edge_index, mesh.num_nodes)
    with pytest.raises(ValueError, match="sigma"):
        graph_spectral_basis(mesh.pos, mesh.edge_index, 2, sigma=0.0)


# -- model --------------------------------------------------------------------
def test_sign_flip_invariance():
    model = _model()
    mesh = _mesh()
    ref = model(mesh)[0]
    flip = torch.tensor([1.0, -1.0] * 4, dtype=torch.float64)
    mesh.gsno_basis = mesh.gsno_basis * flip
    torch.testing.assert_close(model(mesh)[0], ref)


def _rotate_pair(basis, a, b, angle):
    out = basis.clone()
    c, s = np.cos(angle), np.sin(angle)
    out[:, a], out[:, b] = (
        c * basis[:, a] - s * basis[:, b],
        s * basis[:, a] + c * basis[:, b],
    )
    return out


def test_degenerate_rotation_invariance_needs_tying():
    pos, edge_index = _ring(16)
    from torch_geometric.data import Data

    data = Data(pos=pos, edge_index=edge_index, num_nodes=16)
    data.x = torch.randn(
        16, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    attach_graph_spectral_basis(data, 5)
    assert data.gsno_groups.tolist() == [[0, 1, 1, 3, 3]]
    rotated = data.clone()
    rotated.gsno_basis = _rotate_pair(data.gsno_basis, 1, 2, 0.7)
    tied = _model(spatial_modes=5, tie_degenerate=True)
    torch.testing.assert_close(tied(rotated)[0], tied(data)[0])
    untied = _model(spatial_modes=5)
    assert (untied(rotated)[0] - untied(data)[0]).abs().max() > 1e-6


def test_full_basis_reconstruction_is_identity_operator():
    """With all N modes and R = I the spectral path is the identity."""
    mesh = unit_square_mesh(3, 3)
    n = mesh.num_nodes
    laplacian, _ = normalized_laplacian(mesh.pos, mesh.edge_index)
    _, vectors = np.linalg.eigh(laplacian.toarray())
    basis = torch.from_numpy(vectors)
    conv = GraphSpectralConv(2, n).double()
    with torch.no_grad():
        conv.weight.copy_(torch.eye(2).expand(n, 2, 2))
    v = torch.randn(n, 1, 2, dtype=torch.float64)
    out = conv(v, basis, torch.zeros(n, dtype=torch.long), 1)
    torch.testing.assert_close(out, v)


def test_temporal_modes_shapes_and_low_pass():
    conv = GraphSpectralConv(1, 4, temporal_modes=2).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[..., 0, 0, 0] = 1.0  # R = 1 on kept temporal modes
    mesh = _mesh(k=4)
    t = torch.arange(10, dtype=torch.float64) / 10
    phi = mesh.gsno_basis[:, :1]
    v = phi * (1 + torch.cos(2 * torch.pi * t) + torch.cos(2 * torch.pi * 4 * t))
    graph = torch.zeros(mesh.num_nodes, dtype=torch.long)
    out = conv(v.unsqueeze(-1), mesh.gsno_basis, graph, 1)[..., 0]
    torch.testing.assert_close(out, phi * (1 + torch.cos(2 * torch.pi * t)))

    model = GSNO(
        2 * 3,
        8,
        [2 * 6],
        ["node"],
        1,
        spatial_modes=4,
        temporal_modes=3,
        input_time_steps=3,
        output_time_steps=6,
        projection_channels=8,
    ).double()
    mesh.x = torch.randn(mesh.num_nodes, 6, dtype=torch.float64)
    assert model(mesh)[0].shape == (mesh.num_nodes, 12)


def test_batch_of_different_meshes_matches_single():
    model = _model()
    graphs = [_mesh(5, seed=0), _mesh(7, seed=1), _mesh(6, seed=2)]
    batched = model(Batch.from_data_list(graphs))[0]
    torch.testing.assert_close(batched, torch.cat([model(g)[0] for g in graphs]))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(spatial_modes=0), "spatial_modes"),
        (dict(output_time_steps=2), "1 time step"),
        (dict(temporal_modes=5, output_time_steps=4), "T_out // 2"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GSNO(2, 8, [4], ["node"], 1, **kwargs)


def test_missing_basis_or_groups():
    mesh = _mesh(k=4)
    with pytest.raises(ValueError, match="gsno_basis"):
        _model()(mesh)
    mesh = _mesh()
    del mesh.gsno_groups
    with pytest.raises(ValueError, match="gsno_groups"):
        _model(tie_degenerate=True)(mesh)


def test_tiny_training_reduces_loss():
    torch.manual_seed(0)
    model = _model().train()
    mesh = _mesh(8)
    target = mesh.gsno_basis[:, 1:3] @ torch.tensor(
        [[1.0], [-0.5]], dtype=torch.float64
    )
    target = target + 0.3 * mesh.x[:, :1]
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((model(mesh)[0] - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.2 * losses[0]


def test_registry_capabilities_and_create_model():
    caps = get_capabilities("GSNO")
    assert "gsno_basis" in caps.requires
    model = create_model(**_args(gsno_projection_channels=16))
    assert isinstance(model, GSNO) and model.spatial_modes == 4


def test_registry_rejects_global_attention():
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))
