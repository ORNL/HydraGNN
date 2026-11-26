import copy

import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import unit_square_mesh
from hydragnn.models.create import create_model
from hydragnn.models.registry import Capabilities, get_capabilities

pytestmark = pytest.mark.mpi_skip()


def _args(**overrides):
    args = dict(
        mpnn_type="MeshGraphNet",
        input_dim=3,
        hidden_dim=16,
        output_dim=[1, 2],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node", "node"],
        output_heads={},
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0, 1.0],
        num_conv_layers=3,
        use_gpu=False,
        mgn_spatial_dim=2,
    )
    args.update(overrides)
    return args


def _mesh(nx, ny, seed):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(nx, ny, dtype=torch.float32)
    mesh.x = torch.randn(mesh.num_nodes, 3, generator=g)
    return mesh


@pytest.fixture
def model():
    return create_model(**_args()).eval()


def test_capabilities():
    caps = get_capabilities("MeshGraphNet")
    assert caps.output_levels == frozenset({"node"})
    assert "pos" in caps.requires
    assert get_capabilities("GIN") == Capabilities()


def test_trainer_contract(model):
    assert model.num_heads == 2
    assert model.head_dims == [1, 2]
    assert model.head_type == ["node", "node"]
    assert model.var_output == 0
    assert model.loss_weights == [0.5, 0.5]
    mesh = _mesh(3, 3, 0)
    out = model(mesh)
    assert [o.shape for o in out] == [(mesh.num_nodes, 1), (mesh.num_nodes, 2)]
    y = torch.cat([torch.zeros(mesh.num_nodes, 1), torch.ones(mesh.num_nodes, 2)], 1)
    head_index = [
        torch.arange(mesh.num_nodes),
        torch.arange(mesh.num_nodes, 3 * mesh.num_nodes),
    ]
    total, per_head = model.loss(out, y.reshape(-1, 1), head_index)
    total.backward()
    assert len(per_head) == 2
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_rejects_graph_heads():
    with pytest.raises(ValueError, match="supports output levels"):
        create_model(**_args(output_dim=[1], output_type=["graph"], task_weights=[1.0]))


def test_node_permutation_equivariance(model):
    mesh = _mesh(3, 2, 1)
    perm = torch.randperm(mesh.num_nodes, generator=torch.Generator().manual_seed(2))
    inv = torch.argsort(perm)
    permuted = copy.copy(mesh)
    permuted.x = mesh.x[perm]
    permuted.pos = mesh.pos[perm]
    permuted.edge_index = inv[mesh.edge_index]
    with torch.no_grad():
        ref = model(mesh)
        out = model(permuted)
    for r, o in zip(ref, out):
        torch.testing.assert_close(o, r[perm], rtol=1e-5, atol=1e-5)


def test_translation_invariance(model):
    mesh = _mesh(2, 2, 3)
    shifted = copy.copy(mesh)
    shifted.pos = mesh.pos + torch.tensor([5.0, -2.0])
    with torch.no_grad():
        for r, o in zip(model(mesh), model(shifted)):
            torch.testing.assert_close(o, r, rtol=1e-5, atol=1e-5)


def test_batched_matches_single(model):
    meshes = [_mesh(2, 2, 4), _mesh(3, 1, 5), _mesh(1, 3, 6)]
    batch = Batch.from_data_list(meshes)
    with torch.no_grad():
        batched = model(batch)
        singles = [model(m) for m in meshes]
    for ihead in range(model.num_heads):
        torch.testing.assert_close(
            batched[ihead], torch.cat([s[ihead] for s in singles]), rtol=1e-5, atol=1e-5
        )


def test_state_dict_round_trip_and_determinism(model):
    other = create_model(**_args()).eval()
    mesh = _mesh(2, 3, 7)
    with torch.no_grad():
        for a, b in zip(model(mesh), other(mesh)):
            torch.testing.assert_close(a, b)
    fresh = create_model(**_args()).eval()
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(1.0)
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        for a, b in zip(model(mesh), fresh(mesh)):
            torch.testing.assert_close(a, b)


def test_gaussian_nll_outputs_positive_variance():
    model = create_model(**_args(loss_function_type="GaussianNLLLoss")).eval()
    mesh = _mesh(2, 2, 8)
    with torch.no_grad():
        means, variances = model(mesh)
    assert [m.shape[1] for m in means] == [1, 2]
    assert all((v > 0).all() for v in variances)


def test_conv_checkpointing_matches(model):
    ckpt = create_model(**_args(conv_checkpointing=True))
    ckpt.load_state_dict(model.state_dict())
    ckpt.train()
    model.train()
    mesh = _mesh(2, 2, 9)
    for a, b in zip(model(mesh), ckpt(mesh)):
        torch.testing.assert_close(a, b)
