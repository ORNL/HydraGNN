import copy

import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import from_triangles, locate_queries, unit_square_mesh
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


def _mesh(nx, ny, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(nx, ny, dtype=dtype)
    mesh.x = torch.randn(mesh.num_nodes, 3, generator=g, dtype=dtype)
    return mesh


@pytest.fixture
def model():
    return create_model(**_args()).eval()


def test_capabilities():
    caps = get_capabilities("MeshGraphNet")
    assert caps.output_levels == frozenset({"node", "face", "cell", "query"})
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
    with pytest.raises(ValueError, match=r"cannot predict \['graph'\]"):
        create_model(**_args(output_dim=[1], output_type=["graph"], task_weights=[1.0]))


def test_create_model_rejects_mesh_levels_for_legacy_models():
    with pytest.raises(ValueError, match=r"GIN cannot predict \['face'\]"):
        create_model(
            **_args(
                mpnn_type="GIN",
                output_dim=[1],
                output_type=["face"],
                task_weights=[1.0],
            )
        )


def _mesh_model(conservative=True, **overrides):
    args = _args(
        output_dim=[1, 2, 1],
        output_type=["node", "face", "cell"],
        task_weights=[1.0, 1.0, 1.0],
        mgn_conservative_faces=conservative,
    )
    args.update(overrides)
    return create_model(**args).double().eval()


def _flip_faces(mesh):
    flipped = copy.copy(mesh)
    flipped.face_index = mesh.face_index.flip(0)
    flipped.face_nodes = mesh.face_nodes.flip(0)
    flipped.face_normal = -mesh.face_normal
    return flipped


def test_face_and_cell_heads_shapes():
    model = _mesh_model()
    mesh = _mesh(3, 2, 0, torch.float64)
    node, face, cell = model(mesh)
    assert node.shape == (mesh.num_nodes, 1)
    assert face.shape == (mesh.num_faces, 2)
    assert cell.shape == (mesh.num_cells, 1)


def test_conservative_face_head_is_antisymmetric():
    model = _mesh_model()
    mesh = _mesh(3, 3, 1, torch.float64)
    face = model(mesh)[1]
    face_flipped = model(_flip_faces(mesh))[1]
    torch.testing.assert_close(face_flipped, -face, rtol=0, atol=1e-12)
    # Each interior flux leaves one cell and enters the other, so the net
    # contribution of interior faces to the total balance is exactly zero.
    net = torch.zeros(mesh.num_cells, 2, dtype=torch.float64)
    net.index_add_(0, mesh.face_index[0], face)
    net.index_add_(0, mesh.face_index[1], -face)
    torch.testing.assert_close(
        net.sum(0), torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12
    )


def test_non_conservative_face_head_is_not_antisymmetric():
    model = _mesh_model(conservative=False)
    mesh = _mesh(3, 3, 1, torch.float64)
    face = model(mesh)[1]
    face_flipped = model(_flip_faces(mesh))[1]
    assert not torch.allclose(face_flipped, -face)


def test_mesh_heads_batched_equals_single():
    model = _mesh_model()
    meshes = [_mesh(2, 3, 2, torch.float64), _mesh(3, 1, 3, torch.float64)]
    batched = model(Batch.from_data_list(meshes))
    singles = [model(m) for m in meshes]
    for head in range(3):
        expected = torch.cat([out[head] for out in singles])
        torch.testing.assert_close(batched[head], expected, rtol=0, atol=1e-12)


def _query_model(**overrides):
    args = _args(output_dim=[1, 1], output_type=["node", "query"])
    args.update(overrides)
    return create_model(**args).double().eval()


def _query_mesh(nx, ny, nq, seed):
    mesh = _mesh(nx, ny, seed, torch.float64)
    g = torch.Generator().manual_seed(seed + 100)
    return locate_queries(mesh, torch.rand(nq, 2, generator=g, dtype=torch.float64))


def test_query_head_interpolates_node_latent():
    model = _query_model()
    mesh = _mesh(3, 2, 0, torch.float64)
    mesh = locate_queries(mesh, mesh.pos.clone())
    with torch.no_grad():
        state = model.process(model.encode(mesh), mesh)
        latent = model._query_latent(state["node"], mesh)
    # At mesh nodes the barycentric interpolant returns the node latent.
    torch.testing.assert_close(latent, state["node"], rtol=0, atol=1e-12)
    node, query = model(mesh)
    assert query.shape == (mesh.num_nodes, 1)


def test_query_head_batched_equals_single():
    model = _query_model()
    meshes = [_query_mesh(2, 3, 4, 1), _query_mesh(3, 1, 6, 2)]
    with torch.no_grad():
        batched = model(Batch.from_data_list(meshes))
        singles = [model(m) for m in meshes]
    assert batched[1].shape == (10, 1)
    for head in range(2):
        expected = torch.cat([out[head] for out in singles])
        torch.testing.assert_close(batched[head], expected, rtol=0, atol=1e-12)


def test_query_head_low_precision_latent_keeps_fp64_geometry():
    model = _query_model()
    mesh = _query_mesh(3, 3, 5, 3)
    node = torch.randn(mesh.num_nodes, 16, dtype=torch.float64)
    expected = model._query_latent(node, mesh).to(torch.bfloat16)
    latent = model._query_latent(node.to(torch.bfloat16), mesh)
    assert latent.dtype == torch.bfloat16
    torch.testing.assert_close(latent, expected, rtol=2e-2, atol=2e-2)


def test_freeze_conv_freezes_trunk_only():
    model = create_model(**_args(freeze_conv=True))
    for module in (model.node_encoder, model.edge_encoder, model.processor):
        assert not any(p.requires_grad for p in module.parameters())
    assert all(p.requires_grad for p in model.decoders.parameters())


def test_prelu_not_shared_across_mlps():
    model = create_model(**_args(activation_function="prelu", freeze_conv=True))
    prelus = [m for m in model.modules() if isinstance(m, torch.nn.PReLU)]
    assert len({id(m) for m in prelus}) == len(prelus) > 1
    decoder_prelus = [
        m for m in model.decoders.modules() if isinstance(m, torch.nn.PReLU)
    ]
    assert decoder_prelus and all(m.weight.requires_grad for m in decoder_prelus)


def test_query_head_rejects_non_triangle_cells():
    model = _query_model()
    mesh = _query_mesh(3, 2, 4, 0)
    # 12 triangles relabelled as 9 quads: reshape to triangles would succeed.
    mesh.cell_size = torch.full((9,), 4, dtype=torch.long)
    with pytest.raises(ValueError, match="triangle meshes only"):
        model._query_latent(torch.zeros(mesh.num_nodes, 16), mesh)


def test_face_head_distance_uses_coordinate_precision():
    model = _mesh_model()
    mesh = unit_square_mesh(2, 1, dtype=torch.float64)
    # Shift far from the origin: BF16 cannot resolve the centroid spacing.
    mesh.pos = mesh.pos * 1e-2 + 256.0
    mesh.x = torch.randn(mesh.num_nodes, 3, dtype=torch.float64)
    captured = []
    first = model.decoders[1][0]
    hook = first.register_forward_hook(lambda m, i, o: captured.append(i[0]))
    node = torch.randn(mesh.num_nodes, 16, dtype=torch.bfloat16)
    model.decoders.to(torch.bfloat16)
    with torch.no_grad():
        model._decode_face(model.decoders[1], 2, node, mesh)
    hook.remove()
    dist = captured[0][:, -1].double()
    assert (dist > 0).all()


def test_empty_face_head_gives_finite_loss():
    model = create_model(
        **_args(
            output_dim=[1, 1],
            output_type=["node", "face"],
            mgn_conservative_faces=True,
        )
    )
    # One triangle per sample: no interior faces anywhere in the batch.
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    meshes = []
    for _ in range(2):
        tri = from_triangles(pos, torch.tensor([[0, 1, 2]]))
        tri.x = torch.randn(3, 3)
        meshes.append(tri)
    batch = Batch.from_data_list(meshes)
    assert batch.face_index.shape[1] == 0
    node, face = model(batch)
    assert face.shape == (0, 1)
    y = torch.randn(batch.num_nodes, 1)
    head_index = [torch.arange(batch.num_nodes), torch.arange(0)]
    model.loss_component_names = [["u"], ["q"]]
    total, per_head = model.loss([node, face], y, head_index)
    assert torch.isfinite(total) and per_head[1].item() == 0.0
    assert all(torch.isfinite(v["raw"]) for v in model.last_loss_components.values())
    total.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_query_head_requires_2d():
    with pytest.raises(ValueError, match="mgn_spatial_dim=2"):
        _query_model(mgn_spatial_dim=3)


def test_conservative_face_head_gaussian_variance_positive():
    model = _mesh_model(loss_function_type="GaussianNLLLoss")
    mesh = _mesh(2, 2, 4, torch.float64)
    means, variances = model(mesh)
    assert means[1].shape == variances[1].shape == (mesh.num_faces, 2)
    assert (variances[1] > 0).all()
    _, flipped_var = model(_flip_faces(mesh))
    torch.testing.assert_close(flipped_var[1], variances[1], rtol=0, atol=1e-12)


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


@pytest.mark.parametrize("residual", [False, True])
def test_rollout_feeds_node_head_back(residual):
    from hydragnn.models.rollout import rollout

    model = create_model(
        **_args(output_dim=[1], output_type=["node"], task_weights=[1.0])
    )
    model.eval()
    mesh = _mesh(3, 2, 10)
    states = rollout(model, mesh, 3, [1], level="node", residual=residual)
    x = mesh.x.clone()
    data = copy.copy(mesh)
    with torch.no_grad():
        for state in states:
            out = model(data)[0]
            x = x.clone()
            x[:, 1:2] = x[:, 1:2] + out if residual else out
            data.x = x
            torch.testing.assert_close(state, x[:, 1:2])
    assert mesh.x.data_ptr() != data.x.data_ptr()


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


def test_rollout_unpacks_gaussian_nll_means():
    from hydragnn.models.rollout import rollout

    model = create_model(
        **_args(
            output_dim=[1],
            output_type=["node"],
            task_weights=[1.0],
            loss_function_type="GaussianNLLLoss",
        )
    ).eval()
    mesh = _mesh(3, 2, 11)
    states = rollout(model, mesh, 2, [1], level="node")
    with torch.no_grad():
        means, _ = model(mesh)
    torch.testing.assert_close(states[0], means[0])
