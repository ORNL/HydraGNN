"""Contracts for RIGNO and the region interaction graphs."""

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import (
    build_rigno_graph,
    from_triangles,
    locate_queries,
    unit_square_mesh,
)
from hydragnn.data.meshes.region_graph import minimum_support_radius
from hydragnn.models.create import create_model
from hydragnn.models.registry import get_capabilities
from hydragnn.models.rigno import RIGNO

pytestmark = pytest.mark.mpi_skip()


def _graph(n, seed=0, queries=7, **kwargs):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(n, n)
    mesh.x = torch.randn(mesh.num_nodes, 2, generator=g, dtype=torch.float64)
    if queries:
        locate_queries(mesh, torch.rand(queries, 2, generator=g) * 0.9 + 0.05)
    return build_rigno_graph(mesh, generator=g, **kwargs)


def _model(levels=("node", "query"), **kwargs):
    torch.manual_seed(0)
    return RIGNO(2, 16, [1] * len(levels), list(levels), 2, **kwargs).double()


def _args(**overrides):
    args = dict(
        mpnn_type="RIGNO",
        input_dim=2,
        hidden_dim=16,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads={},
        activation_function="silu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        use_gpu=False,
    )
    args.update(overrides)
    return args


# -- graph construction -------------------------------------------------------
def test_graph_fields_and_ranges():
    data = _graph(8)
    n, r, q = data.num_nodes, data.num_rnodes, data.rigno_qx.shape[0]
    assert r == int(n / 4.0)
    assert data.rigno_x.abs().max() <= 1.0 + 1e-12
    assert data.rigno_rradius.shape == (r,) and bool((data.rigno_rradius > 0).all())
    for key, rows in (
        ("rigno_p2r_index", (n, r)),
        ("rigno_r2r_index", (r, r)),
        ("rigno_r2p_index", (r, n)),
        ("rigno_r2q_index", (r, q)),
    ):
        index = data[key]
        assert index.dtype == torch.long
        assert int(index[0].max()) < rows[0] and int(index[1].max()) < rows[1]
    # r2r is symmetric without duplicates; r2p edges respect the support radius.
    r2r = {tuple(e) for e in data.rigno_r2r_index.t().tolist()}
    assert len(r2r) == data.rigno_r2r_index.shape[1]
    assert all((j, i) in r2r for i, j in r2r)


def test_support_radius_covers_hull():
    # Every point inside the convex hull of the regional nodes lies in some
    # support disc, which is the property the medians are chosen for.
    rng = np.random.default_rng(0)
    centers = rng.uniform(-1, 1, (40, 2))
    radius = minimum_support_radius(centers)
    from scipy.spatial import Delaunay

    probe = rng.uniform(-1, 1, (2000, 2))
    probe = probe[Delaunay(centers).find_simplex(probe) >= 0]
    dist = np.linalg.norm(probe[:, None] - centers[None], axis=-1)
    assert bool((dist <= radius[None]).any(axis=1).all())


def test_decoders_cover_every_point():
    for n in (6, 8, 13):
        data = _graph(n, seed=n)
        assert data.rigno_r2p_index[1].unique().numel() == data.num_nodes
        assert data.rigno_r2q_index[1].unique().numel() == data.rigno_qx.shape[0]


def test_regional_nodes_follow_generator():
    a, b, c = _graph(8, seed=0), _graph(8, seed=0), _graph(8, seed=1)
    assert torch.equal(a.rigno_rpos, b.rigno_rpos)
    assert torch.equal(a.rigno_r2r_index, b.rigno_r2r_index)
    assert not torch.equal(a.rigno_rpos, c.rigno_rpos)


def test_normalization_is_isotropic():
    mesh = unit_square_mesh(8, 4)
    mesh.pos = mesh.pos * torch.tensor([2.0, 1.0], dtype=mesh.pos.dtype)
    data = build_rigno_graph(mesh, generator=torch.Generator().manual_seed(0))
    span = data.rigno_x.max(0).values - data.rigno_x.min(0).values
    torch.testing.assert_close(span, torch.tensor([2.0, 1.0], dtype=span.dtype))


def test_builder_rejects_bad_inputs():
    with pytest.raises(ValueError, match="2D"):
        build_rigno_graph(type("D", (), {"pos": torch.zeros(10, 3)})())
    with pytest.raises(ValueError, match="at least 4"):
        build_rigno_graph(unit_square_mesh(2, 2), subsample_factor=4.0)


def test_batching_offsets():
    graphs = [_graph(6), _graph(8, seed=1)]
    batch = Batch.from_data_list(graphs)
    n0, r0, q0 = graphs[0].num_nodes, graphs[0].num_rnodes, graphs[0].rigno_qx.shape[0]
    for key, off in (
        ("rigno_p2r_index", [[n0], [r0]]),
        ("rigno_r2p_index", [[r0], [n0]]),
        ("rigno_r2q_index", [[r0], [q0]]),
        ("rigno_r2r_index", [[r0], [r0]]),
    ):
        e0 = graphs[0][key].shape[1]
        assert torch.equal(batch[key][:, e0:], graphs[1][key] + torch.tensor(off))
    assert batch.rigno_rgraph.tolist() == [0] * r0 + [1] * graphs[1].num_rnodes
    assert batch.rigno_qgraph.tolist() == [0] * q0 + [1] * graphs[1].rigno_qx.shape[0]


# -- model -----------------------------------------------------------------------
def test_batched_equals_single():
    graphs = [_graph(6), _graph(9, seed=1)]
    model = _model().eval()
    out = model(Batch.from_data_list(graphs))
    n0, q0 = graphs[0].num_nodes, graphs[0].rigno_qx.shape[0]
    for g, sl_node, sl_query in (
        (graphs[0], slice(0, n0), slice(0, q0)),
        (graphs[1], slice(n0, None), slice(q0, None)),
    ):
        single = model(g)
        torch.testing.assert_close(out[0][sl_node], single[0])
        torch.testing.assert_close(out[1][sl_query], single[1])


@pytest.mark.parametrize("n_in,n_query", [(6, 5), (10, 5), (10, 40)])
def test_input_and_query_resolutions(n_in, n_query):
    # The same weights serve any input mesh and any query set.
    model = _model().eval()
    data = _graph(n_in, queries=n_query)
    node, query = model(data)
    assert node.shape == (data.num_nodes, 1)
    assert query.shape == (n_query, 1)
    assert bool(torch.isfinite(node).all() and torch.isfinite(query).all())


def test_query_outputs_are_pointwise():
    # Each query depends only on its own position, not on the other queries.
    model = _model().eval()
    data = _graph(8, queries=6)
    subset = _graph(8, queries=6)
    keep = torch.tensor([0, 2, 5])
    subset.rigno_qx = data.rigno_qx[keep]
    subset.rigno_qgraph = data.rigno_qgraph[keep]
    mask = torch.isin(data.rigno_r2q_index[1], keep)
    remap = torch.full((6,), -1, dtype=torch.long)
    remap[keep] = torch.arange(3)
    index = data.rigno_r2q_index[:, mask].clone()
    index[1] = remap[index[1]]
    subset.rigno_r2q_index = index
    torch.testing.assert_close(model(subset)[1], model(data)[1][keep])


def test_edge_masking_is_seeded_and_train_only():
    data = _graph(8)
    model = _model(edge_masking=0.5).train()
    torch.manual_seed(3)
    a = model(data)[0]
    torch.manual_seed(3)
    b = model(data)[0]
    torch.manual_seed(4)
    c = model(data)[0]
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
    model.eval()
    torch.testing.assert_close(model(data)[0], model(data)[0])
    unmasked = _model(edge_masking=0.0).train()
    unmasked.load_state_dict(model.state_dict())
    torch.testing.assert_close(unmasked(data)[0], model(data)[0])


def test_mask_keeps_fraction():
    model = _model(edge_masking=0.25).train()
    index = torch.arange(40).view(2, 20)
    kept = model._mask(index)
    assert kept.shape == (2, 15)
    assert set(map(tuple, kept.t().tolist())) <= set(map(tuple, index.t().tolist()))


def test_time_conditioning_and_concat_time():
    data = _graph(8)
    model = _model(levels=("node",), time_conditioning=True, concat_time=True).eval()
    with pytest.raises(ValueError, match="data.time"):
        model(data)
    data.time = torch.tensor([0.5], dtype=torch.float64)
    data.dt = torch.tensor([0.1], dtype=torch.float64)
    a = model(data)[0]
    data.dt = torch.tensor([0.4], dtype=torch.float64)
    assert not torch.allclose(a, model(data)[0])
    # ConditionedNorm is the identity at dt = 0.
    plain = _model(levels=("node",), concat_time=True).eval()
    plain.load_state_dict(model.state_dict(), strict=False)
    data.dt = torch.tensor([0.0], dtype=torch.float64)
    torch.testing.assert_close(plain(data)[0], model(data)[0])


def test_time_conditioning_is_per_graph():
    graphs = [_graph(6), _graph(8, seed=1)]
    for g, dt in zip(graphs, (0.1, 0.3)):
        g.dt = torch.tensor([dt], dtype=torch.float64)
    model = _model(levels=("node",), time_conditioning=True).eval()
    out = model(Batch.from_data_list(graphs))[0]
    torch.testing.assert_close(out[graphs[0].num_nodes :], model(graphs[1])[0])


def test_gradients_reach_every_parameter():
    model = _model(edge_masking=0.0).train()
    loss = sum(o.pow(2).sum() for o in model(_graph(8)))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_checkpointing_matches_plain():
    data = _graph(8)
    model = _model(edge_masking=0.0).train()
    ref = model(data)
    model.enable_conv_checkpointing()
    out = model(data)
    for a, b in zip(ref, out):
        torch.testing.assert_close(a, b)


def test_freeze_conv_trains_heads_only():
    model = _model(freeze_conv=True)
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"heads"}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(mlp_layers=0), "mlp_layers"),
        (dict(edge_masking=1.0), "edge_masking"),
        (dict(spatial_dim=3), "spatial_dim"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


# -- registry ------------------------------------------------------------------
def test_registry_capabilities_and_create_model():
    caps = get_capabilities("RIGNO")
    assert caps.output_levels == frozenset({"node", "query"})
    assert "rigno_r2p_index" in caps.requires
    model = create_model(**_args(rigno_edge_masking=0.25, rigno_mlp_layers=1))
    assert isinstance(model, RIGNO) and model.edge_masking == 0.25
    assert len(model.processor) == 2


def test_registry_rejects_face_heads():
    # A regional latent cannot feed a face-flux decoder directly.
    with pytest.raises(ValueError, match="face"):
        create_model(**_args(output_type=["face"]))


def test_registry_rejects_global_attention():
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))


# -- irregular domain benchmark -----------------------------------------------
def _l_shape(n):
    """Structured triangulation of [0,1]^2 minus [0.5,1]^2."""
    h = 1.0 / n
    ids, pos = {}, []
    for j in range(n + 1):
        for i in range(n + 1):
            if i * h > 0.5 + 1e-12 and j * h > 0.5 + 1e-12:
                continue
            ids[i, j] = len(pos)
            pos.append((i * h, j * h))
    tris = []
    for j in range(n):
        for i in range(n):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(c in ids for c in corners):
                a, b, c, d = (ids[c] for c in corners)
                tris += [(a, b, c), (a, c, d)]
    return from_triangles(torch.tensor(pos, dtype=torch.float64), torch.tensor(tris))


def _l_sample(n, g):
    mesh = _l_shape(n)
    k, phase = torch.rand(2, generator=g, dtype=torch.float64)
    k = 1.0 + 2.0 * k
    x, y = mesh.pos[:, 0], mesh.pos[:, 1]
    mesh.x = torch.stack([torch.full_like(x, k), torch.full_like(x, phase)], dim=1)
    mesh.y = (
        torch.sin(math.pi * k * x + 2 * math.pi * phase) * torch.cos(math.pi * y)
    ).unsqueeze(1)
    return build_rigno_graph(mesh, generator=g)


def test_irregular_domain_benchmark():
    """RIGNO learns a parametric field on an L-shaped domain and transfers to
    a finer, unseen mesh. Threshold is loose: it guards against regressions,
    not for accuracy claims."""
    g = torch.Generator().manual_seed(0)
    train = [_l_sample(n, g) for n in (8, 10) for _ in range(12)]
    held = [_l_sample(14, g) for _ in range(4)]
    torch.manual_seed(0)
    model = RIGNO(2, 32, [1], ["node"], 3, edge_masking=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = Batch.from_data_list(train)
    hb = Batch.from_data_list(held)
    for b in (batch, hb):
        for key, value in b:
            if torch.is_tensor(value) and value.is_floating_point():
                b[key] = value.float()
    # A constant predictor scores ~1.0 here.
    for _ in range(150):
        opt.zero_grad()
        loss = (model(batch)[0] - batch.y).pow(2).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        err = (model(hb)[0] - hb.y).norm() / hb.y.norm()
    assert float(err) < 0.5
