"""Contracts for the GINO port (integral transforms, batching, queries)."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from hydragnn.models.create import create_model
from hydragnn.models.gino import GINO, SinusoidalEmbedding, radius_neighbors
from hydragnn.models.registry import get_capabilities

pytestmark = pytest.mark.mpi_skip()


def _cloud(n, d=2, queries=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = Data(
        pos=torch.rand(n, d, generator=g, dtype=torch.float64),
        x=torch.randn(n, 2, generator=g, dtype=torch.float64),
        num_nodes=n,
    )
    if queries:
        data.query_pos = torch.rand(queries, d, generator=g, dtype=torch.float64)
    return data


def _model(levels=("node",), d=2, **kwargs):
    torch.manual_seed(0)
    settings = dict(
        coord_dim=d,
        n_modes=(4,) * d,
        latent_resolution=(8,) * d,
        in_radius=0.3,
        out_radius=0.3,
        embed_channels=4,
        in_channel_mlp_hidden_layers=(16,),
        out_channel_mlp_hidden_layers=(16,),
    )
    settings.update(kwargs)
    model = GINO(2, 8, [1] * len(levels), list(levels), 2, **settings)
    return model.double().eval()


def _args(**overrides):
    args = dict(
        mpnn_type="GINO",
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
        gino_n_modes=[4, 4],
        gino_latent_resolution=[8, 8],
        gino_in_radius=0.3,
        gino_out_radius=0.3,
    )
    args.update(overrides)
    return args


def test_sinusoidal_embedding_layout():
    emb = SinusoidalEmbedding(2, 3, max_positions=100)
    x = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    out = emb(x)
    assert out.shape == (1, emb.out_channels) == (1, 12)
    freqs = (1.0 / 100) ** (torch.arange(3, dtype=torch.float64) * 2 / 3)
    torch.testing.assert_close(out[0, 0:6:2], torch.sin(0.5 * freqs))
    torch.testing.assert_close(out[0, 1:6:2], torch.cos(0.5 * freqs))
    torch.testing.assert_close(out[0, 6::2], torch.sin(2.0 * freqs))


def test_radius_neighbors_respects_radius_and_graphs():
    q = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([[0.0, 0.1], [0.9, 0.0], [0.0, 0.5]])
    qi, yi = radius_neighbors(q, y, 0.2)
    assert sorted(zip(qi.tolist(), yi.tolist())) == [(0, 0), (1, 1)]
    qi, _ = radius_neighbors(q, y, 0.2, torch.tensor([0, 1]), torch.tensor([1, 1, 0]))
    assert qi.tolist() == [1]


@pytest.mark.parametrize("d", [2, 3])
def test_nonmatching_query_heads(d):
    data = _cloud(60, d, queries=23)
    node, query = _model(("node", "query"), d=d)(data)
    assert node.shape == (60, 1) and query.shape == (23, 1)


def test_query_at_node_positions_equals_node_head():
    data = _cloud(50, queries=0)
    data.query_pos = data.pos.clone()
    model = _model(("node", "query"))
    with torch.no_grad():
        model.heads[1].load_state_dict(model.heads[0].state_dict())
    node, query = model(data)
    torch.testing.assert_close(node, query)


def test_batch_matches_single_graphs():
    model = _model(("node", "query"))
    graphs = [_cloud(40, queries=9, seed=s) for s in range(3)]
    graphs[1] = _cloud(55, queries=4, seed=7)
    batch = Batch.from_data_list(graphs)
    batch.query_graph = torch.repeat_interleave(
        torch.arange(3), torch.tensor([g.query_pos.shape[0] for g in graphs])
    )
    node, query = model(batch)
    singles = [model(g) for g in graphs]
    torch.testing.assert_close(node, torch.cat([s[0] for s in singles]))
    torch.testing.assert_close(query, torch.cat([s[1] for s in singles]))


def test_points_outside_radius_contribute_nothing():
    model = _model()
    data = _cloud(30)
    far = Data(
        pos=torch.cat([data.pos, torch.full((5, 2), 50.0, dtype=torch.float64)]),
        x=torch.cat([data.x, torch.randn(5, 2, dtype=torch.float64)]),
        num_nodes=35,
    )
    torch.testing.assert_close(model(far)[0][:30], model(data)[0])


def test_batched_query_heads_need_graph_ids():
    model = _model(("node", "query"))
    batch = Batch.from_data_list([_cloud(20, queries=3), _cloud(20, queries=3)])
    with pytest.raises(ValueError, match="query_cell or query_graph"):
        model(batch)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(n_modes=(4, 4, 4)), "coord_dim entries"),
        (dict(in_radius=0.0), "radii"),
        (dict(fno_in_channels=5), "fno_in_channels"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


def test_complex_weights_survive_double():
    model = GINO(2, 8, [1], ["node"], 1, n_modes=(4, 4), latent_resolution=(8, 8))
    w = model.fno_blocks.convs[0].complex_weight().clone()
    model = model.double()
    torch.testing.assert_close(
        model.fno_blocks.convs[0].complex_weight(), w.to(torch.complex128)
    )


def test_tiny_training_reduces_loss():
    model = _model().train()
    data = _cloud(80)
    target = torch.sin(3 * data.pos[:, :1]) * data.pos[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = ((model(data)[0] - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


def test_registry_capabilities_and_create_model():
    caps = get_capabilities("GINO")
    assert caps.output_levels == frozenset({"node", "query"})
    model = create_model(
        **_args(output_type=["node", "query"], output_dim=[1, 1], task_weights=[1, 1])
    )
    assert isinstance(model, GINO) and model.num_latent == 64


def test_registry_rejects_global_attention():
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))
