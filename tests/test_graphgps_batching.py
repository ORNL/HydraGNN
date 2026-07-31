import types

import torch

from hydragnn.globalAtt.gps import GPSConv
from hydragnn.models.Base import Base


def _make_toy_base_for_embedding(hidden_dim=8, pe_dim=3, x_dim=5):
    model = Base.__new__(Base)
    model.use_edge_attr = False
    model.use_global_attn = True
    model.input_dim = x_dim
    model.is_edge_model = False

    model.pos_emb = torch.nn.Linear(pe_dim, hidden_dim, bias=False)
    model.node_emb = torch.nn.Linear(x_dim, hidden_dim, bias=False)
    model.node_lin = torch.nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

    return model


def test_base_embedding_adds_graph_batch_for_global_attention():
    torch.manual_seed(0)

    model = _make_toy_base_for_embedding()
    data = types.SimpleNamespace(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_shifts=torch.zeros(3, 3),
        pe=torch.randn(4, 3),
        x=torch.randn(4, 5),
        pos=torch.randn(4, 3),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )

    _, _, conv_args = Base._embedding(model, data)

    assert "graph_batch" in conv_args
    assert torch.equal(conv_args["graph_batch"], data.batch)


def test_gpsconv_attention_isolated_per_graph_in_minibatch():
    torch.manual_seed(1)

    conv = GPSConv(channels=8, conv=None, heads=2, dropout=0.0, attn_type="multihead")
    conv.eval()

    # Two graphs in one mini-batch, each with two nodes.
    x = torch.randn(4, 8)
    equiv = torch.zeros(4, 3)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    y_batched, _ = conv(inv_node_feat=x, equiv_node_feat=equiv, graph_batch=batch)

    # Running each graph independently should match the corresponding slice
    # of the batched output when graph_batch is respected.
    y_g0, _ = conv(inv_node_feat=x[:2], equiv_node_feat=equiv[:2], graph_batch=None)
    y_g1, _ = conv(inv_node_feat=x[2:], equiv_node_feat=equiv[2:], graph_batch=None)

    assert torch.allclose(y_batched[:2], y_g0, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y_batched[2:], y_g1, atol=1e-6, rtol=1e-6)

    # Perturb only graph 1 features. Graph 0 output must not change.
    x_perturbed = x.clone()
    x_perturbed[2:] = x_perturbed[2:] + 10.0
    y_perturbed, _ = conv(
        inv_node_feat=x_perturbed,
        equiv_node_feat=equiv,
        graph_batch=batch,
    )

    assert torch.allclose(y_batched[:2], y_perturbed[:2], atol=1e-6, rtol=1e-6)


def test_gpsconv_single_graph_batch_none_unchanged():
    torch.manual_seed(2)

    conv = GPSConv(channels=8, conv=None, heads=2, dropout=0.0, attn_type="multihead")
    conv.eval()

    x = torch.randn(3, 8)
    equiv = torch.zeros(3, 3)

    y_none, _ = conv(inv_node_feat=x, equiv_node_feat=equiv, graph_batch=None)
    y_single_batch, _ = conv(
        inv_node_feat=x,
        equiv_node_feat=equiv,
        graph_batch=torch.zeros(3, dtype=torch.long),
    )

    assert torch.allclose(y_none, y_single_batch, atol=1e-6, rtol=1e-6)
