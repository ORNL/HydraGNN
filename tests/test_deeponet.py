"""Contracts for the DeepONet port (DeepXDE strategies, sensors, queries)."""

import os

import pytest
import torch
from torch_geometric.data import Batch, Data

from hydragnn.distributed import setup_ddp
from hydragnn.models.create import create_model
from hydragnn.models.deeponet import DeepONet, GroupedFNN
from hydragnn.models.registry import get_capabilities
from hydragnn.models.utils import load_existing_model, save_model

pytestmark = pytest.mark.mpi_skip()

M, C = 4, 2  # sensors, channels per sensor


def _graph(n=12, queries=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = Data(
        pos=torch.rand(n, 2, generator=g, dtype=torch.float64),
        x=torch.randn(n, C, generator=g, dtype=torch.float64),
        deeponet_sensors=torch.randn(1, M * C, generator=g, dtype=torch.float64),
        num_nodes=n,
    )
    if queries:
        data.query_pos = torch.rand(queries, 2, generator=g, dtype=torch.float64)
    return data


def _model(levels=("node",), dims=None, **kwargs):
    torch.manual_seed(0)
    settings = dict(num_sensors=M, basis_dim=8)
    settings.update(kwargs)
    dims = dims or [1] * len(levels)
    return DeepONet(C, 16, dims, list(levels), 2, **settings).double().eval()


def _args(**overrides):
    args = dict(
        mpnn_type="DeepONet",
        input_dim=C,
        hidden_dim=16,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads={},
        activation_function="tanh",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        use_gpu=False,
        deeponet_num_sensors=M,
        deeponet_basis_dim=8,
    )
    args.update(overrides)
    return args


def test_grouped_fnn_matches_separate_nets():
    torch.manual_seed(0)
    net = GroupedFNN(3, [4, 5, 2], torch.tanh).double()
    x = torch.randn(7, 4, dtype=torch.float64)
    out = net(x)
    assert out.shape == (7, 6)
    for g in range(3):
        h = torch.tanh(x @ net.weights[0][g] + net.biases[0][g])
        ref = h @ net.weights[1][g] + net.biases[1][g]
        torch.testing.assert_close(out[:, 2 * g : 2 * g + 2], ref)


@pytest.mark.parametrize("groups,rows", [(1, 7), (4, 7), (3, 1)])
def test_grouped_fnn_grads_keep_parameter_strides(groups, rows):
    # DDP warns (and copies) when a grad's strides differ from its bucket view.
    net = GroupedFNN(groups, [5, 6, 3], torch.tanh).double()
    net(torch.randn(rows, 5, dtype=torch.float64)).square().sum().backward()
    for p in net.parameters():
        assert p.grad.stride() == p.stride()


def _reference(model, data):
    """DeepXDE's strategy formulas from the raw branch and trunk outputs."""
    b = model.branch(data.deeponet_sensors)[0]
    t = model.act(model.trunk(data.pos))
    k, p = model.num_outputs, model.basis_dim_ref
    if model.strategy is None:
        out = (t * b).sum(-1, keepdim=True)
    elif model.strategy == "independent":
        out = (t.reshape(-1, k, p) * b.reshape(k, p)).sum(-1)
    elif model.strategy == "split_both":
        out = (t.reshape(-1, k, p // k) * b.reshape(k, p // k)).sum(-1)
    elif model.strategy == "split_branch":
        out = t @ b.reshape(k, p).T
    else:  # split_trunk
        out = (t.reshape(-1, k, p) * b).sum(-1)
    return out + model.bias


@pytest.mark.parametrize(
    "strategy", [None, "independent", "split_both", "split_branch", "split_trunk"]
)
@pytest.mark.parametrize("stacked", [False, True])
def test_strategies_match_deepxde_formulas(strategy, stacked):
    dims = [1] if strategy is None else [2]
    model = _model(dims=dims, multi_output_strategy=strategy, stacked=stacked)
    model.basis_dim_ref = 8
    with torch.no_grad():
        model.bias.normal_()
    data = _graph()
    (out,) = model(data)
    assert out.shape == (data.num_nodes, dims[0])
    torch.testing.assert_close(out, _reference(model, data))


def test_k_greater_than_one_defaults_to_independent():
    assert _model(dims=[3]).strategy == "independent"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(multi_output_strategy="split_both"), "more than one output"),
        (dict(dims=[3], multi_output_strategy="split_both"), "divisible"),
        (dict(multi_output_strategy="bogus"), "must be one of"),
        (dict(num_sensors=None), "num_sensors"),
        (dict(basis_dim=0), "basis_dim"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


def test_stacked_branch_nets_are_independent():
    model = _model(stacked=True)
    data = _graph()
    before = model.branch(data.deeponet_sensors)
    with torch.no_grad():
        model.branch.weights[-1][3].add_(1.0)
    after = model.branch(data.deeponet_sensors)
    changed = (before - after).abs().squeeze(0) > 0
    assert changed.tolist() == [i == 3 for i in range(8)]


def test_sensors_from_x_need_exactly_m_nodes():
    model = _model()
    data = _graph(n=M)
    del data.deeponet_sensors
    torch.testing.assert_close(model.sensors(data), data.x.reshape(1, M * C))
    batch = Batch.from_data_list([data, _graph(n=M, seed=1)])
    del batch.deeponet_sensors
    assert model.sensors(batch).shape == (2, M * C)
    with pytest.raises(ValueError, match="exactly deeponet_num_sensors"):
        bad = _graph(n=M + 1)
        del bad.deeponet_sensors
        model(bad)
    with pytest.raises(ValueError, match="channels"):
        bad = _graph(n=M)
        del bad.deeponet_sensors
        bad.x = bad.x[:, :1]
        model(bad)


def test_sensor_tensor_shape_is_checked():
    data = _graph()
    data.deeponet_sensors = data.deeponet_sensors[:, :-1]
    with pytest.raises(ValueError, match="deeponet_sensors must have shape"):
        _model()(data)


def test_batch_matches_single_graphs():
    model = _model(("node", "query"))
    graphs = [_graph(10, 3, 0), _graph(17, 5, 1), _graph(8, 2, 2)]
    batch = Batch.from_data_list(graphs)
    batch.query_graph = torch.repeat_interleave(
        torch.arange(3), torch.tensor([3, 5, 2])
    )
    node, query = model(batch)
    singles = [model(g) for g in graphs]
    torch.testing.assert_close(node, torch.cat([s[0] for s in singles]))
    torch.testing.assert_close(query, torch.cat([s[1] for s in singles]))


def test_batched_query_heads_need_graph_ids():
    model = _model(("node", "query"))
    batch = Batch.from_data_list([_graph(queries=2), _graph(queries=2, seed=1)])
    with pytest.raises(ValueError, match="query_cell or query_graph"):
        model(batch)


def test_query_at_node_positions_equals_node_head():
    model = _model(("node", "query"))
    with torch.no_grad():
        for net in (model.branch, model.trunk):
            for w in (*net.weights, *net.biases):
                w[1].copy_(w[0])
        model.bias[1] = model.bias[0]
    data = _graph()
    data.query_pos = data.pos.clone()
    node, query = model(data)
    torch.testing.assert_close(node, query)


def test_gradcheck():
    model = _model(("node", "query"), basis_dim=4)
    data = _graph(n=5, queries=3)

    def f(sensors, query_pos):
        data.deeponet_sensors, data.query_pos = sensors, query_pos
        return tuple(model(data))

    inputs = (
        data.deeponet_sensors.clone().requires_grad_(),
        data.query_pos.clone().requires_grad_(),
    )
    assert torch.autograd.gradcheck(f, inputs)


def test_gaussian_nll_counts_variance_columns():
    model = _model(loss_function_type="GaussianNLLLoss")
    assert model.num_outputs == 2 and model.strategy == "independent"
    means, variances = model(_graph())
    assert means[0].shape == variances[0].shape == (12, 1)
    assert (variances[0] > 0).all()


def test_freeze_conv_keeps_bias_trainable():
    model = _model(freeze_conv=True)
    assert not any(p.requires_grad for p in model.branch.parameters())
    assert not any(p.requires_grad for p in model.trunk.parameters())
    assert model.bias.requires_grad


def test_tiny_training_reduces_loss():
    model = _model(dims=[1]).train()
    graphs = [_graph(30, seed=s) for s in range(4)]
    for g in graphs:
        amp = g.deeponet_sensors.mean()
        g.y = amp * torch.sin(3 * g.pos[:, :1]) + g.pos[:, 1:]
    batch = Batch.from_data_list(graphs)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = ((model(batch)[0] - batch.y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


def test_registry_capabilities_and_create_model():
    caps = get_capabilities("DeepONet")
    assert caps.output_levels == frozenset({"node", "query"})
    assert "pos" in caps.requires
    model = create_model(
        **_args(
            output_type=["node", "query"],
            output_dim=[1, 1],
            task_weights=[1, 1],
            deeponet_multi_output_strategy="split_branch",
        )
    )
    assert isinstance(model, DeepONet) and model.strategy == "split_branch"


def test_registry_rejects_unsupported_levels_and_attention():
    with pytest.raises(ValueError, match=r"cannot predict \['face'\]"):
        create_model(**_args(output_type=["face"]))
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))


def test_checkpoint_round_trip(tmp_path):
    setup_ddp()  # save_model ends with dist.barrier()
    model = create_model(**_args()).double()
    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs(tmp_path / "deeponet")
    save_model(model, optimizer, "deeponet", path=str(tmp_path))
    torch.manual_seed(123)
    restored = create_model(**_args()).double()
    load_existing_model(restored, "deeponet", path=str(tmp_path))
    data = _graph()
    torch.testing.assert_close(restored(data)[0], model(data)[0])
