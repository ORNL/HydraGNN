"""Contracts for the FNO port (spectral convolution, padding, batching)."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from hydragnn.models.create import create_model
from hydragnn.models.fno import FNO, DomainPadding, SpectralConv
from hydragnn.models.registry import get_capabilities

pytestmark = pytest.mark.mpi_skip()


def _grid(shape, channels=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = int(torch.tensor(shape).prod())
    x = torch.randn(n, channels, generator=g, dtype=torch.float64)
    return Data(x=x, grid_shape=torch.tensor([list(shape)]), num_nodes=n)


def _model(n_modes=(4, 4), **kwargs):
    torch.manual_seed(0)
    return FNO(2, 8, [1], ["node"], 2, n_modes=n_modes, **kwargs).double().eval()


def _args(**overrides):
    args = dict(
        mpnn_type="FNO",
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
        fno_n_modes=[4, 4],
    )
    args.update(overrides)
    return args


# -- spectral convolution -------------------------------------------------------
@pytest.mark.parametrize("sizes", [(16,), (15,), (8, 10), (9, 7), (6, 5, 8)])
def test_spectral_conv_matches_dense_truncated_multiplier(sizes):
    """Retaining all modes of a real input with a constant multiplier w is
    multiplication by w (up to the dropped imaginary DC/Nyquist parts)."""
    torch.manual_seed(0)
    n_modes = [2 * s for s in sizes]  # above the resolution limit: truncation
    conv = SpectralConv(1, 1, n_modes, bias=False).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[..., 0] = 1.7
    x = torch.randn(3, 1, *sizes, dtype=torch.float64)
    torch.testing.assert_close(conv(x), 1.7 * x)


def test_spectral_conv_low_pass_removes_high_modes():
    torch.manual_seed(0)
    conv = SpectralConv(1, 1, [4], bias=False).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[..., 0] = 1.0
    t = torch.arange(32, dtype=torch.float64) / 32
    low = torch.cos(2 * torch.pi * t)
    x = (low + torch.cos(2 * torch.pi * 9 * t)).view(1, 1, -1)
    torch.testing.assert_close(conv(x)[0, 0], low)


def test_spectral_weights_survive_double():
    model = FNO(2, 8, [1], ["node"], 1, n_modes=(4, 4))
    before = model.fno_blocks.convs[0].complex_weight().clone()
    model = model.double()
    after = model.fno_blocks.convs[0].complex_weight()
    assert after.dtype == torch.complex128
    torch.testing.assert_close(after, before.to(torch.complex128))


def test_domain_padding_round_trip():
    x = torch.randn(2, 3, 10, 12)
    padded, pads = DomainPadding([0.25, 0.5]).pad(x)
    assert padded.shape == (2, 3, 10 + 2 * 2, 12 + 2 * 6)
    torch.testing.assert_close(DomainPadding.unpad(padded, pads), x)


# -- model --------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_modes,shape", [((4,), (12,)), ((4, 4), (8, 6)), ((2, 2, 2), (4, 5, 3))]
)
def test_forward_shapes_1d_2d_3d(n_modes, shape):
    data = _grid(shape)
    out = _model(n_modes, domain_padding=0.25)(data)
    assert out[0].shape == (data.num_nodes, 1)


def test_batch_with_mixed_resolutions_matches_single():
    model = _model()
    graphs = [_grid((8, 8), seed=0), _grid((6, 10), seed=1), _grid((8, 8), seed=2)]
    batched = model(Batch.from_data_list(graphs))[0]
    single = torch.cat([model(g)[0] for g in graphs])
    torch.testing.assert_close(batched, single)


def test_grid_shape_validation():
    data = _grid((4, 4))
    data.grid_shape = torch.tensor([[4, 5]])
    with pytest.raises(ValueError, match="does not match"):
        _model()(data)
    del data.grid_shape
    with pytest.raises(ValueError, match="grid_shape"):
        _model()(data)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(n_modes=(2, 2, 2, 2)), "1, 2 or 3"),
        (dict(positional_embedding="sin"), "positional_embedding"),
        (dict(domain_padding=[0.1]), "one entry per dim"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FNO(2, 8, [1], ["node"], 1, **{"n_modes": (4, 4), **kwargs})


def test_gradcheck_fp64():
    torch.manual_seed(0)
    model = FNO(1, 4, [1], ["node"], 1, n_modes=(3, 3)).double()
    x = torch.randn(1, 1, 5, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: model._core(v), (x,))


def test_tiny_training_reduces_loss():
    torch.manual_seed(0)
    model = FNO(2, 8, [1], ["node"], 2, n_modes=(4, 4)).double()
    data = _grid((8, 8))
    target = data.x[:, :1] * 2 - data.x[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = ((model(data)[0] - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.2 * losses[0]


# -- registry ------------------------------------------------------------------
def test_registry_capabilities_and_create_model():
    caps = get_capabilities("FNO")
    assert caps.output_levels == frozenset({"node"})
    assert "grid_shape" in caps.requires
    model = create_model(**_args(fno_domain_padding=0.125))
    assert isinstance(model, FNO) and model.domain_padding is not None


def test_registry_rejects_global_attention():
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))


def test_registry_rejects_query_heads():
    with pytest.raises(ValueError):
        create_model(**_args(output_type=["query"]))
