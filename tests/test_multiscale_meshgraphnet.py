"""Contracts for MultiscaleMeshGraphNet and the hierarchy node-transfer graphs."""

import copy

import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import (
    build_hierarchy,
    prolong_nodes,
    restrict_nodes,
    unit_square_mesh,
)
from hydragnn.models.create import create_model
from hydragnn.models.multiscale_meshgraphnet import default_schedule, parse_schedule
from hydragnn.models.registry import get_capabilities

pytestmark = pytest.mark.mpi_skip()


def _hierarchy(n, seed=0, dtype=torch.float64, coarse=None):
    h = build_hierarchy(
        unit_square_mesh(n, n, dtype=dtype),
        unit_square_mesh(coarse or n // 2, coarse or n // 2, dtype=dtype),
        node_transfer=True,
    )
    g = torch.Generator().manual_seed(seed)
    h.x = torch.randn(h.num_nodes, 3, generator=g, dtype=dtype)
    return h


def _args(**overrides):
    args = dict(
        mpnn_type="MultiscaleMeshGraphNet",
        input_dim=3,
        hidden_dim=16,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads={},
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=7,
        use_gpu=False,
        mgn_spatial_dim=2,
        msmgn_schedule="1H 3L 1H",
    )
    args.update(overrides)
    return args


def _model(**overrides):
    return create_model(**_args(**overrides)).double().eval()


# -- schedule ----------------------------------------------------------------
def test_parse_schedule_inserts_transfers():
    assert parse_schedule("1H 2L 1H") == ["H", "D", "L", "L", "U", "H"]
    # Paper processors: 15 and 25 message-passing steps counting D and U.
    assert len(parse_schedule("1H 11L 1H")) == 15
    assert len(parse_schedule("3H 6L 3H 6L 3H")) == 25
    assert default_schedule(15) == "1H 11L 1H"


@pytest.mark.parametrize("bad", ["", "2L", "1H 2L", "1H x 1H", "0H 1L 1H"])
def test_parse_schedule_rejects(bad):
    with pytest.raises(ValueError):
        parse_schedule(bad)


def test_schedule_length_must_match_num_conv_layers():
    with pytest.raises(ValueError, match="num_conv_layers is 8"):
        create_model(**_args(num_conv_layers=8))


# -- hierarchy node transfer -------------------------------------------------
def test_node_transfer_indices():
    h = _hierarchy(4)
    assert h.down_index.shape == (2, 3 * h.num_nodes)
    assert h.up_index.shape == (2, 3 * h.num_coarse_nodes)
    assert torch.equal(h.down_index[0], torch.arange(h.num_nodes).repeat_interleave(3))
    # Every corner lies on the coarse/fine mesh and weights are a partition of unity.
    assert int(h.down_index[1].max()) < h.num_coarse_nodes
    assert int(h.up_index[1].max()) < h.num_nodes
    for index, weight, n in (
        (h.down_index, h.down_weight, h.num_nodes),
        (h.up_index, h.up_weight, h.num_coarse_nodes),
    ):
        assert bool((weight >= -1e-12).all())
        total = torch.zeros(n, dtype=weight.dtype).index_add_(0, index[0], weight)
        torch.testing.assert_close(total, torch.ones_like(total))


def test_node_transfer_preserves_constants_and_linears():
    h = _hierarchy(4)
    const = torch.full((h.num_coarse_nodes, 2), 2.5, dtype=torch.float64)
    torch.testing.assert_close(
        prolong_nodes(h, const), torch.full((h.num_nodes, 2), 2.5, dtype=torch.float64)
    )

    def linear(pos):
        return (1.0 + 2.0 * pos[:, :1] - 3.0 * pos[:, 1:2]).to(torch.float64)

    torch.testing.assert_close(prolong_nodes(h, linear(h.coarse_pos)), linear(h.pos))
    torch.testing.assert_close(restrict_nodes(h, linear(h.pos)), linear(h.coarse_pos))


def test_node_transfer_batching():
    levels = [_hierarchy(2), _hierarchy(4, seed=1)]
    batch = Batch.from_data_list(levels)
    n0, c0 = levels[0].num_nodes, levels[0].num_coarse_nodes
    d0, u0 = levels[0].down_index.shape[1], levels[0].up_index.shape[1]
    offset_down = torch.tensor([[n0], [c0]])
    offset_up = torch.tensor([[c0], [n0]])
    assert torch.equal(batch.down_index[:, d0:], levels[1].down_index + offset_down)
    assert torch.equal(batch.up_index[:, u0:], levels[1].up_index + offset_up)
    singles = torch.cat([prolong_nodes(h, h.coarse_pos) for h in levels])
    torch.testing.assert_close(prolong_nodes(batch, batch.coarse_pos), singles)
    for orig, back in zip(levels, batch.to_data_list()):
        assert torch.equal(back.down_index, orig.down_index)
        assert torch.equal(back.up_index, orig.up_index)


# -- model -------------------------------------------------------------------
def test_capabilities():
    caps = get_capabilities("MultiscaleMeshGraphNet")
    assert caps.output_levels == frozenset({"node", "face", "cell", "query"})
    assert {"coarse_pos", "down_index", "up_index"} <= caps.requires
    assert "2210.00612" in caps.reference


def test_processor_follows_schedule():
    model = _model()
    assert model.ops == ["H", "D", "L", "L", "L", "U", "H"]
    assert len(model.processor) == 7
    # Every block has its own parameters.
    ids = {id(p) for block in model.processor for p in block.parameters()}
    assert len(ids) == sum(len(list(b.parameters())) for b in model.processor)


def test_gradients_reach_every_parameter():
    model = _model().train()
    h = _hierarchy(4)
    model(h)[0].pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if not name.startswith("decoders"):
            assert float(p.grad.abs().sum()) > 0, name


def test_batched_matches_single():
    model = _model()
    levels = [_hierarchy(4), _hierarchy(6, seed=1, coarse=3)]
    batch = Batch.from_data_list(levels)
    with torch.no_grad():
        out = model(batch)[0]
        singles = torch.cat([model(h)[0] for h in levels])
    torch.testing.assert_close(out, singles)


def test_translation_invariance():
    model = _model()
    h = _hierarchy(4)
    shifted = copy.copy(h)
    shift = torch.tensor([3.0, -2.0], dtype=torch.float64)
    shifted.pos = h.pos + shift
    shifted.coarse_pos = h.coarse_pos + shift
    with torch.no_grad():
        torch.testing.assert_close(model(shifted)[0], model(h)[0])


def test_coarse_level_extends_receptive_field():
    """With 6 message-passing steps a flat MGN cannot see across an 8x8 mesh;
    the coarse level carries information corner to corner."""
    h = _hierarchy(8, coarse=1)
    far = int(((h.pos - 1.0) ** 2).sum(dim=1).argmin())
    near = int((h.pos**2).sum(dim=1).argmin())
    perturbed = copy.copy(h)
    perturbed.x = h.x.clone()
    perturbed.x[near] += 1.0
    flat = create_model(
        **_args(mpnn_type="MeshGraphNet", num_conv_layers=6, msmgn_schedule=None)
    ).double()
    multi = _model(num_conv_layers=6, msmgn_schedule="1H 2L 1H")
    with torch.no_grad():
        assert torch.equal(flat(perturbed)[0][far], flat(h)[0][far])
        assert not torch.allclose(multi(perturbed)[0][far], multi(h)[0][far])


def test_mesh_heads_and_conservative_faces():
    model = _model(
        output_dim=[1, 1, 1],
        output_type=["node", "face", "cell"],
        task_weights=[1.0, 1.0, 1.0],
    )
    h = _hierarchy(4)
    flipped = copy.copy(h)
    flipped.face_index = h.face_index.flip(0)
    flipped.face_nodes = h.face_nodes.flip(0)
    flipped.face_normal = -h.face_normal
    with torch.no_grad():
        node, face, cell = model(h)
        _, face_flipped, _ = model(flipped)
    assert face.shape == (h.face_index.shape[1], 1)
    assert cell.shape == (h.num_cells, 1)
    torch.testing.assert_close(face_flipped, -face)


def test_freeze_conv_keeps_only_decoders_trainable():
    model = create_model(**_args(freeze_conv=True))
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"decoders"}


def test_conv_checkpointing_matches():
    model = _model().train()
    h = _hierarchy(4)
    ref = model(h)[0]
    model.enable_conv_checkpointing()
    torch.testing.assert_close(model(h)[0], ref)


def test_rejects_3d():
    with pytest.raises(ValueError, match="mgn_spatial_dim=2"):
        create_model(**_args(mgn_spatial_dim=3))
