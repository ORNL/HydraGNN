"""Contracts for CompositeNeuralOperator (typed pipeline validation, blocks)."""

import copy
import os

import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import (
    attach_graph_spectral_basis,
    build_rigno_graph,
    locate_queries,
    unit_square_mesh,
)
from hydragnn.distributed import setup_ddp
from hydragnn.models.composite import CompositeNeuralOperator, resolve_pipeline
from hydragnn.models.create import create_model
from hydragnn.models.registry import get_capabilities
from hydragnn.models.utils import (
    checkpoint_metadata,
    load_checkpoint_metadata,
    load_existing_model,
    save_model,
)

pytestmark = pytest.mark.mpi_skip()

MGN_GSNO = dict(
    composite_encoder={"type": "NodeEncoder"},
    composite_processors=[
        {"type": "LocalMessagePassing", "layers": 2},
        {"type": "GraphSpectral", "layers": 1, "modes": 4, "skip": True},
    ],
)
REGIONAL = dict(
    composite_encoder={"type": "RegionalGraphEncoder"},
    composite_processors=[
        {"type": "RegionalMessagePassing", "layers": 2},
        {"type": "RegionToNode"},
        {"type": "LocalMessagePassing", "layers": 1, "skip": True},
    ],
)


def _mesh(nx=4, ny=4, seed=0, queries=0):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(nx, ny, dtype=torch.float64)
    mesh.x = torch.randn(mesh.num_nodes, 2, generator=g, dtype=torch.float64)
    if queries:
        locate_queries(mesh, torch.rand(queries, 2, generator=g, dtype=torch.float64))
    mesh = build_rigno_graph(mesh, generator=g)
    attach_graph_spectral_basis(mesh, 6)
    return mesh


def _args(pipeline=MGN_GSNO, **overrides):
    args = dict(
        mpnn_type="CompositeNeuralOperator",
        input_dim=2,
        hidden_dim=8,
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
        **pipeline,
    )
    args.update(overrides)
    return args


def _model(pipeline=MGN_GSNO, levels=("node",), **overrides):
    torch.manual_seed(0)
    levels = list(levels)
    args = _args(
        pipeline,
        output_type=levels,
        output_dim=[1] * len(levels),
        task_weights=[1.0] * len(levels),
    )
    args.update(overrides)
    return create_model(**args).double().eval()


def _resolve(encoder, processors, decoders=None, levels=("node",), width=8, dim=2):
    return resolve_pipeline(encoder, processors, decoders, list(levels), width, dim)


def test_registry_capabilities():
    caps = get_capabilities("CompositeNeuralOperator")
    assert caps.output_levels == frozenset({"node", "cell", "face", "query"})


@pytest.mark.parametrize("pipeline", [MGN_GSNO, REGIONAL], ids=["mgn_gsno", "rigno"])
def test_pipelines_train(pipeline):
    model = _model(pipeline).train()
    mesh = _mesh()
    target = torch.sin(3 * mesh.pos[:, :1]) * mesh.pos[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = ((model(mesh)[0] - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("pipeline", [MGN_GSNO, REGIONAL], ids=["mgn_gsno", "rigno"])
def test_batched_equals_single(pipeline):
    levels = ("node", "face", "cell", "query")
    model = _model(pipeline, levels)
    meshes = [_mesh(3, 4, 0, queries=3), _mesh(4, 3, 1, queries=5)]
    batched = model(Batch.from_data_list(meshes))
    singles = [model(m) for m in meshes]
    for b, *s in zip(batched, *singles):
        torch.testing.assert_close(b, torch.cat(s))


def test_conservative_face_decoder_is_antisymmetric():
    model = _model(REGIONAL, ("node", "face"))
    mesh = _mesh(seed=2)
    flipped = copy.copy(mesh)
    flipped.face_index = mesh.face_index.flip(0)
    flipped.face_nodes = mesh.face_nodes.flip(0)
    flipped.face_normal = -mesh.face_normal
    face = model(mesh)[1]
    torch.testing.assert_close(model(flipped)[1], -face, rtol=0, atol=1e-12)
    plain = _model(
        REGIONAL, ("node", "face"), composite_decoders={"face": "FluxDecoder"}
    )
    assert not torch.allclose(plain(flipped)[1], -plain(mesh)[1])


def test_skip_is_a_residual():
    model = _model()
    mesh = _mesh()
    state = model.encode(mesh)
    lmp, gsno = model.components[1], model.components[2]
    node, _ = lmp(state["node"], None, mesh)
    expected = gsno(node, None, mesh)[0] + node
    torch.testing.assert_close(model.process(state, mesh)["node"], expected)


def test_conv_checkpointing_matches():
    model = _model(REGIONAL).train()
    ckpt = _model(REGIONAL, conv_checkpointing=True).train()
    ckpt.load_state_dict(model.state_dict())
    mesh = _mesh()
    torch.testing.assert_close(model(mesh)[0], ckpt(mesh)[0])


def test_missing_data_attributes_are_reported():
    model = _model(REGIONAL)
    mesh = _mesh()
    del mesh.rigno_r2p_index
    with pytest.raises(
        ValueError, match=r"needs data attributes \['rigno_r2p_index'\]"
    ):
        model(mesh)


def test_projection_changes_width():
    stages, _, width, _ = _resolve(
        {"type": "NodeEncoder"},
        [
            {"type": "LocalMessagePassing", "layers": 1},
            {"type": "Projection", "hidden_dim": 12},
            {"type": "LocalMessagePassing", "layers": 1, "hidden_dim": 12},
        ],
    )
    assert width == 12 and [w for _, _, w in stages] == [8, 8, 8, 12]
    model = _model(
        composite_processors=[
            {"type": "Projection", "hidden_dim": 12},
            {"type": "LocalMessagePassing", "layers": 1},
        ]
    )
    assert model(_mesh())[0].shape == (25, 1)


@pytest.mark.parametrize(
    "encoder,processors,kwargs,match",
    [
        (
            {"type": "RegionalGraphEncoder"},
            [{"type": "RegionalMessagePassing"}],
            {},
            "add a RegionToNode adapter",
        ),
        (
            {"type": "NodeEncoder"},
            [{"type": "RegionalMessagePassing"}],
            {},
            "reads region latents but the previous component produces node",
        ),
        ({"type": "NodeEncoder"}, [{"type": "Transformer"}], {}, "unknown component"),
        (
            {"type": "LocalMessagePassing"},
            [],
            {},
            "must start with exactly one encoder",
        ),
        (
            {"type": "NodeEncoder"},
            [{"type": "NodeEncoder"}],
            {},
            "must start with exactly one encoder",
        ),
        (
            {"type": "RegionalGraphEncoder"},
            [{"type": "RegionToNode"}],
            dict(dim=3),
            "supports spatial_dim",
        ),
        (
            {"type": "RegionalGraphEncoder"},
            [{"type": "RegionToNode", "skip": True}],
            {},
            "skip is only allowed on processors",
        ),
        (
            {"type": "NodeEncoder"},
            [{"type": "LocalMessagePassing", "hidden_dim": 16}],
            {},
            "insert a Projection",
        ),
        (
            {"type": "NodeEncoder"},
            [{"type": "LocalMessagePassing", "modes": 4}],
            {},
            r"does not accept \['modes'\]",
        ),
        ({"type": "NodeEncoder"}, [{"type": "Projection"}], {}, "needs hidden_dim"),
        (
            {"type": "NodeEncoder"},
            [],
            dict(decoders={"face": "CellDecoder"}, levels=("face",)),
            "cannot decode face",
        ),
        (
            {"type": "NodeEncoder"},
            [],
            dict(decoders={"edge": "NodeDecoder"}),
            "unknown levels",
        ),
        ({"type": "NodeEncoder"}, [], dict(levels=("query",), dim=3), "spatial_dim 2"),
    ],
)
def test_pipeline_rejections(encoder, processors, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _resolve(encoder, processors, **kwargs)


def test_constructor_rejections():
    with pytest.raises(ValueError, match="does not support activation 'gelu'"):
        _model(activation_function="gelu")
    with pytest.raises(ValueError, match="composite_spatial_dim"):
        _model(composite_spatial_dim=1)
    with pytest.raises(ValueError, match="global attention"):
        create_model(**_args(global_attn_engine="GPS"))


def test_defaults_build_local_message_passing():
    model = CompositeNeuralOperator(2, 8, [1], ["node"], 2).double()
    assert [type(c).__name__ for c in model.components] == [
        "NodeEncoder",
        "LocalMessagePassing",
    ]
    assert model(_mesh())[0].shape == (25, 1)


def test_checkpoint_round_trip_with_config(tmp_path):
    setup_ddp()  # save_model ends with dist.barrier()
    model = _model(REGIONAL, ("node", "face"))
    arch = {k: v for k, v in REGIONAL.items()}
    config = {"NeuralNetwork": {"Architecture": arch}, "Variables": {}}
    os.makedirs(tmp_path / "composite")
    save_model(
        model,
        torch.optim.Adam(model.parameters()),
        "composite",
        path=str(tmp_path),
        metadata=checkpoint_metadata(config),
    )
    loaded, _, _ = load_checkpoint_metadata("composite", path=str(tmp_path))
    restored_arch = loaded["NeuralNetwork"]["Architecture"]
    assert restored_arch == arch
    torch.manual_seed(321)
    restored = create_model(
        **_args(
            restored_arch,
            output_type=["node", "face"],
            output_dim=[1, 1],
            task_weights=[1.0, 1.0],
        )
    ).double()
    load_existing_model(restored, "composite", path=str(tmp_path))
    mesh = _mesh()
    for a, b in zip(model(mesh), restored.eval()(mesh)):
        torch.testing.assert_close(a, b)
