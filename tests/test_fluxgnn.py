"""FluxGNN: conservation, symmetry, batching, config wiring and rollout (FP64)."""

import math

import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.config.config_utils import configure_fluxgnn
from hydragnn.config.variable_schema import parse_variable_schema
from hydragnn.data.meshes import unit_square_mesh
from hydragnn.models.create import create_model
from hydragnn.models.fluxgnn import FluxGNN
from hydragnn.models.registry import get_capabilities
from hydragnn.models.rollout import rollout
from hydragnn.physics import balance_residual, fv_increment, net_outflow

pytestmark = pytest.mark.mpi_skip()

LAYOUT = {"state": [0, 1], "vectors": [[2, 3]], "scalars": [4]}
DT = 0.01


def _mesh(n, seed, tags=2):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(n, n + 1, dtype=torch.float64)
    c = mesh.num_cells
    mesh.cell_x = torch.cat(
        [
            torch.rand(c, 2, generator=g, dtype=torch.float64),
            torch.randn(c, 2, generator=g, dtype=torch.float64),
            torch.rand(c, 1, generator=g, dtype=torch.float64),
        ],
        dim=1,
    )
    nb = mesh.bface_owner.numel()
    mesh.bface_tag = torch.randint(0, tags, (nb,), generator=g)
    return mesh


def _model(boundary="learned", source=True, heads=("cell", "face"), **kwargs):
    torch.manual_seed(0)
    model = FluxGNN(
        5,
        8,
        [2] * len(heads),
        list(heads),
        3,
        layout=LAYOUT,
        dt=DT,
        num_bface_tags=2,
        boundary=boundary,
        source=source,
        **kwargs,
    ).double()
    # Leave the zero-initialised regime so every learned term is active.
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(std=0.3)
            else:
                param.uniform_(-0.2, 0.2)
    return model.eval()


def _rotate(mesh, angle, reflect=False):
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    if reflect:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    shift = torch.tensor([0.3, -1.7], dtype=torch.float64)
    out = mesh.clone()
    for key in ("pos", "cell_centroid"):
        out[key] = mesh[key] @ R.T + shift
    for key in ("face_normal", "bface_normal"):
        out[key] = mesh[key] @ R.T
    x = mesh.cell_x.clone()
    x[:, 2:4] = mesh.cell_x[:, 2:4] @ R.T
    out.cell_x = x
    return out


# -- conservation -------------------------------------------------------------
def test_interior_fluxes_cancel():
    mesh = _mesh(4, 0)
    flux = torch.randn(mesh.num_faces, 3, dtype=torch.float64)
    total = (net_outflow(mesh, flux)).sum(0)
    torch.testing.assert_close(total, torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize("boundary,source", [("learned", True), ("closed", False)])
def test_global_balance(boundary, source):
    batch = Batch.from_data_list([_mesh(3, 0), _mesh(5, 1)])
    model = _model(boundary=boundary, source=source)
    terms = model.fluxes(batch)
    u_old = batch.cell_x[:, :2]
    assert (terms["bface_flux"] is None) == (boundary == "closed")
    assert (terms["source"] is None) == (not source)
    residual, scale = balance_residual(
        batch, u_old, terms["u_new"], terms["bface_flux"], terms["source"], DT
    )
    assert residual.shape == (2, 2)
    assert float((residual.abs() / scale).max()) < 1e-13
    # The learned terms really change the state.
    assert float((terms["u_new"] - u_old).abs().max()) > 1e-6


def test_closed_without_source_conserves_total():
    mesh = _mesh(4, 2)
    model = _model(boundary="closed", source=False)
    u_new = model(mesh)[0]
    V = mesh.cell_volume
    torch.testing.assert_close(
        (V * u_new).sum(0), (V * mesh.cell_x[:, :2]).sum(0), rtol=1e-13, atol=0
    )


def test_face_head_matches_fv_update():
    mesh = _mesh(3, 3)
    model = _model()
    u_new, face = model(mesh)
    terms = model.fluxes(mesh)
    torch.testing.assert_close(face, terms["face_flux"])
    # One step of the averaged terms reproduces the substepped update exactly
    # (the update is linear in the per-substep terms).
    expected = mesh.cell_x[:, :2] + fv_increment(
        mesh, terms["face_flux"], terms["bface_flux"], terms["source"], DT
    )
    torch.testing.assert_close(u_new, expected, rtol=1e-12, atol=1e-14)


# -- symmetry -----------------------------------------------------------------
def test_face_orientation_flip_antisymmetric():
    mesh = _mesh(3, 4)
    flipped = mesh.clone()
    flipped.face_index = mesh.face_index.flip(0)
    flipped.face_normal = -mesh.face_normal
    model = _model()
    u, face = model(mesh)
    u_f, face_f = model(flipped)
    torch.testing.assert_close(u_f, u, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(face_f, -face, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("reflect", [False, True])
def test_rotation_reflection_translation_invariance(reflect):
    mesh = _mesh(4, 5)
    model = _model()
    u, face = model(mesh)
    u_r, face_r = model(_rotate(mesh, 0.7, reflect))
    torch.testing.assert_close(u_r, u, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(face_r, face, rtol=1e-11, atol=1e-13)


def test_cell_permutation_equivariance():
    mesh = _mesh(3, 6)
    perm = torch.randperm(mesh.num_cells, generator=torch.Generator().manual_seed(0))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    permuted = mesh.clone()
    for key in ("cell_x", "cell_volume", "cell_centroid", "cell_size"):
        permuted[key] = mesh[key][perm]
    # Rebuild cell_conn in the permuted order.
    offsets = torch.cumsum(mesh.cell_size, 0) - mesh.cell_size
    permuted.cell_conn = torch.cat(
        [mesh.cell_conn[offsets[i] : offsets[i] + mesh.cell_size[i]] for i in perm]
    )
    permuted.face_index = inv[mesh.face_index]
    permuted.bface_owner = inv[mesh.bface_owner]
    model = _model()
    u, face = model(mesh)
    u_p, face_p = model(permuted)
    torch.testing.assert_close(u_p, u[perm], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(face_p, face, rtol=1e-12, atol=1e-14)


def test_batched_equals_single():
    meshes = [_mesh(3, 7), _mesh(4, 8)]
    model = _model()
    u, face = model(Batch.from_data_list(meshes))
    u_single = torch.cat([model(m)[0] for m in meshes])
    face_single = torch.cat([model(m)[1] for m in meshes])
    torch.testing.assert_close(u, u_single, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(face, face_single, rtol=1e-12, atol=1e-14)


def test_untrained_model_is_upwind_diffusion_scheme():
    """Zero-initialised corrections: pure upwind + two-point diffusion."""
    torch.manual_seed(0)
    model = FluxGNN(5, 2, [2], ["cell"], 1, layout=LAYOUT, dt=DT, num_bface_tags=2)
    model = model.double().eval()
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2, dtype=torch.float64))
    mesh = _mesh(3, 9)
    o, n = mesh.face_index
    u, v = mesh.cell_x[:, :2], mesh.cell_x[:, 2:4]
    un = 0.5 * ((v[o] + v[n]) * mesh.face_normal).sum(1, keepdim=True)
    dist = (mesh.cell_centroid[n] - mesh.cell_centroid[o]).norm(dim=1, keepdim=True)
    kappa = torch.nn.functional.softplus(model.diffusivity)
    flux = un * torch.where(un > 0, u[o], u[n]) - kappa * (u[n] - u[o]) / dist
    bvn = (v[mesh.bface_owner] * mesh.bface_normal).sum(1, keepdim=True)
    bflux = bvn.clamp_min(0.0) * u[mesh.bface_owner]
    expected = u + fv_increment(mesh, flux, bflux, None, DT)
    torch.testing.assert_close(model(mesh)[0], expected, rtol=1e-12, atol=1e-14)


# -- construction ---------------------------------------------------------------
def test_freeze_conv_trains_encoder_only():
    model = FluxGNN(
        5, 8, [2], ["cell"], 2, layout=LAYOUT, source=True, freeze_conv=True
    )
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"encoder.weight"}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(layout={"state": [0], "vectors": [], "scalars": []}), "partition"),
        (dict(layout={"state": [0, 1], "vectors": [[2]], "scalars": [3, 4]}), "dim 2"),
        (dict(hidden_dim=1), "hidden_dim"),
        (dict(output_dim=[1]), "state dimension"),
        (dict(output_dim=[2, 2], output_type=["cell", "cell"]), "at most one"),
        (dict(boundary="open"), "fluxgnn_boundary"),
        (dict(loss_function_type="GaussianNLLLoss"), "GaussianNLL"),
    ],
)
def test_constructor_errors(kwargs, match):
    args = dict(
        input_dim=5,
        hidden_dim=8,
        output_dim=[2],
        output_type=["cell"],
        num_steps=2,
        layout=LAYOUT,
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FluxGNN(**args)


def test_registry_and_create_model():
    caps = get_capabilities("FluxGNN")
    assert caps.output_levels == frozenset({"cell", "face"})
    assert {"cell_x", "face_index", "cell_volume"} <= caps.requires
    model = create_model(**_create_args(), fluxgnn_dt=0.5, fluxgnn_source=True)
    assert isinstance(model, FluxGNN)
    assert model.dt == 0.5 and model.source_mlp is not None
    with pytest.raises(ValueError, match="cannot predict"):
        create_model(**{**_create_args(), "output_type": ["node"]})


def _create_args():
    return dict(
        mpnn_type="FluxGNN",
        input_dim=5,
        hidden_dim=8,
        output_dim=[2],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["cell"],
        output_heads={},
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        use_gpu=False,
        fluxgnn_layout=LAYOUT,
    )


# -- config wiring ----------------------------------------------------------------
def _schema(normalize_u="none"):
    return parse_variable_schema(
        {
            "graph_type": "homogeneous",
            "spatial_dim": 2,
            "inputs": [
                {"name": "pos", "level": "node", "dim": 2, "role": "position"},
                {"name": "s", "level": "cell", "dim": 1, "normalize": "scale_only"},
                {"name": "u", "level": "cell", "dim": 2, "normalize": normalize_u},
                {"name": "v", "level": "cell", "dim": 2},
            ],
            "outputs": [{"name": "u_next", "level": "cell", "dim": 2}],
        }
    )


def test_configure_fluxgnn_layout():
    arch = {"fluxgnn_vector_inputs": ["v"], "fluxgnn_scalar_inputs": ["s"]}
    configure_fluxgnn(arch, _schema())
    assert arch["input_dim"] == 5
    assert arch["fluxgnn_spatial_dim"] == 2
    assert arch["fluxgnn_layout"] == {
        "state": [1, 2],
        "vectors": [[3, 4]],
        "scalars": [0],
    }


@pytest.mark.parametrize(
    "arch,normalize_u,match",
    [
        ({"fluxgnn_vector_inputs": ["w"]}, "none", "not cell inputs"),
        (
            {"fluxgnn_vector_inputs": ["v"], "fluxgnn_scalar_inputs": ["v"]},
            "none",
            "disjoint",
        ),
        ({"fluxgnn_vector_inputs": ["v"]}, "standard", "normalize 'none'"),
    ],
)
def test_configure_fluxgnn_errors(arch, normalize_u, match):
    with pytest.raises(ValueError, match=match):
        configure_fluxgnn(arch, _schema(normalize_u))


# -- rollout ------------------------------------------------------------------------
def test_rollout_feeds_state_back():
    mesh = _mesh(3, 10)
    model = _model(heads=("cell",))
    states = rollout(model, mesh, 3, model.state_cols, level="cell")
    assert len(states) == 3
    data = mesh.clone()
    for state in states:
        expected = model(data)[0]
        torch.testing.assert_close(state, expected)
        x = data.cell_x.clone()
        x[:, :2] = expected
        data.cell_x = x
    torch.testing.assert_close(mesh.cell_x, _mesh(3, 10).cell_x)  # input untouched


def test_bface_tag_out_of_range():
    mesh = _mesh(3, 12, tags=3)
    mesh.bface_tag[0] = 2
    with pytest.raises(ValueError, match="bface_tag"):
        _model()(mesh)


def test_rollout_errors():
    mesh = _mesh(3, 11)
    model = _model(heads=("cell",))
    with pytest.raises(ValueError, match="level must be one of"):
        rollout(model, mesh, 1, [0, 1], level="face")
    with pytest.raises(ValueError, match="shape"):
        rollout(model, mesh, 1, [0], level="cell")
