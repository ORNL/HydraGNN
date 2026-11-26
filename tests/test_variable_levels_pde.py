"""Face/cell output levels and ``Variables.spatial_dim`` in the named schema."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from hydragnn.config.variable_schema import (
    parse_variable_schema,
    prepare_data_from_schema,
)
from hydragnn.data.meshes import unit_square_mesh

pytestmark = pytest.mark.mpi_skip()


def _variables(outputs, spatial_dim=2, inputs=None):
    variables = {
        "graph_type": "homogeneous",
        "inputs": inputs
        or [
            {"name": "pos", "level": "node", "dim": spatial_dim, "role": "position"},
            {"name": "f", "level": "node", "dim": 1},
        ],
        "outputs": outputs,
    }
    if spatial_dim != 3:
        variables["spatial_dim"] = spatial_dim
    return variables


def _mesh():
    mesh = unit_square_mesh(3, 2, dtype=torch.float32)
    mesh.f = torch.randn(mesh.num_nodes, 1)
    mesh.u = torch.randn(mesh.num_nodes, 1)
    mesh.q = torch.randn(mesh.num_faces, 2)
    mesh.c = torch.randn(mesh.num_cells, 1)
    return mesh


def test_default_spatial_dim_keeps_atomistic_rule():
    schema = parse_variable_schema(
        _variables([{"name": "u", "level": "node", "dim": 1}], spatial_dim=3)
    )
    assert schema.spatial_dim == 3
    with pytest.raises(ValueError, match="level 'node', and dim 3"):
        parse_variable_schema(
            {
                "graph_type": "homogeneous",
                "inputs": [
                    {"name": "pos", "level": "node", "dim": 2, "role": "position"},
                    {"name": "f", "level": "node", "dim": 1},
                ],
                "outputs": [],
            }
        )


@pytest.mark.parametrize("spatial_dim", [1, 2])
def test_spatial_dim_relaxes_position_dim(spatial_dim):
    schema = parse_variable_schema(_variables([], spatial_dim=spatial_dim))
    assert schema.spatial_dim == spatial_dim
    with pytest.raises(ValueError, match=f"dim {spatial_dim}"):
        parse_variable_schema(
            _variables(
                [],
                spatial_dim=spatial_dim,
                inputs=[
                    {"name": "pos", "level": "node", "dim": 3, "role": "position"},
                    {"name": "f", "level": "node", "dim": 1},
                ],
            )
        )


@pytest.mark.parametrize("value", [0, 4, True, "2"])
def test_invalid_spatial_dim(value):
    variables = _variables([])
    variables["spatial_dim"] = value
    with pytest.raises(ValueError, match="spatial_dim must be one of"):
        parse_variable_schema(variables)


@pytest.mark.parametrize("level", ["face", "cell"])
def test_mesh_levels_rejected_as_inputs(level):
    inputs = [
        {"name": "pos", "level": "node", "dim": 2, "role": "position"},
        {"name": "f", "level": "node", "dim": 1},
        {"name": "g", "level": level, "dim": 1},
    ]
    with pytest.raises(ValueError, match="supported only for outputs"):
        parse_variable_schema(_variables([], inputs=inputs))


def test_face_and_cell_outputs_compile_to_y():
    schema = parse_variable_schema(
        _variables(
            [
                {"name": "u", "level": "node", "dim": 1},
                {"name": "q", "level": "face", "dim": 2},
                {"name": "c", "level": "cell", "dim": 1},
            ]
        )
    )
    mesh = _mesh()
    data = prepare_data_from_schema(mesh.clone(), schema)
    n, f, c = mesh.num_nodes, mesh.num_faces, mesh.num_cells
    assert data.y_loc.tolist() == [[0, n, n + 2 * f, n + 2 * f + c]]
    torch.testing.assert_close(data.y[n : n + 2 * f], mesh.q.reshape(-1, 1))
    torch.testing.assert_close(data.face_output, mesh.q)
    torch.testing.assert_close(data.cell_output, mesh.c)
    assert data.pos.shape == (n, 2)

    batch = Batch.from_data_list(
        [data, prepare_data_from_schema(_mesh().clone(), schema)]
    )
    assert batch.y_loc.shape == (2, 4)
    assert batch.face_output.shape == (2 * f, 2)


def test_face_output_shape_is_validated():
    schema = parse_variable_schema(
        _variables([{"name": "q", "level": "face", "dim": 2}])
    )
    mesh = _mesh()
    mesh.q = torch.randn(mesh.num_faces + 1, 2)
    with pytest.raises(ValueError, match=r"must have shape \("):
        prepare_data_from_schema(mesh, schema)


@pytest.mark.parametrize("level,missing", [("face", "face_index"), ("cell", "cell_size")])
def test_mesh_outputs_require_mesh_data(level, missing):
    schema = parse_variable_schema(_variables([{"name": "t", "level": level, "dim": 1}]))
    data = Data(
        pos=torch.zeros(3, 2),
        f=torch.zeros(3, 1),
        t=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    with pytest.raises(ValueError, match=missing):
        prepare_data_from_schema(data, schema)
