import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import cell_batch, cell_offsets, unit_square_mesh

pytestmark = pytest.mark.mpi_skip()


def _divergence_of_constant(mesh):
    """Sum of n * area over each closed cell; must vanish."""
    c = mesh.num_cells
    total = torch.zeros(c, 2, dtype=mesh.pos.dtype)
    flux = mesh.face_normal * mesh.face_area
    total.index_add_(0, mesh.face_index[0], flux)
    total.index_add_(0, mesh.face_index[1], -flux)
    total.index_add_(0, mesh.bface_owner, mesh.bface_normal * mesh.bface_area)
    return total


def test_unit_square_counts_and_geometry():
    nx, ny = 4, 3
    mesh = unit_square_mesh(nx, ny)
    assert mesh.num_nodes == (nx + 1) * (ny + 1)
    assert mesh.num_cells == 2 * nx * ny
    edges = nx * (ny + 1) + ny * (nx + 1) + nx * ny
    assert mesh.num_faces + mesh.bface_owner.numel() == edges
    assert mesh.bface_owner.numel() == 2 * (nx + ny)
    assert mesh.edge_index.shape[1] == 2 * edges
    torch.testing.assert_close(mesh.cell_volume.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(mesh.bface_area.sum(), torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(
        mesh.face_normal.norm(dim=1), torch.ones(mesh.num_faces, dtype=torch.float64)
    )
    assert int(mesh.boundary_node_mask.sum()) == 2 * (nx + ny)


def test_face_normals_point_owner_to_neighbour():
    mesh = unit_square_mesh(3, 3)
    c = mesh.cell_centroid
    d = c[mesh.face_index[1]] - c[mesh.face_index[0]]
    assert ((d * mesh.face_normal).sum(dim=1) > 0).all()
    out = mesh.cell_centroid[mesh.bface_owner]
    mid = 0.5 * torch.ones_like(out)
    # Boundary normals of the unit square point away from its centre.
    assert (((out - mid) * mesh.bface_normal).sum(dim=1) > 0).all()


def test_closed_cells_have_zero_net_normal():
    mesh = unit_square_mesh(5, 4)
    torch.testing.assert_close(
        _divergence_of_constant(mesh),
        torch.zeros(mesh.num_cells, 2, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_batching_preserves_entity_indices():
    meshes = [unit_square_mesh(2, 2), unit_square_mesh(3, 1), unit_square_mesh(1, 4)]
    batch = Batch.from_data_list(meshes)
    assert batch.num_cells == sum(m.num_cells for m in meshes)
    assert batch.face_index.shape[1] == sum(m.num_faces for m in meshes)

    graph_of_cell = cell_batch(batch)
    expected = torch.cat(
        [torch.full((m.num_cells,), i) for i, m in enumerate(meshes)]
    )
    assert torch.equal(graph_of_cell, expected)
    # Faces never cross graphs.
    assert torch.equal(
        graph_of_cell[batch.face_index[0]], graph_of_cell[batch.face_index[1]]
    )
    assert torch.equal(
        batch.batch[batch.cell_conn[cell_offsets(batch)]], graph_of_cell
    )
    torch.testing.assert_close(
        _divergence_of_constant(batch),
        torch.zeros(batch.num_cells, 2, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    for i, mesh in enumerate(meshes):
        again = batch.get_example(i)
        for key in ("face_index", "bface_owner", "cell_conn", "edge_index"):
            assert torch.equal(again[key], mesh[key]), key


def test_clockwise_input_is_reoriented():
    from hydragnn.data.meshes import from_triangles

    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cw = torch.tensor([[0, 2, 1], [1, 2, 3]])
    mesh = from_triangles(pos, cw)
    assert (mesh.cell_volume > 0).all()
    assert mesh.num_faces == 1 and mesh.bface_owner.numel() == 4


def test_rejects_degenerate_and_non_manifold():
    from hydragnn.data.meshes import from_triangles

    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="degenerate"):
        from_triangles(pos, torch.tensor([[0, 1, 2]]))
    with pytest.raises(ValueError, match="non-manifold"):
        from_triangles(pos, torch.tensor([[0, 1, 3], [0, 1, 4], [0, 1, 5]]))
