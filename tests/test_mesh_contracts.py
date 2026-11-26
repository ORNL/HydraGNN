import pytest
import torch
from torch_geometric.data import Batch

from hydragnn.config.variable_schema import parse_variable_schema
from hydragnn.data.meshes import (
    MeshHierarchy,
    Trajectory,
    build_hierarchy,
    cell_batch,
    from_tetrahedra,
    prolong,
    restrict,
    unit_cube_mesh,
    unit_square_mesh,
    validate_mesh,
)

pytestmark = pytest.mark.mpi_skip()


def _closure(mesh):
    total = torch.zeros(mesh.num_cells, mesh.pos.shape[1], dtype=mesh.pos.dtype)
    flux = mesh.face_normal * mesh.face_area
    total.index_add_(0, mesh.face_index[0], flux)
    total.index_add_(0, mesh.face_index[1], -flux)
    total.index_add_(0, mesh.bface_owner, mesh.bface_normal * mesh.bface_area)
    return total


# -- 3D tetrahedra ---------------------------------------------------------
def test_unit_cube_counts_and_geometry():
    nx, ny, nz = 2, 3, 2
    mesh = validate_mesh(unit_cube_mesh(nx, ny, nz))
    one = torch.tensor(1.0, dtype=torch.float64)
    assert mesh.num_cells == 6 * nx * ny * nz
    assert mesh.face_nodes.shape[0] == 3
    torch.testing.assert_close(mesh.cell_volume.sum(), one)
    # Six unit faces, each split into 2 * n_a * n_b triangles.
    torch.testing.assert_close(mesh.bface_area.sum(), 6 * one)
    assert mesh.bface_owner.numel() == 4 * (nx * ny + ny * nz + nx * nz)
    # Each interior face is shared by exactly two tets: 4C = 2F + Fb.
    assert 4 * mesh.num_cells == 2 * mesh.num_faces + mesh.bface_owner.numel()
    torch.testing.assert_close(
        _closure(mesh), torch.zeros_like(_closure(mesh)), atol=1e-13, rtol=0
    )
    c = mesh.cell_centroid
    d = c[mesh.face_index[1]] - c[mesh.face_index[0]]
    assert ((d * mesh.face_normal).sum(dim=1) > 0).all()
    out = c[mesh.bface_owner] - 0.5
    assert ((out * mesh.bface_normal).sum(dim=1) > 0).all()
    assert mesh.boundary_node_mask.sum() == (nx + 1) * (ny + 1) * (nz + 1) - (
        (nx - 1) * (ny - 1) * (nz - 1)
    )


def test_tetrahedra_reorient_and_reject():
    pos = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64
    )
    mesh = validate_mesh(from_tetrahedra(pos, [[0, 2, 1, 3]]))
    torch.testing.assert_close(mesh.cell_volume, torch.full((1, 1), 1 / 6).double())
    assert mesh.num_faces == 0 and mesh.bface_owner.numel() == 4
    with pytest.raises(ValueError, match="zero volume"):
        from_tetrahedra(pos, [[0, 1, 2, 2]])
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        from_tetrahedra(pos, [[0, 1, 2, 4]])
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        from_tetrahedra(pos[:, :2], [[0, 1, 2, 3]])


def test_tet_batching_offsets():
    meshes = [unit_cube_mesh(1, 1, 1), unit_cube_mesh(2, 1, 1)]
    batch = Batch.from_data_list(meshes)
    assert batch.face_nodes.shape[0] == 3
    c0, n0 = meshes[0].num_cells, meshes[0].num_nodes
    f0 = meshes[0].num_faces
    assert torch.equal(batch.face_index[:, f0:], meshes[1].face_index + c0)
    assert torch.equal(batch.face_nodes[:, f0:], meshes[1].face_nodes + n0)
    assert torch.equal(
        cell_batch(batch),
        torch.tensor([0] * c0 + [1] * meshes[1].num_cells),
    )


# -- validate_mesh ------------------------------------------------------------
@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda m: m.face_normal.neg_(), "not closed"),
        (lambda m: m.face_normal.mul_(2.0), "unit length"),
        (lambda m: m.cell_volume.neg_(), "cell_volume"),
        (lambda m: m.face_index[1].fill_(10**6), "out of range"),
        (lambda m: setattr(m, "face_area", m.face_area[1:]), "face_area"),
        (lambda m: setattr(m, "material_id", torch.zeros(3).long()), "material_id"),
        (lambda m: setattr(m, "dt", torch.zeros(1, 1)), "dt"),
        (lambda m: setattr(m, "time", torch.zeros(2)), "time"),
        (lambda m: m.cell_conn.__setitem__(0, -1), "cell_conn"),
        (lambda m: setattr(m, "cell_volume", m.cell_volume[:, 0]), "cell_volume"),
        (lambda m: setattr(m, "cell_centroid", m.cell_centroid[:, :1]), "centroid"),
        (
            lambda m: setattr(
                m, "face_index", torch.cat([m.face_index, m.face_index[:1]])
            ),
            "face_index",
        ),
        (lambda m: setattr(m, "bface_owner", m.bface_owner[None]), "bface_owner"),
        (lambda m: setattr(m, "face_index", m.face_index.int()), "face_index"),
        (
            lambda m: (m.face_normal.neg_(), m.bface_normal.neg_()),
            "owner to neighbour",
        ),
        (lambda m: m.bface_normal.neg_(), "not closed"),
        (
            lambda m: (
                setattr(m, "query_pos", torch.zeros(2, 2, dtype=m.pos.dtype)),
                setattr(m, "query_cell", torch.tensor([0, -1])),
            ),
            "query_cell",
        ),
        (
            lambda m: (
                setattr(m, "query_pos", torch.zeros(3, 2, dtype=m.pos.dtype)),
                setattr(m, "query_cell", torch.tensor([0, 1])),
            ),
            "query_pos",
        ),
        (lambda m: setattr(m, "query_cell", torch.tensor([0])), "together"),
        (lambda m: setattr(m, "bface_nodes", m.bface_nodes.float()), "bface_nodes"),
        (lambda m: setattr(m, "bface_nodes", m.bface_nodes[0]), "bface_nodes"),
        (lambda m: setattr(m, "face_nodes", m.face_nodes.float()), "face_nodes"),
        (lambda m: setattr(m, "face_nodes", m.face_nodes[0]), "face_nodes"),
    ],
)
def test_validate_mesh_rejects(corrupt, match):
    mesh = unit_square_mesh(3, 2)
    validate_mesh(mesh)
    corrupt(mesh)
    with pytest.raises(ValueError, match=match):
        validate_mesh(mesh)


def test_validate_mesh_rejects_inward_3d_boundary_normals():
    mesh = unit_cube_mesh(1, 1, 1)
    mesh.face_normal.neg_()
    mesh.bface_normal.neg_()
    with pytest.raises(ValueError, match="owner to neighbour"):
        validate_mesh(mesh)
    mesh = from_tetrahedra(
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).double(),
        [[0, 1, 2, 3]],
    )
    mesh.bface_normal.neg_()
    with pytest.raises(ValueError, match="out of the owner"):
        validate_mesh(mesh)


def test_validate_mesh_accepts_located_queries():
    from hydragnn.data.meshes.mesh_data import locate_queries

    mesh = locate_queries(unit_square_mesh(2, 2), [[0.1, 0.2], [0.9, 0.6]])
    validate_mesh(mesh)


def test_metadata_batches_per_graph():
    meshes = []
    for i, n in enumerate((2, 3)):
        m = unit_square_mesh(n, n)
        m.material_id = torch.arange(m.num_cells) % 2
        m.fidelity = torch.tensor([[float(i)]])
        meshes.append(validate_mesh(m))
    batch = Batch.from_data_list(meshes)
    assert batch.fidelity.shape == (2, 1)
    assert batch.material_id.shape == (batch.num_cells,)


# -- Trajectory ---------------------------------------------------------------
def _trajectory(n=2, steps=5):
    mesh = unit_square_mesh(n, n)
    times = torch.linspace(0.0, 0.4, steps).double()
    u = torch.stack([mesh.cell_centroid[:, :1] * (1 + t) for t in times])
    return Trajectory(
        mesh, times, {"u": u}, static={"s": torch.ones(mesh.num_cells, 1)}
    )


def test_trajectory_pairs():
    traj = _trajectory()
    pairs = traj.pairs(horizon=2)
    assert len(pairs) == 3
    for t, p in enumerate(pairs):
        torch.testing.assert_close(p.u, traj.fields["u"][t])
        torch.testing.assert_close(p.u_next, traj.fields["u"][t + 2])
        torch.testing.assert_close(p.time, traj.times[t].view(1, 1))
        torch.testing.assert_close(p.dt, (traj.times[t + 2] - traj.times[t]).view(1, 1))
        assert "s" in p
        validate_mesh(p)
    assert len(traj.pairs(stride=2)) == 2
    batch = Batch.from_data_list(pairs)
    assert batch.time.shape == (3, 1) and batch.dt.shape == (3, 1)
    assert batch.u_next.shape[0] == batch.num_cells
    # Samples must not alias the shared mesh.
    pairs[0].pos.add_(1.0)
    assert traj.mesh.pos.max() == 1.0


def test_trajectory_rejects():
    mesh = unit_square_mesh(1, 1)
    u = torch.zeros(3, mesh.num_cells, 1)
    with pytest.raises(ValueError, match="increasing"):
        Trajectory(mesh, [0.0, 0.0, 1.0], {"u": u})
    with pytest.raises(ValueError, match="shape"):
        Trajectory(mesh, [0.0, 1.0], {"u": u})
    with pytest.raises(ValueError, match="clashes"):
        Trajectory(mesh, [0.0, 0.5, 1.0], {"pos": u})
    with pytest.raises(ValueError, match="horizon"):
        Trajectory(mesh, [0.0, 0.5, 1.0], {"u": u}).pairs(horizon=0)
    times = [0.0, 0.5, 1.0]
    for static in ({"u": u[0]}, {"pos": u[0]}, {"dt": u[0]}):
        with pytest.raises(ValueError, match="static name"):
            Trajectory(mesh, times, {"u": u}, static=static)
    with pytest.raises(ValueError, match="target names"):
        Trajectory(mesh, times, {"u": u, "u_next": u})
    with pytest.raises(ValueError, match="target names"):
        Trajectory(mesh, times, {"u": u}, static={"u_next": u[0]})
    traj = Trajectory(mesh, times, {"u": u})
    with pytest.raises(ValueError, match="non-empty"):
        traj.pairs(target_suffix="")
    with pytest.raises(ValueError, match="target names"):
        Trajectory(mesh, times, {"p": u}).pairs(target_suffix="os")  # -> "pos"
    assert "u_tgt" in traj.pairs(target_suffix="_tgt")[0]


# -- MeshHierarchy ------------------------------------------------------------
def _hierarchy(n=2):
    return build_hierarchy(unit_square_mesh(2 * n, 2 * n), unit_square_mesh(n, n))


def test_hierarchy_restrict_prolong():
    h = _hierarchy()
    assert isinstance(h, MeshHierarchy)
    assert h.num_coarse_cells == 8 and h.num_cells == 32
    # Each coarse triangle contains 4 fine ones in a uniform refinement.
    assert torch.equal(torch.bincount(h.fine_to_coarse), torch.full((8,), 4))
    const = torch.full((h.num_cells, 2), 3.0, dtype=torch.float64)
    torch.testing.assert_close(restrict(h, const), const[: h.num_coarse_cells])
    # Restriction preserves the integral (mass).
    f = h.cell_centroid[:, :1] ** 2
    torch.testing.assert_close(
        (restrict(h, f) * h.coarse_cell_volume).sum(), (f * h.cell_volume).sum()
    )
    coarse = torch.randn(h.num_coarse_cells, 1, dtype=torch.float64)
    torch.testing.assert_close(restrict(h, prolong(h, coarse)), coarse)


def test_hierarchy_batching():
    levels = [_hierarchy(1), _hierarchy(2)]
    batch = Batch.from_data_list(levels)
    c0 = levels[0].num_coarse_cells
    n0 = levels[0].num_coarse_nodes
    k = levels[0].num_cells
    assert torch.equal(batch.fine_to_coarse[k:], levels[1].fine_to_coarse + c0)
    fc = levels[0].coarse_face_index.shape[1]
    assert torch.equal(
        batch.coarse_face_index[:, fc:], levels[1].coarse_face_index + c0
    )
    assert torch.equal(
        batch.coarse_cell_conn[3 * c0 :], levels[1].coarse_cell_conn + n0
    )
    e0 = levels[0].coarse_edge_index.shape[1]
    assert torch.equal(
        batch.coarse_edge_index[:, e0:], levels[1].coarse_edge_index + n0
    )
    f = batch.cell_centroid[:, :1]
    singles = torch.cat([restrict(h, h.cell_centroid[:, :1]) for h in levels])
    torch.testing.assert_close(restrict(batch, f), singles)
    for orig, back in zip(levels, batch.to_data_list()):
        assert torch.equal(back.fine_to_coarse, orig.fine_to_coarse)
        assert torch.equal(back.coarse_face_index, orig.coarse_face_index)


def test_hierarchy_rejects():
    fine, coarse = unit_square_mesh(2, 2), unit_square_mesh(1, 1)
    with pytest.raises(ValueError, match="one entry"):
        build_hierarchy(fine, coarse, torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="at least one"):
        build_hierarchy(fine, coarse, torch.zeros(fine.num_cells, dtype=torch.long))
    # Covers both coarse cells, but child volumes do not tile them.
    bad = torch.zeros(fine.num_cells, dtype=torch.long)
    bad[0] = 1
    with pytest.raises(ValueError, match="non-nested"):
        build_hierarchy(fine, coarse, bad)


# -- units in the variable schema --------------------------------------------
def test_variable_units():
    raw = {
        "graph_type": "homogeneous",
        "inputs": [{"name": "f", "level": "node", "dim": 1, "units": "1/s"}],
        "outputs": [{"name": "u", "level": "node", "dim": 1}],
    }
    schema = parse_variable_schema(raw)
    assert schema.inputs[0].units == "1/s"
    assert schema.outputs[0].units is None
    raw["inputs"][0]["units"] = ""
    with pytest.raises(ValueError, match="units"):
        parse_variable_schema(raw)
