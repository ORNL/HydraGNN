##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import pytest
import torch
from torch_geometric.data import Batch

pytest.importorskip("adios2")
MPI = pytest.importorskip("mpi4py.MPI")
from hydragnn.data.adiosdataset import AdiosDataset, AdiosWriter
from hydragnn.data.meshes import MeshData, unit_square_mesh


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"preload": True},
        {"enable_cache": True},
        {"ddstore": True},
        {"ddstore": True, "ddstore_store_per_sample": False},
    ],
    ids=["file", "preload", "cache", "ddstore", "ddstore_feature"],
)
def test_mesh_data_adios_round_trip(tmp_path, kwargs):
    if kwargs.get("ddstore"):
        pytest.importorskip("pyddstore")
    meshes = [unit_square_mesh(n, n + 1) for n in (2, 4, 3)]
    for m in meshes:
        m.u = m.pos[:, :1] ** 2
    filename = str(tmp_path / "meshes.bp")
    writer = AdiosWriter(filename, MPI.COMM_SELF)
    writer.add("trainset", meshes)
    writer.save()

    dataset = AdiosDataset(filename, "trainset", MPI.COMM_SELF, **kwargs)
    loaded = [dataset[i] for i in range(len(dataset))]
    for expected, sample in zip(meshes, loaded):
        assert isinstance(sample, MeshData)
        assert set(sample.keys()) == set(expected.keys())
        for key in expected.keys():
            assert sample[key].dtype == expected[key].dtype, key
            assert torch.equal(sample[key], expected[key]), key

    # Face/cell indices must still be offset per graph after reading back.
    batch = Batch.from_data_list(loaded)
    ref = Batch.from_data_list(meshes)
    for key in ("face_index", "bface_owner", "cell_conn", "face_nodes"):
        assert torch.equal(batch[key], ref[key]), key


@pytest.mark.parametrize("kwargs", [{}, {"preload": True}], ids=["file", "preload"])
def test_hierarchy_and_tet_adios_round_trip(tmp_path, kwargs):
    from hydragnn.data.meshes import (
        MeshHierarchy,
        build_hierarchy,
        restrict,
        unit_cube_mesh,
    )

    levels = [
        build_hierarchy(unit_square_mesh(2 * n, 2 * n), unit_square_mesh(n, n))
        for n in (1, 2)
    ]
    for h in levels:
        h.time = torch.zeros(1, 1, dtype=torch.float64)
    tets = [unit_cube_mesh(1, 1, n) for n in (1, 2)]
    filename = str(tmp_path / "levels.bp")
    writer = AdiosWriter(filename, MPI.COMM_SELF)
    writer.add("hier", levels)
    writer.add("tets", tets)
    writer.save()

    for label, expected_list, cls in (
        ("hier", levels, MeshHierarchy),
        ("tets", tets, MeshData),
    ):
        dataset = AdiosDataset(filename, label, MPI.COMM_SELF, **kwargs)
        loaded = [dataset[i] for i in range(len(dataset))]
        for expected, sample in zip(expected_list, loaded):
            assert type(sample) is cls
            assert set(sample.keys()) == set(expected.keys())
            for key in expected.keys():
                assert torch.equal(sample[key], expected[key]), key
        batch = Batch.from_data_list(loaded)
        ref = Batch.from_data_list(expected_list)
        for key in ref.keys():
            assert torch.equal(batch[key], ref[key]), key
    batch = Batch.from_data_list(
        [AdiosDataset(filename, "hier", MPI.COMM_SELF)[i] for i in range(2)]
    )
    f = batch.cell_centroid[:, :1]
    torch.testing.assert_close(
        restrict(batch, f),
        torch.cat([restrict(h, h.cell_centroid[:, :1]) for h in levels]),
    )
