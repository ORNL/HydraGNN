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
