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
"""Data import through the installed FAIR-Chem AseDBDataset used by examples."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("fairchem", reason="native FAIR-Chem is optional")

from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase_db_backends.aselmdb import LMDBDatabase  # noqa: E402
from fairchem.core.datasets import AseDBDataset  # noqa: E402

from hydragnn.preprocess.graph_samples_checks_and_updates import (  # noqa: E402
    RadiusGraph,
    RadiusGraphPBC,
)

pytestmark = pytest.mark.fairchem

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
ROWS_PER_FILE = 3


def _load_example(monkeypatch, subdir, name):
    path = os.path.join(EXAMPLES, subdir)
    monkeypatch.syspath_prepend(path)
    # Examples import sibling modules (e.g. ``utils``) by bare name.
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    spec = importlib.util.spec_from_file_location(
        f"_example_{name}", os.path.join(path, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structure(file_idx, row_idx):
    rng = np.random.default_rng(10 * file_idx + row_idx)
    atoms = Atoms(
        numbers=[8, 1, 1, 26],
        positions=rng.uniform(0.0, 4.0, size=(4, 3)),
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0 * (10 * file_idx + row_idx),
        forces=rng.normal(size=(4, 3)),
        stress=rng.normal(size=6),
    )
    return atoms


@pytest.fixture
def aselmdb_dir(tmp_path):
    # Written in reverse so the test catches unsorted file discovery.
    for file_idx in reversed(range(2)):
        db = LMDBDatabase(str(tmp_path / f"shard{file_idx}.aselmdb"), readonly=False)
        for row_idx in range(ROWS_PER_FILE):
            db.write(_structure(file_idx, row_idx), data={"tag": row_idx})
        db.close()
    return tmp_path


def test_asedbdataset_reads_aselmdb_folder(aselmdb_dir):
    dataset = AseDBDataset(config=dict(src=str(aselmdb_dir)))

    assert dataset.num_samples == len(dataset) == 2 * ROWS_PER_FILE
    for idx in range(dataset.num_samples):
        file_idx, row_idx = divmod(idx, ROWS_PER_FILE)
        expected = _structure(file_idx, row_idx)
        atoms = dataset.get_atoms(idx)
        np.testing.assert_array_equal(
            atoms.get_atomic_numbers(), expected.get_atomic_numbers()
        )
        np.testing.assert_allclose(atoms.get_positions(), expected.get_positions())
        np.testing.assert_allclose(atoms.get_cell(), expected.get_cell())
        assert atoms.get_pbc().all()
        assert atoms.get_potential_energy() == expected.get_potential_energy()
        np.testing.assert_allclose(atoms.get_forces(), expected.get_forces())
        np.testing.assert_allclose(atoms.get_stress(), expected.get_stress())
        assert atoms.info["tag"] == row_idx


def test_asedbdataset_internals_used_by_oc25(aselmdb_dir):
    dataset = AseDBDataset(config=dict(src=str(aselmdb_dir)))

    assert list(dataset._idlen_cumulative) == [ROWS_PER_FILE, 2 * ROWS_PER_FILE]
    assert len(dataset.dbs) == len(dataset.db_ids) == 2
    row = dataset.dbs[1]._get_row(dataset.db_ids[1][0])
    np.testing.assert_allclose(row.positions, _structure(1, 0).get_positions())


def test_oc25_reads_aselmdb_rows(aselmdb_dir, monkeypatch):
    oc25 = _load_example(monkeypatch, "open_catalyst_2025", "oc25")
    dataset = AseDBDataset(config=dict(src=str(aselmdb_dir / "shard1.aselmdb")))
    expected = _structure(1, 2)

    atoms = oc25._get_atoms(dataset, 2)
    assert atoms.get_potential_energy() == expected.get_potential_energy()
    np.testing.assert_allclose(atoms.get_forces(), expected.get_forces())
    assert atoms.info["sid"] == 2

    data = oc25._create_pytorch_data_object(
        2,
        dataset,
        energy_per_atom_bool=False,
        radius_graph_pbc=RadiusGraphPBC(5.0, loop=False, max_num_neighbors=20),
        radius_graph=RadiusGraph(5.0, loop=False, max_num_neighbors=20),
        graphgps_transform=None,
        forces_norm_threshold=1000.0,
    )
    assert data is not None
    assert data.atomic_numbers.tolist() == [8, 1, 1, 26]
    torch.testing.assert_close(data.pos, torch.tensor(expected.positions).float())
    assert data.energy.item() == pytest.approx(expected.get_potential_energy())
    assert data.edge_index.size(1) > 0
    assert data.pbc.tolist() == [1, 1, 1]


class _LegacyRow:
    """Row layout of old OC25 shards whose arrays stay ``__ndarray__`` dicts."""

    def __init__(self):
        self._row = {}
        self.numbers = {"__ndarray__": [[3], "int64", [1, 8, 1]]}
        self.positions = {"__ndarray__": [[3, 3], "float64", list(np.arange(9.0))]}
        # Some shards dropped the shape/dtype header and stored flat values only.
        self.cell = {"__ndarray__": list(np.eye(3).ravel() * 5.0)}
        self.pbc = [1, 1, 1]
        self.data = {"energy": [-3.5], "forces": list(np.ones(9))}

    def toatoms(self):
        raise KeyError("__ndarray__")


class _LegacyDataset:
    _idlen_cumulative = [1]
    db_ids = [[1]]

    class _DB:
        def _get_row(self, _id):
            return _LegacyRow()

    dbs = [_DB()]


def test_oc25_repairs_legacy_rows(monkeypatch):
    oc25 = _load_example(monkeypatch, "open_catalyst_2025", "oc25")

    atoms = oc25._get_atoms(_LegacyDataset(), 0)

    assert atoms.get_atomic_numbers().tolist() == [1, 8, 1]
    np.testing.assert_allclose(atoms.positions, np.arange(9.0).reshape(3, 3))
    np.testing.assert_allclose(atoms.cell.array, np.eye(3) * 5.0)
    assert atoms.pbc.all()
    assert atoms.get_potential_energy() == -3.5
    np.testing.assert_allclose(atoms.get_forces(), np.ones((3, 3)))


def test_omat24_builds_graph_from_aselmdb(aselmdb_dir, monkeypatch):
    omat24 = _load_example(monkeypatch, "open_materials_2024", "omat24")
    dataset = AseDBDataset(config=dict(src=str(aselmdb_dir)))
    builder = omat24.OMat2024.__new__(omat24.OMat2024)
    builder.dataset = []
    builder.rank = 0
    builder.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=20)
    builder.radius_graph_pbc = RadiusGraphPBC(5.0, loop=False, max_num_neighbors=20)
    builder.graphgps_transform = None
    builder.forces_norm_threshold = 1000.0

    builder._create_pytorch_data_object(dataset, 4)

    assert len(builder.dataset) == 1
    data = builder.dataset[0]
    expected = _structure(1, 1)
    assert data.atomic_numbers.tolist() == [8, 1, 1, 26]
    assert data.energy.item() == pytest.approx(expected.get_potential_energy())
    torch.testing.assert_close(data.forces, torch.tensor(expected.get_forces()).float())
    assert data.stress.shape == (3, 3)
    assert data.edge_index.size(1) > 0
