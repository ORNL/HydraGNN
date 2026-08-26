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
from rdkit import Chem

from examples.qm9 import qm9


def test_qm9_raw_builder_limits_supplier_before_pyg_processing(monkeypatch):
    original_supplier = Chem.SDMolSupplier
    monkeypatch.setattr(Chem, "SDMolSupplier", lambda *args, **kwargs: range(5000))

    observed = {}

    def fake_qm9(**kwargs):
        observed["records"] = list(Chem.SDMolSupplier("raw.sdf"))
        return kwargs

    monkeypatch.setattr(qm9.torch_geometric.datasets, "QM9", fake_qm9)
    result = qm9.build_qm9_from_raw("cache", pre_transform=lambda data: data)

    assert len(observed["records"]) == qm9.num_samples
    assert result["root"] == "cache"
    assert Chem.SDMolSupplier is not original_supplier


def test_qm9_cache_reuses_raw_and_removes_obsolete_processed_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_file = raw / "gdb9.sdf"
    raw_file.write_text("raw data", encoding="utf-8")

    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "data_v3.pt").write_text("obsolete", encoding="utf-8")
    for name in qm9.QM9_LEGACY_CACHE_DIRECTORIES:
        legacy = tmp_path / name
        legacy.mkdir()
        (legacy / "duplicate.raw").write_text("duplicate", encoding="utf-8")

    qm9.prepare_qm9_cache(tmp_path)

    assert raw_file.read_text(encoding="utf-8") == "raw data"
    assert not processed.exists()
    assert all(
        not (tmp_path / name).exists() for name in qm9.QM9_LEGACY_CACHE_DIRECTORIES
    )


def test_qm9_cache_keeps_matching_processed_data(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    artifact = processed / "data_v3.pt"
    artifact.write_text("current", encoding="utf-8")
    qm9.mark_qm9_cache_current(tmp_path)

    qm9.prepare_qm9_cache(tmp_path)

    assert artifact.read_text(encoding="utf-8") == "current"
