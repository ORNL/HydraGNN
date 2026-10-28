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
import json

import pytest
import torch

from examples.qm9 import qm9


def _processor(report_directory):
    pytest.importorskip("rdkit", reason="raw QM9 preprocessing requires RDKit")
    from examples.qm9.qm9_raw_processor import RobustQM9

    processor = object.__new__(RobustQM9)
    processor.report_directory = report_directory
    return processor


def test_qm9_conversion_preserves_original_target_index(tmp_path):
    pytest.importorskip("rdkit", reason="raw QM9 preprocessing requires RDKit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=0) == 0
    molecule.SetProp("_Name", "methane")
    targets = torch.arange(3 * 19, dtype=torch.float).reshape(3, 19)

    data = _processor(tmp_path)._convert_molecule(molecule, 1, targets)

    assert data.idx == 1
    assert data.name == "methane"
    torch.testing.assert_close(data.y, targets[1].unsqueeze(0))


def test_qm9_rejection_report_records_identity_stage_and_reason(tmp_path):
    processor = _processor(tmp_path)
    failure = processor._failure(
        17, "conversion", "conversion failed", ValueError("bad valence"), "mol-18"
    )
    summary = {"converted": 17, "rejected": 1, "completed": True}

    processor._write_reports([failure], summary)

    records = [
        json.loads(line)
        for line in (tmp_path / "unconverted_molecules.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records == [failure]
    assert records[0]["record_index"] == 17
    assert records[0]["qm9_id"] == 18
    assert records[0]["stage"] == "conversion"
    assert records[0]["exception"] == "ValueError"
    assert records[0]["reason"] == "bad valence"
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


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

    qm9.prepare_qm9_cache(tmp_path, "expected-mode")

    assert raw_file.read_text(encoding="utf-8") == "raw data"
    assert not processed.exists()
    assert all(
        not (tmp_path / name).exists() for name in qm9.QM9_LEGACY_CACHE_DIRECTORIES
    )


def test_qm9_cache_keeps_only_matching_mode(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    artifact = processed / "data_v3.pt"
    artifact.write_text("current", encoding="utf-8")
    qm9.mark_qm9_cache_current(tmp_path, "subset-1000")

    qm9.prepare_qm9_cache(tmp_path, "subset-1000")

    assert artifact.read_text(encoding="utf-8") == "current"

    qm9.prepare_qm9_cache(tmp_path, "full")
    assert not processed.exists()
