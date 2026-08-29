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
from hydragnn.utils.datasets import mark_pyg_cache_current, prepare_pyg_cache


def test_pyg_cache_migrates_raw_and_discards_legacy_processed_data(tmp_path):
    legacy = tmp_path / "schema-v1"
    (legacy / "raw").mkdir(parents=True)
    (legacy / "processed").mkdir()
    (legacy / "raw" / "dataset.npz").write_text("raw", encoding="utf-8")
    (legacy / "processed" / "data.pt").write_text("old", encoding="utf-8")

    prepare_pyg_cache(tmp_path, "schema-v2", ("schema-v1",))

    assert (tmp_path / "raw" / "dataset.npz").read_text() == "raw"
    assert not legacy.exists()
    assert not (tmp_path / "processed").exists()


def test_pyg_cache_preserves_matching_processed_artifact(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    artifact = processed / "data.pt"
    artifact.write_text("current", encoding="utf-8")
    mark_pyg_cache_current(tmp_path, "schema-v2")

    prepare_pyg_cache(tmp_path, "schema-v2")

    assert artifact.read_text() == "current"


def test_pyg_cache_replaces_only_processed_data_on_version_change(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    source = raw / "dataset.npz"
    source.write_text("raw", encoding="utf-8")
    (processed / "data.pt").write_text("old", encoding="utf-8")
    mark_pyg_cache_current(tmp_path, "schema-v1")

    prepare_pyg_cache(tmp_path, "schema-v2")

    assert source.read_text() == "raw"
    assert not processed.exists()
