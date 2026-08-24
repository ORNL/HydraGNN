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

import torch

from examples.lsms.lsms_preprocess import (
    parse_lsms_file,
    prepare_lsms_dataset,
    split_and_write_lsms_dataset,
)
from hydragnn.utils.datasets.serializeddataset import SerializedDataset


def _write_lsms_record(path, energy=-8.0):
    path.write_text(
        f"{energy} metadata\n"
        "26 0 0.0 0.0 0.0 27.5 1.2\n"
        "78 0 1.0 0.0 0.0 79.0 -0.2\n",
        encoding="utf-8",
    )


def _config(stratified=False):
    return {
        "Dataset": {
            "name": "lsms-test",
            "compositional_stratified_splitting": stratified,
        },
        "NeuralNetwork": {
            "Architecture": {"radius": 2.0, "max_neighbours": 8},
            "Training": {"perc_train": 0.5},
        },
        "Variables": {
            "inputs": [{"name": "num_of_protons", "level": "node", "dim": 1}],
            "outputs": [
                {"name": "free_energy_per_atom", "level": "graph", "dim": 1},
                {"name": "charge_density", "level": "node", "dim": 1},
                {"name": "magnetic_moment", "level": "node", "dim": 1},
            ],
        },
    }


def test_lsms_parser_creates_named_attributes(tmp_path):
    record = tmp_path / "sample.lsms"
    _write_lsms_record(record)

    data = parse_lsms_file(record)

    torch.testing.assert_close(data.num_of_protons, torch.tensor([[26.0], [78.0]]))
    torch.testing.assert_close(data.free_energy_per_atom, torch.tensor([[-4.0]]))
    torch.testing.assert_close(data.charge_density, torch.tensor([[1.5], [1.0]]))
    torch.testing.assert_close(data.magnetic_moment, torch.tensor([[1.2], [-0.2]]))
    assert data.pos.shape == (2, 3)
    assert data.x is None
    assert data.y is None


def test_lsms_preprocessing_compiles_schema_and_edges(tmp_path):
    _write_lsms_record(tmp_path / "sample.lsms")

    data = prepare_lsms_dataset(tmp_path, _config())[0]

    assert data.x.shape == (2, 1)
    assert data.edge_index.shape == (2, 2)
    assert data.y.shape == (5, 1)
    torch.testing.assert_close(data.y_loc, torch.tensor([[0, 1, 3, 5]]))
    assert data.edge_attr is None


def test_lsms_preprocessing_writes_reusable_splits(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(4):
        _write_lsms_record(raw / f"sample-{index}.lsms", energy=-8.0 - index)
    config = _config()
    prepared = prepare_lsms_dataset(raw, config)
    output = tmp_path / "prepared"

    splits = split_and_write_lsms_dataset(prepared, config, output)

    assert [len(split) for split in splits] == [2, 1, 1]
    for label, expected_size in (("trainset", 2), ("valset", 1), ("testset", 1)):
        loaded = SerializedDataset(output, "lsms-test", label)
        assert len(loaded) == expected_size
        assert loaded[0].x.shape == (2, 1)
        assert loaded[0].y_loc.shape == (1, 4)
