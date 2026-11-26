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

import os
from types import SimpleNamespace

import torch

from hydragnn.config.variable_schema import SCHEMA_VERSION
from hydragnn.distributed import setup_ddp
from hydragnn.models.utils import (
    checkpoint_metadata,
    load_checkpoint_metadata,
    load_existing_model,
    save_model,
)
from hydragnn.preprocess import FieldNormalizer


def _save(tmp_path, name, metadata):
    setup_ddp()  # save_model ends with dist.barrier()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    os.makedirs(tmp_path / name)
    save_model(model, optimizer, name, path=str(tmp_path), metadata=metadata)
    return model


def test_metadata_round_trip(tmp_path):
    normalizer = FieldNormalizer(
        {"q": {"policy": "scale_only", "shift": 0.0, "scale": 2.5}}
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(normalizer=normalizer))
    config = {"NeuralNetwork": {"hist": torch.tensor([1, 2])}, "Variables": {}}
    model = _save(tmp_path, "m", checkpoint_metadata(config, loader))

    loaded_config, version, loaded_norm = load_checkpoint_metadata(
        "m", path=str(tmp_path)
    )
    assert loaded_config == {"NeuralNetwork": {"hist": [1, 2]}, "Variables": {}}
    assert version == SCHEMA_VERSION
    assert loaded_norm.state_dict() == normalizer.state_dict()

    restored = torch.nn.Linear(2, 1)
    load_existing_model(restored, "m", path=str(tmp_path))
    torch.testing.assert_close(restored.weight, model.weight)


def test_checkpoint_without_metadata(tmp_path):
    _save(tmp_path, "old", None)
    assert load_checkpoint_metadata("old", path=str(tmp_path)) == (None, None, None)
