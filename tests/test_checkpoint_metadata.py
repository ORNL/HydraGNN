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

import pytest
import torch

from hydragnn.config.variable_schema import SCHEMA_VERSION
from hydragnn.distributed import setup_ddp
from hydragnn.models.utils import (
    checkpoint_metadata,
    load_and_validate,
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
    restored = torch.nn.Linear(2, 1)
    assert (
        load_and_validate(restored, "old", {}, path=str(tmp_path), strict=True) is None
    )


def _config(hidden_dim=8, out_dim=1):
    return {
        "NeuralNetwork": {
            "Architecture": {"mpnn_type": "MeshGraphNet", "hidden_dim": hidden_dim}
        },
        "Variables": {"outputs": [{"name": "u", "level": "node", "dim": out_dim}]},
    }


def test_load_and_validate(tmp_path):
    normalizer = FieldNormalizer(
        {"f": {"policy": "scale_only", "shift": 0.0, "scale": 3.0}}
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(normalizer=normalizer))
    model = _save(tmp_path, "m", checkpoint_metadata(_config(), loader))

    restored = torch.nn.Linear(2, 1)
    saved = load_and_validate(
        restored, "m", _config(), path=str(tmp_path), normalizer=normalizer, strict=True
    )
    assert saved.state_dict() == normalizer.state_dict()
    torch.testing.assert_close(restored.weight, model.weight)

    with pytest.raises(
        ValueError, match="Architecture.hidden_dim: checkpoint 8, current 16"
    ):
        load_and_validate(
            restored, "m", _config(hidden_dim=16), path=str(tmp_path), strict=True
        )
    with pytest.raises(ValueError, match="Variables.outputs"):
        load_and_validate(
            restored, "m", _config(out_dim=2), path=str(tmp_path), strict=True
        )
    other = FieldNormalizer({"f": {"policy": "scale_only", "shift": 0.0, "scale": 4.0}})
    with pytest.raises(ValueError, match="normalizer"):
        load_and_validate(
            restored, "m", _config(), path=str(tmp_path), normalizer=other, strict=True
        )
    with pytest.warns(UserWarning, match="hidden_dim"):
        load_and_validate(restored, "m", _config(hidden_dim=16), path=str(tmp_path))
    with pytest.raises(ValueError, match="normalizer: checkpoint has one"):
        load_and_validate(restored, "m", _config(), path=str(tmp_path), strict=True)


def test_metadata_path_is_branch_aware(tmp_path, monkeypatch):
    import hydragnn.models

    class FakeMultiTask(torch.nn.Module):
        branch_id = 3

    monkeypatch.setattr(hydragnn.models, "MultiTaskModelMP", FakeMultiTask)
    config = {"Variables": {}}
    _save(tmp_path, "m", checkpoint_metadata(config))
    os.rename(tmp_path / "m" / "m.pk", tmp_path / "m" / "m_branch3.pk")
    loaded, _, _ = load_checkpoint_metadata(
        "m", path=str(tmp_path), model=FakeMultiTask()
    )
    assert loaded == config


class _FakeDeepSpeedEngine(torch.nn.Module):
    """Stores client_state like DeepSpeed's save/load_checkpoint."""

    store = {}

    def save_checkpoint(self, save_dir, tag, client_state=None):
        self.store[(save_dir, tag)] = dict(client_state or {})

    def load_checkpoint(self, load_dir, tag):
        return load_dir, dict(self.store.get((load_dir, tag), {}))


def test_deepspeed_round_trips_metadata(tmp_path):
    normalizer = FieldNormalizer(
        {"f": {"policy": "scale_only", "shift": 0.0, "scale": 3.0}}
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(normalizer=normalizer))
    engine = _FakeDeepSpeedEngine()
    save_model(
        engine,
        None,
        "ds",
        path=str(tmp_path),
        use_deepspeed=True,
        metadata=checkpoint_metadata(_config(), loader),
    )
    saved = load_and_validate(
        engine,
        "ds",
        _config(),
        path=str(tmp_path),
        normalizer=normalizer,
        strict=True,
        use_deepspeed=True,
    )
    assert saved.state_dict() == normalizer.state_dict()
    with pytest.raises(ValueError, match="hidden_dim"):
        load_and_validate(
            engine,
            "ds",
            _config(hidden_dim=16),
            path=str(tmp_path),
            strict=True,
            use_deepspeed=True,
        )


def test_normalized_outputs_rejected_with_interatomic_loss():
    from hydragnn.config.config_utils import reject_normalized_derivative_outputs
    from hydragnn.config import parse_variable_schema

    schema = parse_variable_schema(
        {
            "graph_type": "homogeneous",
            "inputs": [{"name": "z", "level": "node", "dim": 1}],
            "outputs": [
                {"name": "e", "level": "graph", "dim": 1, "normalize": "standard"}
            ],
        }
    )
    config = {
        "NeuralNetwork": {
            "Training": {"loss": {"enabled": True, "provider": "interatomic_potential"}}
        }
    }
    with pytest.raises(ValueError, match="interatomic_potential"):
        reject_normalized_derivative_outputs(config, schema)
    reject_normalized_derivative_outputs({"NeuralNetwork": {}}, schema)
