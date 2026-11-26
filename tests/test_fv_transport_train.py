"""End-to-end smoke test: FluxGNN and the matched MeshGraphNet baseline on the
finite-volume transport example through the standard training path."""

import importlib.util
import json
import os

import pytest
import torch.distributed as dist

import hydragnn
from hydragnn.models.utils import load_checkpoint_metadata

pytestmark = pytest.mark.mpi_skip()

HERE = os.path.join(os.path.dirname(__file__), "..", "examples", "pde")
EXAMPLE = os.path.join(HERE, "fv_transport.py")


def _module():
    spec = importlib.util.spec_from_file_location("fv_transport", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "config_name", ["fv_transport_fluxgnn.json", "fv_transport_mgn.json"]
)
def test_fv_transport_train(tmp_path, monkeypatch, config_name):
    monkeypatch.chdir(tmp_path)
    module = _module()
    results = module.main(num_samples=36, num_epoch=2, config_name=config_name)
    assert results["heldout_resolution"] == module.HELDOUT_RESOLUTION
    assert len(results["rollout_rel_l2"]) == module.ROLLOUT_STEPS
    for value in [results["test_rel_l2"], *results["rollout_rel_l2"]]:
        assert 0.0 <= value < 10.0  # also rejects NaN
    if "fluxgnn" in config_name:
        assert results["max_rel_balance_residual"] < 1e-12
        # Untrained-or-barely-trained FluxGNN is already a consistent scheme.
        assert results["test_rel_l2"] < 0.2
    else:
        assert "max_rel_balance_residual" not in results

    config, _, normalizer = load_checkpoint_metadata(os.path.splitext(config_name)[0])
    assert sorted(normalizer.state_dict()) == ["s"]
    if "fluxgnn" in config_name:
        assert config["NeuralNetwork"]["Architecture"]["fluxgnn_layout"] == {
            "state": [0],
            "vectors": [[1, 2]],
            "scalars": [3],
        }


def test_fv_transport_fp32_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = _module()
    with open(os.path.join(HERE, "fv_transport_fluxgnn.json")) as f:
        config = json.load(f)
    config["NeuralNetwork"]["Training"]["precision"] = "fp32"
    path = tmp_path / "fv_transport_fp32.json"
    path.write_text(json.dumps(config))
    results = module.main(num_samples=36, num_epoch=1, config_name=str(path))
    for value in [results["test_rel_l2"], *results["rollout_rel_l2"]]:
        assert 0.0 <= value < 10.0
    assert results["max_rel_balance_residual"] < 1e-4


def test_balance_maxima_treat_nan_as_failure():
    import math

    import torch

    assert _module().finite_or_inf(math.nan) == math.inf
    spec = importlib.util.spec_from_file_location(
        "verify_fluxgnn", os.path.join(HERE, "verify_fluxgnn.py")
    )
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    mesh = verify.make_mesh(4, 0)
    mesh.cell_x[0, 0] = math.nan
    model = verify.make_model("learned", True, 0)
    with torch.no_grad():
        assert verify.worst_balance(model, mesh) == math.inf


def _update_config_error(module, mutate, match):
    config, loaders = _config_and_loaders(module, mutate)
    hydragnn.distributed.setup_ddp()
    try:
        with pytest.raises(ValueError, match=match):
            hydragnn.config.update_config(config, *loaders)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _config_and_loaders(module, mutate):
    with open(os.path.join(HERE, "fv_transport_fluxgnn.json")) as f:
        config = json.load(f)
    mutate(config)
    dataset = module.make_dataset(4, (4,), seed=0, steps=2)
    for sample in dataset:
        sample.pos_x = sample.pos[:, :1]
    loaders = hydragnn.preprocess.create_dataloaders(
        dataset, dataset, dataset, 2, variables=config["Variables"]
    )
    return config, loaders


def test_update_config_rejects_cell_inputs_for_other_models():
    module = _module()

    def mutate(config):
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = "MeshGraphNet"

    _update_config_error(module, mutate, "only by FluxGNN")


def test_update_config_requires_cell_inputs_for_fluxgnn():
    module = _module()

    def mutate(config):
        config["Variables"]["inputs"] = [
            config["Variables"]["inputs"][0],
            {"name": "pos_x", "level": "node", "dim": 1},
        ]

    _update_config_error(module, mutate, "needs cell-level inputs")
