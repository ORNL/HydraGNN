"""End-to-end smoke test: MeshGraphNet on the manufactured Poisson example
through the standard HydraGNN data, config and training path."""

import importlib.util
import os

import pytest

from hydragnn.config.variable_schema import SCHEMA_VERSION
from hydragnn.models.utils import load_checkpoint_metadata

pytestmark = pytest.mark.mpi_skip()

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "pde", "poisson_manufactured.py"
)


@pytest.mark.parametrize(
    "config_name,names",
    [
        ("poisson_mgn.json", ["u"]),
        ("poisson_mgn_flux.json", ["u", "q"]),
        ("poisson_msmgn.json", ["u"]),
        ("poisson_rigno.json", ["u"]),
        ("darcy_mgn.json", ["u"]),
        ("darcy_msmgn.json", ["u"]),
        ("darcy_rigno.json", ["u"]),
        ("poisson_deeponet.json", ["u"]),
        ("poisson_composite_mgn_gsno.json", ["u"]),
        ("poisson_composite_regional_flux.json", ["u", "q"]),
    ],
)
def test_poisson_mgn_train(tmp_path, monkeypatch, config_name, names):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("poisson_manufactured", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = module.main(num_samples=40, num_epoch=2, config_name=config_name)
    assert results["heldout_resolution"] == module.HELDOUT_RESOLUTION
    for key in ("test_rel_l2", "heldout_rel_l2"):
        assert sorted(results[key]) == sorted(names)
        for value in results[key].values():
            assert 0.0 <= value < 10.0  # also rejects NaN

    log_name = os.path.splitext(config_name)[0]
    config, version, normalizer = load_checkpoint_metadata(log_name)
    assert config["Variables"]["spatial_dim"] == 2
    assert version == SCHEMA_VERSION
    assert sorted(normalizer.state_dict()) == sorted({"f", *names} - {"u"})
