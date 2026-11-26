"""End-to-end smoke test: MeshGraphNet on the manufactured Poisson example
through the standard HydraGNN data, config and training path."""

import importlib.util
import os

import pytest

pytestmark = pytest.mark.mpi_skip()

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "pde", "poisson_manufactured.py"
)


@pytest.mark.parametrize(
    "config_name,names",
    [("poisson_mgn.json", ["u"]), ("poisson_mgn_flux.json", ["u", "q"])],
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
