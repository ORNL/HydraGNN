"""End-to-end smoke test: MeshGraphNet on the manufactured Poisson example
through the standard HydraGNN data, config and training path."""

import importlib.util
import os

import pytest

pytestmark = pytest.mark.mpi_skip()

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "pde", "poisson_manufactured.py"
)


def test_poisson_mgn_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("poisson_manufactured", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = module.main(num_samples=40, num_epoch=2)
    assert results["heldout_resolution"] == module.HELDOUT_RESOLUTION
    for key in ("test_rel_l2", "heldout_rel_l2"):
        assert results[key] == results[key]  # finite, not NaN
        assert 0.0 <= results[key] < 10.0
