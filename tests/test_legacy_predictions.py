"""Frozen fixed-seed predictions for every pre-PDE architecture.

Regenerate with ``HYDRAGNN_REGEN_LEGACY_FIXTURE=1`` only on the base commit.
"""

import os
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from tests.test_model_registry import (
    HETEROGENEOUS_MODELS,
    HOMOGENEOUS_MODELS,
    _batch,
    _build_homogeneous,
    _hetero_graph,
    _heterogeneous_args,
)
from hydragnn.models.create import create_model

pytestmark = pytest.mark.mpi_skip()

FIXTURE = Path(__file__).parent / "inputs" / "legacy_predictions_fp64.pt"
REGEN = os.getenv("HYDRAGNN_REGEN_LEGACY_FIXTURE") == "1"


def _predict(mpnn_type):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        if mpnn_type in HOMOGENEOUS_MODELS:
            model = _build_homogeneous(mpnn_type).double().eval()
            data = _batch()
        else:
            model = create_model(**_heterogeneous_args(mpnn_type)).double().eval()
            data = Batch.from_data_list([_hetero_graph(1), _hetero_graph(2)])
        data = data.apply(lambda t: t.double() if t.is_floating_point() else t)
        with torch.no_grad():
            return [out.detach().clone() for out in model(data)]
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def reference():
    if REGEN:
        return {}
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; generate it on the base commit")
    return torch.load(FIXTURE)


@pytest.mark.parametrize("mpnn_type", HOMOGENEOUS_MODELS + HETEROGENEOUS_MODELS)
def test_legacy_prediction_unchanged(mpnn_type, reference):
    actual = _predict(mpnn_type)
    if REGEN:
        reference[mpnn_type] = actual
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference, FIXTURE)
        return
    if mpnn_type not in reference:
        pytest.skip(f"{mpnn_type} absent from fixture (dependency unavailable)")
    expected = reference[mpnn_type]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)
