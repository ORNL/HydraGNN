"""Focused regressions for HydraGNN's FAIR-Chem backbone adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn
import importlib.util
import importlib

from hydragnn.models.AllScAIPStack import _resolve_frequency_list
from hydragnn.models.AllScAIPStack import AllScAIPStack
from hydragnn.models.Base import Base
from hydragnn.utils.model.allscaip.AllScAIP import AllScAIPBackbone
from hydragnn.utils.model.allscaip.utils.allscaip_radius_graph import (
    build_radius_graph,
    build_radius_graph_chunked,
)


def pytest_allscaip_frequency_list_explicit():
    assert _resolve_frequency_list(8, [3, 2, 3], True) == [3, 2, 3]


def pytest_allscaip_frequency_list_fairchem_default():
    assert _resolve_frequency_list(64, None, True) == [20, 10, 4, 10, 20]


def pytest_allscaip_frequency_list_invalid_sum():
    with pytest.raises(ValueError, match="must sum"):
        _resolve_frequency_list(8, [4, 3], True)


def pytest_allscaip_frequency_list_unused_without_mask():
    assert _resolve_frequency_list(17, None, False) == [17]


def pytest_external_backbone_embedding_skips_hydragnn_activation():
    model = Base.__new__(Base)
    nn.Module.__init__(model)
    model.skip_post_conv_processing = True
    model.activation_function = nn.ReLU()
    embedding = torch.tensor([[-2.0, 3.0]])
    delivered = model._postprocess_conv_output(embedding, None, object(), nn.Identity())
    assert torch.equal(delivered, embedding)


def pytest_allscaip_embedding_applies_output_normalization():
    class FakeBackbone(nn.Module):
        def forward(self, _data):
            return {"node_reps": torch.tensor([[1.0, 2.0]])}

    class TrackingNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def forward(self, value):
            self.called = True
            return value + 5

    model = AllScAIPStack.__new__(AllScAIPStack)
    nn.Module.__init__(model)
    model.allscaip_backbone = FakeBackbone()
    model.allscaip_output_norm = TrackingNorm()
    model._build_adapter = lambda data: data
    inv, equiv, _ = model._embedding(object())
    assert model.allscaip_output_norm.called
    assert torch.equal(inv, torch.tensor([[6.0, 7.0]]))
    assert equiv.shape == (1, 0)


def pytest_allscaip_construction_preserves_global_matmul_precision(monkeypatch):
    # Avoid constructing the large real blocks: this regression is solely
    # about constructor ownership of process-global PyTorch state.
    monkeypatch.setattr(
        "hydragnn.utils.model.allscaip.AllScAIP.InputBlock",
        lambda **_: nn.Identity(),
    )
    monkeypatch.setattr(
        "hydragnn.utils.model.allscaip.AllScAIP.GraphAttentionBlock",
        lambda **_: nn.Identity(),
    )
    before = torch.get_float32_matmul_precision()
    AllScAIPBackbone(
        regress_forces=False,
        direct_forces=False,
        hidden_size=64,
        num_layers=0,
        max_num_elements=119,
        max_radius=5.0,
        knn_k=8,
        atten_name="math",
        atten_num_heads=1,
    )
    assert torch.get_float32_matmul_precision() == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def pytest_allscaip_chunked_graph_matches_dense(dtype):
    torch.manual_seed(7)
    pos = torch.rand(8, 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * 20
    images = torch.zeros(1, 3, dtype=dtype)
    args = (pos, cell, images, 3.0, 0, pos.device, 4, False, 0.2, 0.1, True)
    dense = build_radius_graph(*args)
    chunked = build_radius_graph_chunked(*args, chunk_size=3)
    # The edge set and numeric geometry agree exactly. Rank values may choose a
    # different (but valid) order for equal bidirectional envelopes.
    assert torch.equal(dense[0], chunked[0])
    assert torch.equal(dense[1], chunked[1])
    for index in (4, 5, 6):
        assert dense[index].dtype == dtype
        assert torch.allclose(dense[index], chunked[index], atol=1e-6, rtol=1e-6)
    for result in (dense, chunked):
        assert (result[2] >= 0).all() and (result[2] < 4).all()
        assert (result[3] >= 0).all() and (result[3] < 4).all()


@pytest.mark.skipif(
    importlib.util.find_spec("fairchem") is None,
    reason="native FAIR-Chem is optional",
)
def pytest_native_fairchem_available_for_parity_suite():
    """Compare graph preprocessing to native FAIR-Chem when it is installed."""
    native_module = importlib.import_module(
        "fairchem.core.models.allscaip.utils.allscaip_radius_graph"
    )
    pos = torch.rand(6, 3)
    cell = torch.eye(3) * 10
    images = torch.zeros(1, 3)
    args = (pos, cell, images, 2.0, 0, pos.device, 4, False, 0.2, 0.1, True)
    ours = build_radius_graph(*args)
    native = native_module.build_radius_graph(*args)
    for ours_value, native_value in zip(ours, native):
        assert torch.allclose(ours_value, native_value)

