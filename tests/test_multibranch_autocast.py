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

from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hydragnn.models.Base import Base


class _NodeHead(torch.nn.Linear):
    def forward(self, x, batch):
        return super().forward(x)


class _DecoderHarness(Base):
    def __init__(self, head_type, num_branches, variance):
        torch.nn.Module.__init__(self)
        self.graph_convs = torch.nn.ModuleList()
        self.feature_layers = torch.nn.ModuleList()
        self.head_dims = [1]
        self.head_type = [head_type]
        self.num_branches = num_branches
        self.var_output = int(variance)
        self.config_heads = {"node": [{"architecture": {"type": "mlp"}}]}
        self.graph_shared = torch.nn.ModuleDict(
            {f"branch-{i}": torch.nn.Identity() for i in range(num_branches)}
        )
        head_class = torch.nn.Linear if head_type == "graph" else _NodeHead
        self.heads_NN = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        f"branch-{i}": head_class(3, 1 + self.var_output)
                        for i in range(num_branches)
                    }
                )
            ]
        )

    def _embedding(self, data):
        return data.pos.square(), None, {}

    def _pool_graph_features(self, x, batch):
        return torch.stack([x[batch == index].sum(dim=0) for index in batch.unique()])

    def _apply_graph_pool_conditioning(self, x, data):
        return x


@pytest.fixture
def decoder(monkeypatch, request):
    if request.param == "base":
        return lambda model, data: model(data)
    example_dir = (
        Path(__file__).resolve().parents[1] / "examples" / "multidataset_hpo_sc26"
    )
    monkeypatch.syspath_prepend(str(example_dir))
    fused = import_module("inference_fused")
    return lambda model, data: fused._decode_branch(model, data, model._embedding(data))


@pytest.mark.parametrize("decoder", ["base", "encoder_reuse"], indirect=True)
@pytest.mark.parametrize("head_type", ["graph", "node"])
@pytest.mark.parametrize("variance", [False, True])
@pytest.mark.parametrize("dataset_ids", [[0, 0], [0, 1]])
@pytest.mark.parametrize("precision", ["fp32", "fp64", "bf16"])
def test_multibranch_autocast_preserves_outputs_and_force_gradients(
    decoder, head_type, variance, dataset_ids, precision
):
    dtype = torch.float64 if precision == "fp64" else torch.float32
    torch.manual_seed(42)
    model = _DecoderHarness(head_type, 2, variance).to(dtype=dtype)
    data = SimpleNamespace(
        pos=torch.randn(5, 3, dtype=dtype, requires_grad=True),
        x=torch.ones(5, 1, dtype=dtype),
        batch=torch.tensor([0, 0, 1, 1, 1]),
        dataset_name=torch.tensor(dataset_ids).reshape(2, 1),
    )
    context = (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    with context:
        result = decoder(model, data)
        outputs, variances = result if variance else (result, None)
        energy = outputs[0]
        assert energy.dtype == dtype
        assert energy.shape == (2 if head_type == "graph" else 5, 1)
        assert torch.isfinite(energy).all()
        if variance:
            assert variances[0].dtype == dtype
            assert variances[0].shape == energy.shape
            assert torch.isfinite(variances[0]).all()
        forces = -torch.autograd.grad(energy.sum(), data.pos, create_graph=True)[0]
        loss = energy.square().mean() + forces.square().mean()
        if variance:
            loss = loss + variances[0].mean()
    assert torch.isfinite(forces).all()
    loss.backward()
    for dataset_id in set(dataset_ids):
        head = model.heads_NN[0][f"branch-{dataset_id}"]
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
