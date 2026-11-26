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

from types import SimpleNamespace

import pytest
import torch
from mpi4py import MPI
from torch_geometric.data import Data

from hydragnn.config import parse_variable_schema
from hydragnn.preprocess import FieldNormalizer, create_dataloaders


def _schema(f_policy="none", v_policy="none", t_policy="none"):
    return parse_variable_schema(
        {
            "graph_type": "homogeneous",
            "inputs": [
                {"name": "pos", "role": "position", "level": "node", "dim": 3},
                {"name": "f", "level": "node", "dim": 1, "normalize": f_policy},
                {"name": "v", "level": "node", "dim": 2, "normalize": v_policy},
            ],
            "outputs": [
                {"name": "t", "level": "node", "dim": 1, "normalize": t_policy}
            ],
        }
    )


def _sample(seed, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    n = 5
    return Data(
        f=torch.randn(n, 1, generator=g, dtype=torch.float64) + offset,
        v=torch.randn(n, 2, generator=g, dtype=torch.float64)
        * torch.tensor([1.0, 10.0], dtype=torch.float64),
        t=torch.randn(n, 1, generator=g, dtype=torch.float64),
        pos=torch.zeros(n, 3, dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=n,
    )


def test_standard_variance_is_numerically_stable():
    schema = _schema(f_policy="standard")
    data = []
    for k in range(4):
        sample = _sample(k)
        sample.f = torch.tensor([[1e8 - 0.25], [1e8 + 0.25]] * 2, dtype=torch.float64)
        sample.num_nodes = 4
        data.append(sample)
    stats = FieldNormalizer.fit(data, schema).stats["f"]
    assert stats["shift"] == pytest.approx(1e8)
    assert stats["scale"] == pytest.approx(0.25, rel=1e-9)


def test_normalize_policy_validation():
    with pytest.raises(ValueError, match="normalize must be one of"):
        _schema(f_policy="log")
    with pytest.raises(ValueError, match="not supported for positions"):
        parse_variable_schema(
            {
                "graph_type": "homogeneous",
                "inputs": [
                    {
                        "name": "pos",
                        "role": "position",
                        "level": "node",
                        "dim": 3,
                        "normalize": "standard",
                    },
                    {"name": "f", "level": "node", "dim": 1},
                ],
                "outputs": [{"name": "t", "level": "node", "dim": 1}],
            }
        )
    assert _schema().inputs[1].normalize == "none"


@pytest.mark.parametrize("policy", ["minmax", "standard", "scale_only"])
def test_round_trip_and_statistics(policy):
    data = [_sample(i, offset=3.0) for i in range(4)]
    norm = FieldNormalizer.fit(data, _schema(f_policy=policy))
    f = torch.cat([d.f for d in data])
    out = torch.cat([norm.transform(d.clone()).f for d in data])
    if policy == "minmax":
        assert out.min().item() == pytest.approx(0.0, abs=1e-12)
        assert out.max().item() == pytest.approx(1.0, abs=1e-12)
    elif policy == "standard":
        assert out.mean().item() == pytest.approx(0.0, abs=1e-12)
        assert out.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-12)
    else:
        assert out.square().mean().item() == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(norm.inverse("f", out), f, atol=1e-12, rtol=0)
    restored = FieldNormalizer.from_state_dict(norm.state_dict())
    torch.testing.assert_close(restored.inverse("f", out), f, atol=1e-12, rtol=0)


def test_scale_only_preserves_sign_and_antisymmetry():
    data = [_sample(i) for i in range(3)]
    norm = FieldNormalizer.fit(data, _schema(t_policy="scale_only"))
    for d in data:
        d.t[1] = -d.t[0]
        out = norm.transform(d.clone()).t
        assert torch.equal(torch.sign(out), torch.sign(d.t))
        assert out[1].item() == -out[0].item()


def test_vector_field_uses_one_scale():
    data = [_sample(i) for i in range(4)]
    norm = FieldNormalizer.fit(data, _schema(v_policy="scale_only"))
    d = data[0]
    out = norm.transform(d.clone()).v
    ratio = d.v / out
    torch.testing.assert_close(ratio, torch.full_like(ratio, ratio[0, 0].item()))


def test_dataloaders_fit_on_training_split_only():
    schema = _schema(f_policy="standard", t_policy="scale_only")
    train = [_sample(i) for i in range(4)]
    val = [_sample(10 + i, offset=100.0) for i in range(2)]
    loaders = create_dataloaders(train, val, val, 2, variables=schema)
    norm = loaders[0].dataset.normalizer
    assert loaders[1].dataset.normalizer is norm
    expected = FieldNormalizer.fit(train, schema)
    assert norm.state_dict() == expected.state_dict()
    # Input normalization reaches x; output normalization reaches y.
    sample = loaders[0].dataset[0]
    torch.testing.assert_close(
        sample.x[:, :1], norm.transform(train[0].clone()).f, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        sample.y, norm.transform(train[0].clone()).t.reshape(-1, 1)
    )
    # Source data is not mutated.
    assert train[0].f.mean().abs() < 10


def _spec(name, normalize):
    return SimpleNamespace(name=name, normalize=normalize, node_type=None)


def test_shared_name_policies():
    schema = SimpleNamespace(
        inputs=(_spec("f", "standard"),), outputs=(_spec("f", "none"),)
    )
    with pytest.raises(ValueError, match="conflicting normalize policies"):
        FieldNormalizer.specs(schema)
    schema = SimpleNamespace(
        inputs=(_spec("f", "standard"),), outputs=(_spec("f", "standard"),)
    )
    assert [s.name for s in FieldNormalizer.specs(schema)] == ["f"]


def test_empty_tensors_and_mpi_group():
    data = [_sample(i) for i in range(3)]
    empty = _sample(9)
    empty.f = empty.f[:0]
    schema = _schema(f_policy="minmax")
    norm = FieldNormalizer.fit([*data, empty], schema, group=MPI.COMM_SELF)
    assert norm.state_dict() == FieldNormalizer.fit(data, schema).state_dict()


def test_no_normalizer_without_policies():
    data = [_sample(i) for i in range(2)]
    loaders = create_dataloaders(data, data, data, 2, variables=_schema())
    assert loaders[0].dataset.normalizer is None
