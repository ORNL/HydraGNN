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

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.models.Base import Base
from hydragnn.models.CGCNNStack import CGCNNStack
from hydragnn.models.DIMEStack import DIMEStack
from hydragnn.models.EGCLStack import EGCLStack
from hydragnn.models.GATStack import GATStack
from hydragnn.models.GINStack import GINStack
from hydragnn.models.MACEStack import MACEStack
from hydragnn.models.MFCStack import MFCStack
from hydragnn.models.PAINNStack import PAINNStack
from hydragnn.models.PNAEqStack import PNAEqStack
from hydragnn.models.PNAPlusStack import PNAPlusStack
from hydragnn.models.PNAStack import PNAStack
from hydragnn.models.SAGEStack import SAGEStack
from hydragnn.models.SCFStack import SCFStack
from hydragnn.models.create import create_model
from hydragnn.utils.model.model import update_multibranch_heads

GENERIC_STACKS = (
    PAINNStack,
    PNAEqStack,
    DIMEStack,
    SCFStack,
    EGCLStack,
    PNAPlusStack,
    PNAStack,
    GATStack,
    GINStack,
    SAGEStack,
    CGCNNStack,
    MFCStack,
)
NATIVE_STACKS = (MACEStack,)


def _model(atomistic):
    heads = update_multibranch_heads(
        {
            "node": {
                "num_sharedlayers": 1,
                "dim_sharedlayers": 8,
                "num_headlayers": 1,
                "dim_headlayers": [8],
                "type": "mlp",
            }
        }
    )
    return create_model(
        mpnn_type="EGNN",
        input_dim=1,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads=heads,
        activation_function="elu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        num_nodes=3,
        enable_interatomic_potential=atomistic,
        energy_weight=0.1,
        force_weight=1.0,
        use_gpu=False,
    )


def _data(atomic_numbers=None):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]],
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    values = torch.tensor([[7.0], [1.0], [8.0]])
    return Data(
        x=values,
        atomic_numbers=atomic_numbers,
        pos=pos,
        edge_index=edge_index,
        edge_shifts=torch.zeros(edge_index.shape[1], 3),
        batch=torch.zeros(3, dtype=torch.long),
        energy=torch.zeros(1),
        forces=torch.zeros_like(pos),
    )


def pytest_stack_species_capabilities_are_explicit():
    assert all(not stack.uses_native_species_encoder for stack in GENERIC_STACKS)
    assert all(stack.uses_native_species_encoder for stack in NATIVE_STACKS)


def pytest_non_atomistic_generic_input_is_unchanged():
    model = _model(atomistic=False)
    data = _data()
    features, _, _ = model._embedding(data)

    assert model.species_embedding is None
    assert torch.equal(features, data.x)


def pytest_atomistic_generic_input_is_a_hidden_width_species_embedding():
    model = _model(atomistic=True)
    data = _data(torch.tensor([7, 1, 8], dtype=torch.long))
    features, _, _ = model.model._embedding(data)

    assert isinstance(model.model.species_embedding, torch.nn.Embedding)
    assert model.model.species_embedding.num_embeddings == 119
    assert features.shape == (3, 8)
    assert torch.equal(features, model.model.species_embedding(data.atomic_numbers))


def pytest_native_capability_prevents_double_embedding():
    for stack_type in NATIVE_STACKS:
        stack = stack_type.__new__(stack_type)
        torch.nn.Module.__init__(stack)
        stack.hidden_dim = 8
        stack.species_embedding = None
        Base.configure_atomistic_species_encoding(stack, True)
        assert stack.enable_atomistic_species_encoding is False
        assert stack.species_embedding is None


def pytest_atomic_number_boundaries_are_supported():
    model = _model(atomistic=True).model
    data = _data(torch.tensor([1, 118, 1], dtype=torch.long))
    features = model._input_node_features(data)
    assert features.shape == (3, model.hidden_dim)


@pytest.mark.parametrize(
    "atomic_numbers,error",
    [
        (torch.tensor([0, 1, 2]), ValueError),
        (torch.tensor([1, 2, 119]), ValueError),
        (torch.tensor([1.0, 2.0, 3.0]), TypeError),
    ],
)
def pytest_invalid_atomic_numbers_are_rejected(atomic_numbers, error):
    model = _model(atomistic=True).model
    with pytest.raises(error):
        model._input_node_features(_data(atomic_numbers))


def pytest_legacy_scalar_atomic_number_fallback_is_supported():
    model = _model(atomistic=True).model
    data = _data()
    del data.atomic_numbers
    with pytest.warns(DeprecationWarning, match="data.atomic_numbers"):
        features = model._input_node_features(data)
    assert features.shape == (3, model.hidden_dim)


def pytest_atomic_number_count_must_match_nodes():
    model = _model(atomistic=True).model
    with pytest.raises(ValueError, match="one value per node"):
        model._input_node_features(_data(torch.tensor([1, 8])))


def pytest_distinct_species_have_distinct_learned_representations():
    model = _model(atomistic=True).model
    data = _data(torch.tensor([1, 2, 3]))
    features = model._input_node_features(data)
    assert torch.unique(features, dim=0).shape[0] == 3


def pytest_atomistic_forward_backward_force_path():
    model = _model(atomistic=True)
    data = _data(torch.tensor([7, 1, 8], dtype=torch.long))
    predictions = model(data)
    energy = predictions[0].sum()
    forces = -torch.autograd.grad(energy, data.pos, create_graph=True)[0]

    assert forces.shape == data.pos.shape
    assert torch.isfinite(forces).all()
    forces.square().sum().backward()
    assert model.model.species_embedding.weight.grad is not None
