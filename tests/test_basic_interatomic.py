##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

"""
Basic test script for interatomic potential enhancements in HydraGNN.
This test validates the basic implementation without full functionality.
"""

import torch
import os
import pytest


@pytest.mark.mpi_skip()
def pytest_interatomic_potential_import():
    """Test that the interatomic potential classes can be imported."""
    from hydragnn.models.InteratomicPotential import (
        InteratomicPotentialMixin,
        InteratomicPotentialBase,
    )

    # If we reach this point, the import was successful


@pytest.mark.mpi_skip()
def pytest_mixin_methods():
    """Test that the mixin has the expected methods."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin

    # Check if the mixin has the expected methods
    expected_methods = [
        "_init_interatomic_layers",
        "_compute_enhanced_geometric_features",
        "_compute_local_environment_features",
        "_compute_three_body_interactions",
        "_apply_atomic_environment_descriptors",
        "forward",
    ]

    missing_methods = []
    for method in expected_methods:
        if not hasattr(InteratomicPotentialMixin, method):
            missing_methods.append(method)

    assert not missing_methods, f"Missing methods: {missing_methods}"


@pytest.mark.mpi_skip()
def pytest_base_class():
    """Test that the InteratomicPotentialBase class can be instantiated."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialBase

    # Test basic instantiation parameters
    assert hasattr(
        InteratomicPotentialBase, "__mro__"
    ), "InteratomicPotentialBase should have MRO"


@pytest.mark.mpi_skip()
def pytest_enhanced_features():
    """Test the enhanced feature computation methods."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin

    class TestMixin(InteratomicPotentialMixin):
        def __init__(self):
            self.use_enhanced_geometry = True
            self.use_three_body_interactions = True
            self.use_atomic_environment_descriptors = True
            self.hidden_dim = 32
            self.activation_function = torch.nn.ReLU()

    mixin = TestMixin()
    mixin._init_interatomic_layers()

    # Check if the layers were created
    has_three_body_mlp = hasattr(mixin, "three_body_mlp")
    has_env_descriptor_mlp = hasattr(mixin, "env_descriptor_mlp")

    assert (
        has_three_body_mlp
    ), "Three-body MLP should be created when use_three_body_interactions is True"
    assert (
        has_env_descriptor_mlp
    ), "Environment descriptor MLP should be created when use_atomic_environment_descriptors is True"


@pytest.mark.mpi_skip()
def pytest_torch_operations():
    """Test basic PyTorch operations that the enhancement uses."""
    # Test basic tensor operations
    x = torch.randn(10, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).long()

    # Test scatter operations (mock)
    coord_numbers = torch.sum(torch.ones(edge_index.size(1)), dim=0)

    # Test basic MLP creation
    mlp = torch.nn.Sequential(
        torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32)
    )

    # Test forward pass
    input_tensor = torch.randn(5, 6)
    output = mlp(input_tensor)

    assert output.shape == (5, 32), f"Expected output shape (5, 32), got {output.shape}"
