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
Direct test of the InteratomicPotential module.
This test validates the implementation by importing it directly.
"""

import torch
import os
import pytest

@pytest.mark.mpi_skip()
def pytest_direct_import():
    """Test importing the InteratomicPotential module directly."""
    # Import specific files needed for the InteratomicPotential
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin, InteratomicPotentialBase
    # If we reach this point, the import was successful

@pytest.mark.mpi_skip()
def pytest_mixin_instantiation():
    """Test creating an instance of the mixin."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin
    
    class TestMixin(InteratomicPotentialMixin):
        def __init__(self):
            self.use_enhanced_geometry = True
            self.use_three_body_interactions = True
            self.use_atomic_environment_descriptors = True
            self.hidden_dim = 32
            self.activation_function = torch.nn.ReLU()
            
    mixin = TestMixin()
    
    # Test if methods exist
    methods = ['_init_interatomic_layers', '_compute_enhanced_geometric_features', 
              '_compute_local_environment_features', '_compute_three_body_interactions',
              '_apply_atomic_environment_descriptors']
    
    missing_methods = []
    for method in methods:
        if not hasattr(mixin, method):
            missing_methods.append(method)
    
    assert not missing_methods, f"Missing methods: {missing_methods}"

@pytest.mark.mpi_skip()
def pytest_layer_initialization():
    """Test the layer initialization functionality."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin
    
    class TestMixin(InteratomicPotentialMixin):
        def __init__(self):
            self.use_enhanced_geometry = True
            self.use_three_body_interactions = True
            self.use_atomic_environment_descriptors = True
            self.hidden_dim = 32
            self.activation_function = torch.nn.ReLU()
            self._init_interatomic_layers()
            
    mixin = TestMixin()
    
    # Check if layers were created
    has_three_body = hasattr(mixin, 'three_body_mlp')
    has_env_desc = hasattr(mixin, 'env_descriptor_mlp')
    
    # The layers should exist based on the enabled features
    assert has_three_body, "Three-body MLP should be created when use_three_body_interactions=True"
    assert has_env_desc, "Environment descriptor MLP should be created when use_atomic_environment_descriptors=True"

@pytest.mark.mpi_skip()
def pytest_geometric_features():
    """Test the enhanced geometric feature computation."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin
    
    class TestMixin(InteratomicPotentialMixin):
        def __init__(self):
            self.use_enhanced_geometry = True
            self.use_three_body_interactions = True
            self.use_atomic_environment_descriptors = True
            self.hidden_dim = 32
            self.activation_function = torch.nn.ReLU()
            self._init_interatomic_layers()
    
    mixin = TestMixin()
    
    # Create mock data
    class MockData:
        def __init__(self):
            self.pos = torch.randn(5, 3)
            self.edge_shifts = torch.zeros(6, 3)
    
    data = MockData()
    conv_args = {"edge_index": torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 4]]).long()}
    
    # Test local environment features
    edge_vectors = torch.randn(6, 3)
    edge_lengths = torch.randn(6, 1)
    
    env_features = mixin._compute_local_environment_features(
        data.pos, conv_args["edge_index"], edge_vectors, edge_lengths
    )
    
    # Verify the output shape and type
    assert isinstance(env_features, torch.Tensor), "Environment features should be a tensor"
    assert env_features.shape[0] == data.pos.shape[0], "Environment features should have same batch size as nodes"

@pytest.mark.mpi_skip()
def pytest_enhanced_creation():
    """Test that InteratomicPotentialBase can be accessed."""
    from hydragnn.models.InteratomicPotential import InteratomicPotentialBase
    
    # Verify the class is accessible and has expected attributes
    assert hasattr(InteratomicPotentialBase, '__name__'), "InteratomicPotentialBase should have a name"
    assert hasattr(InteratomicPotentialBase, '__bases__'), "InteratomicPotentialBase should have base classes"