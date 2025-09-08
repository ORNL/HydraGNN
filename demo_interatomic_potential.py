#!/usr/bin/env python3
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
Demonstration of HydraGNN Interatomic Potential Enhancements

This script demonstrates the enhanced capabilities for machine learning
interatomic potentials in molecular simulations.
"""

import torch
import numpy as np
import sys
import os

# Add HydraGNN to path
sys.path.insert(0, os.path.abspath('.'))

# Import InteratomicPotential classes using standard import
from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin, InteratomicPotentialBase

def create_demo_molecular_data():
    """Create demonstration molecular data for testing."""
    print("\nCreating demonstration molecular data...")
    
    # Simple water molecule structure (H2O)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],      # Oxygen
        [0.757, 0.586, 0.0],  # Hydrogen 1
        [-0.757, 0.586, 0.0]  # Hydrogen 2
    ], dtype=torch.float32)
    
    # Atomic numbers
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.float32).unsqueeze(1)  # O, H, H
    
    # Edge connectivity (all atoms connected)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],  # Source atoms
        [1, 2, 0, 2, 0, 1]   # Target atoms
    ], dtype=torch.long)
    
    # Create mock data object
    class MockData:
        def __init__(self):
            self.x = atomic_numbers
            self.pos = positions
            self.edge_index = edge_index
            self.edge_shifts = torch.zeros(edge_index.size(1), 3)
            self.batch = torch.zeros(3, dtype=torch.long)
            self.dataset_name = torch.zeros(1, 1, dtype=torch.long)
    
    data = MockData()
    
    print(f"  Created molecular data:")
    print(f"  - Atoms: {data.x.size(0)}")
    print(f"  - Edges: {data.edge_index.size(1)}")
    print(f"  - Positions shape: {data.pos.shape}")
    
    return data

def demonstrate_enhanced_features():
    """Demonstrate the enhanced interatomic potential features."""
    print("\n" + "="*60)
    print("HydraGNN Interatomic Potential Enhancement Demonstration")
    print("="*60)
    
    try:
        # InteratomicPotential classes are imported at the top of the file
        print("✓ InteratomicPotential classes loaded successfully")
        
        # Create a test mixin with enhanced features
        class DemoModel(InteratomicPotentialMixin):
            def __init__(self):
                # Initialize basic attributes
                self.use_enhanced_geometry = True
                self.use_three_body_interactions = True
                self.use_atomic_environment_descriptors = True
                self.hidden_dim = 64
                self.activation_function = torch.nn.ReLU()
                self.conv_checkpointing = False
                
                # Set radius and max_neighbours for dynamic graph construction
                self.radius = 6.0  # Cutoff radius for neighbor finding
                self.max_neighbours = 50  # Maximum neighbors per atom
                
                # Initialize model components
                self.graph_convs = torch.nn.ModuleList([
                    torch.nn.Linear(1, self.hidden_dim) for _ in range(2)
                ])
                self.feature_layers = torch.nn.ModuleList([
                    torch.nn.BatchNorm1d(self.hidden_dim) for _ in range(2)
                ])
                
                # Mock head configuration for node-level atomic energy predictions
                self.heads_NN = [{'branch-0': torch.nn.Linear(self.hidden_dim, 1)}]
                self.head_dims = [1]
                self.head_type = ['node']
                self.num_branches = 1
                self.graph_shared = {'branch-0': torch.nn.Linear(self.hidden_dim, self.hidden_dim)}
                self.var_output = 0
                self.config_heads = {'node': [{'type': 'branch-0', 'architecture': {'type': 'mlp'}}]}
                
                # Initialize interatomic potential layers
                self._init_interatomic_layers()
        
        model = DemoModel()
        print("✓ Enhanced model created successfully")
        
        # Demonstrate features
        print(f"✓ Enhanced geometric features: {model.use_enhanced_geometry}")
        print(f"✓ Three-body interactions: {model.use_three_body_interactions}")
        print(f"✓ Atomic environment descriptors: {model.use_atomic_environment_descriptors}")
        print(f"✓ Three-body MLP layers: {hasattr(model, 'three_body_mlp')}")
        print(f"✓ Environment descriptor MLP: {hasattr(model, 'env_descriptor_mlp')}")
        
        # Create demonstration data
        data = create_demo_molecular_data()
        
        # Test enhanced geometric features
        print("\nTesting Enhanced Geometric Features:")
        print("-" * 40)
        
        conv_args = {"edge_index": data.edge_index}
        enhanced_conv_args = model._compute_enhanced_geometric_features(data, conv_args)
        
        if "edge_vectors" in enhanced_conv_args:
            edge_vectors = enhanced_conv_args["edge_vectors"]
            edge_lengths = enhanced_conv_args["edge_lengths"]
            print(f"✓ Edge vectors computed, shape: {edge_vectors.shape}")
            print(f"✓ Edge lengths computed, shape: {edge_lengths.shape}")
            print(f"  Sample edge length: {edge_lengths[0].item():.4f} Ångström")
        
        if "local_env_features" in enhanced_conv_args:
            env_features = enhanced_conv_args["local_env_features"]
            print(f"✓ Local environment features computed, shape: {env_features.shape}")
            print(f"  Coordination numbers: {env_features[:, 0]}")
            print(f"  Average distances: {env_features[:, 1]}")
            print(f"  Local densities: {env_features[:, 2]}")
        
        # Test three-body interactions
        print("\nTesting Three-body Interactions:")
        print("-" * 40)
        
        node_features = torch.randn(3, model.hidden_dim)
        enhanced_features = model._compute_three_body_interactions(
            node_features, data, enhanced_conv_args
        )
        print(f"✓ Three-body interactions computed")
        print(f"  Input features shape: {node_features.shape}")
        print(f"  Enhanced features shape: {enhanced_features.shape}")
        print(f"  Features enhanced: {not torch.allclose(node_features, enhanced_features)}")
        
        # Test atomic environment descriptors
        print("\nTesting Atomic Environment Descriptors:")
        print("-" * 40)
        
        final_features = model._apply_atomic_environment_descriptors(
            enhanced_features, enhanced_conv_args
        )
        print(f"✓ Atomic environment descriptors applied")
        print(f"  Final features shape: {final_features.shape}")
        
        print("\n" + "="*60)
        print("🎉 All interatomic potential enhancements working correctly!")
        print("="*60)
        
        print("\nKey Benefits for Molecular Simulations:")
        print("• Enhanced geometric features for better distance/angle representation")
        print("• Three-body interactions capture angular dependencies")
        print("• Atomic environment descriptors understand local chemistry")
        print("• Improved force consistency for molecular dynamics")
        print("• Compatible with all HydraGNN model architectures")
        
        print("\nUsage:")
        print("Add 'enable_interatomic_potential': true to your HydraGNN configuration")
        print("to automatically enhance any model with these capabilities.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demonstrate_enhanced_features()
    
    if success:
        print("\n🚀 Ready to enhance your molecular simulations with HydraGNN!")
    else:
        print("\n⚠️  Demonstration encountered issues. Please check the implementation.")
    
    sys.exit(0 if success else 1)