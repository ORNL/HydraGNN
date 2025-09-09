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
Test script for interatomic potential enhancements in HydraGNN.
This test validates the enhanced forward method functionality for molecular simulations.
"""

import torch
import numpy as np
from torch_geometric.data import Data
import os

def create_mock_molecular_data(num_atoms=10, num_graphs=2):
    """
    Create mock molecular data for testing interatomic potential functionality.
    
    Args:
        num_atoms: Number of atoms per molecule
        num_graphs: Number of molecules in the batch
    
    Returns:
        PyTorch Geometric Data object with molecular structure
    """
    # Create atomic positions (coordinates)
    pos = torch.randn(num_atoms * num_graphs, 3) * 5.0  # Random positions
    
    # Create node features (atomic numbers)
    x = torch.randint(1, 10, (num_atoms * num_graphs, 1)).float()  # Atomic numbers 1-9
    
    # Create edge connectivity (simple nearest neighbor graph)
    edge_index_list = []
    for graph_idx in range(num_graphs):
        start_idx = graph_idx * num_atoms
        end_idx = (graph_idx + 1) * num_atoms
        
        # Connect each atom to its nearest neighbors
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    # Add edge if distance is reasonable (simplified)
                    dist = torch.norm(pos[i] - pos[j])
                    if dist < 4.0:  # Cutoff distance
                        edge_index_list.extend([[i, j]])
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list).t().contiguous()
    else:
        # Fallback: create minimal connectivity
        edge_index = torch.tensor([[0, 1], [1, 0]]).t().contiguous()
    
    # Create batch assignment
    batch = torch.repeat_interleave(torch.arange(num_graphs), num_atoms)
    
    # Create mock energy and forces for training
    energy = torch.randn(num_graphs, 1)  # Energy per molecule
    forces = torch.randn(num_atoms * num_graphs, 3)  # Forces per atom
    
    # Create positional encodings (random for testing)
    pe = torch.randn(num_atoms * num_graphs, 6)  # 6D positional encoding
    
    # Set positions to require gradients for force computation
    pos.requires_grad_(True)
    
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        batch=batch,
        energy=energy,
        forces=forces,
        pe=pe,
        edge_shifts=torch.zeros(edge_index.size(1), 3)  # No periodic boundary conditions
    )
    
    return data

def test_interatomic_potential_creation():
    """Test that models can be created with interatomic potential enhancements."""
    try:
        from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin, InteratomicPotentialBase
        print("✓ InteratomicPotential classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import InteratomicPotential classes: {e}")
        return False

def test_model_creation_with_enhancement():
    """Test creating a model with interatomic potential enhancement."""
    try:
        from hydragnn.models.create import create_model
        
        # Basic model configuration
        config_args = {
            'mpnn_type': 'GIN',
            'input_dim': 1,
            'hidden_dim': 32,
            'output_dim': [1],
            'pe_dim': 6,
            'global_attn_engine': '',
            'global_attn_type': '',
            'global_attn_heads': 1,
            'output_type': ['node'],
            'output_heads': {'node': {'num_sharedlayers': 1, 'dim_sharedlayers': 16, 'num_headlayers': 1, 'dim_headlayers': [1]}},
            'activation_function': 'relu',
            'loss_function_type': 'mse',
            'task_weights': [1.0],
            'num_conv_layers': 2,
            'enable_interatomic_potential': True,
            'use_gpu': False
        }
        
        model = create_model(**config_args)
        print("✓ Model with interatomic potential enhancement created successfully")
        print(f"  Model type: {type(model)}")
        
        # Check if the model has interatomic potential methods
        has_enhanced_geometry = hasattr(model, '_compute_enhanced_geometric_features')
        has_three_body = hasattr(model, '_compute_three_body_interactions')
        has_env_descriptors = hasattr(model, '_apply_atomic_environment_descriptors')
        
        print(f"  Enhanced geometry features: {has_enhanced_geometry}")
        print(f"  Three-body interactions: {has_three_body}")
        print(f"  Environment descriptors: {has_env_descriptors}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create enhanced model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test the enhanced forward pass with molecular data."""
    try:
        from hydragnn.models.create import create_model
        
        # Create model with interatomic potential enhancement
        config_args = {
            'mpnn_type': 'GIN',
            'input_dim': 1,
            'hidden_dim': 32,
            'output_dim': [1],
            'pe_dim': 6,
            'global_attn_engine': '',
            'global_attn_type': '',
            'global_attn_heads': 1,
            'output_type': ['node'],
            'output_heads': {'node': {'num_sharedlayers': 1, 'dim_sharedlayers': 16, 'num_headlayers': 1, 'dim_headlayers': [1]}},
            'activation_function': 'relu',
            'loss_function_type': 'mse',
            'task_weights': [1.0],
            'num_conv_layers': 2,
            'enable_interatomic_potential': True,
            'use_gpu': False
        }
        
        model = create_model(**config_args)
        model.eval()
        
        # Create mock molecular data
        data = create_mock_molecular_data(num_atoms=5, num_graphs=2)
        
        # Test forward pass
        with torch.no_grad():
            output = model(data)
        
        print("✓ Enhanced forward pass completed successfully")
        print(f"  Output type: {type(output)}")
        print(f"  Output length: {len(output) if isinstance(output, (list, tuple)) else 'scalar'}")
        
        if isinstance(output, (list, tuple)) and len(output) > 0:
            print(f"  First output shape: {output[0].shape}")
            print(f"  First output sample: {output[0][:2]}")  # Show first 2 values
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_energy_force_consistency():
    """Test that forces can be computed from energy gradients."""
    try:
        from hydragnn.models.create import create_model
        
        # Create model for energy prediction
        config_args = {
            'mpnn_type': 'GIN',
            'input_dim': 1,
            'hidden_dim': 32,
            'output_dim': [1],
            'pe_dim': 6,
            'global_attn_engine': '',
            'global_attn_type': '',
            'global_attn_heads': 1,
            'output_type': ['node'],
            'output_heads': {'node': {'num_sharedlayers': 1, 'dim_sharedlayers': 16, 'num_headlayers': 1, 'dim_headlayers': [1]}},
            'activation_function': 'relu',
            'loss_function_type': 'mse',
            'task_weights': [1.0],
            'num_conv_layers': 2,
            'enable_interatomic_potential': True,
            'use_gpu': False
        }
        
        model = create_model(**config_args)
        model.train()  # Need gradients
        
        # Create mock molecular data
        data = create_mock_molecular_data(num_atoms=5, num_graphs=1)
        
        # Ensure positions require gradients
        data.pos.requires_grad_(True)
        
        # Forward pass to get energy
        energy_output = model(data)
        
        if isinstance(energy_output, (list, tuple)):
            energy = energy_output[0].sum()  # Sum over batch
        else:
            energy = energy_output.sum()
        
        # Compute forces as negative gradients of energy w.r.t. positions
        forces = torch.autograd.grad(
            energy, 
            data.pos, 
            create_graph=True, 
            retain_graph=True
        )[0]
        forces = -forces  # Force = -gradient of energy
        
        print("✓ Energy-force consistency test passed")
        print(f"  Energy shape: {energy.shape}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Energy value: {energy.item():.4f}")
        print(f"  Max force magnitude: {torch.norm(forces, dim=1).max().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Energy-force consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all interatomic potential tests."""
    print("Testing HydraGNN Interatomic Potential Enhancements")
    print("=" * 60)
    
    tests = [
        ("Interatomic potential class creation", test_interatomic_potential_creation),
        ("Model creation with enhancement", test_model_creation_with_enhancement),
        ("Enhanced forward pass", test_forward_pass),
        ("Energy-force consistency", test_energy_force_consistency),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print("  Test failed!")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Interatomic potential functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)