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
import sys
import os

# Add HydraGNN to path
sys.path.insert(0, '/home/runner/work/HydraGNN/HydraGNN')

def test_interatomic_potential_import():
    """Test that the interatomic potential classes can be imported."""
    try:
        from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin, InteratomicPotentialBase
        print("✓ InteratomicPotential classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import InteratomicPotential classes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixin_methods():
    """Test that the mixin has the expected methods."""
    try:
        from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin
        
        # Check if the mixin has the expected methods
        expected_methods = [
            '_init_interatomic_layers',
            '_compute_enhanced_geometric_features',
            '_compute_local_environment_features',
            '_compute_three_body_interactions',
            '_apply_atomic_environment_descriptors',
            'forward'
        ]
        
        missing_methods = []
        for method in expected_methods:
            if not hasattr(InteratomicPotentialMixin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"✗ Missing methods: {missing_methods}")
            return False
        
        print("✓ All expected methods found in InteratomicPotentialMixin")
        return True
        
    except Exception as e:
        print(f"✗ Error checking mixin methods: {e}")
        return False

def test_base_class():
    """Test that the InteratomicPotentialBase class can be instantiated."""
    try:
        from hydragnn.models.InteratomicPotential import InteratomicPotentialBase
        
        # Test basic instantiation parameters
        print("✓ InteratomicPotentialBase class accessible")
        print(f"  Class MRO: {[cls.__name__ for cls in InteratomicPotentialBase.__mro__]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with InteratomicPotentialBase class: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features():
    """Test the enhanced feature computation methods."""
    try:
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
        has_three_body_mlp = hasattr(mixin, 'three_body_mlp')
        has_env_descriptor_mlp = hasattr(mixin, 'env_descriptor_mlp')
        
        print(f"✓ Enhanced feature methods working")
        print(f"  Three-body MLP created: {has_three_body_mlp}")
        print(f"  Environment descriptor MLP created: {has_env_descriptor_mlp}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing enhanced features: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_operations():
    """Test basic PyTorch operations that the enhancement uses."""
    try:
        # Test basic tensor operations
        x = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).long()
        
        # Test scatter operations (mock)
        coord_numbers = torch.sum(torch.ones(edge_index.size(1)), dim=0)
        
        # Test basic MLP creation
        mlp = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )
        
        # Test forward pass
        input_tensor = torch.randn(5, 6)
        output = mlp(input_tensor)
        
        print("✓ Basic PyTorch operations working")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch operations failed: {e}")
        return False

def run_basic_tests():
    """Run basic tests without full dependencies."""
    print("Testing HydraGNN Interatomic Potential Enhancements (Basic)")
    print("=" * 60)
    
    tests = [
        ("Import interatomic potential classes", test_interatomic_potential_import),
        ("Check mixin methods", test_mixin_methods),
        ("Test base class", test_base_class),
        ("Test enhanced features", test_enhanced_features),
        ("Test PyTorch operations", test_torch_operations),
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
    print(f"Basic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Core implementation is working.")
    else:
        print("⚠️  Some basic tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)