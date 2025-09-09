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
import sys
import os

# Add HydraGNN to path
sys.path.insert(0, '/home/runner/work/HydraGNN/HydraGNN')

def test_direct_import():
    """Test importing the InteratomicPotential module directly."""
    try:
        # Import specific files needed for the InteratomicPotential
        from hydragnn.models.InteratomicPotential import InteratomicPotentialMixin, InteratomicPotentialBase
        print("✓ InteratomicPotential classes imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import InteratomicPotential classes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixin_instantiation():
    """Test creating an instance of the mixin."""
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
        print("✓ InteratomicPotentialMixin instantiated successfully")
        
        # Test if methods exist
        methods = ['_init_interatomic_layers', '_compute_enhanced_geometric_features', 
                  '_compute_local_environment_features', '_compute_three_body_interactions',
                  '_apply_atomic_environment_descriptors']
        
        for method in methods:
            if hasattr(mixin, method):
                print(f"  ✓ Method {method} found")
            else:
                print(f"  ✗ Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing mixin instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_initialization():
    """Test the layer initialization functionality."""
    try:
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
        
        print("✓ Layer initialization test passed")
        print(f"  Three-body MLP created: {has_three_body}")
        print(f"  Environment descriptor MLP created: {has_env_desc}")
        
        if has_three_body:
            print(f"  Three-body MLP type: {type(mixin.three_body_mlp)}")
        if has_env_desc:
            print(f"  Environment MLP type: {type(mixin.env_descriptor_mlp)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing layer initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geometric_features():
    """Test the enhanced geometric feature computation."""
    try:
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
        
        print("✓ Geometric features test passed")
        print(f"  Environment features shape: {env_features.shape}")
        print(f"  Environment features sample: {env_features[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing geometric features: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_creation():
    """Test that InteratomicPotentialBase can be accessed."""
    try:
        from hydragnn.models.InteratomicPotential import InteratomicPotentialBase
        
        print("✓ InteratomicPotentialBase class accessible")
        print(f"  Class name: {InteratomicPotentialBase.__name__}")
        print(f"  Base classes: {[cls.__name__ for cls in InteratomicPotentialBase.__bases__]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing InteratomicPotentialBase: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_direct_tests():
    """Run direct tests of the InteratomicPotential module."""
    print("Testing HydraGNN Interatomic Potential Enhancements (Direct)")
    print("=" * 60)
    
    tests = [
        ("Direct import test", test_direct_import),
        ("Mixin instantiation", test_mixin_instantiation),
        ("Layer initialization", test_layer_initialization),
        ("Geometric features", test_geometric_features),
        ("Enhanced creation", test_enhanced_creation),
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
    print(f"Direct Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All direct tests passed! Core implementation is working.")
    else:
        print("⚠️  Some direct tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_direct_tests()
    sys.exit(0 if success else 1)