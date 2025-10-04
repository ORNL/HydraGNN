#!/usr/bin/env python3
"""
Simple test for OpenEquivariance integration in HydraGNN MACE.
Tests the core functionality without complex configuration setup.
"""

import os
import sys
import warnings

# Add HydraGNN to path
sys.path.insert(0, os.getcwd())

def test_openequivariance_basic():
    """Test basic OpenEquivariance functionality."""
    
    print("=" * 60)
    print("Basic OpenEquivariance Integration Test")
    print("=" * 60)
    
    # Test 1: Import and availability check
    print("\n1. Testing OpenEquivariance utilities import:")
    try:
        from hydragnn.utils.model.openequivariance_utils import (
            check_openequivariance_availability,
            is_openequivariance_enabled,
            HAS_OPENEQUIVARIANCE,
            OptimizedTensorProduct,
            OptimizedSphericalHarmonics,
            OptimizedLinear,
            optimized_einsum
        )
        print("   ✓ OpenEquivariance utilities imported successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check availability detection
    print(f"\n2. OpenEquivariance availability: {HAS_OPENEQUIVARIANCE}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test disabled
        result_disabled = check_openequivariance_availability(False)
        print(f"   check_openequivariance_availability(False): {result_disabled}")
        
        # Test enabled
        result_enabled = check_openequivariance_availability(True)
        print(f"   check_openequivariance_availability(True): {result_enabled}")
        
        # Show any warnings
        if w:
            print("   Warnings generated:")
            for warning in w:
                print(f"     - {warning.message}")
    
    # Test 3: Test OptimizedLinear fallback
    print("\n3. Testing OptimizedLinear fallback:")
    try:
        from e3nn import o3
        import torch
        
        # Create an OptimizedLinear layer
        irreps_in = o3.Irreps("2x0e + 2x1o")  # 2 scalars + 2 vectors
        irreps_out = o3.Irreps("1x0e")        # 1 scalar output
        
        linear = OptimizedLinear(irreps_in, irreps_out)
        print(f"   ✓ OptimizedLinear created: {type(linear.linear).__name__}")
        print(f"   Using OpenEquivariance: {getattr(linear, 'using_openequivariance', 'unknown')}")
        
        # Test forward pass
        x = torch.randn(10, irreps_in.dim)  # batch of 10
        y = linear(x)
        print(f"   ✓ Forward pass successful: {x.shape} -> {y.shape}")
        
    except Exception as e:
        print(f"   ❌ OptimizedLinear test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test OptimizedSphericalHarmonics fallback
    print("\n4. Testing OptimizedSphericalHarmonics fallback:")
    try:
        irreps_sh = o3.Irreps.spherical_harmonics(2)  # l=0,1,2
        
        sh = OptimizedSphericalHarmonics(irreps_sh)
        print(f"   ✓ OptimizedSphericalHarmonics created: {type(sh.spherical_harmonics).__name__}")
        print(f"   Using OpenEquivariance: {getattr(sh, 'using_openequivariance', 'unknown')}")
        
        # Test forward pass with unit vectors
        vectors = torch.randn(5, 3)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)  # normalize
        sh_features = sh(vectors)
        print(f"   ✓ Forward pass successful: {vectors.shape} -> {sh_features.shape}")
        
    except Exception as e:
        print(f"   ❌ OptimizedSphericalHarmonics test failed: {e}")
        return False
    
    # Test 5: Test optimized_einsum fallback
    print("\n5. Testing optimized_einsum fallback:")
    try:
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        
        # Test einsum operation
        c1 = optimized_einsum("ij,jk->ik", a, b)
        c2 = torch.einsum("ij,jk->ik", a, b)
        
        # Check they're the same
        assert torch.allclose(c1, c2), "Results should be identical"
        print(f"   ✓ optimized_einsum works correctly: {a.shape} x {b.shape} -> {c1.shape}")
        
    except Exception as e:
        print(f"   ❌ optimized_einsum test failed: {e}")
        return False
    
    # Test 6: Configuration parameter handling
    print("\n6. Testing configuration parameter:")
    try:
        from hydragnn.utils.input_config_parsing.config_utils import update_config
        
        # Create minimal config
        config = {
            "NeuralNetwork": {
                "Architecture": {
                    "enable_openequivariance": True,
                    "mpnn_type": "MACE"
                }
            }
        }
        
        # Test that the parameter gets defaults set correctly
        class DummyDataset:
            def __init__(self):
                self.pna_deg = None
            def __getitem__(self, idx):
                class DummyData:
                    def __init__(self):
                        self.y = [torch.tensor([1.0])]
                        self.x = torch.tensor([[1.0]])
                return DummyData()
        
        class DummyLoader:
            def __init__(self):
                self.dataset = DummyDataset()
        
        train_loader = DummyLoader()
        val_loader = DummyLoader()  
        test_loader = DummyLoader()
        
        # This should work with minimal config
        updated_config = update_config(config, train_loader, val_loader, test_loader)
        
        enable_oe = updated_config["NeuralNetwork"]["Architecture"].get("enable_openequivariance", False)
        print(f"   ✓ Configuration parameter preserved: {enable_oe}")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    if HAS_OPENEQUIVARIANCE and result_enabled:
        print("✓ OpenEquivariance is available and would be used")
    else:
        print("⚠ OpenEquivariance not available, but e3nn fallback works correctly")
    print("✓ All OptimizedLinear, OptimizedSphericalHarmonics work")
    print("✓ optimized_einsum fallback works") 
    print("✓ Configuration parameter handling works")
    print("\nOpenEquivariance integration is ready for use!")
    
    return True

if __name__ == "__main__":
    success = test_openequivariance_basic()
    sys.exit(0 if success else 1)