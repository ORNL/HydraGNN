#!/usr/bin/env python3
"""
Simple test script for OpenEquivariance integration.
"""

import sys
import torch
from e3nn import o3

# Add the current directory to path
sys.path.insert(0, '.')

def test_compatibility_module():
    """Test basic functionality of the compatibility module."""
    print("Testing OpenEquivariance compatibility module...")
    
    try:
        from hydragnn.utils.model.equivariance_compat import (
            get_backend_info, 
            is_openequivariance_available,
            TensorProduct
        )
        print("✓ Successfully imported compatibility module")
    except ImportError as e:
        print(f"✗ Failed to import compatibility module: {e}")
        return False
    
    # Test backend detection
    print(f"OpenEquivariance available: {is_openequivariance_available()}")
    backend_info = get_backend_info()
    print(f"Backend info: {backend_info}")
    
    # Test tensor product creation
    irreps_in1 = o3.Irreps("1x1e")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x2e")
    
    try:
        tp = TensorProduct(
            irreps_in1, irreps_in2, irreps_out,
            shared_weights=False,
            internal_weights=False
        )
        print(f"✓ Successfully created TensorProduct: {tp}")
    except Exception as e:
        print(f"✗ Failed to create TensorProduct: {e}")
        return False
    
    # Test forward pass
    try:
        batch_size = 2
        x1 = torch.randn(batch_size, irreps_in1.dim)
        x2 = torch.randn(batch_size, irreps_in2.dim)
        weight = torch.randn(batch_size, tp.weight_numel)
        
        result = tp(x1, x2, weight)
        print(f"✓ Forward pass successful, output shape: {result.shape}")
        print(f"✓ Expected output shape: ({batch_size}, {irreps_out.dim})")
        
        if result.shape != (batch_size, irreps_out.dim):
            print("✗ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_mace_integration():
    """Test MACE modules with the new compatibility layer."""
    print("\nTesting MACE integration...")
    
    try:
        from hydragnn.utils.model.mace_utils.tools.cg import U_matrix_real
        print("✓ Successfully imported cg.py with accelerated functions")
    except ImportError as e:
        print(f"✗ Failed to import cg.py: {e}")
        return False
    
    try:
        # Test U_matrix_real function
        irreps_in = o3.Irreps("1x0e + 1x1e")
        irreps_out = o3.Irreps("1x0e + 1x1e")
        correlation = 2
        
        result = U_matrix_real(irreps_in, irreps_out, correlation)
        print(f"✓ U_matrix_real computation successful, result type: {type(result)}")
        
        if not isinstance(result, list):
            print("✗ U_matrix_real should return a list")
            return False
            
    except Exception as e:
        print(f"✗ U_matrix_real computation failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=== OpenEquivariance Integration Test ===\n")
    
    success = True
    
    # Test compatibility module
    if not test_compatibility_module():
        success = False
    
    # Test MACE integration
    if not test_mace_integration():
        success = False
    
    if success:
        print("\n✓ All tests passed!")
        print("OpenEquivariance integration is working correctly.")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()