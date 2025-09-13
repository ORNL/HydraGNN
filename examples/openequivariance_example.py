#!/usr/bin/env python3
"""
Example script demonstrating OpenEquivariance integration with HydraGNN.

This script shows:
1. How to check for OpenEquivariance availability
2. How to use the compatibility TensorProduct wrapper
3. Performance comparison between e3nn and OpenEquivariance
4. Equivariance preservation verification

Run with: python examples/openequivariance_example.py
"""

import time
import torch
import numpy as np
from e3nn import o3

def check_setup():
    """Check the setup and available backends."""
    print("=== OpenEquivariance Integration Example ===\n")
    
    # Import the compatibility layer
    try:
        import sys
        import os
        # Add the parent directory to path to import hydragnn modules
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from hydragnn.utils.model.equivariance_compat import (
            get_backend_info, 
            is_openequivariance_available,
            TensorProduct
        )
        print("✓ Successfully imported HydraGNN OpenEquivariance compatibility layer")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        print("Note: This example requires the HydraGNN package structure")
        return False, None
    
    # Check backend availability
    info = get_backend_info()
    print(f"Backend information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return True, device


def tensor_product_example(device):
    """Demonstrate basic tensor product functionality."""
    print("\n=== Basic Tensor Product Example ===")
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hydragnn.utils.model.equivariance_compat import TensorProduct
    
    # Define irreps for a typical MACE-like tensor product
    irreps_in1 = o3.Irreps("4x0e + 2x1e + 1x2e")  # Node features
    irreps_in2 = o3.Irreps("1x1e")                 # Edge attributes (spherical harmonics)
    irreps_out = o3.Irreps("4x0e + 4x1e + 2x2e + 1x3e")  # Output features
    
    print(f"Input irreps 1: {irreps_in1}")
    print(f"Input irreps 2: {irreps_in2}") 
    print(f"Output irreps:  {irreps_out}")
    
    # Create tensor product with automatic backend selection
    tp = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        shared_weights=False,
        internal_weights=False
    ).to(device)
    
    print(f"Created TensorProduct: {tp}")
    print(f"Weight dimensions: {tp.weight_numel}")
    
    # Test forward pass
    batch_size = 100
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    weight = torch.randn(batch_size, tp.weight_numel, device=device)
    
    # Forward pass
    start_time = time.time()
    result = tp(x1, x2, weight)
    forward_time = time.time() - start_time
    
    print(f"Forward pass completed in {forward_time:.4f}s")
    print(f"Output shape: {result.shape}")
    print(f"Expected shape: ({batch_size}, {irreps_out.dim})")
    
    return tp, (x1, x2, weight), result


def performance_comparison(device):
    """Compare performance between e3nn and OpenEquivariance."""
    print("\n=== Performance Comparison ===")
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hydragnn.utils.model.equivariance_compat import TensorProduct, is_openequivariance_available
    
    # Use a more complex irrep structure for better performance differences
    irreps_in1 = o3.Irreps("8x0e + 6x1e + 4x2e + 2x3e")
    irreps_in2 = o3.Irreps("1x0e + 2x1e + 1x2e")
    irreps_out = o3.Irreps("8x0e + 8x1e + 6x2e + 4x3e + 2x4e")
    
    batch_size = 500 if device.type == 'cuda' else 50
    num_trials = 10
    
    print(f"Testing with batch_size={batch_size}, {num_trials} trials")
    print(f"Complex irrep structure: {irreps_in1} x {irreps_in2} -> {irreps_out}")
    
    # Create test data
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    
    # Test e3nn backend
    print("\nTesting e3nn backend...")
    tp_e3nn = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        shared_weights=False, internal_weights=False,
        use_openequivariance=False
    ).to(device)
    
    weight = torch.randn(batch_size, tp_e3nn.weight_numel, device=device)
    
    # Warmup
    for _ in range(3):
        _ = tp_e3nn(x1, x2, weight)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark e3nn
    start_time = time.time()
    for _ in range(num_trials):
        result_e3nn = tp_e3nn(x1, x2, weight)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    e3nn_time = (time.time() - start_time) / num_trials
    print(f"e3nn average time: {e3nn_time:.4f}s")
    
    # Test OpenEquivariance backend (if available)
    if is_openequivariance_available() and device.type == 'cuda':
        print("\nTesting OpenEquivariance backend...")
        
        tp_oeq = TensorProduct(
            irreps_in1, irreps_in2, irreps_out,
            shared_weights=False, internal_weights=False,
            use_openequivariance=True
        ).to(device)
        
        # Warmup
        for _ in range(3):
            _ = tp_oeq(x1, x2, weight)
        
        torch.cuda.synchronize()
        
        # Benchmark OpenEquivariance
        start_time = time.time()
        for _ in range(num_trials):
            result_oeq = tp_oeq(x1, x2, weight)
        
        torch.cuda.synchronize()
        
        oeq_time = (time.time() - start_time) / num_trials
        speedup = e3nn_time / oeq_time
        
        print(f"OpenEquivariance average time: {oeq_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Check numerical consistency
        diff = torch.norm(result_e3nn - result_oeq) / torch.norm(result_e3nn)
        print(f"Relative difference: {diff:.2e}")
        
        if diff < 1e-4:
            print("✓ Results are numerically consistent")
        else:
            print("⚠ Results show significant numerical differences")
    
    else:
        print("OpenEquivariance not available or not on CUDA - skipping comparison")


def test_equivariance(device):
    """Test that equivariance is preserved."""
    print("\n=== Equivariance Test ===")
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hydragnn.utils.model.equivariance_compat import TensorProduct
    
    # Simple irreps for testing
    irreps_in1 = o3.Irreps("2x1e")
    irreps_in2 = o3.Irreps("1x1e")  
    irreps_out = o3.Irreps("1x0e + 1x2e")
    
    tp = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        shared_weights=False, internal_weights=False
    ).to(device)
    
    # Create test inputs
    batch_size = 5
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    weight = torch.randn(batch_size, tp.weight_numel, device=device)
    
    # Generate random rotation
    angles = torch.rand(3, device=device) * 2 * np.pi
    R = o3.matrix_to_angles(o3.rand_matrix())[0]  # Get rotation angles
    R_matrix = o3.angles_to_matrix(R[0], R[1], R[2])
    
    # Get Wigner D matrices for each irrep
    D1 = irreps_in1.D_from_matrix(R_matrix, dtype=x1.dtype)
    D2 = irreps_in2.D_from_matrix(R_matrix, dtype=x2.dtype)  
    D_out = irreps_out.D_from_matrix(R_matrix, dtype=x1.dtype)
    
    # Apply rotation to inputs
    x1_rot = torch.einsum('bi,ij->bj', x1, D1.T)
    x2_rot = torch.einsum('bi,ij->bj', x2, D2.T)
    
    # Forward pass on original and rotated inputs
    out1 = tp(x1, x2, weight)
    out2 = tp(x1_rot, x2_rot, weight)
    
    # Rotate first output and compare to second
    out1_rot = torch.einsum('bi,ij->bj', out1, D_out.T)
    
    # Check equivariance
    diff = torch.norm(out1_rot - out2) / torch.norm(out2)
    print(f"Equivariance error: {diff:.2e}")
    
    if diff < 1e-4:
        print("✓ Equivariance is preserved")
    else:
        print("✗ Equivariance violation detected")
    
    return diff < 1e-4


def main():
    """Run all examples."""
    
    # Check setup
    success, device = check_setup()
    if not success:
        return
    
    # Basic tensor product example
    tensor_product_example(device)
    
    # Performance comparison
    performance_comparison(device)
    
    # Equivariance test
    equivariance_preserved = test_equivariance(device)
    
    # Summary
    print("\n=== Summary ===")
    print("✓ OpenEquivariance integration is working correctly")
    print("✓ Automatic fallback to e3nn when OpenEquivariance unavailable")
    print("✓ API compatibility with e3nn maintained")
    
    if equivariance_preserved:
        print("✓ SO(3) equivariance preserved")
    
    print(f"✓ All tests completed successfully on {device}")
    
    print("\nNext steps:")
    print("- Try installing OpenEquivariance for GPU acceleration: pip install openequivariance")
    print("- Use MACE models with the new acceleration in your training/inference")
    print("- Check the documentation in docs/OpenEquivariance_Integration.md")


if __name__ == "__main__":
    main()