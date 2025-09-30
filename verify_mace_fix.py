#!/usr/bin/env python3
"""
Reproduction script for MACE IndexError fix.

This script demonstrates the original error and shows that it's now fixed.
Run this script to verify that the fix is working correctly.

Original error:
[rank9]: IndexError: tuple index out of range
[rank9]:     num_params = self.U_tensors(i).size()[-1]
"""

import sys
import os
from pathlib import Path

try:
    import torch
    from e3nn import o3
    print("✓ Required dependencies (torch, e3nn) are available")
except ImportError as e:
    print(f"✗ Missing required dependencies: {e}")
    print("Please install: pip install torch e3nn")
    sys.exit(1)

# Add HydraGNN to path
hydragnn_path = Path(__file__).parent / "hydragnn"
if hydragnn_path.exists():
    sys.path.insert(0, str(Path(__file__).parent))
else:
    print("Note: Running from HydraGNN repository root, adjusting paths...")
    sys.path.insert(0, str(Path(__file__).parent))

def test_original_error_scenario():
    """Test the scenario that caused the original IndexError."""
    print("\n=== Testing Original Error Scenario ===")
    print("Configuration that caused IndexError:")
    print("  irreps_in: 1x0e (scalar)")
    print("  irreps_out: 1x1o (vector with odd parity)")
    print("  correlation: 2")
    print()
    
    try:
        # Import the fixed modules
        from hydragnn.utils.model.mace_utils.modules.symmetric_contraction import Contraction
        
        # This configuration would have caused the original error
        irreps_in = o3.Irreps("1x0e")
        irrep_out = o3.Irreps("1x1o")
        correlation = 2
        
        print("Attempting to create Contraction...")
        try:
            contraction = Contraction(
                irreps_in=irreps_in,
                irrep_out=irrep_out,
                correlation=correlation
            )
            print("✗ Expected ValueError for incompatible irreps, but succeeded")
            return False
        except ValueError as e:
            if "No valid tensor contractions found" in str(e):
                print("✓ Got expected clear error message:")
                print(f"    {e}")
                return True
            else:
                print(f"✗ Got unexpected ValueError: {e}")
                return False
        except Exception as e:
            print(f"✗ Got unexpected error type: {type(e).__name__}: {e}")
            return False
            
    except ImportError as e:
        print(f"Cannot import Contraction class: {e}")
        print("This suggests you're not running from the HydraGNN repository.")
        print("Please run this script from the HydraGNN root directory.")
        return False

def test_valid_configuration():
    """Test that valid configurations still work."""
    print("\n=== Testing Valid Configuration ===")
    print("Configuration that should work:")
    print("  irreps_in: 1x0e (scalar)")
    print("  irreps_out: 1x0e (scalar)")
    print("  correlation: 2")
    print()
    
    try:
        from hydragnn.utils.model.mace_utils.modules.symmetric_contraction import Contraction
        
        irreps_in = o3.Irreps("1x0e")
        irrep_out = o3.Irreps("1x0e")
        correlation = 2
        
        print("Attempting to create Contraction...")
        contraction = Contraction(
            irreps_in=irreps_in,
            irrep_out=irrep_out,
            correlation=correlation
        )
        print("✓ Valid configuration works correctly")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 2
        x = torch.randn(batch_size, irreps_in.dim)
        y = torch.randn(batch_size, 1, irreps_in.dim)
        
        result = contraction(x, y)
        print(f"✓ Forward pass successful, output shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("MACE IndexError Fix Verification")
    print("=" * 40)
    print()
    print("This script verifies that the MACE IndexError has been fixed.")
    print("Original error: 'IndexError: tuple index out of range'")
    
    success = True
    
    # Test the original error scenario
    success &= test_original_error_scenario()
    
    # Test that valid configurations still work
    success &= test_valid_configuration()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 SUCCESS: MACE IndexError fix is working correctly!")
        print()
        print("What was fixed:")
        print("  • UnboundLocalError in _U_matrix_real_e3nn")
        print("  • IndexError when accessing empty tensor dimensions")
        print("  • Added proper validation and clear error messages")
        print("  • Incompatible irreps now give helpful error instead of crash")
    else:
        print("❌ FAILURE: Some tests failed!")
        print("The fix may not be working correctly.")
        sys.exit(1)

if __name__ == "__main__":
    main()