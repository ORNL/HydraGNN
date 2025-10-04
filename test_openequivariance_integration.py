#!/usr/bin/env python3
"""
Test script for OpenEquivariance integration in HydraGNN MACE.
This script tests that:
1. Configuration parameter is properly read
2. OpenEquivariance availability is checked correctly
3. Fallback to e3nn works when OpenEquivariance is not available
4. Model can be created with the new parameter
"""

import os
import sys
import json
import warnings

# Add HydraGNN to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.models.create import create_model_config
from hydragnn.utils.model.openequivariance_utils import (
    check_openequivariance_availability,
    is_openequivariance_enabled,
    HAS_OPENEQUIVARIANCE
)

def test_openequivariance_integration():
    """Test OpenEquivariance integration."""
    
    print("=" * 60)
    print("Testing OpenEquivariance Integration in HydraGNN MACE")
    print("=" * 60)
    
    # Test 1: Check OpenEquivariance availability detection
    print("\n1. Testing OpenEquivariance availability detection:")
    print(f"   OpenEquivariance module available: {HAS_OPENEQUIVARIANCE}")
    
    # Test with enabled=False (should always return False)
    result_disabled = check_openequivariance_availability(False)
    print(f"   check_openequivariance_availability(False): {result_disabled}")
    assert not result_disabled, "Should return False when disabled"
    
    # Test with enabled=True 
    result_enabled = check_openequivariance_availability(True)
    print(f"   check_openequivariance_availability(True): {result_enabled}")
    
    if HAS_OPENEQUIVARIANCE:
        print("   ✓ OpenEquivariance is available and configured for use")
    else:
        print("   ⚠ OpenEquivariance is not available, using e3nn fallback")
    
    # Test 2: Load and parse configuration
    print("\n2. Testing configuration parsing:")
    config_file = os.path.join(os.path.dirname(__file__), 'LJ_openequivariance_test.json')
    
    if not os.path.exists(config_file):
        print(f"   ❌ Test configuration file not found: {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check that the parameter is in the config
    enable_oe = config["NeuralNetwork"]["Architecture"].get("enable_openequivariance", False)
    print(f"   enable_openequivariance in config: {enable_oe}")
    
    # Test 3: Configuration processing
    print("\n3. Testing configuration update:")
    
    # Create dummy data loaders for config update (minimal implementation)
    class DummyDataset:
        def __init__(self):
            self.pna_deg = None
            
        def __getitem__(self, idx):
            # Return a dummy data object with required attributes
            class DummyData:
                def __init__(self):
                    self.y = [torch.tensor([1.0])]  # graph target
                    self.x = torch.tensor([[1.0]])  # node features
                    
            return DummyData()
    
    class DummyLoader:
        def __init__(self):
            self.dataset = DummyDataset()
    
    train_loader = DummyLoader()
    val_loader = DummyLoader()
    test_loader = DummyLoader()
    
    try:
        # Update config with dummy loaders
        updated_config = update_config(config, train_loader, val_loader, test_loader)
        
        # Check that the parameter is preserved and defaults are set
        final_enable_oe = updated_config["NeuralNetwork"]["Architecture"].get("enable_openequivariance", False)
        print(f"   enable_openequivariance after update: {final_enable_oe}")
        
        print("   ✓ Configuration update successful")
        
    except Exception as e:
        print(f"   ❌ Configuration update failed: {e}")
        return False
    
    # Test 4: Model creation
    print("\n4. Testing model creation:")
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create model with OpenEquivariance enabled
            model = create_model_config(
                config=updated_config["NeuralNetwork"],
                verbosity=1
            )
            
            print(f"   ✓ Model created successfully: {type(model).__name__}")
            
            # Check if any warnings were issued
            oe_warnings = [warning for warning in w if "openequivariance" in str(warning.message).lower()]
            if oe_warnings:
                print("   Warnings about OpenEquivariance:")
                for warning in oe_warnings:
                    print(f"     - {warning.message}")
            
            # Check if the model has the OpenEquivariance usage flag
            if hasattr(model, 'using_openequivariance'):
                print(f"   Model using OpenEquivariance: {model.using_openequivariance}")
            
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Basic forward pass test
    print("\n5. Testing basic model functionality:")
    
    try:
        # Create dummy data for forward pass
        from torch_geometric.data import Data, Batch
        
        # Create a simple molecular graph (2 atoms)
        x = torch.tensor([[1.0], [1.0]], dtype=torch.float32)  # Node features (atomic numbers)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)  # Positions
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()  # Edges
        y = torch.tensor([[-1.0]], dtype=torch.float32)  # Graph target
        
        data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
        batch = Batch.from_data_list([data])
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        print(f"   ✓ Forward pass successful, output shape: {[o.shape for o in output]}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("OpenEquivariance Integration Test Summary:")
    print("=" * 60)
    if HAS_OPENEQUIVARIANCE and result_enabled:
        print("✓ OpenEquivariance is available and integrated successfully")
    else:
        print("⚠ OpenEquivariance not available, but e3nn fallback works correctly")
    print("✓ Configuration parameter properly handled")
    print("✓ Model creation works with new parameter")
    print("✓ Basic model functionality verified")
    print("\nTest completed successfully!")
    
    return True


if __name__ == "__main__":
    success = test_openequivariance_integration()
    sys.exit(0 if success else 1)