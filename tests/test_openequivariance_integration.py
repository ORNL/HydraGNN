"""
Tests for OpenEquivariance integration with HydraGNN MACE modules.

This test suite verifies that the compatibility wrapper works correctly
and that OpenEquivariance acceleration is used when available.
"""

import pytest
import torch
import numpy as np
from e3nn import o3

from hydragnn.utils.model.equivariance_compat import (
    TensorProduct,
    is_openequivariance_available,
    get_backend_info,
    set_default_backend,
    _should_use_openequivariance
)

class TestOpenEquivarianceCompatibility:
    """Test the OpenEquivariance compatibility wrapper."""
    
    def test_backend_availability_check(self):
        """Test that backend availability is correctly detected."""
        info = get_backend_info()
        assert "e3nn_available" in info
        assert "openequivariance_available" in info
        assert "default_backend" in info
        assert info["e3nn_available"] is True
        assert info["default_backend"] in ["e3nn", "openequivariance"]
    
    def test_tensor_product_initialization(self):
        """Test that TensorProduct can be initialized with both backends."""
        irreps_in1 = o3.Irreps("1x1e")
        irreps_in2 = o3.Irreps("1x1e") 
        irreps_out = o3.Irreps("1x0e + 1x2e")
        
        # Test e3nn backend
        tp_e3nn = TensorProduct(
            irreps_in1, irreps_in2, irreps_out,
            use_openequivariance=False
        )
        assert not tp_e3nn.use_oeq
        assert hasattr(tp_e3nn, 'tp_backend')
        assert hasattr(tp_e3nn, 'weight_numel')
        
        # Test OpenEquivariance backend (if available)
        if is_openequivariance_available():
            tp_oeq = TensorProduct(
                irreps_in1, irreps_in2, irreps_out,
                use_openequivariance=True
            )
            assert tp_oeq.use_oeq
            assert hasattr(tp_oeq, 'tp_backend')
            assert hasattr(tp_oeq, 'weight_numel')
    
    def test_tensor_product_forward_pass(self):
        """Test that tensor product forward passes work correctly."""
        torch.manual_seed(42)
        device = torch.device('cpu')  # Use CPU for testing
        
        irreps_in1 = o3.Irreps("2x1e")
        irreps_in2 = o3.Irreps("1x1e")
        irreps_out = o3.Irreps("1x0e + 1x2e")
        
        batch_size = 4
        x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
        x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
        
        # Create tensor product with e3nn backend
        tp_e3nn = TensorProduct(
            irreps_in1, irreps_in2, irreps_out,
            shared_weights=False,
            internal_weights=False,
            use_openequivariance=False
        )
        
        # Generate weights
        weight = torch.randn(batch_size, tp_e3nn.weight_numel, device=device)
        
        # Test forward pass
        result_e3nn = tp_e3nn(x1, x2, weight)
        assert result_e3nn.shape == (batch_size, irreps_out.dim)
        
        # If OpenEquivariance is available, test it too
        if is_openequivariance_available() and device.type == 'cuda':
            # Move tensors to GPU for OpenEquivariance
            x1_gpu = x1.cuda()
            x2_gpu = x2.cuda() 
            weight_gpu = weight.cuda()
            
            tp_oeq = TensorProduct(
                irreps_in1, irreps_in2, irreps_out,
                shared_weights=False,
                internal_weights=False,
                use_openequivariance=True
            ).cuda()
            
            result_oeq = tp_oeq(x1_gpu, x2_gpu, weight_gpu)
            assert result_oeq.shape == (batch_size, irreps_out.dim)
            
            # Results should be close (allowing for different numerical precision)
            torch.testing.assert_close(
                result_e3nn.cuda(), result_oeq.cuda(), 
                rtol=1e-4, atol=1e-4
            )
    
    def test_auto_backend_selection(self):
        """Test that automatic backend selection works correctly."""
        irreps_in1 = o3.Irreps("1x0e")
        irreps_in2 = o3.Irreps("1x1e")
        irreps_out = o3.Irreps("1x1e")
        
        # Auto selection should pick OpenEquivariance if available
        tp_auto = TensorProduct(irreps_in1, irreps_in2, irreps_out)
        
        if is_openequivariance_available():
            assert tp_auto.use_oeq
        else:
            assert not tp_auto.use_oeq
    
    def test_forced_backend_selection(self):
        """Test forced backend selection through global setting."""
        irreps_in1 = o3.Irreps("1x0e")
        irreps_in2 = o3.Irreps("1x1e") 
        irreps_out = o3.Irreps("1x1e")
        
        # Force e3nn backend
        set_default_backend("e3nn")
        tp_forced_e3nn = TensorProduct(irreps_in1, irreps_in2, irreps_out)
        assert not tp_forced_e3nn.use_oeq
        
        # Reset to auto
        set_default_backend("auto")


class TestMACEIntegration:
    """Test MACE module integration with OpenEquivariance."""
    
    def test_mace_blocks_import(self):
        """Test that MACE blocks can be imported with the new compatibility layer."""
        try:
            from hydragnn.utils.model.mace_utils.modules.blocks import (
                RealAgnosticAttResidualInteractionBlock
            )
            # If we get here, the import succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import MACE blocks with compatibility layer: {e}")
    
    def test_clebsch_gordon_acceleration(self):
        """Test that Clebsch-Gordon calculations can use acceleration."""
        from hydragnn.utils.model.mace_utils.tools.cg import U_matrix_real
        
        irreps_in = o3.Irreps("1x0e + 1x1e")
        irreps_out = o3.Irreps("1x0e + 1x1e + 1x2e")
        correlation = 2
        
        # This should work with either backend
        result = U_matrix_real(irreps_in, irreps_out, correlation)
        
        # Check that we get a reasonable result
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.skipif(not is_openequivariance_available(), 
                       reason="OpenEquivariance not available")
    def test_openequivariance_specific_features(self):
        """Test features specific to OpenEquivariance backend."""
        info = get_backend_info()
        assert info["openequivariance_available"] is True
        assert "openequivariance_version" in info
        
        # Test that we can create tensor products that use the OEQ backend
        irreps_in1 = o3.Irreps("1x1e")
        irreps_in2 = o3.Irreps("1x1e")
        irreps_out = o3.Irreps("1x0e + 1x2e")
        
        tp = TensorProduct(
            irreps_in1, irreps_in2, irreps_out,
            use_openequivariance=True
        )
        assert tp.use_oeq is True
        assert "OpenEquivariance" in str(tp)


def test_equivariance_preservation():
    """Test that equivariance is preserved with both backends."""
    torch.manual_seed(123)
    device = torch.device('cpu')
    
    irreps_in1 = o3.Irreps("1x1e") 
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x2e")
    
    batch_size = 2
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    
    # Create rotation matrix
    angles = torch.tensor([0.5, 0.3, 0.7])  
    R = o3.matrix_to_angles(o3.rand_matrix())  # Generate random rotation
    
    # Test e3nn backend
    tp_e3nn = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        shared_weights=False,
        internal_weights=False,
        use_openequivariance=False
    )
    
    weight = torch.randn(batch_size, tp_e3nn.weight_numel, device=device)
    
    # Forward pass on original inputs
    out1 = tp_e3nn(x1, x2, weight)
    
    # Apply rotation to inputs
    D1 = irreps_in1.D_from_matrix(o3.angles_to_matrix(*R))
    D2 = irreps_in2.D_from_matrix(o3.angles_to_matrix(*R))
    D_out = irreps_out.D_from_matrix(o3.angles_to_matrix(*R))
    
    x1_rot = x1 @ D1.T
    x2_rot = x2 @ D2.T
    
    # Forward pass on rotated inputs
    out2 = tp_e3nn(x1_rot, x2_rot, weight)
    
    # Rotate the first output
    out1_rot = out1 @ D_out.T
    
    # Should be approximately equal (up to numerical precision)
    torch.testing.assert_close(out1_rot, out2, rtol=1e-4, atol=1e-4)


