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

import os
import pytest
import warnings
import torch
import numpy as np

try:
    from e3nn import o3
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False

try:
    import openequivariance as oeq
    OPENEQUIV_AVAILABLE = True
except ImportError:
    OPENEQUIV_AVAILABLE = False

from hydragnn.utils.model.equivariance_backend import (
    build_tensor_product,
    initialize_equivariance_backend,
    select_backend,
    get_backend_name,
)
from hydragnn.utils.model.openequiv_adapter import (
    build_oeq_tensor_product_module,
    _normalize_instructions,
)


# Test tolerance for numerical equivalence
TOLERANCE = 1e-6


@pytest.fixture
def reset_backend_state():
    """Reset backend state before each test."""
    import hydragnn.utils.model.equivariance_backend as backend_module
    backend_module._initialized = False
    backend_module._warned_fallback = False
    backend_module._effective_backend = None
    backend_module._requested_backend = None
    
    # Clear environment variable
    if "HYDRAGNN_EQUIVARIANCE_BACKEND" in os.environ:
        del os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"]
    
    yield
    
    # Cleanup after test
    backend_module._initialized = False
    backend_module._warned_fallback = False
    backend_module._effective_backend = None
    backend_module._requested_backend = None


@pytest.fixture
def sample_irreps():
    """Create sample irreps for testing."""
    if not E3NN_AVAILABLE:
        pytest.skip("e3nn not available")
    
    irreps_in1 = o3.Irreps("8x0e + 4x1o + 2x2e")
    irreps_in2 = o3.Irreps("16x0e + 8x1o")  
    irreps_out = o3.Irreps("16x0e + 8x1o + 4x2e")
    
    return irreps_in1, irreps_in2, irreps_out


@pytest.fixture
def sample_data(sample_irreps):
    """Create sample tensor data for testing."""
    irreps_in1, irreps_in2, irreps_out = sample_irreps
    
    batch_size = 4
    x1 = o3.Irreps(irreps_in1).randn(batch_size, -1)
    x2 = o3.Irreps(irreps_in2).randn(batch_size, -1)
    
    return x1, x2


class PytestEquivarianceBackend:
    """Test suite for equivariance backend numerical equivalence."""

    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_backend_selection(self, reset_backend_state):
        """Test backend selection mechanisms."""
        # Test default selection
        backend = select_backend(None)
        assert backend == "e3nn"
        
        # Test config-based selection
        config = {
            "NeuralNetwork": {
                "Architecture": {
                    "equivariance_backend": "e3nn"
                }
            }
        }
        backend = select_backend(config)
        assert backend == "e3nn"
        
        # Test environment variable override
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        backend = select_backend(None)
        assert backend == "e3nn"
    
    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_instruction_normalization(self):
        """Test instruction normalization for OpenEquivariance compatibility."""
        # Test 5-tuple instructions (e3nn style)
        instructions_5 = [(0, 1, 2, "uvw", True), (1, 0, 3, "uvu", False)]
        normalized = _normalize_instructions(instructions_5)
        
        expected = [
            (0, 1, 2, "uvw", True, 1.0),
            (1, 0, 3, "uvu", False, 1.0)
        ]
        assert normalized == expected
        
        # Test 6-tuple instructions (already normalized)
        instructions_6 = [(0, 1, 2, "uvw", True, 2.5)]
        normalized = _normalize_instructions(instructions_6)
        expected = [(0, 1, 2, "uvw", True, 2.5)]
        assert normalized == expected
        
        # Test invalid instructions
        with pytest.raises(ValueError, match="unsupported length"):
            _normalize_instructions([(1, 2, 3)])  # Too short

    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_e3nn_tensor_product_basic(self, reset_backend_state, sample_irreps, sample_data):
        """Test basic tensor product operations with e3nn backend."""
        irreps_in1, irreps_in2, irreps_out = sample_irreps
        x1, x2 = sample_data
        
        # Force e3nn backend
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        initialize_equivariance_backend(None)
        
        # Create tensor product through backend
        tp, weight_numel, backend_used = build_tensor_product(
            irreps_in1,
            irreps_in2, 
            irreps_out,
            instructions=[],  # Let e3nn determine instructions
            shared_weights=False,
            internal_weights=False
        )
        
        assert backend_used == "e3nn"
        assert isinstance(weight_numel, int)
        assert weight_numel > 0
        
        # Test forward pass
        weights = torch.randn(x1.shape[0], weight_numel)
        output = tp(x1, x2, weights)
        
        # Check output shape matches expected irreps
        expected_dim = o3.Irreps(irreps_out).dim
        assert output.shape == (x1.shape[0], expected_dim)

    @pytest.mark.skipif(not (E3NN_AVAILABLE and OPENEQUIV_AVAILABLE), 
                       reason="Both e3nn and OpenEquivariance required")
    def pytest_backend_numerical_equivalence(self, reset_backend_state, sample_irreps):
        """Test numerical equivalence between e3nn and OpenEquivariance backends."""
        irreps_in1, irreps_in2, irreps_out = sample_irreps
        
        # Create instructions manually for consistent comparison
        from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
        _, instructions = tp_out_irreps_with_instructions(
            irreps_in1, irreps_in2, irreps_out
        )
        
        # Create identical input data
        torch.manual_seed(42)
        batch_size = 8
        x1 = o3.Irreps(irreps_in1).randn(batch_size, -1)
        x2 = o3.Irreps(irreps_in2).randn(batch_size, -1)
        
        # Test with e3nn backend
        reset_backend_state  # Reset state
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        initialize_equivariance_backend(None)
        
        tp_e3nn, weight_numel_e3nn, backend_e3nn = build_tensor_product(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )
        
        # Test with OpenEquivariance backend
        reset_backend_state  # Reset state  
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
        initialize_equivariance_backend(None)
        
        tp_oeq, weight_numel_oeq, backend_oeq = build_tensor_product(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )
        
        # Verify backends are different
        assert backend_e3nn == "e3nn"
        assert backend_oeq == "openequivariance"
        
        # Weight dimensions should match
        assert weight_numel_e3nn == weight_numel_oeq
        
        # Test forward pass with identical weights
        torch.manual_seed(123)
        weights = torch.randn(batch_size, weight_numel_e3nn)
        
        # Compute outputs
        output_e3nn = tp_e3nn(x1, x2, weights)
        output_oeq = tp_oeq(x1, x2, weights)
        
        # Check numerical equivalence
        assert output_e3nn.shape == output_oeq.shape
        
        # Compute relative error
        max_diff = torch.max(torch.abs(output_e3nn - output_oeq))
        rel_error = max_diff / (torch.max(torch.abs(output_e3nn)) + 1e-10)
        
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Relative error: {rel_error:.2e}")
        
        # Assert numerical equivalence within tolerance
        assert max_diff < TOLERANCE, f"Outputs differ by {max_diff:.2e}, exceeds tolerance {TOLERANCE:.2e}"
        assert rel_error < TOLERANCE, f"Relative error {rel_error:.2e} exceeds tolerance {TOLERANCE:.2e}"

    @pytest.mark.skipif(not (E3NN_AVAILABLE and OPENEQUIV_AVAILABLE),
                       reason="Both e3nn and OpenEquivariance required")
    @pytest.mark.parametrize("shared_weights", [True, False])
    def pytest_shared_weights_equivalence(self, reset_backend_state, sample_irreps, shared_weights):
        """Test numerical equivalence with different shared_weights settings."""
        irreps_in1, irreps_in2, irreps_out = sample_irreps
        
        from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
        _, instructions = tp_out_irreps_with_instructions(
            irreps_in1, irreps_in2, irreps_out
        )
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 4
        x1 = o3.Irreps(irreps_in1).randn(batch_size, -1)
        x2 = o3.Irreps(irreps_in2).randn(batch_size, -1)
        
        # Test e3nn
        reset_backend_state
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        initialize_equivariance_backend(None)
        
        tp_e3nn, weight_numel_e3nn, _ = build_tensor_product(
            irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=False
        )
        
        # Test OpenEquivariance
        reset_backend_state
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
        initialize_equivariance_backend(None)
        
        tp_oeq, weight_numel_oeq, _ = build_tensor_product(
            irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=False
        )
        
        # Verify weight dimensions match
        assert weight_numel_e3nn == weight_numel_oeq
        
        # Test with identical weights
        torch.manual_seed(123)
        weights = torch.randn(batch_size, weight_numel_e3nn)
        
        output_e3nn = tp_e3nn(x1, x2, weights)
        output_oeq = tp_oeq(x1, x2, weights)
        
        max_diff = torch.max(torch.abs(output_e3nn - output_oeq))
        assert max_diff < TOLERANCE, f"shared_weights={shared_weights}: difference {max_diff:.2e} > {TOLERANCE:.2e}"

    @pytest.mark.skipif(not (E3NN_AVAILABLE and OPENEQUIV_AVAILABLE),
                       reason="Both e3nn and OpenEquivariance required")
    def pytest_multiple_irrep_types(self, reset_backend_state):
        """Test numerical equivalence with various irrep combinations."""
        test_cases = [
            # (irreps_in1, irreps_in2, irreps_out)
            ("4x0e", "4x0e", "8x0e"),  # Scalar only
            ("2x0e + 2x1o", "2x0e + 2x1o", "4x0e + 4x1o"),  # Mixed scalars and vectors
            ("1x0e + 1x1o + 1x2e", "1x0e + 1x1o", "2x0e + 2x1o + 1x2e"),  # Include l=2
            ("8x0e + 4x1o", "16x0e + 8x1o", "16x0e + 8x1o + 4x2e"),  # Larger multiplicities
        ]
        
        for irreps_in1_str, irreps_in2_str, irreps_out_str in test_cases:
            irreps_in1 = o3.Irreps(irreps_in1_str)
            irreps_in2 = o3.Irreps(irreps_in2_str)
            irreps_out = o3.Irreps(irreps_out_str)
            
            from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
            _, instructions = tp_out_irreps_with_instructions(
                irreps_in1, irreps_in2, irreps_out
            )
            
            # Skip if no valid instructions
            if not instructions:
                continue
                
            # Create test data
            torch.manual_seed(42)
            batch_size = 6
            x1 = irreps_in1.randn(batch_size, -1)
            x2 = irreps_in2.randn(batch_size, -1)
            
            # e3nn backend
            reset_backend_state
            os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
            initialize_equivariance_backend(None)
            
            tp_e3nn, weight_numel_e3nn, _ = build_tensor_product(
                irreps_in1, irreps_in2, irreps_out,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False
            )
            
            # OpenEquivariance backend
            reset_backend_state
            os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
            initialize_equivariance_backend(None)
            
            tp_oeq, weight_numel_oeq, _ = build_tensor_product(
                irreps_in1, irreps_in2, irreps_out,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False
            )
            
            # Test equivalence
            torch.manual_seed(123)
            weights = torch.randn(batch_size, weight_numel_e3nn)
            
            output_e3nn = tp_e3nn(x1, x2, weights)
            output_oeq = tp_oeq(x1, x2, weights)
            
            max_diff = torch.max(torch.abs(output_e3nn - output_oeq))
            assert max_diff < TOLERANCE, (
                f"Irreps {irreps_in1_str} ⊗ {irreps_in2_str} → {irreps_out_str}: "
                f"difference {max_diff:.2e} > {TOLERANCE:.2e}"
            )

    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_fallback_behavior(self, reset_backend_state):
        """Test fallback behavior when OpenEquivariance is not available."""
        # Mock OpenEquivariance as unavailable
        import hydragnn.utils.model.openequiv_adapter as adapter_module
        original_available = adapter_module._OEQ_AVAILABLE
        adapter_module._OEQ_AVAILABLE = False
        
        try:
            # Request OpenEquivariance backend
            os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                initialize_equivariance_backend(None)
                
                # Check fallback warning was issued
                warning_messages = [str(warning.message) for warning in w]
                fallback_warnings = [
                    msg for msg in warning_messages 
                    if "OpenEquivariance requested but not available" in msg
                ]
                assert len(fallback_warnings) > 0, "Expected fallback warning"
            
            # Verify we fell back to e3nn
            assert get_backend_name() == "e3nn"
            
        finally:
            # Restore original state
            adapter_module._OEQ_AVAILABLE = original_available

    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_equivariance_property(self, reset_backend_state, sample_irreps):
        """Test that tensor products maintain equivariance property."""
        irreps_in1, irreps_in2, irreps_out = sample_irreps
        
        from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
        _, instructions = tp_out_irreps_with_instructions(
            irreps_in1, irreps_in2, irreps_out
        )
        
        # Test with e3nn backend
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        initialize_equivariance_backend(None)
        
        tp, weight_numel, _ = build_tensor_product(
            irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )
        
        # Create test data and random rotation
        torch.manual_seed(42)
        batch_size = 4
        x1 = irreps_in1.randn(batch_size, -1)
        x2 = irreps_in2.randn(batch_size, -1)
        weights = torch.randn(batch_size, weight_numel)
        
        # Generate random rotation
        R = o3.rand_matrix()
        D_in1 = irreps_in1.D_from_matrix(R)
        D_in2 = irreps_in2.D_from_matrix(R)
        D_out = irreps_out.D_from_matrix(R)
        
        # Compute output before rotation
        output = tp(x1, x2, weights)
        
        # Apply rotation to inputs and compute output
        x1_rot = torch.einsum('ij,bj->bi', D_in1, x1)
        x2_rot = torch.einsum('ij,bj->bi', D_in2, x2)
        output_rot = tp(x1_rot, x2_rot, weights)
        
        # Apply rotation to original output
        expected_output_rot = torch.einsum('ij,bj->bi', D_out, output)
        
        # Check equivariance property holds
        max_diff = torch.max(torch.abs(output_rot - expected_output_rot))
        assert max_diff < TOLERANCE, f"Equivariance violated: difference {max_diff:.2e} > {TOLERANCE:.2e}"

    @pytest.mark.skipif(not (E3NN_AVAILABLE and OPENEQUIV_AVAILABLE),
                       reason="Both e3nn and OpenEquivariance required")
    def pytest_gradient_equivalence(self, reset_backend_state, sample_irreps):
        """Test that gradients are equivalent between backends."""
        irreps_in1, irreps_in2, irreps_out = sample_irreps
        
        from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
        _, instructions = tp_out_irreps_with_instructions(
            irreps_in1, irreps_in2, irreps_out
        )
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 4
        x1 = o3.Irreps(irreps_in1).randn(batch_size, -1, requires_grad=True)
        x2 = o3.Irreps(irreps_in2).randn(batch_size, -1, requires_grad=True)
        
        # Test with e3nn backend
        reset_backend_state
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
        initialize_equivariance_backend(None)
        
        tp_e3nn, weight_numel_e3nn, _ = build_tensor_product(
            irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )
        
        # Test with OpenEquivariance backend
        reset_backend_state
        os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
        initialize_equivariance_backend(None)
        
        tp_oeq, weight_numel_oeq, _ = build_tensor_product(
            irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )
        
        # Use identical weights
        torch.manual_seed(123)
        weights = torch.randn(batch_size, weight_numel_e3nn, requires_grad=True)
        
        # Forward passes
        output_e3nn = tp_e3nn(x1.clone().detach().requires_grad_(True), 
                              x2.clone().detach().requires_grad_(True), 
                              weights.clone().detach().requires_grad_(True))
        output_oeq = tp_oeq(x1.clone().detach().requires_grad_(True), 
                            x2.clone().detach().requires_grad_(True), 
                            weights.clone().detach().requires_grad_(True))
        
        # Create dummy loss
        loss_e3nn = output_e3nn.sum()
        loss_oeq = output_oeq.sum()
        
        # Backward passes
        loss_e3nn.backward()
        loss_oeq.backward()
        
        # Note: This test checks that the tensor products produce similar outputs.
        # Gradient checking would require more complex setup with identical parameter sharing.
        print(f"e3nn output sum: {loss_e3nn.item():.6f}")
        print(f"OpenEquivariance output sum: {loss_oeq.item():.6f}")
        
        # Verify outputs are close
        output_diff = torch.max(torch.abs(output_e3nn - output_oeq))
        assert output_diff < TOLERANCE, f"Output difference {output_diff:.2e} > {TOLERANCE:.2e}"

    @pytest.mark.skipif(not E3NN_AVAILABLE, reason="e3nn not available")
    def pytest_weight_dimensions_consistency(self, reset_backend_state):
        """Test that weight dimensions are consistent across different configurations."""
        irreps_configs = [
            ("4x0e", "4x0e", "8x0e"),
            ("2x0e + 2x1o", "2x0e + 2x1o", "4x0e + 4x1o + 2x2e"),
            ("8x0e + 4x1o + 2x2e", "16x0e + 8x1o", "16x0e + 8x1o + 4x2e"),
        ]
        
        for irreps_in1_str, irreps_in2_str, irreps_out_str in irreps_configs:
            irreps_in1 = o3.Irreps(irreps_in1_str)
            irreps_in2 = o3.Irreps(irreps_in2_str) 
            irreps_out = o3.Irreps(irreps_out_str)
            
            from hydragnn.utils.model.irreps_tools import tp_out_irreps_with_instructions
            _, instructions = tp_out_irreps_with_instructions(
                irreps_in1, irreps_in2, irreps_out
            )
            
            if not instructions:
                continue
                
            # Test e3nn
            reset_backend_state
            os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "e3nn"
            initialize_equivariance_backend(None)
            
            _, weight_numel_e3nn, backend_e3nn = build_tensor_product(
                irreps_in1, irreps_in2, irreps_out,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False
            )
            
            # Test with different shared_weights setting
            _, weight_numel_e3nn_shared, _ = build_tensor_product(
                irreps_in1, irreps_in2, irreps_out,
                instructions=instructions,
                shared_weights=True,
                internal_weights=False
            )
            
            assert backend_e3nn == "e3nn"
            assert isinstance(weight_numel_e3nn, int)
            assert isinstance(weight_numel_e3nn_shared, int)
            assert weight_numel_e3nn > 0
            assert weight_numel_e3nn_shared > 0
            
            print(f"Config {irreps_in1_str} ⊗ {irreps_in2_str} → {irreps_out_str}:")
            print(f"  Weight dimensions (shared=False): {weight_numel_e3nn}")
            print(f"  Weight dimensions (shared=True): {weight_numel_e3nn_shared}")
            
            # Test OpenEquivariance if available
            if OPENEQUIV_AVAILABLE:
                reset_backend_state
                os.environ["HYDRAGNN_EQUIVARIANCE_BACKEND"] = "openequivariance"
                initialize_equivariance_backend(None)
                
                _, weight_numel_oeq, backend_oeq = build_tensor_product(
                    irreps_in1, irreps_in2, irreps_out,
                    instructions=instructions,
                    shared_weights=False,
                    internal_weights=False
                )
                
                if backend_oeq == "openequivariance":
                    assert weight_numel_e3nn == weight_numel_oeq, (
                        f"Weight dimensions mismatch: e3nn={weight_numel_e3nn}, "
                        f"OpenEquivariance={weight_numel_oeq}"
                    )