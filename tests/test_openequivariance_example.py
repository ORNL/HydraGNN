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
Test script demonstrating OpenEquivariance integration with HydraGNN.

This test shows:
1. How to check for OpenEquivariance availability
2. How to use the compatibility TensorProduct wrapper
3. Performance comparison between e3nn and OpenEquivariance
4. Equivariance preservation verification
"""

import time
import pytest
import torch
import numpy as np
from e3nn import o3

from hydragnn.utils.model.equivariance_compat import (
    get_backend_info,
    is_openequivariance_available,
    TensorProduct,
)


@pytest.mark.mpi_skip()
def pytest_setup_and_backend_info():
    """Check the setup and available backends."""
    # Check backend availability
    info = get_backend_info()
    assert "e3nn_available" in info
    assert "openequivariance_available" in info
    assert "default_backend" in info
    assert info["e3nn_available"] is True
    assert info["default_backend"] in ["e3nn", "openequivariance"]

    device = torch.device("cpu")  # Always use CPU for CI testing
    assert device.type == "cpu"


@pytest.mark.mpi_skip()
def pytest_tensor_product_basic_functionality():
    """Demonstrate basic tensor product functionality."""
    device = torch.device("cpu")  # Use CPU for CI

    # Define irreps for a typical MACE-like tensor product
    irreps_in1 = o3.Irreps("4x0e + 2x1e + 1x2e")  # Node features
    irreps_in2 = o3.Irreps("1x1e")  # Edge attributes (spherical harmonics)
    irreps_out = o3.Irreps("4x0e + 4x1e + 2x2e + 1x3e")  # Output features

    try:
        # Create tensor product with automatic backend selection
        tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=False,
            internal_weights=False,
        ).to(device)

        # Basic assertions
        assert tp.weight_numel > 0
        assert hasattr(tp, "tp_backend")

        # Test forward pass
        batch_size = 100
        x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
        x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
        weight = torch.randn(batch_size, tp.weight_numel, device=device)

        # Forward pass
        result = tp(x1, x2, weight)

        # Basic shape check - use the actual output irreps from the tensor product
        # (which may be different from the target irreps due to instruction generation)
        assert result.shape == (batch_size, tp.irreps_out.dim)
        assert not torch.isnan(result).any()

    except Exception as e:
        # If there's still an error, at least give us some information
        pytest.fail(f"TensorProduct test failed with error: {e}")

    # If we get here, the test passed


@pytest.mark.mpi_skip()
def pytest_performance_comparison():
    """Compare performance between e3nn and OpenEquivariance on CPU."""
    device = torch.device("cpu")  # Use CPU for CI testing

    # Use a moderate irrep structure for CPU testing
    irreps_in1 = o3.Irreps("4x0e + 3x1e + 2x2e + 1x3e")
    irreps_in2 = o3.Irreps("1x0e + 1x1e + 1x2e")
    irreps_out = o3.Irreps("4x0e + 4x1e + 3x2e + 2x3e + 1x4e")

    batch_size = 50  # Smaller batch size for CPU testing
    num_trials = 5  # Fewer trials for CI

    # Create test data
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)

    # Test e3nn backend
    tp_e3nn = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        shared_weights=False,
        internal_weights=False,
        use_openequivariance=False,
    ).to(device)

    weight = torch.randn(batch_size, tp_e3nn.weight_numel, device=device)

    # Warmup
    for _ in range(2):
        _ = tp_e3nn(x1, x2, weight)

    # Benchmark e3nn
    start_time = time.time()
    for _ in range(num_trials):
        result_e3nn = tp_e3nn(x1, x2, weight)

    e3nn_time = (time.time() - start_time) / num_trials
    assert e3nn_time > 0  # Sanity check

    # Test that the result has the correct shape
    assert result_e3nn.shape == (batch_size, tp_e3nn.irreps_out.dim)

    # If OpenEquivariance is available, test it (even on CPU)
    if is_openequivariance_available():
        tp_oeq = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=False,
            internal_weights=False,
            use_openequivariance=True,
        ).to(device)

        # Warmup
        for _ in range(2):
            _ = tp_oeq(x1, x2, weight)

        # Benchmark OpenEquivariance
        start_time = time.time()
        for _ in range(num_trials):
            result_oeq = tp_oeq(x1, x2, weight)

        oeq_time = (time.time() - start_time) / num_trials
        assert oeq_time > 0  # Sanity check

        # Check numerical consistency
        diff = torch.norm(result_e3nn - result_oeq) / torch.norm(result_e3nn)
        assert diff < 1e-4, f"Results differ by {diff:.2e}, expected < 1e-4"


@pytest.mark.mpi_skip()
def pytest_equivariance_preservation():
    """Test that equivariance is preserved."""
    device = torch.device("cpu")  # Use CPU for CI

    # Simple irreps for testing
    irreps_in1 = o3.Irreps("2x1e")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x2e")

    tp = TensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=False, internal_weights=False
    ).to(device)

    # Create test inputs
    batch_size = 5
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    weight = torch.randn(batch_size, tp.weight_numel, device=device)

    # Generate random rotation
    R_matrix = o3.rand_matrix()

    # Get Wigner D matrices for each irrep
    D1 = irreps_in1.D_from_matrix(R_matrix)
    D2 = irreps_in2.D_from_matrix(R_matrix)
    D_out = tp.irreps_out.D_from_matrix(R_matrix)

    # Apply rotation to inputs
    x1_rot = torch.einsum("bi,ij->bj", x1, D1.T)
    x2_rot = torch.einsum("bi,ij->bj", x2, D2.T)

    # Forward pass on original and rotated inputs
    out1 = tp(x1, x2, weight)
    out2 = tp(x1_rot, x2_rot, weight)

    # Rotate first output and compare to second
    out1_rot = torch.einsum("bi,ij->bj", out1, D_out.T)

    # Check equivariance
    diff = torch.norm(out1_rot - out2) / torch.norm(out2)
    assert diff < 1e-4, f"Equivariance error {diff:.2e} exceeds tolerance 1e-4"


@pytest.mark.mpi_skip()
@pytest.mark.skipif(
    not is_openequivariance_available(), reason="OpenEquivariance not available"
)
def pytest_openequivariance_specific_features():
    """Test features specific to OpenEquivariance backend."""
    info = get_backend_info()
    assert info["openequivariance_available"] is True
    assert "openequivariance_version" in info

    # Test that we can create tensor products that use the OEQ backend
    irreps_in1 = o3.Irreps("1x1e")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x2e")

    tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, use_openequivariance=True)
    assert tp.use_oeq is True
    assert "OpenEquivariance" in str(tp)


@pytest.mark.mpi_skip()
def pytest_mace_integration():
    """Test basic MACE module integration with OpenEquivariance."""
    # Test that MACE blocks can be imported with the new compatibility layer
    try:
        from hydragnn.utils.model.mace_utils.modules.blocks import (
            RealAgnosticAttResidualInteractionBlock,
        )

        # If we get here, the import succeeded
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import MACE blocks with compatibility layer: {e}")

    # Test that Clebsch-Gordon calculations can use acceleration
    from hydragnn.utils.model.mace_utils.tools.cg import U_matrix_real

    irreps_in = o3.Irreps("1x0e + 1x1e")
    irreps_out = o3.Irreps("1x0e + 1x1e + 1x2e")
    correlation = 2

    # This should work with either backend
    result = U_matrix_real(irreps_in, irreps_out, correlation)

    # Check that we get a reasonable result
    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.mpi_skip()
def pytest_comprehensive_integration_summary():
    """Test comprehensive summary of integration features."""
    # Verify backend information is accessible
    info = get_backend_info()
    assert isinstance(info, dict)

    # Verify e3nn is always available
    assert info["e3nn_available"] is True

    # Verify compatibility layer works
    irreps_in1 = o3.Irreps("1x0e + 1x1e")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x1e + 1x2e")

    # This should work regardless of OpenEquivariance availability
    tp = TensorProduct(irreps_in1, irreps_in2, irreps_out)
    assert hasattr(tp, "weight_numel")
    assert hasattr(tp, "use_oeq")

    # Test forward pass works
    device = torch.device("cpu")
    batch_size = 2
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    weight = torch.randn(batch_size, tp.weight_numel, device=device)

    tp = tp.to(device)
    result = tp(x1, x2, weight)
    assert result.shape == (batch_size, tp.irreps_out.dim)
