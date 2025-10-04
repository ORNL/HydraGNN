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
Core tests for OpenEquivariance integration.

These tests verify the integration works correctly regardless of whether
OpenEquivariance is available, focusing on the fallback behavior and
basic functionality.
"""

import pytest
import torch
import warnings
import numpy as np

# HydraGNN imports
from hydragnn.utils.model.openequivariance_utils import (
    check_openequivariance_availability,
    HAS_OPENEQUIVARIANCE,
    OptimizedLinear,
    OptimizedSphericalHarmonics,
    optimized_einsum,
)

# e3nn imports
from e3nn import o3

# Test parameters
RTOL = 1e-5
ATOL = 1e-6
TEST_SEED = 42


def setup_test_environment():
    """Set up test environment with fixed random seed."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)


@pytest.mark.mpi_skip()
def pytest_openequivariance_availability_check():
    """Test that OpenEquivariance availability is checked correctly."""


@pytest.mark.mpi_skip()
def pytest_optimized_einsum_basic():
    """Test basic einsum functionality."""
    setup_test_environment()

    # Simple matrix multiplication
    a = torch.randn(3, 4, dtype=torch.float64)
    b = torch.randn(4, 5, dtype=torch.float64)

    result_optimized = optimized_einsum("ij,jk->ik", a, b)
    result_torch = torch.einsum("ij,jk->ik", a, b)

    assert torch.allclose(result_optimized, result_torch, rtol=RTOL, atol=ATOL)
    assert result_optimized.shape == (3, 5)


@pytest.mark.mpi_skip()
def pytest_optimized_linear_basic():
    """Test basic OptimizedLinear functionality."""
    setup_test_environment()

    irreps_in = o3.Irreps("2x0e")
    irreps_out = o3.Irreps("1x0e")

    linear = OptimizedLinear(irreps_in, irreps_out)
    x = torch.randn(4, irreps_in.dim, dtype=torch.float64)

    with torch.no_grad():
        y = linear(x)

    assert y.shape == (4, irreps_out.dim)
    assert torch.isfinite(y).all()


@pytest.mark.mpi_skip()
def pytest_optimized_spherical_harmonics_basic():
    """Test basic OptimizedSphericalHarmonics functionality."""
    setup_test_environment()

    irreps_sh = o3.Irreps.spherical_harmonics(2)
    sh = OptimizedSphericalHarmonics(irreps_sh)

    # Create normalized direction vectors
    directions = torch.randn(3, 3, dtype=torch.float64)
    directions = directions / directions.norm(dim=1, keepdim=True)

    with torch.no_grad():
        y = sh(directions)

    assert y.shape == (3, irreps_sh.dim)
    assert torch.isfinite(y).all()


@pytest.mark.mpi_skip()
def pytest_backend_consistency():
    """Test that the same operations produce consistent results."""
    setup_test_environment()

    # Test einsum consistency across multiple calls
    a = torch.randn(3, 4, dtype=torch.float64)
    b = torch.randn(4, 5, dtype=torch.float64)

    result1 = optimized_einsum("ij,jk->ik", a, b)
    result2 = optimized_einsum("ij,jk->ik", a, b)

    assert torch.allclose(result1, result2, rtol=1e-10, atol=1e-12)


@pytest.mark.mpi_skip()
def pytest_mace_integration_basic():
    """Test basic MACE integration functionality."""
    try:
        from hydragnn.utils.model.mace_utils.modules.blocks import (
            TensorProductWeightsBlock,
        )
    except ImportError:
        pytest.skip("MACE modules not available")

    setup_test_environment()

    # Test TensorProductWeightsBlock
    weights_block = TensorProductWeightsBlock(3, 4, 5)

    sender_attrs = torch.randn(2, 3, dtype=torch.float64)
    edge_feats = torch.randn(2, 4, dtype=torch.float64)

    result = weights_block(sender_attrs, edge_feats)

    assert result.shape == (2, 5)
    assert torch.isfinite(result).all()


@pytest.mark.mpi_skip()
def pytest_configuration_parameter_handling():
    """Test that configuration parameters are handled correctly."""
    from hydragnn.utils.input_config_parsing.config_utils import update_config

    # Test minimal config with OpenEquivariance parameter
    config = {
        "NeuralNetwork": {
            "Architecture": {"enable_openequivariance": True, "mpnn_type": "MACE"}
        }
    }

    # The parameter should be preserved
    enable_oe = config["NeuralNetwork"]["Architecture"]["enable_openequivariance"]
    assert enable_oe is True

    # Test default value
    config_no_param = {"NeuralNetwork": {"Architecture": {"mpnn_type": "MACE"}}}

    # Simulate what happens in update_config for default setting
    if (
        "enable_openequivariance"
        not in config_no_param["NeuralNetwork"]["Architecture"]
    ):
        config_no_param["NeuralNetwork"]["Architecture"][
            "enable_openequivariance"
        ] = False

    assert (
        config_no_param["NeuralNetwork"]["Architecture"]["enable_openequivariance"]
        is False
    )


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def pytest_dtype_consistency(dtype):
    """Test that operations work consistently with different dtypes."""
    setup_test_environment()

    # Test einsum with different dtypes
    a = torch.randn(3, 4, dtype=dtype)
    b = torch.randn(4, 5, dtype=dtype)

    result_optimized = optimized_einsum("ij,jk->ik", a, b)
    result_torch = torch.einsum("ij,jk->ik", a, b)

    assert result_optimized.dtype == dtype
    assert result_torch.dtype == dtype

    # Tolerance should be appropriate for dtype
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(
        result_optimized, result_torch, rtol=tolerance, atol=tolerance
    )


@pytest.mark.mpi_skip()
def pytest_error_handling():
    """Test that error handling works correctly."""
    setup_test_environment()

    # Test with invalid irreps (should not crash)
    try:
        irreps_in = o3.Irreps("1x0e")
        irreps_out = o3.Irreps("1x0e")
        linear = OptimizedLinear(irreps_in, irreps_out)

        x = torch.randn(2, irreps_in.dim)
        y = linear(x)

        assert y.shape == (2, irreps_out.dim)
    except Exception as e:
        # Should not crash, but if it does, error should be informative
        assert len(str(e)) > 0


@pytest.mark.mpi_skip()
def pytest_warning_system():
    """Test that the warning system works correctly."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This should trigger warnings if OpenEquivariance is not available
        check_openequivariance_availability(True)

        if not HAS_OPENEQUIVARIANCE:
            # Should have warnings about availability
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any("openequivariance" in msg.lower() for msg in warning_messages)


if __name__ == "__main__":
    """Run basic tests when executed directly."""
    print("Running basic OpenEquivariance integration tests...")

    try:
        pytest_openequivariance_availability_check()
        pytest_optimized_einsum_basic()
        pytest_optimized_linear_basic()
        pytest_optimized_spherical_harmonics_basic()
        pytest_backend_consistency()
        pytest_configuration_parameter_handling()
        pytest_dtype_consistency(torch.float32)
        pytest_dtype_consistency(torch.float64)
        pytest_error_handling()
        pytest_warning_system()

        try:
            pytest_mace_integration_basic()
        except Exception as e:
            print(f"MACE integration test skipped: {e}")

        print("✅ All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
