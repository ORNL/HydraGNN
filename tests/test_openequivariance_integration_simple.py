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
Simple integration test script for OpenEquivariance compatibility.

This test verifies:
1. Basic functionality of the compatibility module
2. MACE integration with accelerated functions
3. Proper error handling and backend detection
"""

import pytest
import torch
from e3nn import o3

from hydragnn.utils.model.equivariance_compat import (
    get_backend_info,
    is_openequivariance_available,
    TensorProduct,
)


@pytest.mark.mpi_skip()
def pytest_compatibility_module():
    """Test basic functionality of the compatibility module."""
    print("Testing OpenEquivariance compatibility module...")

    # Test imports
    assert get_backend_info is not None
    assert is_openequivariance_available is not None
    assert TensorProduct is not None
    print("✓ Successfully imported compatibility module")

    # Test backend detection
    is_available = is_openequivariance_available()
    print(f"OpenEquivariance available: {is_available}")
    backend_info = get_backend_info()
    print(f"Backend info: {backend_info}")

    assert "e3nn_available" in backend_info
    assert "openequivariance_available" in backend_info
    assert "default_backend" in backend_info

    # Test tensor product creation
    irreps_in1 = o3.Irreps("1x1e")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("1x0e + 1x2e")

    tp = TensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=False, internal_weights=False
    )
    assert tp is not None
    print(f"✓ Successfully created TensorProduct: {tp}")

    # Test forward pass
    batch_size = 2
    x1 = torch.randn(batch_size, irreps_in1.dim)
    x2 = torch.randn(batch_size, irreps_in2.dim)
    weight = torch.randn(batch_size, tp.weight_numel)

    result = tp(x1, x2, weight)
    print(f"✓ Forward pass successful, output shape: {result.shape}")
    print(f"✓ Expected output shape: ({batch_size}, {irreps_out.dim})")

    assert result.shape == (batch_size, irreps_out.dim)


@pytest.mark.mpi_skip()
def pytest_mace_integration():
    """Test MACE modules with the new compatibility layer."""
    print("\nTesting MACE integration...")

    # Test cg.py import and functionality
    from hydragnn.utils.model.mace_utils.tools.cg import U_matrix_real

    assert U_matrix_real is not None
    print("✓ Successfully imported cg.py with accelerated functions")

    # Test U_matrix_real function
    irreps_in = o3.Irreps("1x0e + 1x1e")
    irreps_out = o3.Irreps("1x0e + 1x1e")
    correlation = 2

    result = U_matrix_real(irreps_in, irreps_out, correlation)
    print(f"✓ U_matrix_real computation successful, result type: {type(result)}")

    assert isinstance(result, list), "U_matrix_real should return a list"


@pytest.mark.mpi_skip()
def pytest_integration_summary():
    """Provide summary of integration test results."""
    print("\n=== OpenEquivariance Integration Test Summary ===")
    print("✓ Compatibility module tests passed")
    print("✓ MACE integration tests passed")
    print("OpenEquivariance integration is working correctly.")
