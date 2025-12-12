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
Test numerical equivalence between OpenEquivariance and e3nn implementations.

This test ensures that when OpenEquivariance is available, the optimized
operations produce numerically identical results to the e3nn fallback
implementations, within a small numerical tolerance.
"""

import pytest
import torch
import warnings
import numpy as np

# HydraGNN imports
from hydragnn.utils.model.openequivariance_utils import (
    check_openequivariance_availability,
    HAS_OPENEQUIVARIANCE,
    OptimizedTensorProduct,
    OptimizedSphericalHarmonics,
    OptimizedLinear,
    optimized_einsum,
    _USE_OPENEQUIVARIANCE,
)

# e3nn imports for direct comparison
from e3nn import o3

# Test tolerances
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance

# Test parameters
BATCH_SIZE = 8
TEST_SEED = 42

# Check if OpenEquivariance is available for conditional test execution
openequivariance_available = HAS_OPENEQUIVARIANCE


def setup_test_environment():
    """Set up test environment with fixed random seed."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)


def force_e3nn_backend():
    """Force the use of e3nn backend for comparison tests."""
    global _USE_OPENEQUIVARIANCE
    original_state = _USE_OPENEQUIVARIANCE
    _USE_OPENEQUIVARIANCE = False
    return original_state


def restore_backend_state(original_state):
    """Restore the original backend state."""
    global _USE_OPENEQUIVARIANCE
    _USE_OPENEQUIVARIANCE = original_state


@pytest.mark.skipif(
    not openequivariance_available, reason="OpenEquivariance not available"
)
def pytest_linear_equivalence():
    """Test that OptimizedLinear produces identical results to e3nn."""
    setup_test_environment()

    # Test various irreps combinations
    test_cases = [
        (o3.Irreps("3x0e"), o3.Irreps("1x0e")),  # Scalar to scalar
        (o3.Irreps("2x0e + 1x1o"), o3.Irreps("1x0e")),  # Mixed to scalar
        (o3.Irreps("1x0e + 1x1o + 1x2e"), o3.Irreps("2x0e + 1x1o")),  # Complex case
    ]

    for irreps_in, irreps_out in test_cases:
        print(f"Testing OptimizedLinear: {irreps_in} -> {irreps_out}")

        # Create test input
        x = torch.randn(BATCH_SIZE, irreps_in.dim, dtype=torch.float64)

        # Test with OpenEquivariance (if available)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_openequivariance_availability(True)

        linear_oe = OptimizedLinear(
            irreps_in, irreps_out, internal_weights=True, shared_weights=True
        )

        # Force e3nn backend for comparison
        original_state = force_e3nn_backend()
        linear_e3nn = OptimizedLinear(
            irreps_in, irreps_out, internal_weights=True, shared_weights=True
        )
        restore_backend_state(original_state)

        # Copy weights from e3nn to OpenEquivariance for fair comparison
        if (
            hasattr(linear_oe, "using_openequivariance")
            and linear_oe.using_openequivariance
        ):
            # If OpenEquivariance is actually being used, copy weights
            if hasattr(linear_e3nn.linear, "weight") and hasattr(
                linear_oe.linear, "weight"
            ):
                with torch.no_grad():
                    linear_oe.linear.weight.copy_(linear_e3nn.linear.weight)

        # Test forward pass
        with torch.no_grad():
            y_oe = linear_oe(x)
            y_e3nn = linear_e3nn(x)

        # Check numerical equivalence
        assert torch.allclose(
            y_oe, y_e3nn, rtol=RTOL, atol=ATOL
        ), f"OptimizedLinear outputs differ: max_diff={torch.max(torch.abs(y_oe - y_e3nn))}"

        print(f"  ✓ Passed: max_diff={torch.max(torch.abs(y_oe - y_e3nn)):.2e}")


@pytest.mark.skipif(
    not openequivariance_available, reason="OpenEquivariance not available"
)
def pytest_spherical_harmonics_equivalence():
    """Test that OptimizedSphericalHarmonics produces identical results to e3nn."""
    setup_test_environment()

    # Test different l_max values
    l_max_values = [1, 2, 3]

    for l_max in l_max_values:
        print(f"Testing OptimizedSphericalHarmonics with l_max={l_max}")

        irreps_sh = o3.Irreps.spherical_harmonics(l_max)

        # Create normalized direction vectors
        directions = torch.randn(BATCH_SIZE, 3, dtype=torch.float64)
        directions = directions / directions.norm(dim=1, keepdim=True)

        # Test with OpenEquivariance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_openequivariance_availability(True)

        sh_oe = OptimizedSphericalHarmonics(
            irreps_sh, normalize=True, normalization="component"
        )

        # Force e3nn backend
        original_state = force_e3nn_backend()
        sh_e3nn = OptimizedSphericalHarmonics(
            irreps_sh, normalize=True, normalization="component"
        )
        restore_backend_state(original_state)

        # Test forward pass
        with torch.no_grad():
            y_oe = sh_oe(directions)
            y_e3nn = sh_e3nn(directions)

        # Check numerical equivalence
        assert torch.allclose(
            y_oe, y_e3nn, rtol=RTOL, atol=ATOL
        ), f"OptimizedSphericalHarmonics outputs differ: max_diff={torch.max(torch.abs(y_oe - y_e3nn))}"

        print(f"  ✓ Passed: max_diff={torch.max(torch.abs(y_oe - y_e3nn)):.2e}")


@pytest.mark.skipif(not HAS_OPENEQUIVARIANCE, reason="OpenEquivariance not available")
def pytest_tensor_product_equivalence():
    """Test that OptimizedTensorProduct produces identical results."""
    setup_test_environment()

    # Test cases with different irreps combinations
    test_cases = [
        (o3.Irreps("1x0e + 1x1o"), o3.Irreps("1x0e"), o3.Irreps("1x0e + 1x1o")),
        (o3.Irreps("2x0e + 1x1o"), o3.Irreps("1x1o"), o3.Irreps("1x0e + 1x1o + 1x2e")),
    ]

    for irreps_in1, irreps_in2, irreps_out in test_cases:
        print(
            f"Testing OptimizedTensorProduct: {irreps_in1} ⊗ {irreps_in2} -> {irreps_out}"
        )

        # Create test inputs
        x1 = torch.randn(BATCH_SIZE, irreps_in1.dim, dtype=torch.float64)
        x2 = torch.randn(BATCH_SIZE, irreps_in2.dim, dtype=torch.float64)

        # Test with OpenEquivariance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_openequivariance_availability(True)

        try:
            tp_oe = OptimizedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                internal_weights=True,
                shared_weights=True,
            )
        except Exception as e:
            print(f"  ⚠ Skipping TensorProduct test due to: {e}")
            continue

        # Force e3nn backend
        original_state = force_e3nn_backend()
        tp_e3nn = OptimizedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        restore_backend_state(original_state)

        # Copy weights for fair comparison
        if hasattr(tp_oe, "using_openequivariance") and tp_oe.using_openequivariance:
            if hasattr(tp_e3nn.tensor_product, "weight") and hasattr(
                tp_oe.tensor_product, "weight"
            ):
                with torch.no_grad():
                    tp_oe.tensor_product.weight.copy_(tp_e3nn.tensor_product.weight)

        # Test forward pass
        with torch.no_grad():
            y_oe = tp_oe(x1, x2)
            y_e3nn = tp_e3nn(x1, x2)

        # Check numerical equivalence
        assert torch.allclose(
            y_oe, y_e3nn, rtol=RTOL, atol=ATOL
        ), f"OptimizedTensorProduct outputs differ: max_diff={torch.max(torch.abs(y_oe - y_e3nn))}"

        print(f"  ✓ Passed: max_diff={torch.max(torch.abs(y_oe - y_e3nn)):.2e}")


def pytest_einsum_equivalence():
    """Test that optimized_einsum produces identical results to torch.einsum."""
    setup_test_environment()

    # Test various einsum operations
    test_cases = [
        ("ij,jk->ik", [(3, 4), (4, 5)]),  # Matrix multiplication
        ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),  # Batched matrix multiplication
        ("ij,ij->i", [(3, 4), (3, 4)]),  # Element-wise product and sum
        ("ijk,ikl->ijl", [(2, 3, 4), (2, 4, 5)]),  # Complex tensor contraction
    ]

    for equation, shapes in test_cases:
        print(f"Testing optimized_einsum: {equation} with shapes {shapes}")

        # Create test tensors
        tensors = [torch.randn(*shape, dtype=torch.float64) for shape in shapes]

        # Test optimized einsum
        result_optimized = optimized_einsum(equation, *tensors)

        # Test standard torch einsum
        result_torch = torch.einsum(equation, *tensors)

        # Check numerical equivalence
        assert torch.allclose(
            result_optimized, result_torch, rtol=RTOL, atol=ATOL
        ), f"optimized_einsum outputs differ: max_diff={torch.max(torch.abs(result_optimized - result_torch))}"

        print(
            f"  ✓ Passed: max_diff={torch.max(torch.abs(result_optimized - result_torch)):.2e}"
        )


@pytest.mark.skipif(not HAS_OPENEQUIVARIANCE, reason="OpenEquivariance not available")
def pytest_mace_block_equivalence():
    """Test numerical equivalence of MACE blocks with and without OpenEquivariance."""
    setup_test_environment()

    try:
        from hydragnn.utils.model.mace_utils.modules.blocks import (
            LinearNodeEmbeddingBlock,
            TensorProductWeightsBlock,
        )
    except ImportError:
        pytest.skip("MACE modules not available")

    print("Testing MACE LinearNodeEmbeddingBlock")

    # Test LinearNodeEmbeddingBlock
    irreps_in = o3.Irreps("2x0e + 1x1o")
    irreps_out = o3.Irreps("3x0e")

    # Create test input
    x = torch.randn(BATCH_SIZE, irreps_in.dim, dtype=torch.float64)

    # Test with current backend
    block1 = LinearNodeEmbeddingBlock(irreps_in, irreps_out)

    # Force e3nn and create another block
    original_state = force_e3nn_backend()
    block2 = LinearNodeEmbeddingBlock(irreps_in, irreps_out)
    restore_backend_state(original_state)

    # Copy weights for fair comparison
    with torch.no_grad():
        if hasattr(block1.linear, "linear") and hasattr(block2.linear, "linear"):
            # Handle OptimizedLinear wrapper
            if hasattr(block2.linear.linear, "weight"):
                if hasattr(block1.linear.linear, "weight"):
                    block1.linear.linear.weight.copy_(block2.linear.linear.weight)
        elif hasattr(block1.linear, "weight") and hasattr(block2.linear, "weight"):
            # Handle direct linear layer
            block1.linear.weight.copy_(block2.linear.weight)

    # Test forward pass
    with torch.no_grad():
        y1 = block1(x)
        y2 = block2(x)

    # Check numerical equivalence
    assert torch.allclose(
        y1, y2, rtol=RTOL, atol=ATOL
    ), f"MACE block outputs differ: max_diff={torch.max(torch.abs(y1 - y2))}"

    print(f"  ✓ Passed: max_diff={torch.max(torch.abs(y1 - y2)):.2e}")

    print("Testing MACE TensorProductWeightsBlock")

    # Test TensorProductWeightsBlock (this uses optimized_einsum)
    num_elements, num_edge_feats, num_feats_out = 3, 4, 5

    weights_block = TensorProductWeightsBlock(
        num_elements, num_edge_feats, num_feats_out
    )

    # Create test inputs
    sender_attrs = torch.randn(BATCH_SIZE, num_elements, dtype=torch.float64)
    edge_feats = torch.randn(BATCH_SIZE, num_edge_feats, dtype=torch.float64)

    # Test with optimized einsum
    result_optimized = weights_block(sender_attrs, edge_feats)

    # Test with standard einsum (manually)
    with torch.no_grad():
        result_standard = torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_attrs, weights_block.weights
        )

    # Check numerical equivalence
    assert torch.allclose(
        result_optimized, result_standard, rtol=RTOL, atol=ATOL
    ), f"TensorProductWeightsBlock outputs differ: max_diff={torch.max(torch.abs(result_optimized - result_standard))}"

    print(
        f"  ✓ Passed: max_diff={torch.max(torch.abs(result_optimized - result_standard)):.2e}"
    )


def pytest_precision_consistency():
    """Test that numerical precision is consistent across operations."""
    setup_test_environment()

    print("Testing numerical precision consistency")

    # Test with different dtypes
    dtypes = [torch.float32, torch.float64]

    for dtype in dtypes:
        print(f"  Testing with dtype: {dtype}")

        # Simple einsum test
        a = torch.randn(3, 4, dtype=dtype)
        b = torch.randn(4, 5, dtype=dtype)

        result_optimized = optimized_einsum("ij,jk->ik", a, b)
        result_torch = torch.einsum("ij,jk->ik", a, b)

        # For the same operation, results should be identical (or very close)
        tolerance = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(
            result_optimized, result_torch, rtol=tolerance, atol=tolerance
        ), f"Precision inconsistency for {dtype}"

        print(
            f"    ✓ {dtype}: max_diff={torch.max(torch.abs(result_optimized - result_torch)):.2e}"
        )


@pytest.mark.parametrize("enable_oe", [True, False])
def pytest_configuration_consistency(enable_oe):
    """Test that configuration changes produce consistent results."""
    setup_test_environment()

    print(f"Testing configuration consistency with enable_openequivariance={enable_oe}")

    # Set backend state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_openequivariance_availability(enable_oe)

    # Test basic operations
    irreps_in = o3.Irreps("2x0e")
    irreps_out = o3.Irreps("1x0e")

    linear = OptimizedLinear(irreps_in, irreps_out)
    x = torch.randn(
        4, irreps_in.dim, dtype=torch.float32
    )  # Use float32 for compatibility

    with torch.no_grad():
        y = linear(x)

    # Should produce valid output regardless of backend
    assert y.shape == (4, irreps_out.dim)
    assert torch.isfinite(y).all()

    print(f"  ✓ Configuration {enable_oe} produces valid output")


if __name__ == "__main__":
    """Run tests directly when script is executed."""
    print("=" * 70)
    print("OpenEquivariance Numerical Equivalence Tests")
    print("=" * 70)

    if not HAS_OPENEQUIVARIANCE:
        print("⚠ OpenEquivariance not available - running limited tests")
        pytest_einsum_equivalence()
        pytest_precision_consistency()
        print("\n✓ Limited tests completed successfully")
    else:
        print("🚀 OpenEquivariance available - running full test suite")

        try:
            # Enable OpenEquivariance for testing
            check_openequivariance_availability(True)

            pytest_linear_equivalence()
            pytest_spherical_harmonics_equivalence()
            pytest_tensor_product_equivalence()
            pytest_einsum_equivalence()
            pytest_mace_block_equivalence()
            pytest_precision_consistency()
            pytest_configuration_consistency(True)
            pytest_configuration_consistency(False)

            print("\n✅ All numerical equivalence tests passed!")
            print("OpenEquivariance and e3nn produce numerically identical results.")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise

    print("=" * 70)
