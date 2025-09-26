##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory
# All rights reserved.
#
# This file is part of HydraGNN and is distributed under a BSD 3-clause
# license. For the licensing terms see the LICENSE file in the top-level
# directory.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

"""
Unified, parametrized tests for comparing e3nn and OpenEquivariance tensor products.

What this file tests:
- Shape & finiteness checks for outputs across backends
- Optional comparison vs OpenEquivariance if installed
- Lightweight timing measurements (warmup + trials)

Redundancy removed by:
- Shared fixtures for device and inputs
- Single parametrized test body
- Common helpers for model construction and timing
"""

import os
import sys
import time
import pytest
import torch
from e3nn import o3

# Import the compatibility module directly to avoid hydragnn deps
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "hydragnn", "utils", "model"
    ),
)
from equivariance_compat import is_openequivariance_available

# Optional: small, consistent tolerance if you later add numeric equality checks
_TOL = 1e-6


# ---------- Helpers ----------


def _make_inputs(irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, batch_size: int, device):
    torch.manual_seed(0)
    x1 = torch.randn(batch_size, irreps_in1.dim, device=device)
    x2 = torch.randn(batch_size, irreps_in2.dim, device=device)
    return x1, x2


def _make_e3nn_pure(irreps_in1, irreps_in2, irreps_out, device):
    return o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=True, internal_weights=True
    ).to(device)


def _make_compat(irreps_in1, irreps_in2, irreps_out, device, use_oeq: bool):
    # Import locally to avoid import cost if unused
    from equivariance_compat import TensorProduct as CompatTensorProduct

    return CompatTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        shared_weights=True,
        internal_weights=True,
        use_openequivariance=use_oeq,
    ).to(device)


def _time_op(fn, x1, x2, warmup: int = 3, trials: int = 8) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn(x1, x2)
    # Timed runs
    start = time.time()
    for _ in range(trials):
        _ = fn(x1, x2)
    return (time.time() - start) / trials


# ---------- Fixtures & parameters ----------


@pytest.fixture(scope="module")
def device():
    # Keep CPU for CI portability; easy to switch to CUDA later
    return torch.device("cpu")


# Irreps cases and batch sizes (covers your previous choices)
IRREPS_CASES = [
    # Case 1: direct L=1 ⊗ L=1 → L=0 + L=2 (your “direct e3nn comparison” case)
    (
        o3.Irreps("1x1e"),
        o3.Irreps("1x1e"),
        o3.Irreps("1x0e + 1x2e"),
        50,
        "basic_l1_l1",
    ),
    # Case 2: a slightly richer mix (your “timing measurement” case)
    (
        o3.Irreps("2x0e + 1x1e"),
        o3.Irreps("1x1e"),
        o3.Irreps("2x0e + 2x1e + 1x2e"),
        30,
        "mixed_0_1_2",
    ),
    # Case 3: smaller batch, same L=1 ⊗ L=1 for OEQ vs e3nn
    (
        o3.Irreps("1x1e"),
        o3.Irreps("1x1e"),
        o3.Irreps("1x0e + 1x2e"),
        20,
        "oeq_compare",
    ),
]


# Backend modes to test:
# - ("pure_e3nn", None)        → e3nn native TP only
# - ("compat_e3nn", False)     → compat wrapper forcing e3nn
# - ("compat_oeq", True)       → compat wrapper forcing OpenEquivariance (skip if unavailable)
BACKENDS = [
    ("pure_e3nn", None),
    ("compat_e3nn", False),
    ("compat_oeq", True),
]


# ---------- The single, unified test ----------


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "irreps_in1,irreps_in2,irreps_out,batch_size,case_name",
    IRREPS_CASES,
    ids=[c[-1] for c in IRREPS_CASES],
)
@pytest.mark.parametrize("backend_name,use_oeq", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_tensorproduct_shapes_finiteness_and_timing(
    device,
    irreps_in1,
    irreps_in2,
    irreps_out,
    batch_size,
    case_name,
    backend_name,
    use_oeq,
    capsys,
):
    """
    Unified test:
    - Builds inputs once
    - Selects backend (pure e3nn / compat-e3nn / compat-OEQ)
    - Runs timing
    - Checks shape & finiteness
    - Prints consistent timing summary

    Notes:
    - We don't enforce identical values across backends because weights/implementations differ.
    - If OpenEquivariance is not present, the OEQ case is skipped (not failed).
    """

    # Handle optional dependency
    if use_oeq is True and not is_openequivariance_available():
        pytest.skip("OpenEquivariance not available; skipping compat_oeq backend.")

    # Build inputs
    x1, x2 = _make_inputs(irreps_in1, irreps_in2, batch_size, device)

    # Build model according to backend
    if backend_name == "pure_e3nn":
        model = _make_e3nn_pure(irreps_in1, irreps_in2, irreps_out, device)
    else:
        model = _make_compat(irreps_in1, irreps_in2, irreps_out, device, use_oeq)

    # Time it and compute one result for assertions
    avg_time = _time_op(model, x1, x2, warmup=3, trials=8)
    result = model(x1, x2)

    # Assertions shared by all previous tests
    assert result.shape == (
        batch_size,
        irreps_out.dim,
    ), f"Unexpected shape for {backend_name}/{case_name}: {result.shape} vs {(batch_size, irreps_out.dim)}"
    assert torch.isfinite(
        result
    ).all(), f"Non-finite values for {backend_name}/{case_name}"

    # Print a concise timing line to the test log
    print(
        f"[{case_name:>12}] backend={backend_name:>12}  time/op={avg_time:.6f}s  "
        f"batch={batch_size:<4} out_dim={irreps_out.dim:<4} elems/sec={batch_size*irreps_out.dim/avg_time:,.0f}"
    )
    # Ensure the line appears in test output if you run with -s
    capsys.readouterr()
