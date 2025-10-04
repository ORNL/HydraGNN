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
OpenEquivariance integration utilities for optimized tensor operations.
Provides fallback to e3nn when OpenEquivariance is not available.
"""

import warnings
import torch
from typing import Optional, Union, List, Tuple, Any

# Try to import OpenEquivariance
try:
    import openequivariance

    HAS_OPENEQUIVARIANCE = True
except ImportError as e:
    HAS_OPENEQUIVARIANCE = False
    openequivariance = None
    # Store the import error for more detailed diagnostics if needed
    _IMPORT_ERROR = str(e)
except Exception as e:
    # Handle other exceptions like CUDA setup issues
    HAS_OPENEQUIVARIANCE = False
    openequivariance = None
    _IMPORT_ERROR = f"OpenEquivariance import failed: {str(e)}"

# Always import e3nn as fallback
from e3nn import o3, nn as e3nn_nn
from e3nn.util.jit import compile_mode

# Global flag to control usage
_USE_OPENEQUIVARIANCE = False


def check_openequivariance_availability(enable_openequivariance: bool = False) -> bool:
    """
    Check if OpenEquivariance is available and configure its usage.

    Args:
        enable_openequivariance: Whether to attempt to enable OpenEquivariance

    Returns:
        bool: True if OpenEquivariance is available and will be used, False otherwise
    """
    global _USE_OPENEQUIVARIANCE

    if not enable_openequivariance:
        _USE_OPENEQUIVARIANCE = False
        return False

    if not HAS_OPENEQUIVARIANCE:
        if "_IMPORT_ERROR" in globals():
            if "CUDA" in _IMPORT_ERROR:
                warnings.warn(
                    "OpenEquivariance is installed but CUDA is not properly configured. "
                    "Please ensure CUDA is installed and CUDA_HOME environment variable is set "
                    "to enable OpenEquivariance optimizations. Falling back to e3nn.\n"
                    f"Detailed error: {_IMPORT_ERROR}",
                    UserWarning,
                )
            else:
                warnings.warn(
                    f"OpenEquivariance import failed: {_IMPORT_ERROR}. "
                    "Falling back to e3nn.",
                    UserWarning,
                )
        else:
            warnings.warn(
                "OpenEquivariance is not available in the environment. "
                "Please install OpenEquivariance from https://github.com/PASSIONLab/OpenEquivariance "
                "to enable optimized tensor operations. Falling back to e3nn.",
                UserWarning,
            )
        _USE_OPENEQUIVARIANCE = False
        return False

    # Check if OpenEquivariance has the required functionality
    try:
        # Verify key components are available
        if not hasattr(openequivariance, "TensorProduct"):
            warnings.warn(
                "OpenEquivariance installation appears incomplete (missing TensorProduct). "
                "Falling back to e3nn.",
                UserWarning,
            )
            _USE_OPENEQUIVARIANCE = False
            return False

        _USE_OPENEQUIVARIANCE = True
        print(
            "OpenEquivariance is available and will be used for optimized tensor operations."
        )
        return True

    except Exception as e:
        warnings.warn(
            f"OpenEquivariance check failed with error: {e}. " "Falling back to e3nn.",
            UserWarning,
        )
        _USE_OPENEQUIVARIANCE = False
        return False


def is_openequivariance_enabled() -> bool:
    """Check if OpenEquivariance is currently enabled."""
    return _USE_OPENEQUIVARIANCE


class OptimizedTensorProduct(torch.nn.Module):
    """
    Tensor product operation with OpenEquivariance optimization when available.
    Falls back to e3nn when OpenEquivariance is not available.
    """

    def __init__(
        self,
        irreps_in1: Union[str, o3.Irreps],
        irreps_in2: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        instructions: Optional[List] = None,
        shared_weights: bool = True,
        internal_weights: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Convert to e3nn Irreps format
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        if _USE_OPENEQUIVARIANCE and openequivariance is not None:
            try:
                # Use OpenEquivariance implementation
                self.tensor_product = openequivariance.TensorProduct(
                    irreps_in1=str(self.irreps_in1),
                    irreps_in2=str(self.irreps_in2),
                    irreps_out=str(self.irreps_out),
                    instructions=instructions,
                    shared_weights=shared_weights,
                    internal_weights=internal_weights,
                    **kwargs,
                )
                self.using_openequivariance = True
            except Exception as e:
                warnings.warn(
                    f"Failed to create OpenEquivariance TensorProduct: {e}. "
                    "Falling back to e3nn.",
                    UserWarning,
                )
                self._create_e3nn_tensor_product(
                    instructions, shared_weights, internal_weights, kwargs
                )
                self.using_openequivariance = False
        else:
            self._create_e3nn_tensor_product(
                instructions, shared_weights, internal_weights, kwargs
            )
            self.using_openequivariance = False

    def _create_e3nn_tensor_product(
        self, instructions, shared_weights, internal_weights, kwargs
    ):
        """Create fallback e3nn tensor product."""
        # Filter out None values from kwargs to prevent issues
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self.tensor_product = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            **filtered_kwargs,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through tensor product."""
        if weight is not None:
            return self.tensor_product(x1, x2, weight)
        else:
            return self.tensor_product(x1, x2)


class OptimizedSphericalHarmonics(torch.nn.Module):
    """
    Spherical harmonics computation with OpenEquivariance optimization when available.
    Falls back to e3nn when OpenEquivariance is not available.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps],
        normalize: bool = True,
        normalization: str = "component",
        **kwargs,
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)

        if _USE_OPENEQUIVARIANCE and openequivariance is not None:
            try:
                # Use OpenEquivariance implementation if available
                if hasattr(openequivariance, "SphericalHarmonics"):
                    self.spherical_harmonics = openequivariance.SphericalHarmonics(
                        irreps_out=str(self.irreps_out),
                        normalize=normalize,
                        normalization=normalization,
                        **kwargs,
                    )
                    self.using_openequivariance = True
                else:
                    # Fallback if SphericalHarmonics not available in OpenEquivariance
                    self._create_e3nn_spherical_harmonics(
                        normalize, normalization, kwargs
                    )
                    self.using_openequivariance = False
            except Exception as e:
                warnings.warn(
                    f"Failed to create OpenEquivariance SphericalHarmonics: {e}. "
                    "Falling back to e3nn.",
                    UserWarning,
                )
                self._create_e3nn_spherical_harmonics(normalize, normalization, kwargs)
                self.using_openequivariance = False
        else:
            self._create_e3nn_spherical_harmonics(normalize, normalization, kwargs)
            self.using_openequivariance = False

    def _create_e3nn_spherical_harmonics(self, normalize, normalization, kwargs):
        """Create fallback e3nn spherical harmonics."""
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.irreps_out, normalize=normalize, normalization=normalization, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spherical harmonics."""
        return self.spherical_harmonics(x)


class OptimizedLinear(torch.nn.Module):
    """
    Linear layer with OpenEquivariance optimization when available.
    Falls back to e3nn when OpenEquivariance is not available.
    """

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        internal_weights: bool = True,
        shared_weights: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        if _USE_OPENEQUIVARIANCE and openequivariance is not None:
            try:
                # Use OpenEquivariance implementation if available
                if hasattr(openequivariance, "Linear"):
                    self.linear = openequivariance.Linear(
                        irreps_in=str(self.irreps_in),
                        irreps_out=str(self.irreps_out),
                        internal_weights=internal_weights,
                        shared_weights=shared_weights,
                        **kwargs,
                    )
                    self.using_openequivariance = True
                else:
                    # Fallback if Linear not available in OpenEquivariance
                    self._create_e3nn_linear(internal_weights, shared_weights, kwargs)
                    self.using_openequivariance = False
            except Exception as e:
                warnings.warn(
                    f"Failed to create OpenEquivariance Linear: {e}. "
                    "Falling back to e3nn.",
                    UserWarning,
                )
                self._create_e3nn_linear(internal_weights, shared_weights, kwargs)
                self.using_openequivariance = False
        else:
            self._create_e3nn_linear(internal_weights, shared_weights, kwargs)
            self.using_openequivariance = False

    def _create_e3nn_linear(self, internal_weights, shared_weights, kwargs):
        """Create fallback e3nn linear layer."""
        self.linear = o3.Linear(
            self.irreps_in,
            self.irreps_out,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            **kwargs,
        )

    def forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through linear layer."""
        if weight is not None:
            return self.linear(x, weight)
        else:
            return self.linear(x)


def optimized_einsum(equation: str, *operands) -> torch.Tensor:
    """
    Optimized einsum operation using OpenEquivariance when available.
    Falls back to torch.einsum when OpenEquivariance is not available.

    Args:
        equation: Einstein summation equation
        *operands: Input tensors

    Returns:
        torch.Tensor: Result of einsum operation
    """
    if _USE_OPENEQUIVARIANCE and openequivariance is not None:
        try:
            # Use OpenEquivariance optimized einsum if available
            if hasattr(openequivariance, "einsum"):
                return openequivariance.einsum(equation, *operands)
        except Exception as e:
            warnings.warn(
                f"Failed to use OpenEquivariance einsum: {e}. "
                "Falling back to torch.einsum.",
                UserWarning,
            )

    # Fallback to standard torch.einsum
    return torch.einsum(equation, *operands)
