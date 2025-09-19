###########################################################################################
# Compatibility wrapper for accelerated Clebsch-Gordon tensor products
# This module provides a unified interface that can use OpenEquivariance when available
# for faster tensor products, falling back to e3nn when OpenEquivariance is not installed
# Authors: HydraGNN team
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import warnings
from typing import Optional, List, Any, Dict, Tuple
from e3nn import o3
import torch

# Import the instruction generation function
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    """Generate tensor product instructions compatible with e3nn (copied from irreps_tools)."""
    trainable = True

    # Convert target_irreps to a set of Irrep objects for efficient lookup
    target_irrep_set = set(ir for mul, ir in target_irreps)

    # Collect possible irreps and their instructions
    irreps_out_list = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irrep_set:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


# Global flag to track OpenEquivariance availability
_OPENEQUIVARIANCE_AVAILABLE = None
_OPENEQUIVARIANCE_MODULE = None


def _check_openequivariance():
    """Check if OpenEquivariance is available and can be imported."""
    global _OPENEQUIVARIANCE_AVAILABLE, _OPENEQUIVARIANCE_MODULE

    if _OPENEQUIVARIANCE_AVAILABLE is not None:
        return _OPENEQUIVARIANCE_AVAILABLE

    try:
        import openequivariance as oeq

        _OPENEQUIVARIANCE_MODULE = oeq
        _OPENEQUIVARIANCE_AVAILABLE = True
        logging.info(
            "OpenEquivariance is available and will be used for tensor products"
        )
        return True
    except ImportError as e:
        logging.debug(f"OpenEquivariance not available: {e}")
        _OPENEQUIVARIANCE_AVAILABLE = False
        return False


def is_openequivariance_available() -> bool:
    """Check if OpenEquivariance is available for use."""
    return _check_openequivariance()


class TensorProduct(torch.nn.Module):
    """
    A compatibility wrapper for tensor products that uses OpenEquivariance when available,
    falling back to e3nn's TensorProduct when not available.

    This module provides the same interface as e3nn.o3.TensorProduct but with potential
    acceleration from OpenEquivariance.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        normalization: str = "component",
        path_normalization: str = "element",
        gradient_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        checkname: bool = True,
        use_openequivariance: Optional[bool] = None,
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        # Generate instructions if not provided - let e3nn handle default case
        if instructions is None:
            # Don't generate custom instructions, let e3nn handle it
            instructions = None
        self.instructions = instructions

        # Determine whether to use OpenEquivariance
        if use_openequivariance is None:
            self.use_oeq = _check_openequivariance()
        else:
            self.use_oeq = use_openequivariance and _check_openequivariance()

        if use_openequivariance and not self.use_oeq:
            warnings.warn(
                "OpenEquivariance was requested but is not available, falling back to e3nn",
                UserWarning,
            )

        # Initialize the appropriate backend
        try:
            if self.use_oeq:
                logging.debug("Initializing OpenEquivariance backend")
                self._init_openequivariance()
            else:
                logging.debug("Initializing e3nn backend")
                self._init_e3nn(
                    normalization=normalization,
                    path_normalization=path_normalization,
                    gradient_normalization=gradient_normalization,
                    checkname=checkname,
                )
        except Exception as e:
            logging.error(f"Failed to initialize TensorProduct: {e}")
            raise

    def _generate_instructions(self):
        """Generate proper tensor product instructions using MACE-style logic."""
        return tp_out_irreps_with_instructions(
            self.irreps_in1, self.irreps_in2, self.irreps_out
        )[
            1
        ]  # Return just the instructions, not the irreps

    def _init_openequivariance(self):
        """Initialize OpenEquivariance tensor product."""
        try:
            oeq = _OPENEQUIVARIANCE_MODULE

            # Convert e3nn instructions to OpenEquivariance format if needed
            if self.instructions is None:
                # Generate default instructions similar to e3nn
                instructions = []
                target_irreps = set(irrep.ir for mul, irrep in self.irreps_out)
                for i, (mul_ir1, ir1) in enumerate(self.irreps_in1):
                    for j, (mul_ir2, ir2) in enumerate(self.irreps_in2):
                        for ir_out in ir1 * ir2:
                            if ir_out in target_irreps:
                                mode = "uvu" if not self.internal_weights else "uvw"
                                instructions.append((i, j, 0, mode, True))
                self.instructions = instructions

            # Create OpenEquivariance TPProblem
            self.tp_problem = oeq.TPProblem(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                shared_weights=self.shared_weights
                if self.shared_weights is not None
                else True,
                internal_weights=False,  # OpenEquivariance requires internal_weights=False
            )

            # Create OpenEquivariance TensorProduct
            self.tp_backend = oeq.TensorProduct(self.tp_problem, torch_op=True)
            self.weight_numel = self.tp_problem.weight_numel

        except Exception as e:
            logging.warning(
                f"Failed to initialize OpenEquivariance, falling back to e3nn: {e}"
            )
            self.use_oeq = False
            self._init_e3nn()

    def _init_e3nn(
        self,
        normalization="component",
        path_normalization="element",
        gradient_normalization="element",
        checkname=True,
    ):
        """Initialize e3nn tensor product."""
        try:
            # Use the correct parameter names for e3nn 0.5.1
            # Set sensible defaults for None values
            internal_weights = (
                self.internal_weights if self.internal_weights is not None else True
            )
            shared_weights = (
                self.shared_weights if self.shared_weights is not None else False
            )

            # Create the tensor product, letting e3nn generate instructions if None
            if self.instructions is None:
                self.tp_backend = o3.TensorProduct(
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_out,
                    irrep_normalization=normalization,
                    path_normalization=path_normalization,
                    internal_weights=internal_weights,
                    shared_weights=shared_weights,
                )
            else:
                self.tp_backend = o3.TensorProduct(
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_out,
                    instructions=self.instructions,
                    irrep_normalization=normalization,
                    path_normalization=path_normalization,
                    internal_weights=internal_weights,
                    shared_weights=shared_weights,
                )
            self.weight_numel = self.tp_backend.weight_numel
            logging.debug(
                f"e3nn TensorProduct initialized successfully with weight_numel={self.weight_numel}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize e3nn TensorProduct: {e}")
            raise

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, weight: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the tensor product.

        Args:
            x1: First input tensor
            x2: Second input tensor
            weight: Optional weight tensor

        Returns:
            Output tensor from tensor product
        """
        if self.use_oeq:
            # OpenEquivariance requires weight tensor
            if weight is None:
                raise ValueError(
                    "OpenEquivariance tensor product requires weight tensor"
                )
            return self.tp_backend(x1, x2, weight)
        else:
            # e3nn handles internal weights automatically
            if weight is not None:
                return self.tp_backend(x1, x2, weight)
            else:
                return self.tp_backend(x1, x2)

    def __repr__(self):
        backend = "OpenEquivariance" if self.use_oeq else "e3nn"
        return (
            f"TensorProduct({self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} "
            f"using {backend})"
        )


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about available tensor product backends.

    Returns:
        Dictionary containing backend information
    """
    info = {
        "e3nn_available": True,
        "openequivariance_available": _check_openequivariance(),
        "default_backend": "openequivariance" if _check_openequivariance() else "e3nn",
    }

    if _check_openequivariance():
        try:
            info["openequivariance_version"] = _OPENEQUIVARIANCE_MODULE.__version__
        except:
            info["openequivariance_version"] = "unknown"

    return info


def set_default_backend(backend: str):
    """
    Set the default tensor product backend.

    Args:
        backend: Backend to use ("openequivariance", "e3nn", or "auto")
    """
    global _FORCE_BACKEND
    if backend not in ["openequivariance", "e3nn", "auto"]:
        raise ValueError(f"Invalid backend: {backend}")
    _FORCE_BACKEND = backend


# Module-level backend forcing (for testing/debugging)
_FORCE_BACKEND = None


def _should_use_openequivariance(use_openequivariance: Optional[bool] = None) -> bool:
    """Determine whether to use OpenEquivariance based on availability and preferences."""
    if _FORCE_BACKEND == "e3nn":
        return False
    elif _FORCE_BACKEND == "openequivariance":
        if not _check_openequivariance():
            warnings.warn(
                "OpenEquivariance forced but not available, falling back to e3nn",
                UserWarning,
            )
            return False
        return True
    else:  # auto or None
        if use_openequivariance is None:
            return _check_openequivariance()
        return use_openequivariance and _check_openequivariance()
