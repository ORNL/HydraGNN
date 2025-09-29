"""Backend abstraction layer for equivariant tensor operations.

Supports:
  - e3nn (default)
  - OpenEquivariance (if installed) via automatic TPProblem conversion

Selection priority:
  1. Environment variable HYDRAGNN_EQUIVARIANCE_BACKEND
  2. config["NeuralNetwork"]["Architecture"]["equivariance_backend"]
  3. default: "e3nn"
"""
from __future__ import annotations
import os
import warnings
from typing import Sequence, Tuple

from e3nn import o3
from hydragnn.utils.model.openequiv_adapter import (
    build_oeq_tensor_product_module,
)

__all__ = [
    "initialize_equivariance_backend",
    "get_backend_name",
    "Linear",
    "build_tensor_product",
]

_requested_backend = None
_effective_backend = None
_warned_fallback = False
_initialized = False


def initialize_equivariance_backend(config: dict | None):
    select_backend(config)
    maybe_warn_fallback()


def select_backend(config: dict | None):
    global _requested_backend, _effective_backend, _initialized
    if _initialized:
        return _effective_backend
    env_choice = os.environ.get("HYDRAGNN_EQUIVARIANCE_BACKEND", "").strip().lower()
    cfg_choice = None
    if config is not None:
        cfg_choice = (
            config.get("NeuralNetwork", {})
            .get("Architecture", {})
            .get("equivariance_backend", "")
        )
        if cfg_choice:
            cfg_choice = cfg_choice.strip().lower()
    requested = env_choice or (cfg_choice or "e3nn")
    if requested not in ("e3nn", "openequivariance", "openequiv"):
        warnings.warn(
            f"[HydraGNN] Unknown equivariance backend '{requested}', falling back to 'e3nn'."
        )
        requested = "e3nn"
    if requested == "openequiv":
        requested = "openequivariance"
    _requested_backend = requested
    if requested == "openequivariance":
        try:
            import openequivariance  # noqa: F401
            _effective_backend = "openequivariance"
        except Exception:  # pragma: no cover
            _effective_backend = "e3nn"
    else:
        _effective_backend = "e3nn"
    _initialized = True
    return _effective_backend


def get_backend_name():
    return _effective_backend or "e3nn"


def maybe_warn_fallback():
    global _warned_fallback
    if (
        _requested_backend == "openequivariance"
        and get_backend_name() != "openequivariance"
        and not _warned_fallback
    ):
        warnings.warn(
            "[HydraGNN] OpenEquivariance requested but not available. Falling back to e3nn."
        )
        _warned_fallback = True


def Linear(*args, **kwargs):
    """Wrapper (placeholder for future backend-specific Linear)."""
    return o3.Linear(*args, **kwargs)


def build_tensor_product(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: Sequence[Tuple],
    *,
    shared_weights: bool,
    internal_weights: bool,
    **kwargs,
):
    """Create a tensor product module in the active backend.

    Returns (module, weight_numel, backend_used)
    """
    backend = get_backend_name()
    if backend == "openequivariance":
        tp_module, weight_numel = build_oeq_tensor_product_module(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights,
            internal_weights,
            torch_op=kwargs.get("torch_op", True),
            use_opaque=kwargs.get("use_opaque", False),
        )
        if tp_module is not None:
            return tp_module, weight_numel, "openequivariance"
        # fall through to e3nn on failure
    tp = o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    return tp, tp.weight_numel, "e3nn"