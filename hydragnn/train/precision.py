##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Numerical precision selection, autocast, and batch device placement."""

from contextlib import nullcontext

import torch

from hydragnn.distributed import get_device

PRECISION_MAP = {
    # Keep optimizer/master parameters in FP32 for BF16 runs and use autocast for compute.
    # This avoids backend-specific optimizer paths that require FP32 parameters.
    "bf16": {"param_dtype": torch.float32, "autocast_dtype": torch.bfloat16},
    "fp32": {"param_dtype": torch.float32, "autocast_dtype": None},
    "fp64": {"param_dtype": torch.float64, "autocast_dtype": None},
}


def resolve_precision(precision: str):
    """Normalize precision string and return parameter/autocast dtypes."""

    if precision is None:
        precision = "fp32"
    prec = str(precision).lower()
    aliases = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "float64": "fp64",
        "double": "fp64",
    }
    prec = aliases.get(prec, prec)
    if prec not in PRECISION_MAP:
        raise ValueError(
            f"Unsupported precision {precision}. Choose from {list(PRECISION_MAP.keys())}."
        )
    info = PRECISION_MAP[prec]
    return prec, info["param_dtype"], info["autocast_dtype"]


def move_batch_to_device(data, param_dtype):
    device = get_device()

    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=param_dtype)
    else:
        data = data.to(device)
        if hasattr(data, "stores"):
            for store in data.stores:
                for key, value in store.items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(device=device)
                        if torch.is_floating_point(value):
                            value = value.to(dtype=param_dtype)
                        store[key] = value
        else:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device)
                    if torch.is_floating_point(value):
                        value = value.to(dtype=param_dtype)
                    data[key] = value

    return data


def get_autocast_and_scaler(precision):
    precision, _, autocast_dtype = resolve_precision(precision)

    if precision == "bf16":
        device_type = str(get_device())
        cpu_bf16 = bool(getattr(torch.backends.cpu, "has_bf16", False))
        use_bf16 = (
            torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 7
        ) or (hasattr(torch, "xpu") and torch.xpu.is_available())
        use_bf16 = use_bf16 or (device_type == "cpu" and cpu_bf16)
        if not use_bf16:
            print(
                f"Requested bf16 but unsupported on {device_type}; falling back to full precision."
            )

        autocast = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if use_bf16
            else nullcontext()
        )
        return autocast, None

    return nullcontext(), None


def first_float_tensor_dtype(obj):
    if torch.is_tensor(obj):
        return obj.dtype if torch.is_floating_point(obj) else None

    if isinstance(obj, dict):
        values = obj.values()
    elif hasattr(obj, "items"):
        values = (value for _, value in obj.items())
    else:
        values = []

    for value in values:
        dtype = first_float_tensor_dtype(value)
        if dtype is not None:
            return dtype

    return None


def first_optimizer_state_dtype(optimizer):
    if optimizer is None:
        return None

    if hasattr(optimizer, "optimizer1") and hasattr(optimizer, "optimizer2"):
        for subopt in (optimizer.optimizer1, optimizer.optimizer2):
            dtype = first_optimizer_state_dtype(subopt)
            if dtype is not None:
                return dtype
        return None

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                return value.dtype

    return None
