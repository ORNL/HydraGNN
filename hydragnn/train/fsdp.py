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
"""Helpers for FSDP2 sharding control and verification during training."""

import os

import torch.distributed as dist


def is_fsdp2_enabled():
    return (
        bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0")))
        and int(os.getenv("HYDRAGNN_FSDP_VERSION", "1")) == 2
    )


def set_reshard_after_backward(model, enabled):
    setter = getattr(model, "set_reshard_after_backward", None)
    if callable(setter):
        setter(enabled)
        return True

    wrapped = getattr(model, "module", None)
    setter = getattr(wrapped, "set_reshard_after_backward", None)
    if callable(setter):
        setter(enabled)
        return True

    return False


def verify_fsdp2_sharding(model, phase):
    """Assert that FSDP2 parameters or gradients use sharded DTensors."""
    from torch.distributed.tensor import DTensor, Shard

    tensors = []
    unsharded = []
    for name, parameter in model.named_parameters():
        tensor = parameter if phase == "parameters" else parameter.grad
        if tensor is None:
            continue
        tensors.append(tensor)
        if not isinstance(tensor, DTensor) or not any(
            isinstance(placement, Shard) for placement in tensor.placements
        ):
            unsharded.append(name)

    if not tensors:
        raise RuntimeError(f"FSDP2 sharding verification found no {phase}")
    if unsharded:
        names = ", ".join(unsharded[:5])
        raise RuntimeError(
            f"FSDP2 sharding verification found unsharded {phase}: {names}"
        )

    global_numel = sum(tensor.numel() for tensor in tensors)
    local_numel = sum(tensor.to_local().numel() for tensor in tensors)
    if dist.get_world_size() > 1 and local_numel >= global_numel:
        raise RuntimeError(
            f"FSDP2 {phase} were not partitioned: "
            f"local_numel={local_numel}, global_numel={global_numel}"
        )
    if dist.get_rank() == 0:
        print(
            f"[fsdp2-sharding] phase={phase} tensors={len(tensors)} "
            f"local_numel={local_numel} global_numel={global_numel} "
            "placement=Shard(0)"
        )
