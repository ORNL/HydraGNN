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
"""Cross-rank reductions and gathers for loss values and predictions."""

import os

import torch
import torch.distributed as dist

from hydragnn.distributed import get_device


@torch.no_grad()
def reduce_values_ranks_dist(local_tensor):
    if dist.get_world_size() > 1:
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        local_tensor = local_tensor / dist.get_world_size()
    return local_tensor


@torch.no_grad()
def reduce_values_ranks_mpi(local_tensor):
    if dist.get_world_size() > 1:
        from mpi4py import MPI

        local_tensor = MPI.COMM_WORLD.allreduce(
            local_tensor.detach().cpu().numpy(), op=MPI.SUM
        )
        local_tensor = torch.tensor(local_tensor) / dist.get_world_size()
    return local_tensor


def reduce_values_ranks(local_tensor):
    backend = os.getenv("HYDRAGNN_AGGR_BACKEND", "torch")
    if backend == "mpi":
        return reduce_values_ranks_mpi(local_tensor)
    else:
        return reduce_values_ranks_dist(local_tensor)


@torch.no_grad()
def gather_tensor_ranks(head_values):
    if dist.get_world_size() > 1:
        head_values = head_values.to(get_device())
        size_local = torch.tensor(
            [head_values.shape[0]], dtype=torch.int64, device=head_values.device
        )
        size_all = [torch.ones_like(size_local) for _ in range(dist.get_world_size())]
        dist.all_gather(size_all, size_local)
        size_all = torch.cat(size_all, 0)
        max_size = size_all.max()

        padded = torch.empty(
            max_size,
            *head_values.shape[1:],
            dtype=head_values.dtype,
            device=head_values.device,
        )
        padded[: head_values.shape[0]] = head_values

        tensor_list = [torch.ones_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, padded)
        tensor_list = torch.cat(tensor_list, 0)

        head_values = torch.zeros(
            size_all.sum(),
            *head_values.shape[1:],
            dtype=head_values.dtype,
            device=head_values.device,
        )
        for i, size in enumerate(size_all):
            start_idx = i * max_size
            end_idx = start_idx + size.item()
            if end_idx > start_idx:
                head_values[size_all[:i].sum() : size_all[:i].sum() + size.item()] = (
                    tensor_list[start_idx:end_idx]
                )

    return head_values
