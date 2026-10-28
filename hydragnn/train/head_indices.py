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
"""Locate each output head's targets inside the concatenated ``data.y`` tensor."""

import torch


def get_head_indices(model, data):
    """In data.y (the true value here), all feature variables for a mini-batch are concatenated together as a large list.
    To calculate loss function, we need to know true value for each feature in every head.
    This function is to get the feature/head index/location in the large list."""
    if all(ele == "graph" for ele in model.module.head_type):
        return get_head_indices_graph(model, data)
    else:
        return get_head_indices_node_or_mixed(model, data)


def get_head_indices_graph(model, data):
    """this is for cases when outputs are all at graph level"""
    # total length
    nsize = data.y.shape[0]
    # feature index for all heads
    head_index = [None] * model.module.num_heads
    if model.module.num_heads == 1:
        head_index[0] = torch.arange(nsize)
        return head_index
    # dimensions of all heads
    head_dims = model.module.head_dims
    head_dimsum = sum(head_dims)

    batch_size = data.num_graphs
    for ihead in range(model.module.num_heads):
        head_each = torch.arange(head_dims[ihead])
        head_ind_temporary = head_each.repeat(batch_size)
        head_shift_temporary = sum(head_dims[:ihead]) + torch.repeat_interleave(
            torch.arange(batch_size) * head_dimsum, head_dims[ihead]
        )
        head_index[ihead] = head_ind_temporary + head_shift_temporary
    return head_index


def get_head_indices_node_or_mixed(model, data):
    """this is for cases when outputs are node level or mixed graph-node level"""
    # feature index for all heads
    head_index = [None] * model.module.num_heads
    if model.module.num_heads == 1:
        head_index[0] = torch.arange(data.y.shape[0])
        return head_index
    batch_size = data.num_graphs
    y_loc = data.y_loc
    # head size for each sample
    total_size = y_loc[:, -1]
    # intermediate work list
    head_ind_temporary = [None] * batch_size
    # track the start loc of each sample
    sample_start = torch.cumsum(total_size, dim=0) - total_size
    sample_start = sample_start.view(-1, 1)
    # shape (batch_size, model.module.num_heads), start and end of each head for each sample
    start_index = sample_start + y_loc[:, :-1]
    end_index = sample_start + y_loc[:, 1:]

    # a large index tensor pool for all element in data.y
    index_range = torch.arange(0, end_index[-1, -1], device=y_loc.device)
    for ihead in range(model.module.num_heads):
        for isample in range(batch_size):
            head_ind_temporary[isample] = index_range[
                start_index[isample, ihead] : end_index[isample, ihead]
            ]
        head_index[ihead] = torch.cat(head_ind_temporary, dim=0)

    return head_index
