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

"""Shared defaults for optional model-specific architecture settings."""

MODEL_SPECIFIC_ARCHITECTURE_DEFAULTS = {
    "allscaip_num_heads": 8,
    "allscaip_freq_list": None,
    "allscaip_atten_name": "math",
    "allscaip_use_node_path": True,
    "allscaip_use_sincx_mask": True,
    "allscaip_use_freq_mask": True,
    "allscaip_max_num_elements": 119,
    "allscaip_knn_soft": True,
    "allscaip_distance_function": "gaussian",
    "allscaip_normalization": "rmsnorm",
    "allscaip_mlp_dropout": 0.0,
    "allscaip_atten_dropout": 0.0,
    "allscaip_use_residual_scaling": True,
    "allscaip_regress_stress": False,
    "allscaip_use_chunked_graph": False,
    "allscaip_graph_chunk_size": 512,
    "allscaip_knn_use_low_mem": True,
    "allscaip_dataset_list": [],
    "uma_mmax": 2,
    "uma_grid_resolution": None,
    "uma_edge_channels": 128,
    "uma_hidden_channels": None,
    "uma_norm_type": "rms_norm_sh",
    "uma_ff_type": "grid",
    "uma_use_chg_spin": False,
    "uma_max_num_elements": 100,
    "uma_variant": "S",
    "uma_num_experts": None,
    "uma_moe_dropout": 0.0,
    "uma_use_composition_embedding": False,
    "uma_equivariant_vector_head": False,
    "uma_vector_head_index": None,
}
