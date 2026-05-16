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
"""
All-to-all Scaled Attention Interatomic Potential (AllScAIP) backbone.

Vendored and trimmed from FairChem's ``allscaip`` model. Only the backbone
required by HydraGNN's ``AllScAIPStack`` wrapper is retained; the original
FairChem head classes, ASE-calculator validation helpers, and inference-
settings plumbing have been removed since HydraGNN provides its own
multi-head decoder and training loop.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function

from hydragnn.utils.model.allscaip.configs import AllScAIPConfigs, init_configs
from hydragnn.utils.model.allscaip.modules.graph_attention_block import (
    GraphAttentionBlock,
)
from hydragnn.utils.model.allscaip.modules.input_block import InputBlock
from hydragnn.utils.model.allscaip.utils.data_preprocess import (
    data_preprocess_radius_graph,
)
from hydragnn.utils.model.escaip.utils.graph_utils import get_displacement_and_cell
from hydragnn.utils.model.escaip.utils.nn_utils import no_weight_decay

if TYPE_CHECKING:
    from hydragnn.utils.model.allscaip.custom_types import GraphAttentionData


class AllScAIPBackbone(nn.Module):
    """All-to-all Scaled Attention Interatomic Potential backbone."""

    def __init__(self, **kwargs):
        super().__init__()

        cfg = init_configs(AllScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        self.regress_forces = cfg.global_cfg.regress_forces
        self.direct_forces = cfg.global_cfg.direct_forces
        self.regress_stress = cfg.global_cfg.regress_stress
        self.max_num_elements = cfg.molecular_graph_cfg.max_num_elements
        self.max_neighbors = cfg.molecular_graph_cfg.knn_k
        self.cutoff = cfg.molecular_graph_cfg.max_radius

        self.data_preprocess = partial(
            data_preprocess_radius_graph,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                GraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.global_cfg.num_layers)
            ]
        )

        self.init_weights()
        torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(recompiles=True)  # type: ignore

    def compiled_forward(self, data: "GraphAttentionData"):
        with record_function("input_block"):
            neighbor_reps = self.input_block(data)

        for idx in range(self.global_cfg.num_layers):
            neighbor_reps = self.transformer_blocks[idx](
                data, neighbor_reps, layer_idx=idx
            )

        return {
            "data": data,
            "node_reps": neighbor_reps[:, 0].to(torch.float32),
        }

    @torch.compiler.disable()
    def forward(self, data):
        data["atomic_numbers"] = data["atomic_numbers"].long()  # type: ignore
        data["atomic_numbers_full"] = data["atomic_numbers"]  # type: ignore
        data["batch_full"] = data["batch"]  # type: ignore

        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = get_displacement_and_cell(
                data, self.regress_stress, self.regress_forces, self.direct_forces
            )

        with record_function("data_preprocess"), torch.autocast(
            device_type=str(data.pos.device), enabled=False
        ):
            x = self.data_preprocess(data)

        with record_function("backbone_forward"):
            results = self.compiled_forward(x)

        results["displacement"] = displacement
        results["orig_cell"] = orig_cell
        return results

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return no_weight_decay(self)

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
