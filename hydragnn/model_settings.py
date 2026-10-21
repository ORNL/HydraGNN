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
"""Grouped settings for model families with many specific options.

Each settings class maps to config keys sharing one prefix, e.g.
``AllScAIPSettings.num_heads`` <-> ``Architecture.allscaip_num_heads``.
This module only imports the standard library so config parsing can use it.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import ClassVar, List, Optional, Union


class _PrefixedSettings:
    prefix: ClassVar[str]

    @classmethod
    def from_mapping(cls, mapping):
        """Build from flat ``<prefix><name>`` keys; missing keys use defaults."""
        values = {
            f.name: deepcopy(mapping[cls.prefix + f.name])
            for f in fields(cls)
            if cls.prefix + f.name in mapping
        }
        return cls(**values)

    @classmethod
    def pop_from(cls, mapping):
        """Remove this family's flat keys from ``mapping`` and return them unprefixed."""
        names = [f.name for f in fields(cls)]
        return {
            name: mapping.pop(cls.prefix + name)
            for name in names
            if cls.prefix + name in mapping
        }

    def as_kwargs(self):
        return {self.prefix + f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class EquivariantAttentionSettings(_PrefixedSettings):
    prefix: ClassVar[str] = "equivariant_attn_"

    lmax: int = 1
    num_radial: int = 16
    num_hidden_layers: int = 1
    feedforward_multiplier: int = 2
    allow_scalar_only: bool = False
    require_tensor_coupling: bool = True
    chunk_size: Optional[int] = 512
    coupling_mode: str = "parallel"
    periodic: bool = False
    periodic_replication: Union[int, List[int]] = 1


@dataclass(frozen=True)
class AllScAIPSettings(_PrefixedSettings):
    prefix: ClassVar[str] = "allscaip_"

    num_heads: int = 8
    freq_list: Optional[List[int]] = None
    atten_name: str = "math"
    use_node_path: bool = True
    use_sincx_mask: bool = True
    use_freq_mask: bool = True
    max_num_elements: int = 119
    knn_soft: bool = True
    distance_function: str = "gaussian"
    normalization: str = "rmsnorm"
    mlp_dropout: float = 0.0
    atten_dropout: float = 0.0
    use_residual_scaling: bool = True
    regress_stress: bool = False
    use_chunked_graph: bool = False
    graph_chunk_size: int = 512
    knn_use_low_mem: bool = True
    dataset_list: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class UMASettings(_PrefixedSettings):
    prefix: ClassVar[str] = "uma_"

    mmax: int = 2
    grid_resolution: Optional[int] = None
    edge_channels: int = 128
    hidden_channels: Optional[int] = None
    norm_type: str = "rms_norm_sh"
    ff_type: str = "grid"
    use_chg_spin: bool = False
    max_num_elements: int = 100
    variant: str = "S"
    num_experts: Optional[int] = None
    moe_dropout: float = 0.0
    use_composition_embedding: bool = False
    equivariant_vector_head: bool = False
    vector_head_index: Optional[int] = None
