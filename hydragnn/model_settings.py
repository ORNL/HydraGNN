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


@dataclass(frozen=True)
class MeshGraphNetSettings(_PrefixedSettings):
    """MeshGraphNet options; latent width and processor steps come from
    ``hidden_dim`` and ``num_conv_layers``."""

    prefix: ClassVar[str] = "mgn_"

    mlp_layers: int = 2
    spatial_dim: int = 3
    conservative_faces: bool = True


@dataclass(frozen=True)
class MultiscaleMGNSettings(_PrefixedSettings):
    """MultiscaleMeshGraphNet options on top of the ``mgn_`` ones.
    ``schedule`` (e.g. ``"1H 11L 1H"``) defaults to one V-cycle with
    ``num_conv_layers`` steps, counting the down and up blocks."""

    prefix: ClassVar[str] = "msmgn_"

    schedule: Optional[str] = None


@dataclass(frozen=True)
class RIGNOSettings(_PrefixedSettings):
    """RIGNO options; latent width and processor steps come from
    ``hidden_dim`` and ``num_conv_layers``. The overlap factors must match
    the ones passed to ``build_rigno_graph``."""

    prefix: ClassVar[str] = "rigno_"

    mlp_layers: int = 2
    edge_masking: float = 0.5
    overlap_p2r: float = 1.0
    overlap_r2p: float = 2.0
    concat_time: bool = False
    time_conditioning: bool = False
    spatial_dim: int = 2


@dataclass(frozen=True)
class FluxGNNSettings(_PrefixedSettings):
    """FluxGNN options; latent width and substeps per forward come from
    ``hidden_dim`` and ``num_conv_layers``. ``layout`` (filled from the
    variable schema) maps ``state``/``vectors``/``scalars`` to columns of
    ``data.cell_x``."""

    prefix: ClassVar[str] = "fluxgnn_"

    dt: float = 1.0
    spatial_dim: int = 2
    mlp_layers: int = 2
    advection: bool = True
    diffusion: bool = True
    boundary: str = "learned"
    num_bface_tags: int = 1
    source: bool = False
    layout: Optional[dict] = None


@dataclass(frozen=True)
class FNOSettings(_PrefixedSettings):
    """FNO options (neuralop defaults); width and depth come from
    ``hidden_dim`` and ``num_conv_layers``. Nodes need ``data.grid_shape``."""

    prefix: ClassVar[str] = "fno_"

    n_modes: tuple = (16, 16)
    lifting_channel_ratio: float = 2
    projection_channel_ratio: float = 2
    positional_embedding: Optional[str] = "grid"
    domain_padding: object = None
    channel_mlp_expansion: float = 0.5


@dataclass(frozen=True)
class GINOSettings(_PrefixedSettings):
    """GINO options (neuralop defaults except ``fno_in_channels``, which
    defaults to the input width); width and FNO depth come from
    ``hidden_dim`` and ``num_conv_layers``."""

    prefix: ClassVar[str] = "gino_"

    coord_dim: int = 2
    n_modes: tuple = (16, 16)
    latent_resolution: tuple = (32, 32)
    box_min: object = 0.0
    box_max: object = 1.0
    in_radius: float = 0.033
    out_radius: float = 0.033
    fno_in_channels: Optional[int] = None
    embed_channels: int = 32
    embed_max_positions: int = 10000
    in_channel_mlp_hidden_layers: tuple = (80, 80, 80)
    out_channel_mlp_hidden_layers: tuple = (512, 256)
    lifting_channel_ratio: float = 2
    projection_channel_ratio: float = 4
    channel_mlp_expansion: float = 0.5


@dataclass(frozen=True)
class GSNOSettings(_PrefixedSettings):
    """GSNO options (paper defaults); width and depth come from
    ``hidden_dim`` and ``num_conv_layers``. ``temporal_modes = 0`` is the
    steady operator. Nodes need ``data.gsno_basis``."""

    prefix: ClassVar[str] = "gsno_"

    spatial_modes: int = 8
    temporal_modes: int = 0
    input_time_steps: int = 1
    output_time_steps: int = 1
    projection_channels: int = 128
    tie_degenerate: bool = False
