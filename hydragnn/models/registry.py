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
"""Registry mapping ``mpnn_type`` names to model builder functions.

Each builder receives a namespace ``p`` holding every argument of
``hydragnn.models.create.create_model`` and returns an un-moved model.
New architectures register themselves with ``@register_model("Name")``.
"""

from types import SimpleNamespace
from typing import Callable, Dict

from hydragnn.models.GINStack import GINStack
from hydragnn.models.PNAStack import PNAStack
from hydragnn.models.PNAPlusStack import PNAPlusStack
from hydragnn.models.GATStack import GATStack
from hydragnn.models.MFCStack import MFCStack
from hydragnn.models.CGCNNStack import CGCNNStack
from hydragnn.models.SAGEStack import SAGEStack
from hydragnn.models.SCFStack import SCFStack
from hydragnn.models.DIMEStack import DIMEStack
from hydragnn.models.EGCLStack import EGCLStack
from hydragnn.models.PNAEqStack import PNAEqStack
from hydragnn.models.PAINNStack import PAINNStack
from hydragnn.models.MACEStack import MACEStack
from hydragnn.models.AllScAIPStack import AllScAIPStack
from hydragnn.models.UMAStack import UMAStack
from hydragnn.models.heterogeneous import (
    HeteroGINStack,
    HeteroSAGEStack,
    HeteroGATStack,
    HeteroPNAStack,
    HeteroHGTStack,
    HeteroHEATStack,
    HeteroRGATStack,
)

ModelBuilder = Callable[[SimpleNamespace], object]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str):
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def get_model_builder(name: str) -> ModelBuilder:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError("Unknown mpnn_type: {0}".format(name)) from None


def available_models():
    return sorted(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def _base_args(p):
    """Positional arguments shared by every Base subclass, after model-specific ones."""
    return (
        p.input_dim,
        p.hidden_dim,
        p.output_dim,
        p.pe_dim,
        p.global_attn_engine,
        p.global_attn_type,
        p.global_attn_heads,
        p.output_type,
        p.output_heads,
        p.activation_function,
        p.loss_function_type,
        p.equivariance,
    )


def _base_kwargs(p, initial_bias=True):
    kwargs = dict(
        loss_weights=p.task_weights,
        freeze_conv=p.freeze_conv,
        num_conv_layers=p.num_conv_layers,
        num_nodes=p.num_nodes,
        graph_pooling=p.graph_pooling,
        use_graph_attr_conditioning=p.use_graph_attr_conditioning,
        graph_attr_conditioning_mode=p.graph_attr_conditioning_mode,
    )
    if initial_bias:
        kwargs["initial_bias"] = p.initial_bias
    return kwargs


def _equivariant_attn_kwargs(p):
    return dict(
        equivariant_attn_num_hidden_layers=p.equivariant_attn_num_hidden_layers,
        equivariant_attn_lmax=p.equivariant_attn_lmax,
        equivariant_attn_num_radial=p.equivariant_attn_num_radial,
        equivariant_attn_feedforward_multiplier=p.equivariant_attn_feedforward_multiplier,
        equivariant_attn_allow_scalar_only=p.equivariant_attn_allow_scalar_only,
        equivariant_attn_require_tensor_coupling=p.equivariant_attn_require_tensor_coupling,
        equivariant_attn_chunk_size=p.equivariant_attn_chunk_size,
        equivariant_attn_coupling_mode=p.equivariant_attn_coupling_mode,
        equivariant_attn_periodic=p.equivariant_attn_periodic,
        equivariant_attn_periodic_replication=p.equivariant_attn_periodic_replication,
    )


def _hetero_kwargs(p):
    return dict(
        graph_attr_dim=p.graph_attr_dim,
        hetero_pooling_mode=p.hetero_pooling_mode,
        node_target_type=p.node_target_type,
        share_relation_weights=p.share_relation_weights,
        metadata=p.metadata,
        node_input_dims=p.node_input_dims,
        attn_only=p.attn_only,
        attn_node_types=p.attn_node_types,
        structural_encoding=p.structural_encoding,
    )


def _require(p, model_name, *names):
    for name in names:
        assert getattr(p, name) is not None, f"{model_name} requires {name} input."


# ---------------------------------------------------------------------------
# Homogeneous models
# ---------------------------------------------------------------------------


@register_model("GIN")
def _build_gin(p):
    return GINStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("PNA")
def _build_pna(p):
    assert p.pna_deg is not None, "PNA requires degree input."
    return PNAStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        p.pna_deg,
        p.edge_dim,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("PNAPlus")
def _build_pna_plus(p):
    assert p.pna_deg is not None, "PNAPlus requires degree input."
    _require(p, "PNAPlus", "envelope_exponent", "num_radial", "radius")
    return PNAPlusStack(
        "inv_node_feat, equiv_node_feat, edge_index, rbf",
        "inv_node_feat, edge_index, rbf",
        p.pna_deg,
        p.edge_dim,
        p.envelope_exponent,
        p.num_radial,
        p.radius,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("GAT")
def _build_gat(p):
    # FIXME: expose options to users
    heads = 6
    negative_slope = 0.05
    return GATStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        heads,
        negative_slope,
        p.edge_dim,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("MFC")
def _build_mfc(p):
    _require(p, "MFC", "max_neighbours")
    return MFCStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        p.max_neighbours,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("CGCNN")
def _build_cgcnn(p):
    return CGCNNStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        p.edge_dim,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("SAGE")
def _build_sage(p):
    return SAGEStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        *_base_args(p),
        **_base_kwargs(p, initial_bias=False),
    )


@register_model("SchNet")
def _build_schnet(p):
    _require(p, "SchNet", "num_gaussians", "num_filters", "radius")
    return SCFStack(
        "",  # Depends on SchNet usage of edge_features
        "inv_node_feat, equiv_node_feat, edge_index, edge_weight, edge_rbf",
        p.num_filters,
        p.edge_dim,
        p.num_gaussians,
        p.radius,
        *_base_args(p),
        max_neighbours=p.max_neighbours,
        **_base_kwargs(p),
        **_equivariant_attn_kwargs(p),
    )


@register_model("DimeNet")
def _build_dimenet(p):
    _require(
        p,
        "DimeNet",
        "basis_emb_size",
        "envelope_exponent",
        "int_emb_size",
        "out_emb_size",
        "num_after_skip",
        "num_before_skip",
        "num_radial",
        "num_spherical",
        "radius",
    )
    return DIMEStack(
        "inv_node_feat, equiv_node_feat, rbf, sbf, i, j, idx_kj, idx_ji, num_nodes",
        "",
        p.basis_emb_size,
        p.envelope_exponent,
        p.int_emb_size,
        p.out_emb_size,
        p.num_after_skip,
        p.num_before_skip,
        p.num_radial,
        p.num_spherical,
        p.edge_dim,
        p.radius,
        *_base_args(p),
        max_neighbours=p.max_neighbours,
        **_base_kwargs(p),
        **_equivariant_attn_kwargs(p),
    )


@register_model("EGNN")
def _build_egnn(p):
    return EGCLStack(
        "inv_node_feat, equiv_node_feat, edge_index, edge_attr, edge_shifts",
        "",
        p.edge_dim,
        *_base_args(p),
        max_neighbours=p.max_neighbours,
        **_base_kwargs(p),
    )


@register_model("PAINN")
def _build_painn(p):
    return PAINNStack(
        "inv_node_feat, equiv_node_feat, edge_index, diff, dist",
        "inv_node_feat, equiv_node_feat, edge_index, diff, dist",
        p.edge_dim,
        p.num_radial,
        p.radius,
        *_base_args(p),
        **_base_kwargs(p, initial_bias=False),
        **_equivariant_attn_kwargs(p),
    )


@register_model("PNAEq")
def _build_pnaeq(p):
    assert p.pna_deg is not None, "PNAEq requires degree input."
    return PNAEqStack(
        "inv_node_feat, equiv_node_feat, edge_index, edge_rbf, edge_vec",
        "inv_node_feat, equiv_node_feat, edge_index, edge_rbf, edge_vec",
        p.pna_deg,
        p.edge_dim,
        p.num_radial,
        p.radius,
        *_base_args(p),
        **_base_kwargs(p, initial_bias=False),
        **_equivariant_attn_kwargs(p),
    )


@register_model("MACE")
def _build_mace(p):
    _require(p, "MACE", "radius", "num_radial", "max_ell", "node_max_ell")
    assert p.max_ell >= 1, "MACE requires max_ell >= 1."
    assert p.node_max_ell >= 1, "MACE requires node_max_ell >= 1."
    return MACEStack(
        "node_attributes, equiv_node_feat, inv_node_feat, edge_attributes, edge_features, edge_index",
        "node_attributes, edge_attributes, edge_features, edge_index",
        p.radius,
        p.radial_type,
        p.distance_transform,
        p.num_radial,
        p.edge_dim,
        p.max_ell,
        p.node_max_ell,
        p.avg_num_neighbors,
        p.envelope_exponent,
        p.correlation,
        *_base_args(p),
        **_base_kwargs(p),
        **_equivariant_attn_kwargs(p),
    )


@register_model("AllScAIP")
def _build_allscaip(p):
    _require(p, "AllScAIP", "radius", "max_neighbours")
    # input_args/conv_args are unused at runtime; AllScAIP does its own message passing.
    return AllScAIPStack(
        "inv_node_feat, equiv_node_feat",
        "inv_node_feat",
        p.radius,
        p.max_neighbours,
        p.allscaip_num_heads,
        p.allscaip_freq_list,
        p.allscaip_atten_name,
        p.allscaip_use_node_path,
        p.allscaip_use_sincx_mask,
        p.allscaip_use_freq_mask,
        p.allscaip_max_num_elements,
        p.allscaip_knn_soft,
        p.allscaip_distance_function,
        p.allscaip_normalization,
        p.allscaip_mlp_dropout,
        p.allscaip_atten_dropout,
        p.allscaip_use_residual_scaling,
        p.allscaip_regress_stress,
        p.allscaip_dataset_list,
        p.allscaip_use_chunked_graph,
        p.allscaip_graph_chunk_size,
        p.allscaip_knn_use_low_mem,
        *_base_args(p),
        **_base_kwargs(p),
    )


@register_model("UMA")
def _build_uma(p):
    _require(p, "UMA", "radius", "max_neighbours", "max_ell", "num_radial")
    return UMAStack(
        "inv_node_feat, equiv_node_feat, edge_index",
        "inv_node_feat, edge_index",
        p.radius,
        p.max_neighbours,
        p.max_ell,
        p.num_radial,
        p.uma_mmax,
        p.uma_grid_resolution,
        p.uma_edge_channels,
        p.uma_hidden_channels,
        p.uma_norm_type,
        p.uma_ff_type,
        p.uma_use_chg_spin,
        p.uma_max_num_elements,
        p.uma_variant,
        p.uma_num_experts,
        p.uma_moe_dropout,
        p.uma_use_composition_embedding,
        p.periodic_boundary_conditions,
        *_base_args(p),
        uma_equivariant_vector_head=p.uma_equivariant_vector_head,
        uma_vector_head_index=p.uma_vector_head_index,
        **_base_kwargs(p),
    )


# ---------------------------------------------------------------------------
# Heterogeneous models
# ---------------------------------------------------------------------------


def _build_hetero(cls, p, *model_args):
    return cls(
        *model_args,
        p.edge_dim,
        *_base_args(p),
        **_base_kwargs(p),
        **_hetero_kwargs(p),
    )


@register_model("HeteroGIN")
def _build_hetero_gin(p):
    return _build_hetero(HeteroGINStack, p)


@register_model("HeteroSAGE")
def _build_hetero_sage(p):
    return _build_hetero(HeteroSAGEStack, p)


@register_model("HeteroGAT")
def _build_hetero_gat(p):
    heads = 6
    negative_slope = 0.05
    return _build_hetero(HeteroGATStack, p, heads, negative_slope)


@register_model("HeteroPNA")
def _build_hetero_pna(p):
    assert p.pna_deg is not None, "HeteroPNA requires degree input."
    return _build_hetero(HeteroPNAStack, p, p.pna_deg)


@register_model("HeteroRGAT")
def _build_hetero_rgat(p):
    return _build_hetero(
        HeteroRGATStack,
        p,
        p.hetero_attention_heads,
        p.hetero_attention_negative_slope,
    )


@register_model("HeteroHGT")
def _build_hetero_hgt(p):
    return _build_hetero(HeteroHGTStack, p, p.hetero_attention_heads)


@register_model("HeteroHEAT")
def _build_hetero_heat(p):
    return _build_hetero(
        HeteroHEATStack,
        p,
        p.hetero_attention_heads,
        p.hetero_edge_type_emb_dim,
        p.hetero_edge_attr_emb_dim,
    )
