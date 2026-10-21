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

import torch
from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, List, Union

from hydragnn.model_settings import (
    AllScAIPSettings,
    EquivariantAttentionSettings,
    UMASettings,
)
from hydragnn.domain_losses import create_domain_loss
from hydragnn.utils.input_config_parsing.config_utils import edge_type_dims
from hydragnn.models.registry import get_model_builder

from hydragnn.utils.distributed import get_device
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.train.train_validate_test import resolve_precision


def compute_forces_and_hessian(
    energy, positions, *, compute_hessian=False, create_graph=False
):
    """Differentiate scalar graph energy into forces and a Cartesian Hessian."""
    forces = -torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        retain_graph=create_graph or compute_hessian,
        create_graph=create_graph or compute_hessian,
    )[0]
    if not compute_hessian:
        return forces, None

    rows = []
    for component in forces.reshape(-1):
        force_gradient = torch.autograd.grad(
            component,
            positions,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        rows.append(-force_gradient.reshape(-1))
    return forces, torch.stack(rows)


def create_model_config(
    config: dict,
    verbosity: int = 0,
    use_gpu: bool = True,
    metadata=None,
    node_input_dims=None,
):
    model = create_model(
        mpnn_type=config["Architecture"]["mpnn_type"],
        input_dim=config["Architecture"]["input_dim"],
        hidden_dim=config["Architecture"]["hidden_dim"],
        output_dim=config["Architecture"]["output_dim"],
        pe_dim=config["Architecture"]["pe_dim"],
        global_attn_engine=config["Architecture"]["global_attn_engine"],
        global_attn_type=config["Architecture"]["global_attn_type"],
        global_attn_heads=config["Architecture"]["global_attn_heads"],
        attn_only=config["Architecture"].get("attn_only", False),
        attn_node_types=config["Architecture"].get("attn_node_types", None),
        structural_encoding=config["Architecture"].get("structural_encoding", None),
        output_type=config["Architecture"]["output_type"],
        output_heads=config["Architecture"]["output_heads"],
        activation_function=config["Architecture"]["activation_function"],
        loss_function_type=config["Training"]["loss_function_type"],
        task_weights=config["Architecture"]["task_weights"],
        num_conv_layers=config["Architecture"]["num_conv_layers"],
        freeze_conv=config["Architecture"]["freeze_conv_layers"],
        initial_bias=config["Architecture"]["initial_bias"],
        num_nodes=config["Architecture"]["num_nodes"],
        max_neighbours=config["Architecture"]["max_neighbours"],
        edge_dim=(
            edge_type_dims(config["Architecture"]["edge_types"])
            if config["Architecture"].get("edge_types") is not None
            else config["Architecture"]["edge_dim"]
        ),
        pna_deg=config["Architecture"]["pna_deg"],
        num_before_skip=config["Architecture"]["num_before_skip"],
        num_after_skip=config["Architecture"]["num_after_skip"],
        num_radial=config["Architecture"]["num_radial"],
        radial_type=config["Architecture"]["radial_type"],
        distance_transform=config["Architecture"]["distance_transform"],
        basis_emb_size=config["Architecture"]["basis_emb_size"],
        int_emb_size=config["Architecture"]["int_emb_size"],
        out_emb_size=config["Architecture"]["out_emb_size"],
        envelope_exponent=config["Architecture"]["envelope_exponent"],
        num_spherical=config["Architecture"]["num_spherical"],
        num_gaussians=config["Architecture"]["num_gaussians"],
        num_filters=config["Architecture"]["num_filters"],
        radius=config["Architecture"]["radius"],
        periodic_boundary_conditions=config["Architecture"].get(
            "periodic_boundary_conditions", False
        ),
        equivariance=config["Architecture"]["equivariance"],
        correlation=config["Architecture"]["correlation"],
        max_ell=config["Architecture"]["max_ell"],
        node_max_ell=config["Architecture"]["node_max_ell"],
        avg_num_neighbors=config["Architecture"]["avg_num_neighbors"],
        conv_checkpointing=config["Training"]["conv_checkpointing"],
        domain_loss_config=config["Training"].get("loss"),
        input_node_encodings=config["Architecture"].get("input_node_encodings"),
        use_graph_attr_conditioning=config["Architecture"].get(
            "use_graph_attr_conditioning", False
        ),
        graph_attr_dim=config["Architecture"].get("graph_attr_dim", 0),
        graph_attr_conditioning_mode=config["Architecture"].get(
            "graph_attr_conditioning_mode", "concat_node"
        ),
        graph_pooling=config["Architecture"].get("graph_pooling", "mean"),
        equivariant_attn=EquivariantAttentionSettings.from_mapping(
            config["Architecture"]
        ),
        allscaip=AllScAIPSettings.from_mapping(config["Architecture"]),
        uma=UMASettings.from_mapping(config["Architecture"]),
        hetero_pooling_mode=config["Architecture"].get("hetero_pooling_mode", "sum"),
        node_target_type=config["Architecture"].get("node_target_type", None),
        share_relation_weights=config["Architecture"].get(
            "share_relation_weights", False
        ),
        metadata=metadata,
        node_input_dims=node_input_dims,
        hetero_attention_heads=config["Architecture"].get("hetero_attention_heads", 4),
        hetero_attention_negative_slope=config["Architecture"].get(
            "hetero_attention_negative_slope", 0.2
        ),
        hetero_edge_type_emb_dim=config["Architecture"].get(
            "hetero_edge_type_emb_dim", 16
        ),
        hetero_edge_attr_emb_dim=config["Architecture"].get(
            "hetero_edge_attr_emb_dim", 16
        ),
        verbosity=verbosity,
        use_gpu=use_gpu,
    )

    ## Set precision: bf16, fp32, fp64
    training_cfg = config["Training"]
    precision, param_dtype, _ = resolve_precision(training_cfg.get("precision", "fp32"))
    torch.set_default_dtype(param_dtype)

    return model.to(dtype=param_dtype)


_SETTINGS_CLASSES = {
    "equivariant_attn": EquivariantAttentionSettings,
    "allscaip": AllScAIPSettings,
    "uma": UMASettings,
}


def _resolve_settings(settings_cls, settings, flat_settings):
    """Merge legacy flat ``<prefix>_*`` keyword arguments into a settings object."""
    overrides = settings_cls.pop_from(flat_settings)
    settings = settings if settings is not None else settings_cls()
    return replace(settings, **overrides) if overrides else settings


# FIXME: interface does not include ilossweights_hyperp, ilossweights_nll, dropout
def create_model(
    mpnn_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: list,
    pe_dim: int,
    global_attn_engine: str,
    global_attn_type: str,
    global_attn_heads: int,
    output_type: list,
    output_heads: dict,
    activation_function: str,
    loss_function_type: str,
    task_weights: list,
    num_conv_layers: int,
    freeze_conv: bool = False,
    initial_bias: float = None,
    num_nodes: int = None,
    max_neighbours: int = None,
    edge_dim: Union[int, Dict[str, int], None] = None,
    pna_deg: torch.tensor = None,
    num_before_skip: int = None,
    num_after_skip: int = None,
    num_radial: int = None,
    radial_type: str = None,
    distance_transform: str = None,
    basis_emb_size: int = None,
    int_emb_size: int = None,
    out_emb_size: int = None,
    envelope_exponent: int = None,
    num_spherical: int = None,
    num_gaussians: int = None,
    num_filters: int = None,
    radius: float = None,
    equivariance: bool = False,
    correlation: Union[int, List[int]] = None,
    max_ell: int = None,
    node_max_ell: int = None,
    avg_num_neighbors: int = None,
    conv_checkpointing: bool = False,
    domain_loss_config: dict | None = None,
    input_node_encodings: list[dict] | None = None,
    use_graph_attr_conditioning: bool = False,
    graph_attr_dim: int = 0,
    graph_attr_conditioning_mode: str = "fuse_pool",
    graph_pooling: str = "mean",
    hetero_pooling_mode: str = "sum",
    node_target_type: str = None,
    share_relation_weights: bool = False,
    metadata=None,
    node_input_dims=None,
    hetero_attention_heads: int = 4,
    hetero_attention_negative_slope: float = 0.2,
    hetero_edge_type_emb_dim: int = 16,
    hetero_edge_attr_emb_dim: int = 16,
    verbosity: int = 0,
    use_gpu: bool = True,
    periodic_boundary_conditions: bool = False,
    attn_only: bool = False,
    *,
    attn_node_types: List[str] = None,
    structural_encoding: dict = None,
    equivariant_attn: EquivariantAttentionSettings | None = None,
    allscaip: AllScAIPSettings | None = None,
    uma: UMASettings | None = None,
    **flat_settings,
):
    params = dict(locals())
    flat_settings = params.pop("flat_settings")
    for name, settings_cls in _SETTINGS_CLASSES.items():
        params[name] = _resolve_settings(settings_cls, params[name], flat_settings)
    if flat_settings:
        raise TypeError(
            f"create_model() got unexpected keyword arguments: {sorted(flat_settings)}"
        )
    params = SimpleNamespace(**params)

    timer = Timer("create_model")
    timer.start()
    torch.manual_seed(0)

    device = get_device(use_gpu, verbosity_level=verbosity)

    model = get_model_builder(mpnn_type)(params)

    model.configure_input_feature_encoders(input_node_encodings)
    if use_graph_attr_conditioning and graph_attr_dim is not None:
        if model.graph_attr_conditioning_mode == "film":
            model._ensure_graph_conditioner(graph_attr_dim, device)
        elif model.graph_attr_conditioning_mode == "concat_node":
            model._ensure_graph_concat_projector(graph_attr_dim, hidden_dim, device)
        elif model.graph_attr_conditioning_mode == "fuse_pool":
            if "graph" not in output_type:
                raise ValueError(
                    "fuse_pool graph conditioning requires at least one graph output"
                )
            if not getattr(model, "supports_graph_pool_conditioning", True):
                raise ValueError(
                    f"{model.__class__.__name__} does not support fuse_pool graph conditioning"
                )
            model._ensure_graph_pool_projector(graph_attr_dim, hidden_dim, device)

    model.atomistic_mode_enabled = False
    model = create_domain_loss(model, domain_loss_config)

    if conv_checkpointing:
        model.enable_conv_checkpointing()

    timer.stop()

    return model.to(device)
