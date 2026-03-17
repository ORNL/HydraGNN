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
import os
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    check_if_graph_size_variable,
    gather_deg,
)
from hydragnn.utils.model.model import calculate_avg_deg
from hydragnn.utils.distributed import get_comm_size_and_rank
from hydragnn.utils.model import update_multibranch_heads
from copy import deepcopy
import warnings
import json
import hashlib
import re
import torch

from .variable_schema import get_variable_schema, schema_dimensions

_UNSAFE_LOG_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename_component(value, max_length=48):
    """Return a bounded, filesystem-safe representation of one path component.

    The returned value never contains path separators or leading/trailing dots,
    so configuration-derived labels can safely be used as filenames without
    changing the original label shown to users in plots and logs.
    """
    original = str(value)
    sanitized = _UNSAFE_LOG_COMPONENT.sub("-", original).strip("._-")
    if not sanitized:
        sanitized = "variable"

    if sanitized != original or len(sanitized) > max_length:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        prefix = sanitized[: max_length - len(digest) - 1].rstrip("._-")
        sanitized = f"{prefix or 'variable'}-{digest}"

    return sanitized


def update_config(config, train_loader, val_loader, test_loader):
    """check if config input consistent and update config with model and datasets"""

    graph_size_variable = os.getenv("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE")
    if graph_size_variable is None:
        graph_size_variable = check_if_graph_size_variable(
            train_loader, val_loader, test_loader
        )
    else:
        graph_size_variable = bool(int(graph_size_variable))

    named_schema = get_variable_schema(config)

    # Always sync node_input_dims from heterogeneous data.
    arch_cfg = config["NeuralNetwork"].setdefault("Architecture", {})
    data_sample = train_loader.dataset[0]
    if hasattr(data_sample, "node_types"):
        node_input_dims = {}
        for node_type in data_sample.node_types:
            node_store = data_sample[node_type]
            if hasattr(node_store, "x") and node_store.x is not None:
                node_input_dims[str(node_type)] = int(node_store.x.shape[1])
        if node_input_dims:
            if arch_cfg.get("node_input_dims") not in (None, node_input_dims):
                warnings.warn(
                    "Overriding node_input_dims with dataset-derived sizes for hetero model."
                )
            arch_cfg["node_input_dims"] = node_input_dims

    # Set default values for GPS variables
    if "global_attn_engine" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = None
    if "global_attn_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_type"] = None
    if "global_attn_heads" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_heads"] = 0
    if "pe_dim" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["pe_dim"] = 0
    if "global_attn_redraw_interval" not in config["NeuralNetwork"]["Training"]:
        # Used only by Performer attention. None disables projection redraw.
        config["NeuralNetwork"]["Training"]["global_attn_redraw_interval"] = 1000

    architecture = config["NeuralNetwork"]["Architecture"]
    architecture.setdefault("equivariant_attn_lmax", 1)
    architecture.setdefault("equivariant_attn_num_radial", 16)
    architecture.setdefault("equivariant_attn_feedforward_multiplier", 2)
    architecture.setdefault("equivariant_attn_allow_scalar_only", False)
    architecture.setdefault("equivariant_attn_require_tensor_coupling", True)
    architecture.setdefault("equivariant_attn_chunk_size", 512)
    architecture.setdefault("equivariant_attn_coupling_mode", "parallel")
    architecture.setdefault("equivariant_attn_periodic", False)
    architecture.setdefault("equivariant_attn_periodic_replication", 1)
    validate_equivariant_transformer_config(architecture)

    batching = config["NeuralNetwork"]["Training"].get("Batching")
    if batching is not None:
        mode = batching.get("mode", "fixed")
        if mode not in {"fixed", "node_budget"}:
            raise ValueError(f"unsupported batching mode: {mode}")
        if mode == "node_budget" and "max_nodes" not in batching:
            raise ValueError("node_budget batching requires max_nodes")

    validate_local_sgd_config(config["NeuralNetwork"]["Training"])

    # update output_heads with latest config rules
    config["NeuralNetwork"]["Architecture"]["output_heads"] = update_multibranch_heads(
        config["NeuralNetwork"]["Architecture"]["output_heads"]
    )

    outputs = named_schema.outputs
    if any(spec.level == "edge" for spec in outputs):
        raise ValueError(
            "Named edge outputs are valid data attributes, but HydraGNN does "
            "not yet provide an edge prediction head"
        )
    config["NeuralNetwork"]["Architecture"]["input_dim"] = schema_dimensions(
        named_schema, "node", "inputs"
    )
    named_graph_dim = schema_dimensions(named_schema, "graph", "inputs")
    if named_graph_dim:
        config["NeuralNetwork"]["Architecture"]["use_graph_attr_conditioning"] = True
    config["NeuralNetwork"]["Architecture"]["output_dim"] = [
        spec.dim for spec in outputs
    ]
    config["NeuralNetwork"]["Architecture"]["output_type"] = [
        spec.level for spec in outputs
    ]
    config["NeuralNetwork"]["Architecture"]["num_nodes"] = int(
        train_loader.dataset[0].num_nodes
    )
    PNA_models = ["PNA", "PNAPlus", "PNAEq"]
    if config["NeuralNetwork"]["Architecture"]["mpnn_type"] in PNA_models:
        if hasattr(train_loader.dataset, "pna_deg"):
            ## Use max neighbours used in the datasets.
            deg = torch.tensor(train_loader.dataset.pna_deg)
        else:
            deg = gather_deg(train_loader.dataset)
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = deg.tolist()
        config["NeuralNetwork"]["Architecture"]["max_neighbours"] = len(deg) - 1
    else:
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = None

    # Set CGCNN hidden dim to input dim if global attention is not being used
    if (
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] == "CGCNN"
        and not config["NeuralNetwork"]["Architecture"]["global_attn_engine"]
    ):
        config["NeuralNetwork"]["Architecture"]["hidden_dim"] = config["NeuralNetwork"][
            "Architecture"
        ]["input_dim"]

    if config["NeuralNetwork"]["Architecture"]["mpnn_type"] == "MACE":
        if hasattr(train_loader.dataset, "avg_num_neighbors"):
            ## Use avg neighbours used in the dataset.
            avg_num_neighbors = float(train_loader.dataset.avg_num_neighbors)
        else:
            avg_num_neighbors = float(calculate_avg_deg(train_loader.dataset))
        config["NeuralNetwork"]["Architecture"]["avg_num_neighbors"] = avg_num_neighbors
    else:
        config["NeuralNetwork"]["Architecture"]["avg_num_neighbors"] = None

    if "radius" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["radius"] = None
    if "radial_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["radial_type"] = None
    if "distance_transform" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["distance_transform"] = None
    if "num_gaussians" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_gaussians"] = None
    if "num_filters" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_filters"] = None
    if "envelope_exponent" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["envelope_exponent"] = None
    if "num_after_skip" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_after_skip"] = None
    if "num_before_skip" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_before_skip"] = None
    if "basis_emb_size" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["basis_emb_size"] = None
    if "int_emb_size" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["int_emb_size"] = None
    if "out_emb_size" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["out_emb_size"] = None
    if "num_radial" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_radial"] = None
    if "num_spherical" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["num_spherical"] = None
    if "radial_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["radial_type"] = None
    if "correlation" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["correlation"] = None
    if "max_ell" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["max_ell"] = None
    if "node_max_ell" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["node_max_ell"] = None
    if "enable_interatomic_potential" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["enable_interatomic_potential"] = False
    # AllScAIP-specific defaults (used by AllScAIPStack via create_model).
    # Backbone depth is taken from the standard ``num_conv_layers`` key.
    if "allscaip_num_heads" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_num_heads"] = 8
    if "allscaip_freq_list" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_freq_list"] = None
    if "allscaip_atten_name" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_atten_name"] = "math"
    if "allscaip_use_node_path" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_use_node_path"] = True
    if "allscaip_use_sincx_mask" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_use_sincx_mask"] = True
    if "allscaip_use_freq_mask" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_use_freq_mask"] = True
    if "allscaip_max_num_elements" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_max_num_elements"] = 119
    if "allscaip_knn_soft" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_knn_soft"] = True
    if "allscaip_distance_function" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"][
            "allscaip_distance_function"
        ] = "gaussian"
    if "allscaip_normalization" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_normalization"] = "rmsnorm"
    if "allscaip_mlp_dropout" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_mlp_dropout"] = 0.0
    if "allscaip_atten_dropout" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_atten_dropout"] = 0.0
    if "allscaip_use_residual_scaling" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_use_residual_scaling"] = True
    if "allscaip_regress_stress" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_regress_stress"] = False
    if "allscaip_dataset_list" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["allscaip_dataset_list"] = []

    # UMA-specific defaults (used by UMAStack via create_model).
    if "uma_mmax" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_mmax"] = 2
    if "uma_grid_resolution" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_grid_resolution"] = None
    if "uma_edge_channels" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_edge_channels"] = 128
    if "uma_hidden_channels" not in config["NeuralNetwork"]["Architecture"]:
        # Default to None so UMAStack falls back to hidden_dim (sphere_channels).
        config["NeuralNetwork"]["Architecture"]["uma_hidden_channels"] = None
    if "uma_norm_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_norm_type"] = "rms_norm_sh"
    if "uma_ff_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_ff_type"] = "grid"
    if "uma_use_chg_spin" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_use_chg_spin"] = False
    if "uma_max_num_elements" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_max_num_elements"] = 100
    if "uma_variant" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_variant"] = "S"
    if "uma_num_experts" not in config["NeuralNetwork"]["Architecture"]:
        # None -> UMAStack picks the per-variant default (M=8, L=32; S=0).
        config["NeuralNetwork"]["Architecture"]["uma_num_experts"] = None
    if "uma_moe_dropout" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_moe_dropout"] = 0.0
    if "uma_use_composition_embedding" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_use_composition_embedding"] = False
    if "uma_equivariant_vector_head" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_equivariant_vector_head"] = False
    if "uma_vector_head_index" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["uma_vector_head_index"] = None

    config["NeuralNetwork"]["Architecture"] = update_config_edge_dim(
        config["NeuralNetwork"]["Architecture"]
    )
    if named_schema is not None:
        named_edge_dim = schema_dimensions(named_schema, "edge", "inputs")
        if named_edge_dim:
            if config["NeuralNetwork"]["Architecture"]["enable_interatomic_potential"]:
                raise ValueError(
                    "Named edge inputs cannot be used with interatomic-potential "
                    "mode because that mode constructs specialized edge features"
                )
            config["NeuralNetwork"]["Architecture"]["edge_dim"] = named_edge_dim

    config["NeuralNetwork"]["Architecture"] = update_config_equivariance(
        config["NeuralNetwork"]["Architecture"]
    )

    if "freeze_conv_layers" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["freeze_conv_layers"] = False
    if "initial_bias" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["initial_bias"] = None

    if "activation_function" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["activation_function"] = "relu"

    if "SyncBatchNorm" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["SyncBatchNorm"] = False

    if "conv_checkpointing" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["conv_checkpointing"] = False

    if "loss_function_type" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["loss_function_type"] = "mse"

    if "Optimizer" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["Optimizer"]["type"] = "AdamW"

    if "precision" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["precision"] = "fp32"

    return config


def validate_equivariant_transformer_config(config):
    """Validate options whose meaning is specific to the equivariant engine."""
    if config.get("global_attn_engine") != "EquivariantTransformer":
        return
    mpnn_type = config.get("mpnn_type")
    if mpnn_type not in {"PAINN", "PNAEq", "SchNet", "DimeNet", "MACE"}:
        raise ValueError(
            "EquivariantTransformer model integration currently supports "
            "PAINN, PNAEq, SchNet, DimeNet, and MACE; "
            "the other adapters remain unavailable until their integration tests pass"
        )
    if config.get("global_attn_heads", 0) <= 0:
        raise ValueError("EquivariantTransformer requires global_attn_heads > 0")
    if config["equivariant_attn_lmax"] < 0:
        raise ValueError("equivariant_attn_lmax must be nonnegative")
    if config["equivariant_attn_num_radial"] <= 0:
        raise ValueError("equivariant_attn_num_radial must be positive")
    if config["equivariant_attn_feedforward_multiplier"] <= 0:
        raise ValueError("equivariant_attn_feedforward_multiplier must be positive")
    chunk_size = config.get("equivariant_attn_chunk_size", 512)
    if chunk_size is not None:
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError(
                "equivariant_attn_chunk_size must be a positive integer or null"
            )
    if config.get("equivariant_attn_coupling_mode", "parallel") not in {
        "parallel",
        "sequential",
    }:
        raise ValueError(
            "equivariant_attn_coupling_mode must be 'parallel' or 'sequential'"
        )
    if not isinstance(config.get("equivariant_attn_periodic", False), bool):
        raise TypeError("equivariant_attn_periodic must be a boolean")
    replication = config.get("equivariant_attn_periodic_replication", 1)
    if isinstance(replication, int) and not isinstance(replication, bool):
        replication = [replication] * 3
    if (
        not isinstance(replication, (list, tuple))
        or len(replication) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in replication
        )
    ):
        raise ValueError(
            "equivariant_attn_periodic_replication must be a nonnegative "
            "integer or a length-3 list of nonnegative integers"
        )
    if (
        mpnn_type in {"PAINN", "PNAEq", "MACE"}
        and not config["equivariant_attn_require_tensor_coupling"]
    ):
        raise ValueError(
            f"{mpnn_type} provides vector features; tensor coupling "
            "must remain enabled"
        )
    if mpnn_type in {"SchNet", "DimeNet"}:
        if config["equivariant_attn_require_tensor_coupling"]:
            raise ValueError(
                f"{mpnn_type} cannot provide tensor-valued local/global coupling"
            )
        if not config["equivariant_attn_allow_scalar_only"]:
            raise ValueError(
                f"{mpnn_type} requires equivariant_attn_allow_scalar_only=true"
            )
    if mpnn_type == "SchNet" and config.get("equivariance"):
        raise ValueError(
            "SchNet with EquivariantTransformer cannot use coordinate updates; "
            "set Architecture.equivariance=false"
        )
    if mpnn_type == "MACE" and config.get("num_conv_layers", 0) < 2:
        raise ValueError(
            "MACE with EquivariantTransformer requires at least two convolution "
            "layers because MACE's final layer contains scalar irreps only"
        )


def validate_local_sgd_config(training):
    """Validate and fill defaults for optional post-local-SGD training."""
    local_sgd = training.setdefault("LocalSGD", {"enabled": False})
    if not isinstance(local_sgd, dict):
        raise TypeError("Training.LocalSGD must be a JSON object")
    local_sgd.setdefault("enabled", False)
    if not isinstance(local_sgd["enabled"], bool):
        raise TypeError("Training.LocalSGD.enabled must be a boolean")
    if not local_sgd["enabled"]:
        return

    local_sgd.setdefault("warmup_steps", 0)
    local_sgd.setdefault("synchronization_period", 1)
    local_sgd.setdefault("optimizer_state_policy", "local")
    local_sgd.setdefault("optimizer_state_bucket_bytes", 25 * 1024 * 1024)
    if (
        isinstance(local_sgd["warmup_steps"], bool)
        or not isinstance(local_sgd["warmup_steps"], int)
        or local_sgd["warmup_steps"] < 0
    ):
        raise ValueError("Training.LocalSGD.warmup_steps must be an integer >= 0")
    if (
        isinstance(local_sgd["synchronization_period"], bool)
        or not isinstance(local_sgd["synchronization_period"], int)
        or local_sgd["synchronization_period"] < 1
    ):
        raise ValueError(
            "Training.LocalSGD.synchronization_period must be an integer >= 1"
        )
    if local_sgd["optimizer_state_policy"] not in {"local", "synchronize"}:
        raise ValueError(
            "Training.LocalSGD.optimizer_state_policy must be 'local' or "
            "'synchronize'"
        )
    bucket_bytes = local_sgd["optimizer_state_bucket_bytes"]
    if (
        isinstance(bucket_bytes, bool)
        or not isinstance(bucket_bytes, int)
        or bucket_bytes < 1
    ):
        raise ValueError(
            "Training.LocalSGD.optimizer_state_bucket_bytes must be an integer >= 1"
        )


def update_config_equivariance(config):
    equivariance_toggled_models = ["EGNN"]
    if "equivariance" in config:
        if config["mpnn_type"] not in equivariance_toggled_models:
            warnings.warn(
                f"E(3) equivariance can only be toggled for EGNN. Setting it for {config['mpnn_type']} won't break anything,"
                "but won't change anything either."
            )
    else:
        config["equivariance"] = None
    return config


def update_config_edge_dim(config):
    def _normalize_edge_dim(value):
        if value is None:
            return None
        if isinstance(value, dict):
            # Per-edge-type widths (heterogeneous route).
            return {str(k): int(v) for k, v in value.items()}
        try:
            edge_dim = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"edge_dim must be an integer or dict, got: {value}"
            ) from exc
        if edge_dim < 0:
            raise ValueError(f"edge_dim must be >= 0, got: {edge_dim}")
        return edge_dim

    explicit_edge_dim = _normalize_edge_dim(config.get("edge_dim"))
    if explicit_edge_dim is None:
        raise ValueError(
            "NeuralNetwork.Architecture.edge_dim is required. "
            "Set edge_dim explicitly in the input config."
        )

    if isinstance(explicit_edge_dim, int):
        feature_names = config.get("edge_feature_names")
        if feature_names:
            names_len = len(feature_names)
            if names_len != explicit_edge_dim:
                raise ValueError(
                    "NeuralNetwork.Architecture.edge_feature_names length "
                    f"({names_len}) must match edge_dim ({explicit_edge_dim})."
                )

    config["edge_dim"] = explicit_edge_dim
    return config


def get_log_name_config(config):
    input_names = "-".join(
        sanitize_filename_component(spec.name)
        for spec in get_variable_schema(config).inputs
    )
    return (
        config["NeuralNetwork"]["Architecture"]["mpnn_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-data-"
        + config["Dataset"]["name"][
            : (
                config["Dataset"]["name"].rfind("_")
                if config["Dataset"]["name"].rfind("_") > 0
                else None
            )
        ]
        + "-node_ft-"
        + input_names
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )


def save_config(config, log_name, path="./logs/"):
    """Save config"""
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
        fname = os.path.join(path, log_name, "config.json")
        with open(fname, "w") as f:
            json.dump(config, f, indent=4)


def parse_deepspeed_config(config):
    # first, check if we have a ds_config section in the config
    if "ds_config" in config["NeuralNetwork"]:
        ds_config = config["NeuralNetwork"]["ds_config"]
    else:
        ds_config = {}

    if "train_micro_batch_size_per_gpu" not in ds_config:
        ds_config["train_micro_batch_size_per_gpu"] = config["NeuralNetwork"][
            "Training"
        ]["batch_size"]
        ds_config["gradient_accumulation_steps"] = 1

    if "steps_per_print" not in ds_config:
        ds_config["steps_per_print"] = 1e9  # disable printing

    return ds_config


def merge_config(a: dict, b: dict) -> dict:
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = merge_config(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result
