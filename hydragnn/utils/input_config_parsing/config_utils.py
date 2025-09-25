##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import pickle
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
import torch


def update_config(config, train_loader, val_loader, test_loader):
    """check if config input consistent and update config with model and datasets"""

    graph_size_variable = os.getenv("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE")
    if graph_size_variable is None:
        graph_size_variable = check_if_graph_size_variable(
            train_loader, val_loader, test_loader
        )
    else:
        graph_size_variable = bool(int(graph_size_variable))

    if "Dataset" in config:
        check_output_dim_consistent(train_loader.dataset[0], config)

    # Set default values for GPS variables
    if "global_attn_engine" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = None
    if "global_attn_type" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_type"] = None
    if "global_attn_heads" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["global_attn_heads"] = 0
    if "pe_dim" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["pe_dim"] = 0

    # update output_heads with latest config rules
    config["NeuralNetwork"]["Architecture"]["output_heads"] = update_multibranch_heads(
        config["NeuralNetwork"]["Architecture"]["output_heads"]
    )

    config["NeuralNetwork"] = update_config_NN_outputs(
        config["NeuralNetwork"], train_loader.dataset[0], graph_size_variable
    )

    config = normalize_output_config(config)

    config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
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
            avg_num_neighbors = torch.tensor(train_loader.dataset.avg_num_neighbors)
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

    config["NeuralNetwork"]["Architecture"] = update_config_edge_dim(
        config["NeuralNetwork"]["Architecture"]
    )

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

    return config


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
    config["edge_dim"] = None
    edge_models = [
        "GAT",
        "PNA",
        "PNAPlus",
        "PAINN",
        "PNAEq",
        "CGCNN",
        "SchNet",
        "EGNN",
        "DimeNet",
        "MACE",
    ]
    if "edge_features" in config and config["edge_features"]:
        assert (
            config["mpnn_type"] in edge_models
        ), "Edge features can only be used with GAT, PNA, PNAPlus, PAINN, PNAEq, CGCNN, SchNet, EGNN, DimeNet, MACE."
        config["edge_dim"] = len(config["edge_features"])
        if "enable_interatomic_potential" in config:
            assert not config[
                "enable_interatomic_potential"
            ], "Edge features cannot be used with interatomic potentials as the model builds its own specialized features for force computation."
    elif config["mpnn_type"] == "CGCNN":
        # CG always needs an integer edge_dim
        # PNA, PNAPlus, and DimeNet would fail with integer edge_dim without edge_attr
        config["edge_dim"] = 0
    return config


def check_output_dim_consistent(data, config):
    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    if hasattr(data, "y_loc"):
        for ihead in range(len(output_type)):
            if output_type[ihead] == "graph":
                assert (
                    data.y_loc[0, ihead + 1].item() - data.y_loc[0, ihead].item()
                    == config["Dataset"]["graph_features"]["dim"][output_index[ihead]]
                )
            elif output_type[ihead] == "node":
                assert (
                    data.y_loc[0, ihead + 1].item() - data.y_loc[0, ihead].item()
                ) // data.num_nodes == config["Dataset"]["node_features"]["dim"][
                    output_index[ihead]
                ]


def update_config_NN_outputs(config, data, graph_size_variable):
    """ "Extract architecture output dimensions and set node-level prediction architecture"""

    output_type = config["Variables_of_interest"]["type"]
    if config["Architecture"].get("enable_interatomic_potential", False):
        dims_list = config["Variables_of_interest"]["output_dim"]
    elif hasattr(data, "y_loc"):
        dims_list = []
        for ihead in range(len(output_type)):
            if output_type[ihead] == "graph":
                dim_item = data.y_loc[0, ihead + 1].item() - data.y_loc[0, ihead].item()
            elif output_type[ihead] == "node":
                # FIXME: check the first branch only, assuming all branches have the same type
                if (
                    graph_size_variable
                    and config["Architecture"]["output_heads"]["node"][0][
                        "architecture"
                    ]["type"]
                    == "mlp_per_node"
                ):
                    raise ValueError(
                        '"mlp_per_node" is not allowed for variable graph size, Please set config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] to be "mlp" or "conv" in input file.'
                    )
                dim_item = (
                    data.y_loc[0, ihead + 1].item() - data.y_loc[0, ihead].item()
                ) // data.num_nodes
            else:
                raise ValueError("Unknown output type", output_type[ihead])
            dims_list.append(dim_item)
    else:
        for ihead in range(len(output_type)):
            if output_type[ihead] != "graph":
                raise ValueError(
                    "y_loc is needed for outputs that are not at graph levels",
                    output_type[ihead],
                )
        dims_list = config["Variables_of_interest"]["output_dim"]

    config["Architecture"]["output_dim"] = dims_list
    config["Architecture"]["output_type"] = output_type
    config["Architecture"]["num_nodes"] = data.num_nodes
    return config


def normalize_output_config(config):
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    if "denormalize_output" in var_config and var_config["denormalize_output"]:
        if (
            var_config.get("minmax_node_feature") is not None
            and var_config.get("minmax_graph_feature") is not None
        ):
            dataset_path = None
        ###loading min/max values from input data file. Only one path is needed
        elif list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
            dataset_path = list(config["Dataset"]["path"].values())[0]
        else:
            if "total" in config["Dataset"]["path"].keys():
                dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
            else:
                dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_train.pkl"
        var_config = update_config_minmax(dataset_path, var_config)
    else:
        var_config["denormalize_output"] = False

    config["NeuralNetwork"]["Variables_of_interest"] = var_config
    return config


def update_config_minmax(dataset_path, config):
    """load minimum and maximum values from dataset_path, if need denormalize,"""
    ## Check first if "minmax_graph_feature" and "minmax_graph_feature"
    if "minmax_node_feature" not in config and "minmax_graph_feature" not in config:
        with open(dataset_path, "rb") as f:
            node_minmax = pickle.load(f)
            graph_minmax = pickle.load(f)
    else:
        node_minmax = config["minmax_node_feature"]
        graph_minmax = config["minmax_graph_feature"]
    config["x_minmax"] = []
    config["y_minmax"] = []
    feature_indices = [i for i in config["input_node_features"]]
    for item in feature_indices:
        config["x_minmax"].append(node_minmax[:, item].tolist())
    output_type = config["type"]
    output_index = config["output_index"]
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            config["y_minmax"].append(graph_minmax[:, output_index[item]].tolist())
        elif output_type[item] == "node":
            config["y_minmax"].append(node_minmax[:, output_index[item]].tolist())
        else:
            raise ValueError("Unknown output type", output_type[item])
    return config


def get_log_name_config(config):
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
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
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
