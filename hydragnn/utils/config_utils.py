##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
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
from hydragnn.preprocess.utils import check_if_graph_size_variable, gather_deg
from hydragnn.utils.model import calculate_PNA_degree
from hydragnn.utils import get_comm_size_and_rank
import time
import json
from torch_geometric.utils import degree
import torch
import torch.distributed as dist


def update_config(config, train_loader, val_loader, test_loader):
    """check if config input consistent and update config with model and datasets"""

    graph_size_variable = check_if_graph_size_variable(
        train_loader, val_loader, test_loader
    )

    if "Dataset" in config:
        check_output_dim_consistent(train_loader.dataset[0], config)

    config["NeuralNetwork"] = update_config_NN_outputs(
        config["NeuralNetwork"], train_loader.dataset[0], graph_size_variable
    )

    config = normalize_output_config(config)

    config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
    )

    if config["NeuralNetwork"]["Architecture"]["model_type"] == "PNA":
        if hasattr(train_loader.dataset, "pna_deg"):
            ## Use max neighbours used in the dataset.
            deg = torch.tensor(train_loader.dataset.pna_deg)
        else:
            deg = gather_deg(train_loader.dataset)
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = deg.tolist()
        config["NeuralNetwork"]["Architecture"]["max_neighbours"] = len(deg) - 1
    else:
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = None

    if "radius" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["radius"] = None
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

    if "Optimizer" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["Optimizer"]["type"] = "AdamW"

    if "loss_function_type" not in config["NeuralNetwork"]["Training"]:
        config["NeuralNetwork"]["Training"]["loss_function_type"] = "mse"

    if "activation_function" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["activation_function"] = "relu"

    if "SyncBatchNorm" not in config["NeuralNetwork"]["Architecture"]:
        config["NeuralNetwork"]["Architecture"]["SyncBatchNorm"] = False
    return config


def update_config_equivariance(config):
    equivariant_models = ["EGNN", "SchNet"]
    if "equivariance" in config and config["equivariance"]:
        assert (
            config["model_type"] in equivariant_models
        ), "E(3) equivariance can only be ensured for EGNN and SchNet."
    elif "equivariance" not in config:
        config["equivariance"] = False
    return config


def update_config_edge_dim(config):
    config["edge_dim"] = None
    edge_models = ["PNA", "CGCNN", "SchNet", "EGNN"]
    if "edge_features" in config and config["edge_features"]:
        assert (
            config["model_type"] in edge_models
        ), "Edge features can only be used with EGNN, SchNet, PNA and CGCNN."
        config["edge_dim"] = len(config["edge_features"])
    elif config["model_type"] == "CGCNN":
        # CG always needs an integer edge_dim
        # PNA would fail with integer edge_dim without edge_attr
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
    if hasattr(data, "y_loc"):
        dims_list = []
        for ihead in range(len(output_type)):
            if output_type[ihead] == "graph":
                dim_item = data.y_loc[0, ihead + 1].item() - data.y_loc[0, ihead].item()
            elif output_type[ihead] == "node":
                if (
                    graph_size_variable
                    and config["Architecture"]["output_heads"]["node"]["type"]
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
        config["NeuralNetwork"]["Architecture"]["model_type"]
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
            : config["Dataset"]["name"].rfind("_")
            if config["Dataset"]["name"].rfind("_") > 0
            else None
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
            json.dump(config, f)
