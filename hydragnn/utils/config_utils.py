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
from hydragnn.preprocess.utils import check_if_graph_size_variable
from hydragnn.utils.model import calculate_PNA_degree


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

    max_neigh = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
    if config["NeuralNetwork"]["Architecture"]["model_type"] == "PNA":
        deg = calculate_PNA_degree(train_loader.dataset, max_neigh)
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = deg.tolist()
    else:
        config["NeuralNetwork"]["Architecture"]["pna_deg"] = None

    config["NeuralNetwork"]["Architecture"] = update_config_edge_dim(
        config["NeuralNetwork"]["Architecture"]
    )

    return config


def update_config_edge_dim(config):

    config["edge_dim"] = None
    edge_models = ["PNA", "CGCNN"]
    if "edge_features" in config and config["edge_features"]:
        assert (
            config["model_type"] in edge_models
        ), "Edge features can only be used with PNA and CGCNN."
        config["edge_dim"] = len(config["edge_features"])
    elif config["model_type"] == "CGCNN":
        # CG always needs an integer edge_dim
        # PNA would fail with integer edge_dim without edge_attr
        config["edge_dim"] = 0
    return config


def check_output_dim_consistent(data, config):

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]

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
    config["Architecture"]["output_dim"] = dims_list
    config["Architecture"]["output_type"] = output_type
    config["Architecture"]["num_nodes"] = data.num_nodes
    return config


def normalize_output_config(config):
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    if "denormalize_output" in var_config and var_config["denormalize_output"]:
        ###loading min/max values from input data file. Only one path is needed
        if list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
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
    with open(dataset_path, "rb") as f:
        node_minmax = pickle.load(f)
        graph_minmax = pickle.load(f)
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
        + "-mnnn-"
        + str(config["NeuralNetwork"]["Architecture"]["max_neighbours"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-data-"
        + config["Dataset"]["name"]
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
