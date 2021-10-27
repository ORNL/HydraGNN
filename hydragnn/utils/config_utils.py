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
from hydragnn.utils.print_utils import print_distributed


def update_config_NN_outputs(config, graph_size_variable):
    """ "Extract architecture output dimensions and set node-level prediction architecture"""

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]

    dims_list = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["graph_features"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            if graph_size_variable:
                if (
                    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
                        "type"
                    ]
                    == "mlp_per_node"
                ):
                    verbosity = 2
                    print_distributed(
                        verbosity,
                        "'mlp_per_node' is not allowed for variable graph size, switching to 'mlp'",
                    )
                    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
                        "type"
                    ] = "mlp"

            dim_item = config["Dataset"]["node_features"]["dim"][output_index[item]]
        else:
            raise ValueError("Unknown output type", output_type[item])
        dims_list.append(dim_item)
    config["NeuralNetwork"]["Architecture"]["output_dim"] = dims_list
    config["NeuralNetwork"]["Architecture"]["output_type"] = output_type
    return config


def update_config_minmax(dataset_path, config):
    """load minimum and maximum values from dataset_path, if need denormalize,"""
    with open(dataset_path, "rb") as f:
        node_minmax = pickle.load(f)
        graph_minmax = pickle.load(f)
    config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"] = []
    config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"] = []
    feature_indices = [
        i
        for i in config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
    ]
    for item in feature_indices:
        config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"].append(
            node_minmax[:, item].tolist()
        )
    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                graph_minmax[:, output_index[item]].tolist()
            )
        elif output_type[item] == "node":
            config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                node_minmax[:, output_index[item]].tolist()
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
    return config


def get_model_output_name_config(model, config):
    return (
        model.__str__()
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-mnnn-"
        + str(config["NeuralNetwork"]["Architecture"]["max_neighbours"])
        + "-ncl-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
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
