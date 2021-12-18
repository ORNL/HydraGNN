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

import sys, os, json
import pytest

import torch
import shutil

import hydragnn, tests

# Main unit test function called by pytest wrappers.
def unittest_train_model(model_type, ci_input, use_lengths, overwrite_data=False):

    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    """
    to test this locally, set ci.json as
    "Dataset": {
       ...
       "path": {
               "train": "serialized_dataset/unit_test_singlehead_train.pkl",
               "test": "serialized_dataset/unit_test_singlehead_test.pkl",
               "validate": "serialized_dataset/unit_test_singlehead_validate.pkl"}
       ...
    """
    # use pkl files if exist by default
    for dataset_name in config["Dataset"]["path"].keys():
        if dataset_name == "total":
            pkl_file = (
                os.environ["SERIALIZED_DATA_PATH"]
                + "/serialized_dataset/"
                + config["Dataset"]["name"]
                + ".pkl"
            )
        else:
            pkl_file = (
                os.environ["SERIALIZED_DATA_PATH"]
                + "/serialized_dataset/"
                + config["Dataset"]["name"]
                + "_"
                + dataset_name
                + ".pkl"
            )
        if os.path.exists(pkl_file):
            config["Dataset"]["path"][dataset_name] = pkl_file

    # In the unit test runs, it is found MFC favors graph-level features over node-level features, compared with other models;
    # hence here we decrease the loss weight coefficient for graph-level head in MFC.
    if model_type == "MFC" and ci_input == "ci_multihead.json":
        config["NeuralNetwork"]["Architecture"]["task_weights"][0] = 2

    # Only run with edge lengths for models that support them.
    if use_lengths:
        config["NeuralNetwork"]["Architecture"]["edge_features"] = ["lengths"]

    if rank == 0:
        num_samples_tot = 500
        # check if serialized pickle files or folders for raw files provided
        pkl_input = False
        if list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
            pkl_input = True
        # only generate new datasets, if not pkl
        if not pkl_input:
            for dataset_name, data_path in config["Dataset"]["path"].items():
                if overwrite_data:
                    shutil.rmtree(data_path)
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                if dataset_name == "total":
                    num_samples = num_samples_tot
                elif dataset_name == "train":
                    num_samples = int(
                        num_samples_tot
                        * config["NeuralNetwork"]["Training"]["perc_train"]
                    )
                elif dataset_name == "test":
                    num_samples = int(
                        num_samples_tot
                        * (1 - config["NeuralNetwork"]["Training"]["perc_train"])
                        * 0.5
                    )
                elif dataset_name == "validate":
                    num_samples = int(
                        num_samples_tot
                        * (1 - config["NeuralNetwork"]["Training"]["perc_train"])
                        * 0.5
                    )
                if not os.listdir(data_path):
                    tests.deterministic_graph_data(
                        data_path, number_configurations=num_samples
                    )

    # Since the config file uses PNA already, test the file overload here.
    # All the other models need to use the locally modified dictionary.
    if model_type == "PNA" and not use_lengths:
        hydragnn.run_training(config_file)
    else:
        hydragnn.run_training(config)

    (
        error,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = hydragnn.run_prediction(config)

    # Set RMSE and sample error thresholds
    thresholds = {
        "PNA": [0.10, 0.25],
        "MFC": [0.10, 0.50],
        "GIN": [0.15, 0.90],
        "GAT": [0.80, 0.95],
        # fixme: error for cgcnn will be reduced after edge attributes being implemented
        "CGCNN": [0.30, 0.95],
    }
    verbosity = 2

    for ihead in range(len(true_values)):
        error_head_rmse = error_rmse_task[ihead]
        error_str = (
            str("{:.6f}".format(error_head_rmse))
            + " < "
            + str(thresholds[model_type][0])
        )
        hydragnn.utils.print_distributed(verbosity, "head: " + error_str)
        assert (
            error_head_rmse < thresholds[model_type][0]
        ), "RMSE checking failed for components of head " + str(ihead)

        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        # Check individual samples
        sample_error_sum = 0.0
        sample_error_min = 1.0
        sample_error_max = 0.0
        for true_value, predicted_value in zip(head_true, head_pred):
            for idim in range(len(true_value)):
                sample_error = abs(true_value[idim] - predicted_value[idim])
                sample_error_sum += sample_error
                if sample_error < sample_error_min:
                    sample_error_min = sample_error
                if sample_error > sample_error_max:
                    sample_error_max = sample_error
        num_samples = len(head_pred) * len(true_value)
        error_str = (
            "{:.6f}".format(sample_error_sum / num_samples)
            + " / "
            + "{:.6f}".format(sample_error_min)
            + " / "
            + "{:.6f}".format(sample_error_max)
            + " < "
            + str(thresholds[model_type][1])
        )
        hydragnn.utils.print_distributed(verbosity, "samples avg/min/max: " + error_str)
        assert sample_error_max < thresholds[model_type][1], "Samples checking failed!"

    # Check RMSE error
    error_str = str("{:.6f}".format(error)) + " < " + str(thresholds[model_type][0])
    hydragnn.utils.print_distributed(verbosity, "total: " + error_str)
    assert error < thresholds[model_type][0], "Total RMSE checking failed!" + str(error)


# Test across all models with both single/multihead
@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNA", "CGCNN"])
@pytest.mark.parametrize("ci_input", ["ci.json", "ci_multihead.json"])
def pytest_train_model(model_type, ci_input, overwrite_data=False):
    unittest_train_model(model_type, ci_input, False, overwrite_data)


# Test only models
@pytest.mark.parametrize("model_type", ["PNA", "CGCNN"])
def pytest_train_model_lengths(model_type, overwrite_data=False):
    unittest_train_model(model_type, "ci.json", True, overwrite_data)
