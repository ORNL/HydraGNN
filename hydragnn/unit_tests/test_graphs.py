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

import hydragnn, hydragnn.unit_tests


@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNA", "CGCNN"])
@pytest.mark.parametrize("ci_input", ["ci.json", "ci_multihead.json"])
def pytest_train_model(model_type, ci_input, overwrite_data=False):

    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "examples", ci_input)
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    if rank == 0:
        num_samples_tot = 800
        for dataset_name, data_path in config["Dataset"]["path"]["raw"].items():
            if overwrite_data:
                shutil.rmtree(data_path)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            if dataset_name == "total":
                num_samples = num_samples_tot
            elif dataset_name == "train":
                num_samples = int(
                    num_samples_tot * config["NeuralNetwork"]["Training"]["perc_train"]
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
                num_nodes = config["Dataset"]["num_nodes"]
                if num_nodes == 4:
                    hydragnn.unit_tests.deterministic_graph_data(
                        data_path,
                        number_unit_cell_x_range=[1, 2],
                        number_unit_cell_y_range=[1, 2],
                        number_configurations=num_samples,
                    )
                else:
                    hydragnn.unit_tests.deterministic_graph_data(
                        data_path, number_configurations=num_samples
                    )

    tmp_file = "./tmp.json"
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    with open(tmp_file, "w") as f:
        json.dump(config, f)

    hydragnn.run_training(tmp_file)

    (
        error,
        error_sumofnodes_task,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = hydragnn.run_prediction(tmp_file, model_type)

    # Set RMSE and sample error thresholds
    thresholds = {
        "PNA": [0.10, 0.20],
        "MFC": [0.10, 0.25],
        "GIN": [0.10, 0.85],
        "GAT": [0.80, 0.85],
        "CGCNN": [0.10, 0.40],
    }
    verbosity = 2

    for ihead in range(len(true_values)):
        error_head_sum = error_sumofnodes_task[ihead] / len(true_values[ihead][0])
        error_str = (
            str("{:.6f}".format(error_head_sum))
            + " < "
            + str(thresholds[model_type][0])
        )
        hydragnn.utils.print_distributed(verbosity, "head sum: " + error_str)
        assert (
            error_head_sum < thresholds[model_type][0]
        ), "RMSE checking failed for sum of head " + str(ihead)

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
                assert (
                    sample_error < thresholds[model_type][1]
                ), "Samples checking failed!"
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

    # Check RMSE error
    error_str = str("{:.6f}".format(error)) + " < " + str(thresholds[model_type][0])
    hydragnn.utils.print_distributed(verbosity, "total: " + error_str)
    assert error < thresholds[model_type][0], "Total RMSE checking failed!" + str(error)
