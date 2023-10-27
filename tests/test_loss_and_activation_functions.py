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

import os, json
import pytest

import shutil

import hydragnn, tests


# Loss function unit test called by pytest wrappers.
# Note the intent of this test is to make sure all interfaces work - it does not assert anything
def unittest_loss_and_activation_functions(
    activation_function_type, loss_function_type, ci_input, overwrite_data=False
):
    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)

    config["NeuralNetwork"]["Architecture"][
        "activation_function"
    ] = activation_function_type

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

    config["NeuralNetwork"]["Training"]["loss_function_type"] = loss_function_type
    config["NeuralNetwork"]["Training"]["num_epoch"] = 2

    # Make sure training works with each loss function.
    hydragnn.run_training(config)


# Test all supported loss function types. Separate input file because only 2 steps are run.
@pytest.mark.parametrize("loss_function_type", ["mse", "mae", "rmse"])
def pytest_loss_functions(loss_function_type, ci_input="ci.json", overwrite_data=False):
    unittest_loss_and_activation_functions(
        "relu", loss_function_type, ci_input, overwrite_data
    )


# Test all supported activation function types.
@pytest.mark.parametrize(
    "activation_function_type",
    ["relu", "selu", "prelu", "elu", "lrelu_01", "lrelu_025", "lrelu_05"],
)
def pytest_activation_functions_multihead(
    activation_function_type, ci_input="ci_multihead.json", overwrite_data=False
):
    unittest_loss_and_activation_functions(
        activation_function_type, "mse", ci_input, overwrite_data
    )


# Test all supported activation function types.
@pytest.mark.parametrize(
    "activation_function_type",
    ["relu", "selu", "prelu", "elu", "lrelu_01", "lrelu_025", "lrelu_05"],
)
def pytest_activation_functions_vectoroutput(
    activation_function_type, ci_input="ci_vectoroutput.json", overwrite_data=False
):
    unittest_loss_and_activation_functions(
        activation_function_type, "mse", ci_input, overwrite_data
    )
