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
def unittest_optimizers(optimizer_type, use_zero, ci_input, overwrite_data=False):
    world_size, rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)

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

    config["NeuralNetwork"]["Training"]["num_epoch"] = 2
    config["NeuralNetwork"]["Training"]["Optimizer"]["type"] = optimizer_type
    config["NeuralNetwork"]["Training"]["Optimizer"]["use_zero_redundancy"] = use_zero

    hydragnn.run_training(config)


# Test all supported loss function types. Separate input file because only 2 steps are run.
# FusedLAMB should be tested on GPUs
@pytest.mark.parametrize(
    "optimizer_type",
    ["SGD", "Adam", "Adadelta", "Adagrad", "Adamax", "AdamW", "RMSprop"],
)
@pytest.mark.parametrize(
    "use_zero_redundancy",
    [False, True],
)
def pytest_optimizers(
    optimizer_type, use_zero_redundancy, ci_input="ci.json", overwrite_data=False
):
    unittest_optimizers(optimizer_type, use_zero_redundancy, ci_input, overwrite_data)
