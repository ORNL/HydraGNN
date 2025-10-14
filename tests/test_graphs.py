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

import sys, os, json
import pytest

import torch

torch.manual_seed(97)
import shutil

import hydragnn, tests
from hydragnn.utils.input_config_parsing.config_utils import merge_config
from mpi4py import MPI

# Main unit test function called by pytest wrappers.
def unittest_train_model(
    mpnn_type,
    global_attn_engine,
    global_attn_type,
    ci_input,
    use_lengths,
    overwrite_data=False,
    use_deepspeed=False,
    overwrite_config=None,
):
    world_size, rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = global_attn_engine
    config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type
    config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type

    # Overwrite config settings if provided
    if overwrite_config:
        config = merge_config(config, overwrite_config)

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
    if mpnn_type == "MFC" and ci_input == "ci_multihead.json":
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
    MPI.COMM_WORLD.Barrier()

    # Since the config file uses PNA already, test the file overload here.
    # All the other models need to use the locally modified dictionary.
    if mpnn_type == "PNA" and not use_lengths:
        hydragnn.run_training(config_file, use_deepspeed)
    else:
        hydragnn.run_training(config, use_deepspeed)

    (
        error,
        error_mse_task,
        true_values,
        predicted_values,
    ) = hydragnn.run_prediction(config, use_deepspeed)

    # Set RMSE and sample MAE error thresholds
    thresholds = {
        "SAGE": [0.20, 0.20],
        "PNA": [0.20, 0.20],
        "PNAPlus": [0.20, 0.20],
        "MFC": [0.20, 0.30],
        "GIN": [0.25, 0.20],
        "GAT": [0.60, 0.70],
        "CGCNN": [0.50, 0.40],
        "SchNet": [0.20, 0.20],
        "DimeNet": [0.50, 0.50],
        "EGNN": [0.20, 0.20],
        "PNAEq": [0.60, 0.60],
        "PAINN": [0.60, 0.60],
        "MACE": [0.60, 0.70],
    }
    if use_lengths and ("vector" not in ci_input):
        thresholds["CGCNN"] = [0.175, 0.175]
        thresholds["PNA"] = [0.10, 0.10]
        thresholds["PNAPlus"] = [0.10, 0.10]
    if use_lengths and "vector" in ci_input:
        thresholds["PNA"] = [0.2, 0.15]
        thresholds["PNAPlus"] = [0.2, 0.15]
    if ci_input == "ci_conv_head.json":
        thresholds["GIN"] = [0.26, 0.51]
        thresholds["SchNet"] = [0.30, 0.30]

    verbosity = 2

    for ihead in range(len(true_values)):
        error_head_mse = error_mse_task[ihead]
        error_str = (
            str("{:.6f}".format(error_head_mse)) + " < " + str(thresholds[mpnn_type][0])
        )
        hydragnn.utils.print.print_distributed(verbosity, "head: " + error_str)
        assert (
            error_head_mse < thresholds[mpnn_type][0]
        ), "Head RMSE checking failed for " + str(ihead)

        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        # Check individual samples
        mae = torch.nn.L1Loss()
        sample_mean_abs_error = mae(head_true, head_pred)
        error_str = (
            "{:.6f}".format(sample_mean_abs_error)
            + " < "
            + str(thresholds[mpnn_type][1])
        )
        assert (
            sample_mean_abs_error < thresholds[mpnn_type][1]
        ), f"MAE sample checking failed! MAE: {sample_mean_abs_error:.6f} >= threshold: {thresholds[mpnn_type][1]} for model: {mpnn_type}"

    # Check RMSE error
    error_str = str("{:.6f}".format(error)) + " < " + str(thresholds[mpnn_type][0])
    hydragnn.utils.print.print_distributed(verbosity, "total: " + error_str)
    assert error < thresholds[mpnn_type][0], "Total RMSE checking failed!" + str(error)


# Test across all models with both single/multihead
@pytest.mark.parametrize(
    "mpnn_type",
    [
        "SAGE",
        "GIN",
        "GAT",
        "MFC",
        "PNA",
        "PNAPlus",
        "CGCNN",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
        "MACE",
    ],
)
@pytest.mark.parametrize("ci_input", ["ci.json", "ci_multihead.json"])
def pytest_train_model(mpnn_type, ci_input, overwrite_data=False):
    unittest_train_model(mpnn_type, None, None, ci_input, False, overwrite_data)


# Test models that allow edge attributes
@pytest.mark.parametrize(
    "mpnn_type",
    ["GAT", "PNA", "PNAPlus", "CGCNN", "SchNet", "DimeNet", "EGNN", "PNAEq", "PAINN"],
)
def pytest_train_model_lengths(mpnn_type, overwrite_data=False):
    unittest_train_model(mpnn_type, None, None, "ci.json", True, overwrite_data)


# Test models that allow edge attributes with global attention mechanisms
@pytest.mark.parametrize(
    "global_attn_engine",
    ["GPS"],
)
@pytest.mark.parametrize("global_attn_type", ["multihead"])
@pytest.mark.parametrize(
    "mpnn_type",
    ["GAT", "PNA", "PNAPlus", "CGCNN", "SchNet", "DimeNet", "EGNN", "PNAEq", "PAINN"],
)
def pytest_train_model_lengths_global_attention(
    mpnn_type, global_attn_engine, global_attn_type, overwrite_data=False
):
    unittest_train_model(
        mpnn_type, global_attn_engine, global_attn_type, "ci.json", True, overwrite_data
    )


# Test only models
@pytest.mark.parametrize(
    "mpnn_type",
    ["MACE"],
)
def pytest_train_mace_model_lengths(mpnn_type, overwrite_data=False):
    unittest_train_model(mpnn_type, None, None, "ci.json", True, overwrite_data)


# Test across equivariant models
@pytest.mark.parametrize("mpnn_type", ["EGNN", "SchNet", "PNAEq", "PAINN", "MACE"])
def pytest_train_equivariant_model(mpnn_type, overwrite_data=False):
    unittest_train_model(
        mpnn_type, None, None, "ci_equivariant.json", False, overwrite_data
    )


# Test vector output
@pytest.mark.parametrize(
    "mpnn_type",
    [
        "GAT",
        "PNA",
        "PNAPlus",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
    ],
)
def pytest_train_model_vectoroutput(mpnn_type, overwrite_data=False):
    unittest_train_model(
        mpnn_type, None, None, "ci_vectoroutput.json", True, overwrite_data
    )


@pytest.mark.parametrize(
    "mpnn_type",
    [
        "SAGE",
        "GIN",
        "GAT",
        "MFC",
        "PNA",
        "PNAPlus",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
    ],
)
def pytest_train_model_conv_head(mpnn_type, overwrite_data=False):
    unittest_train_model(
        mpnn_type, None, None, "ci_conv_head.json", False, overwrite_data
    )
