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

import sys, os, json, shutil
import pytest
import torch

import hydragnn, tests
from hydragnn.utils.input_config_parsing.config_utils import merge_config
from mpi4py import MPI
from hydragnn.preprocess import graph_samples_checks_and_updates as gscu
from hydragnn.preprocess import serialized_dataset_loader as sdl
from hydragnn.utils.datasets import pickledataset, distdataset, adiosdataset


torch.manual_seed(97)

CONDITIONING_MODES = ["concat_node", "film", "fuse_pool"]


@pytest.fixture(autouse=True)
def add_graph_attr(monkeypatch):
    """Inject graph_attr during dataset preparation without touching core code."""

    orig_update = gscu.update_predicted_values

    def _wrapped(type, index, graph_feature_dim, node_feature_dim, data):
        res = orig_update(type, index, graph_feature_dim, node_feature_dim, data)
        if hasattr(data, "x") and data.x.numel() > 0:
            first_val = data.x[0, 0]
            matches = torch.isclose(data.x[:, 0], first_val)
            data.graph_attr = matches.sum().unsqueeze(0).to(data.x.dtype)
        return res

    monkeypatch.setattr(gscu, "update_predicted_values", _wrapped)
    monkeypatch.setattr(sdl, "update_predicted_values", _wrapped)
    monkeypatch.setattr(pickledataset, "update_predicted_values", _wrapped)
    monkeypatch.setattr(distdataset, "update_predicted_values", _wrapped)
    monkeypatch.setattr(adiosdataset, "update_predicted_values", _wrapped)
    yield
    monkeypatch.setattr(gscu, "update_predicted_values", orig_update)
    monkeypatch.setattr(sdl, "update_predicted_values", orig_update, raising=False)
    monkeypatch.setattr(
        pickledataset, "update_predicted_values", orig_update, raising=False
    )
    monkeypatch.setattr(
        distdataset, "update_predicted_values", orig_update, raising=False
    )
    monkeypatch.setattr(
        adiosdataset, "update_predicted_values", orig_update, raising=False
    )


def unittest_train_model_graphattr(
    mpnn_type,
    global_attn_engine,
    global_attn_type,
    ci_input,
    use_lengths,
    overwrite_data=False,
    use_deepspeed=False,
    overwrite_config=None,
    graph_attr_conditioning_mode="concat_node",
):
    """Replicates test_graphs flow while relying on graph_attr conditioning."""
    world_size, rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = global_attn_engine
    config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type
    config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type
    config["NeuralNetwork"]["Architecture"]["use_graph_attr_conditioning"] = True
    config["NeuralNetwork"]["Architecture"][
        "graph_attr_conditioning_mode"
    ] = graph_attr_conditioning_mode

    # Overwrite config settings if provided
    if overwrite_config:
        config = merge_config(config, overwrite_config)

    # Force regeneration so graph_attr is applied even if prior serialized data exist.
    overwrite_data = True

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

    # Only run with edge lengths for models that support them.
    if use_lengths:
        config["NeuralNetwork"]["Architecture"]["edge_features"] = ["lengths"]

    if rank == 0:
        num_samples_tot = 500
        pkl_input = False
        if list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
            pkl_input = True
        if not pkl_input:
            for dataset_name, data_path in config["Dataset"]["path"].items():
                if overwrite_data:
                    shutil.rmtree(data_path, ignore_errors=True)
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

    hydragnn.run_training(config, use_deepspeed)

    (
        error,
        error_mse_task,
        true_values,
        predicted_values,
    ) = hydragnn.run_prediction(config, use_deepspeed)

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
        # EGNN performs worse with the small conv-head config; relax thresholds
        thresholds["EGNN"] = [0.36, 0.36]

    verbosity = 2

    for ihead in range(len(true_values)):
        error_head_mse = error_mse_task[ihead]
        assert error_head_mse < thresholds[mpnn_type][0]
        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        mae = torch.nn.L1Loss()
        sample_mean_abs_error = mae(head_true, head_pred)
        assert sample_mean_abs_error < thresholds[mpnn_type][1]

    assert error < thresholds[mpnn_type][0]


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
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
def pytest_train_model_graphattr(
    mpnn_type, ci_input, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        ci_input,
        False,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
@pytest.mark.parametrize(
    "mpnn_type",
    ["GAT", "PNA", "PNAPlus", "CGCNN", "SchNet", "DimeNet", "EGNN", "PNAEq", "PAINN"],
)
def pytest_train_model_graphattr_lengths(
    mpnn_type, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        "ci.json",
        True,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize(
    "global_attn_engine",
    ["GPS"],
)
@pytest.mark.parametrize("global_attn_type", ["multihead"])
@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
@pytest.mark.parametrize(
    "mpnn_type",
    ["GAT", "PNA", "PNAPlus", "CGCNN", "SchNet", "DimeNet", "EGNN", "PNAEq", "PAINN"],
)
def pytest_train_model_graphattr_lengths_global_attention(
    mpnn_type,
    global_attn_engine,
    global_attn_type,
    graph_attr_conditioning_mode,
    overwrite_data=False,
):
    unittest_train_model_graphattr(
        mpnn_type,
        global_attn_engine,
        global_attn_type,
        "ci.json",
        True,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
@pytest.mark.parametrize(
    "mpnn_type",
    ["MACE"],
)
def pytest_train_mace_model_graphattr_lengths(
    mpnn_type, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        "ci.json",
        True,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
@pytest.mark.parametrize("mpnn_type", ["EGNN", "SchNet", "PNAEq", "PAINN", "MACE"])
def pytest_train_equivariant_model_graphattr(
    mpnn_type, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        "ci_equivariant.json",
        False,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
@pytest.mark.parametrize(
    "mpnn_type",
    ["GAT", "PNA", "PNAPlus", "SchNet", "DimeNet", "EGNN", "PNAEq"],
)
def pytest_train_model_graphattr_vectoroutput(
    mpnn_type, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        "ci_vectoroutput.json",
        True,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )


@pytest.mark.parametrize("graph_attr_conditioning_mode", CONDITIONING_MODES)
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
def pytest_train_model_graphattr_conv_head(
    mpnn_type, graph_attr_conditioning_mode, overwrite_data=False
):
    unittest_train_model_graphattr(
        mpnn_type,
        None,
        None,
        "ci_conv_head.json",
        False,
        overwrite_data,
        graph_attr_conditioning_mode=graph_attr_conditioning_mode,
    )
