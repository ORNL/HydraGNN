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
import torch.distributed as dist

torch.manual_seed(97)
import shutil

import hydragnn, tests

from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.lsmsdataset import LSMSDataset
from hydragnn.utils.datasets.serializeddataset import (
    SerializedWriter,
    SerializedDataset,
)

from hydragnn.preprocess.load_data import split_dataset


# Main unit test function called by pytest wrappers.
def unittest_train_model_dataset_class_inheritance(
    model_type, ci_input, use_lengths, overwrite_data=False
):
    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type

    if rank == 0:
        num_samples_tot = 500
        for dataset_name, data_path in config["Dataset"]["path"].items():
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
                tests.deterministic_graph_data(
                    data_path, number_configurations=num_samples
                )

    hydragnn.utils.setup_log(get_log_name_config(config))

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    dirpwd = dirpwd.replace("/tests", "")
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    datasetname = config["Dataset"]["name"]
    for dataset_type, raw_data_path in config["Dataset"]["path"].items():
        config["Dataset"]["path"][dataset_type] = os.path.join(dirpwd, raw_data_path)

    # In the unit test runs, it is found MFC favors graph-level features over node-level features, compared with other models;
    # hence here we decrease the loss weight coefficient for graph-level head in MFC.
    if model_type == "MFC" and ci_input == "ci_multihead.json":
        config["NeuralNetwork"]["Architecture"]["task_weights"][0] = 2

    # Only run with edge lengths for models that support them.
    if use_lengths:
        config["NeuralNetwork"]["Architecture"]["edge_features"] = ["lengths"]

    if 0 == rank:

        ## Only rank=0 is enough for pre-processing
        total = LSMSDataset(config)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
        )
        print(len(total), len(trainset), len(valset), len(testset))

        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "serialized_dataset"
        )

        SerializedWriter(
            trainset,
            basedir,
            datasetname,
            "trainset",
            minmax_node_feature=total.minmax_node_feature,
            minmax_graph_feature=total.minmax_graph_feature,
        )
        SerializedWriter(
            valset,
            basedir,
            datasetname,
            "valset",
        )
        SerializedWriter(
            testset,
            basedir,
            datasetname,
            "testset",
        )
    dist.barrier()

    print("Pickle load")
    basedir = os.path.join(os.path.dirname(__file__), "dataset", "serialized_dataset")
    trainset = SerializedDataset(basedir, datasetname, "trainset")
    valset = SerializedDataset(basedir, datasetname, "valset")
    testset = SerializedDataset(basedir, datasetname, "testset")

    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    print(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    if 0 == rank:
        print_model(model)
    dist.barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
    )

    if writer is not None:
        writer.close()


# Test vector output
## FIXME: this is temporarily disabled to avoid multiple instantiations of DDP
@pytest.mark.parametrize("model_type", ["PNA"])
@pytest.mark.skip()
def pytest_train_model_vectoroutput(model_type, overwrite_data=False):
    unittest_train_model_dataset_class_inheritance(
        model_type, "ci_vectoroutput.json", True, overwrite_data
    )
