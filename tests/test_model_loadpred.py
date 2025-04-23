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
import torch
import random
import hydragnn
from tests.test_graphs import unittest_train_model
from hydragnn.utils.input_config_parsing.config_utils import update_config


def unittest_model_prediction(config):
    _, _ = hydragnn.utils.distributed.setup_ddp()

    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.load_data.dataset_loading_and_splitting(config=config)
    config = update_config(config, train_loader, val_loader, test_loader)

    model = hydragnn.models.create.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = hydragnn.utils.distributed.get_distributed_model(
        model, config["Verbosity"]["level"]
    )

    log_name = hydragnn.utils.input_config_parsing.get_log_name_config(config)
    hydragnn.utils.model.load_existing_model(model, log_name)

    model.eval()
    # two checkings
    # 1. entire test set
    thresholds = [0.2]
    _, _, true_values, predicted_values = hydragnn.train.test(
        test_loader, model, config["Verbosity"]["level"]
    )
    # 2.check a random selected sample
    isample = random.randrange(len(test_loader.dataset))
    graph_sample = test_loader.dataset[isample].to(model.device)
    true_sample = graph_sample.y.squeeze()
    predicted_sample = model(graph_sample)
    yloc = graph_sample.y_loc.squeeze()

    mae = torch.nn.L1Loss()
    for ihead in range(model.module.num_heads):
        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        test_mae = mae(head_true, head_pred)
        print("For head ", ihead, "; MAE of test set =", test_mae)
        assert test_mae < thresholds[0], "MAE sample checking failed for test set!"


# test loading and predictiong of a saved model from previous training
def pytest_model_loadpred():
    model_type = "PNA"
    ci_input = "ci_multihead.json"
    config_file = os.path.join(os.getcwd(), "tests/inputs", ci_input)
    with open(config_file, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    # get the directory of trained model
    log_name = hydragnn.utils.input_config_parsing.config_utils.get_log_name_config(
        config
    )
    modelfile = os.path.join("./logs/", log_name, log_name + ".pk")
    # check if pretrained model and pkl datasets files exists
    case_exist = True
    config_file = os.path.join("./logs/", log_name, "config.json")
    if not (os.path.isfile(modelfile) and os.path.isfile(config_file)):
        print("Model or configure file not found: ", modelfile, config_file)
        case_exist = False
    else:
        with open(config_file, "r") as f:
            config = json.load(f)
        for dataset_name, raw_data_path in config["Dataset"]["path"].items():
            if not os.path.isfile(raw_data_path):
                print(dataset_name, "datasets not found: ", raw_data_path)
                case_exist = False
                break
    if not case_exist:
        unittest_train_model(
            config["NeuralNetwork"]["Architecture"]["model_type"],
            "ci_multihead.json",
            False,
            False,
        )
    unittest_model_prediction(config)
