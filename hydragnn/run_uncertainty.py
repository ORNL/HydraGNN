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

import json, os
from functools import singledispatch
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp, get_comm_size_and_rank
from hydragnn.models.create import create, get_device
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.train.train_validate_test import train_validate_test

@singledispatch
def run_uncertainty(config):
    raise TypeError("Input must be filename string or configuration dictionary.")


@run_uncertainty.register
def _(config_file: str):

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    run_uncertainty(config)


@run_uncertainty.register
def _(config: dict):

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp()

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]

    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["graph_features"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            dim_item = (
                config["Dataset"]["node_features"]["dim"][output_index[item]]
                * config["Dataset"]["num_nodes"]
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)
    config["NeuralNetwork"]["Architecture"]["output_type"] = config["NeuralNetwork"][
        "Variables_of_interest"
    ]["type"]


    #### LOAD THE ORIGINAL DATA LOADERS AND THE TRAINED MODEL
    train_loader, val_loader, test_loader, model = load_loaders_and_trained_model(config)

    #### CREATE THE DATASET LOADERS
    (
        up_train_loader, up_val_loader, up_test_loader, 
        down_train_loader, down_val_loader, down_test_loader
    ) = create_loaders([train_loader, val_loader, test_loader], model, config)

    # #### NORMALIZE THE UP AND DOWN DATASETS
    # normalize_loaders(up_train_loader, up_val_loader, up_test_loader,
    #                   down_train_loader, down_val_loader, down_test_loader)

    
    # #### CREATE AND TRAIN UP & DOWN MODELS
    # model_up = train_model(up_train_loader, up_val_loader, up_test_loader, "up", config)
    #model_down = train_model(down_train_loader, down_val_loader, down_test_loader, "down", config)

    # #### COMPUTE ALL 3 PREDICTIONS ON TRAINING DATA
    # pred_mean, pred_up, pred_down, y = compute_predictions(train_loader, (model, model_up, model_down), config)
    # if world_rank == 0:
    #     print(type(pred_mean), type(y))
    #     print(pred_mean.shape, y.shape)
    #     # print(pred_mean[0][0:10])
    #     # print(pred_up[0][0:10])
    #     # print(pred_down[0][0:10])

    # #### MOVE UPPER AND LOWER BOUNDS
    # quantile = 0.90
    # n_train = len(train_loader.dataset)
    # c_up = moveBound(pred_mean, pred_up, y, quantile, n_train, world_rank)

## The function below is incomplete. Computing f0 and f1 on a distributed environment
## is more involved, we need an all-reduce operation on f2 in every iteration of the while loop
def moveBound(p_mean, p_bound, y, quantile, n_train, rank):

    num_outlier = int(n_train * (1-quantile)/2)
    
    c_up0 = 0.0
    c_up1 = 10.0

    f0 = (y >= p_mean + c_up0 * p_bound).sum()
    f1 = (y >= p_mean + c_up1 * p_bound).sum()
    print(rank, f0, f1)

    dist.all_reduce(f0)
    dist.all_reduce(f1)
    print(rank, f0, f1)

    # f0 -= num_outlier
    # f1 -= num_outlier

    # print('init, f0: {}, f1: {}, c0: {}, c1: {}'.format(f0, f1, c_up0, c_up1))
    
    # iter = 0
    # n_iter = 1000
    # while iter <= n_iter and f0*f1 < 0:
    #     c_up2 = (c_up0 + c_up1)/2.0

    #     f2 = 0.0
    #     for data in train_loader:
    #         data.to(device)
    #         f2 += (data.y >= model(data) + c_up2 * model_up(data)).sum()
    #         f2 -= num_outlier
    #         print('{}, f0: {}, f1: {}, f2: {}, c0: {}, c1: {}, c2: {}'.format(iter, f0, f1, f2, c_up0, c_up1, c_up2))
            
    #     if f2 == 0:
    #         break
    #     elif f2 > 0:
    #         c_up0 = c_up2
    #         f0 = f2
    #     else:
    #         c_up1 = c_up2
    #         f1 = f2
    #     iter += 1
        
    # c_up = c_up2
    # return c_up
    return 0

## Compute the predictions on the given dataset  using the given trained model.
def compute_predictions(loader, models, config):
    
    for m in models:
        m.eval()
        
    device = next(models[0].parameters()).device
    verbosity = config["Verbosity"]["level"]

    pred = [None for i in range(len(models))]
    y = None
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
        for i, m in enumerate(models):
            result = models[i](data)
            if pred[i] == None:
                pred[i] = result
            else:
                pred[i] = torch.cat((pred[i], result), 0)
        if y == None:
            y = data.y
        else:
            y = torch.cat((y, data.y), 0)

    pred.append(y)
    return pred

## This function loads the ORIGINAL dataloaders and the TRAINED model, which will 
## be used to compute the MEAN predictions. Note that MEAN predictions will be
## used to compute the UP and DOWN datasets. 
def load_loaders_and_trained_model(config):
    
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=config["Dataset"]["name"],
    )

    # ## Question: Should we normalize/denormalize any input/output?
    # if (
    #     "denormalize_output" in config["NeuralNetwork"]["Variables_of_interest"]
    #     and config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]
    #     == "True"
    # ):
    #     config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = "False"
    
    model = create(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
        verbosity_level=config["Verbosity"]["level"],
    )

    model_with_config_name = (
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

    state_dict = torch.load(
        "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    
    return train_loader, val_loader, test_loader, model

## This is the function which creates the up and down datasets.
## Currently, every GPU separates their local training dataset into up and down,
## and creates the data loaders based on the local portions. I think this is wrong,
## we should create the loaders on the GLOBAL datasets. For this purpose, we need
## to gather all up and down local samples in all GPUs, i.e., make GLOBAL up and
## down datasets visible to all GPUs. 

@torch.no_grad()
def create_loaders(loaders, model, config):
    
    device = next(model.parameters()).device
    verbosity = config["Verbosity"]["level"]
    batch_size=config["NeuralNetwork"]["Training"]["batch_size"]    
    
    model.eval()
    
    rank = 0
    if dist.is_initialized():
        _, rank = get_comm_size_and_rank()
        
    new_loaders = []
    for l, loader in enumerate(loaders):
        up = []
        down = []
        nEq = 0
        for data in iterate_tqdm(loader, verbosity):
            data = data.to(device)
            diff = model(data) - data.y

            data_copy = data.cpu().detach().clone()
            data_copy.y = diff
            data_list = data_copy.to_data_list()
            size = diff.shape[0]
            for i in range(size):
                if data_copy.y[i] > 0:
                    up.append(data_list[i])
                elif data_copy.y[i] < 0:
                    data_copy.y[i] *= -1.0
                    down.append(data_list[i])
                else:
                    nEq += 1

        lengths = torch.tensor([len(up), len(down), nEq])
        lengths = lengths.to(device)
        dist.all_reduce(lengths)

        if rank == 0:
            print('dataset:', l, 'up', lengths[0], 'down', lengths[1], 'equal', lengths[2])

        ## The code below creates the loaders on the LOCAL up and down samples, which is probably wrong.
        ## We should add a mechanism (either a gather-type communication, or a file write and read) to
        ## make the GLOBAL datasets visible in each GPU.
        
        if dist.is_initialized():
            up_sampler = torch.utils.data.distributed.DistributedSampler(up)
            down_sampler = torch.utils.data.distributed.DistributedSampler(up)

            up_loader = DataLoader(up, batch_size=batch_size, shuffle=False, sampler=up_sampler)
            down_loader = DataLoader(down, batch_size=batch_size, shuffle=False, sampler=down_sampler)
        else:
            up_loader = DataLoader(up, batch_size=batch_size, shuffle=True)
            down_loader = DataLoader(down, batch_size=batch_size, shuffle=True)

        new_loaders.append([up_loader, down_loader])


    return (new_loaders[0][0], new_loaders[1][0], new_loaders[2][0], new_loaders[0][1], new_loaders[1][1], new_loaders[2][1])

## The code below will train a model on the upper OR the lower dataset. 
def train_model(train_loader,
                val_loader,
                test_loader,
                up_down,
                config):
    
    model = create(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
        verbosity_level=config["Verbosity"]["level"],
    )

    model_with_config_name = (model.__str__() + "-" + up_down)

    device_name, device = get_device(config["Verbosity"]["level"])
    if dist.is_initialized():
        if device_name == "cpu":
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["NeuralNetwork"]["Training"]["learning_rate"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = None
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        _, world_rank = get_comm_size_and_rank()
        if int(world_rank) == 0:
            writer = SummaryWriter("./logs/" + model_with_config_name)
    else:
        writer = SummaryWriter("./logs/" + model_with_config_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + model_with_config_name + "/config.json", "w") as f:
        json.dump(config, f)

    if (
        "continue" in config["NeuralNetwork"]["Training"]
        and config["NeuralNetwork"]["Training"]["continue"] == 1
    ):

        modelstart = config["NeuralNetwork"]["Training"]["startfrom"]
        if not modelstart:
            modelstart = model_with_config_name

        state_dict = torch.load(
            f"./logs/{modelstart}/{modelstart}.pk",
            map_location="cpu",
        )
        model.load_state_dict(state_dict)

    ## START THE TRAINING
    print_distributed(
        config["Verbosity"]["level"],
        f"Starting the training of the network: \n{json.dumps(config, indent=4, sort_keys=True)}",
    )

    train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        model_with_config_name,
        config["Verbosity"]["level"]
    )

    return model
