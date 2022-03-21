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
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn import run_training
from hydragnn.preprocess import dataset_loading_and_splitting
from hydragnn.models import create_model_config
from hydragnn.train import train_validate_test
from hydragnn.utils import (
    setup_ddp,
    get_comm_size_and_rank,
    get_distributed_model,
    update_config,
    get_log_name_config,
    load_existing_model,
    iterate_tqdm,
    get_summary_writer,
    save_model,
)

from pi3nn.Optimizations import CL_boundary_optimizer


def run_uncertainty(
    config_file_mean, config_file_up_down, retrain_mean=False, retrain_up_down=False
):
    """
    Compute prediction intervals with PI3NN.
    """

    out_name = "uq_"
    mean_name = out_name + "mean"
    if retrain_mean:
        run_training(config_file_mean, mean_name)

    config = {}
    with open(config_file_up_down, "r") as f:
        config = json.load(f)

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp()
    verbosity = config["Verbosity"]["level"]

    #### LOAD THE ORIGINAL DATA LOADERS AND THE TRAINED MEAN MODEL
    ## Question: Should we normalize/denormalize any input/output?
    mean_loaders = dataset_loading_and_splitting(
        config=config,
    )
    config = update_config(config, mean_loaders[0], mean_loaders[1], mean_loaders[2])

    mean_model = load_model(config, mean_loaders[0].dataset, mean_name)

    load_existing_model(mean_model, mean_name)

    mean_model = get_distributed_model(mean_model, verbosity)

    #### CREATE THE DATASET LOADERS
    up_loaders, down_loaders = create_loaders(mean_loaders, mean_model, config)

    #### LOAD OR TRAIN UP & DOWN MODELS
    up_name = out_name + "up"
    down_name = out_name + "down"
    if retrain_up_down:
        # config["NeuralNetwork"]["Architecture"]["hidden_dim"] = 10
        model_up = train_model(up_loaders, up_name, config)
        save_model(model_up, up_name)
        model_down = train_model(down_loaders, down_name, config)
        save_model(model_down, down_name)
    else:
        model_up = load_model(config, up_loaders[0].dataset, up_name)
        model_down = load_model(config, down_loaders[0].dataset, down_name)

    #### COMPUTE ALL 3 PREDICTIONS ON TRAINING DATA
    pred_mean, pred_up, pred_down, y = compute_predictions(
        mean_loaders[0], (mean_model, model_up, model_down), config
    )

    #### COMPUTE PREDICTION INTERVAL BOUNDS
    yx = y.detach().numpy()
    pred_mean_y = pred_mean.detach()
    pred_up_y = pred_up.detach()
    pred_down_y = pred_down.detach()
    boundaryOptimizer = CL_boundary_optimizer(
        yx,
        pred_mean_y,
        pred_up_y,
        pred_down_y,
        c_up0_ini=0.0,
        c_up1_ini=100000.0,
        c_down0_ini=0.0,
        c_down1_ini=100000.0,
        max_iter=100,
    )

    num_samples = len(mean_loaders[0].dataset)
    num_outlier_list = [int(num_samples * (1 - x) / 2) for x in [0.9]]
    c_up = [
        boundaryOptimizer.optimize_up(outliers=x, verbose=0) for x in num_outlier_list
    ]
    c_down = [
        boundaryOptimizer.optimize_down(outliers=x, verbose=0) for x in num_outlier_list
    ]
    print(c_up, c_down)

    ### COMPUTE PREDICTION INTERVAL COVERAGE PROBABILITY
    for loader in mean_loaders:
        compute_picp(
            pred_mean_y,
            pred_up_y,
            pred_down_y,
            y.detach(),
            c_up[0],
            c_down[0],
            num_samples,
        )

    plot_uq(pred_mean, pred_up, pred_down, y, c_up[0], c_down[0])


def plot_uq(pred_mean, pred_up, pred_down, y, c_up, c_down):
    """
    Plot mean, upper, and lower predictions.
    """
    c = ["b", "gray", "r"]
    fig, ax = plt.subplots(1, 1)
    yx = y.detach().numpy()
    pred_mean_y = pred_mean.detach()
    pred_up_y = pred_up.detach()
    pred_down_y = pred_down.detach()
    ax.scatter(yx, pred_mean_y, edgecolor=c[0], marker="o", facecolor="none")
    # ax.scatter(yx, pred_mean_y+pred_up_y, edgecolor=c[1], marker="o", facecolor="none")
    # ax.scatter(yx, pred_mean_y-pred_down_y, edgecolor=c[2], marker="o", facecolor="none")

    ax.scatter(
        yx.squeeze() + 0.001,
        (pred_mean_y + c_up * pred_up_y),
        edgecolor="#9999ff",
        marker="o",
        facecolor="none",
    )
    ax.scatter(
        yx.squeeze() - 0.001,
        (pred_mean_y - c_down * pred_down_y),
        edgecolor="#000066",
        marker="o",
        facecolor="none",
    )
    plt.show()


def load_model(config, dataset, name):

    model = create_model_config(
        config=config["NeuralNetwork"]["Architecture"],
        verbosity=config["Verbosity"]["level"],
    )

    # model_name = model.__str__()
    output_name = name
    load_existing_model(model, output_name)
    return model


def compute_picp(pred_mean_y, pred_up_y, pred_down_y, y, c_up, c_down, num_samples):
    """
    Compute prediction interval coverage probabilty - fraction of data within the bounds.
    """
    covered = 0.0

    up = pred_mean_y + c_up * pred_up_y
    down = pred_mean_y - c_down * pred_down_y
    covered += ((down <= y) & (y <= up)).sum()
    print(
        "COVERED IN PI:",
        covered,
        "IN A TOTAL OF",
        num_samples,
        "PCIP:",
        covered / (num_samples),
    )


def compute_predictions(loader, models, config):
    """
    Compute predictions on the given dataset using mean/up/down trained models.
    """
    for m in models:
        m.eval()

    device = next(models[0].parameters()).device
    verbosity = config["Verbosity"]["level"]

    pred = [None for i in range(len(models))]
    y = None
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
        for i, m in enumerate(models):
            result = models[i](data)[0]
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


## NOTE: with MPI, the total dataset (before DDP splitting) should be used to create up and down, then re-split using DDP.
@torch.no_grad()
def create_loaders(loaders, model, config):
    """
    Create the up and down datasets by splitting on mean model predictions.
    """
    device = next(model.parameters()).device
    verbosity = config["Verbosity"]["level"]
    batch_size = config["NeuralNetwork"]["Training"]["batch_size"]

    model.eval()

    rank = 0
    if dist.is_initialized():
        _, rank = get_comm_size_and_rank()

    up_loaders = []
    down_loaders = []
    for l, loader in enumerate(loaders):
        up = []
        down = []
        nEq = 0
        for data in iterate_tqdm(loader, verbosity):
            data = data.to(device)
            diff = model(data)[0] - data.y

            data_copy = data.cpu().clone()
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
            print(
                "dataset:", l, "up", lengths[0], "down", lengths[1], "equal", lengths[2]
            )

        ## The code below creates the loaders on the LOCAL up and down samples, which is probably wrong.
        ## We should add a mechanism (either a gather-type communication, or a file write and read) to
        ## make the GLOBAL datasets visible in each GPU.
        if dist.is_initialized():
            up_sampler = torch.utils.data.distributed.DistributedSampler(up)
            down_sampler = torch.utils.data.distributed.DistributedSampler(down)

            up_loader = DataLoader(
                up, batch_size=batch_size, shuffle=False, sampler=up_sampler
            )
            down_loader = DataLoader(
                down, batch_size=batch_size, shuffle=False, sampler=down_sampler
            )
        else:
            up_loader = DataLoader(up, batch_size=batch_size, shuffle=True)
            down_loader = DataLoader(down, batch_size=batch_size, shuffle=True)

        up_loaders.append(up_loader)
        down_loaders.append(down_loader)

    return up_loaders, down_loaders


def train_model(loaders, output_name, config):
    """
    Train a model on the upper or lower dataset.
    """
    new_model = create_model_config(
        config=config["NeuralNetwork"]["Architecture"],
        verbosity=config["Verbosity"]["level"],
    )

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = get_summary_writer(output_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + output_name + "/config.json", "w") as f:
        json.dump(config, f)

    train_validate_test(
        new_model,
        optimizer,
        loaders[0],
        loaders[1],
        loaders[2],
        writer,
        scheduler,
        config["NeuralNetwork"],
        output_name,
        config["Verbosity"]["level"],
    )

    return new_model
