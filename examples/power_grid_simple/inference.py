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

import json, os
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import pandas as pd

import torch
import torch_scatter
import numpy as np

from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device, setup_ddp
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config,
)
from hydragnn.utils.print import setup_log
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import pandapower.networks as nw

plt.rcParams.update({"font.size": 16})

# Utility functions and model class (identical to original)
def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size * percentage / 100)]

def make_dataset(dataset, n_bus):
    x_raw, y_raw = [], []
    for i in range(len(dataset)):
        x_sample, y_sample = [], []
        for n in range(n_bus):
            is_pv = 0
            is_pq = 0
            is_slack = 0
            if n == 0:
                is_slack = 1
            elif dataset[i, 4 * n + 2] == 0:
                is_pv = 1
            else:
                is_pq = 1
            x_sample.append([
                dataset[i, 4 * n + 1],
                dataset[i, 4 * n + 2],
                dataset[i, 4 * n + 3],
                dataset[i, 4 * n + 4],
                is_pv,
                is_pq,
                is_slack
            ])
            y_sample.append([
                dataset[i, 4 * n + 3],
                dataset[i, 4 * n + 4]
            ])
        x_raw.append(x_sample)
        y_raw.append(y_sample)
    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    return x_raw, y_raw

def normalize_dataset(x, y):
    x_mean, x_std = torch.mean(x, 0), torch.std(x, 0)
    y_mean, y_std = torch.mean(y, 0), torch.std(y, 0)
    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1
    x_norm = (x - x_mean) / x_std
    x_norm[:, :, 4] = x[:, :, 4]
    y_norm = (y - y_mean) / y_std
    return x_norm, y_norm, x_mean, y_mean, x_std, y_std

def denormalize_output(y_norm, y_mean, y_std):

    return y_norm * y_std + y_mean

def get_log_name_config(config):
    return (
        config["NeuralNetwork"]["Architecture"]["mpnn_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
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


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm


if __name__ == "__main__":

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--log", help="log name")
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="power_grid.json"
    )
    parser.add_argument("--modelname", help="model name")
    group = parser.add_mutually_exclusive_group()
    args = parser.parse_args()

    modelname = "PowerGrid" if args.modelname is None else args.modelname

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")

    ##################################################################################################################
    input_filename = os.path.join(dirpwd, f'./logs/{modelname}/config.json')
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    setup_log(get_log_name_config(config))

    comm.Barrier()
    
    bus_system = "118Bus"  # Change as needed
    n_bus = 14
    if bus_system == "14Bus":
        net = nw.case14()
        dataset1 = pd.read_excel("./dataset/14Bus/PF_Dataset_1.xlsx").values
        dataset2 = pd.read_excel("./dataset/14Bus/PF_Dataset_2.xlsx").values
    elif bus_system == "30Bus":
        n_bus = 30
        net = nw.case30()
        dataset1 = pd.read_excel("./dataset/30Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("./dataset/30Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "57Bus":
        n_bus = 57
        net = nw.case57()
        dataset1 = pd.read_excel("./dataset/57Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("./dataset/57Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "118Bus":
        n_bus = 118
        net = nw.case118()
        dataset1 = pd.read_excel("./dataset/118Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("./dataset/118Bus/PF_Dataset_2_10000.xlsx").values
    else:
        raise ValueError("Invalid bus system.")

    train_percentage = 80
    val_percentage = 20 if bus_system != "14Bus" else 100
    train_dataset = slice_dataset(dataset1, train_percentage)
    val_dataset = slice_dataset(dataset2, val_percentage)
    x_raw_train, y_raw_train = make_dataset(train_dataset, n_bus)
    x_raw_val, y_raw_val = make_dataset(val_dataset, n_bus)
    x_norm_train, y_norm_train, x_train_mean, y_train_mean, x_train_std, y_train_std = normalize_dataset(x_raw_train, y_raw_train)
    x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val)
    from_buses = net.line['from_bus'].values
    to_buses = net.line['to_bus'].values
    edge_index = torch.tensor([list(from_buses) + list(to_buses), list(to_buses) + list(from_buses)], dtype=torch.long)
    train_data_list = []
    output_dims = [2]  # from config: output_dim
    num_nodes = n_bus
    y_loc_indices = [0]
    for dim in output_dims:
        y_loc_indices.append(y_loc_indices[-1] + dim * num_nodes)
    y_loc_tensor = torch.tensor([y_loc_indices])

    train_data_list = []
    for x, y in zip(x_norm_train, y_norm_train):
        y_flat = y.flatten()
        data = Data(x=x, y=y_flat, edge_index=edge_index)
        data.y_loc = y_loc_tensor.clone()
        train_data_list.append(data)

    val_data_list = []
    for x, y in zip(x_norm_val, y_norm_val):
        y_flat = y.flatten()
        data = Data(x=x, y=y_flat, edge_index=edge_index)
        data.y_loc = y_loc_tensor.clone()
        val_data_list.append(data)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train_data_list, val_data_list, val_data_list, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    load_existing_model(model, modelname, path="./logs/")
    model.eval()

    y_val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            y_val_pred = model(batch)
            y_val_pred = y_val_pred[0].view(-1, n_bus, 2)
            y_val_predictions.append(y_val_pred)
    y_val_predictions = torch.cat(y_val_predictions, dim=0)
    y_val_targets = torch.cat([batch.y.view(-1, n_bus, 2) for batch in val_loader], dim=0)

    # Denormalize outputs
    y_val_pred_denorm = denormalize_output(y_val_predictions, y_val_mean, y_val_std)
    y_val_targets_denorm = denormalize_output(y_val_targets, y_val_mean, y_val_std)

    # Parity plots for V and delta
    for i, label in enumerate(["Voltage Magnitude (V)", "Voltage Angle (delta)"]):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val_targets_denorm[:, :, i].flatten(), y_val_pred_denorm[:, :, i].flatten(), alpha=0.5)
        plt.plot([y_val_targets_denorm[:, :, i].min(), y_val_targets_denorm[:, :, i].max()],
                 [y_val_targets_denorm[:, :, i].min(), y_val_targets_denorm[:, :, i].max()],
                 color='red', linestyle='--', label='Ideal')
        plt.xlabel(f'True {label}')
        plt.ylabel(f'Predicted {label}')
        plt.title(f'Parity Plot for {label} (Validation Set)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'parity_plot_{label.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
        print(f"Saved parity plot for {label}.")

