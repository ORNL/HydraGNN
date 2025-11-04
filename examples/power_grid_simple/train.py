import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import random

import pandas as pd

import numpy as np

import torch
import torch.distributed as dist
from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model

from hydragnn.utils.distributed import nsplit

from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.graph_samples_checks_and_updates import RadiusGraph

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils.print.print_utils import log

from hydragnn.utils.distributed import nsplit

import pandapower.networks as nw

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

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


if __name__ == "__main__":
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

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)

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

    train_percentage = 100
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

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "PowerGrid" if args.log is None else args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "PowerGrid" if args.modelname is None else args.modelname

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)


    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

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
        create_plots=False,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)