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

from itertools import chain

import torch
import torch_scatter
import numpy as np

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device, setup_ddp, nsplit
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

plt.rcParams.update({"font.size": 16})


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


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


def getcolordensity(xdata, ydata, nbin=100):
    x = np.asarray(xdata, dtype=float).ravel()
    y = np.asarray(ydata, dtype=float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    # 2D histogram
    hist2d, xedges, yedges = np.histogram2d(x=x, y=y, bins=[nbin, nbin])

    # Normalize safely
    hmax = hist2d.max()
    hist2d_norm = hist2d / hmax if hmax > 0 else np.zeros_like(hist2d, dtype=float)

    # For each (x,y), find its bin index
    ix = np.digitize(x, xedges) - 1
    iy = np.digitize(y, yedges) - 1

    # Clamp to [0, nbin-1] so points on the right/top edge map to the last bin
    ix = np.clip(ix, 0, nbin - 1)
    iy = np.clip(iy, 0, nbin - 1)

    # Color density per point = normalized count in that bin
    return hist2d_norm[ix, iy]


if __name__ == "__main__":

    modelname = "qm7x"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/qm7x/config.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios gan_dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="adios")

    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")

    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = [
        "atomic_number",
        "coordinates",
        "forces",
        "hCHG",
        "hVDIP",
        "hRAT",
    ]
    node_feature_dims = [1, 3, 3, 1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/QM7-X")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    # Transformation to create positional and structural laplacian encoders
    """
    graphgps_transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )
    """
    graphgps_transform = None

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

    log_name = "qm7x"
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(
            os.path.dirname(__file__), "./dataset/%s-v2.bp" % modelname
        )
        # trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        # valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    rx = list(nsplit(range(len(testset)), comm_size))[rank]
    testset.setsubset(rx.start, rx.stop)

    model = torch.nn.parallel.DistributedDataParallel(model)

    load_existing_model(model, modelname, path="./logs/")
    model.eval()

    num_samples = len(testset)
    energy_true_list = []
    energy_pred_list = []
    forces_true_list = []
    forces_pred_list = []

    for data_id, data in enumerate(tqdm(testset)):
        data.pos.requires_grad = True
        node_energy_pred = model(data.to(get_device()))[
            0
        ]  # Note that this is sensitive to energy and forces prediction being single-task (current requirement)
        energy_pred = torch.sum(node_energy_pred, dim=0).float()
        grads_energy = torch.autograd.grad(
            outputs=energy_pred,
            inputs=data.pos,
            grad_outputs=torch.ones_like(energy_pred),
            retain_graph=False,
            create_graph=True,
        )[0]
        energy_pred_list.extend(energy_pred.tolist())
        energy_true_list.extend(data.energy.tolist())
        forces_pred_list.extend((-grads_energy).flatten().tolist())
        forces_true_list.extend(data.forces.flatten().tolist())

    # `MPI_Gather()` is called by all processes
    # Gather lists of lists to rank 0
    all_energy_pred = comm.gather(energy_pred_list, root=0)
    all_energy_true = comm.gather(energy_true_list, root=0)
    all_forces_pred = comm.gather(forces_pred_list, root=0)
    all_forces_true = comm.gather(forces_true_list, root=0)

    if rank == 0:
        # Flatten if you want one big list
        energy_pred_list_global = list(chain.from_iterable(all_energy_pred))
        energy_true_list_global = list(chain.from_iterable(all_energy_true))
        forces_pred_list_global = list(chain.from_iterable(all_forces_pred))
        forces_true_list_global = list(chain.from_iterable(all_forces_true))

    comm.Barrier()

    if rank == 0:
        # Show R2 Metrics
        print(
            f"R2 energy: ",
            r2_score(
                np.array(energy_true_list_global), np.array(energy_pred_list_global)
            ),
        )
        print(
            f"R2 forces: ",
            r2_score(
                np.array(forces_true_list_global), np.array(forces_pred_list_global)
            ),
        )

        hist2d_norm = getcolordensity(energy_true_list_global, energy_pred_list_global)

        fig, ax = plt.subplots()
        plt.scatter(
            energy_true_list_global,
            energy_pred_list_global,
            s=8,
            c=hist2d_norm,
            vmin=0,
            vmax=1,
        )
        plt.clim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
        plt.colorbar()
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"energy")
        plt.draw()
        plt.tight_layout()
        plt.savefig(f"./energy_Scatterplot" + ".png", dpi=400)

        hist2d_norm = getcolordensity(forces_pred_list_global, forces_true_list_global)
        fig, ax = plt.subplots()
        plt.scatter(
            forces_pred_list_global,
            forces_true_list_global,
            s=8,
            c=hist2d_norm,
            vmin=0,
            vmax=1,
        )
        plt.clim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
        plt.colorbar()
        plt.xlabel("Predicted Values")
        plt.ylabel("True Values")
        plt.title("Forces")
        plt.draw()
        plt.tight_layout()
        plt.savefig(f"./Forces_Scatterplot" + ".png", dpi=400)
