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
import logging
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.datasets import SimplePickleDataset
from hydragnn.utils.input_config_parsing import (
    get_log_name_config,
    sanitize_filename_component,
)
from hydragnn.models.create import create_model_config

from scipy.interpolate import griddata

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})


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


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


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
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    datasetname = "qm7x"

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir,
            label="trainset",
            var_config=config["Variables"],
        )
        valset = SimplePickleDataset(
            basedir=basedir,
            label="valset",
            var_config=config["Variables"],
        )
        testset = SimplePickleDataset(
            basedir=basedir,
            label="testset",
            var_config=config["Variables"],
        )
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    load_existing_model(model, modelname, path="./logs/")
    model.eval()

    variable_index = 0
    for output in config["Variables"]["outputs"]:
        output_name = output["name"]
        output_type = output["level"]
        output_dim = output["dim"]

        test_MAE = 0.0

        num_samples = len(testset)
        true_values = []
        predicted_values = []

        for data_id, data in enumerate(tqdm(testset)):
            predicted = model(data.to(get_device()))
            predicted = predicted[variable_index].flatten()
            start = data.y_loc[0][variable_index].item()
            end = data.y_loc[0][variable_index + 1].item()
            true = data.y[start:end, 0]
            test_MAE += torch.norm(predicted - true, p=1).item() / len(testset)
            predicted_values.extend(predicted.tolist())
            true_values.extend(true.tolist())

        hist2d_norm = getcolordensity(true_values, predicted_values)

        fig, ax = plt.subplots()
        plt.scatter(true_values, predicted_values, s=8, c=hist2d_norm, vmin=0, vmax=1)
        plt.clim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
        plt.colorbar()
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"{output_name}")
        plt.draw()
        plt.tight_layout()
        filename = sanitize_filename_component(output_name) + "_Scatterplot.png"
        plt.savefig(filename, dpi=400)

        print(f"Test MAE {output_name}: ", test_MAE)

        variable_index += 1
