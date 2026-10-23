##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import random
import csv

import logging
import sys

from mpi4py import MPI

import torch
import torch.distributed as dist

import hydragnn
from hydragnn import cli
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.data.descriptors_and_embeddings.smiles_utils import (
    generate_graphdata_from_smilestr,
)
from hydragnn.config.variable_schema import parse_variable_schema
from hydragnn.config import sanitize_filename_component
from hydragnn.distributed import nsplit

import numpy as np

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

csce_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def csce_datasets_load(datafile, sampling=None, seed=None, frac=[0.94, 0.02, 0.04]):
    if seed is not None:
        random.seed(seed)
    smiles_all = []
    values_all = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if (sampling is not None) and (random.random() > sampling):
                continue
            smiles_all.append(row[1])
            values_all.append([float(row[-2])])
    print("Total:", len(smiles_all), len(values_all))

    a = list(range(len(smiles_all)))
    a = random.sample(a, len(a))
    ix0, ix1, ix2 = np.split(
        a, [int(frac[0] * len(a)), int((frac[0] + frac[1]) * len(a))]
    )

    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainset = []
    valset = []
    testset = []

    for i in ix0:
        trainsmiles.append(smiles_all[i])
        trainset.append(values_all[i])

    for i in ix1:
        valsmiles.append(smiles_all[i])
        valset.append(values_all[i])

    for i in ix2:
        testsmiles.append(smiles_all[i])
        testset.append(values_all[i])

    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        np.mean(values_all),
        np.std(values_all),
    )


## Torch Dataset for CSCE CSV format
class CSCEDatasetFactory:
    def __init__(
        self, datafile, sampling=1.0, seed=43, var_config=None, norm_yflag=False
    ):
        self.var_config = (
            parse_variable_schema(var_config)
            if isinstance(var_config, dict)
            else var_config
        )

        ## Read full data
        (
            smiles_sets,
            values_sets,
            ymean_feature,
            ystd_feature,
        ) = csce_datasets_load(datafile, sampling=sampling, seed=seed)
        ymean = ymean_feature.tolist()
        ystd = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (valueset - torch.tensor(ymean)) / torch.tensor(ystd)
            self.dataset_lists.append((smileset, valueset))

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        smileset, valueset = self.dataset_lists[index]
        return (smileset, valueset)


class CSCEDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.smileset, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata_from_smilestr(
            smilestr, ytarget, csce_node_types, self.var_config
        )
        return data


def plot_mae(model, loaders, var_config, log_name, verbosity, rank):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for isub, (loader, setname) in enumerate(zip(loaders, ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_values = hydragnn.train.test(
            loader, model, verbosity
        )
        ihead = 0
        head_true = np.asarray(true_values[ihead].cpu()).squeeze()
        head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
        varname = var_config.outputs[ihead].name

        ax = axs[isub]
        error_mae = np.mean(np.abs(head_pred - head_true))
        error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
        print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)

        ax.scatter(
            head_true,
            head_pred,
            s=7,
            linewidth=0.5,
            edgecolor="b",
            facecolor="none",
        )
        minv = np.minimum(np.amin(head_pred), np.amin(head_true))
        maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
        ax.plot([minv, maxv], [minv, maxv], "r--")
        ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
        ax.text(
            minv + 0.1 * (maxv - minv),
            maxv - 0.1 * (maxv - minv),
            "MAE: {:.2f}".format(error_mae),
        )
    if rank == 0:
        filename = sanitize_filename_component(varname) + "_all.png"
        fig.savefig(os.path.join("logs", log_name, filename))
    plt.close()


if __name__ == "__main__":
    parser = cli.build_parser(
        default_inputfile="csce_gap.json", formats=cli.FORMATS + ("csv",)
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirpwd, "dataset/csce_gap_synth.csv")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    verbosity = config["Verbosity"]["level"]
    var_config = parse_variable_schema(config["Variables"])
    rank, log_name, writer = cli.setup_run(args, "csce_gap_eV_fullx")
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    if args.preonly:
        if args.format == "csv":
            raise ValueError("--preonly writes adios or pickle datasets")
        smiles_sets, values_sets, _, _ = csce_datasets_load(
            datafile, sampling=args.sampling, seed=43
        )
        info([len(x) for x in values_sets])
        dataset_lists = [[] for _ in values_sets]
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            rx = list(nsplit(range(len(smileset)), comm_size))[rank]
            info("subset range:", idataset, len(smileset), rx.start, rx.stop)
            _smileset = smileset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            info("local smileset size:", len(_smileset))
            for smilestr, ytarget in iterate_tqdm(
                zip(_smileset, _valueset), verbosity, total=len(_smileset)
            ):
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, csce_node_types, var_config
                )
                data.smiles = smilestr
                dataset_lists[idataset].append(data)

        path = cli.dataset_path(dirpwd, args.modelname or "csce_gap", args.format)
        cli.write_split_datasets(path, args.format, *dataset_lists)
        sys.exit(0)

    if args.format == "csv":
        fact = CSCEDatasetFactory(datafile, args.sampling, var_config=var_config)
        splits = [CSCEDataset(fact, label) for label in cli.SPLITS]
    else:
        path = cli.dataset_path(dirpwd, args.modelname or "csce_gap", args.format)
        splits = cli.load_split_datasets(
            path,
            args.format,
            shmem=args.shmem,
            ddstore=args.ddstore,
            ddstore_width=args.ddstore_width,
        )

    loaders = []
    model, config = cli.run_training(
        config,
        *splits,
        log_name,
        writer,
        precision=args.precision,
        everyone=args.everyone,
        variables=False,
        before_train=lambda model, optimizer, split_loaders, config: loaders.extend(
            split_loaders
        ),
    )
    if args.mae:
        plot_mae(model, loaders, var_config, log_name, verbosity, rank)

    if dist.is_initialized():
        dist.destroy_process_group()
