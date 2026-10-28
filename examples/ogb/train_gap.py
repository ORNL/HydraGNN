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
import os
import random
import pandas
import csv

import logging
import sys
from tqdm import tqdm
import torch
from mpi4py import MPI
import math

import hydragnn
from hydragnn import cli
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.data.descriptors_and_embeddings.smiles_utils import (
    generate_graphdata_from_smilestr,
)
from hydragnn.config.config_utils import sanitize_filename_component
from hydragnn.data import AbstractBaseDataset
from hydragnn.distributed import nsplit

import numpy as np

import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

ogb_node_types = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Si": 6,
    "P": 7,
    "S": 8,
    "Cl": 9,
    "Ca": 10,
    "Ge": 11,
    "As": 12,
    "Se": 13,
    "Br": 14,
    "I": 15,
    "Mg": 16,
    "Ti": 17,
    "Ga": 18,
    "Zn": 19,
    "Ar": 20,
    "Be": 21,
    "He": 22,
    "Al": 23,
    "Kr": 24,
    "V": 25,
    "Na": 26,
    "Li": 27,
    "Cu": 28,
    "Ne": 29,
    "Ni": 30,
}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def smiles_to_graph(datadir, files_list, var_config, comm):
    subset = []
    for filename in files_list:
        df = pandas.read_csv(os.path.join(datadir, filename))
        rx = list(nsplit(range(len(df)), comm.Get_size()))[comm.Get_rank()]

        for smile_id in tqdm(range(len(df))[rx.start : rx.stop]):
            dfrow = df.iloc[smile_id]
            if math.isnan(float(dfrow.iloc[-1])):
                continue

            smilestr = dfrow.iloc[0]
            ytarget = (
                torch.tensor(float(dfrow.iloc[-1]))
                .unsqueeze(0)
                .unsqueeze(1)
                .to(torch.float32)
            )  # HL gap

            try:
                data = generate_graphdata_from_smilestr(
                    smilestr,
                    ytarget,
                    ogb_node_types,
                    var_config,
                )
                subset.append(data)
            except KeyError:
                print("KeyError: ", smilestr)
                continue

    return subset


class OGBDataset(AbstractBaseDataset):
    """Rank-local OGB samples built from every CSV file in ``dirpath``."""

    def __init__(self, dirpath, var_config, comm):
        super().__init__()
        if not os.path.isdir(dirpath):
            raise ValueError("OGBDataset takes dirpath as directory")
        files = sorted(x for x in os.listdir(dirpath) if x.endswith("csv"))
        self.dataset.extend(smiles_to_graph(dirpath, files, var_config, comm))

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


def ogb_datasets_load(datafile, sampling=None, seed=None):
    if seed is not None:
        random.seed(seed)
    trainset = []
    valset = []
    testset = []
    trainsmiles = []
    valsmiles = []
    testsmiles = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if (sampling is not None) and (random.random() > sampling):
                continue
            if row[1] == "train":
                trainsmiles.append(row[0])
                trainset.append([float(row[-1])])
            elif row[1] == "val":
                valsmiles.append(row[0])
                valset.append([float(row[-1])])
            elif row[1] == "test":
                testsmiles.append(row[0])
                testset.append([float(row[-1])])
            else:
                print("unknown file name: ", row[0])
                sys.exit(0)
    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
    )


## Torch Dataset for CSCE CSV format
class OGBRawDatasetFactory:
    def __init__(self, datafile, var_config, sampling=1.0, seed=43, norm_yflag=False):
        self.var_config = var_config

        ## Read full data
        smiles_sets, values_sets = ogb_datasets_load(
            datafile, sampling=sampling, seed=seed
        )
        if norm_yflag:
            ymean = var_config["ymean"]
            ystd = var_config["ystd"]

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


class OGBRawDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.smileset, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata_from_smilestr(
            smilestr,
            ytarget,
            ogb_node_types,
            self.var_config,
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
        varname = var_config["outputs"][ihead]["name"]

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
        default_inputfile="ogb_gap.json", formats=cli.FORMATS + ("csv",)
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use Deepspeed")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    verbosity = config["Verbosity"]["level"]
    var_config = config["Variables"]
    default_log = os.path.splitext(os.path.basename(args.inputfile))[0]
    rank, log_name, writer = cli.setup_run(
        args, default_log, use_deepspeed=args.use_deepspeed
    )
    comm = MPI.COMM_WORLD
    path = (
        cli.dataset_path(dirpwd, args.modelname or default_log, args.format)
        if args.format != "csv"
        else None
    )

    if args.preonly:
        if args.format == "csv":
            raise ValueError("--preonly writes adios or pickle datasets")
        total = OGBDataset(datadir, var_config, comm)
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))
        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    if args.format == "csv":
        fname = os.path.join(datadir, "pcqm4m_gap.csv")
        fact = OGBRawDatasetFactory(
            fname, var_config=var_config, sampling=args.sampling
        )
        splits = [OGBRawDataset(fact, label) for label in cli.SPLITS]
    else:
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
        use_deepspeed=args.use_deepspeed,
    )
    if args.mae:
        plot_mae(model, loaders, var_config, log_name, verbosity, rank)

    if dist.is_initialized():
        dist.destroy_process_group()
