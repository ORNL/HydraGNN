##############################################################################
# Copyright (c) 2025, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

# General
import os
import sys

import torch.distributed as dist

# Distributed
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

# HydraGNN
from hydragnn import cli
from hydragnn.data.pickledataset import SimplePickleDataset
from hydragnn.preprocess.load_data import split_dataset

# Lennard Jones
from LJ_data import create_dataset, LJDataset, info


def cached_dataset_is_compatible(path, fmt, config):
    if fmt == "adios":
        return os.path.exists(path)
    if not os.path.exists(path):
        return False
    try:
        sample = SimplePickleDataset(basedir=path, label="trainset")[0]
    except (FileNotFoundError, EOFError, AttributeError, IndexError):
        return False
    required_names = {
        spec["name"]
        for group in ("inputs", "outputs")
        for spec in config["Variables"][group]
    }
    if all(hasattr(sample, name) for name in required_names):
        return True
    info("Rebuilding an incompatible pre-named-variable LJ cache")
    return False


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="LJ.json", default_format="pickle")
    parser.add_argument("--mpnn_type", help="model type", type=str, default=None)
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    if args.mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = args.mpnn_type
    rank, log_name, writer = cli.setup_run(args, "LJ")
    modelname = args.modelname or "LJ"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly or not cached_dataset_is_compatible(path, args.format, config):
        datadir = os.path.join(dirpwd, "dataset", "data")
        create_dataset(datadir, config)
        total = LJDataset(datadir, config, dist=True)
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))
        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        if args.preonly:
            sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer, preload_train=True)
    dist.destroy_process_group()
