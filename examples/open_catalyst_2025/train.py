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
import logging
import sys
from mpi4py import MPI

import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from hydragnn import cli
from hydragnn.preprocess.load_data import split_dataset

# Ensure the bundled fairchem package is importable when not installed system-wide.
dirpwd = os.path.dirname(os.path.abspath(__file__))
fairchem_path = os.path.join(dirpwd, "fairchem")
if fairchem_path not in sys.path:
    sys.path.insert(0, fairchem_path)

from oc25 import OpenCatalystDataset

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="oc25_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "OC25")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "OC25"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        trainset = OpenCatalystDataset(
            os.path.join(datadir),
            config,
            data_type="train",
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        trainset, valset1, valset2 = split_dataset(
            dataset=trainset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        valset = [*valset1, *valset2]
        testset = OpenCatalystDataset(
            os.path.join(datadir),
            config,
            data_type="val",
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        testset = testset[:]

        comm.Barrier()

        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
