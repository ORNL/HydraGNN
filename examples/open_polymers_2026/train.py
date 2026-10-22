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

###############################################################################################
import importlib.util

# Path to the outer directory that contains 'ase_db_backends'
local_path = "/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN/examples/open_polymers_2026/fairchem/core/datasets"

# Ensure this path is searched first
sys.path.insert(0, local_path)

# Now explicitly load your local ase_db_backends package
package_init = os.path.join(
    local_path, "ase_db_backends", "ase_db_backends", "__init__.py"
)
spec = importlib.util.spec_from_file_location("ase_db_backends", package_init)
mod = importlib.util.module_from_spec(spec)
sys.modules["ase_db_backends"] = mod
spec.loader.exec_module(mod)

###############################################################################################

import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


from hydragnn import cli
from hydragnn.preprocess.load_data import split_dataset

from opoly26 import OPoly2026

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="opoly26_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "OPoly26")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "OPoly26"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        trainset = OPoly2026(
            os.path.join(datadir),
            config,
            data_type="train",
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset1, valset2 = split_dataset(
            dataset=trainset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        valset = [*valset1, *valset2]
        testset = OPoly2026(
            os.path.join(datadir),
            config,
            data_type="val",
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## Need as a list
        testset = testset[:]

        comm.Barrier()

        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
