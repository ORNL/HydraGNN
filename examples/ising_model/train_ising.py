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
import torch
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import sys
from mpi4py import MPI

import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from hydragnn import cli
from hydragnn.config.config_utils import get_log_name_config
from hydragnn.preprocess.load_data import split_dataset
from ising_preprocess import prepare_ising_dataset

if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="ising_model.json")
    parser.add_argument(
        "--natom", type=int, default=3, help="number_atoms_per_dimension"
    )
    parser.add_argument(
        "--cutoff", type=int, default=10, help="configurational_histogram_cutoff"
    )
    parser.add_argument("--seed", type=int, default=43, help="seed")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, get_log_name_config(config))
    comm = MPI.COMM_WORLD

    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    modelname = args.modelname or "ising_model_%d_%d" % (args.natom, args.cutoff)
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly or not os.path.exists(path):
        preprocessing_error = None
        if rank == 0:
            try:
                total = prepare_ising_dataset(
                    args.natom,
                    args.cutoff,
                    config,
                    seed=args.seed,
                    sampling=args.sampling,
                )
                trainset, valset, testset = split_dataset(
                    total,
                    config["NeuralNetwork"]["Training"]["perc_train"],
                    config["Dataset"].get("compositional_stratified_splitting", False),
                )
                cli.write_split_datasets(
                    path, args.format, trainset, valset, testset, comm=MPI.COMM_SELF
                )
            except Exception as error:
                preprocessing_error = f"{type(error).__name__}: {error}"
        preprocessing_error = comm.bcast(preprocessing_error, root=0)
        if preprocessing_error is not None:
            raise RuntimeError(f"Ising preprocessing failed: {preprocessing_error}")
        comm.Barrier()
    if args.preonly:
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    if dist.is_initialized():
        dist.destroy_process_group()
