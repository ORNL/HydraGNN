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
import sys
import torch
from mpi4py import MPI

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import numpy as np

from hydragnn import cli
from hydragnn.data.distdataset import DistDataset
from hydragnn.utils.print.print_utils import log
from hydragnn.distributed import nsplit

from scipy.interpolate import make_interp_spline

## FIMME
torch.backends.cudnn.enabled = False

## FIXME: Hard-coded for now. Need to find common variable names
COMMON_VARIABLE_NAMES = [
    "x",
    "atomic_numbers",
    "edge_index",
    "edge_attr",
    "energy",
    "energy_per_atom",
    "forces",
    "pos",
    "y",
]


def load_multi_datasets(args, datadir, comm):
    """Split ranks across the datasets in ``--multi_model_list`` by size."""
    from hydragnn.data.adiosdataset import AdiosDataset, adios2_open

    rank, comm_size = comm.Get_rank(), comm.Get_size()
    modellist = args.multi_model_list.split(",")
    if rank == 0:
        ndata_list = list()
        pna_deg_list = list()
        for model in modellist:
            fname = os.path.join(datadir, "%s.bp" % model)
            with adios2_open(fname, "r", MPI.COMM_SELF) as f:
                f.__next__()
                ndata = f.read_attribute("trainset/ndata").item()
                attrs = f.available_attributes()
                pna_deg = None
                if "pna_deg" in attrs:
                    pna_deg = f.read_attribute("pna_deg")
                ndata_list.append(ndata)
                pna_deg_list.append(pna_deg)
        ndata_list = np.array(ndata_list, dtype=np.float32)
        process_list = np.ceil(ndata_list / sum(ndata_list) * comm_size).astype(
            np.int32
        )
        imax = np.argmax(process_list)
        process_list[imax] = process_list[imax] - (np.sum(process_list) - comm_size)
        process_list = process_list.tolist()

        ## Merge pna_deg using interpolation
        mlen = min([len(pna_deg) for pna_deg in pna_deg_list])
        pna_deg = np.zeros(mlen)
        for deg in pna_deg_list:
            intp = make_interp_spline(np.linspace(0, 1, num=len(deg)), deg)
            pna_deg += intp(np.linspace(0, 1, num=mlen))
        pna_deg = pna_deg.astype(np.int64).tolist()
    else:
        process_list = None
        pna_deg = None
    process_list = comm.bcast(process_list, root=0)
    pna_deg = comm.bcast(pna_deg, root=0)
    assert comm_size == sum(process_list)

    colorlist = list()
    for color, n in enumerate(process_list):
        colorlist.extend([color] * n)
    mycolor = colorlist[rank]
    mymodel = modellist[mycolor]

    local_comm = comm.Split(mycolor, rank)
    local_comm_rank = local_comm.Get_rank()
    local_comm_size = local_comm.Get_size()

    fname = os.path.join(datadir, "%s.bp" % mymodel)
    trainset, valset, testset = (
        AdiosDataset(fname, label, local_comm, keys=COMMON_VARIABLE_NAMES)
        for label in cli.SPLITS
    )

    ## Set local set
    for dataset in [trainset, valset]:
        rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
        if args.num_samples is not None:
            if args.num_samples > len(rx):
                log(
                    f"WARN: requested samples are larger than what is available. Use only {len(rx)}: {dataset.label}"
                )
            rx = rx[: args.num_samples]

        dataset.setkeys(COMMON_VARIABLE_NAMES)
        dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

    rx = list(nsplit(range(len(testset)), local_comm_size))[local_comm_rank]
    num_samples = len(rx)
    if args.num_test_samples is not None:
        num_samples = args.num_test_samples
    elif args.num_samples is not None:
        num_samples = args.num_samples
    rx = rx[:num_samples]
    testset.setkeys(COMMON_VARIABLE_NAMES)
    testset.setsubset(rx[0], rx[-1] + 1, preload=True)

    if args.shmem and args.ddstore:
        raise ValueError("Cannot use both ddstore and shmem")
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width, "local": True}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)
        for dataset in (trainset, valset, testset):
            dataset.pna_deg = pna_deg
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    return trainset, valset, testset


if __name__ == "__main__":
    parser = cli.build_parser(
        default_inputfile="gfm_multitasking.json",
        preprocessing=False,
        formats=cli.FORMATS + ("multi",),
    )
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--multi_model_list", help="multidataset list", default="OC2020"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="set num samples per process for weak-scaling test",
        default=None,
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        help="set num test samples per process for weak-scaling test",
        default=None,
    )
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "GFM")

    if args.format == "multi":
        splits = load_multi_datasets(args, datadir, MPI.COMM_WORLD)
    else:
        splits = cli.load_split_datasets(
            cli.dataset_path(dirpwd, args.modelname or "GFM", args.format),
            args.format,
            shmem=args.shmem,
            ddstore=args.ddstore,
            ddstore_width=args.ddstore_width,
        )
    cli.run_training(
        config,
        *splits,
        log_name,
        writer,
        precision=args.precision,
        everyone=args.everyone,
    )
    sys.exit(0)
