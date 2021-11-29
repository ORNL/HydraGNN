##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os

import torch
import torch.distributed as dist

from .distributed import get_comm_size_and_rank
from .print_utils import print_distributed


def get_device_list():

    available_gpus = [i for i in range(torch.cuda.device_count())]

    return available_gpus


def get_device(use_gpu=True, rank_per_model=1, verbosity_level=0):

    available_gpus = get_device_list()
    if not use_gpu or not available_gpus:
        print_distributed(verbosity_level, "Using CPU")
        return "cpu", torch.device("cpu")

    world_size, world_rank = get_comm_size_and_rank()
    if rank_per_model != 1:
        raise ValueError("Exactly 1 rank per device currently supported")

    print_distributed(verbosity_level, "Using GPU")
    ## We need to ge a local rank if there are multiple GPUs available.
    localrank = 0
    if torch.cuda.device_count() > 1:
        if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
            ## Summit
            localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        elif os.getenv("SLURM_LOCALID"):
            ## CADES
            localrank = int(os.environ["SLURM_LOCALID"])

        if localrank >= torch.cuda.device_count():
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.cuda.device_count())
            )

    device_name = "cuda:" + str(localrank)

    return device_name, torch.device(device_name)


def get_distributed_model(model, verbosity=0):
    device_name, device = get_device(verbosity)
    if dist.is_initialized():
        if device_name == "cpu":
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )
    return model
