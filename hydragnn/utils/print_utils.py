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

from tqdm import tqdm

import torch.distributed as dist

import logging
import os


"""
Verbosity options for printing
0 - > nothing
1 -> master prints the basic
2 -> Master prints everything, progression bars included
3 -> all MPI processes print the basic
4 -> all MPI processes print the basic, , progression bars included
"""


def print_nothing(*args):
    pass


def print_master(*args):
    log(*args, rank=0)


def print_all_processes(*args):
    log(*args)


switcher = {
    0: print_nothing,
    1: print_master,
    2: print_master,
    3: print_all_processes,
    4: print_all_processes,
}


def print_distributed(verbosity_level, *args):
    print_verbose = switcher.get(verbosity_level)
    return print_verbose(*args)


def iterate_tqdm(iterator, verbosity_level, *args, **kwargs):
    if (0 == dist.get_rank() and 2 == verbosity_level) or 4 == verbosity_level:
        return tqdm(iterator, *args, **kwargs)
    else:
        return iterator


def setup_log(prefix):
    """
    Setup logging to print messages for both screen and file.
    """
    from .distributed import init_comm_size_and_rank

    world_size, world_rank = init_comm_size_and_rank()

    fmt = "%d: %%(message)s" % (world_rank)
    logFormatter = logging.Formatter(fmt)

    logger = logging.getLogger("hydragnn")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    os.makedirs("./logs/%s" % prefix, exist_ok=True)
    fname = "./logs/%s/run.log" % (prefix)
    fileHandler = logging.FileHandler(fname)

    if logger.hasHandlers():
        logger.handlers.clear()

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


def log(*args, sep=" ", rank=None):
    """
    Helper function to print/log messages.
    rank parameter is to limit which rank should print. if rank is None, all processes print.
    """
    logger = logging.getLogger("hydragnn")

    if rank is None:
        logger.info(sep.join(map(str, args)))
    else:
        from .distributed import init_comm_size_and_rank

        world_size, world_rank = init_comm_size_and_rank()
        if rank == world_rank:
            logger.info(sep.join(map(str, args)))
