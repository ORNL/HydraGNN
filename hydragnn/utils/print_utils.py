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
from pathlib import Path
from datetime import datetime


def print_nothing(*args, **kwargs):
    pass


def print_master(*args, **kwargs):

    if not dist.is_initialized():
        info(*args, **kwargs)

    else:
        world_rank = dist.get_rank()
        if 0 == world_rank:
            info(*args, **kwargs)


def print_all_processes(*args, **kwargs):

    info(*args, **kwargs)


"""
Verbosity options for printing
0 - > nothing
1 -> master prints the basic
2 -> Master prints everything, progression bars included
3 -> all MPI processes print the basic
4 -> all MPI processes print the basic, , progression bars included
"""

switcher = {
    0: print_nothing,
    1: print_master,
    2: print_master,
    3: print_all_processes,
    4: print_all_processes,
}


def print_distributed(verbosity_level, *args, **kwargs):

    print_verbose = switcher.get(verbosity_level)
    return print_verbose(*args, **kwargs)


def iterate_tqdm(iterator, verbosity_level):
    if (0 == dist.get_rank() and 2 == verbosity_level) or 4 == verbosity_level:
        return tqdm(iterator)
    else:
        return iterator


"""
Setup logging to print messages for both screen and file.
"""


def setup_log(prefix, rank):
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    fmt = "[%d:%%(levelname)s] %%(message)s" % (rank)
    handlers = [logging.StreamHandler()]
    Path("./logs/%s" % prefix).mkdir(parents=True, exist_ok=True)
    # suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = "./logs/%s/run.log" % (prefix)
    handlers.append(logging.FileHandler(fname, delay=True))

    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=handlers)


"""
uage: info (str1, str2, ...). Use just like print. "[rank:INFO]" will be prefixed.
"""


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


"""
uage: log (str1, str2, ...). Use just like print. "[rank:DEBUG]" will be prefixed.
"""


def log(*args, logtype="debug", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def debug(*args, logtype="debug", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))
