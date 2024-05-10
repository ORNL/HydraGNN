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
import re

import torch
import torch.distributed as dist

from .print_utils import print_distributed

import psutil
import socket
from datetime import timedelta
import time
import subprocess


def find_ifname(myaddr):
    """
    Find socket ifname for a given ip adress. This is for "GLOO" ddp setup.
    Usage example:
        find_ifname("127.0.0.1") will return a network interface name, such as "lo". "lo0", etc.
    """
    ipaddr = socket.gethostbyname(myaddr)
    ifname = None
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address == ipaddr:
                ifname = nic
                break
        if ifname is not None:
            break

    return ifname


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)

    return nlist


def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        ## CADES
        world_size = int(os.environ["SLURM_NPROCS"])
        world_rank = int(os.environ["SLURM_PROCID"])

    ## Fall back to default
    if world_size is None:
        world_size = 1

    return int(world_size), int(world_rank)


def get_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
    else:
        world_size = 1

    return int(world_size), int(world_rank)


def setup_ddp():
    """ "Initialize DDP"""

    if dist.is_initialized():
        world_size, world_rank = init_comm_size_and_rank()
        return world_size, world_rank

    if os.getenv("HYDRAGNN_BACKEND") is not None:
        backend = os.environ["HYDRAGNN_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = "8889"

    if os.getenv("HYDRAGNN_MASTER_ADDR") is not None:
        master_addr = os.environ["HYDRAGNN_MASTER_ADDR"]
    elif os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]

    try:
        if backend in ["nccl", "gloo"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)

        if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
            ifname = find_ifname(master_addr)
            if ifname is not None:
                os.environ["GLOO_SOCKET_IFNAME"] = ifname

        if world_rank == 0:
            print(
                "Distributed data parallel: %s master at %s:%s"
                % (backend, master_addr, master_port),
            )

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend, init_method="env://", timeout=timedelta(seconds=1800)
            )

    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return world_size, world_rank


def get_device_list():

    available_gpus = [i for i in range(torch.cuda.device_count())]

    return available_gpus


def get_device_name(use_gpu=True, rank_per_model=1, verbosity_level=0):

    available_gpus = get_device_list()
    if not use_gpu or not available_gpus:
        print_distributed(verbosity_level, "Using CPU")
        return "cpu"

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

    return device_name


def get_device_from_name(name: str):

    return torch.device(name)


def get_device(use_gpu=True, rank_per_model=1, verbosity_level=0):

    name = get_device_name(use_gpu, rank_per_model, verbosity_level)
    return get_device_from_name(name)


def is_model_distributed(model):
    return isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)


def get_distributed_model(model, verbosity=0, sync_batch_norm=False):
    device_name = get_device_name(verbosity_level=verbosity)

    if dist.is_initialized():
        if device_name == "cpu":
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if sync_batch_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            device = get_device_from_name(device_name)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )
    return model


def print_peak_memory(verbosity_level, prefix):
    # FIXME: this will have to change when the code can run on AMD gpus
    if torch.cuda.is_available():
        device = get_device()
        print_distributed(
            verbosity_level,
            f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ",
        )


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def comm_reduce(x, op):
    """
    All-reduce with numpy array
    """
    tx = torch.tensor(x, requires_grad=True).to(get_device())
    torch.distributed.all_reduce(tx, op=op)
    y = tx.detach().cpu().numpy()
    return y


## For early stop
def timedelta_parse(text):
    """
    Convert input string to timedelta.
    format: [[[d-]h:]m:]s
    """
    tokens = text.replace("-", ":").split(":")
    return timedelta(
        **{
            key: float(val)
            for val, key in zip(tokens[::-1], ("seconds", "minutes", "hours", "days"))
        }
    )


def check_remaining(t0):
    ## Early stop
    world_size, world_rank = get_comm_size_and_rank()
    jobid = os.getenv("SLURM_JOB_ID", None)
    should_stop = False
    device = get_device()
    if jobid is not None:
        if world_rank == 0:
            try:
                cmd = f"squeue -h -j {jobid} -o %L"
                proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
                timestr = proc.stdout.decode("utf-8").strip()
                left = timedelta_parse(timestr).total_seconds()
                esitmated = time.time() - t0
                should_stop = torch.tensor(left < esitmated, dtype=torch.bool).to(device)
                print("should_stop:", left, esitmated, should_stop.item())
            except:
                should_stop = torch.tensor(False, dtype=torch.bool).to(device)
        else:
            should_stop = torch.tensor(False, dtype=torch.bool).to(device)

        dist.broadcast(should_stop, src=0)
        should_stop = should_stop.item()
    return should_stop
