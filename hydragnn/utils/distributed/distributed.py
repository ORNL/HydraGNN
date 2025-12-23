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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

from hydragnn.utils.print.print_utils import print_distributed

import psutil
import socket
from datetime import timedelta
import time
import subprocess

deepspeed_available = True
try:
    import deepspeed
except:
    deepspeed_available = False


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
    else:
        from mpi4py import MPI

        world_size = MPI.COMM_WORLD.Get_size()
        world_rank = MPI.COMM_WORLD.Get_rank()

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


def setup_ddp(use_deepspeed=False):
    """ "Initialize DDP"""

    if dist.is_initialized():
        world_size, world_rank = init_comm_size_and_rank()
        return world_size, world_rank

    if os.getenv("HYDRAGNN_BACKEND") is not None:
        backend = os.environ["HYDRAGNN_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        backend = "ccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = os.getenv("HYDRAGNN_MASTER_PORT", "8889")

    if os.getenv("HYDRAGNN_MASTER_ADDR") is not None:
        master_addr = os.environ["HYDRAGNN_MASTER_ADDR"]
    elif os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_STEP_NODELIST") is not None:
        ## The following is CADES/Frontier/Perlmutter specific with job steps
        master_addr = parse_slurm_nodelist(os.environ["SLURM_STEP_NODELIST"])[0]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]
    elif os.getenv("PBS_O_HOST") is not None:
        if os.environ["PBS_O_HOST"][-19:] == "aurora.alcf.anl.gov":
            from mpi4py import MPI
            import oneccl_bindings_for_pytorch as torch_ccl

            RANK = MPI.COMM_WORLD.Get_rank()
            MASTER_ADDR = socket.gethostname() if RANK == 0 else None
            MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
            master_addr = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
        else:
            ## The following is CADES specific
            master_addr = parse_slurm_nodelist(os.environ["PBS_O_HOST"])[0]

    try:
        if backend in ["nccl", "gloo", "ccl"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)
            ## Setting LOCAL_RANK complicts with DeviceMesh when using the srun "--gpus-per-task=1" option
            # os.environ["LOCAL_RANK"] = str(get_local_rank())

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
            if use_deepspeed:
                assert deepspeed_available, "deepspeed package not installed"
                deepspeed.init_distributed(
                    dist_backend=backend,
                    init_method="env://",
                    timeout=timedelta(seconds=1800),
                )
            else:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    timeout=timedelta(seconds=1800),
                )

    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return world_size, world_rank


def get_device_list():
    # [MODIFIED for Intel XPU]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return [i for i in range(torch.xpu.device_count())]
    elif torch.cuda.is_available():
        return [i for i in range(torch.cuda.device_count())]
    else:
        return []


def get_device_name(use_gpu=True, rank_per_model=1, verbosity_level=0, no_prefix=False):

    available_gpus = get_device_list()
    if not use_gpu or not available_gpus:
        print_distributed(verbosity_level, "Using CPU")
        return "cpu"

    world_size, world_rank = get_comm_size_and_rank()
    if rank_per_model != 1:
        raise ValueError("Exactly 1 rank per device currently supported")

    if torch.cuda.is_available():
        print_distributed(verbosity_level, "Using GPU")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        print_distributed(verbosity_level, "Using XPU")
    ## We need to ge a local rank if there are multiple GPUs available.
    localrank = 0
    if torch.cuda.device_count() > 1 or (
        hasattr(torch, "xpu") and torch.xpu.device_count() > 1
    ):
        if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
            ## Summit
            localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        elif os.getenv("SLURM_LOCALID"):
            ## CADES
            localrank = int(os.environ["SLURM_LOCALID"])
        elif os.getenv("PALS_LOCAL_RANKID"):
            ## Aurora
            localrank = int(os.environ.get("PALS_LOCAL_RANKID"))

        if localrank >= torch.cuda.device_count() and torch.cuda.is_available():
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.cuda.device_count())
            )
        elif (
            hasattr(torch, "xpu")
            and localrank >= torch.xpu.device_count()
            and torch.xpu.is_available()
        ):
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.xpu.device_count())
            )

    if no_prefix:
        device_name = str(localrank)
    elif torch.cuda.is_available():
        device_name = "cuda:" + str(localrank)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        ## (2025/05) jyc: Getting error with xpu:n format. Use only "xpu" with set_device
        # device_name = "xpu:" + str(localrank)
        device_name = "xpu"
        torch.xpu.set_device(localrank)

    return device_name


def get_deepspeed_init_args():
    class Obj(object):
        pass

    obj = Obj()
    obj.device_rank = int(get_device_name(no_prefix=True))
    return obj


def get_local_rank():
    localrank = 0
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
        ## Summit
        localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif os.getenv("SLURM_LOCALID"):
        ## CADES
        localrank = int(os.environ["SLURM_LOCALID"])
    elif os.getenv("PALS_LOCAL_RANKID"):
        localrank = int(os.environ.get("PALS_LOCAL_RANKID"))

    return localrank


def get_device_from_name(name: str):
    # [MODIFIED for Intel XPU]
    # If name starts with xpu, return torch.device("xpu", index)
    if name.startswith("xpu"):
        # e.g. "xpu:0"
        return torch.device(name)
    elif name.startswith("cuda"):
        return torch.device(name)
    else:
        return torch.device("cpu")


def get_device(use_gpu=True, rank_per_model=1, verbosity_level=0):

    name = get_device_name(use_gpu, rank_per_model, verbosity_level)
    return get_device_from_name(name)


def is_model_distributed(model):
    return isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)


def get_distributed_model(
    model,
    verbosity=0,
    sync_batch_norm=False,
    find_unused_parameters=False,
    enhanced_model=False,
):
    device_name = get_device_name(verbosity_level=verbosity)
    print(
        "dist.is_initialized(),sync_batch_norm,device_name:",
        dist.is_initialized(),
        sync_batch_norm,
        device_name,
    )

    if dist.is_initialized():
        if device_name == "cpu":
            ddp_kwargs = {
                "find_unused_parameters": find_unused_parameters,
                "gradient_as_bucket_view": not enhanced_model,
            }
            # Add static_graph=False for enhanced models with dynamic computation
            if enhanced_model:
                ddp_kwargs["static_graph"] = False
        else:
            if sync_batch_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            device = get_device_from_name(device_name)

            ddp_kwargs = {
                "device_ids": [device],
                "find_unused_parameters": find_unused_parameters,
                "gradient_as_bucket_view": not enhanced_model,
            }
            # Add static_graph=False for enhanced models with dynamic computation
            if enhanced_model:
                ddp_kwargs["static_graph"] = False

        ## check if FSDP is to be used
        use_fsdp = bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0")))
        ## List of ShardingStrategy: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, HYBRID_SHARD_ZERO2
        fsdp_strategy = os.getenv("HYDRAGNN_FSDP_STRATEGY", "FULL_SHARD")
        sharding_strategy = eval(f"ShardingStrategy.{fsdp_strategy}")
        print("Using FSDP:", use_fsdp, "Sharding:", sharding_strategy)

        if use_fsdp:
            print_distributed(verbosity, "Using FSDP wrapper")
            model = FSDP(model, sharding_strategy=sharding_strategy)
        else:
            model = DDP(model, **ddp_kwargs)

    return model


def distributed_model_wrapper(
    model,
    optimizer,
    verbosity=0,
    sync_batch_norm=False,
    find_unused_parameters=False,
    use_deepspeed=False,
    config=None,
    zero_opt=False,
    bf16=False,
):

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print_distributed(verbosity, "Using ipex.optimize wrapper")
        import intel_extension_for_pytorch as ipex

        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    else:
        print_distributed(
            verbosity, "CPUs, NVIDIA, and AMD GPUs do not need optimize wrapper"
        )

    if use_deepspeed:
        assert deepspeed_available, "deepspeed package not available"
        assert config is not None, "config is required for deepspeed"

        # create temporary deepspeed configuration
        from hydragnn.utils.input_config_parsing import parse_deepspeed_config

        ds_config = parse_deepspeed_config(config)

        if zero_opt:
            ds_config["zero_optimization"] = {"stage": 1}

        if bf16:
            ## We should choose only one of the following two
            ds_config["bf16"] = {"enabled": False}
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "bfloat16",
                "lower_precision_safe_modules": ["torch.nn.Linear", "torch.nn.Conv2d"],
            }

        # create deepspeed model
        # FIXME: need to check if it also works on ALCF-Aurora with Intel GPUs
        model, optimizer, _, _ = deepspeed.initialize(
            args=get_deepspeed_init_args(),
            model=model,
            config=ds_config,
            dist_init_required=False,
            optimizer=optimizer,  # optimizer is managed by deepspeed
        )  # scheduler is not managed by deepspeed because it is per-epoch instead of per-step
    else:
        # Auto-detect EnhancedModelWrapper and enable find_unused_parameters to avoid DDP gradient stride warnings
        enhanced_model_detected = (
            hasattr(model, "__class__")
            and "EnhancedModelWrapper" in model.__class__.__name__
        )

        if enhanced_model_detected:
            print_distributed(
                verbosity,
                f"EnhancedModelWrapper detected: {model.__class__.__name__}",
            )
            print_distributed(
                verbosity,
                "Applying DDP optimizations: find_unused_parameters=True, gradient_as_bucket_view=False",
            )
            find_unused_parameters = True

        model = get_distributed_model(
            model,
            verbosity=verbosity,
            sync_batch_norm=sync_batch_norm,
            find_unused_parameters=find_unused_parameters,
            enhanced_model=enhanced_model_detected,
        )

    return model, optimizer


def print_peak_memory(verbosity_level, prefix):
    # FIXME: this will have to change when the code can run on AMD gpus
    if torch.cuda.is_available():
        device = get_device()
        print_distributed(
            verbosity_level,
            f"{prefix}: {torch.cuda.max_memory_allocated(device)/1e9} GB",
            f"{torch.cuda.max_memory_reserved()/1e9} GB",
        )
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = get_device()
        print_distributed(
            verbosity_level,
            f"{prefix}: {torch.xpu.max_memory_allocated(device)/1e9} GB",
            f"{torch.xpu.max_memory_reserved()/1e9} GB",
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
                should_stop = torch.tensor(left < esitmated, dtype=torch.bool).to(
                    device
                )
                print("should_stop:", left, esitmated, should_stop.item())
            except:
                should_stop = torch.tensor(False, dtype=torch.bool).to(device)
        else:
            should_stop = torch.tensor(False, dtype=torch.bool).to(device)

        dist.broadcast(should_stop, src=0)
        should_stop = should_stop.item()
    return should_stop
