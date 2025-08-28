import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch

try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
except:
    pass

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import numpy as np

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.print.print_utils import log, log0
from hydragnn.utils.distributed import nsplit
from hydragnn.utils.distributed import get_device

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, adios2_open
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from hydragnn.models import MultiTaskModelMP
from contextlib import nullcontext

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="gfm_multitasking.json"
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--num_epoch", type=int, help="num_epoch", default=None)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
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
        "--task_parallel", action="store_true", help="enable task parallel"
    )
    parser.add_argument("--use_devicemesh", action="store_true", help="use device mesh")
    parser.add_argument("--oversampling", action="store_true", help="use oversampling")
    parser.add_argument(
        "--oversampling_num_samples",
        type=int,
        help="set num samples for oversampling",
        default=None,
    )
    parser.add_argument("--nosync", action="store_true", help="disable gradient sync")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--multi",
        help="Multi dataset",
        action="store_const",
        dest="format",
        const="multi",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "GFM" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "GFM" if args.modelname is None else args.modelname

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "multi":
        ## Reading multiple datasets, which requires the following arguments:
        ## --multi_model_list: the list dataset/model names
        modellist = args.multi_model_list.split(",")
        if rank == 0:
            ndata_list = list()
            pna_deg_list = list()
            for model in modellist:
                # fname = os.path.join(
                #    os.path.dirname(__file__), "./dataset/%s.bp" % model
                # )
                fname = model
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
            intp_list = list()
            mlen = min([len(pna_deg) for pna_deg in pna_deg_list])
            for pna_deg in pna_deg_list:
                x = np.linspace(0, 1, num=len(pna_deg))
                intp = make_interp_spline(x, pna_deg)
                intp_list.append(intp)

            new_pna_deg_list = list()
            for intp in intp_list:
                x = np.linspace(0, 1, num=mlen)
                y = intp(x)
                new_pna_deg_list.append(y)

            pna_deg = np.zeros_like(new_pna_deg_list[0])
            for new_pna_deg in new_pna_deg_list:
                pna_deg += new_pna_deg
            pna_deg = pna_deg.astype(np.int64).tolist()
        else:
            process_list = None
            pna_deg = None
        process_list = comm.bcast(process_list, root=0)
        pna_deg = comm.bcast(pna_deg, root=0)
        assert comm_size == sum(process_list)

        if args.task_parallel and args.use_devicemesh:
            ## task parallel with device mesh
            assert comm_size % len(modellist) == 0
            mesh_2d = init_device_mesh(
                "cuda",
                (len(modellist), comm_size // len(modellist)),
                mesh_dim_names=("dim1", "dim2"),
            )
            world_group = dist.group.WORLD
            dim1_group = mesh_2d["dim1"].get_group()
            dim2_group = mesh_2d["dim2"].get_group()
            dim1_group_size = dist.get_world_size(group=dim1_group)
            dim2_group_size = dist.get_world_size(group=dim2_group)
            dim1_group_rank = dist.get_rank(group=dim1_group)
            dim2_group_rank = dist.get_rank(group=dim2_group)
            # mesh_2d: 0 0 0
            # mesh_2d: 1 0 1
            # mesh_2d: 2 0 2
            # mesh_2d: 3 0 3
            # mesh_2d: 4 1 0
            # mesh_2d: 5 1 1
            # mesh_2d: 6 1 2
            # mesh_2d: 7 1 3
            print(
                "mesh_2d:",
                dist.get_rank(group=world_group),
                dist.get_rank(group=dim1_group),
                dist.get_rank(group=dim2_group),
            )
            device = get_device()

            mycolor = dim1_group_rank  ## branch id
            mymodel = modellist[mycolor]

            branch_id = mycolor
            branch_group = dim2_group
        else:
            colorlist = list()
            color = 0
            for n in process_list:
                for _ in range(n):
                    colorlist.append(color)
                color += 1
            mycolor = colorlist[rank]
            mymodel = modellist[mycolor]

            if args.task_parallel:
                ## non-uniform group size (cf. uniform group size using device mesh)
                subgroup_list = list()
                irank = 0
                for n in process_list:
                    subgroup_ranks = list()
                    for _ in range(n):
                        subgroup_ranks.append(irank)
                        irank += 1
                    subgroup = dist.new_group(ranks=subgroup_ranks)
                    subgroup_list.append(subgroup)

                branch_id = mycolor
                branch_group = subgroup_list[mycolor]

        local_comm = comm.Split(mycolor, rank)
        local_comm_rank = local_comm.Get_rank()
        local_comm_size = local_comm.Get_size()

        ## FIXME: Hard-coded for now. Need to find common variable names
        common_variable_names = [
            "x",
            "edge_index",
            "edge_attr",
            "pos",
            "energy",
            "forces",
            "y",
            # "dataset_name",
        ]
        # fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % mymodel)
        fname = mymodel
        print("mymodel:", rank, mycolor, mymodel)
        trainset = AdiosDataset(
            fname,
            "trainset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )
        valset = AdiosDataset(
            fname,
            "valset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )
        testset = AdiosDataset(
            fname,
            "testset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )

        ## Set local set
        num_samples_list = list()
        for dataset in [trainset, valset, testset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            print(
                f"{rank} {dataset.dataset_name} nsplit:",
                len(dataset),
                local_comm_size,
                len(rx),
            )

            if args.num_samples is not None:
                if args.num_samples > len(rx):
                    print(
                        f"WARN: Requested num_samples is larger than available in {dataset.dataset_name}: {args.num_samples} {len(rx)}"
                    )
                    # args.oversampling = True
                    # args.oversampling_num_samples = args.num_samples
                else:
                    rx = rx[: args.num_samples]

            local_dataset_len = len(rx)
            local_dataset_min = comm.allreduce(local_dataset_len, op=MPI.MIN)
            local_dataset_max = comm.allreduce(local_dataset_len, op=MPI.MAX)

            if args.task_parallel:
                rx = rx[:local_dataset_min]

            if args.oversampling:
                oversampling_num_samples = (
                    args.oversampling_num_samples
                    if args.oversampling_num_samples is not None
                    else local_dataset_max
                )
                num_samples_list.append(oversampling_num_samples)
                print(
                    f"Oversampling {oversampling_num_samples} samples: {oversampling_num_samples/local_dataset_len*100:.2f} (%)"
                )

            print(
                rank,
                "local dataset:",
                local_comm_rank,
                local_comm_size,
                dataset.label,
                len(dataset),
                rx[0],
                rx[-1],
                len(rx),
                dataset.dataset_name,
            )
            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

        print("num_samples_list:", num_samples_list)
        """
        #FIXME: will replace it with Max's new dataset
        datasets=[]
        for dataset in [trainset, valset, testset]:
            dataset_=[]
            for data in dataset:
                data.branchtype = f"branch-{mycolor}"
                print("Pei debugging 1", data)
                dataset_.append(data.to(get_device()))
            datasets.append(dataset_)
        trainset, valset, testset = datasets
        for dataset in [trainset, valset, testset]:
            for data in dataset:
                print("Pei debugging 2", data)
        """
        # print(
        #     rank,
        #     "color, moddelname, local size(trainset,valset,testset):",
        #     mycolor,
        #     mymodel,
        #     len(trainset),
        #     len(valset),
        #     len(testset),
        # )

        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width, "local": True}
            if args.task_parallel:
                trainset = DistDataset(trainset, "trainset", local_comm, **opt)
                valset = DistDataset(valset, "valset", local_comm, **opt)
                testset = DistDataset(testset, "testset", local_comm, **opt)
                trainset.pna_deg = pna_deg
                valset.pna_deg = pna_deg
                testset.pna_deg = pna_deg
            else:
                trainset = DistDataset(trainset, "trainset", comm, **opt)
                valset = DistDataset(valset, "valset", comm, **opt)
                testset = DistDataset(testset, "testset", comm, **opt)
                trainset.pna_deg = pna_deg
                valset.pna_deg = pna_deg
                testset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    log0(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
        group=branch_group if args.task_parallel else None,
        oversampling=args.oversampling,
        num_samples=num_samples_list,
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    # for data in train_loader:
    #    print("Pei debugging 3", data)

    if args.ddstore:
        train_loader.dataset.ddstore.epoch_begin()
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    if args.ddstore:
        train_loader.dataset.ddstore.epoch_end()
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]

    ## task parallel
    if args.task_parallel:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
        )
        model = MultiTaskModelMP(model, branch_id, branch_group)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
        )
        model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
            model, optimizer, verbosity
        )

    # Print details of neural network architecture
    print_model(model)

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    if args.nosync:
        context = model.no_sync()
    else:
        context = nullcontext()

    with context:
        hydragnn.train.train_validate_test(
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            writer,
            scheduler,
            config["NeuralNetwork"],
            log_name,
            verbosity,
            create_plots=False,
        )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    dist.destroy_process_group()
    sys.exit(0)
