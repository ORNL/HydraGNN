import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch

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
from hydragnn.train.train_validate_test import test
try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from hydragnn.models import MultiTaskModelMP
from contextlib import nullcontext
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

## FIMME
torch.backends.cudnn.enabled = False

plt.rcParams.update({"font.size": 20})


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm

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
    parser.add_argument(
        "--task_parallel", action="store_true", help="enable task parallel"
    )
    parser.add_argument(
        "--use_devicemesh", action="store_true", help="use device mesh"
    )
    parser.add_argument(
        "--oversampling", action="store_true", help="use oversampling"
    )
    parser.add_argument(
        "--nosync", action="store_true", help="disable gradient sync"
    )


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
    log_name =  os.path.basename(args.log) 
    #modeldir = os.path.join(dirpwd,f"../../logs/{log_name}")
    modeldir = args.log
    ##################################################################################################################

    input_filename = os.path.join(modeldir, "config.json")
    if not os.path.exists(input_filename):
        raise ValueError(f"Cannot find config file {input_filename}")
    with open(input_filename, "r") as f:
        config = json.load(f)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Training"]["num_epoch"] = 1
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    ##################################################################################################################
    hydragnn.utils.print.setup_log(log_name)
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

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
                with ad2.open(fname, "r", MPI.COMM_SELF) as f:
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
            #"dataset_name",
        ]
        # fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % mymodel)
        fname = mymodel
        print("mymodel:", rank, mycolor, mymodel)
        #comment it out for fast inference
        """
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
        """
        testset = AdiosDataset(
            fname,
            "testset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )

        ## Set local set
        num_samples_list = list()
        """
        for dataset in [trainset, valset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            rx_limit = len(rx)
            if args.task_parallel:
                ## Adjust to use the same number of samples
                rx_limit = comm.allreduce(len(rx), op=MPI.MAX) if args.oversampling else comm.allreduce(len(rx), op=MPI.MIN)

            print("local dataset:", local_comm_rank, local_comm_size, dataset.label, len(rx), rx_limit)
            if args.num_samples is not None:
                if args.num_samples > rx_limit:
                    log(
                        f"WARN: requested samples are larger than what is available. Use only {len(rx)}: {dataset.label}"
                    )
                else:
                    rx_limit = args.num_samples

            if rx_limit < len(rx):
                rx = rx[:rx_limit]
            print(rank, f"Oversampling ratio: {dataset.label} {len(rx)*local_comm_size/len(trainset)*100:.02f} (%)")
            num_samples_list.append(rx_limit)
            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)
        """
        for dataset in [testset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            rx_limit = len(rx)
            if args.task_parallel:
                ## Adjust to use the same number of samples
                rx_limit = comm.allreduce(len(rx), op=MPI.MAX) if args.oversampling else comm.allreduce(len(rx), op=MPI.MIN)
                
            print("local dataset:", local_comm_rank, local_comm_size, dataset.label, len(rx), rx_limit)
            num_samples = rx_limit
            if args.num_test_samples is not None:
                num_samples = args.num_test_samples
            elif args.num_samples is not None:
                num_samples = args.num_samples
            if num_samples < rx_limit:
                rx_limit = num_samples

            if rx_limit < len(rx):
                rx = rx[:rx_limit]
            num_samples_list.append(rx_limit)
            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)
        print("num_samples_list:", num_samples_list)

        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width, "local": True}
            if args.task_parallel:
                #trainset = DistDataset(trainset, "trainset", local_comm, **opt)
                #valset = DistDataset(valset, "valset", local_comm, **opt)
                testset = DistDataset(testset, "testset", local_comm, **opt)
                #trainset.pna_deg = pna_deg
                #valset.pna_deg = pna_deg
                testset.pna_deg = pna_deg
            else:
                #trainset = DistDataset(trainset, "trainset", comm, **opt)
                #valset = DistDataset(valset, "valset", comm, **opt)
                testset = DistDataset(testset, "testset", comm, **opt)
                #trainset.pna_deg = pna_deg
                #valset.pna_deg = pna_deg
                testset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    #log0(
    #    "trainset,valset,testset size: %d %d %d"
    #    % (len(trainset), len(valset), len(testset))
    #)
    log0("testset size: %d"% (len(testset)))

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        testset,
        testset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
        group=branch_group if args.task_parallel else None,
        oversampling=args.oversampling,
        num_samples=num_samples_list,
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()
    timer.stop()
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    ## task parallel
    if args.task_parallel:
        model = MultiTaskModelMP(model, branch_id, branch_group)
    else:
        model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
    # Print details of neural network architecture
    print_model(model)
    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"],path=os.path.dirname(modeldir)
    )

    ##################################################################################################################
    model.eval()
    datasetname = os.path.basename(mymodel)[:-3]
    ##################################################################################################################
    nheads = len(config["NeuralNetwork"]["Variables_of_interest"]["output_names"])
    fig, axs = plt.subplots(1, nheads, figsize=(14, 6))
    for icol, (loader, setname) in enumerate(zip([test_loader], ["test"])):
        total_error, tasks_error, true_values, predicted_values = test(loader, model, verbosity, reduce_ranks=True, return_samples=True) #, num_samples=1024)
        print(rank, "number of heads %d"%len(true_values))
        print(rank, "number of samples %d"%len(true_values[0]))
        if rank==0:
            print(log_name, datasetname,setname, "loss=", total_error, tasks_error)
            assert len(true_values)==len(predicted_values), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_values)))
            for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
            config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
            config["NeuralNetwork"]["Variables_of_interest"]["type"],
            config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
            )): 
                head_true = true_values[ihead].cpu().squeeze().numpy() 
                head_pred = predicted_values[ihead].cpu().squeeze().numpy() 
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = var_config["output_names"][ihead]

                ax = axs[ihead]
                error_mae = np.mean(np.abs(head_pred - head_true))
                error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
                print(log_name, datasetname, setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
                print(rank, head_true.size, head_pred.size)
                hist2d_norm = getcolordensity(head_true, head_pred)
                sc=ax.scatter(head_true[::100], head_pred[::100], s=12, c=hist2d_norm[::100], vmin=0, vmax=1)
                minv = np.minimum(np.amin(head_pred), np.amin(head_true))
                maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
                ax.plot([minv, maxv], [minv, maxv], "r--")
                ax.set_title(setname + "; " + varname, fontsize=24)
                ax.text(
                    minv + 0.1 * (maxv - minv),
                    maxv - 0.1 * (maxv - minv),
                    "MAE: {:.2e}".format(error_mae),
                )
                if icol==0:
                    ax.set_ylabel("Predicted")
                if ihead==1:
                    ax.set_xlabel("True")
                #plt.colorbar(sc)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(sc, cax=cax, orientation='vertical')
                #cbar=plt.colorbar(sc)
                #cbar.ax.set_ylabel('Density', rotation=90)
                #ax.set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.3)
        fig.savefig("./logs/" + f"/parity_plot_{log_name}_{datasetname}.png",dpi=300)
        #fig.savefig("./logs/" + log_name + f"/parity_plot_{datasetname}.pdf")
        plt.close()
    sys.exit(0)

