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

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def check_node_feature_dim(var_config):
    # NOTE: The following check is made to ensure compatibility with the json parsing of node features
    # and compute_grad_energy.
    # NOTE: In short: We need the node feature used to set up data.y to be of dimension 1, since this will dictate our
    # nodal MLP head output dimension. Since we have node_feature_dims[0] == 1 and output_index == 0, this is already true.
    # NOTE: In detail: When using Hydra for physics-informed force prediction for the GFM, we have the following structure:
    # --> Load with ADIOS -->
    # --> update_predicted_values(): defines y_loc = [0, node_feature_dims[output_index]*num_nodes] -->
    # --> update_config_NN_outputs(): y_loc exists, so it defines output_dim = [(node_feature_dims[output_index]*num_nodes)/num_nodes]  = [node_feature_dims[output_index]] -->
    # --> Base() ... MLPNode(): defines node MLP head with output_dim ... This must be equal to 1 as expected for nodal energy predictions
    # NOTE Since changing json parsing functions requires base-level code changes and the imposed requirement is already being obeyed
    # in the data setup for GFM, a quick check has been placed here instead.
    if var_config["node_feature_dims"][var_config["output_index"][0]] != 1:
        raise ValueError("Your node feature dim at the output index is not equal to 1.")


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
        "--num_test_samples",
        type=int,
        help="set num test samples per process for weak-scaling test",
        default=None,
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
    check_node_feature_dim(var_config)

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

        colorlist = list()
        color = 0
        for n in process_list:
            for _ in range(n):
                colorlist.append(color)
            color += 1
        mycolor = colorlist[rank]
        mymodel = modellist[mycolor]

        local_comm = comm.Split(mycolor, rank)
        local_comm_rank = local_comm.Get_rank()
        local_comm_size = local_comm.Get_size()

        ## FIXME: Hard-coded for now. Need to find common variable names
        common_variable_names = [
            "x",
            "edge_index",
            "edge_attr",
            "pos",
            "y",
            "dataset_name",
        ]
        # fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % mymodel)
        fname = mymodel
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
        for dataset in [trainset, valset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            if args.num_samples is not None:
                if args.num_samples > len(rx):
                    log(
                        f"WARN: requested samples are larger than what is available. Use only {len(rx)}: {dataset.label}"
                    )
                rx = rx[: args.num_samples]

            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

        for dataset in [testset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            num_samples = len(rx)
            if args.num_test_samples is not None:
                num_samples = args.num_test_samples
            elif args.num_samples is not None:
                num_samples = args.num_samples
            rx = rx[:num_samples]

            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)
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

    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
    )

    # for data in train_loader:
    #    print("Pei debugging 3", data)

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # Print details of neural network architecture
    print_model(model)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

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

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
