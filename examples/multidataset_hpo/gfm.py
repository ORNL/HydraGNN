import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.print.print_utils import log
from hydragnn.utils.distributed import nsplit

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

## FIMME
torch.backends.cudnn.enabled = False

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="gfm_multitasking.json"
    )
    parser.add_argument("--model_type", help="model_type", default="EGNN")
    parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=5)
    parser.add_argument(
        "--num_conv_layers", type=int, help="num_conv_layers", default=6
    )
    parser.add_argument("--num_headlayers", type=int, help="num_headlayers", default=2)
    parser.add_argument("--dim_headlayers", type=int, help="dim_headlayers", default=10)

    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name", default="gfm_test")
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
    args.parameters = vars(args)

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

    # Update the config dictionary with the suggested hyperparameters
    config["NeuralNetwork"]["Architecture"]["model_type"] = args.parameters[
        "model_type"
    ]
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = args.parameters[
        "hidden_dim"
    ]
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = args.parameters[
        "num_conv_layers"
    ]

    dim_headlayers = [
        args.parameters["dim_headlayers"]
        for i in range(args.parameters["num_headlayers"])
    ]

    for head_type in config["NeuralNetwork"]["Architecture"]["output_heads"]:
        config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "num_headlayers"
        ] = args.parameters["num_headlayers"]
        config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "dim_headlayers"
        ] = dim_headlayers

    if args.parameters["model_type"] not in ["EGNN", "SchNet", "DimeNet"]:
        config["NeuralNetwork"]["Architecture"]["equivariance"] = False

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

    log_name = "gfm_test" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "GFM" if args.modelname is None else args.modelname

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    elif args.format == "multi":
        ## Reading multiple datasets, which requires the following arguments:
        ## --multi_model_list: the list dataset/model names
        modellist = args.multi_model_list.split(",")
        if rank == 0:
            ndata_list = list()
            pna_deg_list = list()
            for model in modellist:
                fname = os.path.join(
                    os.path.dirname(__file__), "./dataset/%s.bp" % model
                )
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
        ]
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % mymodel)
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
        for dataset in [trainset, valset, testset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            if args.num_samples is not None:
                if args.num_samples > len(rx):
                    log(
                        f"WARN: requested samples are larger than what is available. Use only {len(rx)}."
                    )
                rx = rx[: args.num_samples]

            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

        if local_comm_rank == 0:
            print(
                rank,
                "color, moddelname, comm size, local size(trainset,valset,testset):",
                mycolor,
                mymodel,
                local_comm_size,
                len(trainset),
                len(valset),
                len(testset),
            )

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

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("Unexpected error:")
        log(sys.exc_info())
        log(e)
    sys.exit(0)
