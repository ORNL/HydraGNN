import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.preprocess.lsms_raw_dataset_loader import LSMS_RawDataLoader
from hydragnn.utils.model import print_model

import torch
import torch.distributed as dist


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="lsms.json")
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
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    os.environ["SERIALIZED_DATA_PATH"] = dirpwd + "/dataset"
    datasetname = config["Dataset"]["name"]
    fname_adios = dirpwd + "/dataset/%s.bp" % (datasetname)
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    if not args.loadexistingsplit:
        for dataset_type, raw_data_path in config["Dataset"]["path"].items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(dirpwd, raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)
            config["Dataset"]["path"][dataset_type] = raw_data_path

        ## each process saves its own data file
        loader = LSMS_RawDataLoader(config["Dataset"], dist=True)
        loader.load_raw_data()

        ## Read total pkl and split (no graph object conversion)
        hydragnn.preprocess.total_to_train_val_test_pkls(config, isdist=True)

        ## Read each pkl and graph object conversion with max-edge normalization
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)

        if args.format == "adios":
            from hydragnn.utils.adiosdataset import AdiosWriter

            adwriter = AdiosWriter(fname_adios, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", loader.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", loader.minmax_graph_feature)
            adwriter.save()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        from hydragnn.utils.adiosdataset import AdiosDataset

        info("Adios load")
        trainset = AdiosDataset(fname_adios, "trainset", comm)
        valset = AdiosDataset(fname_adios, "valset", comm)
        testset = AdiosDataset(fname_adios, "testset", comm)
        ## Set minmax read from bp file
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_node_feature"
        ] = trainset.minmax_node_feature
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_graph_feature"
        ] = trainset.minmax_graph_feature
    elif args.format == "pickle":
        config["Dataset"]["path"] = {}
        ##set directory to load processed pickle files, train/validate/test
        for dataset_type in ["train", "validate", "test"]:
            raw_data_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_{dataset_type}.pkl"
            config["Dataset"]["path"][dataset_type] = raw_data_path
        info("Pickle load")
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)
        # FIXME: here is a navie implementation with allgather. Need to have better/faster implementation
        trainlist = [None for _ in range(dist.get_world_size())]
        vallist = [None for _ in range(dist.get_world_size())]
        testlist = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(trainlist, trainset)
        dist.all_gather_object(vallist, valset)
        dist.all_gather_object(testlist, testset)
        trainset = [item for sublist in trainlist for item in sublist]
        valset = [item for sublist in vallist for item in sublist]
        testset = [item for sublist in testlist for item in sublist]
    else:
        raise ValueError("Unknown data format: %d" % args.format)

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

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
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
