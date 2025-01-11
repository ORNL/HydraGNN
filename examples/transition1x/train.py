import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random

import h5py

import logging
import sys
import argparse

import hydragnn
from hydragnn.utils.print_utils import iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg

import numpy as np

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance
import torch
import torch.distributed as dist

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr


torch.set_default_dtype(torch.float32)

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
create_graph_fromXYZ = RadiusGraph(r=5.0)  # radius cutoff in angstrom
compute_edge_lengths = Distance(norm=False, cat=True)


class Transition1xDataset(AbstractBaseDataset):
    """Transition1xDataset dataset class"""

    def __init__(self, dirpath, var_config, energy_per_atom=True, dist=False):
        super().__init__()

        self.energy_per_atom = energy_per_atom

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        # loop through all configurations in the data set
        dataloader = Dataloader(args.h5file)
        counter = 0
        for i, configuration in enumerate(dataloader):

            pos = None
            try:
                pos = torch.tensor(
                    configuration['positions']
                ).to(torch.float32)
                assert pos.shape[1] == 3, "pos tensor does not have 3 coordinates per atom"
                assert pos.shape[0] > 0, "pos tensor does not have any atoms"
            except:
                print(f"Structure {entry_id} does not have positional sites", flush=True)
                return data_object
            natoms = torch.IntTensor([pos.shape[0]])

            atomic_numbers = None
            try:
                atomic_numbers = (
                    torch.tensor(
                        [
                            configuration['atomic_numbers']
                        ]
                    )
                    .to(torch.float32)
                ).t()
                print(atomic_numbers.shape)
                assert (
                    pos.shape[0] == atomic_numbers.shape[0]
                ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
            except:
                print(
                    f"Structure {entry_id} does not have positional atomic numbers",
                    flush=True,
                )
                return data_object

            forces = None
            try:
                forces = torch.tensor(configuration['wB97x_6-31G(d).forces']).to(torch.float32)
            except:
                print(f"Structure {entry_id} does not have forces", flush=True)
                return data_object

            total_energy = None
            try:
                total_energy = configuration['wB97x_6-31G(d).energy']
            except:
                print(f"Structure {entry_id} does not have total energy", flush=True)
                return data_object
            total_energy_tensor = (
                torch.tensor(total_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
            )
            total_energy_per_atom_tensor = total_energy_tensor.detach().clone() / natoms

            try:
                # check forces values
                assert self.check_forces_values(
                    forces
                    ), f"transition1x dataset - formula:{configuration['formula']} - confid:{configurantion['rxn']} - L2-norm of atomic forces exceeds {self.forces_norm_threshold}"

                # data = Data(
                #    pos=xyz, x=Z, molid=molid, confid=confid
                # )
                data = Data(pos=xyz, x=Z)
                data.x = torch.cat((data.x, xyz, forces, hCHG, hVDIP, hRAT), dim=1)

                if self.energy_per_atom:
                    data.y = EPBE0 / natoms
                else:
                    data.y = EPBE0

                data = create_graph_fromXYZ(data)

                # Add edge length as edge feature
                data = compute_edge_lengths(data)
                data.edge_attr = data.edge_attr.to(torch.float32)
                subset.append(data)
            except AssertionError as e:
                print(f"Assertion error occurred: {e}")

        return subset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="qm7x.json")
    parser.add_argument(
        "--energy_per_atom",
        help="option to normalize energy by number of atoms",
        type=bool,
        default=True,
    )

    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")

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
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = [
        "atomic_number",
        "coordinates",
        "forces",
        "hCHG",
        "hVDIP",
        "hRAT",
    ]
    node_feature_dims = [1, 3, 3, 1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/Transition1x")
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

    log_name = "transition1x" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "transition1x"
    if args.preonly:

        ## local data
        total = Transition1xDatasetDataset(
            os.path.join(datadir),
            var_config,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## adios
        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % modelname
            )
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        ## pickle
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
        sys.exit(0)

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
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)
        # trainset.minmax_node_feature = minmax_node_feature
        # trainset.minmax_graph_feature = minmax_graph_feature
        trainset.pna_deg = pna_deg

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

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
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

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
