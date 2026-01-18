import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import json
import random
import logging
import sys
import argparse
import sqlite3

from ase.db import connect

import hydragnn
from hydragnn.utils.print.print_utils import iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer

from hydragnn.utils.distributed import nsplit
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

import torch
import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset


torch.set_default_dtype(torch.float32)

transform_coordinates = Distance(norm=False, cat=False)

# Conversion constant from Hartree to electron volt (eV).
# Source: NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?hrjtoeV
# Value: 1 Hartree = 27.2114079527 eV (use at least 10 significant digits for scientific accuracy)
conversion_constant_from_hartree_to_eV = 27.2114079527

# charge and spin are constant across QM7-X dataset
charge = 0.0  # neutral
spin = 1.0  # singlet
graph_attr = torch.tensor([charge, spin], dtype=torch.float32)


class Nabla2RelaxDataset(AbstractBaseDataset):
    """ASE SQLite trajectory snapshots for HydraGNN."""

    def __init__(
        self,
        db_path,
        config,
        energy_per_atom=False,
        dist=False,
        dataset_label="dataset",
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.energy_per_atom = energy_per_atom
        self.dataset_label = dataset_label

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self._load_database(db_path)

    def _load_database(self, db_path):
        db = connect(db_path)

        # Avoid ASE's MPI-aware select helper (which relies on communicator-wide
        # broadcasts) to prevent MPI_ERR_OTHER when multiple ranks read the DB.
        # Using sqlite3 keeps the ID fetch local and read-only. For distributed
        # runs we broadcast the ID list so all ranks partition the same order.
        ids = None
        if not self.dist or self.rank == 0:
            try:
                with sqlite3.connect(db_path) as con:
                    ids = [row[0] for row in con.execute("SELECT id FROM systems")]
            except Exception:
                ids = [row.id for row in db.select()]

        if self.dist:
            comm = MPI.COMM_WORLD
            ids = comm.bcast(ids, root=0)
            if self.rank == 0:
                random.shuffle(ids)
            ids = comm.bcast(ids, root=0)
            chunk = list(nsplit(range(len(ids)), self.world_size))[self.rank]
        else:
            chunk = range(len(ids))

        missing = 0
        for local_idx in iterate_tqdm(
            chunk,
            2,
            desc=f"Load {self.dataset_label} snapshots",
            total=len(chunk),
        ):
            try:
                row = db.get(id=ids[local_idx])
            except KeyError:
                missing += 1
                continue

            data_object = self.row_to_graph(row)
            self.dataset.append(data_object)

        if missing:
            log(
                f"Skipped {missing} missing rows while loading {self.dataset_label}",
                rank=self.rank,
            )

    def row_to_graph(self, row):
        atoms = row.toatoms()

        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.float32).unsqueeze(1)
        natoms = torch.tensor([atoms.positions.shape[0]], dtype=torch.int32)

        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

        energy_raw = row.data["energy"]
        if isinstance(energy_raw, (list, tuple)):
            energy_scalar = energy_raw[0]
        else:
            try:
                energy_scalar = energy_raw.item()
            except Exception:
                energy_scalar = (
                    energy_raw[0] if hasattr(energy_raw, "__len__") else energy_raw
                )

        energy = (
            torch.tensor([float(energy_scalar)], dtype=torch.float32)
            * conversion_constant_from_hartree_to_eV
        )
        forces = (
            torch.tensor(row.data["forces"], dtype=torch.float32)
            * conversion_constant_from_hartree_to_eV
        )

        x = torch.cat((atomic_numbers, forces), dim=1)

        data_object = Data(
            dataset_name="nabla2dft",
            natoms=natoms,
            pos=pos,
            cell=cell,
            pbc=pbc,
            atomic_numbers=atomic_numbers,
            x=x,
            energy=energy,
            forces=forces,
            graph_attr=graph_attr,
        )

        energy_per_atom = energy.detach().clone() / natoms
        data_object.energy_per_atom = energy_per_atom
        data_object.y = energy_per_atom if self.energy_per_atom else energy

        data_object = self.radius_graph(data_object)
        data_object = transform_coordinates(data_object)

        data_object.edge_shifts = torch.zeros(
            (data_object.edge_index.size(1), 3), dtype=torch.float32
        )

        data_object.pbc = data_object.pbc.int()

        return data_object

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preonly", action="store_true", help="preprocess only")
    parser.add_argument(
        "--inputfile", help="input JSON", type=str, default="nabla2_dft.json"
    )
    parser.add_argument(
        "--dbpath_train",
        help="path to training ASE SQLite db",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dbpath_test",
        help="path to testing ASE SQLite db",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--energy_per_atom",
        help="normalize energy by number of atoms",
        action="store_true",
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp64", "bf16"],
        default=None,
        help="Override precision; defaults to fp32 when not set",
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")

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

    random_state = 0
    torch.manual_seed(random_state)
    random.seed(random_state)

    graph_feature_names = ["formation_energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "forces"]
    node_feature_dims = [1, 3]

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    # Resolve datasets relative to this example's directory
    dataset_dir = os.path.join(dirpwd, "dataset")
    default_train_db = os.path.join(
        dataset_dir, "train_2k_v2_formation_energy_w_forces.db"
    )
    default_test_db = os.path.join(
        dataset_dir, "test_full_structures_v2_formation_energy_forces.db"
    )

    def resolve_db_path(user_value, default_value):
        # If user passed a relative path or bare filename, look under dataset_dir
        if user_value is None:
            return default_value
        if os.path.isabs(user_value):
            return user_value
        return os.path.join(dataset_dir, user_value)

    train_db_path = resolve_db_path(args.dbpath_train, default_train_db)
    test_db_path = resolve_db_path(args.dbpath_test, default_test_db)

    input_filename = os.path.join(dirpwd, args.inputfile)
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

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()

    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )

    log_name = "nabla2_dft" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "nabla2_dft"

    if args.preonly:
        if not os.path.exists(train_db_path):
            raise FileNotFoundError(f"Missing training dataset db: {train_db_path}")
        if not os.path.exists(test_db_path):
            raise FileNotFoundError(f"Missing testing dataset db: {test_db_path}")

        train_dataset = Nabla2RelaxDataset(
            train_db_path,
            config,
            energy_per_atom=args.energy_per_atom,
            dist=True,
            dataset_label="train",
        )

        trainset, valset1, valset2 = split_dataset(
            dataset=train_dataset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        valset = [*valset1, *valset2]

        test_dataset = Nabla2RelaxDataset(
            test_db_path,
            config,
            energy_per_atom=args.energy_per_atom,
            dist=True,
            dataset_label="test",
        )
        testset = test_dataset[:]

        comm.Barrier()

        log(
            "Local splitting: %d %d %d %d"
            % (len(train_dataset), len(trainset), len(valset), len(testset))
        )

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        if args.format == "adios":
            if AdiosWriter is None:
                raise ImportError("ADIOS support not available")
            fname = os.path.join(dataset_dir, "%s.bp" % modelname)
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        elif args.format == "pickle":
            basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
            attrs = {"pna_deg": deg}
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                use_subdir=True,
            )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        assert AdiosDataset is not None, "ADIOS support not available"
        log("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(dataset_dir, "%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)

    elif args.format == "pickle":
        log("Pickle load")
        basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    log(
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

    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    precision = args.precision.lower() if args.precision is not None else "fp32"

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
        compute_grad_energy=config["NeuralNetwork"]["Architecture"].get(
            "enable_interatomic_potential", False
        ),
        precision=precision,
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
