##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import torch
import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import random
import sys
import sqlite3
import numpy as np

from ase.db import connect

from hydragnn import cli
from hydragnn.utils.print.print_utils import iterate_tqdm, log

from hydragnn.distributed import nsplit
from hydragnn.preprocess.load_data import split_dataset

import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance

from hydragnn.data.abstractbasedataset import AbstractBaseDataset

torch.set_default_dtype(torch.float32)

transform_coordinates = Distance(norm=False, cat=False)

# Conversion constant from Hartree to electron volt (eV).
# Source: NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?hrjtoeV
# Value: 1 Hartree = 27.2114079527 eV (use at least 10 significant digits for scientific accuracy)
conversion_constant_from_hartree_to_eV = 27.2114079527

# charge and spin are constant across QM7-X dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


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
        # Force serial DB access so ASE does not try to MPI-broadcast selects
        # internally (which conflicts with our own MPI logic).
        db = connect(db_path, serial=True, use_lock_file=False)

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
            torch.tensor([[float(energy_scalar)]], dtype=torch.float32)
            * conversion_constant_from_hartree_to_eV
        )
        forces = (
            torch.tensor(row.data["forces"], dtype=torch.float32)
            * conversion_constant_from_hartree_to_eV
        )

        # Calculate chemical composition
        atomic_number_list = atomic_numbers.tolist()
        assert len(atomic_number_list) == natoms
        ## 118: number of atoms in the periodic table
        hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        energy_per_atom = energy.detach().clone() / natoms

        data_object = Data(
            dataset_name="nabla2dft",
            natoms=natoms,
            pos=pos,
            cell=cell,
            pbc=pbc,
            atomic_numbers=atomic_numbers.long().view(-1),
            chemical_composition=chemical_composition,
            energy=energy,
            energy_per_atom=energy_per_atom,
            forces=forces,
        )

        data_object.energy_per_atom = energy_per_atom

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
    parser = cli.build_parser(default_inputfile="nabla2_dft.json")
    parser.add_argument(
        "--dbpath_train",
        default=None,
        help="training db (absolute or relative to dataset/)",
    )
    parser.add_argument(
        "--dbpath_test",
        default=None,
        help="testing db (absolute or relative to dataset/)",
    )
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    torch.manual_seed(0)
    random.seed(0)

    def resolve_db_path(user_value, default_name):
        if user_value is None:
            return os.path.join(datadir, default_name)
        return os.path.join(datadir, user_value)

    train_db_path = resolve_db_path(
        args.dbpath_train, "train_2k_v2_formation_energy_w_forces.db"
    )
    test_db_path = resolve_db_path(
        args.dbpath_test, "test_full_structures_v2_formation_energy_forces.db"
    )
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "Nabla2DFT")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "Nabla2DFT"
    path = cli.dataset_path(dirpwd, modelname, args.format)

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

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
