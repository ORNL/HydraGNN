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
import os
import logging
import sys
from mpi4py import MPI

import numpy as np

import random
import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
)
from hydragnn.preprocess.load_data import split_dataset

# ===========================
# No authentication necessary
# ===========================
# Alternatively, see https://cloud.google.com/docs/authentication/gcloud.
os.system("gcloud config set auth/disable_credentials True")


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


import tensorflow as tf
import tensorflow_datasets as tfds
from utils import balanced_block

# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

LOCAL_DATA_DIR = "./dataset"
QCML_DATA_DIR = "gs://qcml-datasets/tfds"
GCP_PROJECT = "deepmind-opensource"


# charge and spin are constant across QCML dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


class QCMLDataset(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.convert_tensorflow_data_objects_to_graphs()

    def convert_tensorflow_data_objects_to_graphs(self):

        force_field_ds = tfds.load(
            "qcml/dft_force_field", split="full", data_dir=LOCAL_DATA_DIR
        )

        # Enumerate the dataset to add an index (ID) to each element
        # The elements become tuples of (id, element)
        force_field_ds_with_ids = force_field_ds.enumerate()

        N = force_field_ds.cardinality()
        start, end = balanced_block(self.rank, self.world_size, N)

        list_samples_ids = [int(i) for i in range(start, end)]

        # Convert the list of IDs into a lookup table for efficient filtering
        keys_tensor = tf.constant(list_samples_ids, dtype=tf.int64)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, keys_tensor),
            default_value=-1,
        )

        # Define the filter function
        def filter_by_id(index, element):
            # Lookup the index in the table; it will return -1 if not found
            is_in_table = table.lookup(index)
            # The sample is a match if its index is not -1
            return is_in_table != -1

        # Filter the dataset using the lookup table
        filtered_dataset = force_field_ds_with_ids.filter(filter_by_id)

        # Iterate through the filtered dataset and collect the results
        for index, element in iterate_tqdm(
            filtered_dataset, verbosity_level=4, desc=f"{self.rank}: Load"
        ):
            tensorflow_atomic_numbers = element["atomic_numbers"].numpy()
            atomic_numbers = (
                torch.from_numpy(tensorflow_atomic_numbers).to(torch.int32).unsqueeze(1)
            )
            natoms = torch.IntTensor([atomic_numbers.shape[0]])

            tensorflow_pos = element["positions"].numpy()
            pos = torch.from_numpy(tensorflow_pos).to(torch.float32)

            tensorflow_energy_float = element["pbe0_energy"].numpy()
            tensorflow_energy = np.array([[tensorflow_energy_float]])
            energy = torch.from_numpy(tensorflow_energy).to(torch.float32)
            energy_per_atom = energy.detach().clone() / natoms

            tensorflow_forces = element["pbe0_forces"].numpy()
            forces = torch.from_numpy(tensorflow_forces).to(torch.float32)

            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

            # Calculate chemical composition
            atomic_number_list = atomic_numbers.tolist()
            assert len(atomic_number_list) == natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)
            pos_list = pos.tolist()
            atomic_number_list_int = [int(item[0]) for item in atomic_number_list]

            data_object = Data(
                dataset_name="qcml",
                natoms=natoms,
                pos=pos,
                cell=cell,  # even if not needed, cell needs to be defined because ADIOS requires consistency across datasets
                pbc=pbc,  # even if not needed, pbc needs to be defined because ADIOS requires consistency across datasets
                # edge_index=None,
                # edge_attr=None,
                atomic_numbers=atomic_numbers.long().view(-1),
                chemical_composition=chemical_composition,
                # smiles_string=smiles_string,
                energy=energy,
                energy_per_atom=energy_per_atom,
                forces=forces,
            )

            data_object = self.radius_graph(data_object)

            # Build edge attributes
            data_object = transform_coordinates(data_object)

            # Default edge_shifts for when radius_graph_pbc is not activated
            data_object.edge_shifts = torch.zeros(
                (data_object.edge_index.size(1), 3), dtype=torch.float32
            )

            # FIXME: PBC from bool --> int32 to be accepted by ADIOS
            data_object.pbc = data_object.pbc.int()

            # LPE
            if self.graphgps_transform is not None:
                data_object = self.graphgps_transform(data_object)

            if self.check_forces_values(data_object.forces):
                self.dataset.append(data_object)
            else:
                print(
                    f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold} - atomistic structure: {data_object.forces}",
                    flush=True,
                )

        random.shuffle(self.dataset)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="qcml_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "QCML")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "QCML"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        total = QCMLDataset(
            os.path.join(datadir),
            config,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        comm.Barrier()
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
