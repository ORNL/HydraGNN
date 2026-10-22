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
import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import random


import logging
import sys

import numpy as np

import torch
import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
)
from hydragnn.preprocess.load_data import split_dataset

torch.set_default_dtype(torch.float32)

from utils.create_graph_data import Dataloader


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# charge and spin are constant across Transition1x dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


class Transition1xDataset(AbstractBaseDataset):
    """Transition1xDataset dataset class"""

    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
    ):
        super().__init__()

        self.data_path = os.path.join(dirpath, "transition1x-release.h5")

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.energy_per_atom = energy_per_atom

        self.world_size = 1
        self.rank = 0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        # loop through all configurations in the data set
        dataloader = Dataloader(
            self.data_path, comm_rank=self.rank, comm_size=self.world_size
        )

        for i, configuration in enumerate(dataloader):

            data_object = None

            pos = None
            try:
                pos = torch.tensor(configuration["positions"]).to(torch.float32)
                assert pos.shape[0] > 0, "pos tensor does not have any atoms"
                assert (
                    pos.shape[1] == 3
                ), "pos tensor does not have 3 coordinates per atom"
            except:
                print(
                    f"Structure {configuration} does not have positional sites",
                    flush=True,
                )
                continue
            natoms = torch.IntTensor([pos.shape[0]])

            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

            atomic_numbers = None
            try:
                atomic_numbers = (
                    torch.tensor([configuration["atomic_numbers"]]).to(torch.float32)
                ).t()
                assert (
                    pos.shape[0] == atomic_numbers.shape[0]
                ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
            except:
                print(
                    f"Structure {configuration} does not have positional atomic numbers",
                    flush=True,
                )
                continue

            forces = None
            try:
                forces = torch.tensor(configuration["wB97x_6-31G(d).forces"]).to(
                    torch.float32
                )
            except:
                print(f"Structure {configuration} does not have forces", flush=True)
                continue

            total_energy = None
            try:
                total_energy = configuration["wB97x_6-31G(d).energy"]
            except:
                print(
                    f"Structure {configuration} does not have total energy", flush=True
                )
                continue
            total_energy_tensor = (
                torch.tensor(total_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
            )
            total_energy_per_atom_tensor = total_energy_tensor.detach().clone() / natoms

            # Calculate chemical composition
            atomic_number_list = atomic_numbers.tolist()
            assert len(atomic_number_list) == natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)
            pos_list = pos.tolist()
            atomic_number_list_int = [int(item[0]) for item in atomic_number_list]
            """
            try:
                mol = xyz2mol(
                    atomic_number_list_int,
                    pos_list,
                    charge=0,
                    allow_charged_fragments=True,
                    use_graph=False,
                    use_huckel=False,
                    embed_chiral=True,
                    use_atom_maps=False,
                )

                assert (
                    len(mol) == 1
                ), f"molecule with atomic numbers {atomic_number_list_int}  and positions {pos_list} does not produce RDKit.mol object"
                smiles_string = Chem.MolToSmiles(mol[0])
            except:
                smiles_string = None
            """

            try:
                # check forces values
                assert self.check_forces_values(
                    forces
                ), f"transition1x dataset - formula:{configuration['formula']} - confid:{configuration['rxn']} - L2-norm of atomic forces exceeds {self.forces_norm_threshold}"
            except:
                continue

            data_object = Data(
                dataset_name="transition1x",
                natoms=natoms,
                pos=pos,
                cell=cell,  # even if not needed, cell needs to be defined because ADIOS requires consistency across datasets
                pbc=pbc,  # even if not needed, pbc needs to be defined because ADIOS requires consistency across datasets
                # edge_index=None,
                # edge_attr=None,
                atomic_numbers=atomic_numbers.long().view(-1),
                chemical_composition=chemical_composition,
                # smiles_string=smiles_string,
                energy=total_energy_tensor,
                energy_per_atom=total_energy_per_atom_tensor,
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

            self.dataset.append(data_object)

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
    parser = cli.build_parser(default_inputfile="transition1x_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "transition1x")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "transition1x"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        total = Transition1xDataset(
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
        print(
            "Local splitting: ",
            len(total),
            len(trainset),
            len(valset),
            len(testset),
            flush=True,
        )

        print("Before COMM.Barrier()", flush=True)
        comm.Barrier()
        print("After COMM.Barrier()", flush=True)

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
