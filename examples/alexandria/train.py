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
import bz2

import os
import json
import logging
import sys
import torch
from mpi4py import MPI

import numpy as np

import random

import torch.distributed as dist
from torch_geometric.data import Data

from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    pbc_as_tensor,
    should_skip_self_loops,
)
from hydragnn.preprocess.load_data import split_dataset

from hydragnn.utils.print.print_utils import iterate_tqdm

from generate_dictionaries_pure_elements import (
    generate_dictionary_elements,
)

from hydragnn.distributed import nsplit
import glob


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def list_directories(path):
    # List all items in the given path
    items = os.listdir(path)

    # Filter out items that are directories
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return directories


periodic_table = generate_dictionary_elements()

# Reversing the dictionary so the elements become keys and the atomic numbers become values
reversed_dict_periodic_table = {value: key for key, value in periodic_table.items()}

# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)

# charge and spin are constant across Alexandria dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


class Alexandria(AbstractBaseDataset):
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

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )
        self.radius_graph_pbc = RadiusGraphPBC(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        data_dir = os.path.join(dirpath, "compressed_data", "alexandria.icams.rub.de")
        # print("glob:", os.path.join(data_dir, "**/*.json.bz2"))
        total_file_list = glob.glob(
            os.path.join(data_dir, "**/*.json.bz2"), recursive=True
        )
        if self.dist:
            local_file_list = list(nsplit(total_file_list, self.world_size))[self.rank]
        else:
            local_file_list = total_file_list
        print(
            self.rank,
            "Total flies:",
            len(total_file_list),
            "Local files:",
            len(local_file_list),
        )

        for filepath in iterate_tqdm(local_file_list, verbosity_level=2):
            if filepath.endswith("bz2"):
                self.process_file_content(filepath)
            else:
                print(f"{filepath} is not a .bz2 file to decompress", flush=True)

    def get_data_dict(self, computed_entry_dict):
        """
        Processes the ComputedStructureEntry dictionary to extract the structure, forces and magnetic moments
        and other target properties.
        """

        data_object = None

        def get_forces_array_from_structure(structure):
            forces = [site["properties"]["forces"] for site in structure["sites"]]
            return np.array(forces)

        def get_magmoms_array_from_structure(structure):
            magmoms = [site["properties"]["magmom"] for site in structure["sites"]]
            return np.array(magmoms)

        entry_id = computed_entry_dict["data"]["mat_id"]
        structure = computed_entry_dict["structure"]

        pos = None
        try:
            pos = torch.tensor(
                [item["xyz"] for item in computed_entry_dict["structure"]["sites"]]
            ).to(torch.float32)
            assert pos.shape[1] == 3, "pos tensor does not have 3 coordinates per atom"
            assert pos.shape[0] > 0, "pos tensor does not have any atoms"
        except:
            print(f"Structure {entry_id} does not have positional sites", flush=True)
        natoms = torch.IntTensor([pos.shape[0]])

        cell = None
        try:
            cell = torch.tensor(
                structure["lattice"]["matrix"], dtype=torch.float32
            ).view(3, 3)
        except:
            print(f"Structure {entry_id} does not have cell", flush=True)

        pbc = None
        try:
            pbc = pbc_as_tensor(structure["lattice"]["pbc"])
        except:
            print(f"Structure {entry_id} does not have pbc", flush=True)

        # If either cell or pbc were not read, we set to defaults
        if cell is None or pbc is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        atomic_numbers = None
        try:
            atomic_numbers = (
                torch.tensor(
                    [
                        reversed_dict_periodic_table[item["species"][0]["element"]]
                        for item in computed_entry_dict["structure"]["sites"]
                    ]
                )
                .unsqueeze(1)
                .to(torch.float32)
            )
            assert (
                pos.shape[0] == atomic_numbers.shape[0]
            ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
        except:
            print(
                f"Structure {entry_id} does not have positional atomic numbers",
                flush=True,
            )
            return data_object

        forces_numpy = None
        try:
            forces_numpy = get_forces_array_from_structure(structure)
        except:
            print(f"Structure {entry_id} does not have forces", flush=True)
            return data_object
        forces = torch.tensor(forces_numpy).to(torch.float32)

        # magmoms_numpy = None
        # try:
        #    magmoms_numpy = get_magmoms_array_from_structure(structure)
        # except:
        #    print(f"Structure {entry_id} does not have magnetic moments")
        #    return data_object

        total_energy = None
        try:
            total_energy = computed_entry_dict["data"]["energy_total"]
        except:
            print(f"Structure {entry_id} does not have total energy", flush=True)
            return data_object
        total_energy_tensor = (
            torch.tensor(total_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
        )
        total_energy_per_atom_tensor = total_energy_tensor.detach().clone() / natoms

        # total_mag = None
        # try:
        #    total_mag=computed_entry_dict["data"]["total_mag"]
        # except:
        #    print(f"Structure {entry_id} does not have total magnetization")
        #    return data_object

        # dos_ef = None
        # try:
        #    dos_ef=computed_entry_dict["data"]["dos_ef"]
        # except:
        #    print(f"Structure {entry_id} does not have dos_ef")
        #    return data_object

        # band_gap_ind = None
        # try:
        #    band_gap_ind=computed_entry_dict["data"]["band_gap_ind"]
        # except:
        #    print(f"Structure {entry_id} does not have band_gap_ind")
        #    return data_object

        formation_energy = None
        try:
            formation_energy = computed_entry_dict["data"]["e_form"]
        except:
            print(f"Structure {entry_id} does not have formation energy")
            return data_object
        formation_energy_tensor = (
            torch.tensor(formation_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
        )
        formation_energy_per_atom_tensor = (
            formation_energy_tensor.detach().clone() / natoms
        )

        # energy_above_hull = None
        # try:
        #    energy_above_hull=computed_entry_dict["data"]["e_above_hull"]
        # except:
        #    print(f"Structure {entry_id} does not have e_above_hull")
        #    return data_object

        # Calculate chemical composition
        atomic_number_list = atomic_numbers.tolist()
        assert len(atomic_number_list) == natoms
        ## 118: number of atoms in the periodic table
        hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        data_object = Data(
            dataset_name="alexandria",
            natoms=natoms,
            pos=pos,
            cell=cell,
            pbc=pbc,
            edge_index=None,
            edge_attr=None,
            atomic_numbers=atomic_numbers.long().view(-1),
            chemical_composition=chemical_composition,
            smiles_string=None,
            # entry_id=entry_id,
            energy=formation_energy_tensor,
            energy_per_atom=formation_energy_per_atom_tensor,
            forces=forces,
            # formation_energy=torch.tensor(formation_energy).float(),
            # formation_energy_per_atom=torch.tensor(formation_energy_per_atom).float(),
            # energy_above_hull=energy_above_hull,
            # magmoms=torch.tensor(magmoms_numpy).float(),
            # total_mag=total_mag,
            # dos_ef=dos_ef,
            # band_gap_ind=band_gap_ind,
        )

        # Apply radius graph and build edge attributes accordingly
        if data_object.pbc.any():
            try:
                data_object = self.radius_graph_pbc(data_object)
                data_object = transform_coordinates_pbc(data_object)
            except:
                print(
                    f"Structure {entry_id} could not successfully apply one or both of the pbc radius graph and positional transform",
                    flush=True,
                )
                data_object = self.radius_graph(data_object)
                data_object = transform_coordinates(data_object)
        else:
            data_object = self.radius_graph(data_object)
            data_object = transform_coordinates(data_object)

        # Skip samples that still contain self-loops
        if should_skip_self_loops(data_object, context="alexandria"):
            return None

        # Default edge_shifts for when radius_graph_pbc is not activated
        if not hasattr(data_object, "edge_shifts"):
            data_object.edge_shifts = torch.zeros(
                (data_object.edge_index.size(1), 3), dtype=torch.float32
            )

        # FIXME: PBC from bool --> int32 to be accepted by ADIOS
        data_object.pbc = data_object.pbc.int()

        # LPE
        if self.graphgps_transform is not None:
            data_object = self.graphgps_transform(data_object)

        if self.check_forces_values(data_object.forces):
            return data_object
        else:
            print(
                f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold} - atomistic structure: {data_object}",
                flush=True,
            )
            return None

    def process_file_content(self, filepath):
        """
        Download a file from a dataset of the Alexandria database with the respective index
        and write it to the LMDB file with the respective index.

        Parameters
        ----------
        filepath : int
            path of file to decompress and open. 3D 0-44, 2D 0-1, 1d 0, scan/pbesol 0-4
        """

        # Open the .bz2 file in binary read mode
        with open(filepath, "rb") as file:
            # Read the compressed data
            compressed_data = file.read()

            # Decompress the data
            try:
                decompressed_data = bz2.decompress(compressed_data)
                json_str = decompressed_data.decode("utf-8")
                data = json.loads(json_str)

                computed_entry_dict = [
                    self.get_data_dict(entry)
                    for entry in iterate_tqdm(
                        data["entries"],
                        desc=f"Rank {self.rank} - Processing file {os.path.basename(filepath)}",
                        verbosity_level=2,
                    )
                ]

                # remove None elements
                filtered_computed_entry_dict = [
                    x for x in computed_entry_dict if x is not None
                ]

                random.shuffle(filtered_computed_entry_dict)
                self.dataset.extend(filtered_computed_entry_dict)

            except OSError as e:
                print(
                    "Failed to decompress data:",
                    e,
                    os.path.basename(filepath),
                    flush=True,
                )
                decompressed_data = None
            except json.JSONDecodeError as e:
                print(
                    "Failed to decode JSON:", e, os.path.basename(filepath), flush=True
                )
            except Exception as e:
                print("An error occurred:", e, os.path.basename(filepath), flush=True)

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
    parser = cli.build_parser(default_inputfile="alexandria_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "Alexandria")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "Alexandria"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        total = Alexandria(
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
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        print("Before COMM.Barrier()", flush=True)
        comm.Barrier()
        print("After COMM.Barrier()", flush=True)

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
