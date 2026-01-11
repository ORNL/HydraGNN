import os, random, torch, glob, sys, pickle, pdb
import numpy as np
from mpi4py import MPI
from yaml import full_load

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from torch_geometric.transforms import Distance, Spherical, LocalCartesian
from torch_geometric.data import Data
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    PBCLocalCartesian,
    pbc_as_tensor,
    gather_deg,
)
from hydragnn.utils.distributed import nsplit
from utils import balance_load


# transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)
# transform_coordinates = Distance(norm=False, cat=False)

transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
# transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


import torch
from torch_geometric.data import Data, Dataset
from ase.io import read
from ase import Atoms
import os
from typing import List

from hydragnn.utils.print.print_utils import iterate_tqdm, log


class ExtendedXYZDataset(Dataset):
    def __init__(self, extxyz_filename: str, transform=None, pre_transform=None):
        """
        Args:
            file_list (List[str]): List of paths to `.extxyz` files.
        """
        super().__init__(None, transform, pre_transform)
        self.extxyz_filename = extxyz_filename
        self.structures = []
        self._load_structures()

    def _load_structures(self):
        """Reads all structures from all .extxyz files and stores them in self.structures"""
        if not os.path.isfile(self.extxyz_filename):
            raise FileNotFoundError(f"File not found: {self.extxyz_filename}")
        frames = read(self.extxyz_filename, index=':')  # Read all structures in file
        self.structures.extend(frames)

    def len(self):
        return len(self.structures)

    def get(self, idx):
        atoms: Atoms = self.structures[idx]

        return atoms



class ODAC2023(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        data_type,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
        comm=MPI.COMM_WORLD,
    ):
        super().__init__()

        assert (data_type == "train") or (
            data_type == "val"
        ), "data_type must be a string either equal to 'train' or to 'val'"

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.data_path = os.path.join(dirpath, data_type)
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

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        self.comm = comm

        # Parallelizing over data files for training data. For val set, we parallelize over each molecules.
        dataset_list = self._get_datasets_assigned_to_me(dirpath, data_type)

        for dataset_index, dataset_dict in enumerate(dataset_list):
            fullpath = dataset_dict["dataset_fullpath"]
            try:
                dataset = ExtendedXYZDataset(extxyz_filename=fullpath)
            except ValueError as e:
                print(f"{fullpath} not a valid ase lmdb dataset. Ignoring ...")
                continue

            if data_type == "train":
                rx = list(range(len(dataset)))
            else:
                rx = list(nsplit(list(range(len(dataset))), self.world_size))[
                    self.rank
                ]

            print(
                f"Rank: {self.rank}, dataname: {fullpath}, data_type: {data_type}, num_samples: {len(dataset)}, len(rx): {len(rx)}"
            )

            # for index in iterate_tqdm(
            #     rx,
            #     verbosity_level=2,
            #     desc=f"Rank{self.rank} Dataset {dataset_index}/{len(dataset_list)}",
            # ):

            for index in rx:
                self._create_pytorch_data_object(dataset, index)

        print(
            self.rank,
            f"Rank {self.rank} done creating pytorch data objects for {data_type}. Waiting on barrier.",
            flush=True,
        )
        torch.distributed.barrier()

        random.shuffle(self.dataset)

    def _get_datasets_assigned_to_me(self, dirpath, data_type):
        datasets_info = []

        # get the list of ase lmdb files
        total_file_list = None
        if self.rank == 0:
            total_file_list = glob.glob(
                os.path.join(dirpath, data_type, "**/*.extxyz"), recursive=True
            )
        total_file_list = self.comm.bcast(total_file_list, root=0)

        # evenly distribute amongst all ranks to get num samples
        rx = list(nsplit(total_file_list, self.world_size))[self.rank]
        datasets = rx

        # Get num samples for all datasets assigned to this process
        for d in iterate_tqdm(datasets, verbosity_level=2, desc="Data Parsing"):
            fullpath = os.path.join(dirpath, data_type, d)
            try:
                dataset = ExtendedXYZDataset(extxyz_filename=fullpath)
            except ValueError as e:
                print(f"{fullpath} is not a valid Extended XYZ dataset. Ignoring ...")
                continue

            num_samples = len(dataset)
            datasets_info.append(
                {"dataset_fullpath": fullpath, "num_samples": num_samples}
            )

        # All gather so everyone has all num samples information
        _all_datasets_info = self.comm.allgather(datasets_info)
        all_datasets_info = [item for sublist in _all_datasets_info for item in sublist]

        # Call workload distributor to assign datasets to processes
        my_datasets = balance_load(all_datasets_info, self.world_size, self.rank)
        return my_datasets

    def _create_pytorch_data_object(self, dataset, index):
        try:
            pos = torch.tensor(
                dataset.get(index).get_positions(), dtype=torch.float32
            )
            natoms = torch.IntTensor([pos.shape[0]])
            atomic_numbers = torch.tensor(
                dataset.get(index).get_atomic_numbers(),
                dtype=torch.float32,
            ).unsqueeze(1)

            energy = torch.tensor(
                dataset.get(index).get_total_energy(), dtype=torch.float32
            ).unsqueeze(0)

            energy_per_atom = energy.detach().clone() / natoms
            forces = torch.tensor(
                dataset.get(index).get_forces(), dtype=torch.float32
            )

            chemical_formula = dataset.get(index).get_chemical_formula()

            cell = None
            try:
                cell = torch.tensor(
                    dataset.get(index).get_cell(), dtype=torch.float32
                ).view(3, 3)
                cell = torch.from_numpy(
                    np.asarray(dataset.get(index).get_cell())
                ).to(
                    torch.float32
                )  # dtype conversion in-place
                # shape is already (3, 3) so no .view needed
            except:
                print(
                    f"Atomic structure {chemical_formula} does not have cell",
                    flush=True,
                )

            pbc = None
            try:
                pbc = pbc_as_tensor(dataset.get(index).get_pbc())
            except:
                print(
                    f"Atomic structure {chemical_formula} does not have pbc",
                    flush=True,
                )

            # If either cell or pbc were not read, we set to defaults which are not none.
            if cell is None or pbc is None:
                cell = torch.eye(3, dtype=torch.float32)
                pbc = torch.tensor([False, False, False], dtype=torch.bool)

            x = torch.cat([atomic_numbers, pos, forces], dim=1)

            # Calculate chemical composition
            atomic_number_list = atomic_numbers.tolist()
            assert len(atomic_number_list) == natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

            data_object = Data(
                dataset_name="odac23",
                natoms=natoms,
                pos=pos,
                cell=cell,
                pbc=pbc,
                edge_index=None,
                edge_attr=None,
                atomic_numbers=atomic_numbers,
                chemical_composition=chemical_composition,
                smiles_string=None,
                x=x,
                energy=energy,
                energy_per_atom=energy_per_atom,
                forces=forces,
            )

            if self.energy_per_atom:
                data_object.y = data_object.energy_per_atom
            else:
                data_object.y = data_object.energy

            if data_object.pbc.any():
                try:
                    data_object = self.radius_graph_pbc(data_object)
                    data_object = transform_coordinates_pbc(data_object)
                except:
                    print(
                        f"Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                        flush=True,
                    )
                    data_object = self.radius_graph(data_object)
                    data_object = transform_coordinates(data_object)
            else:
                data_object = self.radius_graph(data_object)
                data_object = transform_coordinates(data_object)

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
                self.dataset.append(data_object)
            else:
                print(
                    f"L2-norm of force tensor is {data_object.forces.norm()} and exceeds threshold {self.forces_norm_threshold} - atomistic structure: {chemical_formula}",
                    flush=True,
                )

        except Exception as e:
            print(f"Rank {self.rank} reading - exception: ", e)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
