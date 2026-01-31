import os
import glob
import random
import sys
import traceback
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from mpi4py import MPI
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
from tqdm import tqdm

from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    pbc_as_tensor,
)
from hydragnn.utils.distributed import nsplit

# Ensure the bundled fairchem package is importable when not installed system-wide.
dirpwd = os.path.dirname(os.path.abspath(__file__))
fairchem_path = os.path.join(dirpwd, "fairchem")
if fairchem_path not in sys.path:
    sys.path.insert(0, fairchem_path)

from fairchem.core.datasets import AseDBDataset  # noqa: E402

# Coordinate transforms
transform_coordinates = Distance(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


def _create_pytorch_data_object(
    index,
    dataset,
    energy_per_atom_bool,
    radius_graph_pbc,
    radius_graph,
    graphgps_transform,
    forces_norm_threshold,
):
    try:
        atoms = dataset.get_atoms(index)

        pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float32)
        natoms = torch.IntTensor([pos.shape[0]])
        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.float32,
        ).unsqueeze(1)

        energy = torch.as_tensor(
            atoms.get_total_energy(), dtype=torch.float32
        ).unsqueeze(0)
        energy_per_atom = energy.detach().clone() / natoms
        forces = torch.as_tensor(atoms.get_forces(), dtype=torch.float32)

        chemical_formula = atoms.get_chemical_formula()

        cell = None
        try:
            cell = torch.from_numpy(np.asarray(atoms.get_cell())).to(torch.float32)
        except Exception:
            print(
                f"Atomic structure {chemical_formula} does not have cell",
                flush=True,
            )

        pbc = None
        try:
            pbc = pbc_as_tensor(atoms.get_pbc())
        except Exception:
            print(f"Atomic structure {chemical_formula} does not have pbc", flush=True)

        if cell is None or pbc is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        x = torch.cat([atomic_numbers, pos, forces], dim=1)

        atomic_number_list = atomic_numbers.tolist()
        assert len(atomic_number_list) == natoms
        hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        charge = atoms.info.get("charge", 0)
        spin = atoms.info.get("spin", 1)
        graph_attr = torch.tensor([charge, spin], dtype=torch.float32)

        data_object = Data(
            dataset_name="oc25",
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
            graph_attr=graph_attr,
        )

        data_object.y = (
            data_object.energy_per_atom if energy_per_atom_bool else data_object.energy
        )

        if data_object.pbc.any():
            try:
                data_object = radius_graph_pbc(data_object)
                data_object = transform_coordinates_pbc(data_object)
            except Exception:
                print(
                    "Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                    flush=True,
                )
                data_object = radius_graph(data_object)
                data_object = transform_coordinates(data_object)
        else:
            data_object = radius_graph(data_object)
            data_object = transform_coordinates(data_object)

        if not hasattr(data_object, "edge_shifts"):
            data_object.edge_shifts = torch.zeros(
                (data_object.edge_index.size(1), 3), dtype=torch.float32
            )

        data_object.pbc = data_object.pbc.int()

        if graphgps_transform is not None:
            data_object = graphgps_transform(data_object)

        if check_forces_values(data_object.forces, forces_norm_threshold):
            return data_object
        else:
            print(
                f"L2-norm of force tensor is {data_object.forces.norm()} and exceeds threshold {forces_norm_threshold} - atomistic structure: {chemical_formula}",
                flush=True,
            )
            return None

    except Exception as e:
        print(f"Exception: {e}\nTraceback: {traceback.format_exc()}")
        return None


def check_forces_values(forces, forces_norm_threshold):
    norms = torch.norm(forces, p=2, dim=1)
    return torch.all(norms < forces_norm_threshold).item()


class OpenCatalystDataset(AbstractBaseDataset):
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

        assert data_type in ["train", "val"], "data_type must be 'train' or 'val'"

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
        else:
            self.world_size = 1
            self.rank = 0
        self.comm = comm

        all_files = self._get_all_files(dirpath, data_type)
        my_files = list(nsplit(all_files, self.world_size))[self.rank]

        self._process_local_files(my_files)

        print(
            self.rank,
            f"Rank {self.rank} done creating pytorch data objects for {data_type}. Waiting on barrier.",
            flush=True,
        )
        MPI.COMM_WORLD.Barrier()

        random.shuffle(self.dataset)

    def _get_all_files(self, dirpath, data_type):
        total_file_list = None
        if self.rank == 0:
            total_file_list = glob.glob(
                os.path.join(dirpath, data_type, "*.aselmdb"), recursive=False
            )
        total_file_list = self.comm.bcast(total_file_list, root=0)
        total_file_list.sort()
        return total_file_list

    def _process_local_files(self, my_files):
        """
        Process files assigned to this MPI rank.
        Rank spawns threads to process samples in parallel.
        """
        for filename in my_files:
            try:
                dataset = AseDBDataset(config=dict(src=filename))
            except ValueError:
                print(f"{filename} not a valid ase lmdb dataset. Ignoring ...")
                continue

            partial_func = partial(
                _create_pytorch_data_object,
                dataset=dataset,
                energy_per_atom_bool=self.energy_per_atom,
                radius_graph_pbc=self.radius_graph_pbc,
                radius_graph=self.radius_graph,
                graphgps_transform=self.graphgps_transform,
                forces_norm_threshold=self.forces_norm_threshold,
            )

            nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1
            with ThreadPoolExecutor(max_workers=nw) as executor:
                futures = [
                    executor.submit(partial_func, index)
                    for index in range(dataset.num_samples)
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {filename}",
                    disable=(self.rank != 0),
                ):
                    data_obj = future.result()
                    if data_obj is not None:
                        self.dataset.append(data_obj)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
