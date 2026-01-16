import os
import glob
import random
import sys
import traceback
import numpy as np
import torch
from mpi4py import MPI
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
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


# Simple greedy load balancer to spread dataset files by sample count.
def balance_load(dataset_list, nranks, me):
    indexed = [
        dict(id=i, num_samples=item["num_samples"])
        for i, item in enumerate(dataset_list)
    ]
    sorted_l = sorted(indexed, key=lambda x: x["num_samples"], reverse=True)
    heap = [(0, r, []) for r in range(nranks)]
    import heapq

    heapq.heapify(heap)
    for item in sorted_l:
        load, rank, assigned = heapq.heappop(heap)
        assigned.append(dataset_list[item["id"]])
        load += item["num_samples"]
        heapq.heappush(heap, (load, rank, assigned))
    for _load, rank, assigned in heap:
        if rank == me:
            print(
                f"load balancing. Rank {rank}: num_samples: {_load}, number of datasets: {len(assigned)}"
            )
            return assigned
    return []


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

        dataset_list = self._get_datasets_assigned_to_me(dirpath, data_type)

        for dataset_dict in dataset_list:
            fullpath = dataset_dict["dataset_fullpath"]
            try:
                dataset = AseDBDataset(config=dict(src=fullpath))
            except ValueError:
                print(f"{fullpath} not a valid ase lmdb dataset. Ignoring ...")
                continue

            if data_type == "train":
                rx = list(range(dataset.num_samples))
            else:
                rx = list(nsplit(list(range(dataset.num_samples)), self.world_size))[
                    self.rank
                ]

            print(
                f"Rank: {self.rank}, dataname: {fullpath}, data_type: {data_type}, num_samples: {dataset.num_samples}, len(rx): {len(rx)}"
            )

            for index in rx:
                self._create_pytorch_data_object(dataset, index)

        print(
            self.rank,
            f"Rank {self.rank} done creating pytorch data objects for {data_type}. Waiting on barrier.",
            flush=True,
        )
        MPI.COMM_WORLD.Barrier()

        random.shuffle(self.dataset)

    def _get_datasets_assigned_to_me(self, dirpath, data_type):
        datasets_info = []

        # Gather the list of ASE LMDB shards
        total_file_list = None
        if self.rank == 0:
            total_file_list = glob.glob(
                os.path.join(dirpath, data_type, "*.aselmdb"), recursive=False
            )
        total_file_list = self.comm.bcast(total_file_list, root=0)

        # Early exit if nothing found
        if total_file_list is None or len(total_file_list) == 0:
            return []

        # Compute sample counts on a subset of files per rank
        rx_files = list(nsplit(total_file_list, self.world_size))[self.rank]
        for fullpath in rx_files:
            try:
                dataset = AseDBDataset(config=dict(src=fullpath))
            except ValueError:
                print(f"{fullpath} is not a valid ASE LMDB dataset. Ignoring ...")
                continue
            datasets_info.append(
                {"dataset_fullpath": fullpath, "num_samples": dataset.num_samples}
            )

        # Share sample counts across ranks
        _all = self.comm.allgather(datasets_info)
        all_datasets_info = [item for sub in _all for item in sub]

        # Assign datasets to ranks for training; for validation every rank sees all datasets
        if data_type == "train":
            my_datasets = balance_load(all_datasets_info, self.world_size, self.rank)
            return my_datasets if my_datasets is not None else []
        else:
            return all_datasets_info

    def _create_pytorch_data_object(self, dataset, index):
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
                print(
                    f"Atomic structure {chemical_formula} does not have pbc", flush=True
                )

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
                data_object.energy_per_atom
                if self.energy_per_atom
                else data_object.energy
            )

            if data_object.pbc.any():
                try:
                    data_object = self.radius_graph_pbc(data_object)
                    data_object = transform_coordinates_pbc(data_object)
                except Exception:
                    print(
                        "Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                        flush=True,
                    )
                    data_object = self.radius_graph(data_object)
                    data_object = transform_coordinates(data_object)
            else:
                data_object = self.radius_graph(data_object)
                data_object = transform_coordinates(data_object)

            if not hasattr(data_object, "edge_shifts"):
                data_object.edge_shifts = torch.zeros(
                    (data_object.edge_index.size(1), 3), dtype=torch.float32
                )

            data_object.pbc = data_object.pbc.int()

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
            print(
                f"Rank {self.rank} reading - exception: {e}\nTraceback: {traceback.format_exc()}",
                flush=True,
            )

    def check_forces_values(self, forces):
        norms = torch.norm(forces, p=2, dim=1)
        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
