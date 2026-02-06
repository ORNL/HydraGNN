import glob
import os
import random
import sys
import traceback
import bisect
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

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    PBCDistance,
    RadiusGraph,
    RadiusGraphPBC,
    pbc_as_tensor,
    should_skip_self_loops,
)
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.distributed import nsplit
from hydragnn.utils.print.print_utils import iterate_tqdm

# Ensure the bundled fairchem package is importable when not installed system-wide.
dirpwd = os.path.dirname(os.path.abspath(__file__))
fairchem_path = os.path.join(dirpwd, "fairchem")
if fairchem_path not in sys.path:
    sys.path.insert(0, fairchem_path)

from fairchem.core.datasets import AseDBDataset  # noqa: E402

# Coordinate transforms
transform_coordinates = Distance(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


def _get_row_fields(dataset, index):
    try:
        db_idx = bisect.bisect(dataset._idlen_cumulative, index)
        el_idx = index if db_idx == 0 else index - dataset._idlen_cumulative[db_idx - 1]
        row = dataset.dbs[db_idx]._get_row(dataset.db_ids[db_idx][el_idx])

        def _decode_ndarray(value):
            if isinstance(value, dict) and "__ndarray__" in value:
                raw = value["__ndarray__"]
                try:
                    shape = raw[0]
                    dtype = raw[1]
                    flat = raw[2]
                    arr = np.asarray(flat, dtype=dtype)
                    return arr.reshape(shape)
                except Exception:
                    try:
                        return np.asarray(raw)
                    except Exception:
                        return value
            return value

        def _get_attr_or_container(key):
            if hasattr(row, key):
                return _decode_ndarray(getattr(row, key))
            for container in (
                getattr(row, "data", None),
                getattr(row, "key_value_pairs", None),
            ):
                if isinstance(container, dict) and key in container:
                    return _decode_ndarray(container.get(key))
            return None

        def _to_scalar(val):
            if val is None:
                return None
            try:
                arr = np.asarray(val, dtype=float).reshape(-1)
                if arr.size > 0:
                    return float(arr[0])
            except Exception:
                return None
            return None

        energy_value = _to_scalar(_get_attr_or_container("energy"))

        return {
            "numbers": _get_attr_or_container("numbers"),
            "positions": _get_attr_or_container("positions"),
            "cell": _get_attr_or_container("cell"),
            "pbc": _get_attr_or_container("pbc"),
            "forces": _get_attr_or_container("forces"),
            "energy": energy_value,
        }
    except Exception:
        return {}


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

        row_fields = _get_row_fields(dataset, index)

        positions = None
        try:
            positions = row_fields.get("positions")
            if positions is None:
                positions = atoms.get_positions()
            pos = torch.as_tensor(positions, dtype=torch.float32)
        except Exception:
            print(
                "Atomic structure does not have valid positions",
                flush=True,
            )
            return None

        natoms = torch.IntTensor([pos.shape[0]])

        numbers_value = None
        try:
            numbers_value = row_fields.get("numbers")
            if numbers_value is None:
                numbers_value = atoms.get_atomic_numbers()
            atomic_numbers = torch.as_tensor(
                numbers_value,
                dtype=torch.float32,
            ).unsqueeze(1)
        except Exception:
            print(
                "Atomic structure does not have valid atomic numbers",
                flush=True,
            )
            return None

        energy_value = None
        try:
            energy_value = row_fields.get("energy")
            if energy_value is None:
                raise ValueError("missing energy")
            energy = torch.as_tensor(energy_value, dtype=torch.float32).unsqueeze(0)
        except Exception:
            print(
                "Atomic structure does not have valid energy",
                flush=True,
            )
            return None

        energy_per_atom = energy.detach().clone() / natoms

        forces_value = None
        try:
            forces_value = row_fields.get("forces")
            if forces_value is None:
                raise ValueError("missing forces")
            forces = torch.as_tensor(forces_value, dtype=torch.float32)
        except Exception:
            print(
                "Atomic structure does not have valid forces",
                flush=True,
            )
            return None

        chemical_formula = atoms.get_chemical_formula()

        cell = None
        try:
            cell_value = row_fields.get("cell")
            if cell_value is None:
                raise ValueError("missing cell")
            cell = torch.from_numpy(np.asarray(cell_value)).to(torch.float32)
        except Exception:
            print(
                f"Atomic structure {chemical_formula} does not have cell",
                flush=True,
            )

        pbc = None
        try:
            pbc_value = row_fields.get("pbc")
            if pbc_value is None:
                raise ValueError("missing pbc")
            pbc = pbc_as_tensor(pbc_value)
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
        energy_per_atom=False,
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
                        if should_skip_self_loops(data_obj, context="oc25"):
                            continue
                        self.dataset.append(data_obj)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
