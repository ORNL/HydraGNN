import os, random, torch, glob, sys, shutil, pdb
import numpy as np
from fairchem.core.datasets import AseDBDataset
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


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


class OMol2025(AbstractBaseDataset):
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

        config_kwargs = {}  # see tutorial on additional configuration

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        self.comm = comm

        # get list of data files and distribute them evenly amongst ranks
        # omol25 has the same number of samples in every data file
        dataset_list = glob.glob(
            os.path.join(dirpath, data_type, "*.aselmdb"), recursive=True
        )

        for dataset in dataset_list:
            fullpath = dataset

            try:
                dataset = AseDBDataset(config=dict(src=fullpath, **config_kwargs))
            except ValueError as e:
                print(f"{fullpath} not a valid ase lmdb dataset. Ignoring ...")
                continue

            rx = list(nsplit(list(range(dataset.num_samples)), self.world_size))[
                self.rank
            ]

            print(
                f"Rank: {self.rank}, dataname: {fullpath}, data_type: {data_type}, num_samples: {dataset.num_samples}, len(rx): {len(rx)}"
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
        MPI.COMM_WORLD.Barrier()

        random.shuffle(self.dataset)

    def _create_pytorch_data_object(self, dataset, index):
        try:
            pos = torch.tensor(
                dataset.get_atoms(index).get_positions(), dtype=torch.float32
            )
            natoms = torch.IntTensor([pos.shape[0]])
            atomic_numbers = torch.tensor(
                dataset.get_atoms(index).get_atomic_numbers(),
                dtype=torch.float32,
            ).unsqueeze(1)

            energy = torch.tensor(
                dataset.get_atoms(index).get_total_energy(), dtype=torch.float32
            ).unsqueeze(0)

            energy_per_atom = energy.detach().clone() / natoms
            forces = torch.tensor(
                dataset.get_atoms(index).get_forces(), dtype=torch.float32
            )

            chemical_formula = dataset.get_atoms(index).get_chemical_formula()

            cell = None
            try:
                cell = torch.tensor(
                    dataset.get_atoms(index).get_cell(), dtype=torch.float32
                ).view(3, 3)
                cell = torch.from_numpy(
                    np.asarray(dataset.get_atoms(index).get_cell())
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
                pbc = pbc_as_tensor(dataset.get_atoms(index).get_pbc())
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
                dataset_name="omol25",
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

            # if not data_object:
            #     return
            # else:
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
