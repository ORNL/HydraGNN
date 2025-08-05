"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import tqdm

import ase.db.sqlite
import ase.io.trajectory
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, Spherical, LocalCartesian

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    PBCLocalCartesian,
    pbc_as_tensor,
)

# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_pbc (bool): Return the periodic boundary conditions with other properties.
        Default is False, so the periodic boundary conditions will not be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_pbc (bool): Return the periodic boundary conditions with other properties.
        Default is False, so the periodic boundary conditions will not be returned.

    """

    def __init__(
        self,
        max_neigh=200,
        radius=6.0,
    ):
        self.max_neigh = max_neigh
        self.radius = radius

        # NOTE Open Catalyst 2020 dataset has PBC:
        #      https://pubs.acs.org/doi/10.1021/acscatal.0c04525#_i3 (Section 2: Tasks, paragraph 2)
        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neigh
        )
        self.radius_graph_pbc = RadiusGraphPBC(
            self.radius, loop=False, max_num_neighbors=self.max_neigh
        )

    def convert(
        self,
        atoms,
        energy_per_atom,
    ):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, atomic_numbers, tags,
            , energy, forces, and optionally periodic boundary conditions.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers()).unsqueeze(1)
        positions = torch.Tensor(atoms.get_positions())
        natoms = torch.IntTensor([positions.shape[0]])
        # initialized to torch.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = torch.Tensor(atoms.get_tags())

        cell = None
        try:
            cell = torch.Tensor(np.array(atoms.get_cell())).view(3, 3)
        except:
            print(f"Structure does not have cell", flush=True)

        pbc = None
        try:
            pbc = pbc_as_tensor(atoms.get_pbc())
        except:
            print(f"Structure does not have pbc", flush=True)

        # If either cell or pbc were not read, we set to defaults which are not none.
        if cell is None or pbc is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        energy = atoms.get_potential_energy(apply_constraint=False)
        energy_tensor = torch.tensor(energy).to(dtype=torch.float32).unsqueeze(0)
        energy_per_atom_tensor = energy_tensor.detach().clone() / natoms

        forces = torch.Tensor(atoms.get_forces(apply_constraint=False))

        x = torch.cat((atomic_numbers, positions, forces), dim=1)

        # put the minimum data in torch geometric data object
        data_object = Data(
            dataset_name="oc2020",
            natoms=natoms,
            pos=positions,
            cell=cell,
            pbc=pbc,
            edge_index=None,
            edge_attr=None,
            atomic_numbers=atomic_numbers,
            x=x,
            energy=energy_tensor,
            energy_per_atom=energy_per_atom_tensor,
            forces=forces,
            tags=tags,
        )

        if energy_per_atom:
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

        return data_object

    def convert_all(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list
