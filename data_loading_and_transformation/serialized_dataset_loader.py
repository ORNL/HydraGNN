from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)
from data_loading_and_transformation.utils import (
    distance_3D,
    remove_collinear_candidates,
    order_candidates,
    resolve_neighbour_conflicts,
)


class SerializedDataLoader:
    """A class used for loading existing structures from files that are lists of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_serialized_data method.

    Methods
    -------
    load_serialized_data(dataset_path: str, atom_features: [AtomFeatures], structure_features: [StructureFeatures], radius: float, max_num_node_neighbours: int,)
        Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.
    """

    def load_serialized_data(
        self,
        dataset_path: str,
        atom_features: [AtomFeatures],
        structure_features: [StructureFeatures],
        radius: float,
        max_num_node_neighbours: int,
    ):
        """Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.

        Parameters
        ----------
        dataset_path: str
            Directory path where files containing serialized structures are stored.
        atom_features: [AtomFeatures]
            List of atom features that are preserved in the returned dataset.
        structure_features: [StructureFeatures]
            List of structure features that are preserved in the returned dataset
        radius: float
            Used when computing edges in the structure. Represents maximum distance of a neighbour atom from an atom.
        max_num_node_neighbours: int
            Used when computing edges in the structure. Represents maximum number of neighbours of an atom.

        Returns
        ----------
        [Data]
            List of Data objects representing atom structures.
        """
        dataset = []
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        edge_index = self.__compute_edges(
            data=dataset[0],
            radius=radius,
            max_num_node_neighbours=max_num_node_neighbours,
        )

        for data in dataset:
            data.edge_index = edge_index
            self.__update_atom_features(atom_features, data)
            self.__update_structure_features(structure_features, data)

        return dataset

    def __update_atom_features(self, atom_features: [AtomFeatures], data: Data):
        """Updates atom features of a structure. An atom is represented with x,y,z coordinates and associated features.

        Parameters
        ----------
        atom_features: [AtomFeatures]
            List of features to update. Each feature is instance of Enum AtomFeatures.
        data: Data
            A Data object representing a structure that has atoms.
        """
        feature_indices = [i.value for i in atom_features]
        data.x = data.x[:, feature_indices]

    def __update_structure_features(
        self, structure_features: [StructureFeatures], data: Data
    ):
        """Updates structure features. A structure is represented with the Data object.

        Parameters
        ----------
        structure_features: [StructureFeatures]
            List of features to update. Each feature is instance of Enum StructureFeatures.
        """

        feature_indices = [i.value for i in structure_features]
        data.y = data.y[feature_indices]

    def __compute_edges(self, data: Data, radius: float, max_num_node_neighbours: int):
        """Computes edges of a structure depending on the maximum number of neighbour atoms that each atom can have
        and radius as a maximum distance of a neighbour.

        Parameters
        ----------
        data: Data
            A Data object representing a structure that has atoms.
        radius: float
            Radius or maximum distance of a neighbour atom.
        max_num_node_neighbours: int
            Maximum number of neighbour atoms an atom can have.

        Returns
        ----------
        torch.tensor
            Tensor filled with pairs (atom1_index, atom2_index) that represent edges or connections between atoms within the structure.
        """
        distance_matrix = np.zeros(
            (StructureFeatures.SIZE.value, StructureFeatures.SIZE.value)
        )
        candidate_neighbours = {k: [] for k in range(StructureFeatures.SIZE.value)}

        for i in range(StructureFeatures.SIZE.value):
            for j in range(StructureFeatures.SIZE.value):
                distance = distance_3D(data.pos[i], data.pos[j])
                distance_matrix[i, j] = distance
                if distance_matrix[i, j] <= radius and i != j:
                    candidate_neighbours[i].append(j)

        ordered_candidate_neighbours = order_candidates(
            candidate_neighbours=candidate_neighbours, distance_matrix=distance_matrix
        )
        collinear_neighbours = remove_collinear_candidates(
            candidate_neighbours=ordered_candidate_neighbours,
            distance_matrix=distance_matrix,
        )

        adjacency_matrix = np.zeros(
            (StructureFeatures.SIZE.value, StructureFeatures.SIZE.value)
        )
        for point, neighbours in ordered_candidate_neighbours.items():
            neighbours = list(neighbours)
            if point in collinear_neighbours.keys():
                collinear_points = list(collinear_neighbours[point])
                neighbours = [x for x in neighbours if x not in collinear_points]

            neighbours = resolve_neighbour_conflicts(
                point, neighbours, adjacency_matrix, max_num_node_neighbours
            )
            adjacency_matrix[point, neighbours] = 1
            adjacency_matrix[neighbours, point] = 1

        return torch.tensor(np.nonzero(adjacency_matrix))