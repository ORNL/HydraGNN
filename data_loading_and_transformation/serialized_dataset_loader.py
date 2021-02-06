from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
)
from data_loading_and_transformation.utils import (
    distance_3D,
    remove_collinear_candidates,
    order_candidates,
    resolve_neighbour_conflicts,
)
from tqdm import tqdm


class SerializedDataLoader:
    """A class used for loading existing structures from files that are lists of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_serialized_data method.

    Methods
    -------
    load_serialized_data(dataset_path: str, config: dict)
        Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.
    """

    def load_serialized_data(
        self,
        dataset_path: str,
        config,
    ):
        """Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.

        Parameters
        ----------
        dataset_path: str
            Directory path where files containing serialized structures are stored.
        config: dict
            Dictionary containing information needed to load the data and transform it, respectively: atom_features, radius, max_num_node_neighbours and predicted_value_option.
        Returns
        ----------
        [Data]
            List of Data objects representing atom structures.
        """
        dataset = []
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        edge_index, edge_distances = self.__compute_edges(
            data=dataset[0],
            radius=config["radius"],
            max_num_node_neighbours=config["max_num_node_neighbours"],
        )

        for data in dataset:
            data.edge_index = edge_index
            data.edge_attr = edge_distances
            self.__update_predicted_values(config["predicted_value_option"], data)
            self.__update_atom_features(config["atom_features"], data)

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
        feature_indices = [i for i in atom_features]
        data.x = data.x[:, feature_indices]

    def __update_predicted_values(self, predicted_value_option: int, data: Data):
        """Updates values of the structure we want to predict. Predicted value is represented by integer value.

        Parameters
        ----------
        predicted_value_option: int
            Integer value that represents one of the options for predict. Possible values and associated output dimensions:
            1)free energy - 1
            2)charge density - 32
            3)magnetic moment - 32
            4)free energy+charge density - 33
            5)free energy+magnetic moment - 33
            6)free energy+charge density+magnetic moment - 65

        """
        free_energy = torch.reshape(data.y[0], (1, 1))
        charge_density = torch.reshape(data.x[:, 1], (len(data.x), 1))
        magnetic_moment = torch.reshape(data.x[:, 2], (len(data.x), 1))
        if predicted_value_option == 1:
            data.y = torch.reshape(data.y[0], (1, 1))
        elif predicted_value_option == 2:
            data.y = torch.reshape(data.x[:, 1], (len(data.x), 1))
        elif predicted_value_option == 3:
            data.y = torch.reshape(data.x[:, 1], (len(data.x), 1))
        elif predicted_value_option == 4:
            data.y = torch.cat([free_energy, charge_density], 0)
        elif predicted_value_option == 5:
            data.y = torch.cat([free_energy, magnetic_moment], 0)
        elif predicted_value_option == 6:
            data.y = torch.cat([free_energy, charge_density, magnetic_moment], 0)

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
        print("Compute edges of the structure=adjacency matrix.")
        num_of_atoms = len(data.x)
        distance_matrix = np.zeros((num_of_atoms, num_of_atoms))
        candidate_neighbours = {k: [] for k in range(num_of_atoms)}

        print("Computing edge distances and adding candidate neighbours.")
        for i in tqdm(range(num_of_atoms)):
            for j in range(num_of_atoms):
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

        print("Removing collinear neighbours and resolving neighbour conflicts.")
        adjacency_matrix = np.zeros((num_of_atoms, num_of_atoms))
        for point, neighbours in tqdm(ordered_candidate_neighbours.items()):
            neighbours = list(neighbours)
            if point in collinear_neighbours.keys():
                collinear_points = list(collinear_neighbours[point])
                neighbours = [x for x in neighbours if x not in collinear_points]

            neighbours = resolve_neighbour_conflicts(
                point, neighbours, adjacency_matrix, max_num_node_neighbours
            )
            adjacency_matrix[point, neighbours] = 1
            adjacency_matrix[neighbours, point] = 1

        edge_index = torch.tensor(np.nonzero(adjacency_matrix))
        edge_lengths = (
            torch.tensor(distance_matrix[np.nonzero(adjacency_matrix)])
            .reshape((edge_index.shape[1], 1))
            .type(torch.FloatTensor)
        )
        # Normalize the lengths using min-max normalization
        edge_lengths = (edge_lengths - min(edge_lengths)) / (
            max(edge_lengths) - min(edge_lengths)
        )

        return edge_index, edge_lengths
