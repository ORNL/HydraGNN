import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch_geometric.data import Data

from data_utils.dataset_descriptors import AtomFeatures
from data_utils.helper_functions import (
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
            x_minmax = pickle.load(f)
            y_minmax = pickle.load(f)
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

        if "subsample_percentage" in config.keys():
            return self.__stratified_sampling(
                dataset=dataset, subsample_percentage=config["subsample_percentage"]
            )

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
            data.y = torch.reshape(data.x[:, 2], (len(data.x), 1))
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

    def __stratified_sampling(self, dataset: [Data], subsample_percentage: float):
        """Given the dataset and the percentage of data you want to extract from it, method will
        apply stratified sampling where X is the dataset and Y is are the category values for each datapoint.
        In the case of the structures dataset where each structure contains 2 types of atoms, the category will
        be constructed in a way: number of atoms of type 1 + number of protons of type 2 * 100.

        Parameters
        ----------
        dataset: [Data]
            A list of Data objects representing a structure that has atoms.
        subsample_percentage: float
            Percentage of the dataset.

        Returns
        ----------
        [Data]
            Subsample of the original dataset constructed using stratified sampling.
        """
        unique_values = torch.unique(dataset[0].x[:, 0]).tolist()
        dataset_categories = []
        print("Computing the categories for the whole dataset.")
        for data in tqdm(dataset):
            frequencies = torch.bincount(data.x[:, 0].int())
            frequencies = sorted(frequencies[frequencies > 0].tolist())
            category = 0
            for index, frequency in enumerate(frequencies):
                category += frequency * (100 ** index)
            dataset_categories.append(category)

        subsample_indices = []
        subsample = []

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=subsample_percentage, random_state=0
        )

        for subsample_index, rest_of_data_index in sss.split(
            dataset, dataset_categories
        ):
            subsample_indices = subsample_index.tolist()

        for index in subsample_indices:
            subsample.append(dataset[index])

        return subsample
