import os
from torch_geometric.data import Data
from torch import tensor
import numpy as np
import pickle
import pathlib
from dataset_descriptors import AtomFeatures, StructureFeatures


class SerializedDataLoader:
    def load_serialized_data(
        self,
        dataset_path: str,
        atom_features: [AtomFeatures],
        structure_features: [StructureFeatures],
        radius: float,
        max_num_node_neighbours: int,
    ):
        dataset = []
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        # Computing adjacency matrix which is the same for each data in the dataset since their
        # x,y,z coordinates are the same. Thus we are computing it only with the first element
        # of the dataset(dataset[0]).
        adjacency_matrix = self.__compute_adjacency_matrix(
            self,
            data=dataset[0],
            radius=radius,
            max_num_node_neighbours=max_num_node_neighbours,
        )

        for data in dataset:
            self.__update_atom_features(self, atom_features, data)
            self.__update_structure_features(self, structure_features, data)

        return dataset

    def __update_atom_features(self, atom_features: [AtomFeatures], data: Data):

        feature_indices = [i.value for i in atom_features]
        data.x = data.x[:, feature_indices]

    def __update_structure_features(
        self, structure_features: [StructureFeatures], data: Data
    ):

        feature_indices = [i.value for i in structure_features]
        data.y = data.y[:, feature_indices]

    def __compute_adjacency_matrix(
        self, data: Data, radius: float, max_num_node_neighbours: int
    ):
        pass
