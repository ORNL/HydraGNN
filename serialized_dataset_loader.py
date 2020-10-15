import os
from torch_geometric.data import Data
from torch import tensor
import numpy as np
import pickle
import pathlib
from dataset_descriptors import (AtomFeatures, StructureFeatures)
from utils import (distance_3D, remove_collinear_candidates, order_candidates)


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
        edge_index = self.__compute_edges(
            self,
            data=dataset[0],
            radius=radius,
            max_num_node_neighbours=max_num_node_neighbours,
        )

        for data in dataset:
            data.edge_index = edge_index
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

    def __compute_edges(
        self, data: Data, radius: float, max_num_node_neighbours: int
    ):  
        distance_matrix = np.zeros((StructureFeatures.SIZE.value, StructureFeatures.SIZE.value))
        candidate_neighbours = {k: [] for k in range(StructureFeatures.SIZE.value)}

        for i in range(StructureFeatures.SIZE.value):
            for j in range(StructureFeatures.SIZE.value):
                distance = distance_3D(data.pos[i], data.pos[j])
                distance_matrix[i, j] = distance
                if distance_matrix[i, j] <= radius and i!=j:
                    candidate_neighbours[i].append(j)

        
        ordered_candidate_neighbours = order_candidates(candidate_neighbours=candidate_neighbours, distance_matrix=distance_matrix)
        collinear_neighbours = remove_collinear_candidates(candidate_neighbours=ordered_candidate_neighbours, distance_matrix=distance_matrix)

        adjacency_matrix = np.zeros((StructureFeatures.SIZE.value, StructureFeatures.SIZE.value))
        for point, neighbours in ordered_candidate_neighbours.items():
            neighbours = list(neighbours)
            if point in collinear_neighbours.keys():
                collinear_points = list(collinear_neighbours[point])
                neighbours = [x for x in neighbours if x not in collinear_points]
            if len(neighbours)>max_num:
                neighbours = neighbours[:max_num]
            adjacency_matrix[point, neighbours] = 1


        return torch.tensor(np.nonzero(adjacency_matrix))
