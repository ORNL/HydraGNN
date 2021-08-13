import os
import numpy as np
import pickle
import pathlib

import torch
from torch_geometric.data import Data
from torch import tensor

from data_utils.helper_functions import tensor_divide


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data(dataset_path: str)
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def load_raw_data(self, dataset_path: str, config):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.

        Parameters
        ----------
        dataset_path: str
            Directory path where raw files are stored.
        config: shows the target variables information, e.g, location and dimension, in data file
        """

        node_feature_dim = config["node_features"]["dim"]
        node_feature_col = config["node_features"]["column_index"]
        graph_feature_dim = config["graph_features"]["dim"]
        graph_feature_col = config["graph_features"]["column_index"]

        dataset = []
        for filename in os.listdir(dataset_path):
            f = open(dataset_path + filename, "r")
            all_lines = f.readlines()
            data_object = self.__transform_input_to_data_object_base(
                lines=all_lines,
                node_feature_dim=node_feature_dim,
                node_feature_col=node_feature_col,
                graph_feature_dim=graph_feature_dim,
                graph_feature_col=graph_feature_col,
            )
            dataset.append(data_object)
            f.close()

        if config["format"] == "LSMS":
            for idx, data_object in enumerate(dataset):
                dataset[idx] = self.__charge_density_update_for_LSMS(data_object)

        dataset_normalized, x_minmax, y_minmax = self.__normalize_dataset(
            dataset=dataset
        )

        serial_data_name = (pathlib.PurePath(dataset_path)).parent.name
        serial_data_path = (
            os.environ["SERIALIZED_DATA_PATH"]
            + "/serialized_dataset/"
            + serial_data_name
            + ".pkl"
        )

        with open(serial_data_path, "wb") as f:
            pickle.dump(x_minmax, f)
            pickle.dump(y_minmax, f)
            pickle.dump(dataset_normalized, f)

    def __transform_input_to_data_object_base(
        self,
        lines: [str],
        node_feature_dim: list,
        node_feature_col: list,
        graph_feature_dim: list,
        graph_feature_col: list,
    ):
        """Transforms lines of strings read from the raw data file to Data object and returns it.

        Parameters
        ----------
        dataset_path: str
            Directory path where raw files are stored.

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        data_object = Data()

        graph_feat = lines[0].split(None, 2)
        g_feature = []
        # collect graph features
        for item in range(len(graph_feature_dim)):
            for icomp in range(graph_feature_dim[item]):
                it_comp = graph_feature_col[item] + icomp
                g_feature.append(float(graph_feat[it_comp].strip()))
        data_object.y = tensor([g_feature])

        node_feature_matrix = []
        node_position_matrix = []
        for line in lines[1:]:
            node_feat = line.split(None, 11)

            x_pos = float(node_feat[2].strip())
            y_pos = float(node_feat[3].strip())
            z_pos = float(node_feat[4].strip())
            node_position_matrix.append([x_pos, y_pos, z_pos])

            node_feature = []
            for item in range(len(node_feature_dim)):
                for icomp in range(node_feature_dim[item]):
                    it_comp = node_feature_col[item] + icomp
                    node_feature.append(float(node_feat[it_comp].strip()))

            node_feature_matrix.append(node_feature)

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)

        return data_object

    def __charge_density_update_for_LSMS(self, data_object: Data):
        """Calculate charge density for LSMS format
        Parameters
        ----------
        data_object: Data
            Data object representing structure of a graph sample.

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        num_of_protons = data_object.x[0]
        charge_density = data_object.x[1]
        charge_density -= num_of_protons
        data_object.x[1] = charge_density
        return data_object

    def __normalize_dataset(self, dataset: [Data]):
        """Performs the normalization on Data objects and returns the normalized dataset.

        Parameters
        ----------
        dataset: [Data]
            List of Data objects representing structures of atoms.

        Returns
        ----------
        [Data]
            Normalized dataset.
        """
        num_of_atoms = len(dataset[0].x)
        max_structure_free_energy = float("-inf")
        min_structure_free_energy = float("inf")
        max_structure_charge_density = float("-inf")
        min_structure_charge_density = float("inf")
        max_structure_magnetic_moment = float("-inf")
        min_structure_magnetic_moment = float("inf")
        max_charge_density = np.full(num_of_atoms, -np.inf)
        min_charge_density = np.full(num_of_atoms, np.inf)
        max_magnetic_moment = np.full(num_of_atoms, -np.inf)
        min_magnetic_moment = np.full(num_of_atoms, np.inf)

        # the minimum and maximum data used for normalization
        x_minmax = np.zeros((2, num_of_atoms, len(dataset[0].x[0, :])))
        y_minmax = np.zeros((2, len(dataset[0].y)))
        x_minmax[1, :, :] = 1.0
        y_minmax[1, :] = 1.0

        for data in dataset:
            max_structure_free_energy = max(abs(data.y[0]), max_structure_free_energy)
            min_structure_free_energy = min(abs(data.y[0]), min_structure_free_energy)
            max_charge_density = np.maximum(data.x[:, 1].numpy(), max_charge_density)
            min_charge_density = np.minimum(data.x[:, 1].numpy(), min_charge_density)
            max_magnetic_moment = np.maximum(data.x[:, 2].numpy(), max_magnetic_moment)
            min_magnetic_moment = np.minimum(data.x[:, 2].numpy(), min_magnetic_moment)
        for data in dataset:
            data.y[0] = tensor_divide(
                (abs(data.y[0]) - min_structure_free_energy),
                (max_structure_free_energy - min_structure_free_energy),
            )
            data.x[:, 1] = tensor_divide(
                (data.x[:, 1] - min_charge_density),
                (max_charge_density - min_charge_density),
            )
            data.x[:, 2] = tensor_divide(
                (data.x[:, 2] - min_magnetic_moment),
                (max_magnetic_moment - min_magnetic_moment),
            )
        x_minmax[0, :, 1] = min_charge_density
        x_minmax[1, :, 1] = max_charge_density
        x_minmax[0, :, 2] = min_magnetic_moment
        x_minmax[1, :, 2] = max_magnetic_moment
        y_minmax[0, 0] = min_structure_free_energy
        y_minmax[1, 0] = max_structure_free_energy

        return dataset, x_minmax, y_minmax
