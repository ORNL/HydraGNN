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

        (
            dataset_normalized,
            minmax_node_feature,
            minmax_graph_feature,
        ) = self.__normalize_dataset(dataset=dataset)

        serial_data_name = (pathlib.PurePath(dataset_path)).parent.name
        serial_data_path = (
            os.environ["SERIALIZED_DATA_PATH"]
            + "/serialized_dataset/"
            + serial_data_name
            + ".pkl"
        )

        with open(serial_data_path, "wb") as f:
            pickle.dump(minmax_node_feature, f)
            pickle.dump(minmax_graph_feature, f)
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
        lines:
          content of data file with all the graph information
        node_feature_dim:
          list of dimensions of node features
        node_feature_col:
          list of column location/index (start location if dim>1) of node features
        graph_feature_dim:
          list of dimensions of graph features
        graph_feature_col: list,
          list of column location/index (start location if dim>1) of graph features

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
        data_object.y = tensor(g_feature)

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
            List of Data objects representing structures of graphs.

        Returns
        ----------
        [Data]
            Normalized dataset.
        """
        num_of_nodes = len(dataset[0].x)
        num_node_features = dataset[0].x.shape[1]
        num_graph_features = len(dataset[0].y)

        minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        minmax_node_feature = np.full((2, num_of_nodes, num_node_features), np.inf)
        minmax_graph_feature[1, :] *= -1
        minmax_node_feature[1, :, :] *= -1

        for data in dataset:
            # find maximum and minimum values for graph level features
            for ifeat in range(num_graph_features):
                minmax_graph_feature[0, ifeat] = min(
                    data.y[ifeat], minmax_graph_feature[0, ifeat]
                )
                minmax_graph_feature[1, ifeat] = max(
                    data.y[ifeat], minmax_graph_feature[1, ifeat]
                )
            # find maximum and minimum values for node level features
            for ifeat in range(num_node_features):
                minmax_node_feature[0, :, ifeat] = np.minimum(
                    data.x[:, ifeat].numpy(), minmax_node_feature[0, :, ifeat]
                )
                minmax_node_feature[1, :, ifeat] = np.maximum(
                    data.x[:, ifeat].numpy(), minmax_node_feature[1, :, ifeat]
                )

        for data in dataset:
            for ifeat in range(num_graph_features):
                data.y[ifeat] = tensor_divide(
                    (data.y[ifeat] - minmax_graph_feature[0, ifeat]),
                    (minmax_graph_feature[1, ifeat] - minmax_graph_feature[0, ifeat]),
                )
            for ifeat in range(num_node_features):
                data.x[:, ifeat] = tensor_divide(
                    (data.x[:, 1] - minmax_node_feature[0, :, ifeat]),
                    (
                        minmax_node_feature[1, :, ifeat]
                        - minmax_node_feature[0, :, ifeat]
                    ),
                )

        return dataset, minmax_node_feature, minmax_graph_feature
