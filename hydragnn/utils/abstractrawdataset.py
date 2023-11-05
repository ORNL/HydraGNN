import os
import numpy as np
import random

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.transforms import (
    Distance,
    NormalizeRotation,
    Spherical,
    PointPairFeatures,
)

from hydragnn.utils import nsplit, tensor_divide, comm_reduce
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.distributed import get_device
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.utils import (
    get_radius_graph,
    get_radius_graph_pbc,
    get_radius_graph_config,
    get_radius_graph_pbc_config,
)
from hydragnn.preprocess import (
    update_predicted_values,
    update_atom_features,
    stratified_sampling,
)

from sklearn.model_selection import StratifiedShuffleSplit

from hydragnn.preprocess.dataset_descriptors import AtomFeatures

from abc import ABC, abstractmethod


class AbstractRawDataset(AbstractBaseDataset, ABC):
    """Raw dataset class"""

    def __init__(self, config, dist=False, sampling=None):
        super().__init__()

        """
        config:
          shows the dataset path the target variables information, e.g, location and dimension, in data file
        ###########
        dataset_list:
          list of datasets read from self.path_dictionary
        serial_data_name_list:
          list of pkl file names
        node_feature_dim:
          list of dimensions of node features
        node_feature_col:
          list of column location/index (start location if dim>1) of node features
        graph_feature_dim:
          list of dimensions of graph features
        graph_feature_col: list,
          list of column location/index (start location if dim>1) of graph features

        dist: True if RawDataLoder is distributed (i.e., each RawDataLoader will read different subset of data)
        """

        # self.serial_data_name_list = []
        self.normalize_features = (
            config["Dataset"]["normalize_features"]
            if config["Dataset"]["normalize_features"] is not None
            else False
        )
        self.node_feature_name = (
            config["Dataset"]["node_features"]["name"]
            if config["Dataset"]["node_features"]["name"] is not None
            else None
        )
        self.node_feature_dim = config["Dataset"]["node_features"]["dim"]
        self.node_feature_col = config["Dataset"]["node_features"]["column_index"]
        self.graph_feature_name = (
            config["Dataset"]["graph_features"]["name"]
            if config["Dataset"]["graph_features"]["name"] is not None
            else None
        )
        self.graph_feature_dim = config["Dataset"]["graph_features"]["dim"]
        self.graph_feature_col = config["Dataset"]["graph_features"]["column_index"]
        self.raw_dataset_name = config["Dataset"]["name"]
        self.data_format = config["Dataset"]["format"]
        self.path_dictionary = config["Dataset"]["path"]

        assert len(self.node_feature_name) == len(self.node_feature_dim)
        assert len(self.node_feature_name) == len(self.node_feature_col)
        assert len(self.graph_feature_name) == len(self.graph_feature_dim)
        assert len(self.graph_feature_name) == len(self.graph_feature_col)

        self.sampling = sampling
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.__load_raw_data()

        ## SerializedDataLoader
        self.verbosity = config["Verbosity"]["level"]
        self.rotational_invariance = config["Dataset"]["rotational_invariance"]
        self.periodic_boundary_conditions = config["NeuralNetwork"]["Architecture"][
            "periodic_boundary_conditions"
        ]
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.variables = config["NeuralNetwork"]["Variables_of_interest"]
        self.variables_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
        self.output_index = config["NeuralNetwork"]["Variables_of_interest"][
            "output_index"
        ]
        self.input_node_features = config["NeuralNetwork"]["Variables_of_interest"][
            "input_node_features"
        ]

        self.spherical_coordinates = False
        self.point_pair_features = False

        if "Descriptors" in config["Dataset"]:
            if "SphericalCoordinates" in config["Dataset"]["Descriptors"]:
                self.spherical_coordinates = config["Dataset"]["Descriptors"][
                    "SphericalCoordinates"
                ]
            if "PointPairFeatures" in config["Dataset"]["Descriptors"]:
                self.point_pair_features = config["Dataset"]["Descriptors"][
                    "PointPairFeatures"
                ]

        # Descriptors about topology of the local environment
        if self.spherical_coordinates and self.point_pair_features:
            # FIXME We need to a new function to include both spherical coordinates and point point features together as edge features
            # Each of the two transformation computes the distance between nodes, and adds it to the set of edge features
            # A naive simuktaneous utilization of both spherical coordinates and point point features together includes the distance multiple times in the edge-feature vector
            raise ValueError(
                "Spherical Coorindates and Point Pair Features cannot be used together in the current version of HydraGNN"
            )

        if self.spherical_coordinates:
            self.edge_feature_transform = Spherical(norm=False, cat=False)

        if self.point_pair_features:
            self.edge_feature_transform = PointPairFeatures(cat=False)

        self.subsample_percentage = None

        self.__build_edge()

    def __load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        # serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        # if not os.path.exists(serialized_dir):
        #     os.makedirs(serialized_dir, exist_ok=True)

        for dataset_type, raw_data_path in self.path_dictionary.items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)

            assert (
                len(os.listdir(raw_data_path)) > 0
            ), "No data files provided in {}!".format(raw_data_path)

            filelist = sorted(os.listdir(raw_data_path))
            if self.dist:
                ## Random shuffle filelist to avoid the same test/validation set
                random.seed(43)
                random.shuffle(filelist)
                if self.sampling is not None:
                    filelist = np.random.choice(
                        filelist, int(len(filelist) * self.sampling)
                    )

                x = torch.tensor(len(filelist), requires_grad=False).to(get_device())
                y = x.clone().detach().requires_grad_(False)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
                assert x == y
                filelist = list(nsplit(filelist, self.world_size))[self.rank]
                log("local filelist", len(filelist))

            for name in iterate_tqdm(filelist, verbosity_level=2, desc="Local files"):
                if name == ".DS_Store":
                    continue
                # if the directory contains file, iterate over them
                if os.path.isfile(os.path.join(raw_data_path, name)):
                    data_object = self.transform_input_to_data_object_base(
                        filepath=os.path.join(raw_data_path, name)
                    )
                    if not isinstance(data_object, type(None)):
                        self.dataset.append(data_object)
                # if the directory contains subdirectories, explore their content
                elif os.path.isdir(os.path.join(raw_data_path, name)):
                    dir_name = os.path.join(raw_data_path, name)
                    for subname in os.listdir(dir_name):
                        if os.path.isfile(os.path.join(dir_name, subname)):
                            data_object = self.transform_input_to_data_object_base(
                                filepath=os.path.join(dir_name, subname)
                            )
                            if not isinstance(data_object, type(None)):
                                self.dataset.append(data_object)
            if self.dist:
                torch.distributed.barrier()

        # scaled features by number of nodes
        self.__scale_features_by_num_nodes()

        if self.normalize_features:
            self.__normalize_dataset()

    def __normalize_dataset(self):

        """Performs the normalization on Data objects and returns the normalized dataset."""
        num_node_features = len(self.node_feature_dim)
        num_graph_features = len(self.graph_feature_dim)

        self.minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        self.minmax_node_feature = np.full((2, num_node_features), np.inf)
        self.minmax_graph_feature[1, :] *= -1
        self.minmax_node_feature[1, :] *= -1
        for data in self.dataset:
            # find maximum and minimum values for graph level features
            g_index_start = 0
            for ifeat in range(num_graph_features):
                g_index_end = g_index_start + self.graph_feature_dim[ifeat]
                self.minmax_graph_feature[0, ifeat] = min(
                    torch.min(data.y[g_index_start:g_index_end]),
                    self.minmax_graph_feature[0, ifeat],
                )
                self.minmax_graph_feature[1, ifeat] = max(
                    torch.max(data.y[g_index_start:g_index_end]),
                    self.minmax_graph_feature[1, ifeat],
                )
                g_index_start = g_index_end

            # find maximum and minimum values for node level features
            n_index_start = 0
            for ifeat in range(num_node_features):
                n_index_end = n_index_start + self.node_feature_dim[ifeat]
                self.minmax_node_feature[0, ifeat] = min(
                    torch.min(data.x[:, n_index_start:n_index_end]),
                    self.minmax_node_feature[0, ifeat],
                )
                self.minmax_node_feature[1, ifeat] = max(
                    torch.max(data.x[:, n_index_start:n_index_end]),
                    self.minmax_node_feature[1, ifeat],
                )
                n_index_start = n_index_end

        ## Gather minmax in parallel
        if self.dist:
            self.minmax_graph_feature[0, :] = comm_reduce(
                self.minmax_graph_feature[0, :], torch.distributed.ReduceOp.MIN
            )
            self.minmax_graph_feature[1, :] = comm_reduce(
                self.minmax_graph_feature[1, :], torch.distributed.ReduceOp.MAX
            )
            self.minmax_node_feature[0, :] = comm_reduce(
                self.minmax_node_feature[0, :], torch.distributed.ReduceOp.MIN
            )
            self.minmax_node_feature[1, :] = comm_reduce(
                self.minmax_node_feature[1, :], torch.distributed.ReduceOp.MAX
            )

        for data in self.dataset:
            g_index_start = 0
            for ifeat in range(num_graph_features):
                g_index_end = g_index_start + self.graph_feature_dim[ifeat]
                data.y[g_index_start:g_index_end] = tensor_divide(
                    (
                        data.y[g_index_start:g_index_end]
                        - self.minmax_graph_feature[0, ifeat]
                    ),
                    (
                        self.minmax_graph_feature[1, ifeat]
                        - self.minmax_graph_feature[0, ifeat]
                    ),
                )
                g_index_start = g_index_end
            n_index_start = 0
            for ifeat in range(num_node_features):
                n_index_end = n_index_start + self.node_feature_dim[ifeat]
                data.x[:, n_index_start:n_index_end] = tensor_divide(
                    (
                        data.x[:, n_index_start:n_index_end]
                        - self.minmax_node_feature[0, ifeat]
                    ),
                    (
                        self.minmax_node_feature[1, ifeat]
                        - self.minmax_node_feature[0, ifeat]
                    ),
                )
                n_index_start = n_index_end

    @abstractmethod
    def transform_input_to_data_object_base(self, filepath):
        pass

    def __scale_features_by_num_nodes(self):
        """Calculate [**]_scaled_num_nodes"""
        scaled_graph_feature_index = [
            i
            for i in range(len(self.graph_feature_name))
            if "_scaled_num_nodes" in self.graph_feature_name[i]
        ]
        scaled_node_feature_index = [
            i
            for i in range(len(self.node_feature_name))
            if "_scaled_num_nodes" in self.node_feature_name[i]
        ]

        for idx, data_object in enumerate(self.dataset):
            if self.dataset[idx].y is not None:
                self.dataset[idx].y[scaled_graph_feature_index] = (
                    self.dataset[idx].y[scaled_graph_feature_index]
                    / data_object.num_nodes
                )
            if self.dataset[idx].x is not None:
                self.dataset[idx].x[:, scaled_node_feature_index] = (
                    self.dataset[idx].x[:, scaled_node_feature_index]
                    / data_object.num_nodes
                )

    def __build_edge(self):
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

        rotational_invariance = NormalizeRotation(max_points=-1, sort=False)
        if self.rotational_invariance:
            self.dataset[:] = [rotational_invariance(data) for data in self.dataset]

        if self.periodic_boundary_conditions:
            # edge lengths already added manually if using PBC, so no need to call Distance.
            compute_edges = get_radius_graph_pbc(
                radius=self.radius,
                loop=False,
                max_neighbours=self.max_neighbours,
            )
        else:
            compute_edges = get_radius_graph(
                radius=self.radius,
                loop=False,
                max_neighbours=self.max_neighbours,
            )
            compute_edge_lengths = Distance(norm=False, cat=True)

        self.dataset[:] = [compute_edges(data) for data in self.dataset]

        #################################
        #### COMPUTE EDGE ATTRIBUTES ####
        #################################

        # edge lengths already added manually if using PBC.
        # if spherical coordinates or pair point is set up, then skip directly to edge_transformation
        if (not self.periodic_boundary_conditions) and (
            not hasattr(self, "edge_feature_transform")
        ):
            self.dataset[:] = [compute_edge_lengths(data) for data in self.dataset]

            max_edge_length = torch.Tensor([float("-inf")])

            for data in self.dataset:
                max_edge_length = torch.max(max_edge_length, torch.max(data.edge_attr))

            if self.dist:
                ## Gather max in parallel
                device = max_edge_length.device
                max_edge_length = max_edge_length.to(get_device())
                torch.distributed.all_reduce(
                    max_edge_length, op=torch.distributed.ReduceOp.MAX
                )
                max_edge_length = max_edge_length.to(device)

            # Normalization of the edges
            for data in self.dataset:
                data.edge_attr = data.edge_attr / max_edge_length

        # Descriptors about topology of the local environment
        elif hasattr(self, "edge_feature_transform"):
            self.dataset[:] = [
                self.edge_feature_transform(data) for data in self.dataset
            ]

        if "subsample_percentage" in self.variables.keys():
            self.subsample_percentage = self.variables["subsample_percentage"]
            sampled = stratified_sampling(
                dataset=self.dataset, subsample_percentage=self.subsample_percentage
            )
            self.dataset = sampled

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
