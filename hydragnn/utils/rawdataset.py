import os
import numpy as np
from mpi4py import MPI
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

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.distributed import get_device
from hydragnn.utils.basedataset import BaseDataset
from hydragnn.preprocess.utils import (
    get_radius_graph,
    get_radius_graph_pbc,
    get_radius_graph_config,
    get_radius_graph_pbc_config,
)
from hydragnn.preprocess.serialized_dataset_loader import update_predicted_values

from ase.io.cfg import read_cfg
from ase.io import read

from sklearn.model_selection import StratifiedShuffleSplit

from hydragnn.preprocess.dataset_descriptors import AtomFeatures

from abc import ABC, abstractmethod


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


## All-reduce with numpy array
def comm_reduce(x, op):
    tx = torch.tensor(x, requires_grad=True).to(get_device())
    torch.distributed.all_reduce(tx, op=op)
    y = tx.detach().cpu().numpy()
    return y


from abc import ABC, abstractmethod


class RawDataset(BaseDataset, ABC):
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

        self.__load_raw_data(sampling=self.sampling)

        ## SerializedDataLoader
        self.verbosity = config["Verbosity"]["level"]
        self.node_feature_name = config["Dataset"]["node_features"]["name"]
        self.node_feature_dim = config["Dataset"]["node_features"]["dim"]
        self.node_feature_col = config["Dataset"]["node_features"]["column_index"]
        self.graph_feature_name = config["Dataset"]["graph_features"]["name"]
        self.graph_feature_dim = config["Dataset"]["graph_features"]["dim"]
        self.graph_feature_col = config["Dataset"]["graph_features"]["column_index"]
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

        self.subsample_percentage = None

        # In situations where someone already provides the .pkl filed with data
        # the asserts from raw_dataset_loader are not performed
        # Therefore, we need to re-check consistency
        assert len(self.node_feature_name) == len(self.node_feature_dim)
        assert len(self.node_feature_name) == len(self.node_feature_col)
        assert len(self.graph_feature_name) == len(self.graph_feature_dim)
        assert len(self.graph_feature_name) == len(self.graph_feature_col)

        self.__load_serialized_data()

    def __load_raw_data(self, sampling=None):
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
                if sampling is not None:
                    filelist = np.random.choice(filelist, int(len(filelist) * sampling))

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
            torch.distributed.barrier()

            if self.data_format == "LSMS":
                for idx, data_object in enumerate(self.dataset):
                    self.dataset[idx] = self.__charge_density_update_for_LSMS(
                        data_object
                    )

        # scaled features by number of nodes
        self.__scale_features_by_num_nodes()

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
        num_of_protons = data_object.x[:, 0]
        charge_density = data_object.x[:, 1]
        charge_density -= num_of_protons
        data_object.x[:, 1] = charge_density
        return data_object

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

    def __load_serialized_data(self):
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

        # edge lengths already added manually if using PBC.
        if not self.periodic_boundary_conditions:
            compute_edge_lengths = Distance(norm=False, cat=True)
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
        for data in self.dataset:
            if self.spherical_coordinates:
                data = Spherical(data)
            if self.point_pair_features:
                data = PointPairFeatures(data)

        # Move data to the device, if used. # FIXME: this does not respect the choice set by use_gpu
        device = get_device(verbosity_level=self.verbosity)
        for data in self.dataset:
            update_predicted_values(
                self.variables_type,
                self.output_index,
                self.graph_feature_dim,
                self.node_feature_dim,
                data,
            )

            self.__update_atom_features(self.input_node_features, data)

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


def stratified_sampling(dataset: [Data], subsample_percentage: float, verbosity=0):
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
    dataset_categories = []
    print_distributed(verbosity, "Computing the categories for the whole dataset.")
    for data in iterate_tqdm(dataset, verbosity):
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

    for subsample_index, rest_of_data_index in sss.split(dataset, dataset_categories):
        subsample_indices = subsample_index.tolist()

    for index in subsample_indices:
        subsample.append(dataset[index])

    return subsample


class LSMSDataset(RawDataset):
    def __init__(self, config, dist=False, sampling=None):
        super().__init__(config, dist, sampling)

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_LSMS_input_to_data_object_base(filepath=filepath)

        return data_object

    def __transform_LSMS_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        data_object = Data()

        f = open(filepath, "r", encoding="utf-8")

        lines = f.readlines()
        graph_feat = lines[0].split(None, 2)
        g_feature = []
        # collect graph features
        for item in range(len(self.graph_feature_dim)):
            for icomp in range(self.graph_feature_dim[item]):
                it_comp = self.graph_feature_col[item] + icomp
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
            for item in range(len(self.node_feature_dim)):
                for icomp in range(self.node_feature_dim[item]):
                    it_comp = self.node_feature_col[item] + icomp
                    node_feature.append(float(node_feat[it_comp].strip()))
            node_feature_matrix.append(node_feature)

        f.close()

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)
        data_object = self.__charge_density_update_for_LSMS(data_object)
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
        num_of_protons = data_object.x[:, 0]
        charge_density = data_object.x[:, 1]
        charge_density -= num_of_protons
        data_object.x[:, 1] = charge_density
        return data_object


class CFGDataset(RawDataset):
    def __init__(self, config, dist=False, sampling=None):
        super().__init__(config, dist, sampling)

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_CFG_input_to_data_object_base(filepath=filepath)
        return data_object

    def __transform_CFG_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data CFG file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        if filepath.endswith(".cfg"):

            data_object = self.__transform_ASE_object_to_data_object(filepath)

            return data_object

        else:
            return None

    def __transform_ASE_object_to_data_object(self, filepath):

        # FIXME:
        #  this still assumes bulk modulus is specific to the CFG format.
        #  To deal with multiple files across formats, one should generalize this function
        #  by moving the reading of the .bulk file in a standalone routine.
        #  Morevoer, this approach assumes tha there is only one global feature to look at,
        #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

        ase_object = read_cfg(filepath)

        data_object = Data()

        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
        c_peratom = np.expand_dims(ase_object.arrays["c_peratom"], axis=1)
        fx = np.expand_dims(ase_object.arrays["fx"], axis=1)
        fy = np.expand_dims(ase_object.arrays["fy"], axis=1)
        fz = np.expand_dims(ase_object.arrays["fz"], axis=1)
        node_feature_matrix = np.concatenate(
            (proton_numbers, masses, c_peratom, fx, fy, fz), axis=1
        )
        data_object.x = tensor(node_feature_matrix).float()

        filename_without_extension = os.path.splitext(filepath)[0]

        if os.path.exists(os.path.join(filename_without_extension + ".bulk")):
            filename_bulk = os.path.join(filename_without_extension + ".bulk")
            f = open(filename_bulk, "r", encoding="utf-8")
            lines = f.readlines()
            graph_feat = lines[0].split(None, 2)
            g_feature = []
            # collect graph features
            for item in range(len(self.graph_feature_dim)):
                for icomp in range(self.graph_feature_dim[item]):
                    it_comp = self.graph_feature_col[item] + icomp
                    g_feature.append(float(graph_feat[it_comp].strip()))
            data_object.y = tensor(g_feature)

        return data_object


class XYZDataset(RawDataset):
    def __init__(self, config, dist=False, sampling=None):
        super().__init__(config, dist, sampling)

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_XYZ_input_to_data_object_base(filepath=filepath)
        return data_object

    def __transform_XYZ_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data XYZ file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        if filepath.endswith(".xyz"):

            data_object = self.__transform_XYZ_ASE_object_to_data_object(filepath)

            return data_object

        else:
            return None
