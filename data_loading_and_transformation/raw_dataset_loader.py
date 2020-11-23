import os
from torch_geometric.data import Data
from torch import tensor
import numpy as np
import pickle
import pathlib
from data_loading_and_transformation.dataset_descriptors import (
    StructureFeatures,
)


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data(dataset_path: str)
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def load_raw_data(self, dataset_path: str):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.

        Parameters
        ----------
        dataset_path: str
            Directory path where raw files are stored.
        """
        dataset = []
        for filename in os.listdir(dataset_path):
            f = open(dataset_path + filename, "r")
            all_lines = f.readlines()
            data_object = self.__transform_input_to_data_object(lines=all_lines)
            dataset.append(data_object)
            f.close()

        dataset_normalized = self.__normalize_dataset(dataset=dataset)

        serial_data_name = (pathlib.PurePath(dataset_path)).parent.name
        serial_data_path = "/home/mburcul/Desktop/Faculty/Master-thesis/GCNN/serialized_dataset/" + serial_data_name + ".pkl"

        with open(serial_data_path, "wb") as f:
            pickle.dump(dataset_normalized, f)

    def __transform_input_to_data_object(self, lines: [str]):
        """Transforms lines of strings read from the raw data file to Data object and returns it.

        Parameters
        ----------
        dataset_path: str
            Directory path where raw files are stored.

        Returns
        ----------
        Data
            Data object representing structure of an atom.
        """
        data_object = Data()

        graph_feat = lines[0].split(None, 2)
        free_energy = float(graph_feat[0].strip())
        magnetic_charge = float(graph_feat[1].strip())
        magnetic_moment = float(graph_feat[2].strip())
        data_object.y = tensor([free_energy, magnetic_charge, magnetic_moment])

        node_feature_matrix = []
        node_position_matrix = []
        for line in lines[1:]:
            node_feat = line.split(None, 11)

            x_pos = float(node_feat[2].strip())
            y_pos = float(node_feat[3].strip())
            z_pos = float(node_feat[4].strip())
            node_position_matrix.append([x_pos, y_pos, z_pos])

            num_of_protons = float(node_feat[0].strip())
            charge_density = float(node_feat[5].strip())
            magnetic_moment = float(node_feat[6].strip())

            charge_density = charge_density - num_of_protons

            node_feature_matrix.append(
                [num_of_protons, charge_density, magnetic_moment]
            )

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)

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
        max_free_energy = float("-inf")
        min_free_energy = float("inf")
        max_proton_number = np.full(StructureFeatures.SIZE.value, -np.inf)
        min_proton_number = np.full(StructureFeatures.SIZE.value, np.inf)
        max_charge_density = np.full(StructureFeatures.SIZE.value, -np.inf)
        min_charge_density = np.full(StructureFeatures.SIZE.value, np.inf)

        # histogram_data_free_energy = []
        # histogram_data_normalized_free_energy = []

        for data in dataset:
            # histogram_data_free_energy.append(data.y[0].item())
            max_free_energy = max(abs(data.y[0]), max_free_energy)
            min_free_energy = min(abs(data.y[0]), min_free_energy)
            max_proton_number = np.maximum(data.x[:, 0].numpy(), max_proton_number)
            min_proton_number = np.minimum(data.x[:, 0].numpy(), min_proton_number)
            max_charge_density = np.maximum(data.x[:, 1].numpy(), max_charge_density)
            min_charge_density = np.minimum(data.x[:, 1].numpy(), min_charge_density)

        for data in dataset:
            data.y[0] = (data.y[0] - min_free_energy) / (
                max_free_energy - min_free_energy
            )
            # histogram_data_normalized_free_energy.append(data.y[0].item())
            data.x[:, 0] = (data.x[:, 0] - min_proton_number) / (
                max_proton_number - min_proton_number
            )
            data.x[:, 1] = (data.x[:, 1] - min_charge_density) / (
                max_charge_density - min_charge_density
            )

        # Visualizing the normalization effect
        # plt.figure("Free energy histogram")
        # plt.hist(histogram_data_free_energy,bins=50 , range=(min(histogram_data_free_energy)-1000, max(histogram_data_free_energy)+1000))
        # plt.figure("Normalized free energy histogram")
        # plt.hist(histogram_data_normalized_free_energy,bins=100, range=(-3,0))
        # plt.show()

        return dataset
