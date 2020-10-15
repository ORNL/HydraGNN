import os
from torch_geometric.data import Data
from torch import tensor
import numpy as np
import pickle
import pathlib
from dataset_descriptors import StructureFeatures
from sys import maxint

class RawDataLoader:
    def load_raw_data(self, dataset_path: str):
        # Reading raw files and creating Data objects from them.
        dataset = []
        for filename in os.listdir(dataset_path):
            print(filename)
            f = open(files_dir + filename, "r")
            all_lines = f.readlines()
            data_object = self.__transform_input_to_data_object(lines=all_lines)
            dataset.append(data_object)
            f.close()
        
        dataset_normalized = self.__normalize_dataset(dataset=dataset)

        # Extracting dataset name and storing it as a serialized object.
        serial_data_name = (pathlib.PurePath(files_dir)).parent.name
        serial_data_path = "./SerializedDataset/" + serial_data_name + ".pkl"

        with open(serial_data_path, "wb") as f:
            pickle.dump(dataset, f)

    def __transform_input_to_data_object(self, lines: [str]):
        data_object = Data()

        # Extracting structure level features.
        graph_feat = lines[0].split(None, 2)
        free_energy = np.float64(graph_feat[0].strip())
        magnetic_charge = np.float64(graph_feat[1].strip())
        magnetic_moment = np.float64(graph_feat[2].strip())
        data_object.y = tensor([free_energy, magnetic_charge, magnetic_moment])

        # Extracting atom level positions(x, y, z coordinates) and features.
        node_feature_matrix = []
        node_position_matrix = []
        for line in lines[1:]:
            node_feat = line.split(None, 11)

            x_pos = np.float64(node_feat[2].strip())
            y_pos = np.float64(node_feat[3].strip())
            z_pos = np.float64(node_feat[4].strip())
            node_position_matrix.append([x_pos, y_pos, z_pos])

            num_of_protons = int(node_feat[0].strip())
            charge_density = np.float64(node_feat[5].strip())
            magnetic_moment = np.float64(node_feat[6].strip())

            # Subtracting number of protons from charge density in order to get real value
            # in terms of how much electrons did it receive or give
            charge_density = charge_density - num_of_protons

            node_feature_matrix.append(
                [num_of_protons, charge_density, magnetic_moment]
            )

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)

        return data_object
    
    def __normalize_dataset(dataset):
        total_free_energy = 0
        max_free_energy = float('-inf')
        total_charge_density = np.zeros(StructureFeatures.SIZE.value)
        max_charge_density = np.full(StructureFeatures.SIZE.value, '-inf')

        for data in dataset:
            total_free_energy += data.y[0]
            total_charge_density += data.x[1]
            max_free_energy = max(data.y[0], max_free_energy)
            max_charge_density = np.maximum(data.x[1], max_charge_density)
        
        mean_free_energy = total_free_energy/StructureFeatures.SIZE.value
        mean_charge_density = total_charge_density/StructureFeatures.SIZE.value
        for data in dataset:
            data.y[0] = (data.y[0] - mean_free_energy)/max_free_energy
            data.x[1] = (data.x[1] - mean_charge_density)/max_charge_density
        
        return dataset
        





cu = "CuAu_32atoms"
fe = "FePt_32atoms"

files_dir = "./Dataset/" + fe + "/output_files/"
loader = RawDataLoader()
loader.load_raw_data(dataset_path=files_dir)
