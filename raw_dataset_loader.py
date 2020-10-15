import os
from torch_geometric.data import Data
from torch import tensor
import numpy as np
import pickle
import pathlib


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
            break

        print(dataset[0].pos)
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
            node_feature_matrix.append(
                [num_of_protons, charge_density, magnetic_moment]
            )

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)

        return data_object


cu = "CuAu_32atoms"
fe = "FePt_32atoms"

files_dir = "./Dataset/" + fe + "/output_files/"
loader = RawDataLoader()
loader.load_raw_data(dataset_path=files_dir)
