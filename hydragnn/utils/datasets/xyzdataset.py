import os
import numpy as np

from torch import tensor
from torch_geometric.data import Data
from hydragnn.utils.datasets.abstractrawdataset import AbstractRawDataset

from ase.io import read


class XYZDataset(AbstractRawDataset):
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

    def __transform_XYZ_ASE_object_to_data_object(self, filepath):

        # FIXME:
        #  this still assumes bulk modulus is specific to the XYZ format.

        ase_object = read(filepath, parallel=False)

        data_object = Data()

        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        node_feature_matrix = proton_numbers
        data_object.x = tensor(node_feature_matrix).float()

        filename_without_extension = os.path.splitext(filepath)[0]

        filename_energy = os.path.join(filename_without_extension + "_energy.txt")
        f = open(filename_energy, "r", encoding="utf-8")
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
