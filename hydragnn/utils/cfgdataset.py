import os
import numpy as np

from torch import tensor
from torch_geometric.data import Data
from hydragnn.utils.abstractrawdataset import AbstractRawDataset

from ase.io.cfg import read_cfg


class CFGDataset(AbstractRawDataset):
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
