from torch import tensor
from torch_geometric.data import Data
from hydragnn.utils.datasets.abstractrawdataset import AbstractRawDataset


class LSMSDataset(AbstractRawDataset):
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
