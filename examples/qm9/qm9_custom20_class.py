import os
import torch
import torch_geometric
import hydragnn
from torch_geometric.data import download_url, extract_zip, extract_tar

class QM9_custom(torch_geometric.datasets.QM9):
    def __init__(self, root: str, var_config=None, pre_filter=None):
        self.graph_feature_names = [
            "mu",
            "alpha",
            "HOMO",
            "LUMO",
            "del-epi",
            "R2",
            "ZPVE",
            "U0",
            "U",
            "H",
            "G",
            "cv",
            "U0atom",
            "Uatom",
            "Hatom",
            "Gatom",
            "A",
            "B",
            "C",
        ]
        self.graph_feature_dims = [1] * len(self.graph_feature_names)
        self.graph_feature_units = [
            "D",
            "a_0^3",
            "eV",
            "eV",
            "eV",
            "a_0^2",
            "eV",
            "eV",
            "eV",
            "eV",
            "eV",
            "cal/(molK)",
            "eV",
            "eV",
            "eV",
            "eV",
            "GHz",
            "GHz",
            "GHz",
        ]
        self.node_attribute_names = [
            "atomH",
            "atomC",
            "atomN",
            "atomO",
            "atomF",
            "atomicnumber",
            "IsAromatic",
            "HSP",
            "HSP2",
            "HSP3",
            "Hprop",
            "chargedensity",
        ]
        self.node_feature_units = [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "e",
        ]
        self.node_feature_dims = [1] * len(self.node_attribute_names)
        self.raw_url_2014 = "https://ndownloader.figstatic.com/files/3195389"
        self.raw_url2 = "https://ndownloader.figshare.com/files/3195404"
        self.var_config = var_config
        # FIXME: current self.qm9_pre_transform and pre_filter are not saved, due to __repr__ AttributeError for self.qm9_pre_transform()
        try:
            super().__init__(
                root, pre_transform=self.qm9_pre_transform, pre_filter=pre_filter
            )
        except:
            if os.path.exists(
                os.path.join(self.processed_dir, self.processed_file_names)
            ):
                print(
                    "Warning: qm9_pre_transform and qm9_pre_filter_test are not saved, but processed data file is saved %s"
                    % os.path.join(self.processed_dir, self.processed_file_names)
                )
            else:
                raise Exception(
                    "Error: processed data file is not saved %s"
                    % os.path.join(self.processed_dir, self.processed_file_names)
                )
            super().__init__(
                root, pre_transform=self.qm9_pre_transform, pre_filter=pre_filter
            )

    def download(self):
        file_path = download_url(self.raw_url_2014, self.raw_dir)
        os.rename(file_path, os.path.join(self.raw_dir, "dsgdb9nsd.xyz.tar.bz2"))
        extract_tar(
            os.path.join(self.raw_dir, "dsgdb9nsd.xyz.tar.bz2"), self.raw_dir, "r:bz2"
        )
        os.unlink(os.path.join(self.raw_dir, "dsgdb9nsd.xyz.tar.bz2"))

        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(
            os.path.join(self.raw_dir, "3195404"),
            os.path.join(self.raw_dir, "uncharacterized.txt"),
        )

    # Update each sample prior to loading.
    def qm9_pre_transform(self, data):
        # Set descriptor as element type.
        self.get_charge(data)
        data.y = data.y.squeeze()
        for iy in range(len(data.y)):
            # predict energy variables per node
            if self.graph_feature_units[iy] == "eV":
                data.y[iy] = data.y[iy] / len(data.x)

        hydragnn.preprocess.update_predicted_values(
            self.var_config["type"],
            self.var_config["output_index"],
            self.graph_feature_dims,
            self.node_feature_dims,
            data,
        )
        hydragnn.preprocess.update_atom_features(
            self.var_config["input_node_features"], data
        )
        # data.x = data.z.float().view(-1, 1)
        return data

    def get_charge(self, data):
        idx = data.idx

        N = data.x.size(dim=0)

        fname = os.path.join(self.raw_dir, "dsgdb9nsd_{:06d}.xyz".format(idx + 1))
        f = open(fname, "r")
        atomlines = f.readlines()[2 : 2 + N]
        f.close()

        try:
            charge = [
                float(line.split("\t")[-1].replace("\U00002013", "-"))
                for line in atomlines
            ]
        except:
            charge = [
                float(
                    line.split("\t")[-1].replace("*^", "e").replace("\U00002013", "-")
                )
                for line in atomlines
            ]
            print("strange charge in ", fname)
            print(charge)
        charge = torch.tensor(charge, dtype=torch.float).view(-1, 1)
        data.x = torch.cat((data.x, charge), 1)
