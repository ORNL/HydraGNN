import os, json
import torch
import torch_geometric
import hydragnn
import matplotlib.pyplot as plt
from torch_geometric.data import download_url, extract_zip, extract_tar
import numpy as np
from qm9_utils import plot_node_graph_features, plot_predictions_all20
import pickle


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


###################################
inputfilesubstr = "all20"
# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(__file__), "qm9_" + inputfilesubstr + ".json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
serial_data_name = "qm9_train_test_val_idx_lists.pkl"
with open(
    os.path.join(os.path.dirname(__file__), "dataset", serial_data_name), "rb"
) as f:
    idx_train_list = pickle.load(f)
    idx_val_list = pickle.load(f)
    idx_test_list = pickle.load(f)


def qm9_pre_filter_train(data):
    return data.idx in idx_train_list


def qm9_pre_filter_val(data):
    return data.idx in idx_val_list


def qm9_pre_filter_test(data):
    return data.idx in idx_test_list


train = QM9_custom(
    root=os.path.join(os.path.dirname(__file__), "dataset/all20/train"),
    var_config=var_config,
    pre_filter=qm9_pre_filter_train,
)
val = QM9_custom(
    root=os.path.join(os.path.dirname(__file__), "dataset/all20/val"),
    var_config=var_config,
    pre_filter=qm9_pre_filter_val,
)
test = QM9_custom(
    root=os.path.join(os.path.dirname(__file__), "dataset/all20/test"),
    var_config=var_config,
    pre_filter=qm9_pre_filter_test,
)
##################################################################################################################
var_config["output_names"] = [
    train.graph_feature_names[item]
    if var_config["type"][ihead] == "graph"
    else train.node_attribute_names[item]
    for ihead, item in enumerate(var_config["output_index"])
]
var_config["output_units"] = [
    train.graph_feature_units[item]
    if var_config["type"][ihead] == "graph"
    else train.node_feature_units[item]
    for ihead, item in enumerate(var_config["output_index"])
]
var_config["input_node_feature_names"] = [
    train.node_attribute_names[item] for item in var_config["input_node_features"]
]

num_output_features = len(var_config["output_names"])
minmax_file = os.path.join(
    os.path.dirname(__file__), "dataset/all20", "train_minmax_output.pt"
)
if not os.path.isfile(minmax_file):
    min_output_feature = torch.zeros([num_output_features, 1]) + 1.0e5
    max_output_feature = torch.zeros([num_output_features, 1]) - 1.0e5
    for data in train:
        num_nodes = data.x.size()[0]
        min_output_feature[:-1, 0] = torch.minimum(
            min_output_feature[:-1, 0], data.y[:-num_nodes, 0]
        )
        max_output_feature[:-1, 0] = torch.maximum(
            max_output_feature[:-1, 0], data.y[:-num_nodes, 0]
        )
        min_output_feature[-1, 0] = torch.minimum(
            min_output_feature[-1, 0],
            torch.min(data.y[-num_nodes:, 0], 0, keepdim=True)[0],
        )
        max_output_feature[-1, 0] = torch.maximum(
            max_output_feature[-1, 0],
            torch.max(data.y[-num_nodes:, 0], 0, keepdim=True)[0],
        )
    torch.save(
        {
            "min_output_feature": min_output_feature,
            "max_output_feature": max_output_feature,
        },
        minmax_file,
    )
else:
    minmax_dict = torch.load(minmax_file)
    min_output_feature = minmax_dict["min_output_feature"]
    max_output_feature = minmax_dict["max_output_feature"]

for dataset in [train, val, test]:
    for data in dataset:
        num_nodes = data.x.size()[0]
        data.y[:-num_nodes, 0] = (
            data.y[:-num_nodes, 0] - min_output_feature[:-1, 0]
        ) / (max_output_feature[:-1, 0] - min_output_feature[:-1, 0])
        data.y[-num_nodes:, 0] = (
            data.y[-num_nodes:, 0] - min_output_feature[-1, 0]
        ) / (max_output_feature[-1, 0] - min_output_feature[-1, 0])
##################################################################################################################
train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)
config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)
learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)
# Run training with the given model and qm9 dataset.
log_name = "qm9_LOG2023_%s_1113" % inputfilesubstr
# Enable print to log file.
hydragnn.utils.setup_log(log_name)
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)
##################################################################################################################
##################################################################################################################
checksplitting = True
if checksplitting:
    plot_node_graph_features(var_config, train, val, test, "./logs/" + log_name)
##################################################################################################################
##################################################################################################################
try:
    hydragnn.utils.model.load_existing_model(model, log_name)
except:
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=config["Visualization"]["create_plots"],
    )
    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

output_dir = "./logs/" + log_name + "/postpred"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_predictions_all20(
    model,
    var_config,
    output_dir,
    min_output_feature,
    max_output_feature,
    val_loader,
    filename="scatter_val",
)
plot_predictions_all20(
    model,
    var_config,
    output_dir,
    min_output_feature,
    max_output_feature,
    test_loader,
    filename="scatter_test",
)
plot_predictions_all20(
    model,
    var_config,
    output_dir,
    min_output_feature,
    max_output_feature,
    train_loader,
    filename="scatter_train",
)
