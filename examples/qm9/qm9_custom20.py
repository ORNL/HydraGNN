import os, json
import torch
import hydragnn
import matplotlib.pyplot as plt
import numpy as np
from qm9_utils import plot_node_graph_features, plot_predictions_all20
import pickle
from qm9_custom20_class import QM9_custom

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
log_name = "qm9_tutorial_%s_032024" % inputfilesubstr
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
