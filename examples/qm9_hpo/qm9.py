import os, json

import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
import argparse

# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def qm9_pre_filter(data):
    return data.idx < num_samples


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", help="model_type", default="EGNN")
parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=5)
parser.add_argument("--num_conv_layers", type=int, help="num_conv_layers", default=6)
parser.add_argument("--num_headlayers", type=int, help="num_headlayers", default=2)
parser.add_argument("--dim_headlayers", type=int, help="dim_headlayers", default=10)
parser.add_argument("--log", help="log name", default="qm9_test")
args = parser.parse_args()
args.parameters = vars(args)

num_samples = 1000

# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Update the config dictionary with the suggested hyperparameters
config["NeuralNetwork"]["Architecture"]["model_type"] = args.parameters["model_type"]
config["NeuralNetwork"]["Architecture"]["hidden_dim"] = args.parameters["hidden_dim"]
config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = args.parameters[
    "num_conv_layers"
]

dim_headlayers = [
    args.parameters["dim_headlayers"] for i in range(args.parameters["num_headlayers"])
]

for head_type in config["NeuralNetwork"]["Architecture"]["output_heads"]:
    config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
        "num_headlayers"
    ] = args.parameters["num_headlayers"]
    config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
        "dim_headlayers"
    ] = dim_headlayers

if args.parameters["model_type"] not in ["EGNN", "SchNet", "DimeNet"]:
    config["NeuralNetwork"]["Architecture"]["equivariance"] = False

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()

log_name = args.log
# Enable print to log file.
hydragnn.utils.setup_log(log_name)

# Use built-in torch_geometric datasets.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
dataset = torch_geometric.datasets.QM9(
    root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter
)
train, val, test = hydragnn.preprocess.split_dataset(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
)
(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
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

# Run training with the given model and qm9 datasets.
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

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
)

hydragnn.utils.save_model(model, optimizer, log_name)
hydragnn.utils.print_timers(verbosity)
