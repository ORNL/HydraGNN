import os, json

import torch
import torch_geometric

import hydragnn

# Update each sample prior to loading.
def md17_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float()
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = data.energy
    hydragnn.preprocess.update_predicted_values(
        var_config["type"],
        var_config["output_index"],
        data,
    )
    data = compute_edges(data)

    return data


# Randomly select ~1000 samples
def md17_pre_filter(data):
    return torch.rand(1) < 0.25


# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(__file__), "md17.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
arch_config = config["NeuralNetwork"]["Architecture"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Set the details of outputs of interest.
arch_config["output_dim"] = [0]
arch_config["output_type"] = ["graph"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

# Use built-in torch_geometric dataset.
compute_edges = hydragnn.preprocess.get_radius_graph(arch_config)
dataset = torch_geometric.datasets.MD17(
    root="dataset/md17",
    name="uracil",
    pre_transform=md17_pre_transform,
    pre_filter=md17_pre_filter,
)
train_loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False)

# Add model changes to config.
config["NeuralNetwork"]["Architecture"] = arch_config
config["NeuralNetwork"]["Variables_of_interest"] = var_config

input_dim = len(config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"])
model = hydragnn.models.create(
    model_type=arch_config["model_type"],
    input_dim=input_dim,
    dataset=dataset,
    config=arch_config,
    verbosity_level=verbosity,
)

learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

# Run training with the given model and md17 dataset.
num_epoch = config["NeuralNetwork"]["Training"]["num_epoch"]
for epoch in range(0, num_epoch):
    train_rmse, _, _ = hydragnn.train.train(train_loader, model, optimizer, verbosity)
    scheduler.step(train_rmse)

    hydragnn.utils.print_distributed(
        verbosity, f"Epoch: {epoch:02d}, Train RMSE: {train_rmse:.8f}"
    )
