import os, json

import torch
import torch_geometric

import hydragnn

# Update each sample prior to loading.
def md17_pre_transform(data):
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = data.energy
    hydragnn.preprocess.update_predicted_values(
        var_config["type"],
        var_config["output_index"],
        data,
    )
    data = compute_edges(data)

    return data


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
    root="dataset/md17", name="salicylic acid", pre_transform=md17_pre_transform
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

# FIXME: this should be a utility function.
# device_name, device = hydragnn.utils.device.get_device(verbosity)
# if torch.distributed.is_initialized():
#    if device_name == "cpu":
#        model = torch.nn.parallel.DistributedDataParallel(model)
#    else:
#        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
# FIXME: should these be utility functions?
learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

# FIXME: should use util function.
# model_name = model.__str__() + "_md17"
# FIXME: this should be a utility function.
# _, world_rank = hydragnn.utils.distributed.get_comm_size_and_rank()
# if int(world_rank) == 0:
#    writer = SummaryWriter("./logs/" + model_name)

# Run training with the given model and md17 dataset.
hydragnn.train.train(train_loader, model, optimizer, verbosity)
