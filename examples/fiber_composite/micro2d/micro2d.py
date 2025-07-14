# %%
import os, json
import importlib
import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

import micro2d_pyg as micro2d_utils
#importlib.reload(vfiber_utils)

import torch_geometric.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

import random


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42) 
#torch.use_deterministic_algorithms(True)
dataset = micro2d_utils.FiberNet(root='.') # ,force_reload=True
# dataset = dataset.to('cuda')
# %%
# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

filename = os.path.join(os.getcwd(), "config_micro2d.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()

# # Determine device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     # specific a gpu for each process
#     device = torch.device(f"cuda:{world_rank}")

log_name = "diff"
hydragnn.utils.setup_log(log_name)

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
# # The following lines are the standard way to handle device placement in hydragnn.
# device = hydragnn.utils.get_device(verbosity_level=verbosity)
# model.to(device)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001
)

# %%
# Run training with the given model and Voronoi dataset.
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
    create_plots=config["Visualization"]["create_plots"],
    # device=device,
)


hydragnn.utils.save_model(model, optimizer, log_name)
hydragnn.utils.print_timers(verbosity)