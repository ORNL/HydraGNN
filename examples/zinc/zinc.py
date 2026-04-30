import sys
import os, json
import pdb
import torch
import torch.distributed as dist
import torch_geometric
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.transforms import AddLaplacianEigenvectorPE

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zinc.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

log_name = "zinc_test"
# Enable print to log file.
hydragnn.utils.print.print_utils.setup_log(log_name)

# Use built-in torch_geometric dataset.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the zinc/processed/ directory is removed.
lapPE = AddLaplacianEigenvectorPE(
    k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
    attr_name="pe",
    is_undirected=True,
)


def zinc_pre_transform(data):
    data.x = data.x.float().view(-1, 1)
    data.edge_attr = data.edge_attr.float().view(-1, 1)
    data = lapPE(data)
    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data


train = ZINC(
    root="dataset/zinc", subset=True, split="train", pre_transform=zinc_pre_transform
)
val = ZINC(
    root="dataset/zinc", subset=True, split="val", pre_transform=zinc_pre_transform
)
test = ZINC(
    root="dataset/zinc", subset=True, split="test", pre_transform=zinc_pre_transform
)

(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

config = hydragnn.utils.input_config_parsing.update_config(
    config, train_loader, val_loader, test_loader
)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
    model, optimizer, verbosity
)

hydragnn.utils.model.model.load_existing_model_config(
    model=model, config=config["NeuralNetwork"]["Training"]
)

# Run training with the given model and zinc dataset.
writer = hydragnn.utils.model.model.get_summary_writer(log_name)
hydragnn.utils.input_config_parsing.save_config(config, log_name)

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

if writer is not None:
    writer.close()

dist.destroy_process_group()
sys.exit(0)
