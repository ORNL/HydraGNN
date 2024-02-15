import os, json

import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

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


# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

num_samples = 1000

# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

def demo_model_parallel(group, group_id):
    global config
    world_size, world_rank = hydragnn.utils.get_comm_size_and_rank()
    print (world_rank, "group_id:", group_id)

    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = group_id*2 + 10
    print (world_rank, "hidden_dim:", config["NeuralNetwork"]["Architecture"]["hidden_dim"])

    log_name = "qm9_test_group%d"%group_id
    print (world_rank, "log_name:", log_name)
    # Enable print to log file.
    hydragnn.utils.setup_log(log_name)

    # Use built-in torch_geometric dataset.
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
    model = hydragnn.utils.get_distributed_model(model, verbosity, group=group)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and qm9 dataset.
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

if __name__ == "__main__":
    world_size, world_rank = hydragnn.utils.setup_ddp()
    print(f"WORLD size {world_size} and WORLD rank {world_rank}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of devices available: {num_gpus}", flush=True)
    num_groups = 2
    num_ranks_per_group = world_size//num_groups
    processes_groups = hydragnn.utils.setup_ddp_groups(num_groups)
    for group_id, group in enumerate(processes_groups):
        if torch.distributed.get_rank(group) >= 0:
           demo_model_parallel(group, group_id)
           ## How to pick next task?
           ## Option 1: no codinator option
           ##    - all writes results to a shared file or with MPI
           ##    - each group will try to find next parameter
           ##    - can be a bottleneck
           ## Option 2: coodinator 
           ##    - may help in a large scale
    torch.distributed.barrier()
    # hydragnn.utils.cleanup(processes_groups)
    print("END")
