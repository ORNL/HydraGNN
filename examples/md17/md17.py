import os, json

import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

# FIXME: temporary treatment for url issue in md17; will update if pytorch geometric corrected
from typing import Optional, Callable, List
import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data


class myMD17(InMemoryDataset):
    url = "http://quantum-machine.org/gdml/data/npz"
    file_names = {"uracil": "md17_uracil.npz"}

    def __init__(
        self,
        root: str,
        name: str,
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        assert name in self.file_names
        super().__init__(root, transform, pre_transform, pre_filter)

        if len(self.processed_file_names) == 1 and train is not None:
            raise ValueError(
                f"'{self.name}' dataset does not provide pre-defined splits "
                f"but 'train' argument is set to '{train}'"
            )
        elif len(self.processed_file_names) == 2 and train is None:
            raise ValueError(
                f"'{self.name}' dataset does provide pre-defined splits but "
                f"'train' argument was not specified"
            )

        idx = 0 if train is None or train else 1
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def mean(self) -> float:
        return float(self.data.energy.mean())

    @property
    def raw_dir(self) -> str:
        print(self.root)
        name = self.file_names[self.name].split(".")[0]
        print(osp.join(self.root, name, "raw"))
        return osp.join(self.root, name, "raw")

    @property
    def processed_dir(self) -> str:
        name = self.file_names[self.name].split(".")[0]
        return osp.join(self.root, name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        name = self.file_names[self.name]
        if name[-4:] == ".zip":
            return [name[:-4] + "-train.npz", name[:-4] + "-test.npz"]
        else:
            return [name]

    @property
    def processed_file_names(self) -> List[str]:
        name = self.file_names[self.name]
        return ["train.pt", "test.pt"] if name[-4:] == ".zip" else ["data.pt"]

    def download(self):
        url = f"{self.url}/{self.file_names[self.name]}"
        path = download_url(url, self.raw_dir)
        print(path)
        if url[-4:] == ".zip":
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)
            z = torch.from_numpy(raw_data["z"]).to(torch.long)
            pos = torch.from_numpy(raw_data["R"]).to(torch.float)
            energy = torch.from_numpy(raw_data["E"]).to(torch.float)
            force = torch.from_numpy(raw_data["F"]).to(torch.float)

            data_list = []
            for i in range(pos.size(0)):
                data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)


# Update each sample prior to loading.
def md17_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = data.energy / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    data = compute_edges(data)
    device = hydragnn.utils.get_device()
    return data.to(device)


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

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

log_name = "md17_test"
# Enable print to log file.
hydragnn.utils.setup_log(log_name)

# Use built-in torch_geometric dataset.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

dataset = myMD17(
    root="dataset/md17",
    name="uracil",
    pre_transform=md17_pre_transform,
    pre_filter=md17_pre_filter,
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

# Run training with the given model and qm9 dataset.
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)

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
