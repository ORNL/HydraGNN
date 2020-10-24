from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import train, test
from models.GNNStack import GNNStack
from models.PNNStack import PNNStack
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import torch

"""
# This is only to be used when running for the first time to process raw files and store them as serialized objects.
cu = "CuAu_32atoms"
fe = "FePt_32atoms"

files_dir = "./Dataset/" + fe + "/output_files/"
loader = RawDataLoader()
loader.load_raw_data(dataset_path=files_dir)
"""

# Dataset parameters
fe = "FePt_32atoms.pkl"
files_dir = "./SerializedDataset/" + fe

atom_features = [
    AtomFeatures.NUM_OF_PROTONS,
    AtomFeatures.CHARGE_DENSITY,
    AtomFeatures.MAGNETIC_MOMENT
]
structure_features = [StructureFeatures.FREE_ENERGY]
radius = 7
max_num_node_neighbours = 5

loader = SerializedDataLoader()
dataset = loader.load_serialized_data(
    dataset_path=files_dir,
    atom_features=atom_features,
    structure_features=structure_features,
    radius=radius,
    max_num_node_neighbours=max_num_node_neighbours,
)

data_size = len(dataset)
batch_size = 64
train_loader = DataLoader(
    dataset[: int(data_size * 0.7)], batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    dataset[int(data_size * 0.7) : int(data_size * 0.85)],
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset[int(data_size * 0.85) :], batch_size=batch_size, shuffle=True
)

## GCNN parameters
# Fixed Parameters
num_node_features = len(atom_features)
input_dim = num_node_features
hidden_dim = 15
output_dim = 1

# Hyperparameters
learning_rate = 0.01
num_epoch = 50


## Setup for PNNStack
deg = torch.zeros(max_num_node_neighbours + 1, dtype=torch.long)
for data in dataset[:int(data_size * 0.7)]:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PNNStack(deg, len(atom_features), hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)


## Setup for GNNstack
'''
model = GNNStack(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
opt = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(opt, step_size=num_epoch/5, gamma=0.9)
'''

model_name = (
    model.__str__()
    + str(radius)
    + "-mnnn-"
    + str(max_num_node_neighbours)
    + "-hd-"
    + str(hidden_dim)
    + "-ne-"
    + str(num_epoch)
    + "-lr-"
    + str(learning_rate)
    + ".pk"
)

writer = SummaryWriter("./logs/" + model_name)

torch.manual_seed(0)

for epoch in range(1, num_epoch):
    loss = train(train_loader, model, optimizer)
    writer.add_scalar("train error", loss, epoch)
    val_mse = test(val_loader, model)
    writer.add_scalar("validate error", val_mse, epoch)
    test_mse = test(test_loader, model)
    writer.add_scalar("validate error", val_mse, epoch)
    scheduler.step(val_mse)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mse:.4f}, '
          f'Test: {test_mse:.4f}')

torch.save(model.state_dict(), "./models_serialized/" + model_name)
