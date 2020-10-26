from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from train import train_validate_test
from models.GNNStack import GNNStack
from models.PNNStack import PNNStack
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import torch
import pickle

# Loading raw data if necessary
raw_dataset_choices = {'1': ["CuAu_32atoms"], '2': ["FePt_32atoms"], '3': ["CuAu_32atoms", "FePt_32atoms"]}
if len(os.listdir("./SerializedDataset")) < 2:
    print("Select which raw dataset you want to load and process: 1) CuAu 2) FePt 3) all")
    chosen_raw_dataset = raw_dataset_choices[input("Selected value: ")]
    for raw_dataset in chosen_raw_dataset:
        files_dir = "./Dataset/" + raw_dataset + "/output_files/"
        loader = RawDataLoader()
        loader.load_raw_data(dataset_path=files_dir)

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

torch.manual_seed(0)

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
hidden_dim = int(input("Hidden layers: "))

# Hyperparameters
learning_rate = float(input("Learning rate: "))
num_epoch = 200
num_conv_layers = 16


## Setup for PNNStack
deg = torch.zeros(max_num_node_neighbours + 1, dtype=torch.long)
for data in dataset[:int(data_size * 0.7)]:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PNNStack(deg, len(atom_features), hidden_dim, num_conv_layers=num_conv_layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                            min_lr=0.00001)

model_name = (
    model.__str__()
    + "-r-"
    + str(radius)
    + "-mnnn-"
    + str(max_num_node_neighbours)
    + "-num_conv_layers-"
    + str(num_conv_layers)
    + "-hd-"
    + str(hidden_dim)
    + "-ne-"
    + str(num_epoch)
    + "-lr-"
    + str(learning_rate)
    + "-ncl-"
    +str(num_conv_layers)
    + ".pk"
)
writer = SummaryWriter("./logs/" + model_name)
train_validate_test(model, optimizer, num_epoch, train_loader, val_loader, test_loader,writer, scheduler)
torch.save(model.state_dict(), "./models_serialized/" + model_name)
'''
model.load_state_dict(torch.load("models_serialized/PNNStack7-mnnn-5-hd-75-ne-200-lr-0.01.pk", map_location=torch.device('cpu')))
print(test(test_loader, model))


'''