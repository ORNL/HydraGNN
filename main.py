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
from utilities.utils import (
    train_validate_test,
    split_dataset,
    combine_and_split_datasets,
)
from utilities.models_setup import generate_model
from tensorboardX import SummaryWriter
import torch

# Loading raw data if necessary
raw_dataset_choices = {
    "1": ["CuAu_32atoms"],
    "2": ["FePt_32atoms"],
    "3": ["CuAu_32atoms", "FePt_32atoms"],
}
if len(os.listdir("./serialized_dataset")) < 2:
    print(
        "Select which raw dataset you want to load and process: 1) CuAu 2) FePt 3) all"
    )
    chosen_raw_dataset = raw_dataset_choices[input("Selected value: ")]
    for raw_dataset in chosen_raw_dataset:
        files_dir = "./dataset/" + raw_dataset + "/output_files/"
        loader = RawDataLoader()
        loader.load_raw_data(dataset_path=files_dir)

# dataset parameters
fe = "FePt_32atoms.pkl"
cu = "CuAu_32atoms.pkl"
files_dir1 = "./serialized_dataset/" + fe
files_dir2 = "./serialized_dataset/" + cu

atom_features = [
    AtomFeatures.NUM_OF_PROTONS,
    AtomFeatures.CHARGE_DENSITY,
    AtomFeatures.MAGNETIC_MOMENT,
]
structure_features = [StructureFeatures.FREE_ENERGY]
radius = 7
max_num_node_neighbours = 5

# loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
loader = SerializedDataLoader()
dataset = loader.load_serialized_data(
    dataset_path=files_dir2,
    atom_features=atom_features,
    structure_features=structure_features,
    radius=radius,
    max_num_node_neighbours=max_num_node_neighbours,
)
# dataset2 = loader.load_serialized_data(
#     dataset_path=files_dir2,
#     atom_features=atom_features,
#     structure_features=structure_features,
#     radius=radius,
#     max_num_node_neighbours=max_num_node_neighbours,
# )


batch_size = int(input("Batch size, integer value: "))
train_loader, val_loader, test_loader = split_dataset(
    dataset=dataset, batch_size=batch_size, perc_train=0.7, perc_val=0.15
)

# train_loader, val_loader, test_loader = combine_and_split_datasets(dataset1=dataset1, dataset2=dataset2, batch_size=batch_size, perc_train=0.8)


# Fixed Parameters
input_dim = len(atom_features)

# Hyperparameters
learning_rate = 0.0001
num_epoch = 200

# model setup
model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
chosen_model = model_choices[input("Selected value: ")]
model = generate_model(model_type=chosen_model, input_dim=input_dim, dataset=dataset, max_num_node_neighbours=max_num_node_neighbours)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20, min_lr=0.00001
)

model_name = (
    model.__str__()
    + "-r-"
    + str(radius)
    + "-mnnn-"
    + str(max_num_node_neighbours)
    + "-num_conv_layers-"
    + str(model.num_conv_layers)
    + "-hd-"
    + str(model.hidden_dim)
    + "-ne-"
    + str(num_epoch)
    + "-lr-"
    + str(learning_rate)
    + "-bs-"
    + str(batch_size)
    + ".pk"
)
writer = SummaryWriter("./logs/" + model_name)
train_validate_test(
    model,
    optimizer,
    num_epoch,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
)
torch.save(model.state_dict(), "./models_serialized/" + model_name)
"""
model.load_state_dict(torch.load("models_serialized/PNNStack7-mnnn-5-hd-75-ne-200-lr-0.01.pk", map_location=torch.device('cpu')))
print(test(test_loader, model))


"""
