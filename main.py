import time
from datetime import datetime

from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)

from models.GNNStack import GNNStack
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
import torch


def train(dataset, writer, model, opt, num_epoch):
    data_size = len(dataset)
    batch_size = 64
    loader = DataLoader(
        dataset[: int(data_size * 0.7)], batch_size=batch_size, shuffle=True
    )
    validate_loader = DataLoader(
        dataset[int(data_size * 0.7) : int(data_size * 0.85)],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset[int(data_size * 0.85) :], batch_size=batch_size, shuffle=True
    )

    validation_error_stopper = 100
    # train
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad()
            pred = model(batch)
            real_value = torch.reshape(batch.y, (batch.y.size()[0], 1))
            loss = model.loss(pred, real_value)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)
        print("Epoch {}. Loss: {:.4f}.".format(epoch, total_loss))

        if epoch % 5 == 0:
            validate_loss = test(validate_loader, model)
            print(
                "Epoch {}. Loss: {:.4f}. Validate loss: {:.4f}".format(
                    epoch, total_loss, validate_loss
                )
            )
            writer.add_scalar("validate loss", validate_loss, epoch)
            if validation_error_stopper / validate_loss < 0.10:
                break
            validation_error_stopper = validate_loss

    test_loss = test(test_loader, model)
    writer.add_scalar("test loss", test_loss, display_name="Test loss")

    return model


def test(loader, model):
    model.eval()

    total_loss = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            real_value = data.y
        loss = model.loss(pred, real_value)
        total_loss += loss.item() * data.num_graphs

    total = len(loader.dataset)

    return total_loss / total

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
    AtomFeatures.MAGNETIC_MOMENT,
]
structure_features = [StructureFeatures.FREE_ENERGY]
radius = 15
max_num_node_neighbours = 7

loader = SerializedDataLoader()
dataset = loader.load_serialized_data(
    dataset_path=files_dir,
    atom_features=atom_features,
    structure_features=structure_features,
    radius=radius,
    max_num_node_neighbours=max_num_node_neighbours,
)

## GCNN parameters
# Fixed Parameters
num_node_features = len(atom_features)
input_dim = num_node_features
output_dim = 1

# Hyperparameters
learning_rate = 0.0125
num_epoch = 200

for i in range(12,150,10):
    hidden_dim = i

    model = GNNStack(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    model_name = (
        "GCNN_stack_meanpool_r-"
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
    model = train(dataset, writer, model, opt, num_epoch)

    torch.save(model.state_dict(), "./models_serialized/" + model_name)
'''

model = GNNStack(input_dim=3, hidden_dim=16, output_dim=1)
model.load_state_dict(torch.load("./models_serialized/GCNN_stack_r-12-mnnn-7-hd-16-ne-100.pk"))
model.eval()
data_size = len(dataset)
test_loader = DataLoader(
        dataset[int(data_size * 0.85) :], batch_size=5, shuffle=True
    )
for data in test_loader:
    with torch.no_grad():
        pred = model(data)
        real_value = data.y
    print("Prediction: "+str(pred)+", real_value: "+str(real_value))
'''