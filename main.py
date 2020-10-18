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
        dataset[: int(data_size * 0.8)], batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset[int(data_size * 0.8) :], batch_size=batch_size, shuffle=True
    )

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

        if epoch % 10 == 0:
            test_loss = test(test_loader, model)
            print(
                "Epoch {}. Loss: {:.4f}. Test loss: {:.4f}".format(
                    epoch, total_loss, test_loss
                )
            )
            writer.add_scalar("test loss", test_loss, epoch)

    return model


def test(loader, model, is_validation=False):
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


# Dataset parameters
fe = "FePt_32atoms.pkl"
files_dir = "./SerializedDataset/" + fe

atom_features = [
    AtomFeatures.NUM_OF_PROTONS,
    AtomFeatures.CHARGE_DENSITY,
    AtomFeatures.MAGNETIC_MOMENT,
]
structure_features = [StructureFeatures.FREE_ENERGY]
radius = int(input("Enter the radius for the atom neighbourhood:"))
max_num_node_neighbours = int(input("Enter the max number of neighbours for the atom:"))

loader = SerializedDataLoader()
dataset = loader.load_serialized_data(
    dataset_path=files_dir,
    atom_features=atom_features,
    structure_features=structure_features,
    radius=radius,
    max_num_node_neighbours=max_num_node_neighbours,
)

# GCNN parameters
num_node_features = 3
input_dim = num_node_features
output_dim = 1
hidden_dim = int(input("Enter the number of hidden dimensions of a neural network:"))

model = GNNStack(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
opt = optim.Adam(model.parameters(), lr=0.01)

num_epoch = int(input("Enter the number of epochs to train the model:"))

writer = SummaryWriter("./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

model = train(dataset, writer, model, opt, num_epoch)
