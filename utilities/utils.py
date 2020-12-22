from torch_geometric.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from ray import tune
from torch import nn
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)
import os
from random import shuffle
from utilities.models_setup import generate_model
from tqdm import tqdm


def train_validate_test_hyperopt(
    config, checkpoint_dir=None, data_dir=None, writer=None
):
    atom_features = [
        AtomFeatures.NUM_OF_PROTONS,
        AtomFeatures.CHARGE_DENSITY,
    ]
    structure_features = [StructureFeatures.FREE_ENERGY]

    input_dim = len(atom_features)
    perc_train = 0.7
    dataset1, dataset2 = load_data(config, structure_features, atom_features)

    model = generate_model(
        model_type="PNN",
        input_dim=input_dim,
        dataset=dataset1[: int(len(dataset1) * perc_train)],
        config=config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=0.00001
    )

    train_loader, val_loader, test_loader = combine_and_split_datasets(
        dataset1=dataset1,
        dataset2=dataset2,
        batch_size=config["batch_size"],
        perc_train=perc_train,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    num_epoch = 200

    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer)
        val_mae = validate(val_loader, model)
        test_rmse = test(test_loader, model)
        scheduler.step(val_mae)
        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
            f"Test RMSE: {test_rmse:.4f}"
        )
        if epoch % 10 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(train_mae=train_mae, val_mae=val_mae, test_rmse=test_rmse)


def train_validate_test_normal(
    model, optimizer, train_loader, val_loader, test_loader, writer, scheduler, num_epoch
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)


    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer)
        val_mae = validate(val_loader, model)
        test_rmse = test(test_loader, model)
        scheduler.step(val_mae)
        writer.add_scalar("train error", train_mae, epoch)
        writer.add_scalar("validate error", val_mae, epoch)
        writer.add_scalar("test error", test_rmse, epoch)

        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
            f"Test RMSE: {test_rmse:.4f}"
        )


def train(loader, model, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.train()
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        loss = model.loss(pred, real_value)
        loss.backward()
        total_error += loss.item() * data.num_graphs
        opt.step()
    return total_error / len(loader.dataset)


@torch.no_grad()
def validate(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        error = model.loss(pred, real_value)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)


@torch.no_grad()
def test(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        error = model.loss_rmse(pred, real_value)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)

# Dataset splitting and manipulation
def dataset_splitting(dataset1: [], dataset2: [], batch_size: int, perc_train: float, chosen_dataset_option: int):

    if chosen_dataset_option==1:
        return split_dataset(dataset=dataset1, batch_size=batch_size, perc_train=perc_train)
    elif chosen_dataset_option==2:
        return split_dataset(dataset=dataset2, batch_size=batch_size, perc_train=perc_train)
    elif chosen_dataset_option==3:
        dataset1.extend(dataset2)
        shuffle(dataset1)
        return split_dataset(dataset=dataset1, batch_size=batch_size, perc_train=perc_train)
    elif chosen_dataset_option==4:
        return combine_and_split_datasets(dataset1=dataset1, dataset2=dataset2, batch_size=batch_size, perc_train=perc_train)
    elif chosen_dataset_option==5:
        return combine_and_split_datasets(dataset1=dataset2, dataset2=dataset1, batch_size=batch_size, perc_train=perc_train)


def split_dataset(dataset: [], batch_size: int, perc_train: float):
    perc_val = (1 - perc_train) / 2
    data_size = len(dataset)
    train_loader = DataLoader(
        dataset[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[int(data_size * perc_train) : int(data_size * (perc_train + perc_val))],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset[int(data_size * (perc_train + perc_val)) :],
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [], dataset2: [], batch_size: int, perc_train: float
):
    data_size = len(dataset1)
    train_loader = DataLoader(
        dataset1[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset1[int(data_size * perc_train) :],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_data(config):
    # Loading raw data if necessary
    raw_datasets = ["CuAu_32atoms", "FePt_32atoms"]
    if len(os.listdir(os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset")) < 2:
        for raw_dataset in raw_datasets:
            files_dir = (
                os.environ["SERIALIZED_DATA_PATH"]
                + "/dataset/"
                + raw_dataset
                + "/output_files/"
            )
            loader = RawDataLoader()
            loader.load_raw_data(dataset_path=files_dir)

    # dataset parameters
    fe = "FePt_32atoms.pkl"
    cu = "CuAu_32atoms.pkl"
    files_dir1 = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset/" + fe
    files_dir2 = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset/" + cu

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset1 = loader.load_serialized_data(
        dataset_path=files_dir1,
        atom_features=config['atom_features'],
        structure_features=config['structure_features'],
        radius=config["radius"],
        max_num_node_neighbours=config["max_num_node_neighbours"],
    )
    dataset2 = loader.load_serialized_data(
        dataset_path=files_dir2,
        atom_features=config['atom_features'],
        structure_features=config['structure_features'],
        radius=config["radius"],
        max_num_node_neighbours=config["max_num_node_neighbours"],
    )

    return dataset1, dataset2
