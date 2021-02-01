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
    Dataset,
)
import os
from random import shuffle
from utilities.models_setup import generate_model
from utilities.visualizer import Visualizer
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
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config,
    model_with_config_name,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    visualizer = Visualizer(model_with_config_name)
    num_epoch = config["num_epoch"]
    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer, config["output_dim"])
        val_mae = validate(val_loader, model, config["output_dim"])
        test_rmse, true_values, predicted_values = test(
            test_loader, model, config["output_dim"]
        )
        visualizer.add_test_values(
            true_values=true_values, predicted_values=predicted_values
        )
        scheduler.step(val_mae)
        writer.add_scalar("train error", train_mae, epoch)
        writer.add_scalar("validate error", val_mae, epoch)
        writer.add_scalar("test error", test_rmse, epoch)

        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
            f"Test RMSE: {test_rmse:.4f}"
        )
    visualizer.create_scatter_plot()


def train(loader, model, opt, output_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.train()
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        loss = model.loss_rmse(pred, data.y)
        loss.backward()
        total_error += loss.item() * data.num_graphs
        opt.step()
    return total_error / len(loader.dataset)


@torch.no_grad()
def validate(loader, model, output_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)


@torch.no_grad()
def test(loader, model, output_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    true_values = []
    predicted_values = []
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        true_values.extend(data.y.tolist())
        predicted_values.extend(pred.tolist())
        error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset), true_values, predicted_values


# Dataset splitting and manipulation
def dataset_splitting(
    dataset_CuAu: [],
    dataset_FePt: [],
    dataset_FeSi: [],
    batch_size: int,
    perc_train: float,
    chosen_dataset_option: int,
):

    if chosen_dataset_option == Dataset.CuAu:
        return split_dataset(
            dataset=dataset_CuAu, batch_size=batch_size, perc_train=perc_train
        )
    elif chosen_dataset_option == Dataset.FePt:
        return split_dataset(
            dataset=dataset_FePt, batch_size=batch_size, perc_train=perc_train
        )
    elif chosen_dataset_option == Dataset.FeSi:
        return split_dataset(
            dataset=dataset_FeSi, batch_size=batch_size, perc_train=perc_train
        )
    elif chosen_dataset_option == Dataset.CuAu_FePt_SHUFFLE:
        dataset_CuAu.extend(dataset_FePt)
        dataset_combined = dataset_CuAu
        shuffle(dataset_combined)
        return split_dataset(
            dataset=dataset_combined, batch_size=batch_size, perc_train=perc_train
        )
    elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:
        return combine_and_split_datasets(
            dataset1=dataset_CuAu,
            dataset2=dataset_FePt,
            batch_size=batch_size,
            perc_train=perc_train,
        )
    elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
        return combine_and_split_datasets(
            dataset1=dataset_FePt,
            dataset2=dataset_CuAu,
            batch_size=batch_size,
            perc_train=perc_train,
        )


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
    raw_datasets = ["CuAu_32atoms", "FePt_32atoms", "FeSi_1024atoms"]
    if len(
        os.listdir(os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset")
    ) < len(raw_datasets):
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
    cu = "CuAu_32atoms.pkl"
    fe = "FePt_32atoms.pkl"
    fesi = "FeSi_1024atoms.pkl"

    files_dir1 = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset/" + cu
    files_dir2 = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset/" + fe
    files_dir3 = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset/" + fesi

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset1 = loader.load_serialized_data(
        dataset_path=files_dir1,
        config=config,
    )
    dataset2 = loader.load_serialized_data(
        dataset_path=files_dir2,
        config=config,
    )
    dataset3 = loader.load_serialized_data(
        dataset_path=files_dir3,
        config=config,
    )

    return dataset1, dataset2, dataset3
