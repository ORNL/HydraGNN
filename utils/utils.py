import os
from random import shuffle
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_utils.raw_dataset_loader import RawDataLoader
from data_utils.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
    Dataset,
)
from utils.visualizer import Visualizer


def get_comm_size_and_rank():
    world_size = 1
    world_rank = 0
    try:
        world_size = os.environ["OMPI_COMM_WORLD_SIZE"]
        world_rank = os.environ["OMPI_COMM_WORLD_RANK"]
    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return int(world_size), int(world_rank)


def setup_ddp():

    """ "Initialize DDP"""

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    distributed_data_parallelism = False
    world_size, world_rank = get_comm_size_and_rank()

    ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
    ## The following is Summit specific
    import subprocess
    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1 )".format(os.environ['LSB_DJOB_HOSTFILE'])
    master_addr = str(subprocess.check_output(get_master, shell=True))[2:-3]
    master_port = "23456"

    try:
        world_size = os.environ["OMPI_COMM_WORLD_SIZE"]
        world_rank = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = world_size
        os.environ["RANK"] = world_rank
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend, rank=int(world_rank), world_size=int(world_size)
            )
        distributed_data_parallelism = True

    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return distributed_data_parallelism, world_size, world_rank


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

    num_epoch = config["num_epoch"]
    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer, config["output_dim"])
        val_mae = validate(val_loader, model, config["output_dim"])
        test_rmse = test(test_loader, model, config["output_dim"])
        scheduler.step(val_mae)
        writer.add_scalar("train error", train_mae, epoch)
        writer.add_scalar("validate error", val_mae, epoch)
        writer.add_scalar("test error", test_rmse[0], epoch)

        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}"
        )
    # At the end of training phase, do the one test run for visualizer to get latest predictions
    visualizer = Visualizer(model_with_config_name)
    test_rmse, true_values, predicted_values = test(
        test_loader, model, config["output_dim"]
    )
    if (
        config["denormalize_output"] == "True"
    ):  ##output predictions with unit/not normalized
        y_minmax = config["y_minmax"]
        for isamp in range(len(predicted_values)):
            for iout in range(len(predicted_values[0])):
                predicted_values[isamp][iout] = (
                    predicted_values[isamp][iout]
                    * (y_minmax[iout][1] - y_minmax[iout][0])
                    + y_minmax[iout][0]
                )
                true_values[isamp][iout] = (
                    true_values[isamp][iout] * (y_minmax[iout][1] - y_minmax[iout][0])
                    + y_minmax[iout][0]
                )

    visualizer.add_test_values(
        true_values=true_values, predicted_values=predicted_values
    )
    visualizer.create_scatter_plot()


def train(loader, model, opt, output_dim):

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    model.train()

    total_error = 0
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
            loss = model.module.loss_rmse(pred, data.y)
        else:
            pred = model(data)
            loss = model.loss_rmse(pred, data.y)
        loss.backward()
        total_error += loss.item() * data.num_graphs
        opt.step()
    return total_error / len(loader.dataset)


@torch.no_grad()
def validate(loader, model, output_dim):

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    total_error = 0
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
            error = model.module.loss_rmse(pred, data.y)
        else:
            pred = model(data)
            error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)


@torch.no_grad()
def test(loader, model, output_dim):

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    total_error = 0
    model.eval()
    true_values = []
    predicted_values = []
    for data in tqdm(loader):
        data = data.to(device)
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
        else:
            pred = model(data)
        true_values.extend(data.y.tolist())
        predicted_values.extend(pred.tolist())

        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            error = model.module.loss_rmse(pred, data.y)
        else:
            error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset), true_values, predicted_values


def dataset_loading_and_splitting(
    config: {},
    chosen_dataset_option: Dataset,
    distributed_data_parallelism: bool = False,
):

    if chosen_dataset_option == Dataset.CuAu:
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        return split_dataset(
            dataset=dataset_CuAu,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
            distributed_data_parallelism=distributed_data_parallelism,
        )
    elif chosen_dataset_option == Dataset.FePt:
        dataset_FePt = load_data(Dataset.FePt.value, config)
        return split_dataset(
            dataset=dataset_FePt,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
            distributed_data_parallelism=distributed_data_parallelism,
        )
    elif chosen_dataset_option == Dataset.FeSi:
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        return split_dataset(
            dataset=dataset_FeSi,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
            distributed_data_parallelism=distributed_data_parallelism,
        )
    elif chosen_dataset_option == Dataset.unit_test:
        dataset_unit_test = load_data(Dataset.unit_test.value, config)
        return split_dataset(
            dataset=dataset_unit_test,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
            distributed_data_parallelism=distributed_data_parallelism,
        )
    else:
        # FIXME, should re-normalize mixed datasets based on joint min_max
        raise ValueError(
            "Chosen dataset option not yet supported", chosen_dataset_option
        )
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        dataset_FePt = load_data(Dataset.FePt.value, config)
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        if chosen_dataset_option == Dataset.CuAu_FePt_SHUFFLE:
            dataset_CuAu.extend(dataset_FePt)
            dataset_combined = dataset_CuAu
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:

            return combine_and_split_datasets(
                dataset1=dataset_CuAu,
                dataset2=dataset_FePt,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
            return combine_and_split_datasets(
                dataset1=dataset_FePt,
                dataset2=dataset_CuAu,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.FePt_FeSi_SHUFFLE:
            dataset_FePt.extend(dataset_FeSi)
            dataset_combined = dataset_FePt
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )


def create_dataloaders(
    distributed_data_parallelism, trainset, valset, testset, batch_size
):

    if distributed_data_parallelism:

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler
        )
        val_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=val_sampler
        )
        test_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=test_sampler
        )

    else:

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True,
        )

    return train_loader, val_loader, test_loader


def split_dataset(
    dataset: [],
    batch_size: int,
    perc_train: float,
    distributed_data_parallelism: bool = False,
):
    perc_val = (1 - perc_train) / 2
    data_size = len(dataset)

    trainset = dataset[: int(data_size * perc_train)]
    valset = dataset[
        int(data_size * perc_train) : int(data_size * (perc_train + perc_val))
    ]
    testset = dataset[int(data_size * (perc_train + perc_val)) :]

    train_loader, val_loader, test_loader = create_dataloaders(
        distributed_data_parallelism, trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [],
    dataset2: [],
    batch_size: int,
    perc_train: float,
    distributed_data_parallelism: bool = False,
):
    data_size = len(dataset1)

    trainset = dataset1[: int(data_size * perc_train)]
    valset = dataset1[int(data_size * perc_train) :]
    testset = dataset2

    train_loader, val_loader, test_loader = create_dataloaders(
        distributed_data_parallelism, trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def load_data(dataset_option, config):
    transform_raw_data_to_serialized(dataset_option)
    files_dir = (
        f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
    )

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset = loader.load_serialized_data(
        dataset_path=files_dir,
        config=config,
    )

    return dataset


def transform_raw_data_to_serialized(raw_dataset: str):

    _, rank = get_comm_size_and_rank()

    if rank == 0:

        raw_datasets = ["CuAu_32atoms", "FePt_32atoms", "FeSi_1024atoms", "unit_test"]
        if raw_dataset not in raw_datasets:
            print("WARNING: requested serialized dataset does not exist.")
            return

        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        if not os.path.exists(serialized_dir):
            os.mkdir(serialized_dir)
        serialized_dataset_dir = os.path.join(serialized_dir, raw_dataset)
        files_dir = (
            os.environ["SERIALIZED_DATA_PATH"]
            + "/dataset/"
            + raw_dataset
            + "/output_files/"
        )
        if not os.path.exists(serialized_dataset_dir):
            loader = RawDataLoader()
            loader.load_raw_data(dataset_path=files_dir)

    dist.barrier()
