import os

import numpy as np
import torch
import torch.distributed as dist

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from gcnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from gcnn.preprocess.raw_dataset_loader import RawDataLoader
from gcnn.preprocess.dataset_descriptors import Dataset
from gcnn.utils.distributed import get_comm_size_and_rank


def dataset_loading_and_splitting(
    config: {},
    chosen_dataset_option: Dataset,
):
    if chosen_dataset_option in [item.value for item in Dataset]:
        dataset_chosen, dataset_names = load_data(chosen_dataset_option, config)
        return split_dataset(
            dataset_list=dataset_chosen,
            dataset_names=dataset_names,
            batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
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
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            )
        elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:

            return combine_and_split_datasets(
                dataset1=dataset_CuAu,
                dataset2=dataset_FePt,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
            return combine_and_split_datasets(
                dataset1=dataset_FePt,
                dataset2=dataset_CuAu,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_FeSi_SHUFFLE:
            dataset_FePt.extend(dataset_FeSi)
            dataset_combined = dataset_FePt
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            )


def create_dataloaders(trainset, valset, testset, batch_size):

    if dist.is_initialized():

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler
        )
        val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, sampler=val_sampler
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=test_sampler
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
    dataset_list: [],
    dataset_names: [],
    batch_size: int,
    perc_train: float,
):

    if len(dataset_names) == 1 and dataset_names[0] == "total":
        dataset = dataset_list[0]
        perc_val = (1 - perc_train) / 2
        data_size = len(dataset)
        trainset = dataset[: int(data_size * perc_train)]
        valset = dataset[
            int(data_size * perc_train) : int(data_size * (perc_train + perc_val))
        ]
        testset = dataset[int(data_size * (perc_train + perc_val)) :]
    elif len(dataset_names) == 3:
        trainset = dataset_list[dataset_names.index("train")]
        valset = dataset_list[dataset_names.index("test")]
        testset = dataset_list[dataset_names.index("validate")]
    else:
        raise ValueError(
            'Must provide "total" OR "train", "test", "validate" data paths: ',
            dataset_names,
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [],
    dataset2: [],
    batch_size: int,
    perc_train: float,
):
    data_size = len(dataset1)

    trainset = dataset1[: int(data_size * perc_train)]
    valset = dataset1[int(data_size * perc_train) :]
    testset = dataset2

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def load_data(dataset_option, config):
    transform_raw_data_to_serialized(config["Dataset"])
    dataset_list = []
    datasetname_list = []
    for dataset_name, raw_data_path in config["Dataset"]["path"]["raw"].items():
        if dataset_name == "total":
            files_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
        else:
            files_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}_{dataset_name}.pkl"

        # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
        loader = SerializedDataLoader(config["Verbosity"]["level"])
        dataset = loader.load_serialized_data(
            dataset_path=files_dir,
            config=config["NeuralNetwork"],
        )
        dataset_list.append(dataset)
        datasetname_list.append(dataset_name)

    return dataset_list, datasetname_list


def transform_raw_data_to_serialized(config):

    _, rank = get_comm_size_and_rank()

    if rank == 0:
        loader = RawDataLoader(config)
        loader.load_raw_data()

    if dist.is_initialized():
        dist.barrier()
