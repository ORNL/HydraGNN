import os
from random import shuffle

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from data_utils.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_utils.raw_dataset_loader import RawDataLoader
from data_utils.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
    Dataset,
)
from utils.print_utils import print_distributed, iterate_tqdm
from utils.visualizer import Visualizer

import re


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)

    return nlist


def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        ## CADES
        world_size = int(os.environ["SLURM_NPROCS"])
        world_rank = int(os.environ["SLURM_PROCID"])

    ## Fall back to default
    if world_size is None:
        world_size = 1
        print("DDP has to be initialized within a job - Running in sequential mode")

    return int(world_size), int(world_rank)


def get_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
    else:
        world_size = 1

    return int(world_size), int(world_rank)


def setup_ddp():

    """ "Initialize DDP"""

    if dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = "8889"

    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]

    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(world_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend, rank=int(world_rank), world_size=int(world_size)
            )
    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return world_size, world_rank


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
    verbosity=0,
    plot_init_solution=True,
    plot_hist_solution=False,
):
    num_epoch = config["Training"]["num_epoch"]
    # total loss tracking for train/vali/test
    total_loss_train = []
    total_loss_val = []
    total_loss_test = []
    # loss tracking of summation across all nodes for node feature predictions
    task_loss_train_sum = []
    task_loss_test_sum = []
    task_loss_val_sum = []
    # loss tracking for each head/task
    task_loss_train = []
    task_loss_test = []
    task_loss_val = []

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module
    else:
        model = model
    # preparing for results visualization
    ## collecting node feature
    node_feature = []
    for data in test_loader.dataset:
        node_feature.append(data.x)
    visualizer = Visualizer(
        model_with_config_name,
        node_feature=node_feature,
        num_heads=model.num_heads,
        head_dims=model.head_dims,
    )

    if plot_init_solution:  # visualizing of initial conditions
        test_rmse = test(test_loader, model, verbosity)
        true_values = test_rmse[3]
        predicted_values = test_rmse[4]
        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=config["Variables_of_interest"]["output_names"],
            iepoch=-1,
        )
    for epoch in range(0, num_epoch):
        train_mae, train_taskserr, train_taskserr_nodes = train(
            train_loader, model, optimizer, verbosity
        )
        val_mae, val_taskserr, val_taskserr_nodes = validate(
            val_loader, model, verbosity
        )
        test_rmse = test(test_loader, model, verbosity)
        scheduler.step(val_mae)
        if writer is not None:
            writer.add_scalar("train error", train_mae, epoch)
            writer.add_scalar("validate error", val_mae, epoch)
            writer.add_scalar("test error", test_rmse[0], epoch)
            for ivar in range(model.num_heads):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}",
        )
        print_distributed(verbosity, "Tasks MAE:", train_taskserr)

        total_loss_train.append(train_mae)
        total_loss_val.append(val_mae)
        total_loss_test.append(test_rmse[0])
        task_loss_train_sum.append(train_taskserr)
        task_loss_val_sum.append(val_taskserr)
        task_loss_test_sum.append(test_rmse[1])

        task_loss_train.append(train_taskserr_nodes)
        task_loss_val.append(val_taskserr_nodes)
        task_loss_test.append(test_rmse[2])

        ###tracking the solution evolving with training
        if plot_hist_solution:
            true_values = test_rmse[3]
            predicted_values = test_rmse[4]
            visualizer.create_scatter_plots(
                true_values,
                predicted_values,
                output_names=config["Variables_of_interest"]["output_names"],
                iepoch=epoch,
            )

    # At the end of training phase, do the one test run for visualizer to get latest predictions
    test_rmse, test_taskserr, test_taskserr_nodes, true_values, predicted_values = test(
        test_loader, model, verbosity
    )

    ##output predictions with unit/not normalized
    if config["Variables_of_interest"]["denormalize_output"] == "True":
        true_values, predicted_values = output_denormalize(
            config["Variables_of_interest"]["y_minmax"], true_values, predicted_values
        )

    ######result visualization######
    visualizer.create_plot_global(
        true_values,
        predicted_values,
        output_names=config["Variables_of_interest"]["output_names"],
    )
    visualizer.create_scatter_plots(
        true_values,
        predicted_values,
        output_names=config["Variables_of_interest"]["output_names"],
    )
    ######plot loss history#####
    visualizer.plot_history(
        total_loss_train,
        total_loss_val,
        total_loss_test,
        task_loss_train_sum,
        task_loss_val_sum,
        task_loss_test_sum,
        task_loss_train,
        task_loss_val,
        task_loss_test,
        model.loss_weights,
        config["Variables_of_interest"]["output_names"],
    )


def output_denormalize(y_minmax, true_values, predicted_values):
    # Fixme, should be improved later
    for ihead in range(len(y_minmax)):
        for isamp in range(len(predicted_values[0])):
            for iatom in range(len(predicted_values[ihead][0])):
                ymin = y_minmax[ihead][0][iatom]
                ymax = y_minmax[ihead][1][iatom]

                predicted_values[ihead][isamp][iatom] = (
                    predicted_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )
                true_values[ihead][isamp][iatom] = (
                    true_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )

    return true_values, predicted_values


def train(loader, model, opt, verbosity):

    device = next(model.parameters()).device
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)

    model.train()

    total_error = 0
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
        opt.zero_grad()

        pred = model(data)
        loss, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)

        loss.backward()
        opt.step()
        total_error += loss.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs
    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def validate(loader, model, verbosity):

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)

        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def test(loader, model, verbosity):

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    true_values = [[] for _ in range(model.num_heads)]
    predicted_values = [[] for _ in range(model.num_heads)]
    IImean = [i for i in range(sum(model.head_dims))]
    if model.ilossweights_nll == 1:
        IImean = [i for i in range(sum(model.head_dims) + model.num_heads)]
        [
            IImean.remove(sum(model.head_dims[: ihead + 1]) + (ihead + 1) * 1 - 1)
            for ihead in range(model.num_heads)
        ]
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)

        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

        ytrue = torch.reshape(data.y, (-1, sum(model.head_dims)))
        for ihead in range(model.num_heads):
            isum = sum(model.head_dims[: ihead + 1])
            true_values[ihead].extend(
                ytrue[:, isum - model.head_dims[ihead] : isum].tolist()
            )
            predicted_values[ihead].extend(
                pred[:, IImean[isum - model.head_dims[ihead] : isum]].tolist()
            )

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
        true_values,
        predicted_values,
    )


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
        raise ValueError('Must provide "total" OR "train", "test", "validate" data paths: ', dataset_names)

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
        raw_dataset = config["name"]
        raw_datasets = ["CuAu_32atoms", "FePt_32atoms", "FeSi_1024atoms", "unit_test"]
        if raw_dataset not in raw_datasets:
            print("WARNING: requested serialized dataset does not exist.")
            return

        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        if not os.path.exists(serialized_dir):
            os.mkdir(serialized_dir)
        serialized_dataset_dir = os.path.join(serialized_dir, raw_dataset)

        if not os.path.exists(serialized_dataset_dir):
            for dataset_name, raw_data_path in config["path"]["raw"].items():
                loader = RawDataLoader()
                if not os.path.isabs(raw_data_path):
                    raw_data_path = os.path.join(os.getcwd(), raw_data_path)
                if not os.path.exists(raw_data_path):
                    os.mkdir(raw_data_path)
                loader.load_raw_data(
                    dataset_path=raw_data_path,
                    config=config,
                    dataset_type=dataset_name,
                )

    if dist.is_initialized():
        dist.barrier()
