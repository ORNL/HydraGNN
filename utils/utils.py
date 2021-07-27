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
import numpy as np

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


def get_comm_size_and_rank():
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


def setup_ddp():

    """ "Initialize DDP"""

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    distributed_data_parallelism = False
    world_size, world_rank = get_comm_size_and_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = "8889"

    if os.getenv("LSB_DJOB_HOSTFILE") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        import subprocess

        get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1 )".format(
            os.environ["LSB_DJOB_HOSTFILE"]
        )
        master_addr = str(subprocess.check_output(get_master, shell=True))[2:-3]
    elif os.getenv("SLURM_NODELIST") is not None:
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
        distributed_data_parallelism = True
    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return distributed_data_parallelism, world_size, world_rank


def train_validate_test_normal(
    model_wrapper,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config,
    model_with_config_name,
    plot_init_sol=True,
    plot_hist_sol=False,
):

    num_epoch = config["Training"]["num_epoch"]
    trainlib = []
    vallib = []
    testlib = []  # total loss tracking for train/vali/test
    tasklib = []
    tasklib_test = []
    tasklib_vali = []  # loss tracking for summation across all atoms/nodes
    tasklib_nodes = []
    tasklib_test_nodes = []
    tasklib_vali_nodes = []  # probably not needed

    if isinstance(model_wrapper, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model_wrapper.module
    else:
        model = model_wrapper

    x_atomfeature = []
    for data in test_loader.dataset:
        x_atomfeature.append(data.x)
    if plot_init_sol:  # visualizing of initial conditions
        test_rmse = test(test_loader, model, config["Architecture"]["output_dim"])
        true_values = test_rmse[3]
        predicted_values = test_rmse[4]
        for ihead in range(model.num_heads):
            visualizer = Visualizer(model_with_config_name)
            visualizer.add_test_values(
                true_values=true_values[ihead], predicted_values=predicted_values[ihead]
            )
            visualizer.create_scatter_plot_atoms(ihead, x_atomfeature, -1)
            visualizer.create_error_histogram_plot_atoms(ihead, x_atomfeature, -1)

    for epoch in range(0, num_epoch):
        train_mae, train_taskserr, train_taskserr_nodes = train(
            train_loader, model, optimizer, config["Architecture"]["output_dim"]
        )
        val_mae, val_taskserr, val_taskserr_nodes = validate(
            val_loader, model, config["Architecture"]["output_dim"]
        )
        test_rmse = test(test_loader, model, config["Architecture"]["output_dim"])
        scheduler.step(val_mae)
        if writer is not None:
            writer.add_scalar("train error", train_mae, epoch)
            writer.add_scalar("validate error", val_mae, epoch)
            writer.add_scalar("test error", test_rmse[0], epoch)
            for ivar in range(model.num_heads):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}"
        )
        print("Tasks MAE:", train_taskserr)

        trainlib.append(train_mae)
        vallib.append(val_mae)
        testlib.append(test_rmse[0])
        tasklib.append(train_taskserr)
        tasklib_vali.append(val_taskserr)
        tasklib_test.append(test_rmse[1])

        tasklib_nodes.append(train_taskserr_nodes)
        tasklib_vali_nodes.append(val_taskserr_nodes)
        tasklib_test_nodes.append(test_rmse[2])

        ###tracking the solution evolving with training
        if plot_hist_sol:
            true_values = test_rmse[3]
            predicted_values = test_rmse[4]
            for ihead in range(model.num_heads):
                visualizer = Visualizer(model_with_config_name)
                visualizer.add_test_values(
                    true_values=true_values[ihead],
                    predicted_values=predicted_values[ihead],
                )
                visualizer.create_scatter_plot_atoms(ihead, x_atomfeature, epoch)
                visualizer.create_error_histogram_plot_atoms(
                    ihead, x_atomfeature, epoch
                )

    # At the end of training phase, do the one test run for visualizer to get latest predictions
    test_rmse, test_taskserr, test_taskserr_nodes, true_values, predicted_values = test(
        test_loader, model, config["Architecture"]["output_dim"]
    )

    if config["Variables_of_interest"]["denormalize_output"] == "True":  ##output predictions with unit/not normalized
        true_values, predicted_values = output_denormalize(
            config["Variables_of_interest"]["y_minmax"], true_values, predicted_values
        )

    for ihead in range(model.num_heads):
        visualizer = Visualizer(model_with_config_name)
        visualizer.add_test_values(
            true_values=true_values[ihead], predicted_values=predicted_values[ihead]
        )
        visualizer.create_plot_global(ihead)
        visualizer.create_scatter_plot_atoms(ihead, x_atomfeature)
        visualizer.create_error_histogram_plot_atoms(ihead, x_atomfeature)

    ######plot loss history#####
    visualizer.plot_history(
        trainlib,
        vallib,
        testlib,
        tasklib,
        tasklib_vali,
        tasklib_test,
        tasklib_nodes,
        tasklib_vali_nodes,
        tasklib_test_nodes,
        model.loss_weights,
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


def train(loader, model_wrapper, opt, output_dim):

    if isinstance(model_wrapper, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model_wrapper.module
    else:
        model = model_wrapper

    device = next(model.parameters()).device
    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)

    model.train()

    total_error = 0
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
            loss, tasks_rmse, tasks_nodes = model.module.loss_rmse(pred, data.y)
        else:
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
def validate(loader, model_wrapper, output_dim):

    if isinstance(model_wrapper, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model_wrapper.module
    else:
        model = model_wrapper

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
            error, tasks_rmse, tasks_nodes = model.module.loss_rmse(pred, data.y)
        else:
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
def test(loader, model_wrapper, output_dim):

    if isinstance(model_wrapper, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model_wrapper.module
    else:
        model = model_wrapper

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
    for data in tqdm(loader):
        data = data.to(device)
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            pred = model.module(data)
        else:
            pred = model(data)

        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            error, tasks_rmse, tasks_nodes = model.module.loss_rmse(pred, data.y)
        else:
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
    distributed_data_parallelism: bool = False,
):
    if chosen_dataset_option in [item.value for item in Dataset]:
        dataset_chosen = load_data(chosen_dataset_option, config)
        return split_dataset(
            dataset=dataset_chosen,
            batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
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
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:

            return combine_and_split_datasets(
                dataset1=dataset_CuAu,
                dataset2=dataset_FePt,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
            return combine_and_split_datasets(
                dataset1=dataset_FePt,
                dataset2=dataset_CuAu,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
                distributed_data_parallelism=distributed_data_parallelism,
            )
        elif chosen_dataset_option == Dataset.FePt_FeSi_SHUFFLE:
            dataset_FePt.extend(dataset_FeSi)
            dataset_combined = dataset_FePt
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
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
    transform_raw_data_to_serialized(config["Dataset"])
    files_dir = (
        f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
    )

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset = loader.load_serialized_data(
        dataset_path=files_dir,
        config=config["NeuralNetwork"],
    )

    return dataset


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
            loader = RawDataLoader()
            raw_data_path = config["path"]
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                os.mkdir(raw_data_path)
            loader.load_raw_data(dataset_path=raw_data_path, config=config)

    if dist.is_initialized():
        dist.barrier()
