import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleDataset

import hydragnn.utils.tracer as tr

from hydragnn.utils.print_utils import log
from hydragnn.utils import nsplit

try:
    from hydragnn.utils.adiosdataset import AdiosDataset
except ImportError:
    pass


## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


parser = argparse.ArgumentParser()
parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
parser.add_argument("--shmem", action="store_true", help="shmem")
parser.add_argument("--model_type", help="model_type", default="EGNN")
parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=5)
parser.add_argument("--num_conv_layers", type=int, help="num_conv_layers", default=6)
parser.add_argument("--num_headlayers", type=int, help="num_headlayers", default=2)
parser.add_argument("--dim_headlayers", type=int, help="dim_headlayers", default=10)
parser.add_argument("--log", help="log name", default="gfm_test")
args = parser.parse_args()
args.parameters = vars(args)


# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gfm_multitasking.json"
)
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Update the config dictionary with the suggested hyperparameters
config["NeuralNetwork"]["Architecture"]["model_type"] = args.parameters["model_type"]
config["NeuralNetwork"]["Architecture"]["hidden_dim"] = args.parameters["hidden_dim"]
config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = args.parameters[
    "num_conv_layers"
]

dim_headlayers = [
    args.parameters["dim_headlayers"] for i in range(args.parameters["num_headlayers"])
]

for head_type in config["NeuralNetwork"]["Architecture"]["output_heads"]:
    head_type["num_headlayers"] = args.parameters["num_headlayers"]
    head_type["dim_headlayers"] = dim_headlayers

if args.parameters["model_type"] not in ["EGNN", "SchNet", "DimeNet"]:
    config["NeuralNetwork"]["Architecture"]["equivariance"] = False

# Always initialize for multi-rank training.
comm_size, rank = hydragnn.utils.setup_ddp()

##################################################################################################################

comm = MPI.COMM_WORLD

log_name = args.log
# Enable print to log file.
hydragnn.utils.setup_log(log_name)

# Use built-in torch_geometric dataset.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
modelname = "GFM" if args.modelname is None else args.modelname

log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

tr.initialize()
tr.disable()
timer = Timer("load_data")
timer.start()

if args.format == "adios":
    info("Adios load")
    assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
    opt = {
        "preload": False,
        "shmem": args.shmem,
        "ddstore": args.ddstore,
        "ddstore_width": args.ddstore_width,
    }
    fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
    trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
    valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
    testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
elif args.format == "pickle":
    info("Pickle load")
    basedir = os.path.join(
        os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
    )
    trainset = SimplePickleDataset(
        basedir=basedir, label="trainset", var_config=var_config
    )
    valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
    testset = SimplePickleDataset(
        basedir=basedir, label="testset", var_config=var_config
    )
    # minmax_node_feature = trainset.minmax_node_feature
    # minmax_graph_feature = trainset.minmax_graph_feature
    pna_deg = trainset.pna_deg
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)
        # trainset.minmax_node_feature = minmax_node_feature
        # trainset.minmax_graph_feature = minmax_graph_feature
        trainset.pna_deg = pna_deg
elif args.format == "multi":
    info("Multi load")
    ## Reading multiple datasets, which requires the following arguments:
    ## --multi_model_list: the list datasets/model names
    ## --multi_process_list: the list of the number of processes
    modellist = args.multi_model_list.split(",")
    processlist = list(map(lambda x: int(x), args.multi_process_list.split(",")))
    assert comm_size == sum(processlist)
    colorlist = list()
    color = 0
    for n in processlist:
        for _ in range(n):
            colorlist.append(color)
        color += 1
    mycolor = colorlist[rank]
    mymodel = modellist[mycolor]

    local_comm = comm.Split(mycolor, rank)
    local_comm_rank = local_comm.Get_rank()
    local_comm_size = local_comm.Get_size()

    fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % mymodel)
    trainset = AdiosDataset(fname, "trainset", local_comm, var_config=var_config)
    valset = AdiosDataset(fname, "valset", local_comm, var_config=var_config)
    testset = AdiosDataset(fname, "testset", local_comm, var_config=var_config)

    ## Set local set
    for dataset in [trainset, valset, testset]:
        rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
        dataset.setsubset(rx)
    print(
        rank,
        "color,moddelname,len:",
        mycolor,
        mymodel,
        len(trainset),
        len(valset),
        len(testset),
    )

    assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width, "local": True}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)
else:
    raise NotImplementedError("No supported format: %s" % (args.format))

info(
    "trainset,valset,testset size: %d %d %d"
    % (len(trainset), len(valset), len(testset))
)

if args.ddstore:
    os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
    os.environ["HYDRAGNN_USE_ddstore"] = "1"

(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)

config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

# Print details of neural network architecture
print_model(model)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

# Run training with the given model and qm9 dataset.
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
)

hydragnn.utils.save_model(model, optimizer, log_name)
hydragnn.utils.print_timers(verbosity)
