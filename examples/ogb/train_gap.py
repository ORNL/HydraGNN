import os, json
import matplotlib.pyplot as plt
from ogb_utils import *

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import pickle

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm

def info(*args, logtype='info', sep=' '):
    getattr(logging, logtype)(sep.join(map(str, args)))

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

parser = argparse.ArgumentParser()
parser.add_argument('inputfilesubstr', help='input file substr')
parser.add_argument('--sampling', type=float, help='sampling ratio', default=None)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

## Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%%(asctime)s,%%(msecs)d %%(levelname)s (rank %d): %%(message)s"%(rank),
    datefmt="%H:%M:%S",
)

graph_feature_names = ["GAP"]
dirpwd = os.path.dirname(__file__)
datafile = os.path.join(dirpwd, "dataset/pcqm4m_gap.csv")
trainset_statistics = os.path.join(dirpwd, "dataset/statistics.pkl")
##################################################################################################################
inputfilesubstr = args.inputfilesubstr
input_filename = os.path.join(dirpwd, "ogb_" + inputfilesubstr + ".json")
##################################################################################################################
# Configurable run choices (JSON file that accompanies this example script).
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["output_names"] = [
    graph_feature_names[item] for ihead, item in enumerate(var_config["output_index"])
]
var_config["input_node_feature_names"] = node_attribute_names
ymax_feature, ymin_feature, ymean_feature, ystd_feature = get_trainset_stat(
    trainset_statistics
)
var_config["ymean"] = ymean_feature.tolist()
var_config["ystd"] = ystd_feature.tolist()
##################################################################################################################
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
norm_yflag = False  # True
smiles_sets, values_sets = datasets_load(datafile, sampling=args.sampling)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
    if norm_yflag:
        valueset = (valueset - torch.tensor(var_config["ymean"])) / torch.tensor(
            var_config["ystd"]
        )
        print(valueset[:, 0].mean(), valueset[:, 0].std())
        print(valueset[:, 1].mean(), valueset[:, 1].std())
        print(valueset[:, 2].mean(), valueset[:, 2].std())

    rx = list(nsplit(range(len(smileset)), comm_size))[rank]
    info ("Subset range:", rx.start, rx.stop)
    ## local portion
    _smileset = smileset[rx.start:rx.stop]
    _valueset = valueset[rx.start:rx.stop]
    info ("local smileset size:", len(_smileset))

    for smilestr, ytarget in iterate_tqdm(zip(_smileset, _valueset), verbosity, total=len(_smileset)):
        data = generate_graphdata(smilestr, ytarget,var_config)
        dataset_lists[idataset].append(data)

## local data
_trainset = dataset_lists[0]
_valset = dataset_lists[1]
_testset = dataset_lists[2]

with open('_ogb-%d.pickle'%rank, 'wb') as f:
    pickle.dump([_trainset, _valset, _testset], f)

## Collect all datas with MPI_Allgather
trainset_list = comm.allgather(_trainset)
valset_list = comm.allgather(_valset)
testset_list = comm.allgather(_testset)

trainset = list(chain(*trainset_list))
valset = list(chain(*valset_list))
testset = list(chain(*testset_list))

(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)


config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"]["Architecture"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

log_name = "ogb_" + inputfilesubstr + "_eV_fullx"
hydragnn.utils.setup_log(log_name)
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)
##################################################################################################################

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
)

hydragnn.utils.save_model(model, log_name)
hydragnn.utils.print_timers(verbosity)

if rank > 0:
    sys.exit(0)
##################################################################################################################
for ifeat in range(len(var_config["output_index"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.scatter(
        range(len(trainset)),
        [trainset[i].cpu().y[ifeat] for i in range(len(trainset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.scatter(
        range(len(valset)),
        [valset[i].cpu().y[ifeat] for i in range(len(valset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.scatter(
        range(len(testset)),
        [testset[i].cpu().y[ifeat] for i in range(len(testset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/ogb_train_val_test_"
        + var_config["output_names"][ifeat]
        + ".png"
    )
    plt.close()

for ifeat in range(len(var_config["input_node_features"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.plot(
        [
            item
            for i in range(len(trainset))
            for item in trainset[i].x[:, ifeat].tolist()
        ],
        "bo",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.plot(
        [item for i in range(len(valset)) for item in valset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.plot(
        [item for i in range(len(testset)) for item in testset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/ogb_train_val_test_"
        + var_config["input_node_feature_names"][ifeat]
        + ".png"
    )
    plt.close()
