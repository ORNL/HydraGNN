import os, json
import matplotlib.pyplot as plt
from ogb_utils import *

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch

import warnings
warnings.filterwarnings("error")

class AdioGGO:
    def __init__(self, filename, comm):
        self.filename = filename
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dataset = list()
        self.adios = ad2.ADIOS()
        self.io = self.adios.DeclareIO(self.filename)

    def add(self, data: torch_geometric.data.Data):
        if isinstance(data, list):
            for x in data:
                self.dataset.append(x)
        elif isinstance(data, torch_geometric.data.Data):
            self.dataset.append(data)
        else:
            raise Exception("Unsuppored data type yet.")

    def save(self):
        t0 = time.time()
        info('Adios saving:', self.filename)
        ns = self.comm.allgather(len(self.dataset))
        ns_offset = sum(ns[:self.rank])

        self.io.DefineAttribute("ndata", np.array(sum(ns)))
        if len(self.dataset)>0:
            data = self.dataset[0]
            self.io.DefineAttribute("keys", data.keys)
            keys = sorted(data.keys)

        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        for k in keys:
            arr_list = [ data[k].cpu().numpy() for data in self.dataset ]
            m0 = np.min([ x.shape for x in arr_list ], axis=0)
            m1 = np.max([ x.shape for x in arr_list ], axis=0)
            wh = np.where(m0 != m1)[0]
            assert (len(wh) < 2)
            vdim = wh[0] if len(wh) == 1 else 1
            val = np.concatenate(arr_list, axis=vdim)
            assert (val.data.contiguous)
            shape_list = self.comm.allgather(list(val.shape))
            offset = [0,]*len(val.shape)
            for i in range(rank):
                offset[vdim] += shape_list[i][vdim]
            global_shape = shape_list[0]
            for i in range(1, self.size):
                global_shape[vdim] += shape_list[i][vdim]
            # info ("k,val shape", k, global_shape, offset, val.shape)
            var = self.io.DefineVariable(k, val, global_shape, offset, val.shape, ad2.ConstantDims)
            self.writer.Put(var, val, ad2.Mode.Sync)

            self.io.DefineAttribute("%s/variable_dim"%k, np.array(vdim))

            vcount = np.array([ x.shape[vdim] for x in arr_list ])
            assert (len(vcount) == len(self.dataset))
            
            offset_arr = np.zeros_like(vcount)
            offset_arr[1:] = np.cumsum(vcount)[:-1]
            offset_arr += offset[vdim]

            var = self.io.DefineVariable("%s/variable_count"%k, vcount, [sum(ns),], [ns_offset,], [len(vcount),], ad2.ConstantDims)
            self.writer.Put(var, vcount, ad2.Mode.Sync)

            var = self.io.DefineVariable("%s/variable_offset"%k, offset_arr, [sum(ns),], [ns_offset,], [len(vcount),], ad2.ConstantDims)
            self.writer.Put(var, offset_arr, ad2.Mode.Sync)

        self.writer.Close()
        t1 = time.time()
        info("Adios saving time (sec): ", (t1-t0))

    def load(self):
        t0 = time.time()
        info('Adios reading:', self.filename)
        with ad2.open(self.filename, "r",  MPI.COMM_SELF) as f:
            vars = f.available_variables()
            keys = f.read_attribute_string('keys')
            ndata = f.read_attribute('ndata').item()

            variable_count = dict()
            variable_offset = dict()
            variable_dim = dict()
            data = dict()
            for k in keys:
                variable_count[k] = f.read("%s/variable_count"%k)
                variable_offset[k] = f.read("%s/variable_offset"%k)
                variable_dim[k] = f.read_attribute("%s/variable_dim"%k).item()
                ## load full data first
                data[k] = f.read(k)
            t2 = time.time()
            info("Adios reading time (sec): ", (t2-t0))

            for i in iterate_tqdm(range(ndata), verbosity_level=2):
                data_object = torch_geometric.data.Data()
                for k in keys:
                    shape = f.available_variables()[k]['Shape']
                    ishape = [ int(x.strip(','), 16) for x in shape.strip().split() ]
                    start = [0,] * len(ishape)
                    count = ishape
                    vdim = variable_dim[k]
                    start[vdim] = variable_offset[k][i]
                    count[vdim] = variable_count[k][i]
                    slice_list = list()
                    for n0,n1 in zip(start, count):
                        slice_list.append(slice(n0,n0+n1))
                    val = data[k][tuple(slice_list)]
                                        
                    _ = torch.tensor(val)
                    exec("data_object.%s = _" % (k))
                self.dataset.append(data_object)
        
        t1 = time.time()
        info("Data loading time (sec): ", (t1-t0))
        return self.dataset
            

    def save_v1(self):
        t0 = time.time()
        info('Adios saving:', self.filename)
        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        ns = self.comm.allgather(len(self.dataset))
        ns.insert(0, 0)
        offset = sum(ns[:rank+1])
        
        self.io.DefineAttribute("ndata", np.array(sum(ns)))
        if len(self.dataset)>0:
            self.io.DefineAttribute("keys", self.dataset[0].keys)

        for i, data in enumerate(self.dataset):
            varinfo_list = list()
            for k in data.keys:
                val = data[k].cpu().numpy()
                assert (val.data.contiguous)
                varinfo_list.append((k, val))
            
            gname = 'data_%d'%(i+offset)
            for vname, val in varinfo_list:
                var = self.io.DefineVariable("%s/%s"%(gname, vname), val, [], [], val.shape, ad2.ConstantDims)
                self.writer.Put(var, val, ad2.Mode.Sync)
            
        self.writer.Close()
        t1 = time.time()
        info("Adios saving time (sec): ", (t1-t0))
    
    def load_v1(self):
        t0 = time.time()
        info('Adios reading:', self.filename)
        with ad2.open(self.filename, "r",  MPI.COMM_SELF) as f:
            keys = f.read_attribute_string('keys')
            ndata = f.read_attribute('ndata').item()
            vars = f.available_variables()
            for i in range(ndata):
                data_object = torch_geometric.data.Data()
                for k in keys:
                    _ = torch.tensor(f.read("data_%d/%s"%(i, k)))
                    exec("data_object.%s = _" % (k))
                self.dataset.append(data_object)

        t1 = time.time()
        info("Adios reading time (sec): ", (t1-t0))
        return self.dataset

class OGBDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        t0 = time.time()
        self.filename = filename
        info('Adios reading:', self.filename)
        with ad2.open(self.filename, "r",  MPI.COMM_SELF) as f:
            self.vars = f.available_variables()
            self.keys = f.read_attribute_string('keys')
            self.ndata = f.read_attribute('ndata').item()

            self.variable_count = dict()
            self.variable_offset = dict()
            self.variable_dim = dict()
            self.data = dict()
            for k in self.keys:
                self.variable_count[k] = f.read("%s/variable_count"%k)
                self.variable_offset[k] = f.read("%s/variable_offset"%k)
                self.variable_dim[k] = f.read_attribute("%s/variable_dim"%k).item()
                ## load full data first
                self.data[k] = f.read(k)
            t2 = time.time()
            info("Adios reading time (sec): ", (t2-t0))
        t1 = time.time()
        info("Data loading time (sec): ", (t1-t0))

    def __len__(self):
        return self.ndata
        
    def __getitem__(self, idx):
        data_object = torch_geometric.data.Data()
        for k in self.keys:
            shape = self.vars[k]['Shape']
            ishape = [ int(x.strip(','), 16) for x in shape.strip().split() ]
            start = [0,] * len(ishape)
            count = ishape
            vdim = self.variable_dim[k]
            start[vdim] = self.variable_offset[k][idx]
            count[vdim] = self.variable_count[k][idx]
            slice_list = list()
            for n0,n1 in zip(start, count):
                slice_list.append(slice(n0,n0+n1))
            val = self.data[k][tuple(slice_list)]
                                
            _ = torch.tensor(val)
            exec("data_object.%s = _" % (k))
        return data_object

def info(*args, logtype='info', sep=' '):
    getattr(logging, logtype)(sep.join(map(str, args)))

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

parser = argparse.ArgumentParser()
parser.add_argument('inputfilesubstr', help='input file substr')
parser.add_argument('--sampling', type=float, help='sampling ratio', default=None)
parser.add_argument('--notrain', action='store_true')
parser.add_argument('--readbp', action='store_true')
parser.add_argument('--usedsclass', action='store_true')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

## Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%%(levelname)s (rank %d): %%(message)s"%(rank),
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

if (not args.readbp) and (not args.usedsclass):
    norm_yflag = False  # True
    smiles_sets, values_sets = datasets_load(datafile, sampling=args.sampling, seed=43)
    info([ len(x) for x in values_sets ])
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
        info ("subset range:", idataset, len(smileset), rx.start, rx.stop)
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

    adwriter = AdioGGO("ogb_gap_trainset.bp", comm)
    adwriter.add(_trainset)
    adwriter.save()

    adwriter = AdioGGO("ogb_gap_valset.bp", comm)
    adwriter.add(_valset)
    adwriter.save()

    adwriter = AdioGGO("ogb_gap_testset.bp", comm)
    adwriter.add(_testset)
    adwriter.save()
elif args.readbp:
    adreader = AdioGGO("ogb_gap_trainset.bp", comm)
    trainset = adreader.load()

    adreader = AdioGGO("ogb_gap_valset.bp", comm)
    valset = adreader.load()

    adreader = AdioGGO("ogb_gap_testset.bp", comm)
    testset = adreader.load()

    info("Adios load")
    info("trainset,valset,testset size: %d %d %d"%(len(trainset), len(valset), len(testset)))

elif args.usedsclass:
    trainset = OGBDataset("ogb_gap_trainset.bp")
    valset = OGBDataset("ogb_gap_valset.bp")
    testset = OGBDataset("ogb_gap_testset.bp")

if args.notrain:
    sys.exit(0)

(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)

t0 = time.time()
config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
t1 = time.time()
info("update_config (sec): ", (t1-t0))

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
