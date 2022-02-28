import os, json
import torch

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
import pickle, csv
import matplotlib.pyplot as plt
import sys

#########################################################
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data

##################################################################################################################
node_attribute_names = [
    "atomH",
    "atomC",
    "atomN",
    "atomO",
    "atomF",
    "atomicnumber",
    "IsAromatic",
    "HSP",
    "HSP2",
    "HSP3",
    "Hprop",
]
graph_feature_names = ["HOMO (eV)", "LUMO (eV)", "GAP (eV)"]
HAR2EV = 27.211386246
datafile_cut = os.path.join(os.path.dirname(__file__), "dataset/gdb9_gap_cut.csv")
trainvaltest_splitlists = os.path.join(
    os.path.dirname(__file__), "dataset/qm9_train_test_val_idx_lists.pkl"
)
trainset_statistics = os.path.join(
    os.path.dirname(__file__), "dataset/qm9_statistics.pkl"
)
input_filename = os.path.join(os.path.dirname(__file__), "qm9_orbitals.json")
##################################################################################################################
# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

# Configurable run choices (JSON file that accompanies this example script).
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["output_names"] = graph_feature_names
var_config["input_node_feature_names"] = node_attribute_names

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
##################################################################################################################
def get_splitlists(filedir):
    with open(filedir, "rb") as f:
        train_filelist = pickle.load(f)
        val_filelist = pickle.load(f)
        test_filelist = pickle.load(f)
    return train_filelist, val_filelist, test_filelist


def get_trainset_stat(filedir):
    with open(filedir, "rb") as f:
        max_feature = pickle.load(f)
        min_feature = pickle.load(f)
        mean_feature = pickle.load(f)
        std_feature = pickle.load(f)
    return max_feature, min_feature, mean_feature, std_feature


def datasets_load(datafile, splitlistfile):
    train_filelist, val_filelist, test_filelist = get_splitlists(splitlistfile)
    trainset = []
    valset = []
    testset = []
    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainidxs = []
    validxs = []
    testidxs = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if row[0] in train_filelist:
                trainsmiles.append(row[1])
                trainset.append([float(x) * HAR2EV for x in row[2:]])
                trainidxs.append(int(row[0][4:]))
            elif row[0] in val_filelist:
                valsmiles.append(row[1])
                valset.append([float(x) * HAR2EV for x in row[2:]])
                validxs.append(int(row[0][4:]))
            elif row[0] in test_filelist:
                testsmiles.append(row[1])
                testset.append([float(x) * HAR2EV for x in row[2:]])
                testidxs.append(int(row[0][4:]))
            else:
                print("unknown file name: ", row[0])
                sys.exit(0)
    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        [trainidxs, validxs, testidxs],
    )


def generate_graphdata(idx, simlestr, ytarget):
    types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    mol = Chem.MolFromSmiles(simlestr, sanitize=False)  # , removeHs=False)
    N = mol.GetNumAtoms()

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = (
        torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
        .t()
        .contiguous()
    )
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    y = ytarget.squeeze()

    data = Data(x=x, z=z, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=idx)

    hydragnn.preprocess.update_predicted_values(
        var_config["type"],
        var_config["output_index"],
        data,
    )

    device = hydragnn.utils.get_device()
    return data.to(device)


##################################################################################################################
##################################################################################################################
smiles_sets, values_sets, idxs_sets = datasets_load(
    datafile_cut, trainvaltest_splitlists
)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset, idxset) in enumerate(
    zip(smiles_sets, values_sets, idxs_sets)
):
    for smilestr, ytarget, idx in zip(smileset, valueset, idxset):
        dataset_lists[idataset].append(generate_graphdata(idx, smilestr, ytarget))
trainset = dataset_lists[0]
valset = dataset_lists[1]
testset = dataset_lists[2]

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

log_name = "qm9_orbitals"
# Enable print to log file.
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
##################################################################################################################
for ifeat in range(len(var_config["output_index"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.scatter(
        [trainset[i].cpu().idx for i in range(len(trainset))],
        [trainset[i].cpu().y[ifeat] for i in range(len(trainset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.scatter(
        [valset[i].cpu().idx for i in range(len(valset))],
        [valset[i].cpu().y[ifeat] for i in range(len(valset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.scatter(
        [testset[i].cpu().idx for i in range(len(testset))],
        [testset[i].cpu().y[ifeat] for i in range(len(testset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/qm9_train_val_test_"
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
        + "/qm9_train_val_test_"
        + var_config["input_node_feature_names"][ifeat]
        + ".png"
    )
    plt.close()
