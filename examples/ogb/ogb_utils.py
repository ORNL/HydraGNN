import sys
import torch
import pickle, csv

#########################################################
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import hydragnn
import random

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


def datasets_load_gap(datafile):
    smiles = []
    strset = []
    yvals = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            smiles.append(row[0])
            strset.append(row[1])
            yvals.append(float(row[-1]))
    return smiles, yvals


def datasets_load(datafile, sampling=None, seed=None):
    if seed is not None:
        random.seed(seed)
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
            if (sampling is not None) and (random.random() > sampling):
                continue
            if row[1] == "train":
                trainsmiles.append(row[0])
                trainset.append([float(row[-1])])
            elif row[1] == "val":
                valsmiles.append(row[0])
                valset.append([float(row[-1])])
            elif row[1] == "test":
                testsmiles.append(row[0])
                testset.append([float(row[-1])])
            else:
                print("unknown file name: ", row[0])
                sys.exit(0)
    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
    )


##################################################################################################################
node_attribute_names = [
    "atomH",
    "atomB",
    "atomC",
    "atomN",
    "atomO",
    "atomF",
    "atomSi",
    "atomP",
    "atomS",
    "atomCl",
    "atomCa",
    "atomGe",
    "atomAs",
    "atomSe",
    "atomBr",
    "atomI",
    "atomMg",
    "atomTi",
    "atomGa",
    "atomZn",
    "atomAr",
    "atomBe",
    "atomHe",
    "atomAl",
    "atomKr",
    "atomV",
    "atomNa",
    "atomLi",
    "atomCu",
    "atomNe",
    "atomNi",
    "atomicnumber",
    "IsAromatic",
    "HSP",
    "HSP2",
    "HSP3",
    "Hprop",
]


def gapfromsmiles(smilestr, model):
    ## gap_true can be replaced by random numbers when use
    gap_rand = 0.0
    data_graph = generate_graphdata(smilestr, gap_rand)
    pred = model(data_graph)
    return pred[0][0].item()


def generate_graphdata_checkelement(simlestr, ytarget, var_config=None):
    types = {
        "H": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "F": 5,
        "Si": 6,
        "P": 7,
        "S": 8,
        "Cl": 9,
        "Ca": 10,
        "Ge": 11,
        "As": 12,
        "Se": 13,
        "Br": 14,
        "I": 15,
        "Mg": 16,
        "Ti": 17,
        "Ga": 18,
        "Zn": 19,
        "Ar": 20,
        "Be": 21,
        "He": 22,
        "Al": 23,
        "Kr": 24,
        "V": 25,
        "Na": 26,
        "Li": 27,
        "Cu": 28,
        "Ne": 29,
        "Ni": 30,
    }
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(simlestr, ps)  # , sanitize=False , removeHs=False)
    mol = Chem.AddHs(mol)
    N = mol.GetNumAtoms()

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        try:
           type_idx.append(types[atom.GetSymbol()])
        except:
           print(atom.GetSymbol())


def generate_graphdata(simlestr, ytarget, var_config=None):
    # types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    # B, C, N, O, F, Si, P, S, Cl, Ca, Ge, As, Se, Br, I
    types = {
        "H": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "F": 5,
        "Si": 6,
        "P": 7,
        "S": 8,
        "Cl": 9,
        "Ca": 10,
        "Ge": 11,
        "As": 12,
        "Se": 13,
        "Br": 14,
        "I": 15,
        "Mg": 16,
        "Ti": 17,
        "Ga": 18,
        "Zn": 19,
        "Ar": 20,
        "Be": 21,
        "He": 22,
        "Al": 23,
        "Kr": 24,
        "V": 25,
        "Na": 26,
        "Li": 27,
        "Cu": 28,
        "Ne": 29,
        "Ni": 30,
    }
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(simlestr, ps)  # , sanitize=False , removeHs=False)
    mol = Chem.AddHs(mol)
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
    # x = torch.tensor([atomic_number], dtype=torch.float).view(-1,1)
    # print(x)

    y = ytarget  # .squeeze()

    data = Data(x=x, z=z, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if var_config is not None:
        hydragnn.preprocess.update_predicted_values(
            var_config["type"],
            var_config["output_index"],
            data,
        )

    device = hydragnn.utils.get_device()
    return data
    #return data.to(device)
