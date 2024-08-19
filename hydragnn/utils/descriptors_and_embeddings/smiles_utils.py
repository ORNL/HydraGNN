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

##################################################################################################################
##################################################################################################################


def get_node_attribute_name(types):
    atom_attr_name = ["atom" + k for k in types]
    extra_attr_name = [
        "atomicnumber",
        "IsAromatic",
        "HSP",
        "HSP2",
        "HSP3",
        "Hprop",
    ]
    name_list = atom_attr_name + extra_attr_name
    dims_list = [
        1,
    ] * len(name_list)
    return name_list, dims_list


def generate_graphdata_from_smilestr(simlestr, ytarget, types, var_config=None):

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(simlestr, ps)  # , sanitize=False , removeHs=False)

    data = generate_graphdata_from_rdkit_molecule(
        mol, ytarget, types, var_config=var_config
    )

    return data


def generate_graphdata_from_rdkit_molecule(
    mol, ytarget, types, atomicdescriptors_torch_tensor=None, var_config=None
):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

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

    if atomicdescriptors_torch_tensor is not None:
        assert (
            atomicdescriptors_torch_tensor.shape[0] == x.shape[0]
        ), "tensor of atomic descriptors MUST have the number of rows equal to the number of atoms in the molecule"
        x = torch.cat([x, atomicdescriptors_torch_tensor], dim=-1).to(torch.float)

    y = ytarget  # .squeeze()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if var_config is not None:
        hydragnn.preprocess.update_predicted_values(
            var_config["type"],
            var_config["output_index"],
            var_config["graph_feature_dims"],
            var_config["input_node_feature_dims"],
            data,
        )

    return data
