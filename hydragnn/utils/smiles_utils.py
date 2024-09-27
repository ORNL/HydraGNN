import sys
import torch
import pickle, csv

#########################################################
# from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import hydragnn

##################################################################################################################
##################################################################################################################


def get_node_attribute_name(types={}):
    atom_attr_name = ["" for k in range(len(types))]
    for k, idx in types.items():
        atom_attr_name[idx] = "atom"+k
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

def get_edge_attribute_name():
    names = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    return names, [1]*len(names)

def generate_graphdata_from_smilestr(simlestr, ytarget, types={}, var_config=None, get_positions=False, pretrained=False):

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(simlestr, ps)  # , sanitize=False , removeHs=False)

    if pretrained:
        data = generate_graphdata_from_rdkit_molecule_pt(
            mol, ytarget, types, var_config=var_config, get_positions=get_positions
        )

    else:
        data = generate_graphdata_from_rdkit_molecule(
            mol, ytarget, types, var_config=var_config, get_positions=get_positions
        )

    return data


def generate_graphdata_from_rdkit_molecule(
    mol, ytarget, types={}, atomicdescriptors_torch_tensor=None, var_config=None, get_positions=False
):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    mol = Chem.AddHs(mol)
    N = mol.GetNumAtoms()

    if get_positions and mol.GetNumConformers() == 0:
        Chem.EmbedMolecule(mol, randomSeed=42, maxAttempts=10)
        Chem.MMFFOptimizeMolecule(mol)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    coordinates = []
    for atom in mol.GetAtoms():
        type_idx.append(types.get(atom.GetSymbol(), 0))
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        if get_positions:
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coordinates.append([pos.x, pos.y, pos.z])

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

    x = (
        torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
        .t()
        .contiguous()
    )

    if len(types) > 0:
        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x = torch.cat([x1.to(torch.float), x], dim=-1)

    if atomicdescriptors_torch_tensor is not None:
        assert (
            atomicdescriptors_torch_tensor.shape[0] == x.shape[0]
        ), "tensor of atomic descriptors MUST have the number of rows equal to the number of atoms in the molecule"
        x = torch.cat([x, atomicdescriptors_torch_tensor], dim=-1).to(torch.float)

    y = ytarget  # .squeeze()
    
    if get_positions:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=torch.tensor(coordinates, dtype=torch.float))
    else:
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

def generate_graphdata_from_rdkit_molecule_pt(
    mol, ytarget, types={}, atomicdescriptors_torch_tensor=None, var_config=None, get_positions=False
):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    mol = Chem.AddHs(mol)
    N = mol.GetNumAtoms()

    if get_positions and mol.GetNumConformers() == 0:
        Chem.EmbedMolecule(mol, randomSeed=42, maxAttempts=10)
        Chem.MMFFOptimizeMolecule(mol)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    coordinates = []
    for atom in mol.GetAtoms():
        type_idx.append(types.get(atom.GetSymbol(), 0))
        atomic_number.append(atom.GetAtomicNum())

        if get_positions:
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coordinates.append([pos.x, pos.y, pos.z])

    z = torch.tensor(atomic_number, dtype=torch.long)


    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float) # this is ultimately overwritten

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = (
        torch.tensor([atomic_number], dtype=torch.float)
        .t()
        .contiguous()
    )
    x = torch.cat([x1, torch.tensor(coordinates, dtype=torch.float)], dim=-1)
    if len(types) > 0:
        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x = torch.cat([x1.to(torch.float), x], dim=-1)

    if atomicdescriptors_torch_tensor is not None:
        assert (
            atomicdescriptors_torch_tensor.shape[0] == x.shape[0]
        ), "tensor of atomic descriptors MUST have the number of rows equal to the number of atoms in the molecule"
        x = torch.cat([x, atomicdescriptors_torch_tensor], dim=-1).to(torch.float)

    y = ytarget  # .squeeze()
    
    if get_positions:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=torch.tensor(coordinates, dtype=torch.float))
    else:
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
