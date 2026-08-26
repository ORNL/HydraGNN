##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Fault-tolerant, report-producing conversion of the official raw QM9 data."""

import json
from pathlib import Path

import rdkit
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.datasets.qm9 import conversion
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm


class RobustQM9(QM9):
    """QM9 raw processor that reports bad records and continues safely.

    Target rows are always indexed by the original SDF record index. Rejected
    records therefore never shift the structure-to-target correspondence.
    """

    def __init__(
        self,
        root,
        *,
        invalid_molecule_policy="report_and_skip",
        max_rejected_molecules=None,
        max_records=None,
        report_directory=None,
        **kwargs,
    ):
        if invalid_molecule_policy not in {"report_and_skip", "error"}:
            raise ValueError(
                "invalid_molecule_policy must be 'report_and_skip' or 'error'"
            )
        if max_rejected_molecules is not None and max_rejected_molecules < 0:
            raise ValueError("max_rejected_molecules must be non-negative")
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive or None")
        self.invalid_molecule_policy = invalid_molecule_policy
        self.max_rejected_molecules = max_rejected_molecules
        self.max_records = max_records
        self.report_directory = Path(
            report_directory or Path(root) / "preprocessing_report"
        )
        super().__init__(root, **kwargs)

    def _write_reports(self, records, summary):
        self.report_directory.mkdir(parents=True, exist_ok=True)
        jsonl = self.report_directory / "unconverted_molecules.jsonl"
        jsonl.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        (self.report_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _failure(record_index, stage, reason, error=None, name=None):
        return {
            "record_index": record_index,
            "qm9_id": record_index + 1,
            "name": name,
            "status": "rejected",
            "stage": stage,
            "exception": type(error).__name__ if error is not None else None,
            "reason": str(error) if error is not None else reason,
            "source": "gdb9.sdf",
        }

    def _convert_molecule(self, mol, record_index, targets):
        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        num_atoms = mol.GetNumAtoms()
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

        type_idx, atomic_number, aromatic, sp, sp2, sp3 = [], [], [], [], [], []
        for atom in mol.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(int(atom.GetIsAromatic()))
            hybridization = atom.GetHybridization()
            sp.append(int(hybridization == HybridizationType.SP))
            sp2.append(int(hybridization == HybridizationType.SP2))
            sp3.append(int(hybridization == HybridizationType.SP3))
        z = torch.tensor(atomic_number, dtype=torch.long)

        rows, cols, edge_types = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_types += 2 * [bonds[bond.GetBondType()]]
        edge_index = torch.tensor([rows, cols], dtype=torch.long).reshape(2, -1)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(bonds))
        permutation = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, permutation]
        edge_attr = edge_attr[permutation]

        row, col = edge_index
        hydrogens = (z == 1).float()
        num_hs = scatter(hydrogens[row], col, dim_size=num_atoms, reduce="sum").tolist()
        x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor(
            [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
        ).t()
        return Data(
            x=torch.cat([x1, x2], dim=-1),
            z=z,
            pos=pos,
            edge_index=edge_index,
            smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
            edge_attr=edge_attr,
            y=targets[record_index].unsqueeze(0),
            name=mol.GetProp("_Name") if mol.HasProp("_Name") else None,
            idx=record_index,
        )

    def process(self):
        with open(self.raw_paths[1], encoding="utf-8") as stream:
            target = [
                [float(value) for value in line.split(",")[1:20]]
                for line in stream.read().split("\n")[1:-1]
            ]
        targets = torch.tensor(target, dtype=torch.float)
        targets = torch.cat([targets[:, 3:], targets[:, :3]], dim=-1)
        targets = targets * conversion.view(1, -1)

        with open(self.raw_paths[2], encoding="utf-8") as stream:
            official_skip = {
                int(line.split()[0]) - 1 for line in stream.read().splitlines()[9:-2]
            }

        supplier = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list, unconverted = [], []
        rejected = filtered = official = seen = 0
        fatal_error = None
        for record_index, mol in enumerate(tqdm(supplier, desc="QM9 raw conversion")):
            if self.max_records is not None and record_index >= self.max_records:
                break
            seen += 1
            if record_index in official_skip:
                official += 1
                unconverted.append(
                    {
                        "record_index": record_index,
                        "qm9_id": record_index + 1,
                        "name": None,
                        "status": "official_exclusion",
                        "stage": "official_uncharacterized_list",
                        "exception": None,
                        "reason": "Listed by QM9 as uncharacterized",
                        "source": "gdb9.sdf",
                    }
                )
                continue

            if mol is None:
                failure = self._failure(
                    record_index,
                    "rdkit_parse",
                    "RDKit could not parse this SDF record; consult the RDKit "
                    "diagnostic emitted on stderr for the chemical parsing detail",
                )
            else:
                try:
                    data = self._convert_molecule(mol, record_index, targets)
                except Exception as error:  # Report the exact molecule and stage.
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
                    failure = self._failure(
                        record_index,
                        "conversion",
                        "Molecule conversion failed",
                        error,
                        name,
                    )
                else:
                    name = data.name
                    try:
                        keep = self.pre_filter is None or self.pre_filter(data)
                    except Exception as error:
                        failure = self._failure(
                            record_index,
                            "pre_filter",
                            "QM9 pre-filter failed",
                            error,
                            name,
                        )
                        keep = None
                    if keep is False:
                        filtered += 1
                        continue
                    if keep is None:
                        pass
                    elif self.pre_transform is None:
                        data_list.append(data)
                        continue
                    else:
                        try:
                            data = self.pre_transform(data)
                        except Exception as error:
                            failure = self._failure(
                                record_index,
                                "pre_transform",
                                "QM9 pre-transform failed",
                                error,
                                name,
                            )
                        else:
                            data_list.append(data)
                            continue

            rejected += 1
            unconverted.append(failure)
            print(
                "[QM9 preprocessing] rejected "
                f"record={record_index} qm9_id={record_index + 1} "
                f"stage={failure['stage']} reason={failure['reason']}",
                flush=True,
            )
            limit_exceeded = (
                self.max_rejected_molecules is not None
                and rejected > self.max_rejected_molecules
            )
            if self.invalid_molecule_policy == "error" or limit_exceeded:
                fatal_error = RuntimeError(
                    f"QM9 conversion rejected record {record_index}: "
                    f"{failure['reason']}"
                )
                break

        summary = {
            "source": "official raw QM9",
            "records_seen": seen,
            "converted": len(data_list),
            "official_exclusions": official,
            "filtered": filtered,
            "rejected": rejected,
            "completed": fatal_error is None,
            "invalid_molecule_policy": self.invalid_molecule_policy,
            "max_rejected_molecules": self.max_rejected_molecules,
            "max_records": self.max_records,
            "rdkit_version": rdkit.__version__,
            "torch_geometric_version": torch_geometric.__version__,
        }
        self._write_reports(unconverted, summary)
        print(
            "[QM9 preprocessing] "
            f"seen={seen} converted={len(data_list)} official={official} "
            f"filtered={filtered} rejected={rejected} "
            f"report={self.report_directory}",
            flush=True,
        )
        if fatal_error is not None:
            raise fatal_error
        self.save(data_list, self.processed_paths[0])
