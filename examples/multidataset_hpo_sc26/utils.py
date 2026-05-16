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
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from mpi4py import MPI

from hydragnn.preprocess import create_dataloaders

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, adios2_open
except ImportError:
    AdiosDataset = None
    adios2_open = None


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


class NormalizedDataset:
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    @staticmethod
    def _normalize_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor.view(1, 1)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        data = self.base_dataset[index]

        for name in [
            "chemical_composition",
            "energy",
            "energy_per_atom",
            "y",
            "graph_attr",
            "natoms",
            "pbc",
        ]:
            if hasattr(data, name):
                value = getattr(data, name)
                if torch.is_tensor(value):
                    setattr(data, name, self._normalize_graph_tensor(value))

        if hasattr(data, "atomic_numbers"):
            value = data.atomic_numbers
            if torch.is_tensor(value):
                if value.dim() == 0:
                    data.atomic_numbers = value.view(1)
                elif value.dim() == 2 and value.size(-1) == 1:
                    data.atomic_numbers = value.squeeze(-1)

        if hasattr(data, "x"):
            value = data.x
            if torch.is_tensor(value) and value.dim() == 1:
                data.x = value.unsqueeze(-1)

        if hasattr(data, "natoms") and torch.is_tensor(data.natoms):
            data.natoms = data.natoms.to(dtype=torch.long)

        if hasattr(data, "dataset_name") and torch.is_tensor(data.dataset_name):
            data.dataset_name = data.dataset_name.to(dtype=torch.long)

        if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
            data.edge_index = data.edge_index.to(dtype=torch.long)

        if hasattr(data, "atomic_numbers") and torch.is_tensor(data.atomic_numbers):
            data.atomic_numbers = data.atomic_numbers.to(dtype=torch.long)

        if hasattr(data, "chemical_composition") and torch.is_tensor(
            data.chemical_composition
        ):
            data.chemical_composition = data.chemical_composition.to(
                dtype=torch.float32
            )

        if hasattr(data, "energy") and torch.is_tensor(data.energy):
            data.energy = data.energy.to(dtype=torch.float32)

        if hasattr(data, "energy_per_atom") and torch.is_tensor(data.energy_per_atom):
            data.energy_per_atom = data.energy_per_atom.to(dtype=torch.float32)

        if hasattr(data, "forces") and torch.is_tensor(data.forces):
            data.forces = data.forces.to(dtype=torch.float32)

        if hasattr(data, "pos") and torch.is_tensor(data.pos):
            data.pos = data.pos.to(dtype=torch.float32)

        if hasattr(data, "graph_attr") and torch.is_tensor(data.graph_attr):
            data.graph_attr = data.graph_attr.to(dtype=torch.float32)

        if hasattr(data, "y") and torch.is_tensor(data.y):
            data.y = data.y.to(dtype=torch.float32)

        if hasattr(data, "edge_attr") and torch.is_tensor(data.edge_attr):
            data.edge_attr = data.edge_attr.to(dtype=torch.float32)

        if hasattr(data, "cell") and torch.is_tensor(data.cell):
            data.cell = data.cell.to(dtype=torch.float32)

        if hasattr(data, "edge_shifts") and torch.is_tensor(data.edge_shifts):
            data.edge_shifts = data.edge_shifts.to(dtype=torch.float32)

        if hasattr(data, "pbc") and torch.is_tensor(data.pbc):
            data.pbc = data.pbc.to(dtype=torch.float32)

        return data


def configure_variable_names(config):
    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims
    var_config["input_node_features"] = [0]


def resolve_selected_precision(
    args_precision: Optional[str], config: dict
) -> Tuple[str, str]:
    cfg_precision = (
        config.get("NeuralNetwork", {}).get("Training", {}).get("precision", None)
    )

    if args_precision is not None:
        source = "cli"
        raw_precision = args_precision
    elif cfg_precision is not None:
        source = "config.json"
        raw_precision = cfg_precision
    else:
        source = "built-in-default"
        raw_precision = "fp32"

    value = str(raw_precision).strip().lower()
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "single": "fp32",
        "float64": "fp64",
        "double": "fp64",
        "bfloat16": "bf16",
    }
    precision = aliases.get(value, value)
    return precision, source


def infer_num_branches(config: dict, model) -> int:
    arch = config.get("NeuralNetwork", {}).get("Architecture", {})
    output_heads = arch.get("output_heads", {})
    graph_heads = output_heads.get("graph") if isinstance(output_heads, dict) else None
    if isinstance(graph_heads, list) and len(graph_heads) > 0:
        return len(graph_heads)

    model_num_branches = getattr(model, "num_branches", None)
    if isinstance(model_num_branches, int) and model_num_branches > 0:
        return model_num_branches

    return 1


def load_multidataset_dataloaders(args, config, var_config):
    if AdiosDataset is None:
        raise ImportError("AdiosDataset is unavailable; install adios2 to use --multi")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not args.multi_model_list or args.multi_model_list.strip() == "":
        raise ValueError("--multi_model_list must be provided")

    modellist = [m for m in args.multi_model_list.split(",") if m.strip()]
    if len(modellist) == 0:
        raise ValueError("--multi_model_list resulted in zero entries")

    if rank == 0:
        pna_deg_list = []
        for model in modellist:
            fname = os.path.join(args.dataset_dir, f"{model}-v2.bp")
            with adios2_open(fname, "r", MPI.COMM_SELF) as f:
                f.__next__()
                attrs = f.available_attributes()
                pna_deg = None
                if "pna_deg" in attrs:
                    pna_deg = f.read_attribute("pna_deg")
                pna_deg_list.append(pna_deg)

        if all(p is None for p in pna_deg_list):
            pna_deg = None
        else:
            valid_pna_deg = [p for p in pna_deg_list if p is not None]
            intp_list = []
            mlen = min(len(p) for p in valid_pna_deg)
            for p in valid_pna_deg:
                x = np.linspace(0, 1, num=len(p))
                intp = np.interp(np.linspace(0, 1, num=mlen), x, p)
                intp_list.append(intp)
            pna_deg = (
                np.sum(np.stack(intp_list, axis=0), axis=0).astype(np.int64).tolist()
            )
    else:
        pna_deg = None

    pna_deg = comm.bcast(pna_deg, root=0)

    common_variable_names = [
        "pbc",
        "edge_attr",
        "energy_per_atom",
        "forces",
        "pos",
        "edge_index",
        "cell",
        "edge_shifts",
        "y",
        "chemical_composition",
        "natoms",
        "x",
        "energy",
        "graph_attr",
        "atomic_numbers",
    ]

    def build_mixed_split(split_label: str, split_index: int):
        datasets = []
        for model_name in modellist:
            fname = os.path.join(args.dataset_dir, f"{model_name}-v2.bp")
            dataset = AdiosDataset(
                fname,
                split_label,
                MPI.COMM_WORLD,
                keys=common_variable_names,
                var_config=var_config,
            )

            dataset.dataset_name_dict = {
                name.lower(): torch.tensor([[i]]) for i, name in enumerate(modellist)
            }

            dataset_len = len(dataset)
            subset_len = dataset_len
            if args.num_samples is not None:
                requested = (
                    args.num_samples
                    if split_index == 0
                    else max(args.num_samples // 10, 1)
                )
                subset_len = min(requested, dataset_len)

            dataset.setkeys(common_variable_names)
            dataset.setsubset(0, subset_len, preload=args.preload)
            datasets.append(NormalizedDataset(dataset))

            if rank == 0:
                print(
                    f"Mixed {split_label}: include {model_name} with {subset_len} samples"
                )

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    trainset = build_mixed_split("trainset", 0)
    valset = build_mixed_split("valset", 1)
    testset = build_mixed_split("testset", 2)

    if pna_deg is not None:
        trainset.pna_deg = pna_deg
        valset.pna_deg = pna_deg
        testset.pna_deg = pna_deg

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
        oversampling=False,
    )
    comm.Barrier()
    return train_loader, val_loader, test_loader


def load_single_dataset_chunk(
    args,
    config,
    var_config,
    model_name: str,
    model_index: int,
    modellist: list,
    pna_deg,
    chunk_start: int,
    chunk_size: int,
):
    """Load one chunk of a single dataset and return train/val loaders.

    Opens the dataset fresh each call so ADIOS2 metadata for previous
    chunks is released before the next chunk is loaded.

    chunk_start and chunk_size apply to the global sample index space
    (before rank partitioning by create_dataloaders).
    """
    if AdiosDataset is None:
        raise ImportError("AdiosDataset is unavailable; install adios2 to use --multi")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    common_variable_names = [
        "pbc",
        "edge_attr",
        "energy_per_atom",
        "forces",
        "pos",
        "edge_index",
        "cell",
        "edge_shifts",
        "y",
        "chemical_composition",
        "natoms",
        "x",
        "energy",
        "graph_attr",
        "atomic_numbers",
    ]

    fname = os.path.join(args.dataset_dir, f"{model_name}-v2.bp")

    def build_split(split_label: str, split_index: int):
        dataset = AdiosDataset(
            fname,
            split_label,
            MPI.COMM_WORLD,
            keys=common_variable_names,
            var_config=var_config,
        )
        dataset.dataset_name_dict = {
            name.lower(): torch.tensor([[i]]) for i, name in enumerate(modellist)
        }
        dataset_len = len(dataset)

        # For val, chunk_size is scaled down proportionally
        this_chunk_size = chunk_size if split_index == 0 else max(chunk_size // 10, 1)
        start = min(chunk_start if split_index == 0 else chunk_start // 10, dataset_len)
        end = min(start + this_chunk_size, dataset_len)

        if start >= end:
            start = 0
            end = min(1, dataset_len)

        dataset.setkeys(common_variable_names)
        dataset.setsubset(start, end, preload=args.preload)

        if rank == 0:
            print(
                f"  {split_label}: {model_name} chunk [{start}:{end}] ({end - start} samples)"
            )

        return NormalizedDataset(dataset)

    trainset = build_split("trainset", 0)
    valset = build_split("valset", 1)
    testset = build_split("testset", 2)

    if pna_deg is not None:
        trainset.pna_deg = pna_deg
        valset.pna_deg = pna_deg
        testset.pna_deg = pna_deg

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
        oversampling=False,
    )
    comm.Barrier()
    return train_loader, val_loader, test_loader


def load_single_dataset_dataloaders(
    args,
    config,
    var_config,
    model_name: str,
    model_index: int,
    modellist: list,
    pna_deg,
):
    """Load a full dataset (no chunking). Kept for backward compatibility and tiny loaders."""
    num_samples = getattr(args, "num_samples", None)
    chunk_start = 0
    chunk_size = num_samples if num_samples is not None else 10 ** 9  # effectively all

    return load_single_dataset_chunk(
        args,
        config,
        var_config,
        model_name=model_name,
        model_index=model_index,
        modellist=modellist,
        pna_deg=pna_deg,
        chunk_start=chunk_start,
        chunk_size=chunk_size,
    )


def get_modellist_and_pna_deg(args, var_config):
    """Read modellist and merged pna_deg from dataset metadata.
    Call once before the per-dataset loop.
    """
    if AdiosDataset is None:
        raise ImportError("AdiosDataset is unavailable; install adios2 to use --multi")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not args.multi_model_list or args.multi_model_list.strip() == "":
        raise ValueError("--multi_model_list must be provided")

    modellist = [m for m in args.multi_model_list.split(",") if m.strip()]
    if len(modellist) == 0:
        raise ValueError("--multi_model_list resulted in zero entries")

    if rank == 0:
        pna_deg_list = []
        for model in modellist:
            fname = os.path.join(args.dataset_dir, f"{model}-v2.bp")
            with adios2_open(fname, "r", MPI.COMM_SELF) as f:
                f.__next__()
                attrs = f.available_attributes()
                pna_deg = None
                if "pna_deg" in attrs:
                    pna_deg = f.read_attribute("pna_deg")
                pna_deg_list.append(pna_deg)

        if all(p is None for p in pna_deg_list):
            pna_deg = None
        else:
            valid_pna_deg = [p for p in pna_deg_list if p is not None]
            intp_list = []
            mlen = min(len(p) for p in valid_pna_deg)
            for p in valid_pna_deg:
                x = np.linspace(0, 1, num=len(p))
                intp = np.interp(np.linspace(0, 1, num=mlen), x, p)
                intp_list.append(intp)
            pna_deg = (
                np.sum(np.stack(intp_list, axis=0), axis=0).astype(np.int64).tolist()
            )
    else:
        pna_deg = None

    pna_deg = comm.bcast(pna_deg, root=0)
    return modellist, pna_deg


def build_dataset_name(data, branch_id: int) -> torch.Tensor:
    if hasattr(data, "dataset_name"):
        return torch.full_like(data.dataset_name, branch_id)
    return torch.full(
        (data.num_graphs, 1), branch_id, dtype=torch.long, device=data.x.device
    )


def energy_from_pred(pred) -> torch.Tensor:
    if isinstance(pred, (list, tuple)):
        energy = pred[0]
    elif isinstance(pred, dict) and "graph" in pred:
        energy = pred["graph"][0]
    else:
        energy = pred
    return energy.squeeze(-1)


def predict_branch_energy_forces(
    model, data, branch_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = build_dataset_name(data, branch_id)

    pred = model(data)
    energy_pred = energy_from_pred(pred)
    forces_pred = torch.autograd.grad(
        energy_pred,
        data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        retain_graph=False,
        create_graph=False,
    )[0]
    forces_pred = -forces_pred

    if original_dataset_name is None:
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name

    return energy_pred.detach(), forces_pred.detach()


def weighted_average(
    energy_preds: torch.Tensor,
    forces_preds: torch.Tensor,
    weights: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Energy: [num_branches, num_graphs] x [num_graphs, num_branches] -> [num_graphs]
    weighted_energy = torch.sum(weights * energy_preds.transpose(0, 1), dim=1)

    # Forces: vectorized over all branches in one operation
    # node_weights: [num_atoms, num_branches] — expand per-graph weights to per-atom
    # forces_preds: [num_branches, num_atoms, 3]
    # output:       [num_atoms, 3]
    node_weights = weights[batch]  # [num_atoms, num_branches]
    weighted_forces = torch.einsum("ab,bac->ac", node_weights, forces_preds)

    return weighted_energy, weighted_forces


def extract_dataset_ids(data, num_branches: int) -> torch.Tensor:
    if not hasattr(data, "dataset_name"):
        return torch.zeros(data.num_graphs, dtype=torch.long, device=data.batch.device)

    dataset_ids = data.dataset_name
    if not torch.is_tensor(dataset_ids):
        dataset_ids = torch.tensor(
            dataset_ids, dtype=torch.long, device=data.batch.device
        )

    dataset_ids = dataset_ids.to(device=data.batch.device, dtype=torch.long)

    if dataset_ids.dim() == 0:
        dataset_ids = dataset_ids.view(1)
    elif dataset_ids.dim() == 2 and dataset_ids.size(-1) == 1:
        dataset_ids = dataset_ids.squeeze(-1)
    elif dataset_ids.dim() > 2:
        dataset_ids = dataset_ids.view(dataset_ids.size(0), -1)
        if dataset_ids.size(1) != 1:
            raise ValueError(
                f"Unsupported dataset_name shape {tuple(data.dataset_name.shape)}"
            )
        dataset_ids = dataset_ids.squeeze(-1)

    if dataset_ids.numel() == 1 and data.num_graphs > 1:
        dataset_ids = dataset_ids.repeat(data.num_graphs)

    if dataset_ids.numel() != data.num_graphs:
        raise ValueError(
            f"dataset_name has {dataset_ids.numel()} entries but batch has num_graphs={data.num_graphs}"
        )

    if torch.any(dataset_ids < 0) or torch.any(dataset_ids >= num_branches):
        bad_min = int(dataset_ids.min().item())
        bad_max = int(dataset_ids.max().item())
        raise ValueError(
            f"dataset_name IDs out of range [0, {num_branches - 1}]: min={bad_min}, max={bad_max}"
        )

    return dataset_ids


def teacher_from_dataset_id(
    energy_preds: torch.Tensor,
    forces_preds: torch.Tensor,
    batch: torch.Tensor,
    dataset_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_branches = energy_preds.size(0)
    target_weights = F.one_hot(dataset_ids, num_classes=num_branches).to(
        dtype=energy_preds.dtype
    )
    return weighted_average(energy_preds, forces_preds, target_weights, batch)
