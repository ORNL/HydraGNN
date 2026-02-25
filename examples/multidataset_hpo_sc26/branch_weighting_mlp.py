#!/usr/bin/env python3
"""Train a composition-conditioned MLP to weight branch predictions.

This script loads a pretrained multi-branch HydraGNN model, computes per-branch
energy/force predictions, and trains a small MLP on data.chemical_composition
that outputs per-branch weights for a weighted average prediction.
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import ConcatDataset

from mpi4py import MPI
from torch.distributed.device_mesh import init_device_mesh

import hydragnn
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.models.create import create_model_config
from hydragnn.utils.distributed import get_device, nsplit
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.train.train_validate_test import resolve_precision, move_batch_to_device

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, adios2_open
except ImportError:
    AdiosDataset = None
    adios2_open = None

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset


class _NormalizedDataset:
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


def _configure_variable_names(config):
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


def _load_multidataset_dataloaders(args, config, var_config):
    if args.format == "pickle":
        raise NotImplementedError("Multi-dataset loading from pickle is not supported")
    if AdiosDataset is None:
        raise ImportError("AdiosDataset is unavailable; install adios2 to use --multi")

    if args.ddstore:
        raise NotImplementedError(
            "--ddstore is not supported with mixed multi-dataset batching in branch_weighting_mlp.py"
        )
    if args.task_parallel or args.use_devicemesh:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                "INFO: Mixed multi-dataset mode ignores --task_parallel/--use_devicemesh and uses global DistributedSampler."
            )
    if args.oversampling:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                "INFO: Mixed multi-dataset mode ignores --oversampling and uses standard DistributedSampler."
            )

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if not args.multi_model_list or args.multi_model_list.strip() == "":
        raise ValueError("--multi_model_list must be provided for --multi")

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
        for model_idx, model_name in enumerate(modellist):
            fname = os.path.join(args.dataset_dir, f"{model_name}-v2.bp")
            dataset = AdiosDataset(
                fname,
                split_label,
                MPI.COMM_SELF,
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
            datasets.append(_NormalizedDataset(dataset))

            if rank == 0:
                print(
                    f"Mixed {split_label}: include {model_name} with {subset_len} samples"
                )

        if len(datasets) == 1:
            return datasets[0]

        mixed_dataset = ConcatDataset(datasets)
        return mixed_dataset

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


def _load_single_dataset_dataloaders(args, config, var_config):
    comm = MPI.COMM_WORLD
    if args.format == "adios":
        if AdiosDataset is None:
            raise ImportError(
                "AdiosDataset is unavailable; install adios2 to use --adios"
            )
        opt = {
            "preload": False,
            "shmem": args.shmem,
        }
        fname = os.path.join(args.dataset_dir, f"{args.modelname}.bp")
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        basedir = os.path.join(args.dataset_dir, f"{args.modelname}.pickle")
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = getattr(trainset, "pna_deg", None)
    else:
        raise NotImplementedError(f"No supported format: {args.format}")

    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)

        if "pna_deg" in locals() and pna_deg is not None:
            trainset.pna_deg = pna_deg
            valset.pna_deg = pna_deg
            testset.pna_deg = pna_deg

        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )
    return train_loader, val_loader, test_loader


class BranchWeightMLP(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int],
        hidden_dims: Tuple[int, ...],
        num_branches: int,
    ):
        super().__init__()
        layers = []
        if input_dim is None:
            if len(hidden_dims) > 0:
                layers.append(nn.LazyLinear(hidden_dims[0]))
                layers.append(nn.ReLU())
                in_dim = hidden_dims[0]
                for h in hidden_dims[1:]:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(nn.ReLU())
                    in_dim = h
                layers.append(nn.Linear(in_dim, num_branches))
            else:
                layers.append(nn.LazyLinear(num_branches))
        else:
            in_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, num_branches))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _reshape_composition(data) -> torch.Tensor:
    """Return composition as [num_graphs, comp_dim]."""
    comp = data.chemical_composition
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    if comp.dim() == 2:
        if comp.size(0) == data.num_graphs:
            return comp
        if comp.size(1) == data.num_graphs:
            return comp.t()
        if comp.size(1) == 1 and comp.size(0) % data.num_graphs == 0:
            return comp.view(data.num_graphs, -1)
    if comp.dim() == 3:
        if comp.size(0) == data.num_graphs:
            return comp.view(data.num_graphs, -1)
        if comp.size(1) == data.num_graphs:
            return comp.transpose(0, 1).contiguous().view(data.num_graphs, -1)
    raise ValueError(
        f"Unsupported chemical_composition shape {tuple(comp.shape)} for num_graphs={data.num_graphs}"
    )


def _infer_num_branches(config: dict, model) -> int:
    arch = config.get("NeuralNetwork", {}).get("Architecture", {})
    output_heads = arch.get("output_heads", {})
    graph_heads = output_heads.get("graph") if isinstance(output_heads, dict) else None
    if isinstance(graph_heads, list) and len(graph_heads) > 0:
        return len(graph_heads)

    model_num_branches = getattr(model, "num_branches", None)
    if isinstance(model_num_branches, int) and model_num_branches > 0:
        return model_num_branches

    return 1


def _resolve_selected_precision(
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


def _build_dataset_name(data, branch_id: int) -> torch.Tensor:
    if hasattr(data, "dataset_name"):
        return torch.full_like(data.dataset_name, branch_id)
    return torch.full(
        (data.num_graphs, 1),
        branch_id,
        dtype=torch.long,
        device=data.x.device,
    )


def _energy_from_pred(pred) -> torch.Tensor:
    if isinstance(pred, (list, tuple)):
        energy = pred[0]
    elif isinstance(pred, dict) and "graph" in pred:
        energy = pred["graph"][0]
    else:
        energy = pred
    return energy.squeeze(-1)


def _predict_branch_energy_forces(
    model, data, branch_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)

    pred = model(data)
    energy_pred = _energy_from_pred(pred)
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


def _weighted_average(
    energy_preds: torch.Tensor,
    forces_preds: torch.Tensor,
    weights: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # energy_preds: [num_branches, num_graphs]
    # forces_preds: [num_branches, num_nodes, 3]
    # weights: [num_graphs, num_branches]
    weighted_energy = torch.sum(weights * energy_preds.transpose(0, 1), dim=1)

    node_counts = torch.bincount(batch)
    weighted_forces = torch.zeros_like(forces_preds[0])
    for branch_idx in range(energy_preds.size(0)):
        node_weights = torch.repeat_interleave(weights[:, branch_idx], node_counts)
        weighted_forces = (
            weighted_forces + node_weights.unsqueeze(-1) * forces_preds[branch_idx]
        )

    return weighted_energy, weighted_forces


def train_epoch(
    model,
    mlp,
    loader,
    optimizer,
    loss_fn,
    num_branches,
    energy_weight,
    force_weight,
    precision,
):
    model.eval()
    mlp.train()
    device = get_device()
    total_loss = 0.0
    total_samples = 0
    timing = {
        "total": 0.0,
        "move_batch": 0.0,
        "mlp_forward": 0.0,
        "branch_inference": 0.0,
        "weighting_loss": 0.0,
        "optimizer_step": 0.0,
    }

    precision, param_dtype, _ = resolve_precision(precision)

    for data in iterate_tqdm(loader, 2, desc="MLP train", leave=False):
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        data = move_batch_to_device(data, param_dtype)
        timing["move_batch"] += time.perf_counter() - t0
        if not hasattr(data, "chemical_composition"):
            raise ValueError(
                "data.chemical_composition is required for branch weighting"
            )

        data.pos.requires_grad_(True)

        t0 = time.perf_counter()
        comp = _reshape_composition(data).to(device=device, dtype=param_dtype)

        logits = mlp(comp)
        weights = F.softmax(logits, dim=-1)
        timing["mlp_forward"] += time.perf_counter() - t0

        energy_preds = []
        forces_preds = []
        t0 = time.perf_counter()
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = _predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)
        timing["branch_inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = _weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        energy_true = data.energy.squeeze().to(dtype=weighted_energy.dtype)
        forces_true = data.forces.to(dtype=weighted_forces.dtype)

        loss_energy = loss_fn(weighted_energy, energy_true)
        loss_forces = loss_fn(weighted_forces, forces_true)
        loss = energy_weight * loss_energy + force_weight * loss_forces
        timing["weighting_loss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        timing["optimizer_step"] += time.perf_counter() - t0

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(len(loader), 1)
    return total_loss / max(total_samples, 1), timing


def validate_epoch(
    model,
    mlp,
    loader,
    loss_fn,
    num_branches,
    energy_weight,
    force_weight,
    precision,
):
    model.eval()
    mlp.eval()
    device = get_device()
    total_loss = 0.0
    total_samples = 0
    timing = {
        "total": 0.0,
        "move_batch": 0.0,
        "mlp_forward": 0.0,
        "branch_inference": 0.0,
        "weighting_loss": 0.0,
    }

    precision, param_dtype, _ = resolve_precision(precision)

    for data in iterate_tqdm(loader, 2, desc="MLP val", leave=False):
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        data = move_batch_to_device(data, param_dtype)
        timing["move_batch"] += time.perf_counter() - t0
        data.pos.requires_grad_(True)

        t0 = time.perf_counter()
        comp = _reshape_composition(data).to(device=device, dtype=param_dtype)

        logits = mlp(comp)
        weights = F.softmax(logits, dim=-1)
        timing["mlp_forward"] += time.perf_counter() - t0

        energy_preds = []
        forces_preds = []
        t0 = time.perf_counter()
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = _predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)
        timing["branch_inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = _weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        energy_true = data.energy.squeeze().to(dtype=weighted_energy.dtype)
        forces_true = data.forces.to(dtype=weighted_forces.dtype)

        loss_energy = loss_fn(weighted_energy, energy_true)
        loss_forces = loss_fn(weighted_forces, forces_true)
        loss = energy_weight * loss_energy + force_weight * loss_forces
        timing["weighting_loss"] += time.perf_counter() - t0

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(len(loader), 1)
    return total_loss / max(total_samples, 1), timing


def _write_timing_plot(timing_history, plot_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: could not import matplotlib for timing plot: {exc}")
        return

    if len(timing_history) == 0:
        return

    epochs = [item["epoch"] for item in timing_history]
    train_branch = [item["train_timing"]["branch_inference"] for item in timing_history]
    train_mlp = [
        item["train_timing"]["mlp_forward"] + item["train_timing"]["optimizer_step"]
        for item in timing_history
    ]
    train_other = [
        item["train_timing"]["total"]
        - item["train_timing"]["branch_inference"]
        - item["train_timing"]["mlp_forward"]
        - item["train_timing"]["optimizer_step"]
        for item in timing_history
    ]

    val_branch = [item["val_timing"]["branch_inference"] for item in timing_history]
    val_mlp = [item["val_timing"]["mlp_forward"] for item in timing_history]
    val_other = [
        item["val_timing"]["total"]
        - item["val_timing"]["branch_inference"]
        - item["val_timing"]["mlp_forward"]
        for item in timing_history
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(epochs, train_branch, marker="o", label="branch_inference")
    axes[0].plot(epochs, train_mlp, marker="o", label="mlp_forward+opt")
    axes[0].plot(epochs, train_other, marker="o", label="other")
    axes[0].set_title("Train timing by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_branch, marker="o", label="branch_inference")
    axes[1].plot(epochs, val_mlp, marker="o", label="mlp_forward")
    axes[1].plot(epochs, val_other, marker="o", label="other")
    axes[1].set_title("Val timing by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP weights for multi-branch predictions"
    )
    parser.add_argument("--inputfile", required=True, help="Path to JSON config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to pretrained model checkpoint (.pk)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing <dataset>-v2.bp files",
    )
    parser.add_argument(
        "--modelname",
        default="GFM",
        help="Base dataset name for single-dataset adios/pickle loading",
    )
    parser.add_argument(
        "--multi_model_list",
        help="Comma-separated dataset/model names (required for --multi)",
        default=None,
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--num_samples",
        type=int,
        help="set num samples per process for weak-scaling test",
        default=None,
    )
    parser.add_argument(
        "--task_parallel", action="store_true", help="enable task parallel"
    )
    parser.add_argument("--use_devicemesh", action="store_true", help="use device mesh")
    parser.add_argument("--oversampling", action="store_true", help="use oversampling")
    parser.add_argument(
        "--oversampling_num_samples",
        type=int,
        help="set num samples for oversampling",
        default=None,
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload ADIOS subset into memory (default: disabled)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override Training.batch_size from JSON",
    )
    parser.add_argument(
        "--energy_weight",
        type=float,
        default=None,
        help="Override energy loss weight",
    )
    parser.add_argument(
        "--force_weight",
        type=float,
        default=None,
        help="Override force loss weight",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help=(
            "Precision override for this run (examples: fp16, fp32, fp64, bf16). "
            "If omitted, uses NeuralNetwork.Training.precision from input config.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="mlp_weights",
        help="Subfolder to save MLP checkpoints",
    )
    parser.add_argument(
        "--output",
        default="branch_weight_mlp.pt",
        help="Output path for MLP weights",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a saved MLP checkpoint to resume from",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--multi",
        help="Multi dataset",
        action="store_const",
        dest="format",
        const="multi",
    )
    parser.set_defaults(format="multi")
    args = parser.parse_args()

    with open(args.inputfile, "r") as f:
        config = json.load(f)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    _configure_variable_names(config)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    hydragnn.utils.distributed.setup_ddp()

    if args.multi_model_list:
        train_loader, val_loader, _ = _load_multidataset_dataloaders(
            args, config, var_config
        )
    else:
        train_loader, val_loader, _ = _load_single_dataset_dataloaders(
            args, config, var_config
        )

    config = update_config(config, train_loader, val_loader, val_loader)

    precision, precision_source = _resolve_selected_precision(args.precision, config)
    precision, param_dtype, _ = resolve_precision(precision)
    torch.set_default_dtype(param_dtype)

    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    )
    device = get_device()
    model = model.to(device)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print(f"Using precision={precision} (source={precision_source})")
        if torch.cuda.is_available():
            local_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(local_idx)
            visible = torch.cuda.device_count()
            print(
                f"GPU mapping: device={device}, local_gpu_index={local_idx}, visible_gpus={visible}, device_name={device_name}"
            )
        else:
            print(f"GPU mapping: device={device}, CUDA unavailable")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_branches = _infer_num_branches(config, model)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    mlp = BranchWeightMLP(None, hidden_dims, num_branches).to(
        device=device, dtype=param_dtype
    )
    if args.resume is not None:
        ckpt_mlp = torch.load(args.resume, map_location=device)
        mlp.load_state_dict(ckpt_mlp["mlp_state_dict"], strict=True)
        print(f"Loaded MLP weights from {args.resume}")

    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = F.mse_loss

    energy_weight = (
        args.energy_weight
        if args.energy_weight is not None
        else config["NeuralNetwork"]["Architecture"].get("energy_weight", 1.0)
    )
    force_weight = (
        args.force_weight
        if args.force_weight is not None
        else config["NeuralNetwork"]["Architecture"].get("force_weight", 1.0)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    output_stem, _ = os.path.splitext(output_path)
    timing_json_path = f"{output_stem}.timing.json"
    timing_plot_path = f"{output_stem}.timing.png"
    timing_history = []

    for epoch in range(args.epochs):
        train_loss, train_timing = train_epoch(
            model,
            mlp,
            train_loader,
            optimizer,
            loss_fn,
            num_branches,
            energy_weight,
            force_weight,
            precision,
        )
        val_loss, val_timing = validate_epoch(
            model,
            mlp,
            val_loader,
            loss_fn,
            num_branches,
            energy_weight,
            force_weight,
            precision,
        )
        timing_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_timing": {k: float(v) for k, v in train_timing.items()},
                "val_timing": {k: float(v) for k, v in val_timing.items()},
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f} "
            f"| train(branch={train_timing['branch_inference']:.2f}s, mlp+opt={train_timing['mlp_forward'] + train_timing['optimizer_step']:.2f}s, total={train_timing['total']:.2f}s) "
            f"| val(branch={val_timing['branch_inference']:.2f}s, mlp={val_timing['mlp_forward']:.2f}s, total={val_timing['total']:.2f}s)"
        )

    torch.save({"mlp_state_dict": mlp.state_dict()}, output_path)
    print(f"Saved MLP weights to {output_path}")

    if rank == 0:
        with open(timing_json_path, "w") as f:
            json.dump(timing_history, f, indent=2)
        _write_timing_plot(timing_history, timing_plot_path)
        print(f"Saved timing history to {timing_json_path}")
        print(f"Saved timing plot to {timing_plot_path}")


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_distributed()
