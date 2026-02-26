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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mpi4py import MPI

import hydragnn
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.models.create import create_model_config
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.train.train_validate_test import resolve_precision, move_batch_to_device

try:
    from .utils import (
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        predict_branch_energy_forces,
        weighted_average,
        extract_dataset_ids,
        teacher_from_dataset_id,
        cleanup_distributed,
    )
except ImportError:
    from utils import (
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        predict_branch_energy_forces,
        weighted_average,
        extract_dataset_ids,
        teacher_from_dataset_id,
        cleanup_distributed,
    )

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, adios2_open
except ImportError:
    AdiosDataset = None
    adios2_open = None

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset


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
                energy_pred, forces_pred = predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)
        timing["branch_inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        dataset_ids = extract_dataset_ids(data, num_branches)
        energy_true, forces_true = teacher_from_dataset_id(
            energy_preds, forces_preds, data.batch, dataset_ids
        )

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
                energy_pred, forces_pred = predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)
        timing["branch_inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        dataset_ids = extract_dataset_ids(data, num_branches)
        energy_true, forces_true = teacher_from_dataset_id(
            energy_preds, forces_preds, data.batch, dataset_ids
        )

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

    configure_variable_names(config)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    hydragnn.utils.distributed.setup_ddp()

    if args.multi_model_list:
        train_loader, val_loader, _ = load_multidataset_dataloaders(
            args, config, var_config
        )
    else:
        train_loader, val_loader, _ = _load_single_dataset_dataloaders(
            args, config, var_config
        )

    config = update_config(config, train_loader, val_loader, val_loader)

    precision, precision_source = resolve_selected_precision(args.precision, config)
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

    num_branches = infer_num_branches(config, model)

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


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
