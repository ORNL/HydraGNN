#!/usr/bin/env python3
"""Cached stacking for branch weighting with optional top-k gating.

Workflow:
1) Build cache: run frozen HydraGNN branches once and save per-batch predictions.
2) Train combiner MLP from cache: no repeated HydraGNN inference in epochs.
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

import hydragnn
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import move_batch_to_device, resolve_precision
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.print.print_utils import iterate_tqdm

from branch_weighting_mlp import BranchWeightMLP

try:
    from .utils import (
        cleanup_distributed,
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        predict_branch_energy_forces,
        weighted_average,
        extract_dataset_ids,
        teacher_from_dataset_id,
    )
except ImportError:
    from utils import (
        cleanup_distributed,
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        predict_branch_energy_forces,
        weighted_average,
        extract_dataset_ids,
        teacher_from_dataset_id,
    )


def _allreduce_mean(value: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    device = get_device()
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _resolve_cache_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in ["fp16", "float16", "half"]:
        return torch.float16
    if value in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if value in ["fp64", "float64", "double"]:
        return torch.float64
    return torch.float32


def _resolve_weights_from_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return F.softmax(logits, dim=-1)

    top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
    sparse_logits = torch.full_like(logits, float("-inf"))
    sparse_logits.scatter_(dim=-1, index=top_idx, src=top_vals)
    return F.softmax(sparse_logits, dim=-1)


def _prepare_rank_dir(path: str, clear_existing: bool):
    os.makedirs(path, exist_ok=True)
    if not clear_existing:
        return
    for file_path in glob.glob(os.path.join(path, "*.pt")):
        os.remove(file_path)


def _cache_split(
    model,
    loader,
    split: str,
    cache_root: str,
    num_branches: int,
    precision: str,
    cache_dtype: torch.dtype,
    rebuild_cache: bool,
) -> int:
    device = get_device()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    _, param_dtype, _ = resolve_precision(precision)

    rank_dir = os.path.join(cache_root, split, f"rank{rank:05d}")
    _prepare_rank_dir(rank_dir, clear_existing=rebuild_cache)

    if (not rebuild_cache) and len(glob.glob(os.path.join(rank_dir, "*.pt"))) > 0:
        return len(glob.glob(os.path.join(rank_dir, "*.pt")))

    num_files = 0
    model.eval()
    for batch_idx, data in enumerate(
        iterate_tqdm(loader, 2, desc=f"Cache {split}", leave=False)
    ):
        data = move_batch_to_device(data, param_dtype)
        data.pos.requires_grad_(True)

        comp = data.chemical_composition
        if comp.dim() == 1:
            comp = comp.unsqueeze(0)
        elif (
            comp.dim() == 2
            and comp.size(0) != data.num_graphs
            and comp.size(1) == data.num_graphs
        ):
            comp = comp.t()
        comp = comp.to(device=device, dtype=param_dtype)
        energy_preds = []
        forces_preds = []
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)

        energy_preds_stacked = torch.stack(energy_preds, dim=0)
        forces_preds_stacked = torch.stack(forces_preds, dim=0)
        dataset_ids = extract_dataset_ids(data, num_branches)
        energy_target, forces_target = teacher_from_dataset_id(
            energy_preds_stacked, forces_preds_stacked, data.batch, dataset_ids
        )

        record = {
            "comp": comp.detach().to(dtype=cache_dtype).cpu(),
            "dataset_ids": dataset_ids.detach().to(dtype=torch.long).cpu(),
            "energy_target": energy_target.detach().to(dtype=cache_dtype).cpu(),
            "forces_target": forces_target.detach().to(dtype=cache_dtype).cpu(),
            "batch": data.batch.detach().to(dtype=torch.long).cpu(),
            "energy_preds": energy_preds_stacked.detach().to(dtype=cache_dtype).cpu(),
            "forces_preds": forces_preds_stacked.detach().to(dtype=cache_dtype).cpu(),
            "num_graphs": int(data.num_graphs),
            "num_branches": int(num_branches),
        }
        out_path = os.path.join(rank_dir, f"batch_{batch_idx:07d}.pt")
        torch.save(record, out_path)
        num_files += 1

    return num_files


def _get_rank_cache_files(cache_root: str, split: str):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rank_dir = os.path.join(cache_root, split, f"rank{rank:05d}")
    files = sorted(glob.glob(os.path.join(rank_dir, "batch_*.pt")))
    return files


def _train_epoch_from_cache(
    mlp,
    cache_files,
    optimizer,
    loss_fn,
    energy_weight,
    force_weight,
    top_k,
    precision,
) -> Tuple[float, Dict[str, float]]:
    mlp.train()
    device = get_device()
    _, param_dtype, _ = resolve_precision(precision)

    total_loss = 0.0
    total_samples = 0
    timing = {"total": 0.0, "load": 0.0, "mlp": 0.0, "loss": 0.0, "opt": 0.0}

    for cache_file in iterate_tqdm(cache_files, 2, desc="Cached train", leave=False):
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        record = torch.load(cache_file, map_location="cpu")
        timing["load"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        comp = record["comp"].to(device=device, dtype=param_dtype)
        logits = mlp(comp)
        weights = _resolve_weights_from_logits(logits, top_k=top_k)
        timing["mlp"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = record["energy_preds"].to(device=device, dtype=param_dtype)
        forces_preds = record["forces_preds"].to(device=device, dtype=param_dtype)
        batch = record["batch"].to(device=device, dtype=torch.long)
        weighted_energy, weighted_forces = weighted_average(
            energy_preds, forces_preds, weights, batch
        )
        energy_target = record["energy_target"].to(device=device, dtype=param_dtype)
        forces_target = record["forces_target"].to(device=device, dtype=param_dtype)
        loss_energy = loss_fn(weighted_energy, energy_target)
        loss_forces = loss_fn(weighted_forces, forces_target)
        loss = energy_weight * loss_energy + force_weight * loss_forces
        timing["loss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        timing["opt"] += time.perf_counter() - t0

        num_graphs = int(record["num_graphs"])
        total_loss += loss.item() * num_graphs
        total_samples += num_graphs
        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(len(cache_files), 1)
    return total_loss / max(total_samples, 1), timing


@torch.no_grad()
def _validate_epoch_from_cache(
    mlp,
    cache_files,
    loss_fn,
    energy_weight,
    force_weight,
    top_k,
    precision,
) -> Tuple[float, Dict[str, float]]:
    mlp.eval()
    device = get_device()
    _, param_dtype, _ = resolve_precision(precision)

    total_loss = 0.0
    total_samples = 0
    timing = {"total": 0.0, "load": 0.0, "mlp": 0.0, "loss": 0.0}

    for cache_file in iterate_tqdm(cache_files, 2, desc="Cached val", leave=False):
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        record = torch.load(cache_file, map_location="cpu")
        timing["load"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        comp = record["comp"].to(device=device, dtype=param_dtype)
        logits = mlp(comp)
        weights = _resolve_weights_from_logits(logits, top_k=top_k)
        timing["mlp"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        energy_preds = record["energy_preds"].to(device=device, dtype=param_dtype)
        forces_preds = record["forces_preds"].to(device=device, dtype=param_dtype)
        batch = record["batch"].to(device=device, dtype=torch.long)
        weighted_energy, weighted_forces = weighted_average(
            energy_preds, forces_preds, weights, batch
        )
        energy_target = record["energy_target"].to(device=device, dtype=param_dtype)
        forces_target = record["forces_target"].to(device=device, dtype=param_dtype)
        loss_energy = loss_fn(weighted_energy, energy_target)
        loss_forces = loss_fn(weighted_forces, forces_target)
        loss = energy_weight * loss_energy + force_weight * loss_forces
        timing["loss"] += time.perf_counter() - t0

        num_graphs = int(record["num_graphs"])
        total_loss += loss.item() * num_graphs
        total_samples += num_graphs
        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(len(cache_files), 1)
    return total_loss / max(total_samples, 1), timing


def main():
    parser = argparse.ArgumentParser(
        description="Cached branch-stacking with optional top-k gating"
    )
    parser.add_argument("--inputfile", required=True, help="Path to JSON config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Optional precision override; defaults to config.json precision",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=(
            "Optional top-k gating. If omitted, defaults to total number of branches. "
            "Use 0 to include all branches."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="branch_cache",
        help="Directory for cached branch predictions",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="fp32",
        help="Storage precision for cache tensors: fp16|bf16|fp32|fp64",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild cache even if per-rank cache files exist",
    )
    parser.add_argument(
        "--build_cache_only",
        action="store_true",
        help="Build cache and exit without MLP training",
    )
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing dataset files",
    )
    parser.add_argument("--modelname", default="GFM")
    parser.add_argument("--multi_model_list", default=None)
    parser.add_argument("--ddstore", action="store_true")
    parser.add_argument("--ddstore_width", type=int, default=None)
    parser.add_argument("--shmem", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--task_parallel", action="store_true")
    parser.add_argument("--use_devicemesh", action="store_true")
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument("--oversampling_num_samples", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--energy_weight", type=float, default=None)
    parser.add_argument("--force_weight", type=float, default=None)
    parser.add_argument("--output_dir", default="mlp_weights")
    parser.add_argument("--output", default="branch_weight_mlp_cached.pt")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    group.add_argument("--multi", action="store_const", dest="format", const="multi")
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
        raise NotImplementedError(
            "Cached script currently supports multi-dataset mode only"
        )

    config = update_config(config, train_loader, val_loader, val_loader)

    precision, precision_source = resolve_selected_precision(args.precision, config)
    precision, param_dtype, _ = resolve_precision(precision)
    torch.set_default_dtype(param_dtype)

    device = get_device()
    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_branches = infer_num_branches(config, model)
    if args.top_k is None:
        top_k = num_branches
    else:
        top_k = args.top_k

    cache_dtype = _resolve_cache_dtype(args.cache_dtype)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print(f"Using precision={precision} (source={precision_source})")
        print(
            f"Cached stacking config: num_branches={num_branches}, top_k={top_k}, cache_dtype={cache_dtype}"
        )

    train_cached = _cache_split(
        model,
        train_loader,
        split="train",
        cache_root=args.cache_dir,
        num_branches=num_branches,
        precision=precision,
        cache_dtype=cache_dtype,
        rebuild_cache=args.rebuild_cache,
    )
    val_cached = _cache_split(
        model,
        val_loader,
        split="val",
        cache_root=args.cache_dir,
        num_branches=num_branches,
        precision=precision,
        cache_dtype=cache_dtype,
        rebuild_cache=args.rebuild_cache,
    )

    if rank == 0:
        print(
            f"Cache ready under {args.cache_dir}: train_files_per_rank={train_cached}, val_files_per_rank={val_cached}"
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if args.build_cache_only:
        return

    train_cache_files = _get_rank_cache_files(args.cache_dir, split="train")
    val_cache_files = _get_rank_cache_files(args.cache_dir, split="val")
    if len(train_cache_files) == 0 or len(val_cache_files) == 0:
        raise RuntimeError("Cache files are missing; run with --rebuild_cache")

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    mlp = BranchWeightMLP(None, hidden_dims, num_branches).to(
        device=device, dtype=param_dtype
    )
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
    timing_path = f"{output_stem}.timing.json"
    timing_history = []

    for epoch in range(args.epochs):
        train_loss, train_timing = _train_epoch_from_cache(
            mlp,
            train_cache_files,
            optimizer,
            loss_fn,
            energy_weight,
            force_weight,
            top_k,
            precision,
        )
        val_loss, val_timing = _validate_epoch_from_cache(
            mlp,
            val_cache_files,
            loss_fn,
            energy_weight,
            force_weight,
            top_k,
            precision,
        )

        train_loss_global = _allreduce_mean(train_loss)
        val_loss_global = _allreduce_mean(val_loss)

        timing_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_global),
                "val_loss": float(val_loss_global),
                "train_timing": {k: float(v) for k, v in train_timing.items()},
                "val_timing": {k: float(v) for k, v in val_timing.items()},
            }
        )

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}: train={train_loss_global:.6f} val={val_loss_global:.6f} "
                f"| train(total={train_timing['total']:.2f}s, load={train_timing['load']:.2f}s, mlp={train_timing['mlp']:.2f}s, loss={train_timing['loss']:.2f}s, opt={train_timing['opt']:.2f}s) "
                f"| val(total={val_timing['total']:.2f}s, load={val_timing['load']:.2f}s, mlp={val_timing['mlp']:.2f}s, loss={val_timing['loss']:.2f}s)"
            )

    torch.save({"mlp_state_dict": mlp.state_dict()}, output_path)
    if rank == 0:
        with open(timing_path, "w") as f:
            json.dump(timing_history, f, indent=2)
        print(f"Saved cached-MLP weights to {output_path}")
        print(f"Saved cached-MLP timing to {timing_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
