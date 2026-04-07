#!/usr/bin/env python3
"""Cached stacking for branch weighting with optional top-k gating.

Workflow:
1) Build cache: run frozen HydraGNN branches once and save per-batch predictions.
2) Train combiner MLP from cache: no repeated HydraGNN inference in epochs.
"""

import argparse
import copy
import glob
import json
import os
import pickle
import random
import time
import gc
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import hydragnn
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import move_batch_to_device, resolve_precision
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.print.print_utils import iterate_tqdm

from branch_weighting_mlp import BranchWeightMLP
from branch_weighting_mlp import _reshape_composition

try:
    from .utils import (
        cleanup_distributed,
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        load_single_dataset_dataloaders,
        load_single_dataset_chunk,
        get_modellist_and_pna_deg,
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
        load_single_dataset_dataloaders,
        load_single_dataset_chunk,
        get_modellist_and_pna_deg,
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


def _append_records(path: str, records: list):
    """Append a list of records to a pickle file, creating it if needed."""
    with open(path, "ab") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def _iter_records(path: str):
    """Stream records from a pickle file one chunk at a time — no full-file RAM load."""
    with open(path, "rb") as f:
        while True:
            try:
                records = pickle.load(f)
                yield from records
            except EOFError:
                break


def _count_chunks(path: str) -> int:
    """Count how many chunks have been appended to a pickle file."""
    count = 0
    with open(path, "rb") as f:
        while True:
            try:
                pickle.load(f)
                count += 1
            except EOFError:
                break
    return count


def _collect_records(
    model,
    loader,
    num_branches: int,
    precision: str,
    cache_dtype: torch.dtype,
    param_dtype,
    device,
) -> list:
    """Run HydraGNN inference on all batches in loader and return list of records."""
    records = []
    model.eval()
    for data in iterate_tqdm(loader, 2, desc="Inference", leave=False):
        data = move_batch_to_device(data, param_dtype)
        data.pos.requires_grad_(True)

        comp = _reshape_composition(data).to(device=device, dtype=param_dtype)

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

        records.append({
            "comp": comp.detach().to(dtype=cache_dtype).cpu(),
            "dataset_ids": dataset_ids.detach().to(dtype=torch.long).cpu(),
            "energy_target": energy_target.detach().to(dtype=cache_dtype).cpu(),
            "forces_target": forces_target.detach().to(dtype=cache_dtype).cpu(),
            "batch": data.batch.detach().to(dtype=torch.long).cpu(),
            "energy_preds": energy_preds_stacked.detach().to(dtype=cache_dtype).cpu(),
            "forces_preds": forces_preds_stacked.detach().to(dtype=cache_dtype).cpu(),
            "num_graphs": int(data.num_graphs),
            "num_branches": int(num_branches),
        })

    return records


def _get_rank_cache_files(cache_root: str, split: str) -> List[str]:
    """Collect all batches.pkl files across all dataset subdirectories."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    all_files = []
    if not os.path.isdir(cache_root):
        return all_files
    for dataset_dir in sorted(os.listdir(cache_root)):
        rank_dir = os.path.join(cache_root, dataset_dir, split, f"rank{rank:05d}")
        f = os.path.join(rank_dir, "batches.pkl")
        if os.path.exists(f):
            all_files.append(f)
    return all_files


def _train_epoch_from_cache(
    mlp,
    cache_files: List[str],
    optimizer,
    loss_fn,
    energy_weight: float,
    force_weight: float,
    top_k: int,
    precision: str,
) -> Tuple[float, Dict[str, float]]:
    mlp.train()
    device = get_device()
    _, param_dtype, _ = resolve_precision(precision)

    total_loss = 0.0
    total_samples = 0
    timing = {"total": 0.0, "load": 0.0, "mlp": 0.0, "loss": 0.0, "opt": 0.0}
    total_batches = 0

    for cache_file in iterate_tqdm(cache_files, 2, desc="Cached train", leave=False):
        iter_t0 = time.perf_counter()

        for record in _iter_records(cache_file):
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

            total_loss += loss.item() * int(record["num_graphs"])
            total_samples += int(record["num_graphs"])
            total_batches += 1

        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(total_batches, 1)
    return total_loss / max(total_samples, 1), timing


@torch.no_grad()
def _validate_epoch_from_cache(
    mlp,
    cache_files: List[str],
    loss_fn,
    energy_weight: float,
    force_weight: float,
    top_k: int,
    precision: str,
) -> Tuple[float, Dict[str, float]]:
    mlp.eval()
    device = get_device()
    _, param_dtype, _ = resolve_precision(precision)

    total_loss = 0.0
    total_samples = 0
    timing = {"total": 0.0, "load": 0.0, "mlp": 0.0, "loss": 0.0}
    total_batches = 0

    for cache_file in iterate_tqdm(cache_files, 2, desc="Cached val", leave=False):
        iter_t0 = time.perf_counter()

        for record in _iter_records(cache_file):
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

            total_loss += loss.item() * int(record["num_graphs"])
            total_samples += int(record["num_graphs"])
            total_batches += 1

        timing["total"] += time.perf_counter() - iter_t0

    timing["num_batches"] = max(total_batches, 1)
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
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Max samples per dataset per chunk during cache building. "
            "If None, loads all num_samples at once. "
            "Use this to avoid OOM on large datasets."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--energy_weight", type=float, default=None)
    parser.add_argument("--force_weight", type=float, default=None)
    parser.add_argument("--output_dir", default="mlp_weights")
    parser.add_argument("--output", default="branch_weight_mlp_cached.pt")
    parser.add_argument(
        "--resume_mlp",
        default=None,
        help="Path to a saved MLP checkpoint to resume training from",
    )

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
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # Get modellist and pna_deg once from metadata only
    modellist, pna_deg = get_modellist_and_pna_deg(args, var_config)

    # update_config once using a tiny loader from the first dataset
    args_tiny = copy.copy(args)
    args_tiny.num_samples = 100
    args_tiny.preload = False
    first_train_loader, first_val_loader, _ = load_single_dataset_dataloaders(
        args_tiny, config, var_config,
        model_name=modellist[0],
        model_index=0,
        modellist=modellist,
        pna_deg=pna_deg,
    )
    config = update_config(config, first_train_loader, first_val_loader, first_val_loader)
    del first_train_loader, first_val_loader
    gc.collect()

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

    if rank == 0:
        print(f"Using precision={precision} (source={precision_source})")
        print(
            f"Cached stacking config: num_branches={num_branches}, top_k={top_k}, "
            f"cache_dtype={cache_dtype}, datasets={len(modellist)}"
        )

    # ------------------------------------------------------------------
    # Cache building — one dataset at a time, with optional chunking
    # to avoid OOM on large datasets
    # ------------------------------------------------------------------
    total_train_cached = 0
    total_val_cached = 0

    chunk_size = args.chunk_size or (args.num_samples or 10**9)

    for model_index, model_name in enumerate(modellist):
        if rank == 0:
            print(f"Caching dataset {model_index + 1}/{len(modellist)}: {model_name}")

        dataset_cache_dir = os.path.join(args.cache_dir, model_name)
        out_train = os.path.join(dataset_cache_dir, "train", f"rank{rank:05d}", "batches.pkl")
        out_val   = os.path.join(dataset_cache_dir, "val",   f"rank{rank:05d}", "batches.pkl")

        # Skip entirely if both splits already cached
        if (not args.rebuild_cache) and os.path.exists(out_train) and os.path.exists(out_val):
            n_train = _count_chunks(out_train)
            n_val   = _count_chunks(out_val)
            if rank == 0:
                print(f"  Already cached ({n_train} train chunks, {n_val} val chunks), skipping")
            total_train_cached += n_train
            total_val_cached   += n_val
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            continue

        # Remove partial files if rebuilding
        if args.rebuild_cache:
            for p in [out_train, out_val]:
                if os.path.exists(p):
                    os.remove(p)

        os.makedirs(os.path.dirname(out_train), exist_ok=True)
        os.makedirs(os.path.dirname(out_val),   exist_ok=True)

        num_samples = args.num_samples or 10**9
        train_batches_written = 0
        val_batches_written   = 0

        for chunk_start in range(0, num_samples, chunk_size):
            if rank == 0:
                print(f"  Chunk [{chunk_start} : {chunk_start + chunk_size}]")

            train_loader, val_loader, _ = load_single_dataset_chunk(
                args, config, var_config,
                model_name=model_name,
                model_index=model_index,
                modellist=modellist,
                pna_deg=pna_deg,
                chunk_start=chunk_start,
                chunk_size=chunk_size,
            )

            train_records = _collect_records(
                model, train_loader, num_branches, precision, cache_dtype, param_dtype, device
            )
            _append_records(out_train, train_records)
            train_batches_written += len(train_records)

            val_records = _collect_records(
                model, val_loader, num_branches, precision, cache_dtype, param_dtype, device
            )
            _append_records(out_val, val_records)
            val_batches_written += len(val_records)

            del train_loader, val_loader, train_records, val_records
            gc.collect()

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if rank == 0:
            print(f"  Saved {train_batches_written} train, {val_batches_written} val batches")

        total_train_cached += train_batches_written
        total_val_cached   += val_batches_written

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if rank == 0:
        print(
            f"Cache ready under {args.cache_dir}: "
            f"train_batches_per_rank={total_train_cached}, "
            f"val_batches_per_rank={total_val_cached}"
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if args.build_cache_only:
        return

    # ------------------------------------------------------------------
    # MLP training
    # ------------------------------------------------------------------
    train_cache_files = _get_rank_cache_files(args.cache_dir, split="train")
    val_cache_files   = _get_rank_cache_files(args.cache_dir, split="val")
    if len(train_cache_files) == 0 or len(val_cache_files) == 0:
        raise RuntimeError("Cache files are missing; run with --rebuild_cache")

    if rank == 0:
        print(f"MLP training on {len(train_cache_files)} train files, {len(val_cache_files)} val files per rank")

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    mlp = BranchWeightMLP(None, hidden_dims, num_branches).to(
        device=device, dtype=param_dtype
    )

    # Initialise LazyLinear before wrapping in DDP
    with torch.no_grad():
        first_record = next(_iter_records(train_cache_files[0]))
        sample_comp = first_record["comp"].to(device=device, dtype=param_dtype)
        _ = mlp(sample_comp)

    # Resume from checkpoint if provided
    start_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    output_stem, _ = os.path.splitext(output_path)
    timing_path = f"{output_stem}.timing.json"

    if args.resume_mlp is not None and os.path.exists(args.resume_mlp):
        ckpt_mlp = torch.load(args.resume_mlp, map_location=device)
        mlp.load_state_dict(ckpt_mlp["mlp_state_dict"])
        start_epoch = ckpt_mlp.get("epoch", 0)
        if rank == 0:
            print(f"Resumed MLP from {args.resume_mlp} at epoch {start_epoch}")

    # Wrap in DDP so all ranks train collectively
    if dist.is_available() and dist.is_initialized():
        mlp = DDP(mlp, device_ids=[torch.cuda.current_device()])

    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = F.mse_loss

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Restore optimizer and scheduler state if resuming
    if args.resume_mlp is not None and os.path.exists(args.resume_mlp):
        ckpt_mlp = torch.load(args.resume_mlp, map_location=device)
        if "optimizer_state_dict" in ckpt_mlp:
            optimizer.load_state_dict(ckpt_mlp["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt_mlp:
            scheduler.load_state_dict(ckpt_mlp["scheduler_state_dict"])

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

    # Load existing timing history if resuming
    timing_history = []
    if os.path.exists(timing_path):
        try:
            with open(timing_path, "r") as f:
                timing_history = json.load(f)
            if rank == 0:
                print(f"Loaded timing history ({len(timing_history)} epochs) from {timing_path}")
        except Exception:
            timing_history = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Shuffle dataset order each epoch so MLP sees datasets in different
        # order — prevents catastrophic forgetting of early datasets.
        # Seed with epoch so all ranks shuffle identically.
        random.seed(epoch)
        random.shuffle(train_cache_files)

        train_loss, train_timing = _train_epoch_from_cache(
            mlp, train_cache_files, optimizer, loss_fn,
            energy_weight, force_weight, top_k, precision,
        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        val_loss, val_timing = _validate_epoch_from_cache(
            mlp, val_cache_files, loss_fn,
            energy_weight, force_weight, top_k, precision,
        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        train_loss_global = _allreduce_mean(train_loss)
        val_loss_global   = _allreduce_mean(val_loss)

        scheduler.step(val_loss_global)

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  current_lr={current_lr:.2e}")

        timing_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_global),
                "val_loss": float(val_loss_global),
                "train_timing": {k: float(v) for k, v in train_timing.items()},
                "val_timing":   {k: float(v) for k, v in val_timing.items()},
            }
        )

        # Save after every epoch so progress is never lost
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            mlp_to_save = mlp.module if isinstance(mlp, DDP) else mlp
            tmp_output = output_path + ".tmp"
            torch.save(
                {
                    "mlp_state_dict": mlp_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                },
                tmp_output,
            )
            os.replace(tmp_output, output_path)

            tmp_timing = timing_path + ".tmp"
            with open(tmp_timing, "w") as f:
                json.dump(timing_history, f, indent=2)
            os.replace(tmp_timing, timing_path)
            print(
                f"Epoch {epoch + 1}/{start_epoch + args.epochs}: "
                f"train={train_loss_global:.6f} val={val_loss_global:.6f} "
                f"| train(total={train_timing['total']:.2f}s, "
                f"load={train_timing['load']:.2f}s, "
                f"mlp={train_timing['mlp']:.2f}s, "
                f"loss={train_timing['loss']:.2f}s, "
                f"opt={train_timing['opt']:.2f}s) "
                f"| val(total={val_timing['total']:.2f}s, "
                f"load={val_timing['load']:.2f}s, "
                f"mlp={val_timing['mlp']:.2f}s, "
                f"loss={val_timing['loss']:.2f}s)"
            )

    if rank == 0:
        print(f"Saved cached-MLP weights to {output_path}")
        print(f"Saved cached-MLP timing to {timing_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
