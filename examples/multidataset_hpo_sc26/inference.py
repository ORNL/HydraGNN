#!/usr/bin/env python3

import argparse
import glob
import json
import os

import torch
import torch.distributed as dist

import hydragnn
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import move_batch_to_device, resolve_precision
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.print.print_utils import iterate_tqdm

try:
    from .utils import (
        cleanup_distributed,
        configure_variable_names,
        resolve_selected_precision,
        infer_num_branches,
        load_multidataset_dataloaders,
        predict_branch_energy_forces,
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
        extract_dataset_ids,
        teacher_from_dataset_id,
    )


def _find_checkpoint(logdir: str, checkpoint: str = None) -> str:
    if checkpoint is not None:
        if os.path.isabs(checkpoint):
            ckpt_path = checkpoint
        else:
            ckpt_path = os.path.join(logdir, checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(glob.glob(os.path.join(logdir, "*.pk")))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .pk checkpoint found in {logdir}")
    return candidates[-1]


def _allreduce_pair(sum_value: float, count_value: int):
    if not (dist.is_available() and dist.is_initialized()):
        return sum_value, count_value

    device = get_device()
    pair = torch.tensor([sum_value, float(count_value)], dtype=torch.float64, device=device)
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0].item()), int(pair[1].item())


def _graph_natoms(data, num_graphs: int, device: torch.device) -> torch.Tensor:
    if hasattr(data, "natoms") and torch.is_tensor(data.natoms):
        natoms = data.natoms.to(device=device, dtype=torch.float64).view(-1)
        if natoms.numel() == 1 and num_graphs > 1:
            natoms = natoms.repeat(num_graphs)
        if natoms.numel() != num_graphs:
            raise ValueError(
                f"Invalid natoms shape {tuple(data.natoms.shape)} for num_graphs={num_graphs}"
            )
        return natoms.clamp_min(1.0)

    if hasattr(data, "batch") and torch.is_tensor(data.batch):
        natoms = torch.bincount(data.batch.to(device=device, dtype=torch.long), minlength=num_graphs)
        return natoms.to(dtype=torch.float64).clamp_min(1.0)

    raise ValueError("Cannot infer per-graph atom counts; expected `natoms` or `batch` in data")


def _evaluate_split(model, loader, num_branches, precision, split_name):
    _, param_dtype, _ = resolve_precision(precision)

    model.eval()
    energy_abs_sum = 0.0
    energy_count = 0
    energy_per_atom_abs_sum = 0.0
    energy_per_atom_count = 0
    force_abs_sum = 0.0
    force_count = 0

    for data in iterate_tqdm(loader, 2, desc=f"Eval {split_name}", leave=False):
        data = move_batch_to_device(data, param_dtype)
        data.pos.requires_grad_(True)

        energy_preds = []
        forces_preds = []
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)

        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        dataset_ids = extract_dataset_ids(data, num_branches)
        energy_pred, forces_pred = teacher_from_dataset_id(
            energy_preds, forces_preds, data.batch, dataset_ids
        )

        energy_true = data.energy.squeeze().to(dtype=energy_pred.dtype)
        forces_true = data.forces.to(dtype=forces_pred.dtype)
        natoms = _graph_natoms(data, data.num_graphs, device=energy_pred.device).to(
            dtype=energy_pred.dtype
        )

        energy_pred_per_atom = energy_pred.view(-1) / natoms
        energy_true_per_atom = energy_true.view(-1) / natoms

        energy_abs_sum += torch.abs(energy_pred - energy_true).sum().item()
        energy_count += energy_true.numel()
        energy_per_atom_abs_sum += torch.abs(
            energy_pred_per_atom - energy_true_per_atom
        ).sum().item()
        energy_per_atom_count += energy_true_per_atom.numel()
        force_abs_sum += torch.abs(forces_pred - forces_true).sum().item()
        force_count += forces_true.numel()

    energy_abs_sum, energy_count = _allreduce_pair(energy_abs_sum, energy_count)
    energy_per_atom_abs_sum, energy_per_atom_count = _allreduce_pair(
        energy_per_atom_abs_sum, energy_per_atom_count
    )
    force_abs_sum, force_count = _allreduce_pair(force_abs_sum, force_count)

    energy_mae = energy_abs_sum / max(energy_count, 1)
    energy_per_atom_mae = energy_per_atom_abs_sum / max(energy_per_atom_count, 1)
    force_mae = force_abs_sum / max(force_count, 1)
    return energy_mae, energy_per_atom_mae, force_mae


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model MAE (energy/forces) on val/test ADIOS splits"
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Training log directory containing config.json and checkpoint .pk",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path or filename inside --logdir (defaults to latest .pk)",
    )
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing <dataset>-v2.bp files",
    )
    parser.add_argument(
        "--multi_model_list",
        required=True,
        help="Comma-separated ADIOS dataset list (e.g. Alexandria,ANI1x,...)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--precision", type=str, default=None)

    parser.add_argument("--modelname", default="GFM")
    parser.add_argument("--ddstore", action="store_true")
    parser.add_argument("--ddstore_width", type=int, default=None)
    parser.add_argument("--shmem", action="store_true")
    parser.add_argument("--task_parallel", action="store_true")
    parser.add_argument("--use_devicemesh", action="store_true")
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument("--oversampling_num_samples", type=int, default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    group.add_argument("--multi", action="store_const", dest="format", const="multi")
    parser.set_defaults(format="multi")

    args = parser.parse_args()

    input_config = os.path.join(args.logdir, "config.json")
    if not os.path.isfile(input_config):
        raise FileNotFoundError(f"config.json not found in {args.logdir}")

    with open(input_config, "r") as f:
        config = json.load(f)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    configure_variable_names(config)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    hydragnn.utils.distributed.setup_ddp()

    train_loader, val_loader, test_loader = load_multidataset_dataloaders(
        args, config, var_config
    )
    config = update_config(config, train_loader, val_loader, test_loader)

    precision, precision_source = resolve_selected_precision(args.precision, config)
    precision, param_dtype, _ = resolve_precision(precision)
    torch.set_default_dtype(param_dtype)

    device = get_device()
    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    ).to(device)

    checkpoint_path = _find_checkpoint(args.logdir, args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_branches = infer_num_branches(config, model)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print(f"Using checkpoint: {checkpoint_path}")
        print(f"Using precision={precision} (source={precision_source})")
        print(f"Evaluating datasets: {args.multi_model_list}")

    val_energy_mae, val_energy_per_atom_mae, val_force_mae = _evaluate_split(
        model, val_loader, num_branches, precision, "val"
    )
    test_energy_mae, test_energy_per_atom_mae, test_force_mae = _evaluate_split(
        model, test_loader, num_branches, precision, "test"
    )

    if rank == 0:
        print("Validation MAE:")
        print(f"  Energy: {val_energy_mae:.8f}")
        print(f"  Energy per atom: {val_energy_per_atom_mae:.8f}")
        print(f"  Forces: {val_force_mae:.8f}")
        print("Test MAE:")
        print(f"  Energy: {test_energy_mae:.8f}")
        print(f"  Energy per atom: {test_energy_per_atom_mae:.8f}")
        print(f"  Forces: {test_force_mae:.8f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
