"""Evaluate OPF models under every Laplacian-eigenvector sign convention.

For the default eight Laplacian modes this uniformly samples 64 distinct
patterns from the full set of ``2**8 == 256`` graphwise sign patterns.  The
same pattern is applied to every graph in one test-set pass; signs are constant
across the nodes of a graph and independent across eigenmodes.  Results are
written to TensorBoard and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# TensorBoard only needs its event-file writer here. This installed TensorBoard
# release selects its bundled stub when tensorboard.compat.notf is importable.
# Providing that marker prevents an optional CUDA TensorFlow installation from
# being loaded into the ROCm process and segfaulting before evaluation starts.
sys.modules.setdefault(
    "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset


DEFAULT_RUNS = {
    "LPE": "case4661_lpe",
    "LPE-signflip": "case4661_lpe_signflip",
    "ERLPE": "case4661_lpe_effective_resistance",
    "ERLPE-signflip": "case4661_lpe_signflip_effective_resistance",
}


def enumerate_sign_patterns(width: int) -> torch.Tensor:
    """Return canonical-first sign patterns with shape ``[2**width, width]``."""

    if width <= 0:
        raise ValueError(f"Laplacian width must be positive, got {width}.")
    # Product order starts at all +1; bit/mode zero changes fastest.
    patterns = []
    for pattern_index in range(1 << width):
        patterns.append(
            [
                -1.0 if (pattern_index >> mode_index) & 1 else 1.0
                for mode_index in range(width)
            ]
        )
    return torch.tensor(patterns, dtype=torch.float32)


def select_sign_patterns(
    width: int, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample distinct sign patterns using a reproducible seed."""

    patterns = enumerate_sign_patterns(width)
    total = patterns.size(0)
    if count <= 0 or count > total:
        raise ValueError(f"--num-patterns must be between 1 and {total}, got {count}.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:count]
    return patterns[indices], indices


def apply_graphwise_sign_pattern(data, pattern: torch.Tensor):
    """Apply one eigenmode sign pattern without mutating the stored dataset."""

    if "bus" not in data.node_types:
        raise ValueError("The batch has no 'bus' node type.")
    bus_store = data["bus"]
    eigenvectors = getattr(bus_store, "lap_eigvec", None)
    if eigenvectors is None:
        raise ValueError(
            "The batch is missing bus.lap_eigvec. Re-run OPF preprocessing "
            "with laplacian in positional_encodings.precompute."
        )
    if eigenvectors.dim() != 2:
        raise ValueError(
            f"Expected bus.lap_eigvec to be 2-D, got {tuple(eigenvectors.shape)}."
        )
    pattern = pattern.to(device=eigenvectors.device, dtype=eigenvectors.dtype)
    if pattern.dim() != 1 or pattern.numel() != eigenvectors.size(1):
        raise ValueError(
            f"Expected a sign pattern of width {eigenvectors.size(1)}, got "
            f"shape {tuple(pattern.shape)}."
        )
    # One pattern is used for every graph in this sweep pass.  Broadcasting
    # preserves a single sign per graph/eigenmode and never flips nodes alone.
    bus_store.lap_eigvec = eigenvectors * pattern.view(1, -1)
    return data


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device):
    """Load checkpoints saved with or without DDP/module prefixes."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    model.load_state_dict(state_dict, strict=True)


def _resolve_checkpoint(run_dir: Path, run_name: str) -> Path:
    preferred = run_dir / f"{run_name}.pk"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob(f"{run_name}_epoch_*.pk"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}.")
    return candidates[-1]


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing saved run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _laplacian_width(config: dict) -> int:
    architecture = config["NeuralNetwork"]["Architecture"]
    pe_config = architecture.get("positional_encodings", {})
    active = pe_config.get("use", [])
    if "laplacian" not in active:
        raise ValueError("Every sweep model must use the Laplacian encoding.")
    return int(pe_config.get("laplacian", {}).get("dim", 8))


def _make_test_loader(dataset_path: Path, config: dict, max_samples: int | None):
    dataset = HDF5Dataset(str(dataset_path), "testset")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    batch_size = int(config["NeuralNetwork"]["Training"].get("batch_size", 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _prepare_batch(data, edge_dim, node_target_type: str, pattern, device):
    # Match the training adapters while avoiding mutation of the HDF5 object.
    data = data.clone()
    from opf_solution_utils import ensure_node_y_loc, validate_edge_attr

    validate_edge_attr(data, edge_dim)
    if node_target_type not in data.node_types:
        raise ValueError(f"Batch is missing node target type '{node_target_type}'.")
    target_store = data[node_target_type]
    if getattr(target_store, "y", None) is None:
        raise ValueError(f"Batch is missing {node_target_type}.y targets.")
    data.y = target_store.y
    ensure_node_y_loc(data)
    if not hasattr(data, "batch"):
        data.batch = target_store.batch
    data = apply_graphwise_sign_pattern(data, pattern)
    return data.to(device)


@torch.inference_mode()
def evaluate_pattern(model, loader, config: dict, pattern, device) -> Dict[str, float]:
    architecture = config["NeuralNetwork"]["Architecture"]
    edge_dim = architecture.get("edge_dim")
    node_target_type = architecture.get("node_target_type", "bus")

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    element_count = 0
    per_dim_squared = None
    per_dim_absolute = None
    per_dim_count = 0

    model.eval()
    for data in loader:
        data = _prepare_batch(
            data,
            edge_dim=edge_dim,
            node_target_type=node_target_type,
            pattern=pattern,
            device=device,
        )
        prediction = model(data)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        target = data[node_target_type].y.to(
            device=prediction.device, dtype=prediction.dtype
        )
        target = target.reshape_as(prediction)
        difference = prediction - target

        squared_error_sum += difference.square().sum().item()
        absolute_error_sum += difference.abs().sum().item()
        element_count += difference.numel()
        dim_squared = difference.square().sum(dim=0).detach().cpu().double()
        dim_absolute = difference.abs().sum(dim=0).detach().cpu().double()
        if per_dim_squared is None:
            per_dim_squared = torch.zeros_like(dim_squared)
            per_dim_absolute = torch.zeros_like(dim_absolute)
        per_dim_squared += dim_squared
        per_dim_absolute += dim_absolute
        per_dim_count += difference.size(0)

    if element_count == 0:
        raise RuntimeError("The test loader produced no target elements.")

    mse = squared_error_sum / element_count
    metrics = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error_sum / element_count,
    }
    for dim_index in range(len(per_dim_squared)):
        metrics[f"mse_dim_{dim_index}"] = (
            per_dim_squared[dim_index].item() / per_dim_count
        )
        metrics[f"mae_dim_{dim_index}"] = (
            per_dim_absolute[dim_index].item() / per_dim_count
        )
    return metrics


def _create_model(config: dict, sample, device):
    architecture = config["NeuralNetwork"]["Architecture"]
    node_input_dims = architecture.get("node_input_dims")
    if node_input_dims is None:
        node_input_dims = {
            node_type: int(sample[node_type].x.size(-1))
            for node_type in sample.node_types
        }
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=0,
        use_gpu=device.type != "cpu",
        metadata=sample.metadata(),
        node_input_dims=node_input_dims,
    )
    return model.to(device)


def _parse_run_overrides(values: Sequence[str]) -> Dict[str, str]:
    runs = dict(DEFAULT_RUNS)
    for value in values:
        if "=" not in value:
            raise ValueError("--run entries must use LABEL=RUN_DIRECTORY_NAME.")
        label, run_name = value.split("=", 1)
        if not label or not run_name:
            raise ValueError("--run entries must use LABEL=RUN_DIRECTORY_NAME.")
        runs[label] = run_name
    return runs


def _summary(values: Iterable[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(correction=0).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "median": tensor.median().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all Laplacian eigenvector sign patterns."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset",
        type=Path,
        default=script_dir / "dataset" / "case4661_lpe_er.h5",
        help=(
            "Preprocessed HDF5 dataset containing bus Laplacian eigenvectors "
            "(default: dataset/case4661_lpe_er.h5)."
        ),
    )
    parser.add_argument("--logs-root", type=Path, default=script_dir / "logs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "logs" / "lpe_sign_sweep",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_NAME",
        help="Override or add a model run; repeat for multiple models.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=234,
        help=(
            "Evaluate only the first N test graphs (default: 234, approximately "
            "1/64 of the 15,000-sample test split; use 0 for the full split)."
        ),
    )
    parser.add_argument(
        "--num-patterns",
        type=int,
        default=64,
        help="Number of distinct random sign patterns (default: 64).",
    )
    parser.add_argument(
        "--pattern-seed",
        type=int,
        default=0,
        help="Random seed used to select sign patterns (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if args.max_samples == 0:
        args.max_samples = None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but torch.cuda is unavailable.")

    runs = _parse_run_overrides(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output_dir))
    csv_path = args.output_dir / "sign_sweep.csv"

    all_rows: List[dict] = []
    expected_width = None
    try:
        for label, run_name in runs.items():
            run_dir = args.logs_root / run_name
            config = _load_run_config(run_dir)
            width = _laplacian_width(config)
            if expected_width is None:
                expected_width = width
            elif width != expected_width:
                raise ValueError(
                    f"All models must use the same Laplacian width; {label} uses "
                    f"{width}, expected {expected_width}."
                )
            patterns, original_pattern_indices = select_sign_patterns(
                width, args.num_patterns, args.pattern_seed
            )

            loader = _make_test_loader(args.dataset, config, args.max_samples)
            sample = loader.dataset[0]
            model = _create_model(config, sample, device)
            checkpoint = _resolve_checkpoint(run_dir, run_name)
            load_checkpoint(model, checkpoint, device)

            print(
                f"Evaluating {label}: run={run_name}, checkpoint={checkpoint}, "
                f"patterns={len(patterns)}, device={device}"
            )
            model_rows = []
            for sweep_index, (original_pattern_index, pattern) in enumerate(
                zip(original_pattern_indices.tolist(), patterns)
            ):
                metrics = evaluate_pattern(model, loader, config, pattern, device)
                sign_string = "".join("-" if sign < 0 else "+" for sign in pattern)
                row = {
                    "model": label,
                    "run": run_name,
                    "pattern_index": original_pattern_index,
                    "sweep_index": sweep_index,
                    "signs": sign_string,
                    "flip_count": int((pattern < 0).sum().item()),
                    **metrics,
                }
                all_rows.append(row)
                model_rows.append(row)

                writer.add_scalar(f"{label}/mse", metrics["mse"], sweep_index)
                writer.add_scalar(f"{label}/rmse", metrics["rmse"], sweep_index)
                writer.add_scalar(f"{label}/mae", metrics["mae"], sweep_index)
                writer.add_scalar(
                    f"{label}/flip_count", row["flip_count"], sweep_index
                )
                for metric_name, metric_value in metrics.items():
                    if metric_name not in {"mse", "rmse", "mae"}:
                        writer.add_scalar(
                            f"{label}/{metric_name}", metric_value, sweep_index
                        )
                writer.add_text(
                    f"{label}/sign_pattern", sign_string, global_step=sweep_index
                )
                writer.flush()
                print(
                    f"[{label}] {sweep_index + 1:03d}/{len(patterns)} "
                    f"pattern={original_pattern_index:03d} "
                    f"signs={sign_string} mse={metrics['mse']:.8e} "
                    f"mae={metrics['mae']:.8e}"
                )

            canonical_mse = model_rows[0]["mse"]
            for row in model_rows:
                row["relative_mse_change"] = (
                    (row["mse"] - canonical_mse) / canonical_mse
                    if canonical_mse != 0.0
                    else float("nan")
                )
                writer.add_scalar(
                    f"{label}/relative_mse_change",
                    row["relative_mse_change"],
                    row["pattern_index"],
                )

            for metric_name in ("mse", "mae", "rmse"):
                statistics = _summary(row[metric_name] for row in model_rows)
                for statistic_name, value in statistics.items():
                    writer.add_scalar(
                        f"summary/{label}/{metric_name}_{statistic_name}", value, 0
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_rows)
        writer.add_text("run/csv", str(csv_path), 0)
        print(f"TensorBoard log: {args.output_dir}")
        print(f"CSV results: {csv_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
