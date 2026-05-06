"""Run inference for OPF heterogeneous node-level model and generate parity plots."""

import argparse
import json
import os
from mpi4py import MPI

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

import hydragnn
from hydragnn.postprocess.postprocess import output_denormalize
from hydragnn.train.train_validate_test import test
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model import load_existing_model

from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset

from opf_solution_utils import (
    HeteroFromHomogeneousDataset,
    NodeBatchAdapter,
    NodeTargetDatasetAdapter,
    compute_pna_deg_for_hetero_dataset,
    validate_voi_node_features,
    info,
    resolve_edge_feature_schema,
    resolve_node_target_type,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    AdiosDataset = None


def _plot_parity_per_dim(
    true_values,
    predicted_values,
    output_name,
    output_dim,
    out_dir,
    prefix="test",
):
    true_arr = true_values.detach().cpu().numpy()
    pred_arr = predicted_values.detach().cpu().numpy()
    if true_arr.ndim == 1:
        true_arr = true_arr.reshape(-1, 1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)

    if output_dim is None:
        output_dim = true_arr.shape[1]

    total = true_arr.shape[0]
    if total % output_dim != 0:
        output_dim = true_arr.shape[1]

    true_arr = true_arr.reshape(-1, output_dim)
    pred_arr = pred_arr.reshape(-1, output_dim)

    for dim in range(output_dim):
        t = true_arr[:, dim]
        p = pred_arr[:, dim]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(t, p, s=8, alpha=0.6)
        minv = float(np.min([t.min(), p.min()]))
        maxv = float(np.max([t.max(), p.max()]))
        ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{output_name} dim {dim}")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fname = f"{prefix}_parity_{output_name}_dim{dim}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close(fig)


def _compute_mae_per_quantity(true_values, predicted_values, output_name, output_dim):
    true_arr = true_values.detach().cpu().numpy()
    pred_arr = predicted_values.detach().cpu().numpy()
    if true_arr.ndim == 1:
        true_arr = true_arr.reshape(-1, 1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)

    if output_dim is None:
        output_dim = true_arr.shape[1]

    total = true_arr.shape[0]
    if total % output_dim != 0:
        output_dim = true_arr.shape[1]

    true_arr = true_arr.reshape(-1, output_dim)
    pred_arr = pred_arr.reshape(-1, output_dim)

    abs_err = np.abs(pred_arr - true_arr)
    mae_per_dim = abs_err.mean(axis=0)
    return {
        "quantity": output_name,
        "mae_overall": float(abs_err.mean()),
        "mae_per_dim": [float(v) for v in mae_per_dim],
    }


def _compute_diagnostics_per_quantity(
    true_values, predicted_values, output_name, output_dim
):
    true_arr = true_values.detach().cpu().numpy()
    pred_arr = predicted_values.detach().cpu().numpy()
    if true_arr.ndim == 1:
        true_arr = true_arr.reshape(-1, 1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)

    if output_dim is None:
        output_dim = true_arr.shape[1]

    total = true_arr.shape[0]
    if total % output_dim != 0:
        output_dim = true_arr.shape[1]

    true_arr = true_arr.reshape(-1, output_dim)
    pred_arr = pred_arr.reshape(-1, output_dim)

    residual = pred_arr - true_arr
    abs_err = np.abs(residual)

    bias_per_dim = residual.mean(axis=0)
    p50_per_dim = np.percentile(abs_err, 50, axis=0)
    p90_per_dim = np.percentile(abs_err, 90, axis=0)
    p99_per_dim = np.percentile(abs_err, 99, axis=0)

    high_true_bias_per_dim = []
    for dim in range(output_dim):
        threshold = float(np.percentile(true_arr[:, dim], 90))
        mask = true_arr[:, dim] >= threshold
        if np.any(mask):
            high_true_bias_per_dim.append(float(residual[mask, dim].mean()))
        else:
            high_true_bias_per_dim.append(0.0)

    return {
        "quantity": output_name,
        "bias_per_dim": [float(v) for v in bias_per_dim],
        "abs_error_p50_per_dim": [float(v) for v in p50_per_dim],
        "abs_error_p90_per_dim": [float(v) for v in p90_per_dim],
        "abs_error_p99_per_dim": [float(v) for v in p99_per_dim],
        "high_true_bias_per_dim": high_true_bias_per_dim,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", type=str, default="opf_solution_heterogeneous.json"
    )
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--modelname", type=str, default="OPF_Solution")
    parser.add_argument(
        "--node_target_type",
        type=str,
        default="bus",
        choices=["bus", "generator"],
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--config_from_log",
        action="store_true",
        help="Load config from logs/<modelname>/config.json when available",
    )
    parser.set_defaults(config_from_log=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    group.add_argument("--hdf5", action="store_const", dest="format", const="hdf5")
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size, rank = setup_ddp()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    log_config_path = os.path.join("./logs", args.modelname, "config.json")

    if args.config_from_log and os.path.isfile(log_config_path):
        if rank == 0:
            info(f"Loading config from {log_config_path}")
        with open(log_config_path, "r") as f:
            config = json.load(f)
    else:
        if not os.path.isfile(input_filename):
            raise FileNotFoundError(f"Missing config file: {input_filename}")
        with open(input_filename, "r") as f:
            config = json.load(f)

    arch_config = config.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    raw_edge_dim = arch_config.get("edge_dim")
    if isinstance(raw_edge_dim, dict):
        edge_dim = {str(k): int(v) for k, v in raw_edge_dim.items()}
        edge_feature_schema = None
    elif raw_edge_dim is not None:
        edge_dim = int(raw_edge_dim)
        names = arch_config.get("edge_feature_names")
        if names:
            edge_feature_schema = resolve_edge_feature_schema(names, edge_dim)
        else:
            edge_feature_schema = None
    else:
        raise RuntimeError("edge_dim must be specified in config.")
    arch_config["edge_dim"] = edge_dim

    if "node_target_type" in config.get("NeuralNetwork", {}).get("Architecture", {}):
        args.node_target_type = config["NeuralNetwork"]["Architecture"][
            "node_target_type"
        ]
    validate_voi_node_features(config, args.node_target_type)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        fname = os.path.join(dirpwd, "dataset", f"{args.modelname}.bp")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing ADIOS dataset: {fname}")
        train_base = AdiosDataset(fname, "trainset", comm, var_config=None)
        val_base = AdiosDataset(fname, "valset", comm, var_config=None)
        test_base = AdiosDataset(fname, "testset", comm, var_config=None)
        trainset = HeteroFromHomogeneousDataset(train_base, edge_dim=edge_dim)
        valset = HeteroFromHomogeneousDataset(val_base, edge_dim=edge_dim)
        testset = HeteroFromHomogeneousDataset(test_base, edge_dim=edge_dim)
    elif args.format == "hdf5":
        basedir = os.path.join(dirpwd, "dataset", f"{args.modelname}.h5")
        if not os.path.isdir(basedir):
            raise FileNotFoundError(f"Missing HDF5 dataset dir: {basedir}")
        trainset = HDF5Dataset(basedir, "trainset")
        valset = HDF5Dataset(basedir, "valset")
        testset = HDF5Dataset(basedir, "testset")
    else:
        basedir = os.path.join(dirpwd, "dataset", f"{args.modelname}.pickle")
        if not os.path.isdir(basedir):
            raise FileNotFoundError(f"Missing pickle dataset dir: {basedir}")
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=None
        )
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=None)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=None)

    resolved_node_target_type = resolve_node_target_type(
        trainset[0], args.node_target_type
    )
    if resolved_node_target_type != args.node_target_type:
        info(
            f"Resolved node_target_type '{args.node_target_type}' -> '{resolved_node_target_type}'"
        )
        args.node_target_type = resolved_node_target_type
    config.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})[
        "node_target_type"
    ] = args.node_target_type
    validate_voi_node_features(config, args.node_target_type)

    trainset = NodeTargetDatasetAdapter(
        trainset, args.node_target_type, edge_dim=edge_dim
    )
    valset = NodeTargetDatasetAdapter(valset, args.node_target_type, edge_dim=edge_dim)
    testset = NodeTargetDatasetAdapter(
        testset, args.node_target_type, edge_dim=edge_dim
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    train_loader = NodeBatchAdapter(
        train_loader, args.node_target_type, edge_dim=edge_dim
    )
    val_loader = NodeBatchAdapter(val_loader, args.node_target_type, edge_dim=edge_dim)
    test_loader = NodeBatchAdapter(
        test_loader, args.node_target_type, edge_dim=edge_dim
    )

    config = update_config(config, train_loader, val_loader, test_loader)
    arch_config = config.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    if arch_config.get("mpnn_type") == "HeteroPNA" and not arch_config.get("pna_deg"):
        info("Computing pna_deg for HeteroPNA from inference dataset")
        pna_deg = compute_pna_deg_for_hetero_dataset(trainset, verbosity=2)
        arch_config["pna_deg"] = pna_deg
        arch_config["max_neighbours"] = max(0, len(pna_deg) - 1)

    metadata = None
    try:
        metadata = trainset[0].metadata()
    except Exception as exc:
        if rank == 0:
            info(f"Unable to fetch hetero metadata: {exc}")
    node_input_dims = (
        config.get("NeuralNetwork", {}).get("Architecture", {}).get("node_input_dims")
    )
    if node_input_dims is None:
        raise RuntimeError(
            "Missing NeuralNetwork.Architecture.node_input_dims in config. "
            "Add node_input_dims to the config to initialize node embedders."
        )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
        metadata=metadata,
        node_input_dims=node_input_dims,
    )

    model = hydragnn.utils.distributed.distributed_model_wrapper(
        model, None, config["Verbosity"]["level"]
    )[0]

    load_existing_model(model, args.modelname)

    num_tasks = model.module.num_heads
    test_error, task_errors, true_values, predicted_values = test(
        test_loader,
        model,
        config["Verbosity"]["level"],
        num_tasks=num_tasks,
        precision=config["NeuralNetwork"]["Training"].get("precision", "fp32"),
    )

    if config["NeuralNetwork"]["Variables_of_interest"].get("denormalize_output"):
        true_values, predicted_values = output_denormalize(
            config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"],
            true_values,
            predicted_values,
        )

    if rank == 0:
        out_dir = os.path.join("./logs", args.modelname)
        os.makedirs(out_dir, exist_ok=True)
        var_config = config["NeuralNetwork"]["Variables_of_interest"]
        output_names = var_config.get("output_names", None)
        output_dims = var_config.get("output_dim", None)

        mae_metrics = []
        diagnostics_metrics = []
        for ihead in range(num_tasks):
            name = output_names[ihead] if output_names else f"head{ihead}"
            dim = output_dims[ihead] if output_dims else None
            mae_metrics.append(
                _compute_mae_per_quantity(
                    true_values[ihead],
                    predicted_values[ihead],
                    name,
                    dim,
                )
            )
            diagnostics_metrics.append(
                _compute_diagnostics_per_quantity(
                    true_values[ihead],
                    predicted_values[ihead],
                    name,
                    dim,
                )
            )

        metrics = {
            "test_error": float(test_error.detach().cpu().item()),
            "task_errors": task_errors.detach().cpu().tolist(),
            "mae": mae_metrics,
            "diagnostics": diagnostics_metrics,
        }
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Print key metrics to stdout so they are visible in batch job logs.
        print(f"Inference test_error: {metrics['test_error']}", flush=True)
        print(f"Inference task_errors: {metrics['task_errors']}", flush=True)

        for mae_entry in mae_metrics:
            info(
                "Inference MAE %s: overall=%g per_dim=%s"
                % (
                    mae_entry["quantity"],
                    mae_entry["mae_overall"],
                    mae_entry["mae_per_dim"],
                )
            )
            print(
                "Inference MAE %s: overall=%g per_dim=%s"
                % (
                    mae_entry["quantity"],
                    mae_entry["mae_overall"],
                    mae_entry["mae_per_dim"],
                ),
                flush=True,
            )

        for diag_entry in diagnostics_metrics:
            print(
                "Inference diagnostics %s: bias=%s p90_abs=%s p99_abs=%s high_true_bias=%s"
                % (
                    diag_entry["quantity"],
                    diag_entry["bias_per_dim"],
                    diag_entry["abs_error_p90_per_dim"],
                    diag_entry["abs_error_p99_per_dim"],
                    diag_entry["high_true_bias_per_dim"],
                ),
                flush=True,
            )

        for ihead in range(num_tasks):
            name = output_names[ihead] if output_names else f"head{ihead}"
            dim = output_dims[ihead] if output_dims else None
            _plot_parity_per_dim(
                true_values[ihead],
                predicted_values[ihead],
                name,
                dim,
                out_dir,
                prefix="test",
            )

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
