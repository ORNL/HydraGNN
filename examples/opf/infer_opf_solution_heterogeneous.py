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

from opf_solution_utils import (
    HeteroFromHomogeneousDataset,
    NodeBatchAdapter,
    NodeTargetDatasetAdapter,
    info,
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size, rank = setup_ddp()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)

    with open(input_filename, "r") as f:
        config = json.load(f)

    if "node_target_type" in config.get("NeuralNetwork", {}).get("Architecture", {}):
        args.node_target_type = config["NeuralNetwork"]["Architecture"][
            "node_target_type"
        ]

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
        trainset = HeteroFromHomogeneousDataset(train_base)
        valset = HeteroFromHomogeneousDataset(val_base)
        testset = HeteroFromHomogeneousDataset(test_base)
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

    trainset = NodeTargetDatasetAdapter(trainset, args.node_target_type)
    valset = NodeTargetDatasetAdapter(valset, args.node_target_type)
    testset = NodeTargetDatasetAdapter(testset, args.node_target_type)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    train_loader = NodeBatchAdapter(train_loader, args.node_target_type)
    val_loader = NodeBatchAdapter(val_loader, args.node_target_type)
    test_loader = NodeBatchAdapter(test_loader, args.node_target_type)

    config = update_config(config, train_loader, val_loader, test_loader)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
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
        metrics = {
            "test_error": float(test_error.detach().cpu().item()),
            "task_errors": task_errors.detach().cpu().tolist(),
        }
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        var_config = config["NeuralNetwork"]["Variables_of_interest"]
        output_names = var_config.get("output_names", None)
        output_dims = var_config.get("output_dim", None)
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
