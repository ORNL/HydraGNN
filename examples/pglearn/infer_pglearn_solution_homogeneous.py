import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

import hydragnn
from hydragnn.postprocess.postprocess import output_denormalize
from hydragnn.train.train_validate_test import test
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model import load_existing_model

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    AdiosDataset = None


def _plot_parity(true_values, pred_values, out_dir, name="bus_solution"):
    true_arr = true_values.detach().cpu().numpy().reshape(-1, true_values.shape[-1])
    pred_arr = pred_values.detach().cpu().numpy().reshape(-1, pred_values.shape[-1])

    for dim in range(true_arr.shape[1]):
        t = true_arr[:, dim]
        p = pred_arr[:, dim]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(t, p, s=8, alpha=0.6)
        minv = float(np.min([t.min(), p.min()]))
        maxv = float(np.max([t.max(), p.max()]))
        ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} dim {dim}")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"parity_{name}_dim{dim}.png"), dpi=300)
        plt.close(fig)


def _load_splits(args, datadir, comm):
    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        base = os.path.join(datadir, f"{args.modelname}.bp")
        return (
            AdiosDataset(base, "trainset", comm, var_config=None),
            AdiosDataset(base, "valset", comm, var_config=None),
            AdiosDataset(base, "testset", comm, var_config=None),
        )

    if args.format == "hdf5":
        base = os.path.join(datadir, f"{args.modelname}.h5")
        return HDF5Dataset(base, "trainset"), HDF5Dataset(base, "valset"), HDF5Dataset(base, "testset")

    base = os.path.join(datadir, f"{args.modelname}.pickle")
    return (
        SimplePickleDataset(base, "trainset", var_config=None),
        SimplePickleDataset(base, "valset", var_config=None),
        SimplePickleDataset(base, "testset", var_config=None),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfile", type=str, default="pglearn_solution_homogeneous.json")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--modelname", type=str, default="PGLearn_Solution")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--config_from_log", action="store_true")
    parser.set_defaults(config_from_log=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--hdf5", action="store_const", dest="format", const="hdf5")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    comm_size, rank = setup_ddp()
    comm = MPI.COMM_WORLD

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, args.data_root)
    input_path = os.path.join(dirpwd, args.inputfile)
    log_cfg = os.path.join("./logs", args.modelname, "config.json")

    if args.config_from_log and os.path.isfile(log_cfg):
        with open(log_cfg, "r") as fh:
            config = json.load(fh)
    else:
        with open(input_path, "r") as fh:
            config = json.load(fh)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    trainset, valset, testset = _load_splits(args, datadir, comm)

    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )

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
            "mae": [],
        }

        for ihead in range(num_tasks):
            t = true_values[ihead].detach().cpu().numpy()
            p = predicted_values[ihead].detach().cpu().numpy()
            mae = float(np.mean(np.abs(p - t)))
            metrics["mae"].append(mae)
            _plot_parity(true_values[ihead], predicted_values[ihead], out_dir)

        with open(os.path.join(out_dir, "test_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"Inference test_error: {metrics['test_error']}", flush=True)
        print(f"Inference task_errors: {metrics['task_errors']}", flush=True)
        print(f"Inference MAE per head: {metrics['mae']}", flush=True)

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
