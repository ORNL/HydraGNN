"""Train OPFLearn solution prediction (homogeneous, single-node tabular graphs).

OPFLearn samples map load setpoints to an AC-OPF solution. Each sample is a
single-node graph with no branch topology, so only a homogeneous model is
meaningful (there is nothing to build a heterogeneous bus/branch graph from).

Typical usage:
    # Preprocess + serialize only (no training):
    python train_opflearn_solution_homogeneous.py --case_name case14_ieee --preonly

    # Train:
    python train_opflearn_solution_homogeneous.py --case_name case14_ieee --num_epoch 20
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from mpi4py import MPI

import hydragnn
from __init__ import data_ops
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model import print_model
from hydragnn.utils.profiling_and_tracing import print_timers


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfile", type=str, default="opflearn_solution_homogeneous.json")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--modelname", type=str, default="OPFLearn_Solution")
    parser.add_argument("--case_name", type=str, default="case14_ieee")
    parser.add_argument("--formulation", type=str, default="ACOPF")
    parser.add_argument("--preonly", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-download/re-serialize even if data exists.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, args.data_root)
    cfg_path = os.path.join(dirpwd, args.inputfile)

    with open(cfg_path, "r") as fh:
        config = json.load(fh)

    config.setdefault("Task", {})
    config["Task"]["formulation"] = args.formulation
    config["Task"]["case_name"] = args.case_name

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

    # Download + preprocess (rank 0), producing a processed Parquet file.
    parquet_path = data_ops.ensure_opflearn_prepared(
        root=datadir,
        case_name=args.case_name,
        rank=rank,
        comm=comm,
        overwrite=args.overwrite,
    )

    serialized_dir = os.path.join(datadir, f"{args.modelname}.h5")
    serialized_exists = os.path.isdir(serialized_dir)

    if args.preonly or args.overwrite or not serialized_exists:
        logger = logging.getLogger("serialize_opflearn")
        data_ops.serialize_parquet_to_hdf5(
            input_parquet=Path(parquet_path),
            output_hdf5_dir=Path(serialized_dir),
            max_samples=args.max_samples,
            include_duals=False,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            overwrite=True,
            logger=logger,
        )
        comm.Barrier()

        if args.preonly:
            if dist.is_initialized():
                dist.destroy_process_group()
            return

    trainset = HDF5Dataset(serialized_dir, "trainset")
    valset = HDF5Dataset(serialized_dir, "valset")
    testset = HDF5Dataset(serialized_dir, "testset")

    # Synchronize config dimensions with the actual serialized tensors.
    sample0 = trainset[0]
    x_dim = int(sample0.x.shape[1])
    y_dim = int(sample0.y.shape[1])

    voi = config["NeuralNetwork"]["Variables_of_interest"]
    voi["input_node_features"] = list(range(x_dim))
    voi["node_feature_dims"] = [x_dim]
    voi["output_dim"] = [y_dim]

    log_name = args.modelname
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )

    config = update_config(config, train_loader, val_loader, test_loader)
    config = _to_jsonable(config)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, config["Verbosity"]["level"]
    )

    print_model(model)
    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        config["Verbosity"]["level"],
        create_plots=False,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    print_timers(config["Verbosity"]["level"])
    if writer is not None:
        writer.close()

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
