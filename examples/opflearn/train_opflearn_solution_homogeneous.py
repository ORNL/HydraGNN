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

import logging
import os
from pathlib import Path

import torch.distributed as dist
from mpi4py import MPI

from __init__ import data_ops
from hydragnn import cli


def parse_args():
    parser = cli.build_parser(
        default_inputfile="opflearn_solution_homogeneous.json",
        default_format="hdf5",
        formats=("hdf5",),
    )
    parser.set_defaults(modelname="OPFLearn_Solution")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--case_name", type=str, default="case14_ieee")
    parser.add_argument("--formulation", type=str, default="ACOPF")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download/re-serialize even if data exists.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, args.data_root)
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, args.modelname)
    comm = MPI.COMM_WORLD

    config.setdefault("Task", {})
    config["Task"]["formulation"] = args.formulation
    config["Task"]["case_name"] = args.case_name

    # Download + preprocess (rank 0), producing a processed Parquet file.
    parquet_path = data_ops.ensure_opflearn_prepared(
        root=datadir,
        case_name=args.case_name,
        rank=rank,
        comm=comm,
        overwrite=args.overwrite,
    )

    serialized_dir = os.path.join(datadir, f"{args.modelname}.h5")

    if args.preonly or args.overwrite or not os.path.isdir(serialized_dir):
        data_ops.serialize_parquet_to_hdf5(
            input_parquet=Path(parquet_path),
            output_hdf5_dir=Path(serialized_dir),
            max_samples=args.max_samples,
            include_duals=False,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            overwrite=True,
            logger=logging.getLogger("serialize_opflearn"),
        )
        comm.Barrier()

        if args.preonly:
            if dist.is_initialized():
                dist.destroy_process_group()
            return

    trainset, valset, testset = cli.load_split_datasets(serialized_dir, "hdf5")

    # Synchronize config dimensions with the actual serialized tensors.
    sample0 = trainset[0]
    x_dim = int(sample0.x.shape[1])
    y_dim = int(sample0.y.shape[1])

    voi = config["NeuralNetwork"]["Variables_of_interest"]
    voi["input_node_features"] = list(range(x_dim))
    voi["node_feature_dims"] = [x_dim]
    voi["output_dim"] = [y_dim]

    for variable in config["Variables"]["inputs"]:
        if variable["name"] == "load_setpoints":
            variable["dim"] = x_dim
    for variable in config["Variables"]["outputs"]:
        if variable["name"] == "opf_solution":
            variable["dim"] = y_dim

    cli.run_training(
        config,
        trainset,
        valset,
        testset,
        log_name,
        writer,
        precision=args.precision,
        everyone=args.everyone,
        variables=False,
    )

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
