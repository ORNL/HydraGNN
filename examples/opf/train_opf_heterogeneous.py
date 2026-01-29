import os
import json
import logging
import argparse
import shutil
import subprocess
import sys
from mpi4py import MPI

import torch
import torch.distributed as dist
from torch_geometric.datasets import OPFDataset
import torch_geometric.datasets.opf as tg_opf


def _patch_fast_tar_extraction():
    tar_path = shutil.which("tar")
    if tar_path is None:
        return

    original_extract_tar = tg_opf.extract_tar

    def _fast_extract_tar(path: str, folder: str, mode: str = "r:gz", log: bool = True):
        if log:
            print(f"Extracting {path}", file=sys.stderr)
        try:
            subprocess.run(
                [tar_path, "-xzf", path, "-C", folder],
                check=True,
            )
        except Exception:
            original_extract_tar(path, folder, mode=mode, log=log)

    tg_opf.extract_tar = _fast_extract_tar


def _opf_release_name(topological_perturbations: bool) -> str:
    return (
        "dataset_release_1_nminusone"
        if topological_perturbations
        else "dataset_release_1"
    )


def _opf_raw_dir(root: str, case_name: str, topological_perturbations: bool) -> str:
    return os.path.join(
        root, _opf_release_name(topological_perturbations), case_name, "raw"
    )


def _opf_tmp_dir(root: str, case_name: str, topological_perturbations: bool) -> str:
    return os.path.join(
        _opf_raw_dir(root, case_name, topological_perturbations), "gridopt-dataset-tmp"
    )


def _find_empty_json(root: str):
    empty = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) == 0:
                    empty.append(path)
            except OSError:
                empty.append(path)
    return empty


def _reextract_opf_if_needed(root, case_name, num_groups, topological_perturbations):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    tmp_dir = _opf_tmp_dir(root, case_name, topological_perturbations)
    raw_files = [f"{case_name}_{i}.tar.gz" for i in range(num_groups)]

    if not os.path.isdir(raw_dir):
        return

    missing = [
        name for name in raw_files if not os.path.isfile(os.path.join(raw_dir, name))
    ]
    if missing:
        return

    if os.path.isdir(tmp_dir):
        empty = _find_empty_json(tmp_dir)
        if not empty:
            return
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for name in raw_files:
        tg_opf.extract_tar(os.path.join(raw_dir, name), raw_dir)


import hydragnn
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.utils.distributed import nsplit
from hydragnn.utils.model import print_model
from hydragnn.utils.input_config_parsing.config_utils import update_config

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def _prepare_sample(data, to_homogeneous: bool):
    data.y = data.objective.view(1, 1).to(torch.float32)
    data.graph_attr = data.x.view(1, -1).to(torch.float32)
    if not to_homogeneous:
        return data
    data_h = data.to_homogeneous(
        node_attrs=["x"], add_node_type=True, add_edge_type=True
    )
    data_h.y = data.y
    data_h.graph_attr = data.graph_attr
    return data_h


def _load_split(root, split, case_name, num_groups, topological_perturbations):
    dataset = OPFDataset(
        root=root,
        split=split,
        case_name=case_name,
        num_groups=num_groups,
        topological_perturbations=topological_perturbations,
    )
    return dataset


def _ensure_opf_downloaded(
    root,
    case_name,
    num_groups,
    topological_perturbations,
    rank,
    comm,
):
    if rank == 0:
        _reextract_opf_if_needed(
            root,
            case_name,
            num_groups,
            topological_perturbations,
        )
        try:
            OPFDataset(
                root=root,
                split="train",
                case_name=case_name,
                num_groups=num_groups,
                topological_perturbations=topological_perturbations,
            )
        except json.JSONDecodeError:
            tmp_dir = _opf_tmp_dir(root, case_name, topological_perturbations)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _reextract_opf_if_needed(
                root,
                case_name,
                num_groups,
                topological_perturbations,
            )
            OPFDataset(
                root=root,
                split="train",
                case_name=case_name,
                num_groups=num_groups,
                topological_perturbations=topological_perturbations,
            )
    comm.Barrier()


def _subset_for_rank(dataset, rank, world_size):
    rx = list(nsplit(range(len(dataset)), world_size))[rank]
    return [dataset[i] for i in range(rx.start, rx.stop)]


class HeteroFromHomogeneousDataset:
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        hetero = data.to_heterogeneous()
        if hasattr(data, "y"):
            hetero.y = data.y
        if hasattr(data, "graph_attr"):
            hetero.graph_attr = data.graph_attr
        return hetero


if __name__ == "__main__":
    _patch_fast_tar_extraction()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfile", type=str, default="opf_heterogeneous.json")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument(
        "--case_name",
        type=str,
        default="pglib_opf_case14_ieee",
    )
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--topological_perturbations", action="store_true")
    parser.add_argument("--preonly", action="store_true", help="preprocess only")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--modelname", type=str, default="OPF_Hetero")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, args.data_root)
    input_filename = os.path.join(dirpwd, args.inputfile)

    with open(input_filename, "r") as f:
        config = json.load(f)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )

    log_name = args.modelname
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    _ensure_opf_downloaded(
        datadir,
        args.case_name,
        args.num_groups,
        args.topological_perturbations,
        rank,
        comm,
    )

    info("Loading OPF splits...")
    train_raw = _load_split(
        datadir,
        "train",
        args.case_name,
        args.num_groups,
        args.topological_perturbations,
    )
    val_raw = _load_split(
        datadir,
        "val",
        args.case_name,
        args.num_groups,
        args.topological_perturbations,
    )
    test_raw = _load_split(
        datadir,
        "test",
        args.case_name,
        args.num_groups,
        args.topological_perturbations,
    )

    if args.preonly:
        store_homogeneous = args.format == "adios"
        trainset = [
            _prepare_sample(d, store_homogeneous)
            for d in _subset_for_rank(train_raw, rank, comm_size)
        ]
        valset = [
            _prepare_sample(d, store_homogeneous)
            for d in _subset_for_rank(val_raw, rank, comm_size)
        ]
        testset = [
            _prepare_sample(d, store_homogeneous)
            for d in _subset_for_rank(test_raw, rank, comm_size)
        ]

        if args.format == "adios":
            if AdiosWriter is None:
                raise RuntimeError("adios2 is not available in this environment.")
            fname = os.path.join(dirpwd, "dataset", f"{args.modelname}.bp")
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.save()
        else:
            basedir = os.path.join(dirpwd, "dataset", f"{args.modelname}.pickle")
            SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
            SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
            SimplePickleWriter(testset, basedir, "testset", use_subdir=True)

        dist.destroy_process_group()
        raise SystemExit(0)

    sample = _prepare_sample(train_raw[0], to_homogeneous=False)
    input_dim = max(
        data.x.size(-1) for data in sample.node_stores if hasattr(data, "x")
    )
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["input_node_features"] = list(range(input_dim))
    var_config["node_feature_dims"] = [input_dim]
    var_config["graph_feature_dims"] = [1]

    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        fname = os.path.join(dirpwd, "dataset", f"{args.modelname}.bp")
        train_base = AdiosDataset(fname, "trainset", comm, var_config=None)
        val_base = AdiosDataset(fname, "valset", comm, var_config=None)
        test_base = AdiosDataset(fname, "testset", comm, var_config=None)
        trainset = HeteroFromHomogeneousDataset(train_base)
        valset = HeteroFromHomogeneousDataset(val_base)
        testset = HeteroFromHomogeneousDataset(test_base)
    else:
        basedir = os.path.join(dirpwd, "dataset", f"{args.modelname}.pickle")
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=None
        )
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=None)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=None)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = update_config(config, train_loader, val_loader, test_loader)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    precision = config["NeuralNetwork"]["Training"].get("precision", "fp32")
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
        precision=precision,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(config["Verbosity"]["level"])
    if writer is not None:
        writer.close()

    dist.destroy_process_group()
