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
from __init__ import data_ops
from opf_nvme_utils import stage_case_to_nvme


def _patch_fast_tar_extraction():
    tar_path = shutil.which("tar")
    if tar_path is None:
        return

    original_extract_tar = tg_opf.extract_tar

    def _fast_extract_tar(path: str, folder: str, mode: str = "r:gz", log: bool = True):
        if log:
            print(f"Extracting {path}", file=sys.stderr)
        try:
            try:
                subprocess.run(
                    [
                        tar_path,
                        "--checkpoint=1000",
                        "--checkpoint-action=dot",
                        "-xzf",
                        path,
                        "-C",
                        folder,
                    ],
                    check=True,
                )
                if log:
                    print("", file=sys.stderr)
                return
            except Exception:
                subprocess.run(
                    [tar_path, "-xzf", path, "-C", folder],
                    check=True,
                )
        except Exception:
            original_extract_tar(path, folder, mode=mode, log=log)

    tg_opf.extract_tar = _fast_extract_tar


import hydragnn
import time


def _diag(msg: str):
    if os.getenv("HYDRAGNN_DIAG") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_filter = os.getenv("HYDRAGNN_DIAG_RANK")
    if rank_filter is not None:
        try:
            if int(rank_filter) != int(rank):
                return
        except ValueError:
            pass
    now = time.perf_counter()
    print(f"[diag][rank {rank}][{now:.3f}] {msg}", flush=True)


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
    if dist.is_initialized():
        if dist.get_rank() == 0:
            info(
                f"Loading OPF split: case={case_name} split={split} groups={num_groups}"
            )
    else:
        info(f"Loading OPF split: case={case_name} split={split} groups={num_groups}")

    def _construct(force_reload: bool = False):
        return OPFDataset(
            root=root,
            split=split,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            force_reload=force_reload,
        )

    try:
        return _construct(force_reload=False)
    except Exception as exc:
        msg = str(exc)
        recoverable = isinstance(exc, EOFError) or any(
            token in msg
            for token in (
                "PytorchStreamReader failed reading file",
                "PytorchStreamReader failed reading zip archive",
                "failed finding central directory",
                "Cannot use ``weights_only=True`` with files saved in the legacy .tar format",
                "Weights only load failed",
                "Unsupported operand 80",
            )
        )
        if not recoverable:
            raise

    if dist.is_initialized() and dist.get_world_size() > 1:
        rank = dist.get_rank()
        info(
            f"Corrupted processed cache detected for case={case_name} split={split}; rank {rank} rebuilding with force_reload=True"
        )
        dataset = _construct(force_reload=True)
        MPI.COMM_WORLD.Barrier()
        return dataset

    info(
        f"Corrupted processed cache detected for case={case_name} split={split}; rebuilding with force_reload=True"
    )
    return _construct(force_reload=True)


def _prime_processed_splits_on_rank0(
    root,
    case_name,
    num_groups,
    topological_perturbations,
    rank,
    comm,
):
    if rank == 0:
        for split in ("train", "val", "test"):
            _load_split(
                root,
                split,
                case_name,
                num_groups,
                topological_perturbations,
            )
    comm.Barrier()


def _subset_for_rank(dataset, rank, world_size):
    rx = list(nsplit(range(len(dataset)), world_size))[rank]
    return [dataset[i] for i in range(rx.start, rx.stop)]


def _log_phase_time(comm, rank, label: str, elapsed_local: float):
    elapsed_max = comm.allreduce(float(elapsed_local), op=MPI.MAX)
    elapsed_sum = comm.allreduce(float(elapsed_local), op=MPI.SUM)
    elapsed_avg = elapsed_sum / max(1, comm.Get_size())
    if rank == 0:
        info(f"Timing {label}: max={elapsed_max:.2f}s avg={elapsed_avg:.2f}s")


def _resolve_preonly_case_names(args, datadir):
    if not args.preonly_case_names:
        return [args.case_name]

    case_names = [c.strip() for c in args.preonly_case_names if c.strip()]
    if len(case_names) == 1 and case_names[0].lower() == "all":
        discovered = data_ops.discover_cases(datadir, args.topological_perturbations)
        if discovered:
            return discovered
        return [args.case_name]
    return case_names


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
    parser.add_argument(
        "--inputfile", type=str, default="opf_graph_output_heterogeneous.json"
    )
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument(
        "--case_name",
        type=str,
        default="pglib_opf_case14_ieee",
    )
    parser.add_argument(
        "--num_groups",
        type=str,
        default="1",
        help="Number of groups or 'all'",
    )
    parser.add_argument(
        "--num_groups_max",
        type=int,
        default=20,
        help="Fallback/probe cap when --num_groups all and none on disk",
    )
    parser.add_argument(
        "--no_num_groups_probe",
        action="store_false",
        dest="num_groups_probe",
        help="Disable probing remote storage when --num_groups all and none on disk",
    )
    parser.set_defaults(num_groups_probe=True)
    parser.add_argument("--topological_perturbations", action="store_true")
    parser.add_argument("--preonly", action="store_true", help="preprocess only")
    parser.add_argument(
        "--preonly_case_names",
        nargs="+",
        default=None,
        help="Case names to preprocess when --preonly is set; supports 'all'",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--modelname", type=str, default="OPF_Hetero")
    parser.add_argument(
        "--nvme",
        action="store_true",
        help="Stage selected OPF case onto node-local NVMe/scratch if available",
    )

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

    requested_num_groups = data_ops.parse_num_groups(args.num_groups)
    adios_training_only = args.format == "adios" and not args.preonly

    if args.preonly:
        case_names = _resolve_preonly_case_names(args, datadir)
        trainset, valset, testset = [], [], []
        output_datadir = datadir
        store_homogeneous = args.format == "adios"

        for case_name in case_names:
            t_case = time.perf_counter()
            num_groups = data_ops.resolve_num_groups(
                requested_num_groups,
                datadir,
                case_name,
                args.topological_perturbations,
                args.num_groups_max,
                args.num_groups_probe,
                rank,
                comm,
            )

            t_download = time.perf_counter()
            data_ops.ensure_opf_downloaded(
                datadir,
                case_name,
                num_groups,
                args.topological_perturbations,
                rank,
                comm,
            )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} phase=download_extract groups={num_groups}",
                time.perf_counter() - t_download,
            )

            case_datadir = datadir
            if args.nvme:
                serialized_target = (
                    f"{args.modelname}.bp"
                    if args.format == "adios"
                    else f"{args.modelname}.pickle"
                )
                t_stage = time.perf_counter()
                case_datadir = stage_case_to_nvme(
                    datadir,
                    case_name,
                    args.topological_perturbations,
                    comm,
                    rank,
                    None,
                    serialized_targets=[serialized_target],
                )
                _log_phase_time(
                    comm,
                    rank,
                    f"case={case_name} phase=nvme_stage groups={num_groups}",
                    time.perf_counter() - t_stage,
                )
                if args.format != "adios":
                    output_datadir = case_datadir

            _prime_processed_splits_on_rank0(
                case_datadir,
                case_name,
                num_groups,
                args.topological_perturbations,
                rank,
                comm,
            )

            t_load = time.perf_counter()
            train_raw = _load_split(
                case_datadir,
                "train",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=train phase=load groups={num_groups}",
                time.perf_counter() - t_load,
            )

            t_load = time.perf_counter()
            val_raw = _load_split(
                case_datadir,
                "val",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=val phase=load groups={num_groups}",
                time.perf_counter() - t_load,
            )

            t_load = time.perf_counter()
            test_raw = _load_split(
                case_datadir,
                "test",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=test phase=load groups={num_groups}",
                time.perf_counter() - t_load,
            )

            t_pre = time.perf_counter()
            train_subset = _subset_for_rank(train_raw, rank, comm_size)
            trainset.extend(_prepare_sample(d, store_homogeneous) for d in train_subset)
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=train phase=preprocess local_samples={len(train_subset)}",
                time.perf_counter() - t_pre,
            )

            t_pre = time.perf_counter()
            val_subset = _subset_for_rank(val_raw, rank, comm_size)
            valset.extend(_prepare_sample(d, store_homogeneous) for d in val_subset)
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=val phase=preprocess local_samples={len(val_subset)}",
                time.perf_counter() - t_pre,
            )

            t_pre = time.perf_counter()
            test_subset = _subset_for_rank(test_raw, rank, comm_size)
            testset.extend(_prepare_sample(d, store_homogeneous) for d in test_subset)
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=test phase=preprocess local_samples={len(test_subset)}",
                time.perf_counter() - t_pre,
            )

            _log_phase_time(
                comm,
                rank,
                f"case={case_name} phase=prepare_total groups={num_groups}",
                time.perf_counter() - t_case,
            )

        info(
            f"Local split sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
        )

        if args.format == "adios":
            t_write = time.perf_counter()
            if AdiosWriter is None:
                raise RuntimeError("adios2 is not available in this environment.")
            fname = os.path.join(output_datadir, f"{args.modelname}.bp")
            if rank == 0 and os.path.exists(fname):
                if os.path.isdir(fname):
                    shutil.rmtree(fname, ignore_errors=True)
                else:
                    os.remove(fname)
            comm.Barrier()
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.save()
            _log_phase_time(
                comm,
                rank,
                f"phase=serialize format=adios model={args.modelname}",
                time.perf_counter() - t_write,
            )
        else:
            t_write = time.perf_counter()
            basedir = os.path.join(output_datadir, f"{args.modelname}.pickle")
            SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
            SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
            SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
            _log_phase_time(
                comm,
                rank,
                f"phase=serialize format=pickle model={args.modelname}",
                time.perf_counter() - t_write,
            )

        comm.Barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)

    if not adios_training_only:
        num_groups = data_ops.resolve_num_groups(
            requested_num_groups,
            datadir,
            args.case_name,
            args.topological_perturbations,
            args.num_groups_max,
            args.num_groups_probe,
            rank,
            comm,
        )

        data_ops.ensure_opf_downloaded(
            datadir,
            args.case_name,
            num_groups,
            args.topological_perturbations,
            rank,
            comm,
        )

        if args.nvme:
            serialized_target = (
                f"{args.modelname}.bp"
                if args.format == "adios"
                else f"{args.modelname}.pickle"
            )
            datadir = stage_case_to_nvme(
                datadir,
                args.case_name,
                args.topological_perturbations,
                comm,
                rank,
                None,
                serialized_targets=[serialized_target],
            )

        _prime_processed_splits_on_rank0(
            datadir,
            args.case_name,
            num_groups,
            args.topological_perturbations,
            rank,
            comm,
        )

        info("Loading OPF splits...")
        train_raw = _load_split(
            datadir,
            "train",
            args.case_name,
            num_groups,
            args.topological_perturbations,
        )
        val_raw = _load_split(
            datadir,
            "val",
            args.case_name,
            num_groups,
            args.topological_perturbations,
        )
        test_raw = _load_split(
            datadir,
            "test",
            args.case_name,
            num_groups,
            args.topological_perturbations,
        )

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
        fname = os.path.join(datadir, f"{args.modelname}.bp")
        if adios_training_only and not os.path.isdir(fname):
            raise RuntimeError(
                f"Expected preprocessed ADIOS dataset at '{fname}' for training-only mode. "
                "Run with --preonly --adios first."
            )
        train_base = AdiosDataset(fname, "trainset", comm, var_config=None)
        val_base = AdiosDataset(fname, "valset", comm, var_config=None)
        test_base = AdiosDataset(fname, "testset", comm, var_config=None)
        trainset = HeteroFromHomogeneousDataset(train_base)
        valset = HeteroFromHomogeneousDataset(val_base)
        testset = HeteroFromHomogeneousDataset(test_base)
    else:
        basedir = os.path.join(datadir, f"{args.modelname}.pickle")
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=None
        )
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=None)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=None)

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = update_config(config, train_loader, val_loader, test_loader)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    precision = config["NeuralNetwork"]["Training"].get("precision", "fp32")
    metadata = None
    try:
        metadata = trainset[0].metadata()
    except Exception as exc:
        info(f"Unable to fetch hetero metadata: {exc}")
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
        metadata=metadata,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model,
        optimizer,
        config["Verbosity"]["level"],
        find_unused_parameters=True,
    )

    _diag("Entering print_model")
    print_model(model)
    _diag("Exited print_model")

    _diag("Entering load_existing_model_config")
    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )
    _diag("Exited load_existing_model_config")

    _diag("Entering train_validate_test")
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
    _diag("Exited train_validate_test")

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(config["Verbosity"]["level"])
    if writer is not None:
        writer.close()

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
