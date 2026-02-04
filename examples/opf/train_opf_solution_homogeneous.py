"""Train node-level OPF solution prediction (homogeneous graph).

Arguments summary:
    --case_name <name|all>        Select a single case or load all cases.
    --case_names <a,b,c>          Comma-separated case list (used if --case_name all).
    --case_list_file <path>       File with one case name per line.
    --num_groups <int|all>        Select group count or load all available groups.
    --num_groups_max <int>        Fallback group count when 'all' and none on disk.
    --node_target_type bus|generator   Choose node target type to predict.
    --preonly                     Preprocess/serialize only (no training).
    --adios / --pickle            Serialization format.
    --batch_size / --num_epoch    Override training hyperparameters.
"""

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


_DEFAULT_CASE_NAMES = [
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case500_goc",
    "pglib_opf_case2000_goc",
    "pglib_opf_case6470_rte",
    "pglib_opf_case4661_sdet",
    "pglib_opf_case10000_goc",
    "pglib_opf_case13659_pegase",
]


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
                subprocess.run([tar_path, "-xzf", path, "-C", folder], check=True)
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


def _opf_release_dir(root: str, topological_perturbations: bool) -> str:
    return os.path.join(root, _opf_release_name(topological_perturbations))


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


def _discover_cases(root: str, topological_perturbations: bool):
    release_dir = _opf_release_dir(root, topological_perturbations)
    if not os.path.isdir(release_dir):
        return []
    return sorted(
        name
        for name in os.listdir(release_dir)
        if os.path.isdir(os.path.join(release_dir, name))
    )


def _discover_num_groups(root: str, case_name: str, topological_perturbations: bool):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    if not os.path.isdir(raw_dir):
        return 0
    groups = []
    for name in os.listdir(raw_dir):
        if not name.startswith(f"{case_name}_") or not name.endswith(".tar.gz"):
            continue
        try:
            idx = int(name[len(case_name) + 1 : -len(".tar.gz")])
            groups.append(idx)
        except ValueError:
            continue
    return max(groups) + 1 if groups else 0


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
from hydragnn.utils.print import iterate_tqdm
from hydragnn.utils.input_config_parsing.config_utils import update_config

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def _build_solution_target(data, node_target_type: str):
    if hasattr(data, "node_types") and node_target_type in data.node_types:
        node_store = data[node_target_type]
        if not hasattr(node_store, "y") or node_store.y is None:
            raise RuntimeError(
                f"No targets found for node type '{node_target_type}' in OPF sample."
            )
        return node_store.y.to(torch.float32)

    if hasattr(data, "_node_type_names") and hasattr(data, "node_type"):
        if node_target_type not in data._node_type_names:
            raise RuntimeError(
                f"Node type '{node_target_type}' not found in OPF sample."
            )
        type_index = data._node_type_names.index(node_target_type)
        if not hasattr(data, "y") or data.y is None:
            raise RuntimeError(
                f"No homogeneous targets found for node type '{node_target_type}'."
            )
        mask = data.node_type == type_index
        return data.y[mask].to(torch.float32)

    raise RuntimeError(f"Node type '{node_target_type}' not found in OPF sample.")


def _ensure_node_y_loc(data):
    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError("Missing node targets (data.y) for OPF sample.")
    if data.y.dim() == 1:
        data.y = data.y.unsqueeze(-1)
    num_nodes = int(data.y.shape[0])
    target_dim = int(data.y.shape[1])
    data.y_num_nodes = torch.tensor(num_nodes, dtype=torch.int64, device=data.y.device)
    data.y_loc = torch.tensor(
        [[0, num_nodes * target_dim]],
        dtype=torch.int64,
        device=data.y.device,
    )


def _prepare_sample(
    data, node_target_type: str, case_name: str, to_homogeneous: bool = True
):
    data.y = _build_solution_target(data, node_target_type)
    _ensure_node_y_loc(data)
    data.graph_attr = data.x.view(1, -1).to(torch.float32)
    data.case_name = case_name
    if not to_homogeneous:
        return data
    data_h = data.to_homogeneous(
        node_attrs=["x", "y"], add_node_type=True, add_edge_type=True
    )
    data_h.graph_attr = data.graph_attr
    data_h.case_name = case_name
    return data_h


def _resolve_node_target_type(data, requested: str) -> str:
    if hasattr(data, "node_types"):
        if requested in data.node_types:
            return requested
        if hasattr(data, "_node_type_names") and requested in data._node_type_names:
            idx = data._node_type_names.index(requested)
            if idx < len(data.node_types):
                return data.node_types[idx]
        for name in data.node_types:
            if str(name).lower() == requested.lower():
                return name
        if len(data.node_types) > 0:
            return data.node_types[0]
    if hasattr(data, "_node_type_names") and requested in data._node_type_names:
        return requested
    return requested


def _load_split(root, split, case_name, num_groups, topological_perturbations):
    dataset = OPFDataset(
        root=root,
        split=split,
        case_name=case_name,
        num_groups=num_groups,
        topological_perturbations=topological_perturbations,
    )
    return dataset


def _parse_num_groups(num_groups_arg: str) -> int | None:
    if isinstance(num_groups_arg, int):
        return num_groups_arg
    if str(num_groups_arg).lower() == "all":
        return None
    return int(num_groups_arg)


def _parse_case_list(args):
    cases = []
    if args.case_names:
        cases.extend([c.strip() for c in args.case_names.split(",") if c.strip()])
    if args.case_list_file:
        with open(args.case_list_file, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    cases.append(name)
    return cases


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


class HomogeneousDatasetAdapter:
    def __init__(self, base, node_target_type: str):
        self.base = base
        self.node_target_type = node_target_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        if hasattr(data, "node_types"):
            data = data.to_homogeneous(
                node_attrs=["x", "y"], add_node_type=True, add_edge_type=True
            )
        if not hasattr(data, "node_type") or not hasattr(data, "_node_type_names"):
            raise RuntimeError("Expected homogeneous OPF sample with node_type.")
        if self.node_target_type not in data._node_type_names:
            raise RuntimeError(
                f"Node type '{self.node_target_type}' not found in OPF sample."
            )
        if not hasattr(data, "y") or data.y is None:
            raise RuntimeError(
                f"No targets found for node type '{self.node_target_type}' in OPF sample."
            )
        type_index = data._node_type_names.index(self.node_target_type)
        mask = data.node_type == type_index
        data.y = data.y[mask]
        if hasattr(data, "batch"):
            data.batch = data.batch[mask]
        _ensure_node_y_loc(data)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class HomogeneousBatchAdapter:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        for data in self.loader:
            _ensure_node_y_loc(data)
            yield data

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


if __name__ == "__main__":
    _patch_fast_tar_extraction()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", type=str, default="opf_solution_homogeneous.json"
    )
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument(
        "--case_name",
        type=str,
        default="pglib_opf_case14_ieee",
        help="Case name or 'all'",
    )
    parser.add_argument(
        "--case_names",
        type=str,
        default="",
        help="Comma-separated case list (used if --case_name all)",
    )
    parser.add_argument(
        "--case_list_file",
        type=str,
        default="",
        help="File with one case name per line (used if --case_name all)",
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
        default=1,
        help="Fallback group count when --num_groups all and none on disk",
    )
    parser.add_argument("--topological_perturbations", action="store_true")
    parser.add_argument("--preonly", action="store_true", help="preprocess only")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--modelname", type=str, default="OPF_Solution")
    parser.add_argument(
        "--node_target_type",
        type=str,
        default="bus",
        choices=["bus", "generator"],
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

    if "node_target_type" in config.get("NeuralNetwork", {}).get("Architecture", {}):
        args.node_target_type = config["NeuralNetwork"]["Architecture"][
            "node_target_type"
        ]

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

    requested_num_groups = _parse_num_groups(args.num_groups)
    if args.case_name.lower() == "all":
        case_names = _parse_case_list(args)
        if not case_names:
            case_names = list(_DEFAULT_CASE_NAMES)
        if not case_names:
            raise RuntimeError(
                "No OPF cases found. Provide --case_names or --case_list_file."
            )
    else:
        case_names = [args.case_name]

    for case_name in case_names:
        num_groups = requested_num_groups
        if num_groups is None:
            num_groups = _discover_num_groups(
                datadir, case_name, args.topological_perturbations
            )
            if num_groups == 0:
                if args.num_groups_max <= 0:
                    raise RuntimeError(f"No groups found for case '{case_name}'.")
                num_groups = args.num_groups_max
        _ensure_opf_downloaded(
            datadir,
            case_name,
            num_groups,
            args.topological_perturbations,
            rank,
            comm,
        )

    info("Loading OPF splits...")
    train_raw = []
    val_raw = []
    test_raw = []
    for case_name in case_names:
        num_groups = requested_num_groups
        if num_groups is None:
            num_groups = _discover_num_groups(
                datadir, case_name, args.topological_perturbations
            )
            if num_groups == 0:
                if args.num_groups_max <= 0:
                    raise RuntimeError(f"No groups found for case '{case_name}'.")
                num_groups = args.num_groups_max
        train_raw.append(
            _load_split(
                datadir,
                "train",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )
        val_raw.append(
            _load_split(
                datadir,
                "val",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )
        test_raw.append(
            _load_split(
                datadir,
                "test",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )

    if args.preonly:
        verbosity = config["Verbosity"]["level"]
        trainset = []
        valset = []
        testset = []
        for case_name, train_split in zip(case_names, train_raw):
            subset = _subset_for_rank(train_split, rank, comm_size)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess train {case_name}", leave=False
            ):
                trainset.append(
                    _prepare_sample(d, args.node_target_type, case_name, True)
                )
        for case_name, val_split in zip(case_names, val_raw):
            subset = _subset_for_rank(val_split, rank, comm_size)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess val {case_name}", leave=False
            ):
                valset.append(_prepare_sample(d, args.node_target_type, case_name, True))
        for case_name, test_split in zip(case_names, test_raw):
            subset = _subset_for_rank(test_split, rank, comm_size)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess test {case_name}", leave=False
            ):
                testset.append(
                    _prepare_sample(d, args.node_target_type, case_name, True)
                )

        info(
            f"Local split sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
        )

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

    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        fname = os.path.join(dirpwd, "dataset", f"{args.modelname}.bp")
        train_base = AdiosDataset(fname, "trainset", comm, var_config=None)
        val_base = AdiosDataset(fname, "valset", comm, var_config=None)
        test_base = AdiosDataset(fname, "testset", comm, var_config=None)
        trainset = HomogeneousDatasetAdapter(train_base, args.node_target_type)
        valset = HomogeneousDatasetAdapter(val_base, args.node_target_type)
        testset = HomogeneousDatasetAdapter(test_base, args.node_target_type)
    else:
        basedir = os.path.join(dirpwd, "dataset", f"{args.modelname}.pickle")
        trainset = HomogeneousDatasetAdapter(
            SimplePickleDataset(basedir=basedir, label="trainset", var_config=None),
            args.node_target_type,
        )
        valset = HomogeneousDatasetAdapter(
            SimplePickleDataset(basedir=basedir, label="valset", var_config=None),
            args.node_target_type,
        )
        testset = HomogeneousDatasetAdapter(
            SimplePickleDataset(basedir=basedir, label="testset", var_config=None),
            args.node_target_type,
        )

    resolved_node_target_type = _resolve_node_target_type(
        trainset[0], args.node_target_type
    )
    if resolved_node_target_type != args.node_target_type:
        info(
            f"Resolved node_target_type '{args.node_target_type}' -> '{resolved_node_target_type}'"
        )
        args.node_target_type = resolved_node_target_type

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    train_loader = HomogeneousBatchAdapter(train_loader)
    val_loader = HomogeneousBatchAdapter(val_loader)
    test_loader = HomogeneousBatchAdapter(test_loader)

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
