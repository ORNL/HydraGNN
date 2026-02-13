"""Train node-level OPF solution prediction.

Use --max_samples <int> to cap the total number of samples preprocessed
across train/val/test splits (proportional allocation). For example,
"--max_samples 100 --preonly" will serialize 100 total samples to pickle.

Arguments summary:
    --case_name <name|all> ...    Select one or more cases, or 'all'.
    --num_groups <int|all>        Select group count or load all available groups.
    --num_groups_max <int>        Fallback/probe cap when 'all' and none on disk.
    --no_num_groups_probe         Disable remote probing for 'all'.
    --node_target_type bus|generator   Choose node target type to predict.
    --preonly                     Preprocess/serialize only (no training).
    --max_samples <int>           Limit total samples across splits.
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

import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.datasets import OPFDataset
import torch_geometric.datasets.opf as tg_opf
from __init__ import data_ops
from opf_nvme_utils import stage_case_to_nvme


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
from hydragnn.utils.print import iterate_tqdm
from hydragnn.utils.input_config_parsing.config_utils import update_config

from opf_solution_utils import (
    HeteroFromHomogeneousDataset,
    NodeBatchAdapter,
    NodeTargetDatasetAdapter,
    ensure_node_y_loc as _ensure_node_y_loc,
    info,
    resolve_node_target_type as _resolve_node_target_type,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None


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
    data.y_num_nodes = torch.tensor(
        [num_nodes], dtype=torch.int64, device=data.y.device
    )
    data.y_loc = torch.tensor(
        [[0, num_nodes * target_dim]],
        dtype=torch.int64,
        device=data.y.device,
    )


def _ensure_non_scalar_attrs(data):
    if hasattr(data, "_global_store"):
        store = data._global_store
        for key in list(store.keys()):
            value = store[key]
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                store[key] = value.reshape(1)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                store[key] = value.reshape(1)
        return data
    keys = data.keys() if callable(data.keys) else data.keys
    for key in list(keys):
        value = data[key]
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            data[key] = value.reshape(1)
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            data[key] = value.reshape(1)
    return data


def _to_int_num_nodes(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return None
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return int(value.reshape(1)[0])
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_num_nodes_from_edges(data):
    inferred = {}
    if not hasattr(data, "edge_types"):
        return inferred
    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        if not hasattr(edge_store, "edge_index") or edge_store.edge_index is None:
            continue
        edge_index = edge_store.edge_index
        if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        src_max = int(edge_index[0].max().item()) + 1
        dst_max = int(edge_index[1].max().item()) + 1
        inferred[src_type] = max(inferred.get(src_type, 0), src_max)
        inferred[dst_type] = max(inferred.get(dst_type, 0), dst_max)
    return inferred


def _ensure_node_store_metadata(data, target_dim: int):
    if not hasattr(data, "node_types"):
        return data
    inferred = _infer_num_nodes_from_edges(data)
    for node_type in data.node_types:
        store = data[node_type]
        num_nodes = None
        if "num_nodes" in store:
            num_nodes = store["num_nodes"]
        if num_nodes is None and hasattr(data, "num_nodes_dict"):
            num_nodes = data.num_nodes_dict.get(node_type)
        if num_nodes is None:
            num_nodes = inferred.get(node_type)
        if num_nodes is None:
            if hasattr(store, "x") and store.x is not None:
                num_nodes = store.x.shape[0]
            elif hasattr(store, "y") and store.y is not None:
                num_nodes = store.y.shape[0]
        num_nodes = _to_int_num_nodes(num_nodes)
        if num_nodes is None:
            num_nodes = 0
        store.num_nodes = num_nodes
        if not hasattr(store, "y") or store.y is None:
            store.y = torch.zeros(
                (int(num_nodes), int(target_dim)),
                dtype=torch.float32,
                device=data.y.device,
            )
    return data


def _ensure_node_store_features(data):
    if not hasattr(data, "node_types"):
        return data
    default_x_dim = None
    for node_type in data.node_types:
        store = data[node_type]
        if hasattr(store, "x") and store.x is not None:
            default_x_dim = int(store.x.shape[1]) if store.x.dim() > 1 else 1
            break
    if default_x_dim is None:
        return data
    for node_type in data.node_types:
        store = data[node_type]
        if not hasattr(store, "x") or store.x is None:
            num_nodes = int(getattr(store, "num_nodes", 0))
            store.x = torch.zeros(
                (num_nodes, default_x_dim),
                dtype=torch.float32,
                device=data.y.device,
            )
    return data


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _prepare_sample(
    data, node_target_type: str, case_name: str, to_homogeneous: bool = False
):
    data.y = _build_solution_target(data, node_target_type)
    _ensure_node_y_loc(data)
    data.graph_attr = data.x.view(1, -1).to(torch.float32)
    data.case_name = case_name
    _ensure_non_scalar_attrs(data)
    if hasattr(data, "num_nodes_dict"):
        delattr(data, "num_nodes_dict")
    if not to_homogeneous:
        return data
    _ensure_node_store_metadata(data, target_dim=int(data.y.shape[1]))
    _ensure_node_store_features(data)
    data_h = data.to_homogeneous(
        node_attrs=["x", "y"], add_node_type=True, add_edge_type=True
    )
    data_h.graph_attr = data.graph_attr
    data_h.case_name = case_name
    _ensure_non_scalar_attrs(data_h)
    if hasattr(data_h, "num_nodes_dict"):
        delattr(data_h, "num_nodes_dict")
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


def _parse_case_list(case_name_args):
    if not case_name_args:
        return []
    if isinstance(case_name_args, str):
        return [case_name_args]
    return [c.strip() for c in case_name_args if c.strip()]


def _subset_for_rank(dataset, rank, world_size, max_samples=None):
    if max_samples is None:
        indices = range(len(dataset))
    else:
        max_samples = max(0, min(int(max_samples), len(dataset)))
        indices = range(max_samples)
    rx = list(nsplit(indices, world_size))[rank]
    return [dataset[i] for i in range(rx.start, rx.stop)]


def _allocate_split_caps(max_samples, split_sizes):
    if max_samples is None:
        return {name: None for name in split_sizes}
    total = sum(split_sizes.values())
    if total <= 0:
        return {name: 0 for name in split_sizes}
    raw = {
        name: (max_samples * (size / total)) if size > 0 else 0
        for name, size in split_sizes.items()
    }
    caps = {name: min(split_sizes[name], int(raw[name])) for name in split_sizes}
    remainder = max_samples - sum(caps.values())
    order = sorted(
        split_sizes.keys(),
        key=lambda name: (raw[name] - int(raw[name])),
        reverse=True,
    )
    for name in order:
        if remainder <= 0:
            break
        if caps[name] < split_sizes[name]:
            caps[name] += 1
            remainder -= 1
    return caps


if __name__ == "__main__":
    _patch_fast_tar_extraction()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", type=str, default="opf_solution_heterogeneous.json"
    )
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument(
        "--case_name",
        nargs="+",
        default=["pglib_opf_case14_ieee"],
        help="Case name(s) or 'all'",
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
        default=128,
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
        "--max_samples",
        type=int,
        default=None,
        help="Limit total number of samples across train/val/test splits",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--modelname", type=str, default="OPF_Solution")
    parser.add_argument(
        "--nvme",
        action="store_true",
        help="Stage selected OPF case(s) onto node-local NVMe/scratch if available",
    )
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

    requested_num_groups = data_ops.parse_num_groups(args.num_groups)
    parsed_case_names = _parse_case_list(args.case_name)
    if len(parsed_case_names) == 1 and parsed_case_names[0].lower() == "all":
        case_names = data_ops.discover_cases(datadir, args.topological_perturbations)
        if not case_names:
            case_names = list(_DEFAULT_CASE_NAMES)
        if not case_names:
            raise RuntimeError("No OPF cases found.")
    else:
        case_names = parsed_case_names

    case_num_groups = {}
    shared_datadir = datadir
    active_datadir = datadir
    serialized_target = (
        f"{args.modelname}.bp" if args.format == "adios" else f"{args.modelname}.pickle"
    )
    for case_name in case_names:
        num_groups = data_ops.resolve_num_groups(
            requested_num_groups,
            shared_datadir,
            case_name,
            args.topological_perturbations,
            args.num_groups_max,
            args.num_groups_probe,
            rank,
            comm,
        )
        case_num_groups[case_name] = num_groups
        data_ops.ensure_opf_downloaded(
            shared_datadir,
            case_name,
            num_groups,
            args.topological_perturbations,
            rank,
            comm,
        )
        if args.nvme:
            staged_datadir = stage_case_to_nvme(
                shared_datadir,
                case_name,
                args.topological_perturbations,
                comm,
                rank,
                None,
                serialized_targets=[serialized_target],
            )
            if staged_datadir != shared_datadir:
                active_datadir = staged_datadir
            else:
                active_datadir = shared_datadir
                break

    datadir = active_datadir

    if rank == 0:
        info("Loading OPF splits...")
    train_raw = []
    val_raw = []
    test_raw = []
    for case_name in case_names:
        num_groups = case_num_groups[case_name]
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

    split_sizes = {
        "train": sum(len(d) for d in train_raw),
        "val": sum(len(d) for d in val_raw),
        "test": sum(len(d) for d in test_raw),
    }
    split_caps = _allocate_split_caps(args.max_samples, split_sizes)
    if args.max_samples is not None:
        info(
            "Limiting samples across splits: "
            f"train={split_caps['train']}, val={split_caps['val']}, test={split_caps['test']}"
        )

    if args.preonly:
        store_homogeneous = args.format == "adios"
        verbosity = config["Verbosity"]["level"]
        trainset = []
        valset = []
        testset = []
        remaining_caps = dict(split_caps)
        for case_name, train_split in zip(case_names, train_raw):
            case_cap = remaining_caps["train"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(train_split)))
            subset = _subset_for_rank(train_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess train {case_name}", leave=False
            ):
                trainset.append(
                    _prepare_sample(
                        d, args.node_target_type, case_name, store_homogeneous
                    )
                )
            if remaining_caps["train"] is not None:
                remaining_caps["train"] = max(
                    0, remaining_caps["train"] - min(len(train_split), case_cap or 0)
                )
        for case_name, val_split in zip(case_names, val_raw):
            case_cap = remaining_caps["val"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(val_split)))
            subset = _subset_for_rank(val_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess val {case_name}", leave=False
            ):
                valset.append(
                    _prepare_sample(
                        d, args.node_target_type, case_name, store_homogeneous
                    )
                )
            if remaining_caps["val"] is not None:
                remaining_caps["val"] = max(
                    0, remaining_caps["val"] - min(len(val_split), case_cap or 0)
                )
        for case_name, test_split in zip(case_names, test_raw):
            case_cap = remaining_caps["test"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(test_split)))
            subset = _subset_for_rank(test_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess test {case_name}", leave=False
            ):
                testset.append(
                    _prepare_sample(
                        d, args.node_target_type, case_name, store_homogeneous
                    )
                )
            if remaining_caps["test"] is not None:
                remaining_caps["test"] = max(
                    0, remaining_caps["test"] - min(len(test_split), case_cap or 0)
                )

        info(
            f"Local split sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
        )

        if args.format == "adios":
            if AdiosWriter is None:
                raise RuntimeError("adios2 is not available in this environment.")
            fname = os.path.join(datadir, f"{args.modelname}.bp")
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.save()
        else:
            basedir = os.path.join(datadir, f"{args.modelname}.pickle")
            SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
            SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
            SimplePickleWriter(testset, basedir, "testset", use_subdir=True)

        comm.Barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)

    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        fname = os.path.join(datadir, f"{args.modelname}.bp")
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

    resolved_node_target_type = _resolve_node_target_type(
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
    config = _to_jsonable(config)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    precision = config["NeuralNetwork"]["Training"].get("precision", "fp32")
    metadata = None
    try:
        metadata = trainset[0].metadata()
    except Exception as exc:
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
