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
import copy
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
    EdgeAttrDatasetAdapter,
    OPFEnhancedModelWrapper,
    HeteroFromHomogeneousDataset,
    NodeBatchAdapter,
    NodeTargetDatasetAdapter,
    OPFDomainLoss,
    assemble_edge_attr,
    build_solution_target as _build_solution_target,
    compute_pna_deg_for_hetero_dataset,
    validate_voi_node_features,
    ensure_node_y_loc as _ensure_node_y_loc,
    info,
    resolve_edge_feature_schema,
    resolve_node_target_type as _resolve_node_target_type,
)

from hydragnn.utils.datasets.hdf5dataset import HDF5Writer, HDF5Dataset

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None


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


def _validate_node_stores_for_homogeneous(data):
    """Crash if any node type is missing 'x' or 'y' — no silent zero-fill."""
    if not hasattr(data, "node_types"):
        return
    for node_type in data.node_types:
        store = data[node_type]
        if not hasattr(store, "x") or store.x is None:
            raise RuntimeError(
                f"Node type '{node_type}' is missing feature tensor 'x'. "
                "All node types must have predetermined features for "
                "homogeneous conversion. Refusing to auto-create zeros."
            )
        if not hasattr(store, "y") or store.y is None:
            raise RuntimeError(
                f"Node type '{node_type}' is missing target tensor 'y'. "
                "All node types must have predetermined targets for "
                "homogeneous conversion. Refusing to auto-create zeros."
            )


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


def _raw_json_to_heterodata(filepath):
    """Load a single raw OPF JSON and build a HeteroData (same as OPFDataset.process)."""
    from torch_geometric.data import HeteroData

    with open(filepath) as f:
        obj = json.load(f)

    grid = obj["grid"]
    solution = obj["solution"]
    metadata = obj["metadata"]

    data = HeteroData()
    data.x = torch.tensor(grid["context"]).view(-1)
    data.objective = torch.tensor(metadata["objective"])

    data["bus"].x = torch.tensor(grid["nodes"]["bus"])
    data["bus"].y = torch.tensor(solution["nodes"]["bus"])
    data["generator"].x = torch.tensor(grid["nodes"]["generator"])
    data["generator"].y = torch.tensor(solution["nodes"]["generator"])
    data["load"].x = torch.tensor(grid["nodes"]["load"])
    data["shunt"].x = torch.tensor(grid["nodes"]["shunt"])

    data["bus", "ac_line", "bus"].edge_index = tg_opf.extract_edge_index(obj, "ac_line")
    data["bus", "ac_line", "bus"].edge_attr = torch.tensor(
        grid["edges"]["ac_line"]["features"]
    )
    data["bus", "ac_line", "bus"].edge_label = torch.tensor(
        solution["edges"]["ac_line"]["features"]
    )

    data["bus", "transformer", "bus"].edge_index = tg_opf.extract_edge_index(
        obj, "transformer"
    )
    data["bus", "transformer", "bus"].edge_attr = torch.tensor(
        grid["edges"]["transformer"]["features"]
    )
    data["bus", "transformer", "bus"].edge_label = torch.tensor(
        solution["edges"]["transformer"]["features"]
    )

    data["generator", "generator_link", "bus"].edge_index = tg_opf.extract_edge_index(
        obj, "generator_link"
    )
    data[
        "bus", "generator_link", "generator"
    ].edge_index = tg_opf.extract_edge_index_rev(obj, "generator_link")
    data["load", "load_link", "bus"].edge_index = tg_opf.extract_edge_index(
        obj, "load_link"
    )
    data["bus", "load_link", "load"].edge_index = tg_opf.extract_edge_index_rev(
        obj, "load_link"
    )
    data["shunt", "shunt_link", "bus"].edge_index = tg_opf.extract_edge_index(
        obj, "shunt_link"
    )
    data["bus", "shunt_link", "shunt"].edge_index = tg_opf.extract_edge_index_rev(
        obj, "shunt_link"
    )

    return data


def _iter_raw_split_for_rank(
    datadir,
    case_name,
    num_groups,
    topological_perturbations,
    split,
    rank,
    world_size,
):
    """Yield HeteroData samples from raw JSON files for *split*, only those
    assigned to *rank*.  Reads one file at a time — never holds more than
    one sample in memory.
    """
    release = "dataset_release_1"
    if topological_perturbations:
        release += "_nminusone"
    raw_dir = os.path.join(datadir, release, case_name, "raw")
    tmp_root = os.path.join(raw_dir, "gridopt-dataset-tmp", release, case_name)

    total_samples = 15_000 * num_groups
    train_limit = int(total_samples * 0.9)
    val_limit = train_limit + int(total_samples * 0.05)

    # Collect all (global_index, filepath) for the requested split
    split_files = []
    for group in range(num_groups):
        group_dir = os.path.join(tmp_root, f"group_{group}")
        for name in sorted(os.listdir(group_dir)):
            i = int(name.split(".")[0].split("_")[1])
            if split == "train" and i < train_limit:
                split_files.append((i, os.path.join(group_dir, name)))
            elif split == "val" and train_limit <= i < val_limit:
                split_files.append((i, os.path.join(group_dir, name)))
            elif split == "test" and i >= val_limit:
                split_files.append((i, os.path.join(group_dir, name)))

    # Sort by index for deterministic ordering
    split_files.sort(key=lambda x: x[0])

    # Select only this rank's share
    n = len(split_files)
    chunk = n // world_size
    remainder = n % world_size
    start = rank * chunk + min(rank, remainder)
    end = start + chunk + (1 if rank < remainder else 0)

    skipped = 0
    for _, filepath in split_files[start:end]:
        try:
            yield _raw_json_to_heterodata(filepath)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            skipped += 1
            logging.warning("Skipping corrupt file %s: %s", filepath, exc)
    if skipped:
        logging.warning("Rank skipped %d corrupt JSON file(s) in this split", skipped)


def _prepare_sample(
    data,
    node_target_type: str,
    case_name: str,
    to_homogeneous: bool = False,
    edge_dim=None,
    edge_feature_schema=None,
):
    data.y = _build_solution_target(data, node_target_type)
    _ensure_node_y_loc(data)
    data.graph_attr = data.x.view(1, -1).to(torch.float32)
    data.case_name = case_name
    _ensure_non_scalar_attrs(data)
    if hasattr(data, "num_nodes_dict"):
        delattr(data, "num_nodes_dict")
    data, _ = assemble_edge_attr(
        data, edge_dim=edge_dim, feature_schema=edge_feature_schema
    )
    if not to_homogeneous:
        return data
    _validate_node_stores_for_homogeneous(data)
    data_h = data.to_homogeneous(
        node_attrs=["x", "y"],
        edge_attrs=["edge_attr"],
        add_node_type=True,
        add_edge_type=True,
    )
    data_h.graph_attr = data.graph_attr
    data_h.case_name = case_name
    _ensure_non_scalar_attrs(data_h)
    if hasattr(data_h, "num_nodes_dict"):
        delattr(data_h, "num_nodes_dict")
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
    case_names,
    case_num_groups,
    topological_perturbations,
    rank,
    comm,
):
    if rank == 0:
        _diag("Priming OPF processed split caches on rank 0")
        for case_name in case_names:
            num_groups = case_num_groups[case_name]
            for split in ("train", "val", "test"):
                _load_split(
                    root,
                    split,
                    case_name,
                    num_groups,
                    topological_perturbations,
                )
        _diag("Finished OPF processed split cache priming on rank 0")
    comm.Barrier()


def _parse_case_list(case_name_args):
    if not case_name_args:
        return []
    if isinstance(case_name_args, str):
        return [case_name_args]
    return [c.strip() for c in case_name_args if c.strip()]


def _resolve_preonly_case_names(args, datadir, default_case_names):
    requested = _parse_case_list(args.preonly_case_names)
    if not requested:
        return default_case_names
    if len(requested) == 1 and requested[0].lower() == "all":
        discovered = data_ops.discover_cases(datadir, args.topological_perturbations)
        if discovered:
            return discovered
        return default_case_names
    return requested


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


def _log_phase_time(comm, rank, label: str, elapsed_local: float):
    elapsed_max = comm.allreduce(float(elapsed_local), op=MPI.MAX)
    elapsed_sum = comm.allreduce(float(elapsed_local), op=MPI.SUM)
    elapsed_avg = elapsed_sum / max(1, comm.Get_size())
    if rank == 0:
        info(f"Timing {label}: max={elapsed_max:.2f}s avg={elapsed_avg:.2f}s")


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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit total number of samples across train/val/test splits",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--modelname", type=str, default="OPF_Solution_Hetero")
    parser.add_argument("--mpnn_type", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_conv_layers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--log", type=str, default=None)
    domain_group = parser.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--enable_domain_loss",
        action="store_true",
        help="Enable OPF domain-informed auxiliary loss regardless of config default.",
    )
    domain_group.add_argument(
        "--disable_domain_loss",
        action="store_true",
        help="Disable OPF domain-informed auxiliary loss regardless of config default.",
    )
    parser.add_argument(
        "--domain_loss_smoothness_weight",
        type=float,
        default=None,
        help="Override DomainLoss.smoothness_weight.",
    )
    parser.add_argument(
        "--domain_loss_transformer_smoothness_weight",
        type=float,
        default=None,
        help="Override DomainLoss.transformer_smoothness_weight.",
    )
    parser.add_argument(
        "--domain_loss_voltage_bound_weight",
        type=float,
        default=None,
        help="Override DomainLoss.voltage_bound_weight.",
    )
    parser.add_argument(
        "--domain_loss_voltage_bound_feature_indices",
        nargs=2,
        type=int,
        default=None,
        metavar=("VMIN_IDX", "VMAX_IDX"),
        help="Override DomainLoss.voltage_bound_feature_indices.",
    )
    parser.add_argument(
        "--domain_loss_voltage_output_index",
        type=int,
        default=None,
        help="Override DomainLoss.voltage_output_index (index of Vm in bus_pred; default 1).",
    )
    parser.add_argument(
        "--domain_loss_va_output_index",
        type=int,
        default=None,
        help="Override DomainLoss.va_output_index (index of Va in bus_pred; default 0).",
    )
    parser.add_argument(
        "--domain_loss_angle_diff_weight",
        type=float,
        default=None,
        help="Override DomainLoss.angle_diff_weight (angle-difference-limit penalty).",
    )
    parser.add_argument(
        "--domain_loss_line_flow_weight",
        type=float,
        default=None,
        help="Override DomainLoss.line_flow_weight (DC thermal-limit penalty).",
    )
    parser.add_argument(
        "--domain_loss_ema_momentum",
        type=float,
        default=None,
        help="Override DomainLoss.ema_momentum for per-term EMA normalization (default 0.1).",
    )
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
    group.add_argument("--hdf5", action="store_const", dest="format", const="hdf5")
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, args.data_root)
    input_filename = os.path.join(dirpwd, args.inputfile)

    with open(input_filename, "r") as f:
        config = json.load(f)

    arch_config = config.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})

    # CLI overrides for HPO
    for param in ("mpnn_type", "hidden_dim", "num_conv_layers"):
        val = getattr(args, param, None)
        if val is not None:
            arch_config[param] = val
    if args.learning_rate is not None:
        config["NeuralNetwork"]["Training"]["Optimizer"][
            "learning_rate"
        ] = args.learning_rate
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

    training_config = config.setdefault("NeuralNetwork", {}).setdefault("Training", {})

    # Apply CLI overrides for domain loss.  Any CLI flag takes precedence over
    # whatever is stored in the input config.
    _domain_cli_overrides = {
        "enabled": True if args.enable_domain_loss else (False if args.disable_domain_loss else None),
        "smoothness_weight": args.domain_loss_smoothness_weight,
        "transformer_smoothness_weight": args.domain_loss_transformer_smoothness_weight,
        "voltage_bound_weight": args.domain_loss_voltage_bound_weight,
        "voltage_bound_feature_indices": (
            list(args.domain_loss_voltage_bound_feature_indices)
            if args.domain_loss_voltage_bound_feature_indices is not None
            else None
        ),
        "voltage_output_index": args.domain_loss_voltage_output_index,
        "va_output_index": args.domain_loss_va_output_index,
        "angle_diff_weight": args.domain_loss_angle_diff_weight,
        "line_flow_weight": args.domain_loss_line_flow_weight,
        "ema_momentum": args.domain_loss_ema_momentum,
    }
    if any(v is not None for v in _domain_cli_overrides.values()):
        domain_loss_config = copy.deepcopy(training_config.get("DomainLoss", {}))
        for key, val in _domain_cli_overrides.items():
            if val is not None:
                domain_loss_config[key] = val
        training_config["DomainLoss"] = domain_loss_config

    raw_edge_dim = arch_config.get("edge_dim")
    if isinstance(raw_edge_dim, dict):
        # Heterogeneous route: per-edge-type widths from pre-assembled tensors.
        edge_dim = {str(k): int(v) for k, v in raw_edge_dim.items()}
        edge_feature_schema = None
    elif raw_edge_dim is not None:
        # Homogeneous route: uniform width, optional named-column schema.
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

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )

    log_name = args.log if args.log is not None else args.modelname
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    requested_num_groups = data_ops.parse_num_groups(args.num_groups)
    serialized_training_only = args.format in ("adios", "hdf5") and not args.preonly
    if serialized_training_only:
        case_names = []
        if rank == 0:
            info(
                f"{args.format.upper()} training mode: skipping OPF case discovery/download/split preparation."
            )
    else:
        parsed_case_names = _parse_case_list(args.case_name)
        if len(parsed_case_names) == 1 and parsed_case_names[0].lower() == "all":
            case_names = data_ops.discover_cases(
                datadir, args.topological_perturbations
            )
            if not case_names:
                case_names = list(_DEFAULT_CASE_NAMES)
            if not case_names:
                raise RuntimeError("No OPF cases found.")
        else:
            case_names = parsed_case_names

        if args.preonly:
            case_names = _resolve_preonly_case_names(args, datadir, case_names)

    case_num_groups = {}
    shared_datadir = datadir
    active_datadir = datadir
    _fmt_ext = {"adios": ".bp", "hdf5": ".h5", "pickle": ".pickle"}
    serialized_target = f"{args.modelname}{_fmt_ext[args.format]}"
    preonly_pipeline = args.preonly and args.max_samples is None
    if preonly_pipeline:
        store_homogeneous = args.format == "adios"
        verbosity = config["Verbosity"]["level"]
        # For HDF5 streaming mode we write samples as they are processed,
        # so no large lists are needed.  For other formats we still
        # accumulate.
        hdf5_streaming = args.format == "hdf5"
        if hdf5_streaming:
            serialize_datadir = shared_datadir
            basedir = os.path.join(serialize_datadir, f"{args.modelname}.h5")
            if rank == 0 and os.path.exists(basedir):
                shutil.rmtree(basedir, ignore_errors=True)
            comm.Barrier()
            h5writer = HDF5Writer(basedir, comm)
            trainset_count = 0
            valset_count = 0
            testset_count = 0
        else:
            trainset = []
            valset = []
            testset = []

    # Task-parallel: partition cases across rank groups so each group
    # processes a subset of cases concurrently (embarrassingly parallel).
    _task_parallel = preonly_pipeline and len(case_names) > 1 and comm_size > 1
    _task_cases = case_names
    _saved_comm, _saved_rank, _saved_size = comm, rank, comm_size
    _case_sub_comm = None
    if _task_parallel:
        _n_cases = len(case_names)
        _n_groups = min(_n_cases, comm_size)
        _group_id = min(rank * _n_groups // comm_size, _n_groups - 1)
        _case_sub_comm = comm.Split(_group_id, rank)
        _task_cases = [
            case_names[i]
            for i in range(_n_cases)
            if min(i * _n_groups // _n_cases, _n_groups - 1) == _group_id
        ]
        if rank == 0:
            info(
                f"Task-parallel preprocessing: {_n_cases} cases across "
                f"{_n_groups} groups of {comm_size} total ranks"
            )
        # Shadow comm/rank/comm_size so the loop body uses the sub-communicator
        # for barriers, broadcasts, and work splitting within each case.
        comm = _case_sub_comm
        rank = _case_sub_comm.Get_rank()
        comm_size = _case_sub_comm.Get_size()

    for case_name in _task_cases:
        t_case = time.perf_counter()
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

        t_download = time.perf_counter()
        data_ops.ensure_opf_downloaded(
            shared_datadir,
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

        if args.nvme:
            t_stage = time.perf_counter()
            staged_datadir = stage_case_to_nvme(
                shared_datadir,
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
            if staged_datadir != shared_datadir:
                active_datadir = staged_datadir
            else:
                active_datadir = shared_datadir
                break
        _log_phase_time(
            comm,
            rank,
            f"case={case_name} phase=prepare_total groups={num_groups}",
            time.perf_counter() - t_case,
        )

        if preonly_pipeline:
            for split_name, label in [
                ("train", "trainset"),
                ("val", "valset"),
                ("test", "testset"),
            ]:
                t_pre = time.perf_counter()
                sample_iter = _iter_raw_split_for_rank(
                    active_datadir,
                    case_name,
                    num_groups,
                    args.topological_perturbations,
                    split_name,
                    rank,
                    comm_size,
                )
                local_count = 0
                if hdf5_streaming:
                    h5writer.begin(label)
                    for d in iterate_tqdm(
                        sample_iter,
                        verbosity,
                        desc=f"Preprocess {split_name} {case_name}",
                        leave=False,
                    ):
                        h5writer.put(
                            _prepare_sample(
                                d,
                                args.node_target_type,
                                case_name,
                                store_homogeneous,
                                edge_dim=edge_dim,
                                edge_feature_schema=edge_feature_schema,
                            )
                        )
                        local_count += 1
                    h5writer.end_label()
                    if label == "trainset":
                        trainset_count += local_count
                    elif label == "valset":
                        valset_count += local_count
                    else:
                        testset_count += local_count
                else:
                    target_list = (
                        trainset
                        if label == "trainset"
                        else (valset if label == "valset" else testset)
                    )
                    for d in iterate_tqdm(
                        sample_iter,
                        verbosity,
                        desc=f"Preprocess {split_name} {case_name}",
                        leave=False,
                    ):
                        target_list.append(
                            _prepare_sample(
                                d,
                                args.node_target_type,
                                case_name,
                                store_homogeneous,
                                edge_dim=edge_dim,
                                edge_feature_schema=edge_feature_schema,
                            )
                        )
                        local_count += 1
                _log_phase_time(
                    comm,
                    rank,
                    f"case={case_name} split={split_name} phase=preprocess local_samples={local_count}",
                    time.perf_counter() - t_pre,
                )

    datadir = active_datadir

    # Restore original communicator after task-parallel loop.
    comm, rank, comm_size = _saved_comm, _saved_rank, _saved_size
    if _case_sub_comm is not None:
        _case_sub_comm.Free()

    if preonly_pipeline:
        if hdf5_streaming:
            info(
                f"Local split sizes: train={trainset_count}, val={valset_count}, test={testset_count}"
            )
        else:
            info(
                f"Local split sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
            )

        t_write = time.perf_counter()
        if hdf5_streaming:
            # Samples already written; just finalize metadata.
            h5writer.save()
        elif args.format == "adios":
            if AdiosWriter is None:
                raise RuntimeError("adios2 is not available in this environment.")
            serialize_datadir = shared_datadir
            fname = os.path.join(serialize_datadir, f"{args.modelname}.bp")
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
        elif args.format == "hdf5":
            serialize_datadir = shared_datadir
            basedir = os.path.join(serialize_datadir, f"{args.modelname}.h5")
            if rank == 0 and os.path.exists(basedir):
                shutil.rmtree(basedir, ignore_errors=True)
            comm.Barrier()
            h5writer = HDF5Writer(basedir, comm)
            h5writer.add("trainset", trainset)
            h5writer.add("valset", valset)
            h5writer.add("testset", testset)
            h5writer.save()
        else:
            basedir = os.path.join(datadir, f"{args.modelname}.pickle")
            SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
            SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
            SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
        _log_phase_time(
            comm,
            rank,
            f"phase=serialize format={args.format} model={args.modelname}",
            time.perf_counter() - t_write,
        )

        comm.Barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)

    if case_names:
        _prime_processed_splits_on_rank0(
            datadir,
            case_names,
            case_num_groups,
            args.topological_perturbations,
            rank,
            comm,
        )

    if rank == 0 and case_names:
        info("Loading OPF splits...")
    train_raw = []
    val_raw = []
    test_raw = []
    for case_name in case_names:
        num_groups = case_num_groups[case_name]

        t_load = time.perf_counter()
        train_raw.append(
            _load_split(
                datadir,
                "train",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )
        _log_phase_time(
            comm,
            rank,
            f"case={case_name} split=train phase=load groups={num_groups}",
            time.perf_counter() - t_load,
        )

        t_load = time.perf_counter()
        val_raw.append(
            _load_split(
                datadir,
                "val",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )
        _log_phase_time(
            comm,
            rank,
            f"case={case_name} split=val phase=load groups={num_groups}",
            time.perf_counter() - t_load,
        )

        t_load = time.perf_counter()
        test_raw.append(
            _load_split(
                datadir,
                "test",
                case_name,
                num_groups,
                args.topological_perturbations,
            )
        )
        _log_phase_time(
            comm,
            rank,
            f"case={case_name} split=test phase=load groups={num_groups}",
            time.perf_counter() - t_load,
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
            t_pre = time.perf_counter()
            case_cap = remaining_caps["train"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(train_split)))
            subset = _subset_for_rank(train_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess train {case_name}", leave=False
            ):
                trainset.append(
                    _prepare_sample(
                        d,
                        args.node_target_type,
                        case_name,
                        store_homogeneous,
                        edge_dim=edge_dim,
                        edge_feature_schema=edge_feature_schema,
                    )
                )
            if remaining_caps["train"] is not None:
                remaining_caps["train"] = max(
                    0, remaining_caps["train"] - min(len(train_split), case_cap or 0)
                )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=train phase=preprocess local_samples={len(subset)}",
                time.perf_counter() - t_pre,
            )
        for case_name, val_split in zip(case_names, val_raw):
            t_pre = time.perf_counter()
            case_cap = remaining_caps["val"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(val_split)))
            subset = _subset_for_rank(val_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess val {case_name}", leave=False
            ):
                valset.append(
                    _prepare_sample(
                        d,
                        args.node_target_type,
                        case_name,
                        store_homogeneous,
                        edge_dim=edge_dim,
                        edge_feature_schema=edge_feature_schema,
                    )
                )
            if remaining_caps["val"] is not None:
                remaining_caps["val"] = max(
                    0, remaining_caps["val"] - min(len(val_split), case_cap or 0)
                )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=val phase=preprocess local_samples={len(subset)}",
                time.perf_counter() - t_pre,
            )
        for case_name, test_split in zip(case_names, test_raw):
            t_pre = time.perf_counter()
            case_cap = remaining_caps["test"]
            if case_cap is not None:
                case_cap = max(0, min(case_cap, len(test_split)))
            subset = _subset_for_rank(test_split, rank, comm_size, case_cap)
            for d in iterate_tqdm(
                subset, verbosity, desc=f"Preprocess test {case_name}", leave=False
            ):
                testset.append(
                    _prepare_sample(
                        d,
                        args.node_target_type,
                        case_name,
                        store_homogeneous,
                        edge_dim=edge_dim,
                        edge_feature_schema=edge_feature_schema,
                    )
                )
            if remaining_caps["test"] is not None:
                remaining_caps["test"] = max(
                    0, remaining_caps["test"] - min(len(test_split), case_cap or 0)
                )
            _log_phase_time(
                comm,
                rank,
                f"case={case_name} split=test phase=preprocess local_samples={len(subset)}",
                time.perf_counter() - t_pre,
            )

        info(
            f"Local split sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}"
        )

        t_write = time.perf_counter()
        if args.format == "adios":
            if AdiosWriter is None:
                raise RuntimeError("adios2 is not available in this environment.")
            serialize_datadir = shared_datadir
            fname = os.path.join(serialize_datadir, f"{args.modelname}.bp")
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
        elif args.format == "hdf5":
            serialize_datadir = shared_datadir
            basedir = os.path.join(serialize_datadir, f"{args.modelname}.h5")
            if rank == 0 and os.path.exists(basedir):
                shutil.rmtree(basedir, ignore_errors=True)
            comm.Barrier()
            h5writer = HDF5Writer(basedir, comm)
            h5writer.add("trainset", trainset)
            h5writer.add("valset", valset)
            h5writer.add("testset", testset)
            h5writer.save()
        else:
            basedir = os.path.join(datadir, f"{args.modelname}.pickle")
            SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
            SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
            SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
        _log_phase_time(
            comm,
            rank,
            f"phase=serialize format={args.format} model={args.modelname}",
            time.perf_counter() - t_write,
        )

        comm.Barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)

    if args.format == "adios":
        if AdiosDataset is None:
            raise RuntimeError("adios2 is not available in this environment.")
        fname = os.path.join(datadir, f"{args.modelname}.bp")
        if serialized_training_only and not os.path.isdir(fname):
            raise RuntimeError(
                f"Expected preprocessed ADIOS dataset at '{fname}' for training-only mode. "
                "Run with --preonly --adios first."
            )
        train_base = AdiosDataset(fname, "trainset", comm, var_config=None)
        val_base = AdiosDataset(fname, "valset", comm, var_config=None)
        test_base = AdiosDataset(fname, "testset", comm, var_config=None)
        trainset = HeteroFromHomogeneousDataset(train_base, edge_dim=edge_dim)
        valset = HeteroFromHomogeneousDataset(val_base, edge_dim=edge_dim)
        testset = HeteroFromHomogeneousDataset(test_base, edge_dim=edge_dim)
    elif args.format == "hdf5":
        basedir = os.path.join(datadir, f"{args.modelname}.h5")
        if serialized_training_only and not os.path.isdir(basedir):
            raise RuntimeError(
                f"Expected preprocessed HDF5 dataset at '{basedir}' for training-only mode. "
                "Run with --preonly --hdf5 first."
            )
        trainset = HDF5Dataset(basedir, "trainset")
        valset = HDF5Dataset(basedir, "valset")
        testset = HDF5Dataset(basedir, "testset")
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
    validate_voi_node_features(config, args.node_target_type)

    arch_config = config.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})

    trainset = EdgeAttrDatasetAdapter(trainset, edge_dim=edge_dim)
    valset = EdgeAttrDatasetAdapter(valset, edge_dim=edge_dim)
    testset = EdgeAttrDatasetAdapter(testset, edge_dim=edge_dim)

    trainset = NodeTargetDatasetAdapter(
        trainset, args.node_target_type, edge_dim=edge_dim
    )
    valset = NodeTargetDatasetAdapter(valset, args.node_target_type, edge_dim=edge_dim)
    testset = NodeTargetDatasetAdapter(
        testset, args.node_target_type, edge_dim=edge_dim
    )

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
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
        info("Computing pna_deg for HeteroPNA from training dataset")
        pna_deg = compute_pna_deg_for_hetero_dataset(trainset, verbosity=2)
        arch_config["pna_deg"] = pna_deg
        arch_config["max_neighbours"] = max(0, len(pna_deg) - 1)

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

    domain_loss_config = config["NeuralNetwork"]["Training"].get("DomainLoss")
    if domain_loss_config is not None:
        dl_enabled = domain_loss_config.get("enabled", False)
        if rank == 0:
            info(
                f"[DomainLoss] config (enabled={dl_enabled}): "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in domain_loss_config.items()
                    if k != "enabled"
                )
            )
        if dl_enabled and rank == 0:
            info("[DomainLoss] Wrapping model with OPFEnhancedModelWrapper.")
        model = OPFEnhancedModelWrapper(
            model,
            OPFDomainLoss(
                domain_loss_config,
                node_target_type=args.node_target_type,
            ),
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
