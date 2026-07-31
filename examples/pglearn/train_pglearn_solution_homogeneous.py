import argparse
import copy
import gzip
import json
import logging
import os
import shutil

import h5py
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch_geometric.data import Data

import hydragnn
from __init__ import data_ops
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset, SimplePickleWriter
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model import print_model
from hydragnn.utils.profiling_and_tracing import print_timers
from hydragnn.utils.print import iterate_tqdm

from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset, HDF5Writer

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, AdiosWriter
except ImportError:
    AdiosDataset = None
    AdiosWriter = None


TASK_PF = "pf"
TASK_OPF = "opf"
TASK_TOKEN = {TASK_PF: 0.0, TASK_OPF: 1.0}
PF_PG_KEY_CANDIDATES = ["pg", "pg_setpoint", "pg_spec", "pgen", "gen_pg"]
PF_VG_KEY_CANDIDATES = ["vg", "vg_setpoint", "vm_setpoint", "vset", "gen_vm"]


def _normalize_task(task_name):
    task = str(task_name).strip().lower()
    if task not in (TASK_PF, TASK_OPF):
        raise ValueError(f"Unsupported task '{task_name}'. Use '{TASK_PF}' or '{TASK_OPF}'.")
    return task


def _infer_task_from_formulation(formulation):
    f = str(formulation).strip().upper()
    if "OPF" in f:
        return TASK_OPF
    if f.endswith("PF"):
        return TASK_PF
    raise ValueError(
        f"Cannot infer task from formulation '{formulation}'. "
        "Pass --task explicitly as 'pf' or 'opf'."
    )


def _validate_task_and_formulation(task, formulation):
    task = _normalize_task(task)
    f = str(formulation).strip().upper()
    if task == TASK_OPF and "OPF" not in f:
        raise ValueError(
            f"Task '{TASK_OPF}' requires an OPF formulation; got '{formulation}'."
        )
    if task == TASK_PF and f.endswith("OPF"):
        raise ValueError(
            f"Task '{TASK_PF}' requires a PF formulation; got '{formulation}'."
        )


def _resolve_present_key(h5_file, candidates):
    for key in candidates:
        if key in h5_file:
            return key
    return None


def _assert_shape_second_dim(h5_file, key, expected, name):
    shape = tuple(h5_file[key].shape)
    if len(shape) != 2:
        raise RuntimeError(f"{name} key '{key}' must be rank-2, got shape {shape}.")
    if int(shape[1]) != int(expected):
        raise RuntimeError(
            f"{name} key '{key}' has width {shape[1]}, expected {expected}."
        )


def _validate_and_resolve_schema(task, input_h5, primal_h5, n_gen):
    required_input = ["pd", "qd", "gen_status", "branch_status", "reserve_requirement"]
    missing_input = [k for k in required_input if k not in input_h5]
    if missing_input:
        raise RuntimeError(f"Missing required input keys: {missing_input}")

    required_output = ["va", "vm"]
    missing_output = [k for k in required_output if k not in primal_h5]
    if missing_output:
        raise RuntimeError(f"Missing required output keys: {missing_output}")

    _assert_shape_second_dim(input_h5, "pd", input_h5["pd"].shape[1], "pd")
    _assert_shape_second_dim(input_h5, "qd", input_h5["qd"].shape[1], "qd")
    _assert_shape_second_dim(input_h5, "gen_status", n_gen, "gen_status")

    resolved = {"pg_setpoint_key": None, "vg_setpoint_key": None}
    if task == TASK_PF:
        pg_key = _resolve_present_key(input_h5, PF_PG_KEY_CANDIDATES)
        vg_key = _resolve_present_key(input_h5, PF_VG_KEY_CANDIDATES)
        if pg_key is None:
            raise RuntimeError(
                "PF task requires generator active-power setpoints in input data. "
                f"Tried keys: {PF_PG_KEY_CANDIDATES}"
            )
        if vg_key is None:
            raise RuntimeError(
                "PF task requires generator voltage setpoints in input data. "
                f"Tried keys: {PF_VG_KEY_CANDIDATES}"
            )
        _assert_shape_second_dim(input_h5, pg_key, n_gen, "pf pg setpoints")
        _assert_shape_second_dim(input_h5, vg_key, n_gen, "pf vg setpoints")
        resolved["pg_setpoint_key"] = pg_key
        resolved["vg_setpoint_key"] = vg_key

    return resolved


def _check_task_token(sample, expected_task):
    graph_attr = getattr(sample, "graph_attr", None)
    if not isinstance(graph_attr, torch.Tensor) or graph_attr.numel() < 2:
        raise RuntimeError(
            "Serialized samples do not carry a PF/OPF task token. "
            "Re-run preprocessing with --preonly to regenerate dataset."
        )
    got = float(graph_attr.reshape(-1)[1].item())
    expected = TASK_TOKEN[expected_task]
    if abs(got - expected) > 1e-6:
        raise RuntimeError(
            f"Serialized dataset task token {got} does not match requested task '{expected_task}'."
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


def _subset_for_rank(indices, rank, world_size):
    return indices[rank::world_size]


def _build_case_static(case_data):
    n_bus = int(case_data["N"])

    bus_fr = np.asarray(case_data["bus_fr"], dtype=np.int64) - 1
    bus_to = np.asarray(case_data["bus_to"], dtype=np.int64) - 1

    edge_src = np.concatenate([bus_fr, bus_to])
    edge_dst = np.concatenate([bus_to, bus_fr])
    edge_index = torch.tensor(np.stack([edge_src, edge_dst], axis=0), dtype=torch.long)

    # Static branch channels.
    branch_keys = ["gff", "gft", "gtf", "gtt", "bff", "bft", "btf", "btt", "smax"]
    branch_static = []
    num_edges = len(bus_fr)
    for key in branch_keys:
        vals = case_data.get(key, [0.0] * num_edges)
        vals = np.asarray(vals, dtype=np.float32).reshape(-1)
        if vals.shape[0] != num_edges:
            vals = np.resize(vals, num_edges)
        branch_static.append(vals)
    branch_static = np.stack(branch_static, axis=1)
    branch_static = np.concatenate([branch_static, branch_static], axis=0)

    bus_loads = []
    for item in case_data.get("bus_loads", [[] for _ in range(n_bus)]):
        bus_loads.append([int(x) - 1 for x in item])

    bus_gens = []
    for item in case_data.get("bus_gens", [[] for _ in range(n_bus)]):
        bus_gens.append([int(x) - 1 for x in item])

    pgmax = np.asarray(case_data.get("pgmax", []), dtype=np.float32)
    pgmin = np.asarray(case_data.get("pgmin", []), dtype=np.float32)
    vmin = np.asarray(case_data.get("vmin", [0.9] * n_bus), dtype=np.float32)
    vmax = np.asarray(case_data.get("vmax", [1.1] * n_bus), dtype=np.float32)

    ref_bus = int(case_data.get("ref_bus", 1)) - 1
    ref_vm = float(case_data.get("ref_vm", case_data.get("vm_ref", 1.0)))
    ref_va = float(case_data.get("ref_va", case_data.get("va_ref", 0.0)))

    return {
        "n_bus": n_bus,
        "edge_index": edge_index,
        "branch_static": branch_static,
        "bus_loads": bus_loads,
        "bus_gens": bus_gens,
        "pgmax": pgmax,
        "pgmin": pgmin,
        "vmin": vmin,
        "vmax": vmax,
        "ref_bus": ref_bus,
        "ref_vm": ref_vm,
        "ref_va": ref_va,
    }


def _aggregate_to_bus(values, bus_map, n_bus):
    out = np.zeros((n_bus,), dtype=np.float32)
    for i in range(n_bus):
        idx = bus_map[i]
        if not idx:
            continue
        out[i] = float(np.sum(values[idx]))
    return out


def _build_sample(static, input_h5, primal_h5, idx, task, schema_keys):
    n_bus = static["n_bus"]

    pd = np.asarray(input_h5["pd"][idx], dtype=np.float32)
    qd = np.asarray(input_h5["qd"][idx], dtype=np.float32)
    gen_status = np.asarray(input_h5["gen_status"][idx], dtype=np.float32)
    branch_status = np.asarray(input_h5["branch_status"][idx], dtype=np.float32)
    reserve_req = float(input_h5["reserve_requirement"][idx])

    pd_bus = _aggregate_to_bus(pd, static["bus_loads"], n_bus)
    qd_bus = _aggregate_to_bus(qd, static["bus_loads"], n_bus)
    gen_on_bus = _aggregate_to_bus(gen_status, static["bus_gens"], n_bus)
    pgmax_bus = _aggregate_to_bus(static["pgmax"], static["bus_gens"], n_bus)
    pgmin_bus = _aggregate_to_bus(static["pgmin"], static["bus_gens"], n_bus)

    is_ref = np.zeros((n_bus,), dtype=np.float32)
    if 0 <= static["ref_bus"] < n_bus:
        is_ref[static["ref_bus"]] = 1.0

    reserve = np.full((n_bus,), reserve_req, dtype=np.float32)
    task_token = np.full((n_bus,), TASK_TOKEN[task], dtype=np.float32)

    if task == TASK_OPF:
        x = np.stack(
            [
                pd_bus,
                qd_bus,
                gen_on_bus,
                pgmax_bus,
                pgmin_bus,
                is_ref,
                static["vmin"],
                static["vmax"],
                reserve,
                task_token,
            ],
            axis=1,
        )
    else:
        pg = np.asarray(input_h5[schema_keys["pg_setpoint_key"]][idx], dtype=np.float32)
        vg = np.asarray(input_h5[schema_keys["vg_setpoint_key"]][idx], dtype=np.float32)
        pg_bus = _aggregate_to_bus(pg, static["bus_gens"], n_bus)
        vg_bus = _aggregate_to_bus(vg, static["bus_gens"], n_bus)
        ref_vm = np.zeros((n_bus,), dtype=np.float32)
        ref_va = np.zeros((n_bus,), dtype=np.float32)
        if 0 <= static["ref_bus"] < n_bus:
            ref_vm[static["ref_bus"]] = static["ref_vm"]
            ref_va[static["ref_bus"]] = static["ref_va"]
        x = np.stack(
            [
                pd_bus,
                qd_bus,
                gen_on_bus,
                pg_bus,
                vg_bus,
                is_ref,
                ref_vm,
                ref_va,
                reserve,
                task_token,
            ],
            axis=1,
        )

    va = np.asarray(primal_h5["va"][idx], dtype=np.float32).reshape(n_bus)
    vm = np.asarray(primal_h5["vm"][idx], dtype=np.float32).reshape(n_bus)
    y = np.stack([va, vm], axis=1)

    branch_status_full = np.concatenate([branch_status, branch_status], axis=0).reshape(
        -1, 1
    )
    edge_attr = np.concatenate([static["branch_static"], branch_status_full], axis=1)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=static["edge_index"],
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data.graph_attr = torch.tensor([reserve_req, TASK_TOKEN[task]], dtype=torch.float32)
    return data


def _open_case_data(case_json_gz):
    with gzip.open(case_json_gz, "rt") as fh:
        obj = json.load(fh)
    return obj["data"]


def _prepare_local_datasets(
    datadir,
    case_name,
    formulation,
    task,
    rank,
    world_size,
    max_samples,
):
    case_dir = os.path.join(datadir, case_name)
    case_data = _open_case_data(os.path.join(case_dir, "case.json.gz"))
    static = _build_case_static(case_data)

    train_input_path = os.path.join(case_dir, "train", "input.h5")
    train_primal_path = os.path.join(case_dir, "train", formulation, "primal.h5")
    test_input_path = os.path.join(case_dir, "test", "input.h5")
    test_primal_path = os.path.join(case_dir, "test", formulation, "primal.h5")

    if not os.path.isfile(train_input_path) or not os.path.isfile(train_primal_path):
        raise FileNotFoundError(
            "Missing uncompressed train HDF5 files. Run download_and_uncompress_data.py first."
        )
    if not os.path.isfile(test_input_path) or not os.path.isfile(test_primal_path):
        raise FileNotFoundError(
            "Missing uncompressed test HDF5 files. Run download_and_uncompress_data.py first."
        )

    schema_keys = None
    with h5py.File(train_input_path, "r") as tr_in, h5py.File(train_primal_path, "r") as tr_out:
        schema_keys = _validate_and_resolve_schema(task, tr_in, tr_out, len(static["pgmax"]))

    with h5py.File(train_input_path, "r") as tr_in, h5py.File(train_primal_path, "r") as tr_out:
        n_train_total = int(tr_in["pd"].shape[0])
        val_count = max(1, int(0.1 * n_train_total))
        train_count = n_train_total - val_count

    with h5py.File(test_input_path, "r") as te_in:
        n_test_total = int(te_in["pd"].shape[0])

    # Optional global cap while preserving split proportions.
    if max_samples is not None and max_samples > 0:
        total = train_count + val_count + n_test_total
        ratio = min(1.0, float(max_samples) / float(total))
        train_count = max(1, int(train_count * ratio))
        val_count = max(1, int(val_count * ratio))
        n_test_total = max(1, int(n_test_total * ratio))

    train_indices = list(range(train_count))
    val_indices = list(range(n_train_total - val_count, n_train_total))
    test_indices = list(range(n_test_total))

    local_train_idx = _subset_for_rank(train_indices, rank, world_size)
    local_val_idx = _subset_for_rank(val_indices, rank, world_size)
    local_test_idx = _subset_for_rank(test_indices, rank, world_size)

    local_train = []
    local_val = []
    local_test = []

    with h5py.File(train_input_path, "r") as tr_in, h5py.File(train_primal_path, "r") as tr_out:
        for idx in iterate_tqdm(local_train_idx, 2, desc="Preprocess train", leave=False):
            local_train.append(_build_sample(static, tr_in, tr_out, idx, task, schema_keys))
        for idx in iterate_tqdm(local_val_idx, 2, desc="Preprocess val", leave=False):
            local_val.append(_build_sample(static, tr_in, tr_out, idx, task, schema_keys))

    with h5py.File(test_input_path, "r") as te_in, h5py.File(test_primal_path, "r") as te_out:
        for idx in iterate_tqdm(local_test_idx, 2, desc="Preprocess test", leave=False):
            local_test.append(_build_sample(static, te_in, te_out, idx, task, schema_keys))

    return local_train, local_val, local_test


def _serialize_splits(args, datadir, trainset, valset, testset, comm, rank):
    if args.format == "adios":
        if AdiosWriter is None:
            raise RuntimeError("adios2 is not available in this environment.")
        out_path = os.path.join(datadir, f"{args.modelname}.bp")
        if rank == 0 and os.path.exists(out_path):
            if os.path.isdir(out_path):
                shutil.rmtree(out_path, ignore_errors=True)
            else:
                os.remove(out_path)
        comm.Barrier()
        writer = AdiosWriter(out_path, comm)
        writer.add("trainset", trainset)
        writer.add("valset", valset)
        writer.add("testset", testset)
        writer.save()
        return

    if args.format == "hdf5":
        out_path = os.path.join(datadir, f"{args.modelname}.h5")
        if rank == 0 and os.path.isdir(out_path):
            shutil.rmtree(out_path, ignore_errors=True)
        comm.Barrier()
        writer = HDF5Writer(out_path, comm)
        writer.add("trainset", trainset)
        writer.add("valset", valset)
        writer.add("testset", testset)
        writer.save()
        return

    out_path = os.path.join(datadir, f"{args.modelname}.pickle")
    if rank == 0 and os.path.isdir(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    comm.Barrier()
    SimplePickleWriter(trainset, out_path, "trainset", use_subdir=True)
    SimplePickleWriter(valset, out_path, "valset", use_subdir=True)
    SimplePickleWriter(testset, out_path, "testset", use_subdir=True)


def _load_serialized_splits(args, datadir, comm):
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
    parser.add_argument("--repo", type=str, default="PGLearn/PGLearn-Small")
    parser.add_argument("--case_name", type=str, default="14_ieee")
    parser.add_argument("--formulation", type=str, default="ACOPF")
    parser.add_argument(
        "--task",
        type=str,
        choices=["auto", TASK_PF, TASK_OPF],
        default="auto",
        help="PF/OPF task mode. 'auto' infers task from formulation.",
    )
    parser.add_argument("--preonly", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--hdf5", action="store_const", dest="format", const="hdf5")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    task = _infer_task_from_formulation(args.formulation) if args.task == "auto" else _normalize_task(args.task)
    _validate_task_and_formulation(task, args.formulation)

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
    config["Task"]["kind"] = task
    config["Task"]["formulation"] = args.formulation
    config["Task"]["task_token"] = TASK_TOKEN[task]

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

    data_ops.ensure_pglearn_downloaded(
        root=datadir,
        repo=args.repo,
        case_name=args.case_name,
        formulation=args.formulation,
        rank=rank,
        comm=comm,
    )

    # Preprocess + serialize when explicitly requested or when serialized data is absent.
    if args.format == "adios":
        serialized_exists = os.path.isdir(os.path.join(datadir, f"{args.modelname}.bp"))
    elif args.format == "hdf5":
        serialized_exists = os.path.isdir(os.path.join(datadir, f"{args.modelname}.h5"))
    else:
        serialized_exists = os.path.isdir(os.path.join(datadir, f"{args.modelname}.pickle"))

    if args.preonly or not serialized_exists:
        trainset, valset, testset = _prepare_local_datasets(
            datadir=datadir,
            case_name=args.case_name,
            formulation=args.formulation,
            task=task,
            rank=rank,
            world_size=comm_size,
            max_samples=args.max_samples,
        )

        _serialize_splits(args, datadir, trainset, valset, testset, comm, rank)
        if args.preonly:
            comm.Barrier()
            if dist.is_initialized():
                dist.destroy_process_group()
            return

    trainset, valset, testset = _load_serialized_splits(args, datadir, comm)

    # Synchronize config with actual tensor dimensions from preprocessed samples.
    sample0 = trainset[0]
    _check_task_token(sample0, task)
    x_dim = int(sample0.x.shape[1])
    edge_dim = int(sample0.edge_attr.shape[1])

    voi = config["NeuralNetwork"]["Variables_of_interest"]
    voi["input_node_features"] = list(range(x_dim))
    voi["node_feature_dims"] = [x_dim]
    config["NeuralNetwork"]["Architecture"]["edge_dim"] = edge_dim

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

    if config["NeuralNetwork"]["Architecture"].get("mpnn_type") == "PNA":
        deg = config["NeuralNetwork"]["Architecture"].get("pna_deg")
        if not deg:
            # Keep sane fallback if degree computation could not infer automatically.
            config["NeuralNetwork"]["Architecture"]["pna_deg"] = [0, 1, 1, 1]

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
        precision=config["NeuralNetwork"]["Training"].get("precision", "fp32"),
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
