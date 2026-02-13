#!/usr/bin/env python3
"""Train a composition-conditioned MLP to weight branch predictions.

This script loads a pretrained multi-branch HydraGNN model, computes per-branch
energy/force predictions, and trains a small MLP on data.chemical_composition
that outputs per-branch weights for a weighted average prediction.
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mpi4py import MPI
from torch.distributed.device_mesh import init_device_mesh

import hydragnn
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.models.create import create_model_config
from hydragnn.utils.distributed import get_device, nsplit
from hydragnn.train.train_validate_test import resolve_precision, move_batch_to_device

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, adios2_open
except ImportError:
    AdiosDataset = None
    adios2_open = None

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset


def _configure_variable_names(config):
    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims
    var_config["input_node_features"] = [0]


def _load_multidataset_dataloaders(args, config):
    if args.format == "pickle":
        raise NotImplementedError("Multi-dataset loading from pickle is not supported")
    if AdiosDataset is None:
        raise ImportError("AdiosDataset is unavailable; install adios2 to use --multi")

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if not args.multi_model_list or args.multi_model_list.strip() == "":
        raise ValueError("--multi_model_list must be provided for --multi")

    modellist = [m for m in args.multi_model_list.split(",") if m.strip()]
    if len(modellist) == 0:
        raise ValueError("--multi_model_list resulted in zero entries")

    if rank == 0:
        ndata_list = []
        pna_deg_list = []
        for model in modellist:
            fname = os.path.join(args.dataset_dir, f"{model}-v2.bp")
            with adios2_open(fname, "r", MPI.COMM_SELF) as f:
                f.__next__()
                ndata = f.read_attribute("trainset/ndata").item()
                attrs = f.available_attributes()
                pna_deg = None
                if "pna_deg" in attrs:
                    pna_deg = f.read_attribute("pna_deg")
                ndata_list.append(ndata)
                pna_deg_list.append(pna_deg)
        ndata_list = np.array(ndata_list, dtype=np.float32)
        process_list = np.ceil(ndata_list / sum(ndata_list) * comm_size).astype(
            np.int32
        )
        imax = np.argmax(process_list)
        process_list[imax] = process_list[imax] - (np.sum(process_list) - comm_size)
        process_list = process_list.tolist()

        if all(p is None for p in pna_deg_list):
            pna_deg = None
        else:
            intp_list = []
            mlen = min(
                [len(pna_deg) for pna_deg in pna_deg_list if pna_deg is not None]
            )
            for pna_deg in pna_deg_list:
                if pna_deg is None:
                    continue
                x = np.linspace(0, 1, num=len(pna_deg))
                intp = np.interp(np.linspace(0, 1, num=mlen), x, pna_deg)
                intp_list.append(intp)
            if len(intp_list) > 0:
                pna_deg = (
                    np.sum(np.stack(intp_list, axis=0), axis=0)
                    .astype(np.int64)
                    .tolist()
                )
            else:
                pna_deg = None
    else:
        process_list = None
        pna_deg = None

    process_list = comm.bcast(process_list, root=0)
    pna_deg = comm.bcast(pna_deg, root=0)

    if args.task_parallel and args.use_devicemesh:
        assert comm_size % len(modellist) == 0
        device = get_device()
        device_type = str(device).split(":")[0]
        mesh_2d = init_device_mesh(
            device_type,
            (len(modellist), comm_size // len(modellist)),
            mesh_dim_names=("dim1", "dim2"),
        )
        dim1_group = mesh_2d["dim1"].get_group()
        dim2_group = mesh_2d["dim2"].get_group()
        dim1_group_rank = dist.get_rank(group=dim1_group)
        mycolor = dim1_group_rank
        branch_group = dim2_group
    else:
        colorlist = []
        color = 0
        for n in process_list:
            for _ in range(n):
                colorlist.append(color)
            color += 1
        mycolor = colorlist[rank]
        branch_group = None

    local_comm = comm.Split(mycolor, rank)
    local_comm_rank = local_comm.Get_rank()
    local_comm_size = local_comm.Get_size()

    common_variable_names = [
        "pbc",
        "edge_attr",
        "energy_per_atom",
        "forces",
        "pos",
        "edge_index",
        "cell",
        "edge_shifts",
        "y",
        "chemical_composition",
        "natoms",
        "x",
        "energy",
        "graph_attr",
        "atomic_numbers",
    ]

    mymodel = modellist[mycolor]
    fname = os.path.join(args.dataset_dir, f"{mymodel}-v2.bp")
    trainset = AdiosDataset(fname, "trainset", local_comm, keys=common_variable_names)
    valset = AdiosDataset(fname, "valset", local_comm, keys=common_variable_names)
    testset = AdiosDataset(fname, "testset", local_comm, keys=common_variable_names)

    for ds in [trainset, valset, testset]:
        ds.dataset_name_dict = {
            name.lower(): torch.tensor([[i]]) for i, name in enumerate(modellist)
        }

    num_samples_list = []
    for i, dataset in enumerate([trainset, valset, testset]):
        rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
        if args.num_samples is not None:
            num_samples = args.num_samples if i == 0 else max(args.num_samples // 10, 1)
            rx = rx[: min(num_samples, len(rx))]

        local_dataset_len = len(rx)
        local_dataset_min = comm.allreduce(local_dataset_len, op=MPI.MIN)
        local_dataset_max = comm.allreduce(local_dataset_len, op=MPI.MAX)

        if args.task_parallel:
            rx = rx[:local_dataset_min]

        if args.oversampling:
            oversampling_num_samples = (
                args.oversampling_num_samples
                if args.oversampling_num_samples is not None
                else local_dataset_max
            )
            oversampling_num_samples = (
                oversampling_num_samples
                if i == 0
                else max(oversampling_num_samples // 10, 1)
            )
            num_samples_list.append(oversampling_num_samples)

        dataset.setkeys(common_variable_names)
        dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

    assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width, "local": True}
        if args.task_parallel:
            trainset = DistDataset(trainset, "trainset", local_comm, **opt)
            valset = DistDataset(valset, "valset", local_comm, **opt)
            testset = DistDataset(testset, "testset", local_comm, **opt)
        else:
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)

        if pna_deg is not None:
            trainset.pna_deg = pna_deg
            valset.pna_deg = pna_deg
            testset.pna_deg = pna_deg

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    if mymodel == "OC2020_all":
        config["NeuralNetwork"]["Training"]["batch_size"] = 40
    if mymodel == "OC2022":
        config["NeuralNetwork"]["Training"]["batch_size"] = 4

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        test_sampler_shuffle=False,
        group=branch_group if args.task_parallel else None,
        oversampling=args.oversampling,
        num_samples=num_samples_list if args.oversampling else None,
    )

    comm.Barrier()
    return train_loader, val_loader, test_loader


def _load_single_dataset_dataloaders(args, config, var_config):
    comm = MPI.COMM_WORLD
    if args.format == "adios":
        if AdiosDataset is None:
            raise ImportError(
                "AdiosDataset is unavailable; install adios2 to use --adios"
            )
        opt = {
            "preload": False,
            "shmem": args.shmem,
        }
        fname = os.path.join(args.dataset_dir, f"{args.modelname}.bp")
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        basedir = os.path.join(args.dataset_dir, f"{args.modelname}.pickle")
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = getattr(trainset, "pna_deg", None)
    else:
        raise NotImplementedError(f"No supported format: {args.format}")

    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)

        if "pna_deg" in locals() and pna_deg is not None:
            trainset.pna_deg = pna_deg
            valset.pna_deg = pna_deg
            testset.pna_deg = pna_deg

        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )
    return train_loader, val_loader, test_loader


class BranchWeightMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], num_branches: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_branches))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _reshape_composition(data) -> torch.Tensor:
    """Return composition as [num_graphs, comp_dim]."""
    comp = data.chemical_composition
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    if comp.dim() == 2:
        if comp.size(0) == data.num_graphs:
            return comp
        if comp.size(1) == data.num_graphs:
            return comp.t()
        if comp.size(1) == 1 and comp.size(0) % data.num_graphs == 0:
            return comp.view(data.num_graphs, -1)
    if comp.dim() == 3:
        if comp.size(0) == data.num_graphs:
            return comp.view(data.num_graphs, -1)
        if comp.size(1) == data.num_graphs:
            return comp.transpose(0, 1).contiguous().view(data.num_graphs, -1)
    raise ValueError(
        f"Unsupported chemical_composition shape {tuple(comp.shape)} for num_graphs={data.num_graphs}"
    )


def _build_dataset_name(data, branch_id: int) -> torch.Tensor:
    if hasattr(data, "dataset_name"):
        return torch.full_like(data.dataset_name, branch_id)
    return torch.full(
        (data.num_graphs, 1),
        branch_id,
        dtype=torch.long,
        device=data.x.device,
    )


def _energy_from_pred(pred) -> torch.Tensor:
    if isinstance(pred, (list, tuple)):
        energy = pred[0]
    elif isinstance(pred, dict) and "graph" in pred:
        energy = pred["graph"][0]
    else:
        energy = pred
    return energy.squeeze(-1)


def _predict_branch_energy_forces(
    model, data, branch_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)

    pred = model(data)
    energy_pred = _energy_from_pred(pred)
    forces_pred = torch.autograd.grad(
        energy_pred,
        data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        retain_graph=False,
        create_graph=False,
    )[0]
    forces_pred = -forces_pred

    if original_dataset_name is None:
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name

    return energy_pred.detach(), forces_pred.detach()


def _weighted_average(
    energy_preds: torch.Tensor,
    forces_preds: torch.Tensor,
    weights: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # energy_preds: [num_branches, num_graphs]
    # forces_preds: [num_branches, num_nodes, 3]
    # weights: [num_graphs, num_branches]
    weighted_energy = torch.sum(weights * energy_preds.transpose(0, 1), dim=1)

    node_counts = torch.bincount(batch)
    weighted_forces = torch.zeros_like(forces_preds[0])
    for branch_idx in range(energy_preds.size(0)):
        node_weights = torch.repeat_interleave(weights[:, branch_idx], node_counts)
        weighted_forces = (
            weighted_forces + node_weights.unsqueeze(-1) * forces_preds[branch_idx]
        )

    return weighted_energy, weighted_forces


def train_epoch(
    model,
    mlp,
    loader,
    optimizer,
    loss_fn,
    num_branches,
    energy_weight,
    force_weight,
    precision,
):
    model.eval()
    mlp.train()
    device = get_device()
    total_loss = 0.0
    total_samples = 0

    precision, param_dtype, _ = resolve_precision(precision)

    for data in loader:
        data = move_batch_to_device(data, param_dtype)
        if not hasattr(data, "chemical_composition"):
            raise ValueError(
                "data.chemical_composition is required for branch weighting"
            )

        data.pos.requires_grad_(True)
        comp = _reshape_composition(data).to(device)

        logits = mlp(comp)
        weights = F.softmax(logits, dim=-1)

        energy_preds = []
        forces_preds = []
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = _predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)

        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = _weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        energy_true = data.energy.squeeze().float()
        forces_true = data.forces.float()

        loss_energy = loss_fn(weighted_energy, energy_true)
        loss_forces = loss_fn(weighted_forces, forces_true)
        loss = energy_weight * loss_energy + force_weight * loss_forces

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    return total_loss / max(total_samples, 1)


def validate_epoch(
    model,
    mlp,
    loader,
    loss_fn,
    num_branches,
    energy_weight,
    force_weight,
    precision,
):
    model.eval()
    mlp.eval()
    device = get_device()
    total_loss = 0.0
    total_samples = 0

    precision, param_dtype, _ = resolve_precision(precision)

    for data in loader:
        data = move_batch_to_device(data, param_dtype)
        data.pos.requires_grad_(True)
        comp = _reshape_composition(data).to(device)

        logits = mlp(comp)
        weights = F.softmax(logits, dim=-1)

        energy_preds = []
        forces_preds = []
        with torch.enable_grad():
            for branch_id in range(num_branches):
                energy_pred, forces_pred = _predict_branch_energy_forces(
                    model, data, branch_id
                )
                energy_preds.append(energy_pred)
                forces_preds.append(forces_pred)

        energy_preds = torch.stack(energy_preds, dim=0)
        forces_preds = torch.stack(forces_preds, dim=0)

        weighted_energy, weighted_forces = _weighted_average(
            energy_preds, forces_preds, weights, data.batch
        )

        energy_true = data.energy.squeeze().float()
        forces_true = data.forces.float()

        loss_energy = loss_fn(weighted_energy, energy_true)
        loss_forces = loss_fn(weighted_forces, forces_true)
        loss = energy_weight * loss_energy + force_weight * loss_forces

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP weights for multi-branch predictions"
    )
    parser.add_argument("--inputfile", required=True, help="Path to JSON config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to pretrained model checkpoint (.pk)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing <dataset>-v2.bp files",
    )
    parser.add_argument(
        "--modelname",
        default="GFM",
        help="Base dataset name for single-dataset adios/pickle loading",
    )
    parser.add_argument(
        "--multi_model_list",
        help="Comma-separated dataset/model names (required for --multi)",
        default=None,
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--num_samples",
        type=int,
        help="set num samples per process for weak-scaling test",
        default=None,
    )
    parser.add_argument(
        "--task_parallel", action="store_true", help="enable task parallel"
    )
    parser.add_argument("--use_devicemesh", action="store_true", help="use device mesh")
    parser.add_argument("--oversampling", action="store_true", help="use oversampling")
    parser.add_argument(
        "--oversampling_num_samples",
        type=int,
        help="set num samples for oversampling",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override Training.batch_size from JSON",
    )
    parser.add_argument(
        "--energy_weight",
        type=float,
        default=None,
        help="Override energy loss weight",
    )
    parser.add_argument(
        "--force_weight",
        type=float,
        default=None,
        help="Override force loss weight",
    )
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        default="mlp_weights",
        help="Subfolder to save MLP checkpoints",
    )
    parser.add_argument(
        "--output",
        default="branch_weight_mlp.pt",
        help="Output path for MLP weights",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a saved MLP checkpoint to resume from",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--multi",
        help="Multi dataset",
        action="store_const",
        dest="format",
        const="multi",
    )
    parser.set_defaults(format="multi")
    args = parser.parse_args()

    with open(args.inputfile, "r") as f:
        config = json.load(f)

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    _configure_variable_names(config)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    hydragnn.utils.distributed.setup_ddp()

    if args.multi_model_list:
        train_loader, val_loader, _ = _load_multidataset_dataloaders(args, config)
    else:
        train_loader, val_loader, _ = _load_single_dataset_dataloaders(
            args, config, var_config
        )

    config = update_config(config, train_loader, val_loader, val_loader)

    precision = args.precision or config["NeuralNetwork"]["Training"].get(
        "precision", "fp32"
    )
    precision, param_dtype, _ = resolve_precision(precision)
    torch.set_default_dtype(param_dtype)

    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    )
    device = get_device()
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sample = next(iter(train_loader))
    sample = move_batch_to_device(sample, param_dtype)
    comp_dim = _reshape_composition(sample).shape[1]
    num_branches = getattr(model, "num_branches", 1)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    mlp = BranchWeightMLP(comp_dim, hidden_dims, num_branches).to(device)
    if args.resume is not None:
        ckpt_mlp = torch.load(args.resume, map_location=device)
        mlp.load_state_dict(ckpt_mlp["mlp_state_dict"], strict=True)
        print(f"Loaded MLP weights from {args.resume}")

    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = F.mse_loss

    energy_weight = (
        args.energy_weight
        if args.energy_weight is not None
        else config["NeuralNetwork"]["Architecture"].get("energy_weight", 1.0)
    )
    force_weight = (
        args.force_weight
        if args.force_weight is not None
        else config["NeuralNetwork"]["Architecture"].get("force_weight", 1.0)
    )

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            mlp,
            train_loader,
            optimizer,
            loss_fn,
            num_branches,
            energy_weight,
            force_weight,
            precision,
        )
        val_loss = validate_epoch(
            model,
            mlp,
            val_loader,
            loss_fn,
            num_branches,
            energy_weight,
            force_weight,
            precision,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    torch.save({"mlp_state_dict": mlp.state_dict()}, output_path)
    print(f"Saved MLP weights to {output_path}")


if __name__ == "__main__":
    main()
