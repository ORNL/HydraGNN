##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""FluxGNN surrogate for transient advection–diffusion with a source term.

    du/dt + div(v u - kappa grad u) = s   on [0,1]^2

The reference data come from a cell-centred finite-volume solver on a
triangulated unit square: upwind advection with the exact face-midpoint
velocity, two-point diffusion, upwind outflow and zero inflow at the
boundary, and ``REF_SUBSTEPS`` explicit substeps per step of length ``DT``.
Each sample is one step ``u^n -> u^{n+1}`` from a trajectory with a random
Gaussian initial blob, a random uniform-plus-rotation velocity ``v`` and a
random Gaussian source ``s``. The model sees ``u``, ``v`` and ``s`` per cell.

* ``fv_transport_fluxgnn.json``: FluxGNN on the cells, ``num_conv_layers``
  (fewer) explicit substeps.
* ``fv_transport_mgn.json``: MeshGraphNet baseline on the dual graph (one
  node per cell at its centroid, one edge pair per interior face), same
  samples, split, epochs and precision.

After training it reports the one-step relative L2 error on the test split
and, on a held-out trajectory at a finer unseen resolution, the
autoregressive rollout error, the relative drift of the total amount
``sum V u`` from the reference after the rollout, and (FluxGNN only) the
worst relative global balance residual of the model's own fluxes.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.data.meshes import unit_square_mesh
from hydragnn.models.rollout import rollout
from hydragnn.physics import balance_residual, fv_increment
from hydragnn.preprocess.load_data import SchemaPreparedDataset

TRAIN_RESOLUTIONS = (8, 10, 12)
HELDOUT_RESOLUTION = 16
DT = 0.02
KAPPA = 0.005
REF_SUBSTEPS = 16
STEPS_PER_TRAJECTORY = 6
ROLLOUT_STEPS = 10


def velocity(points, speed, angle, omega):
    x, y = points[:, 0] - 0.5, points[:, 1] - 0.5
    vx = speed * math.cos(angle) - omega * y
    vy = speed * math.sin(angle) + omega * x
    return torch.stack([vx, vy], dim=1)


def reference_step(mesh, u, params):
    """One step of length DT with REF_SUBSTEPS explicit substeps."""
    o, n = mesh.face_index
    mid = mesh.pos[mesh.face_nodes].mean(dim=0)
    vn = (velocity(mid, *params) * mesh.face_normal).sum(1, keepdim=True)
    dist = (mesh.cell_centroid[n] - mesh.cell_centroid[o]).norm(dim=1, keepdim=True)
    bmid = mesh.pos[mesh.bface_nodes].mean(dim=0)
    bvn = (velocity(bmid, *params) * mesh.bface_normal).sum(1, keepdim=True)
    dt = DT / REF_SUBSTEPS
    for _ in range(REF_SUBSTEPS):
        flux = vn * torch.where(vn > 0, u[o], u[n]) - KAPPA * (u[n] - u[o]) / dist
        bflux = bvn.clamp_min(0.0) * u[mesh.bface_owner]
        u = u + fv_increment(mesh, flux, bflux, mesh.s, dt)
    return u


def trajectory(n, generator, num_steps):
    """Meshes for ``num_steps`` consecutive one-step samples."""
    mesh = unit_square_mesh(n, n, dtype=torch.float64)
    rand = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi, generator=generator))
    params = (rand(0.3, 1.0), rand(0.0, 2 * math.pi), rand(-2.0, 2.0))
    c = mesh.cell_centroid
    center = torch.tensor([rand(0.3, 0.7), rand(0.3, 0.7)], dtype=c.dtype)
    width = rand(0.08, 0.15)
    u = torch.exp(-((c - center) ** 2).sum(1, keepdim=True) / (2 * width**2))
    src_center = torch.tensor([rand(0.2, 0.8), rand(0.2, 0.8)], dtype=c.dtype)
    mesh.s = rand(0.0, 2.0) * torch.exp(
        -((c - src_center) ** 2).sum(1, keepdim=True) / (2 * 0.1**2)
    )
    mesh.v = velocity(c, *params)
    samples = []
    for _ in range(num_steps):
        u_next = reference_step(mesh, u, params)
        sample = mesh.clone()
        sample.u, sample.u_next = u, u_next
        samples.append(sample)
        u = u_next
    return samples


def make_dataset(num_samples, resolutions, seed, steps=STEPS_PER_TRAJECTORY):
    g = torch.Generator().manual_seed(seed)
    dataset = []
    i = 0
    while len(dataset) < num_samples:
        dataset += trajectory(resolutions[i % len(resolutions)], g, steps)
        i += 1
    return dataset[:num_samples]


def dataset_hash(dataset):
    """SHA-256 of the generated samples, for provenance in the reported results."""
    h = hashlib.sha256()
    for sample in dataset:
        for key in ("pos", "face_index", "u", "v", "s", "u_next"):
            h.update(sample[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def to_dual(sample):
    """Cell-centred sample as a node graph for node-based baselines."""
    o, n = sample.face_index
    return Data(
        pos=sample.cell_centroid,
        edge_index=torch.stack([torch.cat([o, n]), torch.cat([n, o])]),
        u=sample.u,
        v=sample.v,
        s=sample.s,
        u_next=sample.u_next,
    )


def relative_l2(pred, true):
    return math.sqrt(float(((pred - true) ** 2).sum()) / float((true**2).sum()))


def finite_or_inf(value):
    return value if math.isfinite(value) else math.inf


def to_model(data, device, dtype):
    """Move to ``device`` and cast floating tensors to the model dtype."""
    data = data.to(device)
    for store in data.stores:
        for key, value in store.items():
            if torch.is_tensor(value) and value.is_floating_point():
                store[key] = value.to(dtype)
    return data


def evaluate(model, schema, test_set, device, dual=False):
    model.eval()
    dtype = next(model.parameters()).dtype
    num = den = 0.0
    with torch.no_grad():
        for data in DataLoader(test_set, batch_size=16):
            data = to_model(data, device, dtype)
            pred = model(data)[0]
            num += float(((pred - data.u_next) ** 2).sum())
            den += float((data.u_next**2).sum())

    # Held-out trajectory: rollout error, drift of sum V u, flux balance.
    g = torch.Generator().manual_seed(7)
    traj = trajectory(HELDOUT_RESOLUTION, g, ROLLOUT_STEPS)
    # Apply the training normalizer so rollout inputs match training.
    normalizer = getattr(test_set, "normalizer", None)
    if dual:
        first = SchemaPreparedDataset([to_dual(traj[0])], schema, normalizer)[0]
        first = to_model(first, device, dtype)
        states = rollout(model, first, ROLLOUT_STEPS, [0], level="node")
    else:
        first = SchemaPreparedDataset(traj[:1], schema, normalizer)[0]
        first = to_model(first, device, dtype)
        states = rollout(model, first, ROLLOUT_STEPS, model.state_cols, level="cell")
    errors = [
        relative_l2(state, step.u_next.to(device)) for state, step in zip(states, traj)
    ]
    volume = traj[-1].cell_volume.to(device)
    true = traj[-1].u_next.to(device)
    drift = float(
        ((volume * (states[-1] - true)).sum() / (volume * true.abs()).sum()).abs()
    )
    results = {
        "test_rel_l2": math.sqrt(num / den),
        "heldout_resolution": HELDOUT_RESOLUTION,
        "rollout_rel_l2": errors,
        "rollout_rel_mass_drift": drift,
    }
    if dual:
        return results
    balance = 0.0
    data = first.clone()
    with torch.no_grad():
        for state in states:
            terms = model.fluxes(data)
            u_old = data.cell_x[:, model.state_cols]
            residual, scale = balance_residual(
                data, u_old, terms["u_new"], terms["bface_flux"], terms["source"], DT
            )
            worst = float((residual.abs() / scale).max())
            balance = max(balance, finite_or_inf(worst))
            x = data.cell_x.clone()
            x[:, model.state_cols] = state
            data.cell_x = x
    results["max_rel_balance_residual"] = balance
    return results


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def main(num_samples=360, num_epoch=None, config_name="fv_transport_fluxgnn.json"):
    torch.manual_seed(0)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, config_name)) as f:
        config = json.load(f)
    if num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = num_epoch
    verbosity = config["Verbosity"]["level"]

    hydragnn.distributed.setup_ddp()
    log_name = os.path.splitext(os.path.basename(config_name))[0]
    hydragnn.utils.print.print_utils.setup_log(log_name)

    dual = config["NeuralNetwork"]["Architecture"]["mpnn_type"] != "FluxGNN"
    dataset = make_dataset(num_samples, TRAIN_RESOLUTIONS, seed=1)
    data_hash = dataset_hash(dataset)
    if dual:
        dataset = [to_dual(sample) for sample in dataset]
    random.seed(0)
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    world_size, _ = hydragnn.distributed.get_comm_size_and_rank()
    training = config["NeuralNetwork"]["Training"]
    if training["batch_size"] % world_size != 0:
        raise ValueError(
            f"Global batch_size {training['batch_size']} is not divisible by "
            f"the number of ranks {world_size}"
        )
    training["batch_size"] //= world_size
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        train,
        val,
        test,
        training["batch_size"],
        variables=config["Variables"],
    )
    config = hydragnn.config.update_config(
        config, train_loader, val_loader, test_loader
    )
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity
    )
    lr = training["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    model, optimizer = hydragnn.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )
    writer = hydragnn.models.utils.model.get_summary_writer(log_name)
    hydragnn.config.save_config(config, log_name)

    start = time.perf_counter()
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config,
        log_name,
        verbosity,
        precision=training.get("precision", "fp32"),
    )
    train_seconds = time.perf_counter() - start
    hydragnn.models.utils.save_model(
        model,
        optimizer,
        log_name,
        metadata=hydragnn.models.utils.checkpoint_metadata(config, train_loader),
    )

    device = next(model.parameters()).device
    schema = hydragnn.config.parse_variable_schema(config["Variables"])
    results = evaluate(unwrap(model), schema, test_loader.dataset, device, dual)
    results.update(
        dataset_sha256=data_hash,
        train_seconds=train_seconds,
        device=str(device),
        world_size=world_size,
    )
    hydragnn.utils.print.print_utils.print_master(json.dumps(results))

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, default=360)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--config", default="fv_transport_fluxgnn.json")
    args = parser.parse_args()
    main(args.num_samples, args.num_epoch, args.config)
