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
"""MeshGraphNet surrogate for -Δu = f on [0,1]^2 with u = 0 on the boundary.

Each sample is a manufactured solution
``u = a sin(kπx) sin(lπy)``, ``f = (k² + l²) π² u`` on a triangulated unit
square whose resolution varies per sample. The model maps (f, boundary flag)
on mesh nodes to u. With ``poisson_mgn_flux.json`` it also predicts the
diffusive flux ``q = -∇u·n A`` through every interior face (midpoint rule,
multiplied by the resolution n so it is O(1) at every resolution). f and q
use the ``scale_only`` normalization fitted on the training split. After
training, the relative L2 error of each output, in physical units, is reported
on held-out samples at a finer, unseen resolution. ``batch_size`` in the config
is the global batch, split evenly across MPI ranks. With
``poisson_msmgn.json`` (MultiscaleMeshGraphNet) every sample is a two-level
mesh hierarchy whose coarse level has half the resolution. With
``poisson_rigno.json`` (RIGNO) every sample carries its regional graphs, with
regional nodes drawn from a seeded generator.

Configs whose inputs include ``kappa`` (``darcy_*.json``) switch to the Darcy
problem ``-∇·(κ∇u) = f`` with the same ``u`` and a log-periodic coefficient
``κ = exp(β cos(pπx) cos(qπy))``, ``β ∈ [-1, 1]`` (contrast up to e²), so
``f = -κΔu - ∇κ·∇u``; κ is an extra node input and ``q`` uses ``-κ∇u·n A``.
"""

import argparse
import json
import math
import os
import random

import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.data.meshes import build_hierarchy, build_rigno_graph, unit_square_mesh
from hydragnn.preprocess.load_data import SchemaPreparedDataset

TRAIN_RESOLUTIONS = (6, 8, 10)
HELDOUT_RESOLUTION = 14


GRAPH_MODES = {"MultiscaleMeshGraphNet": "hierarchy", "RIGNO": "rigno"}


def _mesh(n, graph):
    if graph != "hierarchy":
        return unit_square_mesh(n, n, dtype=torch.float32)
    # Build in float64 so the nested-volume check is exact, then cast.
    mesh = build_hierarchy(
        unit_square_mesh(n, n), unit_square_mesh(n // 2, n // 2), node_transfer=True
    )
    for key, value in mesh.items():
        if value.is_floating_point():
            mesh[key] = value.float()
    return mesh


def _coefficient(pos, beta, p, q):
    """κ = exp(β cos(pπx) cos(qπy)) and its gradient at ``pos``."""
    cx, cy = torch.cos(p * math.pi * pos[:, 0]), torch.cos(q * math.pi * pos[:, 1])
    sx, sy = torch.sin(p * math.pi * pos[:, 0]), torch.sin(q * math.pi * pos[:, 1])
    kappa = torch.exp(beta * cx * cy)
    grad = -beta * math.pi * torch.stack([p * sx * cy, q * cx * sy], dim=1)
    return kappa, kappa.unsqueeze(1) * grad


def _solution(pos, amplitude, k, l):
    """u = a sin(kπx) sin(lπy) and its gradient at ``pos``."""
    sx, sy = torch.sin(k * math.pi * pos[:, 0]), torch.sin(l * math.pi * pos[:, 1])
    cx, cy = torch.cos(k * math.pi * pos[:, 0]), torch.cos(l * math.pi * pos[:, 1])
    u = amplitude * sx * sy
    grad = amplitude * math.pi * torch.stack([k * cx * sy, l * sx * cy], dim=1)
    return u, grad


def manufactured_sample(n, generator, max_mode=2, graph=None, darcy=False):
    k, l = torch.randint(1, max_mode + 1, (2,), generator=generator).tolist()
    amplitude = float(torch.empty(1).uniform_(0.5, 1.5, generator=generator))
    if darcy:
        kp, kq = torch.randint(1, 3, (2,), generator=generator).tolist()
        beta = float(torch.empty(1).uniform_(-1.0, 1.0, generator=generator))
    else:
        kp, kq, beta = 1, 1, 0.0  # κ = 1
    mesh = _mesh(n, graph)
    u, grad_u = _solution(mesh.pos, amplitude, k, l)
    kappa, grad_kappa = _coefficient(mesh.pos, beta, kp, kq)
    # -∇·(κ∇u) = -κΔu - ∇κ·∇u, with -Δu = (k² + l²)π² u.
    f = kappa * (k * k + l * l) * math.pi**2 * u - (grad_kappa * grad_u).sum(dim=1)
    mesh.f = f.unsqueeze(1)
    mesh.boundary = mesh.boundary_node_mask.float().unsqueeze(1)
    mesh.u = u.unsqueeze(1)
    if darcy:
        mesh.kappa = kappa.unsqueeze(1)

    mid = mesh.pos[mesh.face_nodes].mean(dim=0)
    _, grad = _solution(mid, amplitude, k, l)
    kappa_mid, _ = _coefficient(mid, beta, kp, kq)
    q = -kappa_mid.unsqueeze(1) * (grad * mesh.face_normal).sum(dim=1, keepdim=True)
    q = q * mesh.face_area
    # A_f ~ 1/n, so q * n has the same magnitude across resolutions.
    mesh.q = q * n
    if graph == "rigno":
        mesh = build_rigno_graph(mesh, generator=generator)
    return mesh


def make_dataset(num_samples, resolutions, seed, graph=None, darcy=False):
    g = torch.Generator().manual_seed(seed)
    return [
        manufactured_sample(
            resolutions[i % len(resolutions)], g, graph=graph, darcy=darcy
        )
        for i in range(num_samples)
    ]


def relative_l2(model, loader, names, device, normalizer=None):
    """Relative L2 error per output head in physical units, keyed by name."""
    model.eval()
    num = dict.fromkeys(names, 0.0)
    den = dict.fromkeys(names, 0.0)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            for name, pred in zip(names, model(data)):
                true = data[name]
                if normalizer is not None:
                    pred = normalizer.inverse(name, pred)
                    true = normalizer.inverse(name, true)
                num[name] += float(((pred - true) ** 2).sum())
                den[name] += float((true**2).sum())
    return {name: math.sqrt(num[name] / den[name]) for name in names}


def main(num_samples=240, num_epoch=None, config_name="poisson_mgn.json"):
    torch.manual_seed(0)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, config_name)) as f:
        config = json.load(f)
    if num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = num_epoch
    verbosity = config["Verbosity"]["level"]

    hydragnn.distributed.setup_ddp()
    log_name = os.path.splitext(config_name)[0]
    hydragnn.utils.print.print_utils.setup_log(log_name)

    graph = GRAPH_MODES.get(config["NeuralNetwork"]["Architecture"]["mpnn_type"])
    darcy = any(v["name"] == "kappa" for v in config["Variables"]["inputs"])
    dataset = make_dataset(
        num_samples, TRAIN_RESOLUTIONS, seed=1, graph=graph, darcy=darcy
    )
    # split_dataset shuffles with Python's random; every rank needs the same split.
    random.seed(0)
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    # batch_size is global: on a small dataset, a fixed per-rank batch would cut
    # the optimizer steps per epoch by the world size and stall convergence.
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
    lr = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    model, optimizer = hydragnn.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )
    writer = hydragnn.models.utils.model.get_summary_writer(log_name)
    hydragnn.config.save_config(config, log_name)

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
    )
    hydragnn.models.utils.save_model(
        model,
        optimizer,
        log_name,
        metadata=hydragnn.models.utils.checkpoint_metadata(config, train_loader),
    )

    schema = hydragnn.config.parse_variable_schema(config["Variables"])
    test_set = test_loader.dataset
    normalizer = test_set.normalizer
    heldout = SchemaPreparedDataset(
        make_dataset(16, (HELDOUT_RESOLUTION,), seed=2, graph=graph, darcy=darcy),
        schema,
        normalizer,
    )
    device = next(model.parameters()).device
    names = [spec.name for spec in schema.outputs]
    results = {
        "test_rel_l2": relative_l2(
            model, DataLoader(test_set, batch_size=16), names, device, normalizer
        ),
        "heldout_resolution": HELDOUT_RESOLUTION,
        "heldout_rel_l2": relative_l2(
            model, DataLoader(heldout, batch_size=16), names, device, normalizer
        ),
    }
    hydragnn.utils.print.print_utils.print_master(json.dumps(results))

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, default=240)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--config", default="poisson_mgn.json")
    args = parser.parse_args()
    main(args.num_samples, args.num_epoch, args.config)
