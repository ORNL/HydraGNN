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
on held-out samples at a finer, unseen resolution.
"""

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.data.meshes import unit_square_mesh
from hydragnn.preprocess.load_data import SchemaPreparedDataset

TRAIN_RESOLUTIONS = (6, 8, 10)
HELDOUT_RESOLUTION = 14


def manufactured_sample(n, generator, max_mode=2):
    k, l = torch.randint(1, max_mode + 1, (2,), generator=generator).tolist()
    amplitude = float(torch.empty(1).uniform_(0.5, 1.5, generator=generator))
    mesh = unit_square_mesh(n, n, dtype=torch.float32)
    x, y = mesh.pos[:, 0], mesh.pos[:, 1]
    u = amplitude * torch.sin(k * math.pi * x) * torch.sin(l * math.pi * y)
    f = (k * k + l * l) * math.pi**2 * u
    mesh.f = f.unsqueeze(1)
    mesh.boundary = mesh.boundary_node_mask.float().unsqueeze(1)
    mesh.u = u.unsqueeze(1)

    mid = mesh.pos[mesh.face_nodes].mean(dim=0)
    sx, sy = torch.sin(k * math.pi * mid[:, 0]), torch.sin(l * math.pi * mid[:, 1])
    cx, cy = torch.cos(k * math.pi * mid[:, 0]), torch.cos(l * math.pi * mid[:, 1])
    grad = amplitude * math.pi * torch.stack([k * cx * sy, l * sx * cy], dim=1)
    q = -(grad * mesh.face_normal).sum(dim=1, keepdim=True) * mesh.face_area
    # A_f ~ 1/n, so q * n has the same magnitude across resolutions.
    mesh.q = q * n
    return mesh


def make_dataset(num_samples, resolutions, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        manufactured_sample(resolutions[i % len(resolutions)], g)
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

    dataset = make_dataset(num_samples, TRAIN_RESOLUTIONS, seed=1)
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        train,
        val,
        test,
        config["NeuralNetwork"]["Training"]["batch_size"],
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

    schema = hydragnn.config.parse_variable_schema(config["Variables"])
    test_set = test_loader.dataset
    normalizer = test_set.normalizer
    heldout = SchemaPreparedDataset(
        make_dataset(16, (HELDOUT_RESOLUTION,), seed=2), schema, normalizer
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
