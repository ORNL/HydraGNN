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
"""Numerical checks for FNO, GINO and GSNO (WP06).

Prints one JSON report and exits non-zero if a check fails. Checks:

FNO
  * ``spectral_identity``: all modes kept and ``R = w`` reproduces ``w * x``
    in 1D/2D/3D for even and odd sizes (Hermitian handling, truncation);
  * ``discretization_invariance``: a band-limited input sampled at two
    resolutions gives the same spectral-layer output at shared points;
  * ``padding_round_trip`` and an FP64 ``gradcheck`` of a full FNO.
GINO
  * ``fit_3d``: GINO fitted on a random 3D point cloud, evaluated at
    non-matching query points, against the spectral reference: an FNO with the
    same latent grid and modes fitted on grid samples of the same field;
  * ``memory``: neighbour pairs, time and (on CUDA) peak allocator memory of
    one forward/backward vs. point count.
GSNO
  * ``basis``: orthonormality, eigen-residual and dense/LOBPCG agreement;
  * ``sign_invariance`` and ``degenerate``: eigenvector sign flips never
    change the output; a rotation inside a degenerate eigenspace changes it
    unless ``tie_degenerate`` is set;
  * ``reconstruction``: with the full basis and ``R = I`` the spectral path is
    the identity; the truncated projection error of a smooth field decays
    with the number of modes;
  * ``temporal``: the temporal FFT keeps exactly the first ``k_t`` modes;
  * ``eigensolve``: dense and LOBPCG wall time vs. mesh size.

``--device cuda`` runs the model checks on a GPU (the eigensolves stay on the
CPU, as in preprocessing).
"""

import argparse
import json
import math
import platform
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

from hydragnn.data.meshes import (
    attach_graph_spectral_basis,
    graph_spectral_basis,
    unit_square_mesh,
)
from hydragnn.data.meshes.graph_spectral import normalized_laplacian
from hydragnn.models.fno import FNO, DomainPadding, SpectralConv
from hydragnn.models.gino import GINO, radius_neighbors
from hydragnn.models.gsno import GSNO, GraphSpectralConv

F64 = torch.float64


def _max_abs(a, b):
    return float((a - b).abs().max())


# -- FNO ---------------------------------------------------------------------
def fno_checks(device):
    report = {}
    errors = {}
    for sizes in [(16,), (15,), (8, 10), (9, 7), (6, 5, 8), (5, 7, 4)]:
        conv = SpectralConv(1, 1, [2 * s for s in sizes], bias=False).to(device, F64)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[..., 0] = 1.7
        x = torch.randn(2, 1, *sizes, dtype=F64, device=device)
        errors["x".join(map(str, sizes))] = _max_abs(conv(x), 1.7 * x)
    report["spectral_identity"] = errors

    torch.manual_seed(0)
    conv = SpectralConv(1, 1, [5, 5], bias=False).to(device, F64)

    def field(n):
        t = torch.arange(n, dtype=F64, device=device) / n
        gx, gy = torch.meshgrid(t, t, indexing="ij")
        f = torch.sin(2 * math.pi * gx) * torch.cos(4 * math.pi * gy) + torch.cos(
            2 * math.pi * (gx + gy)
        )
        return f.view(1, 1, n, n)

    coarse, fine = conv(field(16)), conv(field(48))
    report["discretization_invariance"] = _max_abs(coarse, fine[..., ::3, ::3])

    x = torch.randn(2, 3, 10, 12, device=device)
    padded, pads = DomainPadding([0.25, 0.5]).pad(x)
    report["padding_round_trip"] = _max_abs(DomainPadding.unpad(padded, pads), x)

    torch.manual_seed(0)
    model = FNO(1, 4, [1], ["node"], 1, n_modes=(3, 3), domain_padding=0.2)
    model = model.to(device, F64)
    x = torch.randn(1, 1, 5, 6, dtype=F64, device=device, requires_grad=True)
    report["gradcheck"] = bool(torch.autograd.gradcheck(model._core, (x,)))
    ok = (
        max(errors.values()) < 1e-12
        and report["discretization_invariance"] < 1e-12
        and report["padding_round_trip"] == 0.0
        and report["gradcheck"]
    )
    return report, ok


# -- GINO ----------------------------------------------------------------------
def _field3d(p):
    return torch.sin(math.pi * p[:, 0]) * torch.cos(math.pi * p[:, 1]) + p[:, 2] ** 2


def _fit(model, data_fn, target, steps, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((data_fn() - target) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss)


def gino_checks(device, steps, seed):
    report = {}
    res, modes = (8, 8, 8), (4, 4, 4)
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(600, 3, generator=g, dtype=F64).to(device)
    queries = torch.rand(250, 3, generator=g, dtype=F64).to(device)
    data = Data(pos=pos, x=torch.ones(600, 1, dtype=F64, device=device), num_nodes=600)
    target = _field3d(pos).unsqueeze(1)

    torch.manual_seed(seed)
    gino = GINO(
        1,
        16,
        [1, 1],
        ["node", "query"],
        2,
        coord_dim=3,
        n_modes=modes,
        latent_resolution=res,
        in_radius=0.25,
        out_radius=0.25,
        embed_channels=8,
        in_channel_mlp_hidden_layers=(32,),
        out_channel_mlp_hidden_layers=(32,),
        loss_weights=[1.0, 1.0],
    ).to(device, F64)
    # The query head is its own head: train it on separate query points and
    # evaluate on the held-out ``queries`` (non-matching with nodes and train).
    train_q = torch.rand(400, 3, generator=g, dtype=F64).to(device)
    data.query_pos = train_q
    both = torch.cat([target, _field3d(train_q).unsqueeze(1)])
    t0 = time.perf_counter()
    loss = _fit(gino, lambda: torch.cat(gino(data)), both, steps)
    gino.eval()
    data.query_pos = queries
    with torch.no_grad():
        q_pred = gino(data)[1]
    q_true = _field3d(queries).unsqueeze(1)
    report["gino_train_loss"] = loss
    report["gino_query_rel_l2"] = float((q_pred - q_true).norm() / q_true.norm())
    report["gino_fit_seconds"] = time.perf_counter() - t0

    # Spectral reference: FNO on grid samples of the same field.
    axes = [torch.linspace(0, 1, n, dtype=F64, device=device) for n in res]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    grid_data = Data(
        x=torch.ones(grid.shape[0], 1, dtype=F64, device=device),
        grid_shape=torch.tensor([list(res)], device=device),
        num_nodes=grid.shape[0],
    )
    torch.manual_seed(seed)
    fno = FNO(1, 16, [1], ["node"], 2, n_modes=modes).to(device, F64)
    t0 = time.perf_counter()
    _fit(fno, lambda: fno(grid_data)[0], _field3d(grid).unsqueeze(1), steps)
    with torch.no_grad():
        on_grid = fno(grid_data)[0].view(1, 1, *res)
    # Trilinear interpolation of the FNO grid output at the queries.
    sample = torch.nn.functional.grid_sample(
        on_grid,
        (queries[:, [2, 1, 0]] * 2 - 1).view(1, -1, 1, 1, 3),
        align_corners=True,
    ).view(-1, 1)
    report["fno_reference_query_rel_l2"] = float(
        (sample - q_true).norm() / q_true.norm()
    )
    report["fno_fit_seconds"] = time.perf_counter() - t0

    memory = []
    for n in (500, 2000, 8000):
        pts = torch.rand(n, 3, generator=g, dtype=F64).to(device)
        d = Data(
            pos=pts,
            x=torch.ones(n, 1, dtype=F64, device=device),
            query_pos=pts,
            num_nodes=n,
        )
        pairs_in = int(
            radius_neighbors(gino.latent_queries.to(F64), pts, 0.25)[0].numel()
        )
        pairs_out = int(
            radius_neighbors(pts, gino.latent_queries.to(F64), 0.25)[0].numel()
        )
        entry = {"points": n, "pairs_in": pairs_in, "pairs_out": pairs_out}
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        gino.train()
        gino.zero_grad()
        t0 = time.perf_counter()
        gino(d)[0].sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            entry["peak_bytes"] = torch.cuda.max_memory_allocated() - base
        entry["seconds"] = time.perf_counter() - t0
        memory.append(entry)
    report["memory"] = memory
    ok = (
        report["gino_query_rel_l2"] < 0.1
        and report["fno_reference_query_rel_l2"] < 0.1
        and memory[-1]["pairs_out"] > memory[0]["pairs_out"]
    )
    return report, ok


# -- GSNO ----------------------------------------------------------------------
def _ring(n):
    angle = 2 * np.pi * torch.arange(n, dtype=F64) / n
    pos = torch.stack([angle.cos(), angle.sin()], dim=1)
    i = torch.arange(n)
    edge = torch.stack([i, (i + 1) % n])
    return pos, torch.cat([edge, edge.flip(0)], dim=1)


def gsno_checks(device, sizes):
    report = {}
    mesh = unit_square_mesh(12, 12)
    values, basis, _, info = graph_spectral_basis(mesh.pos, mesh.edge_index, 12)
    lap, _ = normalized_laplacian(mesh.pos, mesh.edge_index, info["sigma"])
    sparse, sbasis, _, _ = graph_spectral_basis(
        mesh.pos, mesh.edge_index, 12, method="lobpcg"
    )
    report["basis"] = {
        "orthonormality": _max_abs(basis.T @ basis, torch.eye(12, dtype=F64)),
        "eigen_residual": float(
            np.abs(lap @ basis.numpy() - basis.numpy() * values.numpy()).max()
        ),
        "lobpcg_eigenvalue_diff": _max_abs(sparse, values),
        "lobpcg_subspace_diff": _max_abs(basis @ basis.T, sbasis @ sbasis.T),
        "sigma": info["sigma"],
    }

    torch.manual_seed(0)
    data = unit_square_mesh(10, 10)
    data.x = torch.randn(data.num_nodes, 2, dtype=F64)
    attach_graph_spectral_basis(data, 8)
    data = data.to(device)
    model = GSNO(2, 8, [1], ["node"], 2, spatial_modes=8, projection_channels=16)
    model = model.to(device, F64).eval()
    with torch.no_grad():
        ref = model(data)[0]
        flipped = data.clone()
        flipped.gsno_basis = data.gsno_basis * torch.tensor(
            [1.0, -1.0] * 4, dtype=F64, device=device
        )
        report["sign_invariance"] = _max_abs(model(flipped)[0], ref)

    pos, edge_index = _ring(24)
    ring = Data(pos=pos, edge_index=edge_index, num_nodes=24)
    ring.x = torch.randn(24, 2, dtype=F64)
    attach_graph_spectral_basis(ring, 5)
    ring = ring.to(device)
    rotated = ring.clone()
    c, s = math.cos(0.7), math.sin(0.7)
    b = ring.gsno_basis.clone()
    b[:, 1], b[:, 2] = c * ring.gsno_basis[:, 1] - s * ring.gsno_basis[:, 2], (
        s * ring.gsno_basis[:, 1] + c * ring.gsno_basis[:, 2]
    )
    rotated.gsno_basis = b
    degenerate = {"groups": ring.gsno_groups.tolist()[0]}
    for tie in (False, True):
        torch.manual_seed(0)
        m = (
            GSNO(
                2,
                8,
                [1],
                ["node"],
                2,
                spatial_modes=5,
                projection_channels=16,
                tie_degenerate=tie,
            )
            .to(device, F64)
            .eval()
        )
        with torch.no_grad():
            degenerate[f"tied_{tie}"] = _max_abs(m(rotated)[0], m(ring)[0])
    report["degenerate"] = degenerate

    small = unit_square_mesh(4, 4)
    n = small.num_nodes
    lap, _ = normalized_laplacian(small.pos, small.edge_index)
    _, vectors = np.linalg.eigh(lap.toarray())
    full = torch.from_numpy(vectors).to(device)
    conv = GraphSpectralConv(2, n).to(device, F64)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(2).expand(n, 2, 2))
        v = torch.randn(n, 1, 2, dtype=F64, device=device)
        graph = torch.zeros(n, dtype=torch.long, device=device)
        identity_err = _max_abs(conv(v, full, graph, 1), v)
    big = unit_square_mesh(16, 16)
    _, phi, _, _ = graph_spectral_basis(big.pos, big.edge_index, 64)
    smooth = torch.sin(math.pi * big.pos[:, 0]) * torch.sin(math.pi * big.pos[:, 1])
    projection = {}
    for k in (4, 16, 64):
        p = phi[:, :k]
        projection[k] = float((p @ (p.T @ smooth) - smooth).norm() / smooth.norm())
    report["reconstruction"] = {
        "identity_full_basis": identity_err,
        "projection_rel_error": projection,
    }

    conv = GraphSpectralConv(1, 4, temporal_modes=2).to(device, F64)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[..., 0, 0, 0] = 1.0
        t = torch.arange(10, dtype=F64, device=device) / 10
        phi0 = data.gsno_basis[:, :1]
        signal = phi0 * (1 + torch.cos(2 * math.pi * t) + torch.cos(8 * math.pi * t))
        graph = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        out = conv(signal.unsqueeze(-1), data.gsno_basis, graph, 1)[..., 0]
        report["temporal_low_pass"] = _max_abs(
            out, phi0 * (1 + torch.cos(2 * math.pi * t))
        )

    timing = []
    for m in sizes:
        mesh = unit_square_mesh(m, m)
        entry = {"nodes": mesh.num_nodes}
        for method in ("dense", "lobpcg"):
            if method == "dense" and mesh.num_nodes > 5000:
                continue
            t0 = time.perf_counter()
            graph_spectral_basis(mesh.pos, mesh.edge_index, 16, method=method)
            entry[f"{method}_seconds"] = time.perf_counter() - t0
        timing.append(entry)
    report["eigensolve"] = timing

    ok = (
        report["basis"]["orthonormality"] < 1e-12
        and report["basis"]["eigen_residual"] < 1e-10
        and report["basis"]["lobpcg_eigenvalue_diff"] < 1e-8
        and report["basis"]["lobpcg_subspace_diff"] < 1e-6
        and report["sign_invariance"] < 1e-12
        and degenerate["tied_True"] < 1e-12
        and degenerate["tied_False"] > 1e-6
        and identity_err < 1e-12
        and projection[64] < projection[16] < projection[4]
        and report["temporal_low_pass"] < 1e-12
    )
    return report, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    results, failed = {}, []
    for name, fn, extra in (
        ("fno", fno_checks, ()),
        ("gino", gino_checks, (args.steps, args.seed)),
        ("gsno", gsno_checks, (args.sizes,)),
    ):
        results[name], ok = fn(device, *extra)
        if not ok:
            failed.append(name)
    results["meta"] = {
        "failed": failed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "precision": "fp64",
        "seed": args.seed,
        "steps": args.steps,
        "wall_seconds": time.perf_counter() - t0,
    }
    print(json.dumps(results, indent=2, default=float))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
