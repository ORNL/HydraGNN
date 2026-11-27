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
"""Numerical checks for DeepONet and CompositeNeuralOperator (WP07).

Prints one JSON report and exits non-zero if a check fails. Checks:

DeepONet
  * ``antiderivative``: the function-to-function map ``u -> int_0^y u`` of
    Lu et al. (2021), fitted in FP64 on random Fourier inputs sampled at
    fixed sensors and evaluated on held-out functions at points that were
    never trained on;
  * ``strategies``: every ``multi_output_strategy`` (unstacked and stacked)
    reproduces the DeepXDE contraction computed from the raw branch and
    trunk outputs, and ``independent`` equals K separate single-output
    DeepONets with the same weights;
  * ``queries``: query heads at arbitrary points equal node heads at the
    same points, and batched evaluation equals per-graph evaluation;
  * ``sensor_errors``: malformed sensor inputs are rejected;
  * ``gradcheck``: FP64 ``gradcheck`` w.r.t. sensors and query points.
Composite
  * ``fit``: the MGN+GSNO and RIGNO-regional+flux combinations fit a smooth
    field on a mesh (loss reduction);
  * ``batching`` and ``conservative``: batched equals single-graph output;
    the conservative face decoder is antisymmetric under face flips;
  * ``rejections``: invalid pipelines fail with the expected messages.
"""

import argparse
import copy
import json
import math
import platform
import sys
import time

import torch
from torch_geometric.data import Batch, Data

from hydragnn.data.meshes import (
    attach_graph_spectral_basis,
    build_rigno_graph,
    locate_queries,
    unit_square_mesh,
)
from hydragnn.models.composite import CompositeNeuralOperator, resolve_pipeline
from hydragnn.models.deeponet import DeepONet

F64 = torch.float64
STRATEGIES = ("independent", "split_both", "split_branch", "split_trunk")


def _max_abs(a, b):
    return float((a - b).abs().max())


def _rel_l2(pred, target):
    return float((pred - target).norm() / target.norm())


# -- DeepONet ----------------------------------------------------------------
def _fourier(n, modes, g):
    """Random ``u(x) = sum_k a_k cos(2 pi k x) + b_k sin(2 pi k x)`` and its
    antiderivative ``U(y) = int_0^y u``."""
    k = torch.arange(1, modes + 1, dtype=F64)
    a = torch.randn(n, modes, generator=g, dtype=F64) / k
    b = torch.randn(n, modes, generator=g, dtype=F64) / k
    c = torch.randn(n, 1, generator=g, dtype=F64)
    w = 2 * math.pi * k

    def u(x):  # x [P] -> [n, P]
        arg = x[None, :, None] * w
        return c + (a[:, None] * arg.cos() + b[:, None] * arg.sin()).sum(-1)

    def big_u(y):
        arg = y[None, :, None] * w
        terms = a[:, None] * arg.sin() / w + b[:, None] * (1 - arg.cos()) / w
        return c * y[None] + terms.sum(-1)

    return u, big_u


def _antiderivative_batch(n, sensors, points, g, device):
    u, big_u = _fourier(n, 4, g)
    graphs = []
    values, targets = u(sensors), big_u(points)
    for i in range(n):
        graphs.append(
            Data(
                pos=points[:, None].clone(),
                x=torch.zeros(points.shape[0], 1, dtype=F64),
                deeponet_sensors=values[i : i + 1],
                y=targets[i][:, None],
                num_nodes=points.shape[0],
            )
        )
    return Batch.from_data_list(graphs).to(device)


def antiderivative_check(device, steps, seed):
    g = torch.Generator().manual_seed(seed)
    m = 50
    sensors = torch.linspace(0, 1, m, dtype=F64)
    train_pts = torch.rand(40, generator=g, dtype=F64)
    test_pts = torch.rand(97, generator=g, dtype=F64)
    train = _antiderivative_batch(512, sensors, train_pts, g, device)
    test = _antiderivative_batch(64, sensors, test_pts, g, device)
    torch.manual_seed(seed)
    model = DeepONet(
        1,
        64,
        [1],
        ["node"],
        3,
        num_sensors=m,
        basis_dim=64,
        coord_dim=1,
        activation_function="relu",
    ).to(device, F64)
    with torch.no_grad():
        initial = _rel_l2(model(test)[0], test.y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(train)[0] - train.y) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    train_time = time.perf_counter() - t0
    with torch.no_grad():
        train_rel = _rel_l2(model(train)[0], train.y)
        test_rel = _rel_l2(model(test)[0], test.y)
    return {
        "num_sensors": m,
        "train_functions": 512,
        "test_functions": 64,
        "steps": steps,
        "initial_test_rel_l2": initial,
        "train_rel_l2": train_rel,
        "heldout_rel_l2": test_rel,
        "train_seconds": train_time,
        # Held-out error plateaus near 0.05 (query-point generalization from
        # 40 random training points); 0.08 leaves margin across devices.
        "ok": test_rel < 0.08 and test_rel < 0.1 * initial,
    }


def _random_graph(n, queries, seed, m=4, c=2):
    g = torch.Generator().manual_seed(seed)
    data = Data(
        pos=torch.rand(n, 2, generator=g, dtype=F64),
        x=torch.randn(n, c, generator=g, dtype=F64),
        deeponet_sensors=torch.randn(1, m * c, generator=g, dtype=F64),
        num_nodes=n,
    )
    if queries:
        data.query_pos = torch.rand(queries, 2, generator=g, dtype=F64)
    return data


def _deepxde(model, data, p):
    b = model.branch(data.deeponet_sensors)[0]
    t = model.act(model.trunk(data.pos))
    k = model.num_outputs
    if model.strategy == "independent":
        out = (t.reshape(-1, k, p) * b.reshape(k, p)).sum(-1)
    elif model.strategy == "split_both":
        out = (t.reshape(-1, k, p // k) * b.reshape(k, p // k)).sum(-1)
    elif model.strategy == "split_branch":
        out = t @ b.reshape(k, p).T
    else:
        out = (t.reshape(-1, k, p) * b).sum(-1)
    return out + model.bias


def strategy_checks(device):
    report, ok = {}, True
    data = _random_graph(15, 0, 0).to(device)
    k, p = 3, 6
    for stacked in (False, True):
        for strategy in STRATEGIES:
            torch.manual_seed(0)
            model = DeepONet(
                2,
                12,
                [k],
                ["node"],
                2,
                num_sensors=4,
                basis_dim=p,
                multi_output_strategy=strategy,
                stacked=stacked,
            ).to(device, F64)
            with torch.no_grad():
                model.bias.normal_()
                err = _max_abs(model(data)[0], _deepxde(model, data, p))
            key = f"{strategy}{'_stacked' if stacked else ''}"
            report[key] = err
            ok &= err < 1e-12
    # independent == K separate single-output DeepONets.
    torch.manual_seed(1)
    multi = DeepONet(2, 12, [k], ["node"], 2, num_sensors=4, basis_dim=p)
    multi = multi.to(device, F64)
    with torch.no_grad():
        multi.bias.normal_()
        columns = []
        for i in range(k):
            single = DeepONet(2, 12, [1], ["node"], 2, num_sensors=4, basis_dim=p)
            single = single.to(device, F64)
            for dst, src in (
                (single.branch, multi.branch),
                (single.trunk, multi.trunk),
            ):
                for w_dst, w_src in zip(
                    (*dst.weights, *dst.biases), (*src.weights, *src.biases)
                ):
                    w_dst.copy_(w_src[i : i + 1])
            single.bias.copy_(multi.bias[i : i + 1])
            columns.append(single(data)[0])
        err = _max_abs(multi(data)[0], torch.cat(columns, 1))
    report["independent_equals_separate"] = err
    ok &= err < 1e-12
    return report, ok


def query_checks(device):
    torch.manual_seed(0)
    model = DeepONet(
        2, 12, [1, 1], ["node", "query"], 2, num_sensors=4, basis_dim=6
    ).to(device, F64)
    with torch.no_grad():
        for net in (model.branch, model.trunk):
            for w in (*net.weights, *net.biases):
                w[1].copy_(w[0])
        model.bias[1] = model.bias[0]
        data = _random_graph(20, 31, 0)
        # Same sensors, nodes moved to the query points.
        moved = data.clone()
        moved.pos = data.query_pos.clone()
        moved.x = torch.zeros(31, 2, dtype=F64)
        moved.num_nodes = 31
        query = model(data.to(device))[1]
        node_at_queries = model(moved.to(device))[0]
        graphs = [_random_graph(n, q, s) for s, (n, q) in enumerate([(9, 4), (14, 7)])]
        batch = Batch.from_data_list(graphs)
        batch.query_graph = torch.repeat_interleave(
            torch.arange(2), torch.tensor([4, 7])
        )
        batched = model(batch.to(device))
        singles = [model(g.to(device)) for g in graphs]
        batch_err = max(
            _max_abs(batched[h], torch.cat([s[h] for s in singles])) for h in (0, 1)
        )
    report = {
        "query_vs_node_at_same_points": _max_abs(query, node_at_queries),
        "batched_vs_single": batch_err,
    }
    return report, all(v < 1e-12 for v in report.values())


def sensor_error_checks(device):
    model = DeepONet(2, 8, [1], ["node"], 2, num_sensors=4, basis_dim=4)
    model = model.to(device, F64)
    cases = {}

    def expect(name, data, match):
        try:
            model(data.to(device))
        except ValueError as err:
            cases[name] = match in str(err)
        else:
            cases[name] = False

    bad = _random_graph(6, 0, 0)
    bad.deeponet_sensors = bad.deeponet_sensors[:, :-1]
    expect("wrong_sensor_width", bad, "deeponet_sensors must have shape")
    bad = _random_graph(6, 0, 0)
    del bad.deeponet_sensors
    expect("x_wrong_node_count", bad, "exactly deeponet_num_sensors")
    bad = _random_graph(4, 0, 0, c=1)
    del bad.deeponet_sensors
    expect("x_wrong_channels", bad, "channels")
    bad = Batch.from_data_list([_random_graph(5, 2, 0), _random_graph(5, 2, 1)])
    query_model = DeepONet(2, 8, [1], ["query"], 2, num_sensors=4, basis_dim=4)
    try:
        query_model.to(device, F64)(bad.to(device))
        cases["batched_query_without_graph_ids"] = False
    except ValueError as err:
        cases["batched_query_without_graph_ids"] = "query_graph" in str(err)
    return cases, all(cases.values())


def gradcheck_check(device):
    torch.manual_seed(0)
    model = DeepONet(
        2,
        6,
        [1, 1],
        ["node", "query"],
        2,
        num_sensors=4,
        basis_dim=4,
        multi_output_strategy="split_both",
    ).to(device, F64)
    data = _random_graph(5, 3, 0).to(device)

    def f(sensors, query_pos):
        data.deeponet_sensors, data.query_pos = sensors, query_pos
        return tuple(model(data))

    inputs = (
        data.deeponet_sensors.clone().requires_grad_(),
        data.query_pos.clone().requires_grad_(),
    )
    ok = bool(torch.autograd.gradcheck(f, inputs))
    return {"passed": ok}, ok


def deeponet_checks(device, steps, seed):
    report, ok = {}, True
    report["antiderivative"] = antiderivative_check(device, steps, seed)
    ok &= report["antiderivative"]["ok"]
    for name, fn in (
        ("strategies", strategy_checks),
        ("queries", query_checks),
        ("sensor_errors", sensor_error_checks),
        ("gradcheck", gradcheck_check),
    ):
        report[name], passed = fn(device)
        ok &= passed
    return report, ok


# -- Composite ---------------------------------------------------------------
PIPELINES = {
    "mgn_gsno": (
        {"type": "NodeEncoder"},
        [
            {"type": "LocalMessagePassing", "layers": 3},
            {"type": "GraphSpectral", "layers": 2, "modes": 8, "skip": True},
        ],
        ["node", "cell", "query"],
        None,
    ),
    "regional_flux": (
        {"type": "RegionalGraphEncoder"},
        [
            {"type": "RegionalMessagePassing", "layers": 3},
            {"type": "RegionToNode"},
            {"type": "LocalMessagePassing", "layers": 2, "skip": True},
        ],
        ["node", "face"],
        {"face": "ConservativeFluxDecoder"},
    ),
}


def _mesh(n, seed, queries=0):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(n, n, dtype=F64)
    mesh.x = torch.randn(mesh.num_nodes, 2, generator=g, dtype=F64)
    if queries:
        locate_queries(mesh, torch.rand(queries, 2, generator=g, dtype=F64))
    mesh = build_rigno_graph(mesh, generator=g)
    attach_graph_spectral_basis(mesh, 10)
    return mesh


def _composite(name, device):
    encoder, processors, levels, decoders = PIPELINES[name]
    torch.manual_seed(0)
    model = CompositeNeuralOperator(
        2,
        16,
        [1] * len(levels),
        levels,
        2,
        encoder=encoder,
        processors=processors,
        decoders=decoders,
        activation_function="silu",
    )
    return model.to(device, F64), levels


def composite_checks(device, steps):
    report, ok = {}, True
    for name in PIPELINES:
        model, levels = _composite(name, device)
        mesh = _mesh(8, 0, queries=8).to(device)
        target = torch.sin(math.pi * mesh.pos[:, :1]) * torch.sin(
            math.pi * mesh.pos[:, 1:]
        )
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = ((model(mesh)[0] - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        model.eval()
        with torch.no_grad():
            meshes = [_mesh(5, 1, queries=6), _mesh(6, 2, queries=4)]
            batched = model(Batch.from_data_list(meshes).to(device))
            singles = [model(m.to(device)) for m in meshes]
            batch_err = max(
                _max_abs(b, torch.cat(s)) for b, *s in zip(batched, *singles)
            )
        entry = {
            "levels": levels,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "batched_vs_single": batch_err,
        }
        passed = losses[-1] < 0.1 * losses[0] and batch_err < 1e-12
        if "face" in levels:
            with torch.no_grad():
                flipped = copy.copy(mesh)
                flipped.face_index = mesh.face_index.flip(0)
                flipped.face_nodes = mesh.face_nodes.flip(0)
                flipped.face_normal = -mesh.face_normal
                face = model(mesh)[1]
                entry["face_antisymmetry"] = _max_abs(model(flipped)[1], -face)
            passed &= entry["face_antisymmetry"] < 1e-12
        entry["ok"] = passed
        report[name] = entry
        ok &= passed

    rejections = {
        "missing_region_to_node": (
            {"type": "RegionalGraphEncoder"},
            [{"type": "RegionalMessagePassing"}],
            "add a RegionToNode adapter",
        ),
        "entity_mismatch": (
            {"type": "NodeEncoder"},
            [{"type": "RegionToNode"}],
            "reads region latents",
        ),
        "unknown_type": ({"type": "NodeEncoder"}, [{"type": "Attention"}], "unknown"),
        "width_mismatch": (
            {"type": "NodeEncoder"},
            [{"type": "GraphSpectral", "hidden_dim": 32}],
            "insert a Projection",
        ),
        "skip_on_adapter": (
            {"type": "NodeEncoder"},
            [{"type": "Projection", "hidden_dim": 16, "skip": True}],
            "skip is only allowed",
        ),
    }
    results = {}
    for key, (encoder, processors, match) in rejections.items():
        try:
            resolve_pipeline(encoder, processors, None, ["node"], 16, 2)
            results[key] = False
        except ValueError as err:
            results[key] = match in str(err)
    try:
        resolve_pipeline(
            {"type": "RegionalGraphEncoder"},
            [{"type": "RegionToNode"}],
            None,
            ["node"],
            16,
            3,
        )
        results["regional_in_3d"] = False
    except ValueError as err:
        results["regional_in_3d"] = "spatial_dim" in str(err)
    report["rejections"] = results
    ok &= all(results.values())
    return report, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--composite-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    results, failed = {}, []
    for name, fn, extra in (
        ("deeponet", deeponet_checks, (args.steps, args.seed)),
        ("composite", composite_checks, (args.composite_steps,)),
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
        "composite_steps": args.composite_steps,
        "wall_seconds": time.perf_counter() - t0,
    }
    print(json.dumps(results, indent=2, default=float))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
