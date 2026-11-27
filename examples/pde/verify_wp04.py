"""Numerical verification of WP04 (Multiscale MeshGraphNet and RIGNO).

Property checks (FP64, random weights, every one compared with ``--tol``):

* ``msmgn_partition_of_unity``: max ``|sum_j w_ij - 1|`` of the down/up
  barycentric weights.
* ``msmgn_constant_preservation`` / ``msmgn_linear_exactness``: max error of
  ``prolong_nodes``/``restrict_nodes`` on constant and linear fields.
* ``msmgn_hierarchy_indices``: max ``|sum_j w_ij p_j - p_i|``, i.e. every
  node is reproduced from the corners it is linked to; negative weights
  (a corner outside the containing triangle) are counted as errors too.
* ``msmgn_batched_vs_single`` / ``rigno_batched_vs_single``.
* ``rigno_mask_reproducibility``: training-mode outputs with edge masking
  under the same ``torch.manual_seed``; ``rigno_eval_deterministic``.
* ``rigno_query_subset``: outputs at a subset of query points minus the
  same points evaluated together with the others.
* ``rigno_resolution_transfer``: one model on input meshes ``n`` = 8, 12, 20
  with query sets of 5, 50 and 500 points; non-finite outputs fail.

With ``--scaling`` the script also times one forward + backward pass of
MeshGraphNet, MultiscaleMeshGraphNet and RIGNO (hyperparameters of the
``poisson_*.json`` presets) on unit-square meshes of growing size, and
reports graph-construction time and, on GPUs, peak allocated memory.
"""

import argparse
import json
import math
import platform
import sys
import time

import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import (
    build_hierarchy,
    build_rigno_graph,
    locate_queries,
    prolong_nodes,
    restrict_nodes,
    unit_square_mesh,
)
from hydragnn.models.create import create_model

DT = torch.float64


def _args(mpnn_type, input_dim, **overrides):
    args = dict(
        mpnn_type=mpnn_type,
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads={"node": {"num_headlayers": 1, "dim_headlayers": [64]}},
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=8,
        use_gpu=False,
    )
    if mpnn_type in ("MeshGraphNet", "MultiscaleMeshGraphNet"):
        args.update(mgn_spatial_dim=2, mgn_mlp_layers=2)
    if mpnn_type == "MultiscaleMeshGraphNet":
        args["msmgn_schedule"] = "2H 2L 2H"
    if mpnn_type == "RIGNO":
        args.update(
            activation_function="silu",
            num_conv_layers=6,
            rigno_mlp_layers=2,
            rigno_edge_masking=0.5,
        )
    args.update(overrides)
    return args


def _model(mpnn_type, input_dim, seed=0, **overrides):
    torch.manual_seed(seed)
    return create_model(**_args(mpnn_type, input_dim, **overrides))


def _hierarchy(n, seed, dtype=DT):
    h = build_hierarchy(
        unit_square_mesh(n, n), unit_square_mesh(n // 2, n // 2), node_transfer=True
    )
    for key, value in h.items():
        if value.is_floating_point():
            h[key] = value.to(dtype)
    g = torch.Generator().manual_seed(seed)
    h.x = torch.randn(h.num_nodes, 3, generator=g, dtype=dtype)
    return h


def _rigno(n, seed, queries=0, dtype=DT):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(n, n)
    mesh.x = torch.randn(mesh.num_nodes, 3, generator=g, dtype=DT)
    if queries:
        locate_queries(mesh, torch.rand(queries, 2, generator=g, dtype=DT))
    data = build_rigno_graph(mesh, generator=g)
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype)
    return data


def _maxdiff(a, b):
    return float((a - b).detach().abs().max())


def verify_msmgn(device):
    h = _hierarchy(8, 0)
    report = {"msmgn_partition_of_unity": 0.0, "msmgn_hierarchy_indices": 0.0}
    for index, weight, n, pos_src, pos_dst in (
        (h.down_index, h.down_weight, h.num_nodes, h.pos, h.coarse_pos),
        (h.up_index, h.up_weight, h.num_coarse_nodes, h.coarse_pos, h.pos),
    ):
        total = torch.zeros(n, dtype=DT).index_add_(0, index[0], weight)
        report["msmgn_partition_of_unity"] = max(
            report["msmgn_partition_of_unity"], _maxdiff(total, torch.ones_like(total))
        )
        rebuilt = torch.zeros(n, 2, dtype=DT).index_add_(
            0, index[0], weight[:, None] * pos_dst[index[1]]
        )
        negative = float((-weight).clamp(min=0).max())
        report["msmgn_hierarchy_indices"] = max(
            report["msmgn_hierarchy_indices"], _maxdiff(rebuilt, pos_src), negative
        )

    const = torch.full((h.num_coarse_nodes, 3), -1.75, dtype=DT)
    fine_const = torch.full((h.num_nodes, 3), -1.75, dtype=DT)
    report["msmgn_constant_preservation"] = max(
        _maxdiff(prolong_nodes(h, const), fine_const),
        _maxdiff(restrict_nodes(h, fine_const), const),
    )

    def linear(p):
        return 0.3 + 2.0 * p[:, :1] - 1.5 * p[:, 1:2]

    report["msmgn_linear_exactness"] = max(
        _maxdiff(prolong_nodes(h, linear(h.coarse_pos)), linear(h.pos)),
        _maxdiff(restrict_nodes(h, linear(h.pos)), linear(h.coarse_pos)),
    )

    model = _model("MultiscaleMeshGraphNet", 3).double().to(device).eval()
    meshes = [_hierarchy(n, n) for n in (4, 6, 10)]
    with torch.no_grad():
        batched = model(Batch.from_data_list(meshes).to(device))[0]
        singles = torch.cat([model(m.to(device))[0] for m in meshes])
    report["msmgn_batched_vs_single"] = _maxdiff(batched, singles)
    return report


def verify_rigno(device):
    report = {}
    model = _model(
        "RIGNO",
        3,
        output_dim=[1, 1],
        output_type=["node", "query"],
        task_weights=[1.0, 1.0],
        output_heads={
            "node": {"num_headlayers": 1, "dim_headlayers": [64]},
            "query": {"num_headlayers": 1, "dim_headlayers": [64]},
        },
    )
    model = model.double().to(device)
    data = _rigno(10, 0, queries=30).to(device)

    model.train()
    torch.manual_seed(7)
    a = model(data)
    torch.manual_seed(7)
    b = model(data)
    torch.manual_seed(8)
    c = model(data)
    report["rigno_mask_reproducibility"] = max(_maxdiff(x, y) for x, y in zip(a, b))
    other_seed = min(_maxdiff(x, y) for x, y in zip(a, c))

    model.eval()
    with torch.no_grad():
        report["rigno_eval_deterministic"] = max(
            _maxdiff(x, y) for x, y in zip(model(data), model(data))
        )
        full = model(data)[1]
        keep = torch.tensor([0, 3, 4, 17, 29], device=device)
        subset = data.clone()
        subset.rigno_qx = data.rigno_qx[keep]
        subset.rigno_qgraph = data.rigno_qgraph[keep]
        mask = torch.isin(data.rigno_r2q_index[1], keep)
        remap = torch.full((data.rigno_qx.shape[0],), -1, device=device)
        remap[keep] = torch.arange(keep.numel(), device=device)
        index = data.rigno_r2q_index[:, mask].clone()
        index[1] = remap[index[1]]
        subset.rigno_r2q_index = index
        report["rigno_query_subset"] = _maxdiff(model(subset)[1], full[keep])

        graphs = [_rigno(n, n, queries=7) for n in (6, 9, 13)]
        batched = model(Batch.from_data_list(graphs).to(device))
        singles = [model(g.to(device)) for g in graphs]
        report["rigno_batched_vs_single"] = max(
            _maxdiff(batched[i], torch.cat([s[i] for s in singles])) for i in (0, 1)
        )

        finite = True
        for n in (8, 12, 20):
            for q in (5, 50, 500):
                node, query = model(_rigno(n, n + q, queries=q).to(device))
                finite &= bool(
                    torch.isfinite(node).all() and torch.isfinite(query).all()
                )
                finite &= node.shape[0] == (n + 1) ** 2 and query.shape[0] == q
        report["rigno_resolution_transfer"] = 0.0 if finite else math.inf
    return report, {"rigno_mask_other_seed_min_diff": other_seed}


def _message_edges(name, data):
    if name == "MeshGraphNet":
        keys = ["edge_index"]
    elif name == "MultiscaleMeshGraphNet":
        keys = ["edge_index", "coarse_edge_index", "down_index", "up_index"]
    else:
        keys = ["rigno_p2r_index", "rigno_r2r_index", "rigno_r2p_index"]
    return sum(int(data[k].shape[1]) for k in keys)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_step(model, data, device, repeats):
    """Median wall time of forward + backward, and peak memory on GPU."""
    model.train()
    times = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        _sync(device)
        start = time.perf_counter()
        model(data)[0].pow(2).mean().backward()
        _sync(device)
        if i:  # first pass is warm-up
            times.append(time.perf_counter() - start)
    times.sort()
    peak = (
        torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
    )
    return times[len(times) // 2], peak


def scaling(device, sizes, repeats):
    rows = []
    models = {
        "MeshGraphNet": _model("MeshGraphNet", 3).float().to(device),
        "MultiscaleMeshGraphNet": _model("MultiscaleMeshGraphNet", 3)
        .float()
        .to(device),
        "RIGNO": _model("RIGNO", 3).float().to(device),
    }
    for n in sizes:
        start = time.perf_counter()
        flat = unit_square_mesh(n, n, dtype=torch.float32)
        flat.x = torch.randn(flat.num_nodes, 3)
        t_flat = time.perf_counter() - start
        start = time.perf_counter()
        hier = _hierarchy(n, 0, dtype=torch.float32)
        t_hier = time.perf_counter() - start
        start = time.perf_counter()
        rig = _rigno(n, 0, dtype=torch.float32)
        t_rig = time.perf_counter() - start
        for name, data, t_build in (
            ("MeshGraphNet", flat, t_flat),
            ("MultiscaleMeshGraphNet", hier, t_hier),
            ("RIGNO", rig, t_rig),
        ):
            torch.manual_seed(0)
            seconds, peak = _time_step(models[name], data.to(device), device, repeats)
            rows.append(
                {
                    "model": name,
                    "n": n,
                    "nodes": int(flat.num_nodes),
                    "message_edges": _message_edges(name, data),
                    "build_s": t_build,
                    "fwd_bwd_s": seconds,
                    "peak_mib": peak,
                }
            )
            print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 96])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    start = time.perf_counter()
    report = verify_msmgn(args.device)
    rigno_report, info = verify_rigno(args.device)
    report.update(rigno_report)
    failed = sorted(k for k, v in report.items() if not v <= args.tol)
    # Different seeds must give different masks, otherwise the check is vacuous.
    if not info["rigno_mask_other_seed_min_diff"] > 1e-6:
        failed.append("rigno_mask_other_seed_min_diff")
    out = {
        "report": report,
        "info": info,
        "tol": args.tol,
        "failed": failed,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name() if args.device.startswith("cuda") else None
        ),
        "precision": "fp64 checks, fp32 scaling",
        "seed": 0,
        "check_wall_s": time.perf_counter() - start,
    }
    if args.scaling:
        out["scaling"] = scaling(args.device, args.sizes, args.repeats)
    print(json.dumps(out, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
