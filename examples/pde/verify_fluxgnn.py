"""Numerical verification of FluxGNN's conservation and symmetry properties.

Runs in FP64 on batched triangulated unit squares with random weights (all
learned terms active) and prints a JSON report. The properties hold by
construction for any weights; the trained transport model is checked by
``fv_transport.py`` (``max_rel_balance_residual``).


* ``interior_cancellation``: ``|sum_i net_outflow_i|`` for random interior
  fluxes, relative to ``sum |A F|``.
* ``balance_learned_source`` / ``balance_closed``: worst per-graph relative
  discrete balance residual of the model's own fluxes and sources.
* ``orientation_flip``: change of the cell update and of ``-face_flux`` when
  every interior face is reversed.
* ``rotation`` / ``reflection``: change of outputs under a rigid motion of
  positions, normals, centroids and vector inputs (plus translation).
* ``batched_vs_single``: batched minus per-mesh outputs.
* ``rollout_balance``: worst balance residual over a 20-step autoregressive
  rollout.

All errors are maxima of absolute differences unless stated otherwise; each
is compared with ``--tol`` and the script exits non-zero if any exceeds it.
"""

import argparse
import json
import math
import platform
import sys

import torch
from torch_geometric.data import Batch

from hydragnn.data.meshes import unit_square_mesh
from hydragnn.models.fluxgnn import FluxGNN
from hydragnn.models.rollout import rollout
from hydragnn.physics import balance_residual, net_outflow

LAYOUT = {"state": [0, 1], "vectors": [[2, 3]], "scalars": [4]}
DT = 0.01


def make_mesh(n, seed):
    g = torch.Generator().manual_seed(seed)
    mesh = unit_square_mesh(n, n + 2, dtype=torch.float64)
    c = mesh.num_cells
    mesh.cell_x = torch.cat(
        [
            torch.rand(c, 2, generator=g, dtype=torch.float64),
            torch.randn(c, 2, generator=g, dtype=torch.float64),
            torch.rand(c, 1, generator=g, dtype=torch.float64),
        ],
        dim=1,
    )
    mesh.bface_tag = torch.randint(0, 2, (mesh.bface_owner.numel(),), generator=g)
    return mesh


def make_model(boundary, source, seed):
    torch.manual_seed(seed)
    model = FluxGNN(
        5,
        16,
        [2, 2],
        ["cell", "face"],
        4,
        layout=LAYOUT,
        dt=DT,
        num_bface_tags=2,
        boundary=boundary,
        source=source,
    ).double()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(std=0.3)
            else:
                param.uniform_(-0.2, 0.2)
    return model.eval()


def transform(mesh, angle, reflect):
    c, s = math.cos(angle), math.sin(angle)
    kw = dict(dtype=torch.float64, device=mesh.pos.device)
    R = torch.tensor([[c, -s], [s, c]], **kw)
    if reflect:
        R = R @ torch.diag(torch.tensor([1.0, -1.0], **kw))
    out = mesh.clone()
    shift = torch.tensor([2.0, -3.0], **kw)
    out.pos = mesh.pos @ R.T + shift
    out.cell_centroid = mesh.cell_centroid @ R.T + shift
    out.face_normal = mesh.face_normal @ R.T
    out.bface_normal = mesh.bface_normal @ R.T
    x = mesh.cell_x.clone()
    x[:, 2:4] = mesh.cell_x[:, 2:4] @ R.T
    out.cell_x = x
    return out


def max_diff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def worst_balance(model, data):
    terms = model.fluxes(data)
    residual, scale = balance_residual(
        data,
        data.cell_x[:, model.state_cols],
        terms["u_new"],
        terms["bface_flux"],
        terms["source"],
        model.dt,
    )
    return float((residual.abs() / scale).max())


@torch.no_grad()
def verify(seed=0, device="cpu"):
    meshes = [make_mesh(n, seed + n).to(device) for n in (4, 6, 9)]
    batch = Batch.from_data_list(meshes)
    model = make_model("learned", True, seed).to(device)
    closed = make_model("closed", False, seed).to(device)

    flux = torch.randn(batch.num_faces, 3, dtype=torch.float64, device=device)
    cancel = float(net_outflow(batch, flux).sum(0).abs().max()) / float(
        (batch.face_area * flux).abs().sum()
    )

    single = meshes[1]
    ref = model(single)
    flipped = single.clone()
    flipped.face_index = single.face_index.flip(0)
    flipped.face_normal = -single.face_normal
    flip = model(flipped)

    batched = model(batch)
    singles = [model(m) for m in meshes]
    batched_err = max_diff(
        batched, [torch.cat([s[i] for s in singles]) for i in range(2)]
    )

    states = rollout(model, single, 20, model.state_cols, level="cell")
    data, roll_balance = single.clone(), 0.0
    for state in states:
        roll_balance = max(roll_balance, worst_balance(model, data))
        x = data.cell_x.clone()
        x[:, model.state_cols] = state
        data.cell_x = x

    return {
        "interior_cancellation": cancel,
        "balance_learned_source": worst_balance(model, batch),
        "balance_closed": worst_balance(closed, batch),
        "orientation_flip": max(
            float((flip[0] - ref[0]).abs().max()), float((flip[1] + ref[1]).abs().max())
        ),
        "rotation": max_diff(model(transform(single, 0.9, False)), ref),
        "reflection": max_diff(model(transform(single, 0.4, True)), ref),
        "batched_vs_single": batched_err,
        "rollout_balance": roll_balance,
        "rollout_final_abs_max": float(states[-1].abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = verify(args.seed, args.device)
    checked = {k: v for k, v in report.items() if k != "rollout_final_abs_max"}
    failed = sorted(k for k, v in checked.items() if not v <= args.tol)
    print(
        json.dumps(
            {
                "report": report,
                "tol": args.tol,
                "failed": failed,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "seed": args.seed,
                "device": args.device,
                "precision": "fp64",
            },
            indent=2,
        )
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
