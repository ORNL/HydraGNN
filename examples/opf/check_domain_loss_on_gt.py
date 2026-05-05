"""Check whether domain loss penalties are zero on ground-truth OPF solutions.

Loads N raw JSON samples, substitutes bus.y (ground truth Va, Vm) as the
"prediction", and evaluates each penalty term.  A correctly-posed feasibility
penalty should return exactly zero (or near-zero up to float precision) on
any strictly feasible OPF solution.

Usage:
    python3 check_domain_loss_on_gt.py [--n_samples N] [--json_dir PATH]
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=20)
parser.add_argument(
    "--json_dir",
    default=(
        "dataset/dataset_release_1/pglib_opf_case10000_goc/raw/"
        "gridopt-dataset-tmp/dataset_release_1/pglib_opf_case10000_goc/group_1"
    ),
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Minimal HeteroData builder (same logic as _raw_json_to_heterodata)
# ---------------------------------------------------------------------------
try:
    import torch_geometric.datasets.opf as tg_opf
    from torch_geometric.data import HeteroData
except ImportError:
    sys.exit("torch_geometric not available")


def load_sample(filepath):
    with open(filepath) as f:
        obj = json.load(f)
    grid = obj["grid"]
    solution = obj["solution"]

    data = HeteroData()
    data["bus"].x = torch.tensor(grid["nodes"]["bus"], dtype=torch.float)
    data["bus"].y = torch.tensor(solution["nodes"]["bus"], dtype=torch.float)

    def _ei(obj, rel):
        try:
            return tg_opf.extract_edge_index(obj, rel)
        except Exception:
            return None

    for rel in ("ac_line", "transformer"):
        ei = _ei(obj, rel)
        if ei is None:
            continue
        ea = grid["edges"].get(rel, {}).get("features")
        if ea is None:
            continue
        data["bus", rel, "bus"].edge_index = ei
        data["bus", rel, "bus"].edge_attr = torch.tensor(ea, dtype=torch.float)

    return data


# ---------------------------------------------------------------------------
# Penalty evaluation on ground-truth bus.y
# ---------------------------------------------------------------------------
VMIN_IDX, VMAX_IDX = 2, 3   # bus.x columns for v_min, v_max
VM_IDX = 1                   # bus.y column for Vm
VA_IDX = 0                   # bus.y column for Va

# ac_line edge_attr indices
AC_X_IDX, AC_RATE_A_IDX = 5, 6
AC_THETA_MIN_IDX, AC_THETA_MAX_IDX = 0, 1

# transformer edge_attr indices
TR_X_IDX, TR_RATE_A_IDX = 3, 4
TR_THETA_MIN_IDX, TR_THETA_MAX_IDX = 0, 1


def evaluate(data):
    bus_true = data["bus"].y   # shape [N_bus, 2]  (Va, Vm)
    bus_x    = data["bus"].x   # shape [N_bus, 4]

    results = {}

    # 1. Voltage bound: vmin <= Vm <= vmax
    Vm    = bus_true[:, VM_IDX]
    vmin  = bus_x[:, VMIN_IDX]
    vmax  = bus_x[:, VMAX_IDX]
    vbound = torch.mean(F.relu(vmin - Vm).pow(2) + F.relu(Vm - vmax).pow(2))
    results["voltage_bound"] = vbound.item()

    # 2. Angle difference limit: theta_min <= Va_i - Va_j <= theta_max
    Va = bus_true[:, VA_IDX]
    for rel, tmin_i, tmax_i, tag in [
        (("bus", "ac_line",      "bus"), AC_THETA_MIN_IDX, AC_THETA_MAX_IDX, "ac_angle_diff"),
        (("bus", "transformer",  "bus"), TR_THETA_MIN_IDX, TR_THETA_MAX_IDX, "tr_angle_diff"),
    ]:
        if rel not in data.edge_types:
            continue
        ea = getattr(data[rel], "edge_attr", None)
        ei = getattr(data[rel], "edge_index", None)
        if ea is None or ei is None or ea.shape[1] <= max(tmin_i, tmax_i):
            continue
        theta_min = ea[:, tmin_i]
        theta_max = ea[:, tmax_i]
        src, dst = ei
        delta = Va[src] - Va[dst]
        p = torch.mean(F.relu(delta - theta_max).pow(2) + F.relu(theta_min - delta).pow(2))
        results[tag] = p.item()

    # 3. DC thermal limit: |P_ij| = |(Va_i - Va_j) / x_ij| <= rate_a
    for rel, x_i, ra_i, tag in [
        (("bus", "ac_line",     "bus"), AC_X_IDX,  AC_RATE_A_IDX,  "ac_line_flow"),
        (("bus", "transformer", "bus"), TR_X_IDX,  TR_RATE_A_IDX,  "tr_line_flow"),
    ]:
        if rel not in data.edge_types:
            continue
        ea = getattr(data[rel], "edge_attr", None)
        ei = getattr(data[rel], "edge_index", None)
        if ea is None or ei is None or ea.shape[1] <= max(x_i, ra_i):
            continue
        x_ij   = ea[:, x_i].clamp(min=1e-6)
        rate_a = ea[:, ra_i].clamp(min=0.0)
        src, dst = ei
        P_ij = (Va[src] - Va[dst]) / x_ij
        p = torch.mean(F.relu(P_ij.abs() - rate_a).pow(2))
        results[tag] = p.item()

    return results


# ---------------------------------------------------------------------------
# Run over N samples
# ---------------------------------------------------------------------------
json_dir = os.path.join(os.path.dirname(__file__), args.json_dir)
files = sorted(
    [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
)[: args.n_samples]

if not files:
    sys.exit(f"No JSON files found in {json_dir}")

print(f"Evaluating {len(files)} samples from {json_dir}\n")

accum = {}
n_ok = 0
for path in files:
    try:
        data = load_sample(path)
        r = evaluate(data)
        for k, v in r.items():
            accum.setdefault(k, []).append(v)
        n_ok += 1
    except Exception as e:
        print(f"  WARNING: {os.path.basename(path)}: {e}")

print(f"{'Term':<25}  {'mean':>12}  {'max':>12}  {'min':>12}  {'#non-zero':>10}")
print("-" * 75)
eps = 1e-9
for term in sorted(accum):
    vals = accum[term]
    mean_v = sum(vals) / len(vals)
    max_v  = max(vals)
    min_v  = min(vals)
    nonzero = sum(1 for v in vals if abs(v) > eps)
    print(f"{term:<25}  {mean_v:>12.6e}  {max_v:>12.6e}  {min_v:>12.6e}  {nonzero:>10}/{len(vals)}")

print(f"\nProcessed {n_ok}/{len(files)} samples successfully.")
