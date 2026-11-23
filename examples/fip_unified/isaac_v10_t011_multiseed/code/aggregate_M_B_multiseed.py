"""aggregate_M_B_multiseed.py — pool M_B (μ_log + σ_log) across seeds.

Reads run_metrics.json from M_B_logmu_logsigma_seed{42,43,44,45,46} run dirs
and reports mean ± std for the key per-cell metrics:

  - HCF top-1 hit rate (paper headline)
  - σ_log R² per cell
  - μ_log R² per cell
  - max R² per cell
  - 4-cell mean top-1

Decision criterion: M_B σ_log R² > 0.5 with std < 0.15 in HCF cells →
robust enough to promote M_B to v9b base.
"""
from __future__ import annotations
from fip_release_paths import release_root
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
RESULTS = Path(str(release_root() / "examples/fip_unified/isaac_v9b_study/results"))
CELLS = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]


def _r2_aux(test, name):
    aux = test.get("pergrain_aux", {})
    if name not in aux:
        return float("nan")
    m = aux[name]
    if isinstance(m, dict):
        mt = m.get("metrics") or {}
        return float(mt.get("r2", float("nan")))
    return float("nan")


def _r2_slot0(test, expect_name="lcf_fs_max"):
    pn0 = test.get("pergrain") or {}
    if isinstance(pn0, dict):
        return float(pn0.get("r2", float("nan")))
    return float("nan")


def _top1_per_cell(test):
    h = test.get("pergrain_per_cell_hotspot") or {}
    out = {}
    for c in CELLS:
        v = h.get(f"{c}_max") or {}
        out[c] = float(v.get("top1_hit_rate", float("nan")))
    return out


def _summary(metric_label, values_per_seed):
    arr = np.array([v for v in values_per_seed if v == v], dtype=float)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "n": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--tag", type=str, default="M_B_logmu_logsigma")
    args = ap.parse_args()
    metrics = []
    seeds_present = []
    for s in args.seeds:
        rp = RESULTS / f"{args.tag}_seed{s}" / "run_metrics.json"
        if not rp.exists():
            print(f"  [skip] seed {s}: no run_metrics.json")
            continue
        m = json.load(open(rp))
        metrics.append((s, m))
        seeds_present.append(s)
    if not metrics:
        print("[no seeds found]")
        return 1
    print()
    print(f" v9b Phase M_B multi-seed aggregate — n_seeds = {len(metrics)}")
    print(f" seeds present: {seeds_present}")
    print("=" * 110)
    print()
    print(" Per-cell top-1 hit rate")
    print("  " + "-" * 86)
    print(
        f"  {'cell':<10s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  per-seed"
    )
    print("  " + "-" * 86)
    for c in CELLS:
        per_seed = []
        for _, m in metrics:
            t1 = _top1_per_cell(m["test"])[c]
            per_seed.append(t1)
        s = _summary(f"top1_{c}", per_seed)
        ps = " ".join((f"{v:.3f}" for v in per_seed))
        print(
            f"  {c:<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {ps}"
        )
    fc = []
    for _, m in metrics:
        v = m["test"].get("pergrain_top1_hit_rate_mean_4cell")
        if v is not None:
            fc.append(float(v))
    s = _summary("top1_4cell_mean", fc)
    print(
        f"  {'4cell_mn':<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {' '.join((f'{v:.3f}' for v in fc))}"
    )
    hcf = [
        (_top1_per_cell(m["test"])["hcf_fs"] + _top1_per_cell(m["test"])["hcf_wp"])
        / 2.0
        for _, m in metrics
    ]
    s = _summary("HCF_mean_top1", hcf)
    print(
        f"  {'HCF_mean':<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {' '.join((f'{v:.3f}' for v in hcf))}"
    )
    print()
    print(" Per-cell σ_log (lognormal_sigma) R² — DISTRIBUTION-SHAPE TARGET")
    print("  " + "-" * 86)
    for c in CELLS:
        per_seed = []
        for _, m in metrics:
            r2 = _r2_aux(m["test"], f"{c}_logsigma_voxel")
            per_seed.append(r2)
        s = _summary(f"r2_logsigma_{c}", per_seed)
        ps = " ".join((f"{v:+.3f}" for v in per_seed))
        marker = (
            "  *** STRONG ***"
            if s["mean"] > 0.7
            else "  *** USABLE ***"
            if s["mean"] > 0.5
            else ""
        )
        print(
            f"  {c:<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {ps}{marker}"
        )
    print()
    print(" Per-cell μ_log (lognormal_mu) R² — DISTRIBUTION-SHAPE TARGET")
    print("  " + "-" * 86)
    for c in CELLS:
        per_seed = []
        for _, m in metrics:
            r2 = _r2_aux(m["test"], f"{c}_logmu_voxel")
            per_seed.append(r2)
        s = _summary(f"r2_logmu_{c}", per_seed)
        ps = " ".join((f"{v:+.3f}" for v in per_seed))
        print(
            f"  {c:<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {ps}"
        )
    print()
    print(" Per-cell MAX R² (paper main outcome — vs L6b2 baseline)")
    print("  " + "-" * 86)
    for c in CELLS:
        per_seed = []
        for _, m in metrics:
            if c == "lcf_fs":
                r2 = _r2_slot0(m["test"])
            else:
                r2 = _r2_aux(m["test"], f"{c}_max")
            per_seed.append(r2)
        s = _summary(f"r2_max_{c}", per_seed)
        ps = " ".join((f"{v:+.3f}" for v in per_seed))
        print(
            f"  {c:<10s}  {s['mean']:>+8.3f}  {s['std']:>8.3f}  {s['min']:>+8.3f}  {s['max']:>+8.3f}  {ps}"
        )
    summary = {
        "tag": args.tag,
        "seeds": seeds_present,
        "n_seeds": len(metrics),
        "per_cell_top1": {
            c: _summary(f"top1_{c}", [_top1_per_cell(m["test"])[c] for _, m in metrics])
            for c in CELLS
        },
        "per_cell_r2_max": {
            c: _summary(
                f"r2_max_{c}",
                [
                    _r2_slot0(m["test"])
                    if c == "lcf_fs"
                    else _r2_aux(m["test"], f"{c}_max")
                    for _, m in metrics
                ],
            )
            for c in CELLS
        },
        "per_cell_r2_logmu_voxel": {
            c: _summary(
                f"r2_logmu_{c}",
                [_r2_aux(m["test"], f"{c}_logmu_voxel") for _, m in metrics],
            )
            for c in CELLS
        },
        "per_cell_r2_logsigma_voxel": {
            c: _summary(
                f"r2_logsigma_{c}",
                [_r2_aux(m["test"], f"{c}_logsigma_voxel") for _, m in metrics],
            )
            for c in CELLS
        },
    }
    out = RESULTS / f"{args.tag}_multiseed_aggregate.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  [saved] {out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
