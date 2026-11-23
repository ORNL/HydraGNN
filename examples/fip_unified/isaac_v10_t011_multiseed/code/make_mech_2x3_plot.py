"""make_mech_2x3_plot.py — 2×3 physics-grouped mech QoI parity & dist plots.

Test split only. Compact paper-grade layout:
  Row 1 (stress)              = [yield_stress, sigma_amp_lcf, sigma_amp_hcf]
  Row 2 (energy / hardening)  = [hardening_modulus, Wp_lcf,  Wp_hcf]
  Cols                        = [monotonic, LCF, HCF]

Reads `pred_test_pg.csv` (the wide v9b pg format). Replaces the prior
6×3 = 18-panel mech grid with a 2×3 = 6-panel grid.

Usage:
  python make_mech_2x3_plot.py                                      # all v9b L6* dirs
  python make_mech_2x3_plot.py --tag L6b2_het_nll_both_50_15        # one dir
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

from plots import parity_grid_plot, dist_overlay_grid_plot

LAYOUT = [
    ("yield_stress", "monotonic"),
    ("sigma_amp_lcf", "LCF"),
    ("sigma_amp_hcf", "HCF"),
    ("hardening_modulus", "monotonic"),
    ("Wp_lcf", "LCF"),
    ("Wp_hcf", "HCF"),
]
ROW_LABELS = ["stress", "energy / hardening"]
COL_LABELS = ["monotonic", "LCF", "HCF"]


def _column_arr(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            out.append(np.nan)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
    return np.array(out)


def make_for_run(run_dir: Path) -> bool:
    p = run_dir / "pred_test_pg.csv"
    if not p.exists():
        print(f"  [skip] {run_dir.name}: no pred_test_pg.csv")
        return False
    rows = list(csv.DictReader(open(p)))
    sb = _column_arr(rows, "size_bucket").astype(int).tolist()
    panels = []
    for q, _regime in LAYOUT:
        col_t = f"true_{q}"
        col_p = f"pred_{q}"
        if col_t not in rows[0] or col_p not in rows[0]:
            panels.append(
                {
                    "title": f"{q}",
                    "y_true": np.array([]),
                    "y_pred": np.array([]),
                    "size_buckets": None,
                }
            )
            continue
        yt = _column_arr(rows, col_t)
        yp = _column_arr(rows, col_p)
        panels.append(
            {
                "title": f"{q}",
                "y_true": yt,
                "y_pred": yp,
                "size_buckets": sb,
            }
        )
    parity_grid_plot(
        run_dir / "parity_mech.png",
        panels,
        nrows=2,
        ncols=3,
        row_labels=ROW_LABELS,
        col_labels=COL_LABELS,
        suptitle=f"{run_dir.name} — mech QoI parity (test split, 2×3 physics-grouped)",
    )
    dist_overlay_grid_plot(
        run_dir / "dist_mech.png",
        panels,
        nrows=2,
        ncols=3,
        row_labels=ROW_LABELS,
        col_labels=COL_LABELS,
        suptitle=f"{run_dir.name} — mech QoI dist overlay (test split, 2×3 physics-grouped)",
    )
    print(f"  [done] {run_dir.name}: parity_mech.png + dist_mech.png (2×3)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--results_dir", type=Path, default=STUDY / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan_pattern", type=str, default="L6*_seed*")
    args = ap.parse_args()

    if args.tag:
        targets = [args.tag]
    else:
        targets = []
        for d in sorted(args.results_dir.glob(args.scan_pattern)):
            if d.is_dir():
                targets.append(d.name.replace(f"_seed{args.seed}", ""))
    if not targets:
        print("[no run dirs found]")
        return 1
    n = 0
    for tag in targets:
        rd = args.results_dir / f"{tag}_seed{args.seed}"
        if not rd.exists():
            rd = args.results_dir / tag
        print(f"  • {tag}")
        if make_for_run(rd):
            n += 1
    print(f"[mech 2×3] done — {n}/{len(targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
