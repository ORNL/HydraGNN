"""make_distshape_plots.py — Phase M distribution-shape parity plots.

Generates 4-cell parity grids for any of:
  - log-mean μ_log    (per-grain lognormal location)
  - log-scale σ_log   (per-grain lognormal scale, the new feature-conditional
                       voxel-UQ output enabled by Phase M / M_B base)
  - p90 / p99         (within-grain quantiles, Phase M proposal C)

Reads the extended `pred_test_pn.csv` whose schema includes
`{true,pred}_<cell>_{logmu,logsigma,p90,p99}` columns when those targets
are present in the per-grain head. Writes one PNG per active stat:

  parity_pn_logmu.png      parity_pn_logsigma.png
  parity_pn_p90.png        parity_pn_p99.png

Plus `parity_pn_distshape_extras.json` summarising R² / Spearman / n
per cell × stat.

If the run did not train with these targets, returns SKIPPED (no error).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

from plots import parity_grid_plot
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
DISTSHAPE_STATS = (
    ("logmu", "log-mean μ_log"),
    ("logsigma", "log-scale σ_log"),
    ("p90", "p90 quantile"),
    ("p99", "p99 quantile"),
)


def _to_f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _column_arr(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(_to_f(v) if v not in (None, "") else float("nan"))
    return np.array(out)


def make_for_run(run_dir: Path) -> bool:
    pn_csv = run_dir / "pred_test_pn.csv"
    if not pn_csv.exists():
        print(f"  [skip] {run_dir.name}: no pred_test_pn.csv")
        return False

    rows = list(csv.DictReader(open(pn_csv)))
    if not rows:
        print(f"  [skip] {run_dir.name}: empty CSV")
        return False

    header_keys = set(rows[0].keys())
    # Find which distribution-shape stats are present in the CSV.
    active_stats = []
    for stat, label in DISTSHAPE_STATS:
        present_cells = [
            c
            for c in CELL_KEYS
            if f"true_{c}_{stat}" in header_keys and f"pred_{c}_{stat}" in header_keys
        ]
        if len(present_cells) >= 2:  # at least 2 cells present
            active_stats.append((stat, label, present_cells))

    if not active_stats:
        print(f"  [skip] {run_dir.name}: no distribution-shape columns in CSV")
        return False

    sb_arr = _column_arr(rows, "size_bucket").astype(int).tolist()

    summary = {}
    for stat, label, present_cells in active_stats:
        panels = []
        cell_metrics = {}
        for c in CELL_KEYS:
            t_col = f"true_{c}_{stat}"
            p_col = f"pred_{c}_{stat}"
            if t_col not in header_keys or p_col not in header_keys:
                panels.append(
                    {
                        "title": f"{c.upper()}/{stat}",
                        "y_true": np.array([]),
                        "y_pred": np.array([]),
                        "size_buckets": None,
                    }
                )
                cell_metrics[c] = {"present": False}
                continue
            yt = _column_arr(rows, t_col)
            yp = _column_arr(rows, p_col)
            mask = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[mask], yp[mask]
            sb_filtered = [int(b) for b, m in zip(sb_arr, mask.tolist()) if m]
            if len(yt) >= 8:
                r2 = float(r2_score(yt, yp))
                rho, _ = spearmanr(yt, yp)
                rho = float(rho)
                pe, _ = pearsonr(yt, yp)
                pe = float(pe)
            else:
                r2 = rho = pe = float("nan")
            panels.append(
                {
                    "title": f"{c.upper()}\nR²={r2:+.3f}  ρ={rho:+.3f}  n={len(yt)}",
                    "y_true": yt,
                    "y_pred": yp,
                    "size_buckets": sb_filtered if sb_filtered else None,
                }
            )
            cell_metrics[c] = {
                "r2": r2,
                "spearman": rho,
                "pearson": pe,
                "n": int(len(yt)),
            }
        outpath = run_dir / f"parity_pn_{stat}.png"
        parity_grid_plot(
            outpath,
            panels,
            nrows=2,
            ncols=2,
            row_labels=["LCF", "HCF"],
            col_labels=["FS", "WP"],
            suptitle=f"{run_dir.name} — per-grain {label} parity (test split)",
        )
        summary[stat] = cell_metrics
        print(f"    [done] {outpath.name}")

    # Save summary JSON
    with open(run_dir / "parity_pn_distshape_extras.json", "w") as f:
        json.dump(summary, f, indent=2)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--scan_pattern", type=str, default="M_*_seed*")
    args = ap.parse_args()

    runs = []
    if args.run:
        runs.append(args.run)
    else:
        for d in sorted((STUDY / "results").glob(args.scan_pattern)):
            if d.is_dir():
                runs.append(d)
    if not runs:
        print("[no runs found]")
        return 1
    n_ok = 0
    for r in runs:
        if make_for_run(r):
            n_ok += 1
    print(f"[distshape plots] done — {n_ok}/{len(runs)} runs had distshape targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
