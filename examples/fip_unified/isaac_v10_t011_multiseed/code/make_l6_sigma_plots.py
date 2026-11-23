"""make_l6_sigma_plots.py — post-hoc σ-output diagnostic plots for L6 cells.

Sister of `make_l5_sigma_plots.py` operating at the per-graph (SVE) level.
Reads `pred_<split>_pg.csv` and emits 5 grid plots per L6 run:

  sigma_vs_residual_pg.png        log_σ vs |residual|, 4 cells × 3 splits.
                                  Spearman annotated.
  sigma_histogram_pg.png          σ distribution histogram.
  parity_pg_max_by_sigma.png      Parity scatter colored by predicted log_σ.
  parity_pg_max_with_uq.png       Parity + ±1.65σ vertical error bars.
  dist_pg_max_with_uq.png         Distribution overlay with pred + σ noise.

No retraining. Operates on whatever L6* run dirs already have the wide CSVs.

Usage:
  python make_l6_sigma_plots.py                              # all L6 dirs
  python make_l6_sigma_plots.py --tag L6a_het_nll_pg         # one dir
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

from plots import (
    sigma_residual_grid_plot,
    sigma_histogram_grid_plot,
    parity_by_sigma_grid_plot,
    parity_with_uq_grid_plot,
    dist_overlay_with_uq_grid_plot,
)

CELLS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
SPLITS = ("train", "val", "test")


def _load_pg_csv(path: Path):
    rows = list(csv.DictReader(open(path)))
    return rows


def _column_arr(rows, key, *, dtype=float):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            out.append(np.nan)
        else:
            try:
                out.append(dtype(v))
            except Exception:
                out.append(np.nan)
    return np.array(out)


def make_plots_for_run(run_dir: Path, run_label: str = "") -> bool:
    if not run_dir.exists():
        print(f"  [skip] {run_dir} does not exist")
        return False
    test_csv = run_dir / "pred_test_pg.csv"
    if not test_csv.exists():
        print(f"  [skip] {run_dir} has no pred_test_pg.csv")
        return False
    test_rows = _load_pg_csv(test_csv)
    if not test_rows:
        print(f"  [skip] {run_dir} test CSV empty")
        return False
    has_sigma = any(
        c in test_rows[0]
        for c in (
            "logsigma_lcf_fs_max_pergraph",
            "logsigma_lcf_wp_max_pergraph",
            "logsigma_hcf_fs_max_pergraph",
            "logsigma_hcf_wp_max_pergraph",
        )
    )
    if not has_sigma:
        print(f"  [skip] {run_dir} has no logsigma_*_pergraph columns (not L6)")
        return False

    # Per-cell raw-target std for σ_raw approximation. Cell-level local
    # Jacobian: σ_raw[c] ≈ exp(log_σ_pred) × std(y_true_raw[c]).
    cell_std_raw = {}
    for cell in CELLS:
        all_yt = []
        for split in SPLITS:
            p = run_dir / f"pred_{split}_pg.csv"
            if not p.exists():
                continue
            rows = _load_pg_csv(p)
            yt = _column_arr(rows, f"true_{cell}_max")
            all_yt.append(yt[np.isfinite(yt)])
        flat = np.concatenate(all_yt) if all_yt else np.zeros(0)
        cell_std_raw[cell] = float(flat.std()) if flat.size else 1.0

    panels_resid, panels_hist, panels_parity, panels_uq, panels_dist = (
        [],
        [],
        [],
        [],
        [],
    )
    for cell in CELLS:
        for split in SPLITS:
            p = run_dir / f"pred_{split}_pg.csv"
            if not p.exists():
                empty = {"title": f"{cell} / {split}"}
                panels_resid.append(
                    {"sigma": np.array([]), "residuals": np.array([]), **empty}
                )
                panels_hist.append({"sigma": np.array([]), **empty})
                panels_parity.append(
                    {
                        "y_true": np.array([]),
                        "y_pred": np.array([]),
                        "sigma": np.array([]),
                        **empty,
                    }
                )
                panels_uq.append(
                    {
                        "y_true": np.array([]),
                        "y_pred": np.array([]),
                        "sigma_raw": np.array([]),
                        **empty,
                    }
                )
                panels_dist.append(
                    {
                        "y_true": np.array([]),
                        "y_pred": np.array([]),
                        "sigma_raw": np.array([]),
                        **empty,
                    }
                )
                continue
            rows = _load_pg_csv(p)
            yt = _column_arr(rows, f"true_{cell}_max")
            yp = _column_arr(rows, f"pred_{cell}_max")
            sg_log = _column_arr(rows, f"logsigma_{cell}_max_pergraph")
            sb = _column_arr(rows, "size_bucket", dtype=int)
            res = np.abs(yp - yt)
            sigma_raw = np.exp(sg_log) * cell_std_raw[cell]
            panels_resid.append(
                {
                    "sigma": sg_log,
                    "residuals": res,
                    "title": f"{cell} / {split}",
                    "bucket": sb,
                }
            )
            panels_hist.append({"sigma": sg_log, "title": f"{cell} / {split}"})
            panels_parity.append(
                {
                    "y_true": yt,
                    "y_pred": yp,
                    "sigma": sg_log,
                    "title": f"{cell} / {split}",
                }
            )
            panels_uq.append(
                {
                    "y_true": yt,
                    "y_pred": yp,
                    "sigma_raw": sigma_raw,
                    "title": f"{cell} / {split}",
                    "bucket": sb,
                }
            )
            panels_dist.append(
                {
                    "y_true": yt,
                    "y_pred": yp,
                    "sigma_raw": sigma_raw,
                    "title": f"{cell} / {split}",
                }
            )

    base = f"{run_label} ({run_dir.name})" if run_label else run_dir.name
    sigma_residual_grid_plot(
        run_dir / "sigma_vs_residual_pg.png",
        panels_resid,
        nrows=4,
        ncols=3,
        row_labels=list(CELLS),
        col_labels=list(SPLITS),
        suptitle=f"{base} — per-graph predicted log_σ vs |residual| (per cell × split)",
    )
    sigma_histogram_grid_plot(
        run_dir / "sigma_histogram_pg.png",
        panels_hist,
        nrows=4,
        ncols=3,
        row_labels=list(CELLS),
        col_labels=list(SPLITS),
        suptitle=f"{base} — per-graph predicted log_σ distribution",
    )
    parity_by_sigma_grid_plot(
        run_dir / "parity_pg_max_by_sigma.png",
        panels_parity,
        nrows=4,
        ncols=3,
        row_labels=list(CELLS),
        col_labels=list(SPLITS),
        suptitle=f"{base} — per-graph SVE-MAX parity colored by predicted log_σ",
    )
    parity_with_uq_grid_plot(
        run_dir / "parity_pg_max_with_uq.png",
        panels_uq,
        nrows=4,
        ncols=3,
        row_labels=list(CELLS),
        col_labels=list(SPLITS),
        suptitle=(
            f"{base} — per-graph SVE-MAX parity with ±1.65σ error bars\n"
            "(σ_raw ≈ exp(log_σ) × std(y_true_raw); cov = empirical 90% PI coverage)"
        ),
        z=1.65,
        n_subsample=None,  # no subsample — pg has only ~47 test SVEs
    )
    dist_overlay_with_uq_grid_plot(
        run_dir / "dist_pg_max_with_uq.png",
        panels_dist,
        nrows=4,
        ncols=3,
        row_labels=list(CELLS),
        col_labels=list(SPLITS),
        suptitle=(
            f"{base} — per-graph SVE-MAX dist: true vs pred vs (pred + σ noise)\n"
            "(green = pred + 1 sample of N(0, σ²) per SVE — predicted distribution incorporating σ)"
        ),
    )
    print(f"  [done] wrote 5 σ-pg plots in {run_dir.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Single run-dir tag (e.g. L6a_het_nll_pg). "
        "If omitted, scans for all L6* dirs.",
    )
    ap.add_argument("--results_dir", type=Path, default=STUDY / "results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.tag:
        targets = [args.tag]
    else:
        targets = []
        for d in sorted(args.results_dir.glob("L6*_seed*")):
            if d.is_dir():
                targets.append(d.name.replace(f"_seed{args.seed}", ""))
        if not targets:
            print("[no L6 run dirs found]")
            return 1
    print(f"[L6-σ-plots] processing {len(targets)} run(s)")
    n_done = 0
    for tag in targets:
        rd = args.results_dir / f"{tag}_seed{args.seed}"
        print(f"  • {tag}")
        if make_plots_for_run(rd, run_label=tag):
            n_done += 1
    print(f"[L6-σ-plots] done — wrote plots in {n_done}/{len(targets)} run dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
