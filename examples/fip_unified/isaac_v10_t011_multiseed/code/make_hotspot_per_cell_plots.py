"""make_hotspot_per_cell_plots.py — post-hoc 4-cell symmetric hotspot
diagnostic plots from existing wide pn CSVs.

For each L*/cell run dir, emits ONE plot per cell (4 plots total):

  hotspot_diagnostic_lcf_fs_max.png   3 splits × 4 metrics for LCF/FS
  hotspot_diagnostic_lcf_wp_max.png   ...
  hotspot_diagnostic_hcf_fs_max.png   ...
  hotspot_diagnostic_hcf_wp_max.png   ...

Works on runs trained BEFORE the train.py hotspot refactor — reads only
`pred_<split>_pn.csv` (the wide format), groups by graph_idx, computes
per-cell hotspot metrics, then calls `hotspot_diagnostic_grid_plot`.

Also computes per-cell metric scalars and writes a JSON summary
`hotspot_per_cell_summary.json` next to the plots so leaderboards can
read them without re-running training.

Usage:
  python make_hotspot_per_cell_plots.py                          # all v9b cells
  python make_hotspot_per_cell_plots.py --tag L6a_het_nll_pg     # one dir
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

from plots import hotspot_diagnostic_grid_plot

CELLS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
SPLITS = ("train", "val", "test")


def _load_pn_csv_grouped(path: Path, cell: str):
    """Load `pred_<split>_pn.csv` and group by graph_idx into per-graph
    arrays of (pred, true) for the requested cell. Drops NaN rows (HCF
    cells on missing-HCF SVEs)."""
    if not path.exists():
        return [], []
    rows = list(csv.DictReader(open(path)))
    col_t, col_p = f"true_{cell}_max", f"pred_{cell}_max"
    by_graph: dict = {}
    for r in rows:
        if r.get(col_t) in (None, "") or r.get(col_p) in (None, ""):
            continue
        try:
            yt = float(r[col_t])
            yp = float(r[col_p])
            gi = int(r["graph_idx"])
        except Exception:
            continue
        if not (np.isfinite(yt) and np.isfinite(yp)):
            continue
        if gi not in by_graph:
            by_graph[gi] = ([], [])
        by_graph[gi][0].append(yp)
        by_graph[gi][1].append(yt)
    pred_per_graph, true_per_graph = [], []
    for gi in sorted(by_graph.keys()):
        p_list, t_list = by_graph[gi]
        if len(p_list) < 2:
            continue
        pred_per_graph.append(np.asarray(p_list))
        true_per_graph.append(np.asarray(t_list))
    return pred_per_graph, true_per_graph


def _per_cell_metrics(pred_per_graph, true_per_graph) -> dict:
    """Compute (within-SVE Spearman, top-1 hit, top-10% recall, hotspot
    loc err) — same recipe as train.py's `_cell_hotspot_metrics`."""
    from scipy.stats import spearmanr

    spears, top1_hits, top10_recall, hotspot_loc_err = [], 0, [], []
    n_graphs_used = 0
    for pred, true in zip(pred_per_graph, true_per_graph):
        n = len(pred)
        if n < 2:
            continue
        n_graphs_used += 1
        try:
            s = spearmanr(true, pred).statistic
            if s == s:
                spears.append(float(s))
        except Exception:
            pass
        if int(np.argmax(pred)) == int(np.argmax(true)):
            top1_hits += 1
        k = max(1, int(np.ceil(0.10 * n)))
        top_true = set(np.argsort(true)[-k:].tolist())
        top_pred = set(np.argsort(pred)[-k:].tolist())
        top10_recall.append(len(top_true & top_pred) / k)
        true_max_idx = int(np.argmax(true))
        pred_rank_of_true_max = int(np.sum(pred >= pred[true_max_idx]))
        hotspot_loc_err.append((pred_rank_of_true_max - 1) / max(n - 1, 1))
    return {
        "n_graphs_used": n_graphs_used,
        "spearman_per_graph_mean": float(np.mean(spears)) if spears else float("nan"),
        "top1_hit_rate": (top1_hits / n_graphs_used) if n_graphs_used else float("nan"),
        "top10_recall": float(np.mean(top10_recall)) if top10_recall else float("nan"),
        "hotspot_loc_err": float(np.mean(hotspot_loc_err))
        if hotspot_loc_err
        else float("nan"),
    }


def make_hotspot_for_run(run_dir: Path, run_label: str = "") -> bool:
    if not run_dir.exists():
        print(f"  [skip] {run_dir} does not exist")
        return False
    if not (run_dir / "pred_test_pn.csv").exists():
        print(f"  [skip] {run_dir} has no pred_test_pn.csv")
        return False
    base = f"{run_label} ({run_dir.name})" if run_label else run_dir.name
    summary: dict = {}
    for cell in CELLS:
        per_split_pairs = {}
        per_split_metrics = {}
        for split in SPLITS:
            p = run_dir / f"pred_{split}_pn.csv"
            pred_pg, true_pg = _load_pn_csv_grouped(p, cell)
            per_split_pairs[split] = (pred_pg, true_pg)
            per_split_metrics[split] = _per_cell_metrics(pred_pg, true_pg)
        # Build 3-row grid input for hotspot_diagnostic_grid_plot.
        hotspot_splits = []
        for split in SPLITS:
            pred_pg, true_pg = per_split_pairs[split]
            hotspot_splits.append(
                {
                    "split": split,
                    "pred_per_graph": pred_pg,
                    "true_per_graph": true_pg,
                }
            )
        hotspot_diagnostic_grid_plot(
            run_dir / f"hotspot_diagnostic_{cell}_max.png",
            hotspot_splits,
            suptitle=f"{base} — hot-spot diagnostic (per-grain MAX, cell={cell})",
        )
        summary[cell] = per_split_metrics
    # Write summary JSON for leaderboard ingestion.
    (run_dir / "hotspot_per_cell_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"  [done] wrote 4 per-cell hotspot plots + summary in {run_dir.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--results_dir", type=Path, default=STUDY / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--scan_pattern",
        type=str,
        default="*_seed*",
        help="Glob for run dirs (default *_seed*; e.g. 'L6*' or 'L5*'",
    )
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
    print(f"[hotspot per-cell] processing {len(targets)} run(s)")
    n_done = 0
    for tag in targets:
        rd = args.results_dir / f"{tag}_seed{args.seed}"
        print(f"  • {tag}")
        if make_hotspot_for_run(rd, run_label=tag):
            n_done += 1
    print(f"[hotspot per-cell] done — {n_done}/{len(targets)} run dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
