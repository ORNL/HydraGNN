"""run_track_L6_pergraph_het.py — Phase L6 per-graph heteroscedastic σ head.

Sister of L5 at the SVE level. Adds 4 σ output channels paired with the 4
per-graph SVE-MAX cells (lcf_fs, lcf_wp, hcf_fs, hcf_wp). Trained with
pure Gaussian NLL (no topdec aux — per-graph head has too few SVEs per
batch for top-decile gradient signal; user decision 2026-05-06).

Cells (single-rep, seed=42, 400 epochs):

  L6a_het_nll_pg          het_nll_pg                   ← pg σ alone (no pn σ)
  L6b_het_nll_both_50     het_nll_both_50              ← pn σ (L5c) AND pg σ
  L6b2_het_nll_both_50_15 het_nll_both_50_15           ← same but top_frac=0.15

L6a isolates the per-graph σ effect. L6b/L6b2 test full two-level
heteroscedastic UQ (per-grain σ within an SVE + per-SVE σ across SVEs).

This runner uses the v9b deslot Recipe API: `per_grain_targets` and
`pergraph_targets` are explicit symmetric lists (no privileged slot 0).

Usage:
  python run_track_L6_pergraph_het.py
  python run_track_L6_pergraph_het.py --post           # aggregate without re-running
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))


CELLS = [
    ("L6a_het_nll_pg", "het_nll_pg"),
    ("L6b_het_nll_both_50", "het_nll_both_50"),
    ("L6b2_het_nll_both_50_15", "het_nll_both_50_15"),
]

SHALLOW_NODE_HEAD = {"num_headlayers": 1, "dim_headlayers": [8]}
CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")


def _build_extras(loss_recipe: str, base_extra: Dict[str, Any]) -> Dict[str, Any]:
    """Inject the locked v9 base + L6 deslotted target lists + het flags."""
    from manifest import _V9_PERGRAIN_TARGETS_8, _V9_PERGRAPH_TARGETS_14

    extras = copy.deepcopy(base_extra)
    extras["Recipe"]["loss_recipe"] = loss_recipe
    extras["Recipe"]["node_head_dims"] = dict(SHALLOW_NODE_HEAD)
    extras["Recipe"]["use_residual_node_head"] = True
    # v9b deslot: symmetric target lists. The recipe-level API has no
    # primary/aux split; FIPDataset.__post_init__ derives the legacy slot-0
    # + aux split internally to keep HydraGNN's head architecture happy.
    extras["Recipe"]["per_grain_targets"] = list(_V9_PERGRAIN_TARGETS_8)
    extras["Recipe"]["pergraph_targets"] = list(_V9_PERGRAPH_TARGETS_14)
    # Drop the legacy slot-0 + aux keys so the deslot translation runs cleanly.
    extras["Recipe"].pop("per_grain_target_field", None)
    extras["Recipe"].pop("per_grain_target_col", None)
    extras["Recipe"].pop("per_grain_aux_targets", None)
    extras["Recipe"].pop("aux_pergraph_targets", None)
    # Heteroscedastic flags: which head(s) emit σ. Recipe naming determines
    # which combos are viable (het_nll_pg only ⇒ pg=True; het_nll_both ⇒ both).
    if loss_recipe == "het_nll_pg":
        extras["Recipe"]["use_heteroscedastic_max"] = False
        extras["Recipe"]["use_heteroscedastic_pergraph_max"] = True
    elif loss_recipe.startswith("het_nll_both"):
        extras["Recipe"]["use_heteroscedastic_max"] = True
        extras["Recipe"]["use_heteroscedastic_pergraph_max"] = True
    else:
        # Defensive default — pn-only het.
        extras["Recipe"]["use_heteroscedastic_max"] = True
        extras["Recipe"]["use_heteroscedastic_pergraph_max"] = False
    extras.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    extras["NeuralNetwork"]["Architecture"]["graph_pooling"] = "max"
    return extras


def _row_for_run(run_dir: Path, tag: str, recipe: str) -> Dict[str, Any]:
    """Aggregate per-cell pg metrics + σ-residual diagnostics from the
    wide pg CSV. Operates on `pred_test_pg.csv`."""
    rp = run_dir / "run_metrics.json"
    row: Dict[str, Any] = {"cell": tag, "loss_recipe": recipe}
    if not rp.exists():
        return row
    m = json.load(open(rp))
    row["train_time_s"] = float(m.get("elapsed_sec", float("nan")))
    p = run_dir / "pred_test_pg.csv"
    if not p.exists():
        return row
    rows = list(csv.DictReader(open(p)))
    bulks, std_ratios, p99s, sig_resid_corrs, sig_pcov = [], [], [], [], []
    bucket_breakdown: Dict[str, Dict[int, Dict[str, Any]]] = {}
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    for cell in CELL_KEYS:
        col_t = f"true_{cell}_max"
        col_p = f"pred_{cell}_max"
        col_s = f"logsigma_{cell}_max_pergraph"
        rs = [r for r in rows if r.get(col_t) not in (None, "")]
        if not rs:
            continue
        yt = np.array([float(r[col_t]) for r in rs])
        yp = np.array([float(r[col_p]) for r in rs])
        sb = np.array([int(r["size_bucket"]) for r in rs])
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt, yp, sb = yt[finite], yp[finite], sb[finite]
        if len(yt) < 5:
            continue
        bulk = float(r2_score(yt, yp))
        std_r = float(yp.std() / yt.std()) if yt.std() > 0 else float("nan")
        p99_r = (
            float(np.quantile(yp, 0.99) / np.quantile(yt, 0.99))
            if np.quantile(yt, 0.99) != 0
            else float("nan")
        )
        row[f"{cell}_bulk_r2"] = bulk
        row[f"{cell}_std_ratio"] = std_r
        row[f"{cell}_p99_ratio"] = p99_r
        bulks.append(bulk)
        std_ratios.append(std_r)
        p99s.append(p99_r)
        # σ-residual diagnostics + Pcov (per cell + per bucket).
        if rs[0].get(col_s) not in (None, ""):
            sig = np.array(
                [
                    float(r[col_s]) if r[col_s] not in (None, "") else float("nan")
                    for r in rs
                ]
            )
            sig = sig[finite]
            res = np.abs(yp - yt)
            try:
                cor, _ = spearmanr(sig, res)
                row[f"{cell}_sigma_residual_spearman"] = (
                    float(cor) if cor == cor else float("nan")
                )
                if cor == cor:
                    sig_resid_corrs.append(float(cor))
            except Exception:
                row[f"{cell}_sigma_residual_spearman"] = float("nan")
            # Pcov at z=1.65 using local-Jacobian raw-σ approximation:
            #   σ_raw ≈ exp(log_σ_pred) × std(y_true_raw_within_cell).
            sigma_raw = np.exp(sig) * yt.std()
            cov = float(((yp - yt).__abs__() <= 1.65 * sigma_raw).mean())
            row[f"{cell}_pcov_z1.65"] = cov
            sig_pcov.append(cov)
            # Per-bucket Pcov.
            bb = bucket_breakdown.setdefault(cell, {})
            for b in sorted(set(int(x) for x in sb.tolist())):
                m_b = sb == b
                if m_b.sum() < 3:
                    continue
                cov_b = float((((yp - yt).__abs__() <= 1.65 * sigma_raw)[m_b]).mean())
                bb[b] = {
                    "n": int(m_b.sum()),
                    "pcov_z1.65": cov_b,
                    "bulk_r2": float(r2_score(yt[m_b], yp[m_b]))
                    if m_b.sum() >= 2
                    else float("nan"),
                }

    def _fmean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    row["mean_bulk_r2_4cell"] = _fmean(bulks)
    row["mean_std_ratio_4cell"] = _fmean(std_ratios)
    row["mean_p99_ratio_4cell"] = _fmean(p99s)
    row["mean_sigma_residual_spearman_4cell"] = _fmean(sig_resid_corrs)
    row["mean_pcov_z1.65_4cell"] = _fmean(sig_pcov)
    if bucket_breakdown:
        row["per_cell_per_bucket"] = bucket_breakdown
    val = m.get("val") or {}
    sig_stats = val.get("pergraph_logsigma_stats")
    if sig_stats:
        for cell, st in sig_stats.items():
            row[f"sigma_{cell}_mean_val"] = float(st.get("mean", float("nan")))
            row[f"sigma_{cell}_std_val"] = float(st.get("std", float("nan")))
    return row


def _print_leaderboard(
    rows: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]]
) -> None:
    print()
    print("v9b Phase L6 — per-graph heteroscedastic Gaussian-NLL head")
    print("=" * 140)
    hdr = (
        f"  {'cell':<28s} {'recipe':<22s}  "
        f"{'mean_bulk':>9s} {'mean_p99':>9s} {'mean_std':>8s} "
        f"{'mean_pcov':>9s} {'σ-resid_sp':>10s}  {'time':>5s}"
    )
    print(hdr)
    print("-" * 140)
    for r in baseline_rows + rows:
        print(
            f"  {r['cell']:<28s} {r.get('loss_recipe','(baseline)'):<22s}  "
            f"{r.get('mean_bulk_r2_4cell', float('nan')):>+9.3f} "
            f"{r.get('mean_p99_ratio_4cell', float('nan')):>9.3f} "
            f"{r.get('mean_std_ratio_4cell', float('nan')):>8.3f} "
            f"{r.get('mean_pcov_z1.65_4cell', float('nan')):>9.3f} "
            f"{r.get('mean_sigma_residual_spearman_4cell', float('nan')):>+10.3f}  "
            f"{r.get('train_time_s', float('nan')):>5.0f}"
        )
    print("=" * 140)
    if rows:
        print()
        print("Per-cell σ-residual Spearman (positive ⇒ σ ranks errors):")
        for r in rows:
            per = " ".join(
                f"{c.upper()}={r.get(f'{c}_sigma_residual_spearman', float('nan')):>+6.3f}"
                for c in CELL_KEYS
            )
            print(f"  {r['cell']:<30s}  {per}")
        print()
        print("Per-cell Pcov at z=1.65 (target ≥ 0.85 for at least 3/4 cells):")
        for r in rows:
            per = " ".join(
                f"{c.upper()}={r.get(f'{c}_pcov_z1.65', float('nan')):>5.3f}"
                for c in CELL_KEYS
            )
            print(f"  {r['cell']:<30s}  {per}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_epoch", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--cells_only", type=str, default=None)
    args = ap.parse_args()

    cells = CELLS
    if args.cells_only:
        wanted = set(args.cells_only.split(","))
        cells = [c for c in CELLS if c[0] in wanted]

    if not args.post:
        from manifest import BASE_RECIPE, TRACK_V7
        from _runner_utils import run_cell

        base_extra = TRACK_V7[0]["extra"]
        for tag, recipe in cells:
            res_path = STUDY / "results" / f"{tag}_seed{args.seed}" / "run_metrics.json"
            if res_path.exists():
                print(f"[L6] {tag} already has metrics — skip", flush=True)
                continue
            extras = _build_extras(recipe, base_extra)
            print(f"[L6] {tag} starting (loss={recipe})", flush=True)
            t0 = time.time()
            try:
                run_cell(
                    tag,
                    seed=args.seed,
                    base=BASE_RECIPE,
                    extra=extras,
                    num_epoch=args.num_epoch,
                    skip_done=True,
                )
                print(f"[L6] {tag} done ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[L6] {tag} FAILED: {e}", flush=True)
                traceback.print_exc()

    out_dir = STUDY / "results" / "L6_pergraph_het_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for tag, recipe in cells:
        rd = STUDY / "results" / f"{tag}_seed{args.seed}"
        rows.append(_row_for_run(rd, tag, recipe))
    if rows:
        (out_dir / "leaderboard.json").write_text(
            json.dumps(rows, indent=2, default=float)
        )
    _print_leaderboard(rows, baseline_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
