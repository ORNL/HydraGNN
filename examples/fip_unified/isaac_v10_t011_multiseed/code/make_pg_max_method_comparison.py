"""make_pg_max_method_comparison.py — direct pg-MAX head prediction vs
order-statistic (max of per-grain predictions) for the same SVE.

For every test SVE, the model gives us TWO predictions of the SVE-MAX:

  1. Direct: per-graph head's MAX channel (slot 0..3 of pg outputs).
     UQ: ±z·σ_pg (per-graph σ, available when use_heteroscedastic_pergraph_max).

  2. Order-statistic: max over grains of the per-grain head's MAX
     channel for that SVE. This is what the model implicitly says the
     SVE-MAX is when interpreted as "the worst grain in the SVE".
     UQ: ±z·σ_pn at the argmax grain (available when use_heteroscedastic_max).

Both should equal the true SVE-MAX. Disagreement between them is a
diagnostic — direct head and order-stat are different inductive biases.

Output (per L6 run dir, test split):

  parity_pg_max_methods.png   4 cells × 3 cols = 12 panels:
                                col 1: direct pg-MAX with ±1.65σ_pg
                                col 2: order-stat pg-MAX with ±1.65σ_pn (argmax grain)
                                col 3: direct vs order-stat agreement scatter

  pg_max_methods_summary.json  per-cell metrics:
                                direct R², order-stat R², direct Pcov, order-stat Pcov,
                                direct-vs-orderstat correlation, mean abs diff

Usage:
  python make_pg_max_method_comparison.py                       # all L6* dirs
  python make_pg_max_method_comparison.py --tag L6b2_*          # one dir
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

CELLS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")


def _load_pg_test(run_dir: Path):
    p = run_dir / "pred_test_pg.csv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p)))


def _load_pn_test(run_dir: Path):
    p = run_dir / "pred_test_pn.csv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p)))


def _orderstat_from_pn(pn_rows, cell: str):
    """Group per-grain rows by graph_idx, take max of pred_<cell>_max for
    each SVE. Also pull log_sigma at the argmax grain (if present).
    Returns dict graph_idx -> (true_max, pred_orderstat, sigma_at_argmax)."""
    by_graph: dict = {}
    col_t = f"true_{cell}_max"
    col_p = f"pred_{cell}_max"
    col_s = f"logsigma_{cell}_max"
    for r in pn_rows:
        if r.get(col_t) in (None, "") or r.get(col_p) in (None, ""):
            continue
        try:
            yt = float(r[col_t])
            yp = float(r[col_p])
            gi = int(r["graph_idx"])
        except Exception:
            continue
        sg = float("nan")
        if r.get(col_s) not in (None, ""):
            try:
                sg = float(r[col_s])
            except Exception:
                pass
        if not (np.isfinite(yt) and np.isfinite(yp)):
            continue
        cur = by_graph.get(gi)
        if cur is None or yp > cur[1]:
            by_graph[gi] = (
                yt,
                yp,
                sg,
                yt,
            )  # (true_at_argmax_pred, pred_argmax, σ_argmax, true_max_so_far)
        # Track true MAX separately (regardless of pred argmax)
        if cur is None:
            by_graph[gi] = (yt, yp, sg, yt)
        else:
            t_at_pred_argmax, p_argmax, s_argmax, true_max = cur
            new_true_max = max(true_max, yt)
            if yp > p_argmax:
                by_graph[gi] = (yt, yp, sg, new_true_max)
            else:
                by_graph[gi] = (t_at_pred_argmax, p_argmax, s_argmax, new_true_max)
    return by_graph


def _draw_parity_uq_panel(
    ax, y_true, y_pred, sigma_raw, *, panel_title=None, z=1.65, color_pts="C0"
):
    """Parity scatter with optional ±z·σ error bars. If σ is all-NaN
    (e.g. running on a config without that head's σ output), draw plain
    parity without error bars and report Pcov=NaN."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    s = np.asarray(sigma_raw)
    finite_pred = np.isfinite(yt) & np.isfinite(yp)
    yt_p, yp_p = yt[finite_pred], yp[finite_pred]
    s_p = s[finite_pred] if s.shape == yt.shape else np.full(yt_p.shape, np.nan)
    has_sigma = np.isfinite(s_p).all() and s_p.size > 0
    if yt_p.size == 0:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        if panel_title:
            ax.set_title(panel_title, fontsize=8)
        return float("nan"), float("nan")
    from sklearn.metrics import r2_score

    r2 = float(r2_score(yt_p, yp_p))
    cov = (
        float(((yp_p - yt_p).__abs__() <= z * s_p).mean())
        if has_sigma
        else float("nan")
    )
    if has_sigma:
        ax.errorbar(
            yt_p,
            yp_p,
            yerr=z * s_p,
            fmt="o",
            ms=3,
            ecolor="grey",
            elinewidth=0.5,
            capsize=0,
            alpha=0.75,
            mec="none",
            color=color_pts,
        )
    else:
        ax.scatter(yt_p, yp_p, s=14, alpha=0.75, edgecolor="none", color=color_pts)
    lo = float(min(yt_p.min(), yp_p.min()))
    hi = float(max(yt_p.max(), yp_p.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=0.7)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ann = f"R²={r2:+.3f}\nN={yt_p.size}"
    if has_sigma:
        ann = f"R²={r2:+.3f}\ncov(±{z:.2f}σ)={cov:.3f}\nN={yt_p.size}"
    else:
        ann += "\n(no σ output)"
    ax.text(
        0.04,
        0.96,
        ann,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)
    return r2, cov


def _draw_agreement_panel(ax, direct, orderstat, true_max, *, panel_title=None):
    direct = np.asarray(direct)
    orderstat = np.asarray(orderstat)
    tm = np.asarray(true_max)
    finite = np.isfinite(direct) & np.isfinite(orderstat) & np.isfinite(tm)
    d, o, t = direct[finite], orderstat[finite], tm[finite]
    if d.size == 0:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        return
    sc = ax.scatter(d, o, c=t, cmap="plasma", s=12, alpha=0.85, edgecolor="none")
    lo = float(min(d.min(), o.min()))
    hi = float(max(d.max(), o.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    corr = float(np.corrcoef(d, o)[0, 1]) if d.size > 1 else float("nan")
    mad = float(np.mean(np.abs(d - o)))
    ax.text(
        0.04,
        0.96,
        f"corr(direct,os)={corr:+.3f}\nMAD={mad:.3g}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)
    return corr, mad


def make_for_run(run_dir: Path) -> bool:
    pg_rows = _load_pg_test(run_dir)
    pn_rows = _load_pn_test(run_dir)
    if pg_rows is None or pn_rows is None:
        print(f"  [skip] {run_dir} missing pred_test_*.csv")
        return False

    summary: dict = {}
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), squeeze=False)
    col_titles = [
        "direct pg-MAX (±σ_pg)",
        "order-stat pn-MAX (±σ_pn at argmax)",
        "direct vs order-stat agreement",
    ]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=10, fontweight="bold")

    for i, cell in enumerate(CELLS):
        # ---- Direct pg-MAX from per-graph head ----
        col_t = f"true_{cell}_max"
        col_p = f"pred_{cell}_max"
        col_s = f"logsigma_{cell}_max_pergraph"
        rows = [r for r in pg_rows if r.get(col_t) not in (None, "")]
        yt_pg = np.array([float(r[col_t]) for r in rows])
        yp_pg = np.array([float(r[col_p]) for r in rows])
        if rows and rows[0].get(col_s) not in (None, ""):
            sg_pg = np.array(
                [
                    float(r[col_s]) if r.get(col_s) not in (None, "") else float("nan")
                    for r in rows
                ]
            )
        else:
            sg_pg = np.full_like(yt_pg, np.nan)
        # Approximate σ_raw via local Jacobian.
        finite_pg = np.isfinite(yt_pg) & np.isfinite(yp_pg)
        yt_std = float(yt_pg[finite_pg].std()) if finite_pg.any() else 1.0
        sigma_raw_pg = np.exp(sg_pg) * yt_std

        # ---- Order-stat pg-MAX from per-grain predictions ----
        os_map = _orderstat_from_pn(pn_rows, cell)
        # Align order: by graph_idx in pg test rows.
        gids_pg = [int(r["graph_idx"]) for r in rows]
        yp_os = []
        sg_os = []
        true_max_align = []
        for gi in gids_pg:
            ent = os_map.get(gi)
            if ent is None:
                yp_os.append(np.nan)
                sg_os.append(np.nan)
                true_max_align.append(np.nan)
            else:
                _, p_argmax, s_argmax, true_max = ent
                yp_os.append(p_argmax)
                sg_os.append(s_argmax)
                true_max_align.append(true_max)
        yp_os = np.array(yp_os)
        sg_os = np.array(sg_os)
        true_max_align = np.array(true_max_align)
        # σ_raw via per-cell std of the order-stat true values.
        finite_os = np.isfinite(true_max_align) & np.isfinite(yp_os)
        os_std = float(true_max_align[finite_os].std()) if finite_os.any() else 1.0
        sigma_raw_os = np.exp(sg_os) * os_std

        # ---- Plot panels ----
        r2_d, cov_d = (
            _draw_parity_uq_panel(
                axes[i][0],
                yt_pg,
                yp_pg,
                sigma_raw_pg,
                panel_title=f"{cell}",
                color_pts="#1f77b4",
            )
            or (float("nan"), float("nan"))
        )
        r2_o, cov_o = (
            _draw_parity_uq_panel(
                axes[i][1],
                true_max_align,
                yp_os,
                sigma_raw_os,
                panel_title=f"{cell}",
                color_pts="#2ca02c",
            )
            or (float("nan"), float("nan"))
        )
        agg = _draw_agreement_panel(
            axes[i][2],
            yp_pg,
            yp_os,
            yt_pg,
            panel_title=f"{cell}",
        )
        corr, mad = agg if agg is not None else (float("nan"), float("nan"))
        axes[i][0].set_ylabel(f"{cell}\npred", fontsize=10, fontweight="bold")
        summary[cell] = {
            "direct": {"r2": r2_d, "pcov_z1.65": cov_d, "n": int(finite_pg.sum())},
            "orderstat": {"r2": r2_o, "pcov_z1.65": cov_o, "n": int(finite_os.sum())},
            "agreement": {"corr": corr, "mean_abs_diff": mad},
        }

    for ax in axes[-1, :]:
        ax.set_xlabel("y_true (raw FIP)", fontsize=9)
    fig.suptitle(
        f"{run_dir.name} — pg-MAX direct head vs order-stat from pn-MAX (test split)",
        fontsize=11,
    )
    fig.tight_layout()
    out = run_dir / "parity_pg_max_methods.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    (run_dir / "pg_max_methods_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(
        f"  [done] {run_dir.name}: parity_pg_max_methods.png + pg_max_methods_summary.json"
    )
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
    print(f"[pg-method-cmp] processing {len(targets)} run(s)")
    n_done = 0
    for tag in targets:
        rd = args.results_dir / f"{tag}_seed{args.seed}"
        if not rd.exists():
            # tag may already include seed suffix
            rd = args.results_dir / tag
        print(f"  • {tag}")
        if make_for_run(rd):
            n_done += 1
    print(f"[pg-method-cmp] done — {n_done}/{len(targets)} run dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
