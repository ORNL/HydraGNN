"""diagnostics_plots.py — per-cell diagnostic plots for FIP regression runs.

Why these plots.  R² + Spearman alone don't reveal the failure mode. The
v3 leaderboard has uniform pn_R²≈0.62, but six different per-cell pictures
of *why per-graph max R² is −3.6*. These plots explain the picture so the
paper can argue from images instead of numbers.

Per-run plots (saved to `<run_dir>/diagnostics/`):

  pgmax_parity.png        true vs pred per-graph max, raw + cal side-by-side,
                          color by SVE bucket, R² + slope annotated.
  distribution_overlay.png  KDE/hist of y_true, y_pred_raw, y_pred_cal —
                          shows the variance shrinkage MSE imposes and how
                          calibration recovers it.
  tail_decile_slope.png   slope of pred-vs-true per y_true decile (1..10).
                          Tail compression shows up as low slope in upper
                          deciles. Two panels: raw / cal.
  gumbel_per_graph.png    fit Gumbel(μ, β) to per-grain MAX distribution
                          inside each test graph, separately for true and
                          predicted; scatter (μ_true vs μ_pred), (β_true
                          vs β_pred). Anto's physics: per-graph max ~ Gumbel.
                          If the model captures the per-grain distribution,
                          μ̂ and β̂ should track μ and β.
  topK_parity.png         per-graph top-K-mean true vs pred for K∈{1,5,10}.
                          Operationally: top-1 IS the per-graph max; top-5/10
                          are smoother proxies the paper can quote.

Cross-cell summary (saved to `<study_root>/diagnostics/`):

  cell_pgmax_R2.png       sorted bar chart, raw vs cal pgmax_R² per cell.
  cell_top10_slope.png    same for top-10 slope.
  cell_calibration_lift.png   per cell: how much cal lifted pgmax_R²
                              and top10_slope.

Usage:

  from diagnostics_plots import make_run_plots, make_summary_plots
  make_run_plots(run_dir)             # writes 5 PNGs under run_dir/diagnostics/
  make_summary_plots(study_root)      # walks results/, writes summary PNGs

CLI: `python diagnostics_plots.py <study_root>` runs both for every cell.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r, linregress, pearsonr, spearmanr

# Import calibration relative to this file (same package).
from calibration import calibrate_run_dir, calibrate_run_dir_pergraph, fit_per_bucket
from plots import parity_plot, per_grain_diagnostic_plot
from tail_metrics import compute_tail_metrics


BUCKET_TO_LSVE = {0: "125 µm", 1: "177 µm", 2: "250 µm", 3: "354 µm", 4: "500 µm"}
BUCKET_COLORS = {0: "#440154", 1: "#3b528b", 2: "#21918c", 3: "#5ec962", 4: "#fde725"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_csv(
    path: Path, pred_col: str = "y_pred_raw"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    y_true = np.array([float(r["y_true_raw"]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])
    gi = np.array([int(r["graph_idx"]) for r in rows])
    bucket = (
        np.array([int(r["size_bucket"]) for r in rows])
        if rows and "size_bucket" in rows[0]
        else np.zeros_like(gi)
    )
    return y_true, y_pred, gi, bucket


def _ensure_calibrated(run_dir: Path) -> bool:
    """Run calibration if it hasn't been run yet."""
    cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
    if cal_csv.exists():
        return True
    val_csv = run_dir / "val_pergrain_predictions.csv"
    if not val_csv.exists():
        return False
    try:
        calibrate_run_dir(str(run_dir), mode="per_bucket")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-graph aggregators
# ---------------------------------------------------------------------------
def _per_graph_max(y: np.ndarray, gi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-graph max(y) along with one bucket per graph."""
    unique = np.unique(gi)
    out = np.empty(len(unique))
    for i, g in enumerate(unique):
        out[i] = y[gi == g].max()
    return unique, out


def _per_graph_topk_mean(y: np.ndarray, gi: np.ndarray, k: int) -> np.ndarray:
    """Return mean of top-k per-graph values (sorted descending)."""
    unique = np.unique(gi)
    out = np.empty(len(unique))
    for i, g in enumerate(unique):
        v = np.sort(y[gi == g])[::-1]
        out[i] = v[:k].mean()
    return out


def _per_graph_gumbel_fit(
    y: np.ndarray, gi: np.ndarray, min_n: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Gumbel(μ, β) per graph via method-of-moments.
    β = std·√6/π,  μ = mean − β·γ_E.   Returns (graph_ids, μ, β)."""
    GAMMA_E = 0.5772156649
    unique = np.unique(gi)
    mu = np.full(len(unique), np.nan)
    beta = np.full(len(unique), np.nan)
    for i, g in enumerate(unique):
        v = y[gi == g]
        if len(v) < min_n:
            continue
        m = v.mean()
        s = v.std(ddof=0)
        b = s * np.sqrt(6.0) / np.pi
        beta[i] = b
        mu[i] = m - b * GAMMA_E
    return unique, mu, beta


def _scatter_one_to_one(ax, x, y, color=None, label=None, scatter_kw=None):
    scatter_kw = scatter_kw or {}
    ax.scatter(
        x,
        y,
        c=color,
        label=label,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.4,
        **scatter_kw,
    )
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = 0.05 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "--",
        color="gray",
        lw=1,
        label="y = x",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", "box")


def _annotate_fit(ax, x_true, y_pred):
    """Annotate a parity-plot axis with R², slope, pearson on (x_true, y_pred).

    Important: R² = 1 − Σ(true − pred)² / Σ(true − mean(true))². The
    denominator is the variance of the *truth*, not the prediction.
    """
    m = np.isfinite(x_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return
    sl = linregress(x_true[m], y_pred[m])
    ss_tot = float(np.sum((x_true[m] - x_true[m].mean()) ** 2))
    ss_res = float(np.sum((x_true[m] - y_pred[m]) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pr = pearsonr(x_true[m], y_pred[m]).statistic
    ax.text(
        0.04,
        0.96,
        f"R²={r2:+.3f}\nslope={sl.slope:.3f}\npearson={pr:+.3f}\nn={int(m.sum())}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )


# ---------------------------------------------------------------------------
# Plot 1: per-graph max parity
# ---------------------------------------------------------------------------
def _plot_pgmax_parity(
    run_dir: Path,
    out_dir: Path,
    *,
    simple: bool = False,
    pred_csv: Optional[Path] = None,
    pred_col: str = "y_pred_raw",
    title_suffix: str = "",
) -> None:
    """Per-graph max parity. If simple, single panel of one prediction set
    (raw by default; passing pred_csv/pred_col overrides). Else two panels
    (raw, calibrated) read from the run dir."""
    csv_in = pred_csv or (run_dir / "test_pergrain_predictions.csv")
    yt_raw, yp_raw, gi_raw, bk_raw = _load_csv(csv_in, pred_col)
    if not simple:
        cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
        if cal_csv.exists():
            yt_cal, yp_cal, gi_cal, bk_cal = _load_csv(cal_csv, "y_pred_cal")
        else:
            yt_cal, yp_cal, gi_cal, bk_cal = yt_raw, yp_raw, gi_raw, bk_raw

    # Per-graph max
    g_raw, mt = _per_graph_max(yt_raw, gi_raw)
    _, mp_raw = _per_graph_max(yp_raw, gi_raw)
    bg = np.array([int(bk_raw[gi_raw == g][0]) for g in g_raw])

    if simple:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6))
        for b in np.unique(bg):
            m = bg == b
            ax.scatter(
                mt[m],
                mp_raw[m],
                c=BUCKET_COLORS.get(int(b), "k"),
                edgecolor="k",
                linewidth=0.4,
                alpha=0.85,
                label=BUCKET_TO_LSVE.get(int(b), str(b)),
            )
        lo = min(mt.min(), mp_raw.min())
        hi = max(mt.max(), mp_raw.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true per-graph max FIP")
        ax.set_ylabel("predicted per-graph max FIP")
        _annotate_fit(ax, mt, mp_raw)
        ax.legend(title="SVE size", fontsize=8, loc="lower right", framealpha=0.85)
        fig.suptitle(
            f"{run_dir.name} — per-graph MAX parity{title_suffix}", fontsize=11
        )
        fig.tight_layout()
        fig.savefig(out_dir / "pgmax_parity.png", dpi=140)
        plt.close(fig)
        return

    _, mp_cal = _per_graph_max(yp_cal, gi_cal)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for b in np.unique(bg):
        m = bg == b
        axes[0].scatter(
            mt[m],
            mp_raw[m],
            c=BUCKET_COLORS.get(int(b), "k"),
            edgecolor="k",
            linewidth=0.4,
            alpha=0.85,
            label=BUCKET_TO_LSVE.get(int(b), str(b)),
        )
        axes[1].scatter(
            mt[m],
            mp_cal[m],
            c=BUCKET_COLORS.get(int(b), "k"),
            edgecolor="k",
            linewidth=0.4,
            alpha=0.85,
            label=BUCKET_TO_LSVE.get(int(b), str(b)),
        )
    for ax, title, mp in zip(axes, ["RAW pred", "CALIBRATED pred"], [mp_raw, mp_cal]):
        lo = min(mt.min(), mp.min())
        hi = max(mt.max(), mp.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true per-graph max FIP")
        ax.set_ylabel("predicted per-graph max FIP")
        ax.set_title(title)
        _annotate_fit(ax, mt, mp)
    axes[0].legend(title="SVE size", fontsize=8, loc="lower right", framealpha=0.85)
    fig.suptitle(f"{run_dir.name} — per-graph MAX parity", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pgmax_parity.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: distribution overlay (true vs raw [vs cal])
# ---------------------------------------------------------------------------
def _plot_distribution_overlay(
    run_dir: Path,
    out_dir: Path,
    *,
    simple: bool = False,
    pred_csv: Optional[Path] = None,
    pred_col: str = "y_pred_raw",
    title_suffix: str = "",
) -> None:
    csv_in = pred_csv or (run_dir / "test_pergrain_predictions.csv")
    yt, yp_raw, _, _ = _load_csv(csv_in, pred_col)
    if simple:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
        bins = np.linspace(min(yt.min(), yp_raw.min()), max(yt.max(), yp_raw.max()), 80)
        ax.hist(yt, bins=bins, alpha=0.45, density=True, label="y_true", color="k")
        ax.hist(
            yp_raw, bins=bins, alpha=0.45, density=True, label="y_pred", color="tab:red"
        )
        ax.set_xlabel("FIP value")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=9)
        ax.text(
            0.04,
            0.96,
            f"std(true)={yt.std():.4f}\nstd(pred)={yp_raw.std():.4f}\nratio={yp_raw.std()/yt.std():.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
        )
        fig.suptitle(
            f"{run_dir.name} — per-grain distribution: pred vs truth{title_suffix}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "distribution_overlay.png", dpi=140)
        plt.close(fig)
        return

    cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
    if cal_csv.exists():
        _, yp_cal, _, _ = _load_csv(cal_csv, "y_pred_cal")
    else:
        yp_cal = yp_raw

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(
        min(yt.min(), yp_raw.min(), yp_cal.min()),
        max(yt.max(), yp_raw.max(), yp_cal.max()),
        80,
    )
    for ax, ydata, label, color in [
        (axes[0], yt, "y_true", "k"),
        (axes[0], yp_raw, "y_pred raw", "tab:red"),
        (axes[1], yt, "y_true", "k"),
        (axes[1], yp_cal, "y_pred cal", "tab:blue"),
    ]:
        ax.hist(ydata, bins=bins, alpha=0.45, density=True, label=label, color=color)
    for ax, title in zip(axes, ["RAW pred vs truth", "CAL pred vs truth"]):
        ax.set_xlabel("FIP value")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title)
        if "RAW" in title:
            ax.text(
                0.04,
                0.96,
                f"std(true)={yt.std():.4f}\nstd(pred)={yp_raw.std():.4f}\nratio={yp_raw.std()/yt.std():.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
            )
        else:
            ax.text(
                0.04,
                0.96,
                f"std(true)={yt.std():.4f}\nstd(pred)={yp_cal.std():.4f}\nratio={yp_cal.std()/yt.std():.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
            )
    fig.suptitle(f"{run_dir.name} — per-grain distribution: pred vs truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "distribution_overlay.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: tail decile slope (raw + cal)
# ---------------------------------------------------------------------------
def _decile_slopes(y_true, y_pred, n_deciles=10) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(y_true, np.linspace(0, 1, n_deciles + 1))
    slopes = np.full(n_deciles, np.nan)
    centers = np.full(n_deciles, np.nan)
    for i in range(n_deciles):
        lo, hi = edges[i], edges[i + 1]
        m = (y_true >= lo) & (y_true <= hi if i == n_deciles - 1 else y_true < hi)
        if m.sum() < 10:
            continue
        sl = linregress(y_true[m], y_pred[m])
        slopes[i] = sl.slope
        centers[i] = 0.5 * (lo + hi)
    return slopes, centers


def _plot_tail_decile_slope(
    run_dir: Path,
    out_dir: Path,
    *,
    simple: bool = False,
    pred_csv: Optional[Path] = None,
    pred_col: str = "y_pred_raw",
    title_suffix: str = "",
) -> None:
    csv_in = pred_csv or (run_dir / "test_pergrain_predictions.csv")
    yt, yp_raw, _, _ = _load_csv(csv_in, pred_col)
    slopes_raw, _ = _decile_slopes(yt, yp_raw)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.2))
    x = np.arange(1, 11)

    if simple:
        ax.bar(
            x,
            slopes_raw,
            0.7,
            color="tab:red",
            edgecolor="k",
            linewidth=0.4,
            label="pred",
        )
    else:
        cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
        yp_cal = _load_csv(cal_csv, "y_pred_cal")[1] if cal_csv.exists() else yp_raw
        slopes_cal, _ = _decile_slopes(yt, yp_cal)
        w = 0.4
        ax.bar(
            x - w / 2,
            slopes_raw,
            w,
            label="raw",
            color="tab:red",
            edgecolor="k",
            linewidth=0.4,
        )
        ax.bar(
            x + w / 2,
            slopes_cal,
            w,
            label="cal",
            color="tab:blue",
            edgecolor="k",
            linewidth=0.4,
        )
    ax.axhline(1.0, color="k", lw=1, ls="--", label="y=x slope")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in x])
    ax.set_xlabel("decile of y_true (D10 = top 10%)")
    ax.set_ylabel("OLS slope of y_pred vs y_true (within decile)")
    ax.set_title(f"{run_dir.name} — tail compression by decile{title_suffix}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tail_decile_slope.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: per-graph Gumbel fit (μ, β) — true vs pred
# ---------------------------------------------------------------------------
def _plot_gumbel_per_graph(
    run_dir: Path,
    out_dir: Path,
    *,
    simple: bool = False,
    pred_csv: Optional[Path] = None,
    pred_col: str = "y_pred_raw",
    title_suffix: str = "",
) -> None:
    csv_in = pred_csv or (run_dir / "test_pergrain_predictions.csv")
    yt, yp_raw, gi, _ = _load_csv(csv_in, pred_col)
    _, mu_t, beta_t = _per_graph_gumbel_fit(yt, gi)
    _, mu_r, beta_r = _per_graph_gumbel_fit(yp_raw, gi)

    if simple:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for axi, (true, pred, name) in enumerate(
            [(mu_t, mu_r, "μ"), (beta_t, beta_r, "β")]
        ):
            axes[axi].scatter(
                true,
                pred,
                color="tab:red",
                alpha=0.7,
                edgecolor="k",
                linewidth=0.4,
                label="pred",
            )
            lo = float(np.nanmin([true, pred]))
            hi = float(np.nanmax([true, pred]))
            pad = 0.05 * (hi - lo) if hi > lo else 1e-6
            axes[axi].plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                "--",
                color="gray",
                lw=1,
                label="y=x",
            )
            axes[axi].set_xlim(lo - pad, hi + pad)
            axes[axi].set_ylim(lo - pad, hi + pad)
            axes[axi].set_aspect("equal", "box")
            axes[axi].set_xlabel(f"Gumbel {name} (true grain-MAX dist)")
            axes[axi].set_ylabel(f"Gumbel {name} (predicted grain-MAX dist)")
            axes[axi].set_title(f"per-graph Gumbel {name}")
            _annotate_fit(axes[axi], true, pred)
            axes[axi].legend(loc="lower right", fontsize=9)
        fig.suptitle(
            f"{run_dir.name} — per-graph Gumbel(μ, β) fit on grain-MAX distribution{title_suffix}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "gumbel_per_graph.png", dpi=140)
        plt.close(fig)
        return

    cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
    if cal_csv.exists():
        _, yp_cal, _, _ = _load_csv(cal_csv, "y_pred_cal")
    else:
        yp_cal = yp_raw
    _, mu_c, beta_c = _per_graph_gumbel_fit(yp_cal, gi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # μ
    axes[0].scatter(
        mu_t,
        mu_r,
        color="tab:red",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.4,
        label="raw pred",
    )
    axes[0].scatter(
        mu_t,
        mu_c,
        color="tab:blue",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.4,
        label="cal pred",
    )
    lo = np.nanmin([mu_t, mu_r, mu_c])
    hi = np.nanmax([mu_t, mu_r, mu_c])
    pad = 0.05 * (hi - lo)
    axes[0].plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "--",
        color="gray",
        lw=1,
        label="y=x",
    )
    axes[0].set_xlabel("Gumbel μ (true grain-MAX dist)")
    axes[0].set_ylabel("Gumbel μ (predicted grain-MAX dist)")
    axes[0].set_title("per-graph Gumbel μ")
    axes[0].set_xlim(lo - pad, hi + pad)
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].set_aspect("equal", "box")
    axes[0].legend(loc="lower right", fontsize=9)
    # Annotate
    for arr, lab, dy in [(mu_r, "raw", 0.0), (mu_c, "cal", 0.18)]:
        m = np.isfinite(mu_t) & np.isfinite(arr)
        if m.sum() < 3:
            continue
        pr = pearsonr(mu_t[m], arr[m]).statistic
        # R² = 1 − Σ(true − pred)² / Σ(true − mean(true))² — denom is var(truth).
        ss_tot = float(np.sum((mu_t[m] - mu_t[m].mean()) ** 2))
        ss_res = float(np.sum((mu_t[m] - arr[m]) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        axes[0].text(
            0.04,
            0.96 - dy,
            f"{lab}: R²={r2:+.2f}, ρ={pr:+.2f}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            color={"raw": "tab:red", "cal": "tab:blue"}[lab],
        )

    # β
    axes[1].scatter(
        beta_t,
        beta_r,
        color="tab:red",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.4,
        label="raw pred",
    )
    axes[1].scatter(
        beta_t,
        beta_c,
        color="tab:blue",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.4,
        label="cal pred",
    )
    lo = np.nanmin([beta_t, beta_r, beta_c])
    hi = np.nanmax([beta_t, beta_r, beta_c])
    pad = 0.05 * (hi - lo)
    axes[1].plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "--",
        color="gray",
        lw=1,
        label="y=x",
    )
    axes[1].set_xlabel("Gumbel β (true grain-MAX dist)")
    axes[1].set_ylabel("Gumbel β (predicted grain-MAX dist)")
    axes[1].set_title("per-graph Gumbel β")
    axes[1].set_xlim(lo - pad, hi + pad)
    axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].set_aspect("equal", "box")
    axes[1].legend(loc="lower right", fontsize=9)
    for arr, lab, dy in [(beta_r, "raw", 0.0), (beta_c, "cal", 0.18)]:
        m = np.isfinite(beta_t) & np.isfinite(arr)
        if m.sum() < 3:
            continue
        pr = pearsonr(beta_t[m], arr[m]).statistic
        ss_tot = float(np.sum((beta_t[m] - beta_t[m].mean()) ** 2))
        ss_res = float(np.sum((beta_t[m] - arr[m]) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        axes[1].text(
            0.04,
            0.96 - dy,
            f"{lab}: R²={r2:+.2f}, ρ={pr:+.2f}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            color={"raw": "tab:red", "cal": "tab:blue"}[lab],
        )

    fig.suptitle(
        f"{run_dir.name} — per-graph Gumbel(μ, β) fit on grain-MAX distribution",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gumbel_per_graph.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: top-K per-graph parity (K=1, 5, 10)
# ---------------------------------------------------------------------------
def _plot_topK_parity(
    run_dir: Path,
    out_dir: Path,
    *,
    simple: bool = False,
    pred_csv: Optional[Path] = None,
    pred_col: str = "y_pred_raw",
    title_suffix: str = "",
) -> None:
    csv_in = pred_csv or (run_dir / "test_pergrain_predictions.csv")
    yt, yp_raw, gi, bk = _load_csv(csv_in, pred_col)
    Ks = [1, 5, 10]
    graphs = np.unique(gi)
    bg = np.array([int(bk[gi == g][0]) for g in graphs])

    if simple:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for j, k in enumerate(Ks):
            t = _per_graph_topk_mean(yt, gi, k)
            p = _per_graph_topk_mean(yp_raw, gi, k)
            ax = axes[j]
            for b in np.unique(bg):
                m = bg == b
                ax.scatter(
                    t[m],
                    p[m],
                    c=BUCKET_COLORS.get(int(b), "k"),
                    edgecolor="k",
                    linewidth=0.4,
                    alpha=0.85,
                    label=BUCKET_TO_LSVE.get(int(b), str(b)) if j == 0 else None,
                )
            lo = min(t.min(), p.min())
            hi = max(t.max(), p.max())
            pad = 0.05 * (hi - lo)
            ax.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", "box")
            ax.set_xlabel(f"top-{k} mean true")
            ax.set_ylabel(f"top-{k} mean pred")
            ax.set_title(f"K={k}")
            _annotate_fit(ax, t, p)
        axes[0].legend(title="SVE size", fontsize=8, loc="lower right", framealpha=0.85)
        fig.suptitle(
            f"{run_dir.name} — per-graph top-K mean parity (K=1,5,10){title_suffix}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "topK_parity.png", dpi=140)
        plt.close(fig)
        return

    cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
    if cal_csv.exists():
        _, yp_cal, _, _ = _load_csv(cal_csv, "y_pred_cal")
    else:
        yp_cal = yp_raw

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    for j, k in enumerate(Ks):
        t = _per_graph_topk_mean(yt, gi, k)
        r = _per_graph_topk_mean(yp_raw, gi, k)
        c = _per_graph_topk_mean(yp_cal, gi, k)
        for i, (pred, lab, _color) in enumerate(
            [(r, "raw", "tab:red"), (c, "cal", "tab:blue")]
        ):
            ax = axes[i, j]
            for b in np.unique(bg):
                m = bg == b
                ax.scatter(
                    t[m],
                    pred[m],
                    c=BUCKET_COLORS.get(int(b), "k"),
                    edgecolor="k",
                    linewidth=0.4,
                    alpha=0.85,
                    label=BUCKET_TO_LSVE.get(int(b), str(b))
                    if (i == 0 and j == 0)
                    else None,
                )
            lo = min(t.min(), pred.min())
            hi = max(t.max(), pred.max())
            pad = 0.05 * (hi - lo)
            ax.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", "box")
            ax.set_xlabel(f"top-{k} mean true")
            ax.set_ylabel(f"top-{k} mean pred ({lab})")
            ax.set_title(f"K={k} — {lab.upper()}")
            _annotate_fit(ax, t, pred)
    axes[0, 0].legend(title="SVE size", fontsize=8, loc="lower right", framealpha=0.85)
    fig.suptitle(
        f"{run_dir.name} — per-graph top-K mean parity (K=1,5,10)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_dir / "topK_parity.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry: per-run + summary
# ---------------------------------------------------------------------------
def _make_calibrated_trainpy_plots(run_dir: Path, cal_dir: Path) -> None:
    """Mirror the train.py-style plots on the calibrated test predictions:

      parity_test_pergrain.png    — per-grain parity, calibrated
      parity_test.png             — per-graph parity, calibrated (if per-graph head ran)
      hotspot_diagnostic_test.png — per-grain rank-quality diagnostic, calibrated

    Per-grain calibration is read from `test_pergrain_predictions_cal.csv`
    (already produced by `_ensure_calibrated`). Per-graph calibration is fit
    on the val per-graph predictions and applied to test.
    """
    import csv as _csv

    # --- per-grain (test) ----------------------------------------------------
    cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
    if cal_csv.exists():
        rows = list(_csv.DictReader(open(cal_csv)))
        if rows:
            yt = np.array([float(r["y_true_raw"]) for r in rows])
            yc = np.array([float(r["y_pred_cal"]) for r in rows])
            gi = np.array([int(r["graph_idx"]) for r in rows])
            bk = np.array([int(r["size_bucket"]) for r in rows])
            # parity_test_pergrain (calibrated)
            parity_plot(
                cal_dir / "parity_test_pergrain.png",
                f"{run_dir.name} test parity (per-grain) [CAL]",
                yt,
                yc,
                size_buckets=bk.tolist(),
                bucket_to_label={i: BUCKET_TO_LSVE.get(i, str(i)) for i in range(5)},
            )
            # hotspot diagnostic — needs per-graph lists of (true, pred)
            true_pg, pred_pg = [], []
            for g in np.unique(gi):
                m = gi == g
                true_pg.append(yt[m])
                pred_pg.append(yc[m])
            per_grain_diagnostic_plot(
                cal_dir / "hotspot_diagnostic_test.png",
                f"{run_dir.name} hotspot diagnostic [CAL]",
                pred_pg,
                true_pg,
            )

    # --- per-graph (test) — if the cell trained the per-graph head -----------
    try:
        ok = calibrate_run_dir_pergraph(str(run_dir), mode="per_bucket")
    except Exception as e:
        ok = False
        print(f"  [warn] {run_dir.name}: per-graph cal failed: {e}")
    if ok:
        pg_cal = run_dir / "test_predictions_cal.csv"
        rows = list(_csv.DictReader(open(pg_cal)))
        if rows:
            yt = np.array([float(r["y_true_raw"]) for r in rows])
            yc = np.array([float(r["y_pred_cal"]) for r in rows])
            bk = np.array([int(r["size_bucket"]) for r in rows])
            parity_plot(
                cal_dir / "parity_test.png",
                f"{run_dir.name} test parity (per-graph) [CAL]",
                yt,
                yc,
                size_buckets=bk.tolist(),
                bucket_to_label={i: BUCKET_TO_LSVE.get(i, str(i)) for i in range(5)},
            )


def make_run_plots(run_dir: str | Path, *, simple_mode: bool = False) -> bool:
    """Generate all 5 diagnostic plots for one run dir.
    Returns True if any plots were made; False if predictions are missing.

    Parameters
    ----------
    simple_mode : bool, default False
        If True (v4 default): single-panel plots of raw predictions written
        directly to `run_dir/`, AND a parallel set of single-panel plots of
        calibrated predictions written to `run_dir/calibrated/`.
        Calibration is fit on val and applied to test (no leakage), so it
        is a valid post-processing step worth showing alongside raw.
        If False (v3 default): two-panel raw-vs-calibrated plots written
        to `run_dir/diagnostics/` (single side-by-side comparison).
    """
    run_dir = Path(run_dir)
    if not (run_dir / "test_pergrain_predictions.csv").exists():
        return False

    if simple_mode:
        # Raw plots → run_dir/
        _plot_pgmax_parity(run_dir, run_dir, simple=True)
        _plot_distribution_overlay(run_dir, run_dir, simple=True)
        _plot_tail_decile_slope(run_dir, run_dir, simple=True)
        try:
            _plot_gumbel_per_graph(run_dir, run_dir, simple=True)
        except Exception as e:
            print(f"  [warn] {run_dir.name}: gumbel plot (raw) failed: {e}")
        _plot_topK_parity(run_dir, run_dir, simple=True)

        # Calibrated plots → run_dir/calibrated/
        if _ensure_calibrated(run_dir):
            cal_csv = run_dir / "test_pergrain_predictions_cal.csv"
            cal_dir = run_dir / "calibrated"
            cal_dir.mkdir(exist_ok=True)

            # 5 simple-mode diagnostic plots (calibrated test per-grain)
            kw = {
                "simple": True,
                "pred_csv": cal_csv,
                "pred_col": "y_pred_cal",
                "title_suffix": "  [CAL]",
            }
            _plot_pgmax_parity(run_dir, cal_dir, **kw)
            _plot_distribution_overlay(run_dir, cal_dir, **kw)
            _plot_tail_decile_slope(run_dir, cal_dir, **kw)
            try:
                _plot_gumbel_per_graph(run_dir, cal_dir, **kw)
            except Exception as e:
                print(f"  [warn] {run_dir.name}: gumbel plot (cal) failed: {e}")
            _plot_topK_parity(run_dir, cal_dir, **kw)

            # ALSO mirror the train.py-style plots on calibrated test:
            #   parity_test_pergrain.png   (per-grain calibrated)
            #   parity_test.png            (per-graph calibrated, if available)
            #   hotspot_diagnostic_test.png (per-grain rank diagnostic, calibrated)
            _make_calibrated_trainpy_plots(run_dir, cal_dir)
        return True

    # Dual-panel mode (v3 style)
    _ensure_calibrated(run_dir)
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(exist_ok=True)
    _plot_pgmax_parity(run_dir, out_dir, simple=False)
    _plot_distribution_overlay(run_dir, out_dir, simple=False)
    _plot_tail_decile_slope(run_dir, out_dir, simple=False)
    try:
        _plot_gumbel_per_graph(run_dir, out_dir, simple=False)
    except Exception as e:
        print(f"  [warn] {run_dir.name}: gumbel plot failed: {e}")
    _plot_topK_parity(run_dir, out_dir, simple=False)
    return True


def _aggregate_e_cells(rows: List[Dict]) -> List[Dict]:
    """Collapse E_<tag>_split<ss>_seed<ii> into one row per <tag> with mean ± std.
    Non-E rows pass through unchanged."""
    from collections import defaultdict

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    out: List[Dict] = []
    for r in rows:
        if r["name"].startswith("E_"):
            # strip "_split{ss}_seed{ii}" → keep the underlying tag for grouping
            base = r["name"]
            for sep in ("_split", "_seed"):
                if sep in base:
                    base = base.rsplit(sep, 1)[0]
            grouped[base].append(r)
        else:
            out.append(r)
    for base, lst in grouped.items():
        agg = {"name": f"{base} (mean of {len(lst)})"}
        for k in lst[0]:
            if k == "name":
                continue
            vals = [r[k] for r in lst if isinstance(r[k], float) and r[k] == r[k]]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(agg)
    return out


def make_summary_plots(study_root: str | Path) -> None:
    """Walk results/{tag}_seed*/ and write 3 cross-cell summary plots
    to study_root/diagnostics/. E-track reps are collapsed into one bar
    each (mean ± std over the 25 splits×inits)."""
    study_root = Path(study_root)
    results = study_root / "results"
    out_dir = study_root / "diagnostics"
    out_dir.mkdir(exist_ok=True)

    raw_rows: List[Dict] = []
    for d in sorted(results.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "test_pergrain_predictions.csv").exists():
            continue
        if not _ensure_calibrated(d):
            continue
        try:
            yt, yp_raw, gi, _ = _load_csv(
                d / "test_pergrain_predictions.csv", "y_pred_raw"
            )
            yt_c, yp_cal, gi_c, _ = _load_csv(
                d / "test_pergrain_predictions_cal.csv", "y_pred_cal"
            )
            tm_raw = compute_tail_metrics(yt, yp_raw, graph_idx=gi)
            tm_cal = compute_tail_metrics(yt_c, yp_cal, graph_idx=gi_c)
            raw_rows.append(
                {
                    "name": d.name,
                    "raw_pgmax_r2": tm_raw.get("per_graph_max_r2", float("nan")),
                    "cal_pgmax_r2": tm_cal.get("per_graph_max_r2", float("nan")),
                    "raw_top10_slope": tm_raw.get("top10_slope", float("nan")),
                    "cal_top10_slope": tm_cal.get("top10_slope", float("nan")),
                    "raw_pgmax_pearson": tm_raw.get(
                        "per_graph_max_pearson", float("nan")
                    ),
                }
            )
        except Exception as e:
            print(f"  [warn] summary skip {d.name}: {e}")

    if not raw_rows:
        print("  no rows for summary")
        return

    rows = _aggregate_e_cells(raw_rows)

    # Sort by cal_pgmax_r2 descending
    rows.sort(
        key=lambda r: -r["cal_pgmax_r2"]
        if r["cal_pgmax_r2"] == r["cal_pgmax_r2"]
        else 1e9
    )
    names = [r["name"] for r in rows]
    raw_r2 = np.array([r["raw_pgmax_r2"] for r in rows])
    cal_r2 = np.array([r["cal_pgmax_r2"] for r in rows])
    raw_r2_std = np.array([r.get("raw_pgmax_r2_std", 0.0) or 0.0 for r in rows])
    cal_r2_std = np.array([r.get("cal_pgmax_r2_std", 0.0) or 0.0 for r in rows])
    raw_sl = np.array([r["raw_top10_slope"] for r in rows])
    cal_sl = np.array([r["cal_top10_slope"] for r in rows])
    raw_sl_std = np.array([r.get("raw_top10_slope_std", 0.0) or 0.0 for r in rows])
    cal_sl_std = np.array([r.get("cal_top10_slope_std", 0.0) or 0.0 for r in rows])

    width = max(8, 0.55 * len(rows))

    # Plot 1: pgmax R² raw vs cal
    fig, ax = plt.subplots(figsize=(width, 5.5))
    x = np.arange(len(rows))
    w = 0.4
    ax.bar(
        x - w / 2,
        raw_r2,
        w,
        yerr=raw_r2_std,
        label="raw",
        color="tab:red",
        edgecolor="k",
        linewidth=0.4,
        capsize=2.5,
    )
    ax.bar(
        x + w / 2,
        cal_r2,
        w,
        yerr=cal_r2_std,
        label="cal",
        color="tab:blue",
        edgecolor="k",
        linewidth=0.4,
        capsize=2.5,
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("per-graph max R²")
    ax.set_title("Per-cell per-graph max R² (raw vs calibrated; E reps aggregated)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_pgmax_R2.png", dpi=140)
    plt.close(fig)

    # Plot 2: top10 slope raw vs cal
    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.bar(
        x - w / 2,
        raw_sl,
        w,
        yerr=raw_sl_std,
        label="raw",
        color="tab:red",
        edgecolor="k",
        linewidth=0.4,
        capsize=2.5,
    )
    ax.bar(
        x + w / 2,
        cal_sl,
        w,
        yerr=cal_sl_std,
        label="cal",
        color="tab:blue",
        edgecolor="k",
        linewidth=0.4,
        capsize=2.5,
    )
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal slope=1")
    ax.axhline(0.30, color="g", lw=1, ls=":", label="ISAAC un-pause gate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("top-10% slope (within-tail)")
    ax.set_title("Per-cell top-10% slope (raw vs calibrated; E reps aggregated)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_top10_slope.png", dpi=140)
    plt.close(fig)

    # Plot 3: calibration lift = cal − raw, for both metrics
    fig, ax = plt.subplots(figsize=(width, 5.5))
    lift_r2 = cal_r2 - raw_r2
    lift_sl = cal_sl - raw_sl
    ax.bar(
        x - w / 2,
        lift_r2,
        w,
        label="ΔR²(pgmax)",
        color="tab:purple",
        edgecolor="k",
        linewidth=0.4,
    )
    ax.bar(
        x + w / 2,
        lift_sl,
        w,
        label="Δslope(top10%)",
        color="tab:orange",
        edgecolor="k",
        linewidth=0.4,
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("calibration lift (cal − raw)")
    ax.set_title("Per-cell calibration lift on pgmax R² and top-10% slope")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_calibration_lift.png", dpi=140)
    plt.close(fig)

    print(f"  wrote summary to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("study_root", help="path to study folder (has results/ inside)")
    ap.add_argument(
        "--no_per_run",
        action="store_true",
        help="skip per-cell plots; just write the cross-cell summary",
    )
    ap.add_argument(
        "--include", nargs="+", default=None, help="only run on these tag substrings"
    )
    ap.add_argument(
        "--simple",
        action="store_true",
        help="single-panel raw-only plots, written directly to run_dir "
        "(v4 style; default is v3 raw-vs-calibrated under diagnostics/).",
    )
    args = ap.parse_args()

    study_root = Path(args.study_root)
    if not args.no_per_run:
        for d in sorted((study_root / "results").iterdir()):
            if not d.is_dir():
                continue
            if args.include and not any(s in d.name for s in args.include):
                continue
            ok = make_run_plots(d, simple_mode=args.simple)
            print(f"  {d.name}: {'ok' if ok else 'skip (no preds)'}")
    make_summary_plots(study_root)


if __name__ == "__main__":
    main()
