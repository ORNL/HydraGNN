"""tail_metrics.py — tail-aware regression metrics.

R² alone is the wrong scoreboard for per-grain FIP prediction because the
metric is dominated by the bulk of the distribution, while the operationally
important regime is the tail (high-FIP grains drive per-graph MAX, which
drives fatigue prediction). This module computes:

  - overall: r², pearson, spearman, regression slope (pred = a·true + b),
             regression intercept
  - top-decile (true ≥ p90): r², pearson, slope, mean-relative-error
  - per-decile: slope of pred vs true within each decile of true (10 numbers)
  - per-graph max (when graph_idx is provided):
      max-of-pred vs max-of-true r²/pearson/relative-error,
      within-graph top-1 hit rate,
      hot-spot localization error.

Single entry point `compute_tail_metrics(y_true, y_pred, graph_idx=None) -> dict`.

No HydraGNN imports — pure numpy + scipy. Reusable from train.py and from
re-scoring scripts walking results dirs.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return float("nan")
    try:
        return float(pearsonr(a, b).statistic)
    except Exception:
        return float("nan")


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    try:
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float("nan")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination = 1 - SS_res / SS_tot."""
    if len(y_true) < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-30:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _ols_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """OLS fit pred = a·true + b.  Slope a = 1 means perfect-magnitude;
    a ≈ 0.5 means the model compresses the dynamic range by half."""
    if len(y_true) < 2:
        return float("nan"), float("nan")
    if np.std(y_true) < 1e-30:
        return float("nan"), float("nan")
    a, b = np.polyfit(y_true, y_pred, 1)
    return float(a), float(b)


def _mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of |pred - true| / |true| (signed predictions / nonzero-true assumption)."""
    if len(y_true) == 0:
        return float("nan")
    eps = 1e-30
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + eps)))


def compute_tail_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    graph_idx: Optional[Sequence[int]] = None,
    top_pct: float = 0.10,
    n_deciles: int = 10,
) -> Dict[str, float]:
    """Compute tail-aware regression metrics.

    Parameters
    ----------
    y_true, y_pred : flat 1-D arrays of equal length, raw (un-normalized) units.
    graph_idx      : optional 1-D array of graph IDs; if given, enables
                     per-graph max metrics.
    top_pct        : top fraction of y_true to define the "tail" (default 0.10).
    n_deciles      : number of bins for the per-decile slope diagnostic.

    Returns
    -------
    dict with keys:
        overall_r2, overall_pearson, overall_spearman, overall_slope, overall_intercept
        top10_r2, top10_pearson, top10_slope, top10_mean_rel_err, top10_n
        slope_per_decile  : list of `n_deciles` floats
        decile_edges      : list of `n_deciles + 1` floats (in y_true)
        (if graph_idx given:)
        per_graph_max_r2, per_graph_max_pearson, per_graph_max_spearman,
        per_graph_max_mean_rel_err, per_graph_max_relative_bias,
        within_graph_top1_hit, within_graph_top10pct_recall
        n_graphs
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    out: Dict[str, float] = {}

    # --- Overall ---
    out["overall_r2"] = _safe_r2(y_true, y_pred)
    out["overall_pearson"] = _safe_pearson(y_true, y_pred)
    out["overall_spearman"] = _safe_spearman(y_true, y_pred)
    a, b = _ols_slope_intercept(y_true, y_pred)
    out["overall_slope"] = a
    out["overall_intercept"] = b
    out["n"] = int(len(y_true))

    # --- Tail (top decile of y_true) ---
    if len(y_true) >= 10:
        thr = float(np.quantile(y_true, 1.0 - top_pct))
        mask = y_true >= thr
        yt_top = y_true[mask]
        yp_top = y_pred[mask]
        out["top10_threshold"] = thr
        out["top10_n"] = int(mask.sum())
        out["top10_r2"] = _safe_r2(yt_top, yp_top)
        out["top10_pearson"] = _safe_pearson(yt_top, yp_top)
        out["top10_spearman"] = _safe_spearman(yt_top, yp_top)
        sa, sb = _ols_slope_intercept(yt_top, yp_top)
        out["top10_slope"] = sa
        out["top10_intercept"] = sb
        out["top10_mean_rel_err"] = _mean_relative_error(yt_top, yp_top)
        # Mean residual sign: positive = under-prediction in tail, negative = over.
        out["top10_signed_residual_mean"] = float(np.mean(yt_top - yp_top))
    else:
        for k in (
            "top10_threshold",
            "top10_n",
            "top10_r2",
            "top10_pearson",
            "top10_spearman",
            "top10_slope",
            "top10_intercept",
            "top10_mean_rel_err",
            "top10_signed_residual_mean",
        ):
            out[k] = float("nan")

    # --- Per-decile slope (where does compression actually happen?) ---
    if len(y_true) >= n_deciles * 2:
        edges = np.quantile(y_true, np.linspace(0.0, 1.0, n_deciles + 1))
        slopes = []
        for i in range(n_deciles):
            lo, hi = edges[i], edges[i + 1]
            if i == n_deciles - 1:
                m = (y_true >= lo) & (y_true <= hi)
            else:
                m = (y_true >= lo) & (y_true < hi)
            if m.sum() < 10:
                slopes.append(float("nan"))
                continue
            sa, _ = _ols_slope_intercept(y_true[m], y_pred[m])
            slopes.append(sa)
        out["slope_per_decile"] = slopes
        out["decile_edges"] = edges.tolist()
    else:
        out["slope_per_decile"] = [float("nan")] * n_deciles
        out["decile_edges"] = [float("nan")] * (n_deciles + 1)

    # --- Per-graph metrics (if graph_idx given) ---
    if graph_idx is not None:
        graph_idx = np.asarray(graph_idx).ravel()
        if graph_idx.shape != y_true.shape:
            raise ValueError(
                f"graph_idx shape {graph_idx.shape} != y shape {y_true.shape}"
            )
        unique_graphs = np.unique(graph_idx)
        max_pred = []
        max_true = []
        top1_hits = 0
        top10_recall = []
        n_eligible = 0
        for g in unique_graphs:
            m = graph_idx == g
            if m.sum() < 2:
                continue
            yt_g = y_true[m]
            yp_g = y_pred[m]
            max_true.append(float(yt_g.max()))
            max_pred.append(float(yp_g.max()))
            # within-graph top-1 hit
            if int(np.argmax(yp_g)) == int(np.argmax(yt_g)):
                top1_hits += 1
            # within-graph top-10% recall
            n = len(yt_g)
            k = max(1, int(np.ceil(0.10 * n)))
            top_t = set(np.argsort(yt_g)[-k:].tolist())
            top_p = set(np.argsort(yp_g)[-k:].tolist())
            top10_recall.append(len(top_t & top_p) / k)
            n_eligible += 1

        max_true_a = np.asarray(max_true)
        max_pred_a = np.asarray(max_pred)
        out["n_graphs"] = int(n_eligible)
        out["per_graph_max_r2"] = _safe_r2(max_true_a, max_pred_a)
        out["per_graph_max_pearson"] = _safe_pearson(max_true_a, max_pred_a)
        out["per_graph_max_spearman"] = _safe_spearman(max_true_a, max_pred_a)
        if len(max_true_a) > 0:
            out["per_graph_max_mean_rel_err"] = _mean_relative_error(
                max_true_a, max_pred_a
            )
            out["per_graph_max_relative_bias"] = float(
                np.mean(max_pred_a - max_true_a) / np.mean(max_true_a)
            )
            sa, sb = _ols_slope_intercept(max_true_a, max_pred_a)
            out["per_graph_max_slope"] = sa
            out["per_graph_max_intercept"] = sb
        out["within_graph_top1_hit"] = (
            top1_hits / n_eligible if n_eligible else float("nan")
        )
        out["within_graph_top10pct_recall"] = (
            float(np.mean(top10_recall)) if top10_recall else float("nan")
        )

    return out


def format_summary(m: Dict[str, float]) -> str:
    """One-line condensed summary of a metrics dict."""
    parts = [
        f"R²={m.get('overall_r2', float('nan')):+.3f}",
        f"slope={m.get('overall_slope', float('nan')):+.2f}",
        f"top10_R²={m.get('top10_r2', float('nan')):+.3f}",
        f"top10_slope={m.get('top10_slope', float('nan')):+.2f}",
        f"top10_relerr={m.get('top10_mean_rel_err', float('nan')):.3f}",
    ]
    if "per_graph_max_r2" in m:
        parts += [
            f"pgmax_R²={m['per_graph_max_r2']:+.3f}",
            f"pgmax_slope={m.get('per_graph_max_slope', float('nan')):+.2f}",
            f"pgmax_relerr={m.get('per_graph_max_mean_rel_err', float('nan')):.3f}",
            f"top1_hit={m.get('within_graph_top1_hit', float('nan')):.3f}",
        ]
    return "  ".join(parts)


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.exponential(1.0, n)
    # Compressed predictor: pred = 0.5*true + 0.5*mean(true) + small noise
    y_pred = 0.5 * y_true + 0.5 * y_true.mean() + 0.05 * rng.standard_normal(n)
    m = compute_tail_metrics(y_true, y_pred)
    print(format_summary(m))
    print(f"  expected: overall_R² high (~0.6), top10_slope ≈ 0.5, top10_R² LOW")
