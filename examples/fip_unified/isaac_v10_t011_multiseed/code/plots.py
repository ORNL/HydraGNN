"""Plot helpers for cp3d_v2 results dirs.

Each function writes a PNG into the given results dir. Inputs come from arrays
already produced by train.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _r2_pearson_mape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan"), float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    pear = (
        float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 else float("nan")
    )
    nz = np.abs(y_true) > 1e-12
    mape = (
        float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)
        if nz.any()
        else float("nan")
    )
    return r2, pear, mape


def parity_plot(
    out_path: Path,
    title: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    size_buckets: Optional[Sequence[int]] = None,
    bucket_to_label: Optional[Dict[int, str]] = None,
) -> None:
    """Per-graph parity scatter with R²/Pearson/MAPE annotation.

    `size_buckets` (optional) — color points by SVE bucket if provided.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # NaN/Inf guard: yeo_johnson inverse can produce NaN for far-tail
    # predictions on small training partitions. Drop those points
    # before axis-limit computation to avoid `Axis limits cannot be NaN`.
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        if size_buckets is not None and len(size_buckets) == len(finite):
            size_buckets = [b for b, ok in zip(size_buckets, finite) if ok]
    if y_true.size == 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(
            0.5, 0.5, "no test data", ha="center", va="center", transform=ax.transAxes
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    r2, pear, mape = _r2_pearson_mape(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if size_buckets is not None and len(size_buckets) == len(y_true):
        unique = sorted(set(int(b) for b in size_buckets))
        cmap = plt.get_cmap("viridis")
        for k, b in enumerate(unique):
            mask = np.array(size_buckets) == b
            label = (
                bucket_to_label.get(b, f"bucket {b}")
                if bucket_to_label
                else f"bucket {b}"
            )
            ax.scatter(
                y_true[mask],
                y_pred[mask],
                s=24,
                alpha=0.85,
                edgecolor="k",
                linewidth=0.4,
                color=cmap(k / max(len(unique) - 1, 1)),
                label=label,
            )
        ax.legend(loc="lower right", fontsize=9, frameon=True)
    else:
        ax.scatter(
            y_true, y_pred, s=24, alpha=0.85, edgecolor="k", linewidth=0.4, color="C0"
        )

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.0, label="y=x")

    ann_extra = f"\n(dropped {n_dropped} non-finite)" if n_dropped > 0 else ""
    ann = f"R² = {r2:+.4f}\nPearson = {pear:+.4f}\nMAPE = {mape:.2f}%\nN = {y_true.size}{ann_extra}"
    ax.text(
        0.04,
        0.96,
        ann,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.4"),
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("y_true (raw FIP)")
    ax.set_ylabel("y_pred (raw FIP)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def per_grain_diagnostic_plot(
    out_path: Path, title: str, pred_per_graph, true_per_graph
) -> None:
    """4-panel per-grain diagnostic:
    (a) within-graph Spearman histogram
    (b) top-K recall curve
    (c) hotspot localization error histogram
    (d) parity scatter colored by within-graph quantile
    """
    from scipy.stats import spearmanr as _spr

    n_graphs = len(pred_per_graph)
    if n_graphs == 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(
            0.5,
            0.5,
            "no per-grain data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    spears = []
    top_k_curves = []
    hotspot_errs = []
    flat_pred, flat_true, flat_color = [], [], []

    for pred, true in zip(pred_per_graph, true_per_graph):
        n = len(pred)
        if n < 5:
            continue
        try:
            s = _spr(true, pred).statistic
            if s == s:
                spears.append(float(s))
        except Exception:
            pass
        # Top-K recall curve, K ∈ {1, 5, 10, 20, 50}%
        kfracs = [0.01, 0.05, 0.10, 0.20, 0.50]
        rec_for_g = []
        for kf in kfracs:
            k = max(1, int(np.ceil(kf * n)))
            top_t = set(np.argsort(true)[-k:].tolist())
            top_p = set(np.argsort(pred)[-k:].tolist())
            rec_for_g.append(len(top_t & top_p) / k)
        top_k_curves.append(rec_for_g)
        # Hotspot loc err
        true_max_idx = int(np.argmax(true))
        rank_of_truth = int(np.sum(pred >= pred[true_max_idx]))
        hotspot_errs.append((rank_of_truth - 1) / max(n - 1, 1))
        # Flat scatter with within-graph rank
        ranks = np.argsort(np.argsort(true)) / max(n - 1, 1)  # quantile within graph
        flat_pred.extend(pred.tolist())
        flat_true.extend(true.tolist())
        flat_color.extend(ranks.tolist())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) Within-graph Spearman histogram
    ax = axes[0, 0]
    if spears:
        ax.hist(spears, bins=30, color="C0", edgecolor="k", alpha=0.85)
        ax.axvline(
            np.mean(spears),
            color="red",
            linestyle="--",
            label=f"mean = {np.mean(spears):+.3f}",
        )
        ax.legend()
    ax.set_xlabel("within-graph Spearman ρ")
    ax.set_ylabel("# graphs")
    ax.set_title("(a) within-graph rank correlation distribution")
    ax.grid(alpha=0.3)

    # (b) Top-K recall curve
    ax = axes[0, 1]
    if top_k_curves:
        kfracs = [1, 5, 10, 20, 50]
        arr = np.array(top_k_curves)  # (n_graphs, 5)
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        ax.errorbar(kfracs, m, yerr=s, fmt="-o", color="C2", capsize=4)
        ax.axhline(0.10, color="gray", linestyle=":", label="random (0.10 for top-10%)")
        for kf, mv in zip(kfracs, m):
            ax.text(kf, mv + 0.02, f"{mv:.2f}", ha="center", fontsize=8)
        ax.set_xlabel("K (%)")
        ax.set_ylabel("recall")
        ax.set_xscale("log")
        ax.set_title("(b) top-K recall (mean ± std across graphs)")
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.grid(alpha=0.3)

    # (c) Hotspot localization error histogram
    ax = axes[1, 0]
    if hotspot_errs:
        ax.hist(hotspot_errs, bins=30, color="C3", edgecolor="k", alpha=0.85)
        ax.axvline(
            np.mean(hotspot_errs),
            color="red",
            linestyle="--",
            label=f"mean = {np.mean(hotspot_errs):.3f}",
        )
        ax.axvline(0.5, color="gray", linestyle=":", label="random = 0.5")
        ax.legend()
    ax.set_xlabel("normalized rank of true-max in prediction (0=perfect)")
    ax.set_ylabel("# graphs")
    ax.set_title("(c) hotspot localization error")
    ax.grid(alpha=0.3)

    # (d) Per-grain parity colored by within-graph quantile
    ax = axes[1, 1]
    if flat_pred:
        flat_pred_arr = np.array(flat_pred)
        flat_true_arr = np.array(flat_true)
        # Subsample for plot legibility
        if flat_pred_arr.size > 5000:
            idx = np.random.default_rng(0).choice(
                flat_pred_arr.size, size=5000, replace=False
            )
            flat_pred_arr = flat_pred_arr[idx]
            flat_true_arr = flat_true_arr[idx]
            flat_color_arr = np.array(flat_color)[idx]
        else:
            flat_color_arr = np.array(flat_color)
        sc = ax.scatter(
            flat_true_arr,
            flat_pred_arr,
            c=flat_color_arr,
            cmap="viridis",
            s=8,
            alpha=0.6,
            edgecolor="none",
        )
        plt.colorbar(sc, ax=ax, label="within-graph rank quantile")
        lo = float(min(flat_true_arr.min(), flat_pred_arr.min()))
        hi = float(max(flat_true_arr.max(), flat_pred_arr.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.0)
    ax.set_xlabel("y_true (raw FIP)")
    ax.set_ylabel("y_pred (raw FIP)")
    ax.set_title("(d) per-grain parity, colored by within-graph rank")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def loss_history_plot(
    out_path: Path,
    hist_pickle: Optional[Path] = None,
    train_loss: Optional[Sequence[float]] = None,
    val_loss: Optional[Sequence[float]] = None,
    test_loss: Optional[Sequence[float]] = None,
    best_epoch: Optional[int] = None,
) -> None:
    """Train/val/test loss curves with best-val marker, log-y, trimmed to actual epochs."""
    if hist_pickle is not None and Path(hist_pickle).exists():
        with open(hist_pickle, "rb") as f:
            hist = pickle.load(f)
        train_loss = list(hist.get("total_loss_train", train_loss or []))
        val_loss = list(hist.get("total_loss_val", val_loss or []))
        test_loss = list(hist.get("total_loss_test", test_loss or []))

    train_loss = list(train_loss or [])
    val_loss = list(val_loss or [])
    test_loss = list(test_loss or [])

    # Trim trailing zeros (HydraGNN pre-allocates arrays).
    def _trim(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return a
        nz_idx = np.where(a != 0)[0]
        if nz_idx.size == 0:
            return a[:0]
        return a[: nz_idx.max() + 1]

    train_loss = _trim(train_loss)
    val_loss = _trim(val_loss)
    test_loss = _trim(test_loss)
    n = max(len(train_loss), len(val_loss), len(test_loss))
    if n == 0:
        # Empty plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "no loss history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    epochs = np.arange(1, n + 1)

    def _safe_log(arr, n_target):
        a = np.asarray(arr, dtype=np.float64)
        if a.size < n_target:
            a = np.pad(a, (0, n_target - a.size), constant_values=np.nan)
        a = np.where(a > 0, a, np.nan)
        return a

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        epochs, _safe_log(train_loss, n), "-", lw=1.6, label="train", color="#1f77b4"
    )
    ax.plot(epochs, _safe_log(val_loss, n), "-", lw=1.6, label="val", color="#ff7f0e")
    if any(test_loss):
        ax.plot(
            epochs,
            _safe_log(test_loss, n),
            "--",
            lw=1.2,
            label="test",
            color="#2ca02c",
            alpha=0.7,
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"Loss history — {n} epochs")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if best_epoch is None and len(val_loss) > 0:
        # NaN-safe: an all-NaN val_loss means validation never reported a
        # finite loss (e.g. degenerate target hit NaN every epoch). Skip
        # the best-epoch annotation rather than crashing.
        try:
            log_val = _safe_log(val_loss, n)
            if np.all(np.isnan(log_val)):
                best_epoch = None
            else:
                best_epoch = int(np.nanargmin(log_val) + 1)
        except (ValueError, TypeError):
            best_epoch = None
    if best_epoch is not None and best_epoch <= n:
        ax.axvline(best_epoch, color="orange", lw=1.0, alpha=0.5)
        ax.annotate(
            f"best val ep {best_epoch}",
            xy=(best_epoch, ax.get_ylim()[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="orange",
        )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pred_vs_true_dist_overlay_plot(
    out_path: Path,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str,
    unit: str = "",
    n_bins: int = 40,
) -> None:
    """Overlay histograms + KDE of predicted vs true marginal distributions.

    Diagnoses whether the model COMPRESSES the dynamic range relative to truth
    (a classic MSE failure mode on heavy-tailed targets). The bulk should
    overlap; tail mismatch on the right side reveals tail compression.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    yt = yt[np.isfinite(yt)]
    yp = yp[np.isfinite(yp)]
    if yt.size == 0 or yp.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "no finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(
        yt,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#1f77b4",
        edgecolor="k",
        linewidth=0.4,
        label=f"true (n={yt.size})",
    )
    ax.hist(
        yp,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#ff7f0e",
        edgecolor="k",
        linewidth=0.4,
        label=f"pred (n={yp.size})",
    )

    try:
        from scipy.stats import gaussian_kde

        x = np.linspace(lo, hi, 256)
        if yt.std() > 1e-30:
            kt = gaussian_kde(yt)(x)
            ax.plot(x, kt, "-", lw=1.6, color="#1f77b4", label="true KDE")
        if yp.std() > 1e-30:
            kp = gaussian_kde(yp)(x)
            ax.plot(x, kp, "-", lw=1.6, color="#ff7f0e", label="pred KDE")
    except Exception:
        pass

    pct_pred_at_99 = float(np.quantile(yp, 0.99))
    pct_true_at_99 = float(np.quantile(yt, 0.99))
    compression = (
        pct_pred_at_99 / pct_true_at_99 if pct_true_at_99 != 0 else float("nan")
    )
    ax.text(
        0.04,
        0.96,
        f"true mean={yt.mean():.3g}, std={yt.std():.3g}\n"
        f"pred mean={yp.mean():.3g}, std={yp.std():.3g}\n"
        f"pct99 ratio (pred/true) = {compression:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )
    ax.set_xlabel(f"value {unit}")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def loss_history_per_head_plot(
    out_path: Path,
    task_history: Dict[int, Sequence[float]],
    head_names: Optional[Sequence[str]] = None,
    best_epoch: Optional[int] = None,
) -> None:
    """Per-task train-loss curves on one log-y panel.

    HydraGNN emits `train error of task<N>` per epoch — one curve per output
    head (e.g. task0 = per-grain node head, task1 = per-graph head with
    1 + n_aux dims). This plot shows them stacked on a shared log axis.
    Per-aux-head granularity is unavailable from HydraGNN's writer.
    """
    if not task_history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "no per-task history captured",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    task_ids = sorted(task_history.keys())
    n = max((len(task_history[t]) for t in task_ids), default=0)
    if n == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "empty per-task history",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    epochs = np.arange(1, n + 1)
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # Detect whether any task has negative loss values (e.g. Gaussian NLL,
    # which goes negative when σ is small). If so, use symlog axis so both
    # positive and negative values are visible — the previous log-only axis
    # masked NaN any value ≤ 0, which made NLL curves disappear after a few
    # epochs even though training was progressing fine.
    any_negative = False
    for t in task_ids:
        arr = np.asarray(task_history[t], dtype=np.float64)
        if arr.size and np.any(np.isfinite(arr) & (arr <= 0)):
            any_negative = True
            break
    for i, t in enumerate(task_ids):
        arr = np.asarray(task_history[t], dtype=np.float64)
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), constant_values=np.nan)
        if not any_negative:
            # Mask only on plain-log axes (else symlog handles negatives natively).
            arr = np.where(arr > 0, arr, np.nan)
        name = head_names[t] if head_names and t < len(head_names) else f"task{t}"
        ax.plot(epochs, arr, "-", lw=1.4, color=cmap(i % 10), label=name)
    if any_negative:
        # symlog with linthresh=0.01 gives a small linear region around zero
        # so the NLL trajectory across the sign change is smooth.
        ax.set_yscale("symlog", linthresh=0.01)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
        ylabel = "train loss (symlog; can go negative for NLL)"
    else:
        ax.set_yscale("log")
        ylabel = "train loss (log) - per output head"
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title("Per-head train-loss history")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    if best_epoch is not None and 0 < best_epoch <= n:
        ax.axvline(best_epoch, color="orange", lw=1.0, alpha=0.5)
        ax.annotate(
            f"best val ep {best_epoch}",
            xy=(best_epoch, ax.get_ylim()[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="orange",
        )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid plot helpers (v9 result-naming refactor 2026-05-06).
#
# Replace the explosion of per-(output × split × plot-type) PNGs with one
# grid PNG per (level × stat-family). E.g. instead of 4 cells × 3 splits = 12
# parity files, write one `parity_pn_max.png` with a 4×3 panel grid.
# ---------------------------------------------------------------------------


def _draw_parity_panel(
    ax,
    y_true,
    y_pred,
    *,
    panel_title=None,
    size_buckets=None,
    bucket_to_label=None,
    annotate=True,
    n_subsample=None,
    rng_seed=0,
):
    """Single-axes parity scatter, mirroring the standalone `parity_plot`
    body but writing to a caller-supplied Axes. Returns (n_used, r2, n_dropped)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        if size_buckets is not None and len(size_buckets) == len(finite):
            size_buckets = [b for b, ok in zip(size_buckets, finite) if ok]
    if y_true.size == 0:
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
        return 0, float("nan"), n_dropped
    # Subsample for legibility.
    if n_subsample is not None and y_true.size > n_subsample:
        idx = np.random.default_rng(rng_seed).choice(
            y_true.size, size=n_subsample, replace=False
        )
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        if size_buckets is not None and len(size_buckets) == y_true.size + (
            n_dropped or 0
        ):
            size_buckets = [size_buckets[i] for i in idx.tolist()]
    r2, pear, mape = _r2_pearson_mape(y_true, y_pred)
    if size_buckets is not None and len(size_buckets) == len(y_true):
        unique = sorted(set(int(b) for b in size_buckets))
        cmap = plt.get_cmap("viridis")
        for k, b in enumerate(unique):
            mask = np.array(size_buckets) == b
            ax.scatter(
                y_true[mask],
                y_pred[mask],
                s=8,
                alpha=0.7,
                edgecolor="none",
                color=cmap(k / max(len(unique) - 1, 1)),
            )
    else:
        ax.scatter(y_true, y_pred, s=8, alpha=0.7, edgecolor="none", color="C0")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=0.8)
    if annotate:
        ann = f"R²={r2:+.3f}\nN={y_true.size}"
        ax.text(
            0.04,
            0.96,
            ann,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
        )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)
    return y_true.size, r2, n_dropped


def parity_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
    bucket_to_label: Optional[Dict[int, str]] = None,
    per_panel_subsample: Optional[int] = 2000,
) -> None:
    """Multi-panel parity grid.

    `panels` is a list of dicts with keys: y_true, y_pred, title (optional),
    size_buckets (optional). Order is row-major: panel index = row * ncols + col.
    `row_labels` / `col_labels` annotate the outer borders.
    """
    n = len(panels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no panels", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig_w = max(2.4 * ncols, 6.0)
    fig_h = max(2.4 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        _draw_parity_panel(
            ax,
            panel["y_true"],
            panel["y_pred"],
            panel_title=panel.get("title"),
            size_buckets=panel.get("size_buckets"),
            bucket_to_label=bucket_to_label,
            n_subsample=per_panel_subsample,
            rng_seed=i,
        )
    # Hide unused axes.
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    # Outer row/col labels.
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(label, fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_dist_overlay_panel(ax, y_true, y_pred, *, panel_title=None, n_bins=40):
    """Single-axes distribution-overlay histogram + KDE. Mirrors
    `pred_vs_true_dist_overlay_plot` body."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    yt = yt[np.isfinite(yt)]
    yp = yp[np.isfinite(yp)]
    if yt.size == 0 or yp.size == 0:
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
        return
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    bins = np.linspace(lo, hi, n_bins + 1)
    ax.hist(
        yt,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#1f77b4",
        edgecolor="k",
        linewidth=0.3,
        label="true",
    )
    ax.hist(
        yp,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#ff7f0e",
        edgecolor="k",
        linewidth=0.3,
        label="pred",
    )
    try:
        from scipy.stats import gaussian_kde

        x = np.linspace(lo, hi, 200)
        if yt.std() > 1e-30:
            ax.plot(x, gaussian_kde(yt)(x), "-", lw=1.0, color="#1f77b4")
        if yp.std() > 1e-30:
            ax.plot(x, gaussian_kde(yp)(x), "-", lw=1.0, color="#ff7f0e")
    except Exception:
        pass
    p99r = (
        np.quantile(yp, 0.99) / np.quantile(yt, 0.99)
        if np.quantile(yt, 0.99) != 0
        else float("nan")
    )
    std_r = yp.std() / yt.std() if yt.std() > 0 else float("nan")
    ax.text(
        0.04,
        0.96,
        f"std_r={std_r:.2f}\np99_r={p99r:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.85, "boxstyle": "round,pad=0.2"},
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)


def dist_overlay_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
) -> None:
    """Multi-panel distribution-overlay grid. Same shape contract as
    `parity_grid_plot`; `panels` need only y_true, y_pred, title."""
    n = len(panels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no panels", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig_w = max(3.0 * ncols, 7.0)
    fig_h = max(2.4 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        _draw_dist_overlay_panel(
            ax, panel["y_true"], panel["y_pred"], panel_title=panel.get("title")
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(label, fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def hotspot_diagnostic_grid_plot(
    out_path: Path, splits_data: List[Dict], *, suptitle: str = ""
) -> None:
    """3-row stacked hotspot diagnostic: one row per split. Each row mirrors
    the 4-panel `per_grain_diagnostic_plot` (Spearman hist, top-K curve,
    hotspot loc err, parity colored by within-graph rank).

    `splits_data` = [{"split":"train","pred_per_graph":[...],"true_per_graph":[...]}, ...]
    """
    from scipy.stats import spearmanr as _spr

    n_splits = len(splits_data)
    if n_splits == 0:
        return
    fig, axes = plt.subplots(n_splits, 4, figsize=(15, 3.4 * n_splits), squeeze=False)
    for r, sd in enumerate(splits_data):
        ax_row = axes[r]
        pred_per_graph = sd["pred_per_graph"]
        true_per_graph = sd["true_per_graph"]
        ax_row[0].set_ylabel(sd.get("split", f"row{r}"), fontsize=10, fontweight="bold")
        if not pred_per_graph:
            for ax in ax_row:
                ax.text(
                    0.5,
                    0.5,
                    "empty",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            continue
        spears, hotspot_errs = [], []
        kfracs = [0.01, 0.05, 0.10, 0.20, 0.50]
        topk_curves = []
        flat_pred, flat_true, flat_color = [], [], []
        for pred, true in zip(pred_per_graph, true_per_graph):
            n = len(pred)
            if n < 5:
                continue
            try:
                s = _spr(true, pred).statistic
                if s == s:
                    spears.append(float(s))
            except Exception:
                pass
            rec = []
            for kf in kfracs:
                k = max(1, int(np.ceil(kf * n)))
                top_t = set(np.argsort(true)[-k:].tolist())
                top_p = set(np.argsort(pred)[-k:].tolist())
                rec.append(len(top_t & top_p) / k)
            topk_curves.append(rec)
            true_max_idx = int(np.argmax(true))
            rank = int(np.sum(np.asarray(pred) >= pred[true_max_idx]))
            hotspot_errs.append((rank - 1) / max(n - 1, 1))
            ranks = np.argsort(np.argsort(true)) / max(n - 1, 1)
            flat_pred.extend(pred)
            flat_true.extend(true)
            flat_color.extend(ranks)
        if spears:
            ax_row[0].hist(
                spears,
                bins=20,
                color="#1f77b4",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.4,
            )
            ax_row[0].set_title(
                f"Spearman per graph (μ={np.mean(spears):.3f})", fontsize=8
            )
        else:
            ax_row[0].text(
                0.5,
                0.5,
                "no Spearman data",
                ha="center",
                transform=ax_row[0].transAxes,
                fontsize=8,
            )
        if topk_curves:
            tc = np.array(topk_curves)
            mean = tc.mean(axis=0)
            p25 = np.quantile(tc, 0.25, axis=0)
            p75 = np.quantile(tc, 0.75, axis=0)
            xs = np.array(kfracs) * 100
            ax_row[1].plot(xs, mean, "-o", color="#1f77b4")
            ax_row[1].fill_between(xs, p25, p75, alpha=0.25, color="#1f77b4")
            ax_row[1].set_xlabel("top-K %", fontsize=7)
            ax_row[1].set_ylim(0, 1)
            ax_row[1].set_title("Top-K recall", fontsize=8)
        if hotspot_errs:
            ax_row[2].hist(
                hotspot_errs,
                bins=20,
                color="#d62728",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.4,
            )
            ax_row[2].set_title(
                f"Hotspot loc err (μ={np.mean(hotspot_errs):.3f})", fontsize=8
            )
        if flat_true:
            yp = np.asarray(flat_pred)
            yt = np.asarray(flat_true)
            cl = np.asarray(flat_color)
            sc = ax_row[3].scatter(
                yt, yp, c=cl, s=8, cmap="plasma", alpha=0.7, edgecolor="none"
            )
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            ax_row[3].plot([lo, hi], [lo, hi], "k--", lw=0.6)
            ax_row[3].set_title("Parity colored by within-graph rank", fontsize=8)
        for ax in ax_row:
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase L5 σ-output diagnostic plots (heteroscedastic head visualization).
# ---------------------------------------------------------------------------


def _draw_sigma_residual_panel(
    ax, sigma, residuals, *, panel_title=None, bucket=None, n_subsample=2000
):
    """log_σ vs |residual| scatter with within-panel Spearman annotation.
    Positive Spearman ⇒ σ ranks errors correctly within this cell/split."""
    s = np.asarray(sigma, dtype=np.float64).ravel()
    r = np.asarray(residuals, dtype=np.float64).ravel()
    finite = np.isfinite(s) & np.isfinite(r)
    s, r = s[finite], r[finite]
    if bucket is not None:
        bucket = np.asarray(bucket)[finite]
    if s.size == 0:
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
        return
    if n_subsample is not None and s.size > n_subsample:
        idx = np.random.default_rng(0).choice(s.size, size=n_subsample, replace=False)
        s, r = s[idx], r[idx]
        if bucket is not None:
            bucket = bucket[idx]
    from scipy.stats import spearmanr

    try:
        spear, _ = spearmanr(s, r)
        spear = float(spear) if spear == spear else float("nan")
    except Exception:
        spear = float("nan")
    if bucket is not None:
        unique = sorted(set(int(b) for b in bucket))
        cmap = plt.get_cmap("viridis")
        for k, b in enumerate(unique):
            m = np.array(bucket) == b
            ax.scatter(
                s[m],
                r[m],
                s=8,
                alpha=0.65,
                edgecolor="none",
                color=cmap(k / max(len(unique) - 1, 1)),
            )
    else:
        ax.scatter(s, r, s=8, alpha=0.6, edgecolor="none", color="C0")
    ax.text(
        0.04,
        0.96,
        f"Spearman={spear:+.3f}\nN={s.size}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)


def sigma_residual_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
) -> None:
    """Multi-panel σ vs |residual| scatter."""
    n = len(panels)
    if n == 0:
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(2.6 * ncols, 7.0), max(2.6 * nrows, 5.0)),
        squeeze=False,
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        _draw_sigma_residual_panel(
            axes[r][c],
            panel["sigma"],
            panel["residuals"],
            panel_title=panel.get("title"),
            bucket=panel.get("bucket"),
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(
                f"{label}\n|residual|", fontsize=10, fontweight="bold"
            )
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    for ax in axes[-1]:
        ax.set_xlabel("predicted log_σ", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_sigma_histogram_panel(ax, sigma, *, panel_title=None, n_bins=40):
    s = np.asarray(sigma, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
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
        return
    ax.hist(s, bins=n_bins, color="#2ca02c", alpha=0.75, edgecolor="k", linewidth=0.3)
    ax.axvline(float(s.mean()), color="r", lw=1.0, linestyle="--")
    ax.text(
        0.04,
        0.96,
        f"N={s.size}\nmean={s.mean():.3f}\nstd={s.std():.3f}\n"
        f"p10={np.quantile(s, 0.10):.3f}\np90={np.quantile(s, 0.90):.3f}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)


def sigma_histogram_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
) -> None:
    """Multi-panel σ histogram. Each panel: {sigma, title (optional)}."""
    n = len(panels)
    if n == 0:
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(2.6 * ncols, 7.0), max(2.4 * nrows, 4.0)),
        squeeze=False,
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        _draw_sigma_histogram_panel(
            axes[r][c], panel["sigma"], panel_title=panel.get("title")
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(f"{label}\nfreq", fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    for ax in axes[-1]:
        ax.set_xlabel("predicted log_σ", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_parity_by_sigma_panel(
    ax, y_true, y_pred, sigma, *, panel_title=None, n_subsample=2000
):
    """Parity scatter colored by predicted log_σ. High-σ points should
    cluster in off-diagonal regions if σ honestly tracks model uncertainty."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(sigma, dtype=np.float64).ravel()
    finite = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(s)
    yt, yp, s = yt[finite], yp[finite], s[finite]
    if yt.size == 0:
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
        return None
    if n_subsample is not None and yt.size > n_subsample:
        idx = np.random.default_rng(0).choice(yt.size, size=n_subsample, replace=False)
        yt, yp, s = yt[idx], yp[idx], s[idx]
    sc = ax.scatter(yt, yp, c=s, cmap="plasma", s=10, alpha=0.75, edgecolor="none")
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)
    return sc


def _draw_parity_with_uq_panel(
    ax,
    y_true,
    y_pred,
    sigma_raw,
    *,
    panel_title=None,
    z=1.65,
    n_subsample=150,
    size_buckets=None,
):
    """Parity scatter with vertical error bars ±z·σ_pred on a subsample.
    Annotates empirical coverage = fraction of points with |y_true − y_pred| ≤ z·σ.
    For z=1.65 a perfectly-calibrated model gives 0.90 coverage.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(sigma_raw, dtype=np.float64).ravel()
    finite = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(s)
    yt, yp, s = yt[finite], yp[finite], s[finite]
    if size_buckets is not None:
        size_buckets = np.asarray(size_buckets)[finite]
    if yt.size == 0:
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
        return
    cov = float(((yp - yt).__abs__() <= z * s).mean())
    cov_50 = float(((yp - yt).__abs__() <= 0.674 * s).mean())  # 50%-PI
    if n_subsample is not None and yt.size > n_subsample:
        idx = np.random.default_rng(0).choice(yt.size, size=n_subsample, replace=False)
        yt_p, yp_p, s_p = yt[idx], yp[idx], s[idx]
        sb_p = size_buckets[idx] if size_buckets is not None else None
    else:
        yt_p, yp_p, s_p = yt, yp, s
        sb_p = size_buckets
    if sb_p is not None:
        unique = sorted(set(int(b) for b in sb_p))
        cmap = plt.get_cmap("viridis")
        for k, b in enumerate(unique):
            m = np.array(sb_p) == b
            ax.errorbar(
                yt_p[m],
                yp_p[m],
                yerr=z * s_p[m],
                fmt="o",
                ms=3,
                ecolor="grey",
                elinewidth=0.5,
                capsize=0,
                alpha=0.7,
                mec="none",
                color=cmap(k / max(len(unique) - 1, 1)),
            )
    else:
        ax.errorbar(
            yt_p,
            yp_p,
            yerr=z * s_p,
            fmt="o",
            ms=3,
            ecolor="grey",
            elinewidth=0.5,
            capsize=0,
            alpha=0.7,
            mec="none",
            color="C0",
        )
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=0.7)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    nominal = (
        z * 100 * 2
        if z == 0.5
        else (90 if abs(z - 1.65) < 0.01 else int(2 * z * 100 / 3.29 * 90))
    )
    cov_label = f"cov(±{z:.2f}σ)={cov:.3f}\ncov(±0.67σ)={cov_50:.3f}"
    ax.text(
        0.04,
        0.96,
        f"{cov_label}\nN={yt.size}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)


def parity_with_uq_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
    z: float = 1.65,
    n_subsample: int = 150,
) -> None:
    """Parity scatter with ±z·σ vertical error bars on a subsample. Each panel
    needs {y_true, y_pred, sigma_raw, title (optional), bucket (optional)}.
    """
    n = len(panels)
    if n == 0:
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(2.6 * ncols, 7.0), max(2.6 * nrows, 5.0)),
        squeeze=False,
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        _draw_parity_with_uq_panel(
            axes[r][c],
            panel["y_true"],
            panel["y_pred"],
            panel["sigma_raw"],
            panel_title=panel.get("title"),
            size_buckets=panel.get("bucket"),
            z=z,
            n_subsample=n_subsample,
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(f"{label}\npred", fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    for ax in axes[-1]:
        ax.set_xlabel("y_true (raw FIP)", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_dist_overlay_with_uq_panel(
    ax, y_true, y_pred, sigma_raw, *, panel_title=None, n_bins=40, rng_seed=0
):
    """Distribution overlay: true (blue) + pred-direct (orange) + pred-with-σ
    (green = one Gaussian sample per grain at μ_pred, σ²). If σ is honestly
    calibrated, the green curve should match the true distribution better
    than the orange (pred-direct) — i.e. σ is "filling in" the missing
    dispersion that point predictions miss.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(sigma_raw, dtype=np.float64).ravel()
    finite = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(s)
    yt, yp, s = yt[finite], yp[finite], s[finite]
    if yt.size == 0:
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
        return
    rng = np.random.default_rng(rng_seed)
    yp_with_sigma = yp + rng.standard_normal(yt.size) * s
    lo = float(min(yt.min(), yp.min(), yp_with_sigma.min()))
    hi = float(max(yt.max(), yp.max(), yp_with_sigma.max()))
    bins = np.linspace(lo, hi, n_bins + 1)
    ax.hist(
        yt,
        bins=bins,
        density=True,
        alpha=0.40,
        color="#1f77b4",
        edgecolor="k",
        linewidth=0.3,
        label="true",
    )
    ax.hist(
        yp,
        bins=bins,
        density=True,
        alpha=0.40,
        color="#ff7f0e",
        edgecolor="k",
        linewidth=0.3,
        label="pred",
    )
    ax.hist(
        yp_with_sigma,
        bins=bins,
        density=True,
        alpha=0.40,
        color="#2ca02c",
        edgecolor="k",
        linewidth=0.3,
        label="pred + σ noise",
    )
    try:
        from scipy.stats import gaussian_kde

        x = np.linspace(lo, hi, 200)
        if yt.std() > 1e-30:
            ax.plot(x, gaussian_kde(yt)(x), "-", lw=1.0, color="#1f77b4")
        if yp.std() > 1e-30:
            ax.plot(x, gaussian_kde(yp)(x), "-", lw=1.0, color="#ff7f0e")
        if yp_with_sigma.std() > 1e-30:
            ax.plot(x, gaussian_kde(yp_with_sigma)(x), "-", lw=1.0, color="#2ca02c")
    except Exception:
        pass
    p99r_pred = (
        np.quantile(yp, 0.99) / np.quantile(yt, 0.99)
        if np.quantile(yt, 0.99) != 0
        else float("nan")
    )
    p99r_uq = (
        np.quantile(yp_with_sigma, 0.99) / np.quantile(yt, 0.99)
        if np.quantile(yt, 0.99) != 0
        else float("nan")
    )
    ax.text(
        0.04,
        0.96,
        f"std_t={yt.std():.3g}\nstd_p={yp.std():.3g}\nstd_p+σ={yp_with_sigma.std():.3g}\n"
        f"p99_p/t={p99r_pred:.2f}\np99_(p+σ)/t={p99r_uq:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
    )
    ax.tick_params(labelsize=7)
    if panel_title:
        ax.set_title(panel_title, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=7)


def dist_overlay_with_uq_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
) -> None:
    """Multi-panel dist overlay with σ-noised predicted distribution as a 3rd curve."""
    n = len(panels)
    if n == 0:
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(3.0 * ncols, 7.5), max(2.6 * nrows, 5.0)),
        squeeze=False,
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        _draw_dist_overlay_with_uq_panel(
            axes[r][c],
            panel["y_true"],
            panel["y_pred"],
            panel["sigma_raw"],
            panel_title=panel.get("title"),
            rng_seed=i,
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(f"{label}\ndensity", fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    for ax in axes[-1]:
        ax.set_xlabel("FIP value (raw)", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parity_by_sigma_grid_plot(
    out_path: Path,
    panels: List[Dict],
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
) -> None:
    """Parity scatters colored by predicted log_σ + shared colorbar."""
    n = len(panels)
    if n == 0:
        return
    if ncols is None and nrows is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(2.6 * ncols, 7.0), max(2.6 * nrows, 5.0)),
        squeeze=False,
    )
    last_sc = None
    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        sc = _draw_parity_by_sigma_panel(
            axes[r][c],
            panel["y_true"],
            panel["y_pred"],
            panel["sigma"],
            panel_title=panel.get("title"),
        )
        if sc is not None:
            last_sc = sc
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if row_labels is not None:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].set_ylabel(f"{label}\npred", fontsize=10, fontweight="bold")
    if col_labels is not None:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(
                (axes[0][c].get_title() + "\n" if axes[0][c].get_title() else "")
                + f"[{label}]",
                fontsize=9,
                fontweight="bold",
            )
    for ax in axes[-1]:
        ax.set_xlabel("y_true", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    if last_sc is not None:
        fig.subplots_adjust(right=0.92)
        cbar = fig.colorbar(
            last_sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, aspect=30
        )
        cbar.set_label("predicted log_σ", fontsize=9)
        cbar.ax.tick_params(labelsize=7)
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
