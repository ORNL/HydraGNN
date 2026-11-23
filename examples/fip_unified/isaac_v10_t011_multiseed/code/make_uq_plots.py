"""make_uq_plots.py — generate all six UQ diagnostic plots from
mcd_predictions.csv across the 25 Track-E reps.

Outputs go to v5_study/uq_plots/0N_<plot_name>/ — each plot in its own
subfolder so the user can browse and pick which ones land in the paper.

Plots:
  01_calibration_curve   — empirical coverage vs nominal probability,
                            for Normal/Gumbel/combined aleatoric.
  02_epi_vs_ale_scatter  — sigma_E (MCD) vs sigma_A (Normal max-of-N),
                            colored by bucket.
  03_ci_width_vs_N       — 90% CI half-width vs N_grains, log scale,
                            with sqrt(log N) trend overlay.
  04_per_bucket_coverage — bar chart of P_cov per L_SVE bucket for
                            Normal vs Gumbel constructions.
  05_posterior_predictive_density — for one representative SVE per
                            bucket, the K=30 MCD passes histogram
                            overlaid with the Normal max-of-N density
                            and the true value as a vertical line.
  06_multi_split_parity  — pooled 1250-SVE parity scatter with 90%
                            combined CI bars, color by split-seed.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
PLOTS = STUDY / "uq_plots"

GAMMA_E = 0.5772156649
BETA_FLOOR = 1e-9  # clip predicted β to be strictly positive before forming
# σ_y = β π/√6 — guards against the inverse-z-score occasionally
# producing a negative β when σ(β̂) is small relative to mean(β̂).
# Global training-set within-SVE std of per-grain MAX. Computed pooled across
# the full dataset (998 SVEs); per-split values differ from this by < 1%.
# Used in place of β̂·π/√6 in the principled aleatoric construction.
SIGMA_POP = 0.001520


def _beta_safe(beta):
    """Floor predicted β so that σ_y = β π/√6 stays non-negative. β is a
    scale parameter and must be > 0; predictions can drift negative after
    inverse z-score when the head collapses near the global mean."""
    return max(float(beta), BETA_FLOOR)


# Mode flag — switch to True to use the principled σ_pop construction
# (Eq. eq:sigma-pop in the paper) in place of β̂·π/√6 from the head.
USE_SIGMA_POP = True
BETA_POP = SIGMA_POP * np.sqrt(6) / np.pi  # back-compute β_pop for ȳ shift


# ---------------------------------------------------------------------------
# Load all MCD predictions across reps
# ---------------------------------------------------------------------------
def load_all_predictions():
    rows = []
    for csv_path in sorted(
        glob.glob(str(RESULTS / "E_M1_K1_dropout_split*_seed*/mcd_predictions.csv"))
    ):
        # parse split / seed from path
        parent = Path(csv_path).parent.name
        split = int(parent.split("split")[1].split("_")[0])
        seed = int(parent.split("seed")[1])
        for r in csv.DictReader(open(csv_path)):
            r["split"] = split
            r["seed"] = seed
            for k in (
                "y_true_raw",
                "pg_fip_mean",
                "pg_fip_std",
                "gumbel_mu_mean",
                "gumbel_mu_std",
                "gumbel_beta_mean",
                "gumbel_beta_std",
                "pn_max_mean",
                "pn_max_std",
            ):
                r[k] = float(r.get(k, "nan"))
            r["bucket"] = int(r["bucket"])
            r["n_grains"] = int(r["n_grains"])
            rows.append(r)
    return rows


def normal_max_quantile(mean, sigma, N, p):
    return mean + sigma * norm.ppf(p ** (1.0 / N))


def gumbel_max_quantile(mu, beta, N, p):
    """p-quantile of max of N i.i.d. Gumbel(mu, beta) is Gumbel(mu+beta*ln N, beta)."""
    mu_max = mu + beta * log(N)
    # F^{-1}(p) for Gumbel(mu_max, beta) = mu_max - beta * ln(-ln p)
    return mu_max - beta * np.log(-np.log(p))


def _bar_y_and_sigma(row):
    """Resolve (ȳ, σ) used in the Normal max-of-N quantile, per
    USE_SIGMA_POP. When True, σ is the global training-set σ_pop and ȳ is
    the head's predicted µ shifted by β_pop·γ_E to align with the within-SVE
    mean (see paper §2.7); when False, fall back to the head's β prediction
    which the original cell trained on."""
    mu_pred = float(row["gumbel_mu_mean"])
    if USE_SIGMA_POP:
        bar_y = mu_pred + BETA_POP * GAMMA_E
        sigma_y = SIGMA_POP
    else:
        beta_pred = _beta_safe(row["gumbel_beta_mean"])
        bar_y = mu_pred + beta_pred * GAMMA_E
        sigma_y = beta_pred * np.pi / np.sqrt(6)
    return bar_y, sigma_y


def combined_normal_ci(row, p):
    """Lower/upper of a (1-2*(1-p))*100 % combined epistemic+aleatoric CI."""
    alpha = (1.0 - p) / 2.0
    p_lo, p_hi = alpha, 1.0 - alpha
    bar_y, sigma_y = _bar_y_and_sigma(row)
    N = row["n_grains"]
    lo_an = normal_max_quantile(bar_y, sigma_y, N, p_lo)
    hi_an = normal_max_quantile(bar_y, sigma_y, N, p_hi)
    sigma_ale = (hi_an - lo_an) / (2.0 * norm.ppf(p_hi))
    sigma_total = np.sqrt(row["pg_fip_std"] ** 2 + sigma_ale ** 2)
    center = (hi_an + lo_an) / 2.0
    z = norm.ppf(p_hi)
    return center - z * sigma_total, center + z * sigma_total


def normal_ale_ci(row, p):
    alpha = (1.0 - p) / 2.0
    bar_y, sigma_y = _bar_y_and_sigma(row)
    N = row["n_grains"]
    return (
        normal_max_quantile(bar_y, sigma_y, N, alpha),
        normal_max_quantile(bar_y, sigma_y, N, 1 - alpha),
    )


def gumbel_ale_ci(row, p):
    alpha = (1.0 - p) / 2.0
    mu_pred = row["gumbel_mu_mean"]
    beta_pred = _beta_safe(row["gumbel_beta_mean"])
    N = row["n_grains"]
    return (
        gumbel_max_quantile(mu_pred, beta_pred, N, alpha),
        gumbel_max_quantile(mu_pred, beta_pred, N, 1 - alpha),
    )


def epi_only_ci(row, p):
    z = norm.ppf((1 + p) / 2.0)
    c = row["pg_fip_mean"]
    s = row["pg_fip_std"]
    return c - z * s, c + z * s


# ---------------------------------------------------------------------------
# 01 calibration curve
# ---------------------------------------------------------------------------
def plot_01_calibration(rows):
    out = PLOTS / "01_calibration_curve"
    ps = np.linspace(0.05, 0.95, 19)
    cov_norm, cov_gum, cov_combo, cov_epi = [], [], [], []
    for p in ps:
        n = 0
        cN = cG = cC = cE = 0
        for r in rows:
            yt = r["y_true_raw"]
            n += 1
            lo, hi = normal_ale_ci(r, p)
            cN += int(lo <= yt <= hi)
            lo, hi = gumbel_ale_ci(r, p)
            cG += int(lo <= yt <= hi)
            lo, hi = combined_normal_ci(r, p)
            cC += int(lo <= yt <= hi)
            lo, hi = epi_only_ci(r, p)
            cE += int(lo <= yt <= hi)
        cov_norm.append(cN / n)
        cov_gum.append(cG / n)
        cov_combo.append(cC / n)
        cov_epi.append(cE / n)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (calibrated)")
    ax.plot(ps, cov_norm, "o-", color="C0", lw=1.5, label="Normal max-of-$N$ aleatoric")
    ax.plot(ps, cov_gum, "s-", color="C3", lw=1.5, label="Gumbel max-of-$N$ aleatoric")
    ax.plot(
        ps,
        cov_combo,
        "^-",
        color="C2",
        lw=1.5,
        label="Normal aleatoric + MCD epistemic",
    )
    ax.plot(
        ps, cov_epi, "v-", color="C5", lw=1.0, alpha=0.7, label="MCD epistemic only"
    )
    ax.set_xlabel("nominal CI probability $p$")
    ax.set_ylabel("empirical coverage $P_{\\mathrm{cov}}(p)$")
    ax.set_title(f"UQ calibration on {len(rows)} test SVEs (25 Track-E reps)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=160)
    plt.close(fig)
    # also save numerical table
    with open(out / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "p_nominal",
                "cov_normal_ale",
                "cov_gumbel_ale",
                "cov_combined_normal",
                "cov_mcd_epi",
            ]
        )
        for i, p in enumerate(ps):
            w.writerow(
                [
                    f"{p:.2f}",
                    f"{cov_norm[i]:.3f}",
                    f"{cov_gum[i]:.3f}",
                    f"{cov_combo[i]:.3f}",
                    f"{cov_epi[i]:.3f}",
                ]
            )
    print(f"  [01] saved {out / 'calibration_curve.png'}")


# ---------------------------------------------------------------------------
# 02 epi vs ale scatter
# ---------------------------------------------------------------------------
def plot_02_epi_vs_ale(rows):
    out = PLOTS / "02_epi_vs_ale_scatter"
    sigma_E, sigma_A, buckets = [], [], []
    for r in rows:
        bar_y, sigma_y = _bar_y_and_sigma(r)
        N = r["n_grains"]
        # aleatoric "1-sigma" on max-of-N: half-width of 90% / (2*1.645)
        z = norm.ppf(0.95)
        Q_lo = normal_max_quantile(bar_y, sigma_y, N, 0.05)
        Q_hi = normal_max_quantile(bar_y, sigma_y, N, 0.95)
        sigma_a = (Q_hi - Q_lo) / (2 * z)
        sigma_E.append(r["pg_fip_std"])
        sigma_A.append(sigma_a)
        buckets.append(r["bucket"])
    sigma_E = np.array(sigma_E)
    sigma_A = np.array(sigma_A)
    buckets = np.array(buckets)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis")
    bucket_labels = {0: "125 µm", 1: "177 µm", 2: "250 µm", 3: "354 µm", 4: "500 µm"}
    for b in sorted(set(buckets)):
        m = buckets == b
        ax.scatter(
            sigma_E[m],
            sigma_A[m],
            s=22,
            alpha=0.55,
            color=cmap(b / 4.0),
            label=bucket_labels[b],
            edgecolor="none",
        )
    diag = np.geomspace(max(sigma_E.min(), 1e-6), sigma_A.max(), 50)
    ax.plot(diag, diag, "k--", lw=1, alpha=0.5, label="$\\sigma_A = \\sigma_E$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\sigma_E$ — MCD epistemic std")
    ax.set_ylabel("$\\sigma_A$ — Normal max-of-$N$ aleatoric std")
    ax.set_title(
        f"epistemic vs aleatoric on {len(rows)} test SVEs\n"
        f"mean $\\sigma_A / \\sigma_E$ = {sigma_A.mean()/sigma_E.mean():.2f}"
    )
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", title="$L_{\\mathrm{SVE}}$")
    fig.tight_layout()
    fig.savefig(out / "epi_vs_ale.png", dpi=160)
    plt.close(fig)
    print(f"  [02] saved {out / 'epi_vs_ale.png'}")


# ---------------------------------------------------------------------------
# 03 CI width vs N_grains
# ---------------------------------------------------------------------------
def plot_03_ci_vs_N(rows):
    out = PLOTS / "03_ci_width_vs_N"
    Ns, halfwidths_combo, halfwidths_aleN, halfwidths_aleG = [], [], [], []
    buckets = []
    for r in rows:
        N = r["n_grains"]
        lo, hi = combined_normal_ci(r, 0.90)
        halfwidths_combo.append((hi - lo) / 2)
        lo, hi = normal_ale_ci(r, 0.90)
        halfwidths_aleN.append((hi - lo) / 2)
        lo, hi = gumbel_ale_ci(r, 0.90)
        halfwidths_aleG.append((hi - lo) / 2)
        Ns.append(N)
        buckets.append(r["bucket"])
    Ns = np.array(Ns)
    buckets = np.array(buckets)
    halfwidths_combo = np.array(halfwidths_combo)
    halfwidths_aleN = np.array(halfwidths_aleN)
    halfwidths_aleG = np.array(halfwidths_aleG)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.get_cmap("viridis")
    bucket_labels = {0: "125 µm", 1: "177 µm", 2: "250 µm", 3: "354 µm", 4: "500 µm"}
    for b in sorted(set(buckets)):
        m = buckets == b
        ax.scatter(
            Ns[m],
            halfwidths_aleN[m],
            s=18,
            alpha=0.45,
            color=cmap(b / 4.0),
            edgecolor="none",
            label=bucket_labels[b],
        )
    # theoretical scaling: half-width ∝ 1 / sqrt(log N) up to a constant
    Ngrid = np.geomspace(Ns.min(), Ns.max(), 60)
    sigma0 = np.median(halfwidths_aleN[buckets == 2])  # anchor at bucket 2 median
    N_anchor = np.median(Ns[buckets == 2])
    scale = sigma0 * np.sqrt(np.log(N_anchor))
    trend = scale / np.sqrt(np.log(Ngrid))
    ax.plot(
        Ngrid, trend, "k--", lw=1, alpha=0.5, label="trend $\\propto 1/\\sqrt{\\log N}$"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ — number of grains in SVE")
    ax.set_ylabel(
        "$90\\%$ CI half-width on $\\widehat{\\mathrm{FIP}^m_{\\mathrm{FS}}}$"
    )
    ax.set_title(f"Aleatoric CI width vs grain count ({len(rows)} test SVEs)")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", title="$L_{\\mathrm{SVE}}$")
    fig.tight_layout()
    fig.savefig(out / "ci_width_vs_N.png", dpi=160)
    plt.close(fig)
    print(f"  [03] saved {out / 'ci_width_vs_N.png'}")


# ---------------------------------------------------------------------------
# 04 per-bucket coverage
# ---------------------------------------------------------------------------
def plot_04_per_bucket_coverage(rows):
    out = PLOTS / "04_per_bucket_coverage"
    buckets = sorted(set(r["bucket"] for r in rows))
    bucket_labels = ["125 µm", "177 µm", "250 µm", "354 µm", "500 µm"]
    cov_norm = []
    cov_gum = []
    cov_combo = []
    n_per = []
    for b in buckets:
        bs = [r for r in rows if r["bucket"] == b]
        n = len(bs)
        cN = cG = cC = 0
        for r in bs:
            yt = r["y_true_raw"]
            lo, hi = normal_ale_ci(r, 0.90)
            cN += int(lo <= yt <= hi)
            lo, hi = gumbel_ale_ci(r, 0.90)
            cG += int(lo <= yt <= hi)
            lo, hi = combined_normal_ci(r, 0.90)
            cC += int(lo <= yt <= hi)
        cov_norm.append(cN / n)
        cov_gum.append(cG / n)
        cov_combo.append(cC / n)
        n_per.append(n)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(buckets))
    w = 0.27
    ax.bar(x - w, cov_norm, w, label="Normal max-of-$N$ aleatoric", color="C0")
    ax.bar(x, cov_combo, w, label="combined Normal (epi+ale)", color="C2")
    ax.bar(x + w, cov_gum, w, label="Gumbel max-of-$N$ aleatoric", color="C3")
    ax.axhline(0.90, color="k", ls="--", lw=1, alpha=0.6, label="nominal 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lab}\n($n={n_per[i]}$)" for i, lab in enumerate(bucket_labels)]
    )
    ax.set_xlabel("$L_{\\mathrm{SVE}}$ bucket")
    ax.set_ylabel("$P_{\\mathrm{cov}}$ on 90% CI")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-bucket UQ coverage (25 Track-E reps)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "per_bucket_coverage.png", dpi=160)
    plt.close(fig)
    print(f"  [04] saved {out / 'per_bucket_coverage.png'}")


# ---------------------------------------------------------------------------
# 05 per-SVE posterior predictive density
# ---------------------------------------------------------------------------
def plot_05_posterior_density(rows):
    out = PLOTS / "05_posterior_predictive_density"
    # Pick one representative SVE per bucket (median-error rep on split-seed=42)
    bucket_labels = ["125 µm", "177 µm", "250 µm", "354 µm", "500 µm"]
    # Group by bucket on split=42
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for b in range(5):
        bs = [r for r in rows if r["bucket"] == b and r["split"] == 42]
        if not bs:
            continue
        # pick the SVE whose pg_fip_mean error is closest to the median
        errs = np.array([abs(r["pg_fip_mean"] - r["y_true_raw"]) for r in bs])
        med_idx = np.argsort(errs)[len(errs) // 2]
        r = bs[med_idx]
        ax = axes[b]
        # K=30 stochastic predictions: we don't have the per-pass values in the
        # CSV, only mean/std. Reconstruct an idealised Gaussian as the
        # epistemic predictive density.
        c = r["pg_fip_mean"]
        s = r["pg_fip_std"]
        x = np.linspace(c - 4 * s, c + 4 * s, 200)
        epi_pdf = norm.pdf(x, c, s)
        ax.fill_between(x, epi_pdf, alpha=0.4, color="C5", label="MCD epistemic")
        # Normal max-of-N aleatoric density
        bar_y, sigma_y = _bar_y_and_sigma(r)
        N = r["n_grains"]
        # max-of-N density: f_max(x) = N * f_Y(x) * F_Y(x)^(N-1)
        x2 = np.linspace(bar_y - 4 * sigma_y, bar_y + 6 * sigma_y, 400)
        f_y = norm.pdf(x2, bar_y, sigma_y)
        F_y = norm.cdf(x2, bar_y, sigma_y)
        with np.errstate(invalid="ignore"):
            f_max = N * f_y * F_y ** (N - 1)
        # rescale axes for visualization on common x-range
        x_all = np.union1d(x, x2)
        ax.plot(x2, f_max, color="C0", lw=2, label=f"Normal max-of-$N={N}$ aleatoric")
        ax.axvline(
            r["y_true_raw"],
            color="k",
            ls="-",
            lw=1.5,
            label="true $\\mathrm{FIP}^m_{\\mathrm{FS}}$",
        )
        ax.axvline(
            r["pg_fip_mean"], color="C5", ls=":", lw=1.5, label="MCD posterior mean"
        )
        ax.set_title(f"{bucket_labels[b]} (N={N})")
        ax.set_xlabel("$\\mathrm{FIP}^m_{\\mathrm{FS}}$")
        if b == 0:
            ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        if b == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Posterior predictive: epistemic (MCD) + aleatoric (Normal max-of-$N$)\n"
        "one representative test SVE per bucket from split-seed 42"
    )
    fig.tight_layout()
    fig.savefig(out / "posterior_density_by_bucket.png", dpi=160)
    plt.close(fig)
    print(f"  [05] saved {out / 'posterior_density_by_bucket.png'}")


# ---------------------------------------------------------------------------
# 06 multi-split aggregate parity with CI bars
# ---------------------------------------------------------------------------
def plot_06_multi_split_parity(rows):
    out = PLOTS / "06_multi_split_parity"
    fig, ax = plt.subplots(figsize=(7, 7))
    splits = sorted(set(r["split"] for r in rows))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(splits):
        bs = [r for r in rows if r["split"] == s]
        yt = np.array([r["y_true_raw"] for r in bs])
        yp = np.array([r["pg_fip_mean"] for r in bs])
        # 90% combined CI
        lo = []
        hi = []
        for r in bs:
            l, h = combined_normal_ci(r, 0.90)
            lo.append(l)
            hi.append(h)
        lo = np.array(lo)
        hi = np.array(hi)
        # error bars
        ax.errorbar(
            yt,
            yp,
            yerr=[yp - lo, hi - yp],
            fmt="o",
            ms=2,
            alpha=0.5,
            color=cmap(i / 5.0),
            ecolor=cmap(i / 5.0),
            elinewidth=0.5,
            label=f"split-seed {s}",
        )
    diag = np.linspace(
        min(min(r["y_true_raw"] for r in rows), min(r["pg_fip_mean"] for r in rows)),
        max(max(r["y_true_raw"] for r in rows), max(r["pg_fip_mean"] for r in rows)),
        50,
    )
    ax.plot(diag, diag, "k--", lw=1, alpha=0.5, label="$\\hat = $ truth")
    ax.set_xlabel("true $\\mathrm{FIP}^m_{\\mathrm{FS}}$")
    ax.set_ylabel(
        "predicted $\\widehat{\\mathrm{FIP}^m_{\\mathrm{FS}}}$ (MCD posterior mean)"
    )
    ax.set_title(
        f"Pooled per-SVE parity with 90% combined CI bars\n"
        f"({len(rows)} test SVEs across {len(splits)} split-seeds × 5 init-seeds)"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "multi_split_parity.png", dpi=160)
    plt.close(fig)
    print(f"  [06] saved {out / 'multi_split_parity.png'}")


def main():
    print(f"[uq_plots] loading MCD predictions...")
    rows = load_all_predictions()
    print(
        f"[uq_plots] loaded {len(rows)} test SVEs from "
        f"{len(set((r['split'], r['seed']) for r in rows))} reps"
    )
    plot_01_calibration(rows)
    plot_02_epi_vs_ale(rows)
    plot_03_ci_vs_N(rows)
    plot_04_per_bucket_coverage(rows)
    plot_05_posterior_density(rows)
    plot_06_multi_split_parity(rows)
    print(f"[uq_plots] all plots written to {PLOTS}")


if __name__ == "__main__":
    main()
