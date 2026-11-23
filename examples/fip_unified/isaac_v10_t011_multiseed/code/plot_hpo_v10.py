"""plot_hpo_v10.py — v10 2-objective HPO visualisation.

Adapted from plot_hpo_evolution_v9d.py for v10's objective pair:
  O1 = HCF tail R²        (maximise)
  O2 = -HCF CRPS          (maximise; CRPS itself is minimise-direction)

Usage:
  python plot_hpo_v10.py [--top-k 8] [--split val]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
OUT_DIR = RESULTS / "hpo_v10"

HCF_CELLS = ("hcf_fs", "hcf_wp")
TRIAL_GLOB = "hpo_v10_t*_seed42"

OBJ_COLS = ["o1_tail_r2", "o2_neg_crps"]
OBJ_LABELS = {
    "o1_tail_r2": r"O1: HCF tail $R^{2}$",
    "o2_neg_crps": r"O2: $-$HCF CRPS  (higher = better)",
    # diagnostic-only:
    "o2_crps": r"HCF CRPS  (lower = better)",
}


def _safe_get(d: Dict, *path) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _hcf_max_r2_mean(metrics: Dict, split: str) -> float:
    rs = []
    for c in HCF_CELLS:
        v = _safe_get(metrics, split, "pergrain_aux", f"{c}_max", "metrics", "r2")
        if v is not None and np.isfinite(v):
            rs.append(float(v))
    return float(np.mean(rs)) if rs else float("nan")


def _hcf_pg_r2_mean(metrics: Dict, split: str) -> float:
    rs = []
    for c in HCF_CELLS:
        v = _safe_get(
            metrics, split, "aux_pergraph", f"{c}_pergraph_max", "metrics", "r2"
        )
        if v is not None and np.isfinite(v):
            rs.append(float(v))
    return float(np.mean(rs)) if rs else float("nan")


def _hcf_logsigma_r2_mean(metrics: Dict, split: str) -> float:
    rs = []
    for c in HCF_CELLS:
        v = _safe_get(
            metrics, split, "pergrain_aux", f"{c}_logsigma_voxel", "metrics", "r2"
        )
        if v is not None and np.isfinite(v):
            rs.append(float(v))
    return float(np.mean(rs)) if rs else float("nan")


def _hcf_mech_r2_mean(metrics: Dict, split: str) -> float:
    rs = []
    for k in [
        "yield_stress",
        "hardening_modulus",
        "Wp_lcf",
        "Wp_hcf",
        "sigma_amp_lcf",
        "sigma_amp_hcf",
    ]:
        v = _safe_get(metrics, split, "aux_pergraph", k, "metrics", "r2")
        if v is not None and np.isfinite(v):
            rs.append(float(v))
    return float(np.mean(rs)) if rs else float("nan")


def _hcf_tail_r2_from_csv(res_dir: Path, split: str) -> float:
    fp = res_dir / f"pred_{split}_pn.csv"
    if not fp.exists():
        return float("nan")
    try:
        df = pd.read_csv(fp)
    except Exception:
        return float("nan")
    rs = []
    for c in HCF_CELLS:
        ycol = f"true_{c}_max"
        pcol = f"pred_{c}_max"
        if ycol not in df.columns or pcol not in df.columns:
            continue
        sub = df[[ycol, pcol]].dropna()
        if len(sub) < 50:
            continue
        thresh = sub[ycol].quantile(0.90)
        tail = sub[sub[ycol] >= thresh]
        if len(tail) < 5:
            continue
        ss_res = float(((tail[ycol] - tail[pcol]) ** 2).sum())
        ss_tot = float(((tail[ycol] - tail[ycol].mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        rs.append(1.0 - ss_res / ss_tot)
    return float(np.mean(rs)) if rs else float("nan")


def _hcf_crps_from_csv(res_dir: Path, split: str) -> float:
    """Read pred CSV and compute mean Gaussian CRPS over HCF cells."""
    sys.path.insert(0, str(HERE))
    from crps import gaussian_crps

    fp = res_dir / f"pred_{split}_pn.csv"
    if not fp.exists():
        return float("nan")
    try:
        df = pd.read_csv(fp)
    except Exception:
        return float("nan")
    crps_vals = []
    for c in HCF_CELLS:
        ct, cp, cs = f"true_{c}_max", f"pred_{c}_max", f"logsigma_{c}_max"
        if cs not in df.columns:
            continue
        sub = df[[ct, cp, cs]].dropna()
        if len(sub) < 50:
            continue
        yt = sub[ct].values
        yp = sub[cp].values
        sg = sub[cs].values
        sig_raw = np.exp(sg) * yt.std()
        cv = gaussian_crps(yt, yp, sig_raw)
        if np.isfinite(cv):
            crps_vals.append(cv)
    return float(np.mean(crps_vals)) if crps_vals else float("nan")


def build_trials_csv(out: Path, split: str) -> pd.DataFrame:
    """Scan results/hpo_v10_t*_seed42/ and emit per-trial CSV."""
    rows: List[Dict[str, Any]] = []
    trial_dirs = sorted(RESULTS.glob(TRIAL_GLOB))
    for tdir in trial_dirs:
        cfg_p = tdir / "config.json"
        met_p = tdir / "run_metrics.json"
        if not cfg_p.exists() or not met_p.exists():
            continue
        try:
            cfg = json.load(open(cfg_p))
            met = json.load(open(met_p))
        except Exception:
            continue
        nn = cfg.get("NeuralNetwork", {})
        arch = nn.get("Architecture", {})
        train = nn.get("Training", {})
        rec = cfg.get("Recipe", {})
        trial_num = int(tdir.name.split("_t")[1].split("_")[0])

        nh = rec.get("node_head_dims") or {}
        gh = rec.get("graph_head_dims") or {}
        node_head_str = "x".join(str(d) for d in (nh.get("dim_headlayers") or []))
        graph_head_str = "x".join(str(d) for d in (gh.get("dim_headlayers") or []))

        crps = _hcf_crps_from_csv(tdir, split)
        crps_test = _hcf_crps_from_csv(tdir, "test")
        row = {
            "trial": trial_num,
            "param.hidden_dim": arch.get("hidden_dim"),
            "param.num_conv": arch.get("num_conv_layers"),
            "param.activation": arch.get("activation_function"),
            "param.dropout": arch.get("dropout"),
            "param.lr": _safe_get(train, "Optimizer", "learning_rate"),
            "param.graph_attr": rec.get("graph_attr_source"),
            "param.node_head": node_head_str or None,
            "param.graph_head": graph_head_str or None,
            "o1_tail_r2": _hcf_tail_r2_from_csv(tdir, split),
            "o2_crps": crps,
            "o2_neg_crps": -crps if np.isfinite(crps) else float("nan"),
            "hcf_max_r2": _hcf_max_r2_mean(met, split),
            "hcf_pg_r2": _hcf_pg_r2_mean(met, split),
            "hcf_logsigma_r2": _hcf_logsigma_r2_mean(met, split),
            "hcf_mech_r2": _hcf_mech_r2_mean(met, split),
            "o1_tail_r2_test": _hcf_tail_r2_from_csv(tdir, "test"),
            "o2_crps_test": crps_test,
            "hcf_pg_r2_test": _hcf_pg_r2_mean(met, "test"),
            "hcf_mech_r2_test": _hcf_mech_r2_mean(met, "test"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} trials)")
    return df


def compute_pareto(df: pd.DataFrame, out: Path) -> set[int]:
    """2-objective Pareto front (both maximised; o2_neg_crps is already flipped)."""
    rows = df.dropna(subset=OBJ_COLS).copy()
    pts = rows[OBJ_COLS].values
    n = len(pts)
    is_p = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_p[i]:
            continue
        for j in range(n):
            if i == j or not is_p[j]:
                continue
            if all(pts[j] >= pts[i]) and any(pts[j] > pts[i]):
                is_p[i] = False
                break
    pareto_trials = [int(t) for t, k in zip(rows["trial"].astype(int), is_p) if k]
    out.write_text(json.dumps([{"trial": t} for t in pareto_trials], indent=2))
    print(f"wrote {out}  Pareto={pareto_trials}")
    return set(pareto_trials)


def plot_objectives_evolution(df: pd.DataFrame, out: Path, pareto: set[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col in zip(axes, OBJ_COLS):
        y = df[col].values.astype(float)
        x = df["trial"].values.astype(int)
        running_best = np.maximum.accumulate(np.where(np.isnan(y), -np.inf, y))
        running_best[running_best == -np.inf] = np.nan
        ax.plot(x, running_best, color="C0", lw=2, label="running max")
        non_p = [i for i, t in enumerate(x) if int(t) not in pareto]
        in_p = [i for i, t in enumerate(x) if int(t) in pareto]
        if non_p:
            ax.scatter(x[non_p], y[non_p], c="#888", s=22, alpha=0.6, label="trial")
        if in_p:
            ax.scatter(
                x[in_p],
                y[in_p],
                c="C3",
                s=46,
                edgecolor="k",
                lw=0.6,
                zorder=5,
                label="Pareto",
            )
            for i in in_p:
                ax.annotate(
                    str(int(x[i])),
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        ax.set_xlabel("trial #")
        ax.set_ylabel(OBJ_LABELS[col])
        ax.set_title(OBJ_LABELS[col])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"v10 HPO objectives across {len(df)} trials", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_params_vs_objectives(df: pd.DataFrame, out: Path) -> None:
    params = [c for c in df.columns if c.startswith("param.")]
    fig, axes = plt.subplots(
        len(OBJ_COLS),
        len(params),
        figsize=(2.4 * len(params), 2.4 * len(OBJ_COLS)),
        sharey="row",
        squeeze=False,
    )
    cmap = plt.cm.viridis
    n = max(1, len(df))
    for i, obj in enumerate(OBJ_COLS):
        for j, p in enumerate(params):
            ax = axes[i, j]
            xs = df[p]
            ys = df[obj]
            colors = cmap(df["trial"].astype(float) / n)
            try:
                xs_num = pd.to_numeric(xs)
                ax.scatter(xs_num, ys, c=colors, s=20, alpha=0.8, edgecolor="k", lw=0.3)
                if "lr" in p:
                    ax.set_xscale("log")
            except (TypeError, ValueError):
                cats = list(xs.dropna().unique())
                cat_idx = xs.map({c: k for k, c in enumerate(cats)}).values
                ax.scatter(
                    cat_idx, ys, c=colors, s=20, alpha=0.8, edgecolor="k", lw=0.3
                )
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=45, fontsize=6, ha="right")
            if i == 0:
                ax.set_title(p.replace("param.", ""), fontsize=8)
            if j == 0:
                ax.set_ylabel(OBJ_LABELS[obj], fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=6)
    fig.suptitle(
        "v10 HPO -- parameter values vs. each objective (color = trial #)", y=1.0
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_pareto_2d(df: pd.DataFrame, out: Path, pareto: set[int]) -> None:
    """Single 2-D scatter: tail R² vs CRPS (lower) — both maximised in -CRPS form."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    is_p = df["trial"].isin(pareto)
    ax.scatter(
        df.loc[~is_p, "o2_crps"],
        df.loc[~is_p, "o1_tail_r2"],
        c="#888",
        s=24,
        alpha=0.6,
        label="trial",
    )
    ax.scatter(
        df.loc[is_p, "o2_crps"],
        df.loc[is_p, "o1_tail_r2"],
        c="C3",
        s=64,
        edgecolor="k",
        lw=0.7,
        zorder=5,
        label="Pareto",
    )
    for _, r in df[is_p].iterrows():
        ax.annotate(
            str(int(r["trial"])),
            (r["o2_crps"], r["o1_tail_r2"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("HCF CRPS  (lower = better)")
    ax.set_ylabel(r"HCF tail $R^{2}$  (higher = better)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("v10 HPO -- Pareto front (objective space)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_parallel_coords_topk(df: pd.DataFrame, out: Path, top_k: int = 8) -> None:
    sub = df.dropna(subset=OBJ_COLS).copy()
    if sub.empty:
        return
    sub["mean_obj"] = sub[OBJ_COLS].mean(axis=1)
    sub = (
        sub.sort_values("mean_obj", ascending=False).head(top_k).reset_index(drop=True)
    )
    cols = [c for c in sub.columns if c.startswith("param.")] + OBJ_COLS
    norm = sub[cols].copy()
    for c in cols:
        col = norm[c]
        try:
            col_num = pd.to_numeric(col, errors="raise")
            mn, mx = float(col_num.min()), float(col_num.max())
            norm[c] = (col_num - mn) / (mx - mn) if mx > mn else 0.5
        except Exception:
            cats = list(col.dropna().unique())
            mp = {v: i / max(1, len(cats) - 1) for i, v in enumerate(cats)}
            norm[c] = col.map(mp).fillna(0.5)
    fig, ax = plt.subplots(figsize=(2.2 + 0.6 * len(cols), 5.0))
    cmap = plt.cm.plasma
    xs = np.arange(len(cols))
    for i, (_, r) in enumerate(norm.iterrows()):
        ys = r[cols].values.astype(float)
        c = cmap(i / max(1, top_k - 1))
        ax.plot(
            xs,
            ys,
            "-o",
            color=c,
            alpha=0.85,
            lw=1.4,
            ms=4.5,
            label=f'trial {int(sub.loc[i, "trial"])}',
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [c.replace("param.", "") for c in cols], rotation=45, fontsize=8, ha="right"
    )
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("normalised value (per column)")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title(
        f"v10 HPO -- top-{top_k} trials parallel coords (rank by mean of objectives)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "all_trials.csv"
    df = build_trials_csv(csv_path, args.split)
    if df.empty:
        print("no trial data found")
        return 1
    pareto = compute_pareto(df, OUT_DIR / "pareto.json")
    plot_objectives_evolution(df, OUT_DIR / "hpo_v10_objectives_evolution.png", pareto)
    plot_params_vs_objectives(df, OUT_DIR / "hpo_v10_param_vs_objective.png")
    plot_pareto_2d(df, OUT_DIR / "hpo_v10_pareto_2d.png", pareto)
    plot_parallel_coords_topk(
        df, OUT_DIR / "hpo_v10_parallel_coords_top.png", args.top_k
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
