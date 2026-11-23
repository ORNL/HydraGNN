"""aggregate_mcd_v9c.py — pool MCD epistemic + aleatoric across 25 reps.

For each E_M_B_split{S}_seed{I} rep:
  - σ_e (epistemic) per max channel = std across MCD passes (mcd_predictions.npz)
  - σ_pn (aleatoric) per max channel = average exp(logsigma) from train-time NLL
  - σ_total = sqrt(σ_pn² + σ_e²)

Aggregate (across 25 reps):
  - per-cell mean(σ_e), mean(σ_pn), mean(σ_total)
  - per-cell ratio σ_e / σ_total (epistemic dominance)
  - per-cell Pcov at z=1.65 with combined σ_total

Also reports cross-rep R² mean ± std for the σ_log channels (Phase M
distinguishing finding) — confirms the σ_log learnability across the
5×5 grid.

Usage:
  python aggregate_mcd_v9c.py
  python aggregate_mcd_v9c.py --split_filter 42  # subset
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

CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
SPLIT_SEEDS = [7, 13, 21, 42, 99, 137, 271, 314, 666, 2024]
INIT_SEEDS = [42]
WINNER_TAG = "t011"


def _per_cell_metrics_from_run(run_dir):
    """Return per-cell σ_pn (aleatoric) + σ_e (epistemic) means.
    σ_pn from pred_test_pn.csv logsigma columns; σ_e from MCD npz."""
    out = {c: {} for c in CELL_KEYS}

    # σ_pn (aleatoric) from CSV — average over test grains
    pn_csv = run_dir / "pred_test_pn.csv"
    if pn_csv.exists():
        rows = list(csv.DictReader(open(pn_csv)))
        for c in CELL_KEYS:
            ls_col = f"logsigma_{c}_max"
            if ls_col in (rows[0].keys() if rows else {}):
                arr = []
                for r in rows:
                    v = r.get(ls_col)
                    if v in (None, ""):
                        continue
                    try:
                        arr.append(float(v))
                    except:
                        pass
                if arr:
                    out[c]["sigma_pn_mean"] = float(np.exp(np.mean(arr)))
                    out[c]["sigma_pn_median"] = float(np.exp(np.median(arr)))

    # σ_e (epistemic) from MCD npz
    npz = run_dir / "mcd_predictions.npz"
    if npz.exists():
        d = np.load(npz)
        pred_std = d["pred_std"]  # (n_grains, n_pn_total)
        # Slot 0..3 = max channels per (lcf_fs, lcf_wp, hcf_fs, hcf_wp)
        for i, c in enumerate(CELL_KEYS):
            if pred_std.shape[1] > i:
                out[c]["sigma_e_mean"] = float(pred_std[:, i].mean())
                out[c]["sigma_e_median"] = float(np.median(pred_std[:, i]))
        # logsigma slots = 12..15 for distB (n_super=16: 4 max + 4 mean + 4 logmu + 4 logsigma)
        if pred_std.shape[1] >= 16:
            for i, c in enumerate(CELL_KEYS):
                out[c]["sigma_e_logsigma_mean"] = float(pred_std[:, 12 + i].mean())
        # NEW: per-graph head MCD ensemble (added 2026-05-08; pg_pred_std
        # exists when re-run with the extended inference). Slot 0..3 = SVE-MAX.
        if "pg_pred_std" in d.files:
            pg_pred_std = d["pg_pred_std"]
            for i, c in enumerate(CELL_KEYS):
                if pg_pred_std.shape[1] > i:
                    out[c]["pg_sigma_e_mean"] = float(pg_pred_std[:, i].mean())
                    out[c]["pg_sigma_e_median"] = float(np.median(pg_pred_std[:, i]))

    # Per-cell σ_log R² (Phase M finding — robust across reps?)
    rp = run_dir / "run_metrics.json"
    if rp.exists():
        m = json.load(open(rp))
        aux = (m.get("test") or {}).get("pergrain_aux") or {}
        for c in CELL_KEYS:
            r2 = (
                (aux.get(f"{c}_logsigma_voxel") or {})
                .get("metrics", {})
                .get("r2", float("nan"))
            )
            out[c]["r2_logsigma"] = float(r2)
            r2_mu = (
                (aux.get(f"{c}_logmu_voxel") or {})
                .get("metrics", {})
                .get("r2", float("nan"))
            )
            out[c]["r2_logmu"] = float(r2_mu)
        # Top-1 per cell
        h = (m.get("test") or {}).get("pergrain_per_cell_hotspot") or {}
        for c in CELL_KEYS:
            v = h.get(f"{c}_max") or {}
            out[c]["top1"] = float(v.get("top1_hit_rate", float("nan")))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split_filter", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [(s, i) for s in SPLIT_SEEDS for i in INIT_SEEDS]
    if args.split_filter is not None:
        pairs = [(s, i) for s, i in pairs if s == args.split_filter]

    rep_metrics = {
        c: {
            k: []
            for k in (
                "sigma_pn_mean",
                "sigma_e_mean",
                "sigma_e_logsigma_mean",
                "pg_sigma_e_mean",
                "r2_logsigma",
                "r2_logmu",
                "top1",
            )
        }
        for c in CELL_KEYS
    }
    # NEW: cross-rep mech σ_e (from pg head channels 8..13)
    MECH_KEYS = [
        "yield_stress",
        "hardening_modulus",
        "Wp_lcf",
        "Wp_hcf",
        "sigma_amp_lcf",
        "sigma_amp_hcf",
    ]
    mech_sigma_e = {k: [] for k in MECH_KEYS}

    n_total = 0
    n_with_mcd = 0
    n_with_metrics = 0
    for split_seed, init_seed in pairs:
        tag = f"{WINNER_TAG}_split{split_seed}"
        rd = STUDY / "results" / f"{tag}_seed{init_seed}"
        if not rd.exists():
            continue
        n_total += 1
        per_cell = _per_cell_metrics_from_run(rd)
        for c in CELL_KEYS:
            for k in rep_metrics[c]:
                v = per_cell[c].get(k)
                if v is not None and v == v:
                    rep_metrics[c][k].append(v)
        # NEW: harvest mech σ_e from pg ensemble (slots 8..13)
        npz_p = rd / "mcd_predictions.npz"
        if npz_p.exists():
            d = np.load(npz_p)
            if "pg_pred_std" in d.files:
                pg = d["pg_pred_std"]
                if pg.shape[1] >= 14:
                    for j, k in enumerate(MECH_KEYS):
                        mech_sigma_e[k].append(float(pg[:, 8 + j].mean()))
            n_with_mcd += 1
        if (rd / "run_metrics.json").exists():
            n_with_metrics += 1

    print()
    print(
        f" v9c E-sweep MCD aggregate — {n_total} dirs, {n_with_metrics} trained, {n_with_mcd} with MCD"
    )
    print("=" * 110)

    summary = {
        "n_reps_total": n_total,
        "n_reps_trained": n_with_metrics,
        "n_reps_with_mcd": n_with_mcd,
        "per_cell": {},
    }

    print()
    print(" Per-cell aggregates (mean ± std across reps)")
    print(
        f"  {'cell':<10s} {'top1':>14s} {'σ_pn (alea)':>14s} {'σ_e (epi)':>14s} {'r2_logmu':>14s} {'r2_logsigma':>14s}"
    )
    print("-" * 100)
    for c in CELL_KEYS:
        rm = rep_metrics[c]

        def stat(key):
            arr = np.array(rm[key])
            if arr.size == 0:
                return ("nan", "nan", 0)
            return (
                f"{arr.mean():+.3f}",
                f"{arr.std(ddof=1) if arr.size>1 else 0:.3f}",
                arr.size,
            )

        ts = stat("top1")
        ps = stat("sigma_pn_mean")
        es = stat("sigma_e_mean")
        rmu = stat("r2_logmu")
        rsg = stat("r2_logsigma")
        print(
            f"  {c:<10s} "
            f"{ts[0]:>8s}±{ts[1]:>5s} "
            f"{ps[0]:>8s}±{ps[1]:>5s} "
            f"{es[0]:>8s}±{es[1]:>5s} "
            f"{rmu[0]:>8s}±{rmu[1]:>5s} "
            f"{rsg[0]:>8s}±{rsg[1]:>5s}"
        )
        summary["per_cell"][c] = {
            "top1": {
                "mean": float(np.mean(rm["top1"])) if rm["top1"] else float("nan"),
                "std": float(np.std(rm["top1"], ddof=1))
                if len(rm["top1"]) > 1
                else float("nan"),
                "n": len(rm["top1"]),
            },
            "sigma_pn_mean": {
                "mean": float(np.mean(rm["sigma_pn_mean"]))
                if rm["sigma_pn_mean"]
                else float("nan"),
                "std": float(np.std(rm["sigma_pn_mean"], ddof=1))
                if len(rm["sigma_pn_mean"]) > 1
                else float("nan"),
                "n": len(rm["sigma_pn_mean"]),
            },
            "sigma_e_mean": {
                "mean": float(np.mean(rm["sigma_e_mean"]))
                if rm["sigma_e_mean"]
                else float("nan"),
                "std": float(np.std(rm["sigma_e_mean"], ddof=1))
                if len(rm["sigma_e_mean"]) > 1
                else float("nan"),
                "n": len(rm["sigma_e_mean"]),
            },
            "r2_logmu": {
                "mean": float(np.mean(rm["r2_logmu"]))
                if rm["r2_logmu"]
                else float("nan"),
                "std": float(np.std(rm["r2_logmu"], ddof=1))
                if len(rm["r2_logmu"]) > 1
                else float("nan"),
                "n": len(rm["r2_logmu"]),
            },
            "r2_logsigma": {
                "mean": float(np.mean(rm["r2_logsigma"]))
                if rm["r2_logsigma"]
                else float("nan"),
                "std": float(np.std(rm["r2_logsigma"], ddof=1))
                if len(rm["r2_logsigma"]) > 1
                else float("nan"),
                "n": len(rm["r2_logsigma"]),
            },
            "pg_sigma_e_mean": {
                "mean": float(np.mean(rm["pg_sigma_e_mean"]))
                if rm["pg_sigma_e_mean"]
                else float("nan"),
                "std": float(np.std(rm["pg_sigma_e_mean"], ddof=1))
                if len(rm["pg_sigma_e_mean"]) > 1
                else float("nan"),
                "n": len(rm["pg_sigma_e_mean"]),
            },
        }
    # NEW: mech-QoI σ_e summary
    summary["mech_sigma_e"] = {
        k: {
            "mean": float(np.mean(v)) if v else float("nan"),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "n": len(v),
        }
        for k, v in mech_sigma_e.items()
    }

    print("=" * 110)
    print()
    print(" Aleatoric vs epistemic split — typical interpretation:")
    print("   σ_pn / σ_total = aleatoric fraction (data noise)")
    print("   σ_e  / σ_total = epistemic fraction (model uncertainty)")
    print("=" * 110)
    print()
    print(
        f"  {'cell':<10s}  {'σ_e_mean':>10s}  {'σ_pn_mean':>10s}  {'σ_total_mean':>12s}  {'epi%':>6s}  {'alea%':>6s}"
    )
    for c in CELL_KEYS:
        per = summary["per_cell"][c]
        if per["sigma_pn_mean"]["n"] == 0 or per["sigma_e_mean"]["n"] == 0:
            continue
        sp = per["sigma_pn_mean"]["mean"]
        se = per["sigma_e_mean"]["mean"]
        st = float(np.sqrt(sp ** 2 + se ** 2))
        pe = (se * se / (sp * sp + se * se) * 100.0) if (sp + se) > 0 else 0
        pa = 100.0 - pe
        print(
            f"  {c:<10s}  {se:>10.4f}  {sp:>10.4f}  {st:>12.4f}  {pe:>5.1f}%  {pa:>5.1f}%"
        )

    out_path = args.out or (STUDY / "results" / "E_aggregate" / "mcd_aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
