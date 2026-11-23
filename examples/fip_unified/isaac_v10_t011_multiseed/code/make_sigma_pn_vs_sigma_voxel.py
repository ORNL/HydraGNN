"""make_sigma_pn_vs_sigma_voxel.py — model UQ vs voxel UQ + descriptive
within-grain MAX-position diagnostic. (NO Gumbel/EVT framing.)

(A) Is model σ_pn physics-coupled to within-grain voxel dispersion?
    Spearman(log σ_pn, σ_voxel) per cell. Spearman is transform-invariant,
    so it doesn't matter that σ_pn lives in yeo-johnson space — only the
    rank relationship matters here.

(B) Where does the within-grain MAX sit relative to the lognormal fit?
    Compute z = (ln MAX − μ_log) / σ_log per grain. This is just a
    descriptive coordinate: how far above the lognormal-fit center
    the within-grain MAX falls. NOT a Gumbel test (iid premise fails;
    voxels are spatially correlated). Reference lines are LOGNORMAL
    PERCENTILES (1.65=95th, 2.33=99th, 3.09=99.9th) — i.e. where the
    MAX would sit *if* the voxel distribution were exactly lognormal
    and the MAX coincided with that percentile.

Outputs:
  - sigma_pn_vs_sigma_voxel.png — 4 cells × 2 rows (scatter + z-hist)
  - sigma_pn_vs_sigma_voxel.json — per-cell + per-bucket numbers
"""
from __future__ import annotations
from fip_release_paths import release_root
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DATA_PATH = Path(
    str(release_root() / "examples/cpfem_fip_data/outputs/full_data_v5.pt")
)
DEFAULT_RUN = STUDY / "results" / "L6b2_het_nll_both_50_15_seed42"
CELLS = [
    ("lcf_fs", "lcf_fs_fip_dist"),
    ("lcf_wp", "lcf_wp_fip_dist"),
    ("hcf_fs", "hcf_fs_fip_dist"),
    ("hcf_wp", "hcf_wp_fip_dist"),
]
COL_MAX, COL_MU, COL_SIGMA = (0, 6, 7)
BUCKET_COLOR = ["C0", "C1", "C2", "C3", "C4"]
BUCKET_LABEL = [
    "125μm/~30g",
    "177μm/~83g",
    "250μm/~233g",
    "354μm/~664g",
    "500μm/~1883g",
]
PCTILE_REF = [(1.645, "95th percentile"), (2.326, "99th"), (3.09, "99.9th")]


def _to_f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def make_for_run(run_dir: Path, data_path: Path = DATA_PATH) -> bool:
    pn_csv = run_dir / "pred_test_pn.csv"
    if not pn_csv.exists():
        print(f"  [skip] {run_dir.name}: no pred_test_pn.csv")
        return False
    rows = list(csv.DictReader(open(pn_csv)))
    if not rows:
        return False
    if not all((f"logsigma_{c}_max" in rows[0] for c, _ in CELLS)):
        print(f"  [skip] {run_dir.name}: σ-columns missing → not heteroscedastic run")
        return False
    print(f"  [load] {data_path}")
    data = torch.load(data_path, weights_only=False)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    summary = {}
    for ci, (cell, attr) in enumerate(CELLS):
        ls_pn, sv, lnmax, mu, bucket = ([], [], [], [], [])
        for r in rows:
            gi = int(_to_f(r["graph_idx"]))
            grn = int(_to_f(r["grain_idx"]))
            sb = int(_to_f(r.get("size_bucket", -1)))
            d = data[gi]
            if not hasattr(d, attr):
                continue
            fd = getattr(d, attr)
            if grn >= fd.shape[0]:
                continue
            mx = float(fd[grn, COL_MAX].item())
            mu_g = float(fd[grn, COL_MU].item())
            sv_g = float(fd[grn, COL_SIGMA].item())
            ls = _to_f(r[f"logsigma_{cell}_max"])
            if not (
                np.isfinite(mx)
                and np.isfinite(mu_g)
                and np.isfinite(sv_g)
                and np.isfinite(ls)
                and (mx > 0)
                and (sv_g > 0)
            ):
                continue
            ls_pn.append(ls)
            sv.append(sv_g)
            lnmax.append(np.log(mx))
            mu.append(mu_g)
            bucket.append(sb)
        ls_pn = np.array(ls_pn)
        sv = np.array(sv)
        lnmax = np.array(lnmax)
        mu = np.array(mu)
        bucket = np.array(bucket)
        cell_summary = {
            "n_grains_total": int(len(ls_pn)),
            "aggregate": {},
            "per_bucket": {},
        }
        ax = axes[0, ci]
        rho = p_rho = rp = np.nan
        if len(ls_pn) >= 8:
            rho, p_rho = spearmanr(ls_pn, sv)
            rp, _ = pearsonr(ls_pn, np.log(sv))
            cell_summary["aggregate"]["spearman_logsigma_pn_vs_sigma_voxel"] = float(
                rho
            )
            cell_summary["aggregate"]["spearman_p"] = float(p_rho)
        for sb_idx in range(5):
            mask = bucket == sb_idx
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            label = f"{BUCKET_LABEL[sb_idx]}"
            if n_b >= 8:
                rb, _ = spearmanr(ls_pn[mask], sv[mask])
                cell_summary["per_bucket"][str(sb_idx)] = {
                    "n": n_b,
                    "spearman": float(rb),
                }
                label += f"  ρ={rb:+.2f}"
            else:
                cell_summary["per_bucket"][str(sb_idx)] = {"n": n_b}
            ax.scatter(
                sv[mask],
                ls_pn[mask],
                s=6,
                alpha=0.4,
                c=BUCKET_COLOR[sb_idx],
                label=label,
                edgecolor="none",
            )
        ax.set_xlabel("σ_voxel  (lognormal_sigma per grain, log-FIP space)")
        ax.set_ylabel(
            "log σ_pn  (model output, yeo-johnson target space —\nrank-relevant only)"
        )
        ax.set_title(
            f"{cell}\nSpearman(σ_pn, σ_voxel) = {rho:+.3f}  (p={p_rho:.1e}, n={len(ls_pn)})\nPearson(log σ_pn, log σ_voxel) = {rp:+.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.25)
        ax2 = axes[1, ci]
        z = (lnmax - mu) / sv if len(sv) else np.array([])
        if len(z):
            for sb_idx in range(5):
                mask = bucket == sb_idx
                if mask.sum() == 0:
                    continue
                z_b = z[mask]
                ax2.hist(
                    z_b,
                    bins=25,
                    alpha=0.45,
                    density=True,
                    color=BUCKET_COLOR[sb_idx],
                    label=f"{BUCKET_LABEL[sb_idx]} z̄={z_b.mean():.2f}",
                )
                cell_summary["per_bucket"][str(sb_idx)] = {
                    **cell_summary["per_bucket"].get(str(sb_idx), {}),
                    "z_mean": float(z_b.mean()),
                    "z_std": float(z_b.std()),
                }
            cell_summary["aggregate"]["z_mean"] = float(np.mean(z))
            cell_summary["aggregate"]["z_std"] = float(np.std(z))
            for zr, lab in PCTILE_REF:
                ax2.axvline(
                    zr,
                    ls="--",
                    lw=1.0,
                    color="0.4",
                    label=f"lognormal {lab} (z={zr:.2f})",
                )
        ax2.set_xlabel("z = (ln MAX − μ_log) / σ_log  (within-grain)")
        ax2.set_ylabel("density")
        ax2.set_title(
            f"{cell} | within-grain MAX position\nz̄={cell_summary['aggregate'].get('z_mean', float('nan')):.2f}  std={cell_summary['aggregate'].get('z_std', float('nan')):.2f}",
            fontsize=9,
        )
        ax2.legend(fontsize=6, loc="upper left")
        ax2.grid(alpha=0.25)
        summary[cell] = cell_summary
    fig.suptitle(
        f"{run_dir.name} — model σ_pn vs voxel σ + within-grain MAX position\n(test split, by SVE size bucket; NOT a Gumbel/EVT test — see make_cross_sve_gumbel.py for the proper block-maxima test)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "sigma_pn_vs_sigma_voxel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    with open(run_dir / "sigma_pn_vs_sigma_voxel.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] {run_dir.name}: sigma_pn_vs_sigma_voxel.png + .json")
    for c, s in summary.items():
        a = s.get("aggregate", {})
        print(
            f"    {c:8s}: ρ(σ_pn,σ_vox)={a.get('spearman_logsigma_pn_vs_sigma_voxel', float('nan')):+.3f}  z̄={a.get('z_mean', float('nan')):.2f}±{a.get('z_std', float('nan')):.2f}"
        )
    return True


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    return 0 if make_for_run(args.run, args.data) else 1


if __name__ == "__main__":
    sys.exit(main())
