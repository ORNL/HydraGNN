"""Representative mean-residual-life curves; graphical diagnostic only."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "revision_R2/studies_2026_11_15/data"
PARTITIONS = (
    ROOT
    / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation/partitions.json"
)
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
PROBABILITIES = np.linspace(0, 0.98, 50)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    meta = pd.read_csv(SOURCE / "all_sve_metadata.csv")
    assert len(meta) == 940 and meta.dataset_idx.nunique() == 940
    chosen = []
    for size in SIZES:
        group = meta[meta.size_um.eq(size)].copy()
        median = float(group.grain_count.median())
        group["distance_to_median"] = (group.grain_count - median).abs()
        row = group.sort_values(["distance_to_median", "dataset_idx"]).iloc[0]
        chosen.append(
            {
                "size_um": size,
                "dataset_idx": int(row.dataset_idx),
                "grain_count": int(row.grain_count),
                "size_population": len(group),
                "median_grain_count": median,
                "distance_to_median": float(row.distance_to_median),
            }
        )
    # This file records the selection before response fields are inspected.
    manifest = {
        "selection": "one SVE per size with grain count nearest the size-specific median; dataset_idx breaks ties",
        "selection_uses_response_values": False,
        "repeat": 0,
        "selected": chosen,
        "grid_probabilities": PROBABILITIES.tolist(),
        "metadata_sha256": sha(SOURCE / "all_sve_metadata.csv"),
        "script_sha256": sha(__file__),
        "partitions_sha256": sha(PARTITIONS),
        "purpose": "graphical matched-field diagnostic, not a deployable observed-response predictor or threshold selector",
    }
    (HERE / "mrl_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parts = [p for p in json.loads(PARTITIONS.read_text()) if p["repeat"] == 0]
    selected_ids = {x["dataset_idx"] for x in chosen}
    checked = set()
    rows = []
    source_hashes = {}
    for part in parts:
        fold = part["fold"]
        path = SOURCE / f"grain_predictions_repeat0_fold{fold}.npz"
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            assert list(z["cases"]) == CASES
            assert set(z["dataset_idx"]) == set(part["evaluation"])
            for j, idx in enumerate(z["dataset_idx"]):
                idx = int(idx)
                if idx not in selected_ids:
                    continue
                assert idx not in checked
                a, b = z["offsets"][j : j + 2]
                selection = next(c for c in chosen if c["dataset_idx"] == idx)
                assert b - a == selection["grain_count"] == z["grain_count"][j]
                assert selection["size_um"] == z["size_um"][j]
                assert np.allclose(
                    z["true_peak"][a:b].max(axis=0), z["y_true"][j], rtol=2e-6
                )
                selection["fold"] = int(fold)
                for c, case in enumerate(CASES):
                    for field, column in [
                        ("observed", "true_peak"),
                        ("predicted", "pred_peak"),
                    ]:
                        x = np.asarray(z[column][a:b, c], dtype=float)
                        assert np.isfinite(x).all()
                        for q in PROBABILITIES:
                            u = float(np.quantile(x, q, method="linear"))
                            excess = x[x > u] - u
                            # Empty points have no empirical conditional mean and
                            # are explicitly marked; they are not fabricated as 0.
                            mean = float(excess.mean()) if len(excess) else None
                            rows.append(
                                {
                                    "dataset_idx": idx,
                                    "size_um": selection["size_um"],
                                    "grain_count": len(x),
                                    "repeat": 0,
                                    "fold": fold,
                                    "case": case,
                                    "field": field,
                                    "quantile": float(q),
                                    "threshold": u,
                                    "mean_excess": mean,
                                    "exceedance_count": len(excess),
                                    "below_ten_exceedances": len(excess) < 10,
                                    "status": "ok" if len(excess) else "no_exceedances",
                                }
                            )
                checked.add(idx)
    assert checked == selected_ids
    values = pd.DataFrame(rows)
    assert len(values) == 5 * 4 * 2 * 50
    assert not values.duplicated(["dataset_idx", "case", "field", "quantile"]).any()
    values.to_csv(HERE / "mrl_values.csv", index=False)
    manifest.update(
        selected=chosen,
        source_hashes=source_hashes,
        rows=len(values),
        empty_tail_points=int(values.status.ne("ok").sum()),
        finite_nonempty_points=bool(
            np.isfinite(values.loc[values.status.eq("ok"), "mean_excess"]).all()
        ),
        all_selected_fields_matched_to_original_evaluation=True,
    )
    (HERE / "mrl_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot(values)
    print(
        json.dumps(
            {
                "selected": chosen,
                "rows": len(values),
                "empty_tail_points": manifest["empty_tail_points"],
            },
            indent=2,
        )
    )


def plot(values):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{lmodern}",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex="row", sharey="row")
    for i, (case, size) in enumerate(
        [("hcf_fs", 125), ("hcf_fs", 500), ("lcf_wp", 125), ("lcf_wp", 500)]
    ):
        ax = axes.flat[i]
        for field, label, color in [
            ("observed", "CPFEM", "#333333"),
            ("predicted", "GNN", "#1f77b4"),
        ]:
            q = values[
                values.case.eq(case)
                & values.size_um.eq(size)
                & values.field.eq(field)
                & values.status.eq("ok")
            ].sort_values("quantile")
            ax.plot(q.threshold, q.mean_excess, color=color, lw=1.45, label=label)
            sparse = q[q.exceedance_count.lt(10)]
            ax.plot(
                sparse.threshold,
                sparse.mean_excess,
                "o",
                ms=4,
                mfc="white",
                mec=color,
                mew=0.9,
                ls="none",
            )
        ax.text(0, 1.115, "(" + chr(97 + i) + ")", transform=ax.transAxes, va="bottom")
        ax.text(
            0.96,
            0.95,
            str(size) + r" $\mu$m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
        )
        ax.set_xlabel(r"Threshold, $u$" + (r" (MPa)" if case == "lcf_wp" else ""))
        if i % 2 == 0:
            ax.set_ylabel(
                ("HCF/FS" if case == "hcf_fs" else "LCF/WP")
                + r" mean excess"
                + (r" (MPa)" if case == "lcf_wp" else "")
            )
        ax.ticklabel_format(
            axis="both", style="sci", scilimits=(-3, 3), useMathText=True
        )
        ax.grid(alpha=0.15, lw=0.35)
    for row, case in enumerate(["hcf_fs", "lcf_wp"]):
        shown = values[
            values.case.eq(case)
            & values.size_um.isin([125, 500])
            & values.status.eq("ok")
        ]
        axes[row, 0].set_ylim(0, 1.07 * shown.mean_excess.max())
    handles = [
        Line2D([], [], color="#333333", lw=1.45, label="CPFEM"),
        Line2D([], [], color="#1f77b4", lw=1.45, label="GNN"),
        Line2D(
            [],
            [],
            color="#555555",
            marker="o",
            mfc="white",
            mec="#555555",
            ls="none",
            ms=4,
            label=r"$k<10$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.52, 0.005),
        columnspacing=2.0,
    )
    fig.subplots_adjust(
        left=0.11, right=0.98, top=0.92, bottom=0.14, wspace=0.16, hspace=0.58
    )
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "mean_residual_life.png", bbox_inches="tight")
    fig.savefig(out / "mean_residual_life.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
