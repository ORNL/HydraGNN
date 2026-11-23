"""Candidate diagnostic figures; source results are never modified."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
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
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
CASE_LABELS = ["LCF/FS", "LCF/WP", "HCF/FS", "HCF/WP"]
BLUE = "#1f77b4"
ORANGE = "#d67a12"
BLACK = "#111111"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    outputs = {}
    for ext in ["pdf", "png"]:
        path = HERE / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        outputs[ext] = {
            "path": str(path),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    plt.close(fig)
    return outputs


def letter(ax, i):
    ax.text(0, 1.025, "(" + chr(97 + i) + ")", transform=ax.transAxes, va="bottom")


def errorbar(ax, x, y, low, high, **kwargs):
    x, y, low, high = map(np.asarray, (x, y, low, high))
    assert np.isfinite(np.r_[y, low, high]).all()
    assert np.all(low <= y) and np.all(y <= high)
    return ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - low, high - y]),
        capsize=2.5,
        elinewidth=0.9,
        capthick=0.9,
        **kwargs,
    )


def readout_budget_figure(budget, preserve_existing_data=False):
    """Render only the reference-SVE comparison, using the fixed source rows."""
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for i, (ax, scope, scope_label) in enumerate(
        zip(axs, ["pooled", "500"], ["All sizes", r"500 $\mu$m"])
    ):
        for j, (group, label, color, marker) in enumerate(
            [
                ("same_case", "One-case grain summaries", BLUE, "o"),
                ("all_cases", "Four-case grain summaries", ORANGE, "s"),
            ]
        ):
            q = (
                budget[
                    (budget.group == group)
                    & (budget["mode"] == "alpha100")
                    & (budget.size_um == scope)
                ]
                .set_index("case")
                .loc[CASES]
            )
            y = np.arange(4) + (j - 0.5) * 0.22
            values = q.full_budget_rmse_reduction_percent.to_numpy()
            low = q.ci_low.to_numpy()
            high = q.ci_high.to_numpy()
            assert np.allclose(
                values,
                100 * (1 - q.rmse_full.to_numpy() / q.rmse_75.to_numpy()),
                rtol=1e-12,
                atol=1e-12,
            )
            ax.errorbar(
                values,
                y,
                xerr=np.vstack([values - low, high - values]),
                fmt=marker,
                ms=4.5,
                color=color,
                capsize=2.5,
                lw=1,
                label=label,
            )
            for case, r in q.iterrows():
                rows.append(
                    dict(
                        group=group,
                        alpha=100,
                        case=case,
                        size_um=scope,
                        n_physical_sves=int(r.n_sve),
                        rmse_reduction_percent=float(
                            r.full_budget_rmse_reduction_percent
                        ),
                        ci_low=float(r.ci_low),
                        ci_high=float(r.ci_high),
                        rmse_75=float(r.rmse_75),
                        rmse_full=float(r.rmse_full),
                    )
                )
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(np.arange(4), CASE_LABELS)
        ax.invert_yaxis()
        ax.set_xlim(-0.2, 6.35)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_ylim(3.5, -0.5)
        ax.set_xlabel(scope_label + r": RMSE reduction (\%)")
        ax.grid(axis="x", alpha=0.17, lw=0.35)
        letter(ax, i)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.043),
        columnspacing=2,
    )
    fig.text(
        0.53, 0.018, r"75 versus 751--753 reference SVEs", ha="center", fontsize=11
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.25, wspace=0.35)
    data_path = HERE / "hgb_readout_budget_alpha100_data.csv"
    frame = pd.DataFrame(rows)
    if preserve_existing_data and data_path.exists():
        existing = pd.read_csv(
            data_path, dtype={"size_um": str}, float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(frame, existing, check_exact=True)
    else:
        frame.to_csv(data_path, index=False)
    outputs = save_figure(fig, "hgb_readout_budget_alpha100")
    return {
        "outputs": outputs,
        "rows": rows,
        "statistic": "100*(1-RMSE_full/RMSE_75); positive favors 751–753 reference SVEs",
        "interval": "Published paired pointwise conditional95% physical-SVE bootstrap intervals",
        "readout_alpha": 100,
        "subset_draws": 100,
        "prediction_repeats": 2,
        "scientific_scope": "Fixed HGB features and fixed numerical alpha. This includes budget/shrinkage sensitivity, not a pure encoder comparison.",
    }


def main(figure="all"):
    if figure == "hgb_readout_budget_alpha100":
        budget_path = ROOT / "readout_budget/paired_budget_comparisons.csv"
        budget = pd.read_csv(
            budget_path, dtype={"size_um": str}, float_precision="round_trip"
        )
        trace_path = HERE / "figure_trace.json"
        trace = (
            json.loads(trace_path.read_text(encoding="utf-8"))
            if trace_path.exists()
            else {"figures": {}}
        )
        trace["figures"]["hgb_readout_budget_alpha100"] = readout_budget_figure(
            budget, preserve_existing_data=True
        )
        trace["figures"]["hgb_readout_budget_alpha100"]["generator_sha256"] = sha(
            Path(__file__)
        )
        trace["figures"]["hgb_readout_budget_alpha100"]["source_sha256"] = sha(
            budget_path
        )
        trace_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            "Generated only hgb_readout_budget_alpha100 (PDF and PNG); numerical data unchanged.",
            flush=True,
        )
        return
    count_path = ROOT / "count_thinning/metrics.csv"
    budget_path = ROOT / "readout_budget/paired_budget_comparisons.csv"
    metadata_path = ROOT.parents[1] / "studies_2026_11_15/data/all_sve_metadata.csv"
    # ROOT is model_redevelopment; ROOT.parents[1] is revision_R2.
    counts = pd.read_csv(metadata_path)
    median_n = float(counts.loc[counts.size_um.eq(500), "grain_count"].median())
    n_range = (
        counts.loc[counts.size_um.eq(500), "grain_count"]
        .agg(["min", "max"])
        .astype(int)
        .tolist()
    )
    data = pd.read_csv(count_path, float_precision="round_trip")
    budget = pd.read_csv(
        budget_path, dtype={"size_um": str}, float_precision="round_trip"
    )
    trace = {
        "status": "generated_not_yet_visually_reviewed",
        "generator": str(Path(__file__)),
        "generator_sha256": sha(Path(__file__)),
        "sources": [
            {"path": str(p), "sha256": sha(p)}
            for p in [count_path, budget_path, metadata_path]
        ],
        "style_source": str(ROOT.parent / "threshold_literature/plot_results.py"),
        "style": dict(
            font="Computer Modern via LaTeX",
            boxed_axes=True,
            inward_ticks=True,
            titles=False,
            colors=[BLUE, ORANGE, BLACK],
        ),
        "figures": {},
    }

    configs = ["common_26", "large_30", "large_83", "large_233", "large_665", "full"]
    x = np.asarray([26, 30, 83, 233, 665, median_n])
    plot_rows = []
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.7))
    for i, (ax, case, label) in enumerate(zip(axs.flat, CASES, CASE_LABELS)):
        q = (
            data[
                (data.physical_size_um == 500)
                & (data.case == case)
                & (data.route == "raw_max")
            ]
            .set_index("configuration")
            .loc[configs]
        )
        y = q.draw_repeat_correlation.to_numpy()
        lo = q.draw_repeat_correlation_lo.to_numpy()
        hi = q.draw_repeat_correlation_hi.to_numpy()
        errorbar(
            ax, x[:-1], y[:-1], lo[:-1], hi[:-1], fmt="o-", color=BLUE, lw=1.1, ms=4
        )
        ax.plot(x[-2:], y[-2:], color="#777777", lw=0.8, ls="--")
        errorbar(ax, x[-1:], y[-1:], lo[-1:], hi[-1:], fmt="D", color=BLACK, ms=4.8)
        ax.axhline(0, color="black", lw=0.45)
        ax.set_xscale("log")
        ax.set_xticks([26, 83, 233, 665, median_n], ["26", "83", "233", "665", r"$N$"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(22, 2300)
        ax.set_ylim(-0.35, 0.75)
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
        ax.set_xlabel(r"Retained grains, $K$ (500 $\mu$m)")
        ax.set_ylabel(label + r" Pearson $r$")
        ax.grid(alpha=0.17, lw=0.35)
        letter(ax, i)
        for j, cfg in enumerate(configs):
            plot_rows.append(
                dict(
                    configuration=cfg,
                    case=case,
                    physical_size_um=500,
                    k="observed_N" if cfg == "full" else int(x[j]),
                    plotted_x=float(x[j]),
                    correlation=float(y[j]),
                    ci_low=float(lo[j]),
                    ci_high=float(hi[j]),
                    endpoint="M_N" if cfg == "full" else "M_K",
                    n_physical_sves=141,
                )
            )
    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            lw=1.1,
            ms=4,
            label=r"Raw subset maximum $\widehat{M}_K$",
        ),
        Line2D(
            [],
            [],
            color=BLACK,
            marker="D",
            ls="None",
            ms=4.8,
            label=r"Raw full-SVE maximum $\widehat{M}_N$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.002),
        columnspacing=2.5,
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, top=0.96, bottom=0.13, wspace=0.28, hspace=0.36
    )
    outputs = save_figure(fig, "count_thinning_500")
    pd.DataFrame(plot_rows).to_csv(HERE / "count_thinning_500_data.csv", index=False)
    trace["figures"]["count_thinning_500"] = {
        "outputs": outputs,
        "rows": plot_rows,
        "statistic": "Equal-SVE draw/repeat-mixture Pearson correlation",
        "interval": "Pointwise conditional95% physical-SVE bootstrap; shared resamples across counts and routes",
        "subset_draws": 100,
        "prediction_repeats": 2,
        "full_N_range": n_range,
        "full_N_plotted_x": "median observed N",
        "scientific_scope": "Random-subset M_K changes with K; full M_N is marked separately. No graph refit or smaller-SVE simulation.",
        "line_semantics": "Lines guide the eye; dashed final segment reaches the distinct full-SVE endpoint.",
    }

    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.05))
    for i, (ax, regime) in enumerate(zip(axs, ["lcf", "hcf"])):
        for suffix, label, color, marker in [
            ("fs", "FS", BLUE, "o"),
            ("wp", "WP", ORANGE, "s"),
        ]:
            case = f"{regime}_{suffix}"
            q = data[
                (data.configuration == "common_26")
                & (data.case == case)
                & (data.route == "raw_max")
            ].sort_values("physical_size_um")
            assert len(q) == 5
            errorbar(
                ax,
                q.physical_size_um,
                q.draw_repeat_correlation,
                q.draw_repeat_correlation_lo,
                q.draw_repeat_correlation_hi,
                fmt=marker + "-",
                color=color,
                lw=1.1,
                ms=4,
                label=label + " subset maximum",
            )
            for _, r in q.iterrows():
                rows.append(
                    dict(
                        case=case,
                        k=26,
                        physical_size_um=int(r.physical_size_um),
                        n_physical_sves=int(r.n_physical_sves),
                        correlation=float(r.draw_repeat_correlation),
                        ci_low=float(r.draw_repeat_correlation_lo),
                        ci_high=float(r.draw_repeat_correlation_hi),
                    )
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([125, 177, 250, 354, 500], ["125", "177", "250", "354", "500"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(112, 557)
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_xlabel(r"SVE edge length ($\mu$m)")
        ax.set_ylabel(regime.upper() + r" Pearson $r$")
        ax.text(0.96, 0.96, r"$K=26$", ha="right", va="top", transform=ax.transAxes)
        ax.grid(alpha=0.17, lw=0.35)
        letter(ax, i)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, -0.008),
        columnspacing=2.5,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.23, wspace=0.26)
    outputs = save_figure(fig, "count_common_26")
    pd.DataFrame(rows).to_csv(HERE / "count_common_26_data.csv", index=False)
    trace["figures"]["count_common_26"] = {
        "outputs": outputs,
        "rows": rows,
        "statistic": "Equal-SVE draw/repeat-mixture Pearson correlation",
        "interval": "Pointwise conditional95% physical-SVE bootstrap",
        "scientific_scope": "All physical sizes retain26random grain rows; this is the subset endpoint M_26, not full-SVE accuracy.",
    }

    trace["figures"]["hgb_readout_budget_alpha100"] = readout_budget_figure(budget)
    (HERE / "figure_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Generated three candidate figures (PDF and PNG); source results unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all diagnostics or only the reference-SVE comparison."
    )
    parser.add_argument(
        "--figure", choices=["all", "hgb_readout_budget_alpha100"], default="all"
    )
    main(parser.parse_args().figure)
