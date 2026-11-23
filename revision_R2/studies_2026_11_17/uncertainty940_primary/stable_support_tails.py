"""Stable outside-inverse-support and negative-physical-support diagnostics.

Computes the upper tail directly, avoiding cancellation in 1-CDF. This is
authoritative for tiny outside masses; analyze.py's first-pass diagnostic may
round these tiny masses to zero. Neither changes the predictive distribution.
"""
from pathlib import Path
import json, pickle
import numpy as np, pandas as pd
from scipy.special import ndtr
from analyze import bounds, sha, CASES, HERE


def main():
    rows = []
    for r in range(2):
        for f in range(5):
            key = f"repeat{r}_fold{f}"
            maps = pickle.load(open(HERE / f"{key}_fitted_maps.pkl", "rb"))
            with np.load(HERE / f"{key}_draws.npz") as archive:
                d = {
                    k: archive[k]
                    for k in [
                        "node",
                        "graph",
                        "dataset_idx",
                        "offsets",
                        "node_counts",
                        "size_bucket",
                    ]
                }
            for level in ["node", "graph"]:
                for c, case in enumerate(CASES):
                    mp = maps[level + "_" + case]
                    a, b, _, _ = bounds(mp)
                    mu = d[level][:, :, c].astype(float)
                    sig = np.exp(np.maximum(d[level][:, :, c + 4].astype(float), -3))
                    left = ndtr((a - mu) / sig).mean(0)
                    right = ndtr((mu - b) / sig).mean(0)
                    tail = left + right
                    zzero = float(
                        mp["scaler"].transform(
                            mp["state"]["power_transformer"].transform(
                                np.array([[0.0]])
                            )
                        )[0, 0]
                    )
                    # Probability of the interval [inverse lower bound, T(0)); the
                    # undefined lower tail must not be counted as finite negative y.
                    belowzero = ndtr((zzero - mu) / sig).mean(0) - left
                    for i, idx in enumerate(d["dataset_idx"]):
                        sl = (
                            slice(d["offsets"][i], d["offsets"][i + 1])
                            if level == "node"
                            else slice(i, i + 1)
                        )
                        rows.append(
                            dict(
                                repeat=r,
                                fold=f,
                                dataset_idx=int(idx),
                                size_um=int(
                                    [125, 177, 250, 354, 500][d["size_bucket"][i]]
                                ),
                                level=level,
                                case=case,
                                n=len(tail[sl]),
                                mean_outside_mass=float(tail[sl].mean()),
                                max_outside_mass=float(tail[sl].max()),
                                mean_lower_outside_mass=float(left[sl].mean()),
                                mean_upper_outside_mass=float(right[sl].mean()),
                                mean_negative_raw_mass=float(belowzero[sl].mean()),
                                max_negative_raw_mass=float(belowzero[sl].max()),
                            )
                        )
    df = pd.DataFrame(rows)
    df.to_csv(
        HERE / "stable_support_tails_per_sve.csv", index=False, float_format="%.17g"
    )
    output = []
    for (r, level, case), g in df.groupby(["repeat", "level", "case"]):
        row = dict(repeat=r, level=level, case=case, n=int(g.n.sum()), n_sves=len(g))
        for col in [
            "mean_outside_mass",
            "mean_lower_outside_mass",
            "mean_upper_outside_mass",
            "mean_negative_raw_mass",
        ]:
            row[col] = float(np.average(g[col], weights=g.n))
        for col in ["max_outside_mass", "max_negative_raw_mass"]:
            row[col] = float(g[col].max())
        output.append(row)
    pd.DataFrame(output).to_csv(
        HERE / "stable_support_tail_mass.csv", index=False, float_format="%.17g"
    )
    report = dict(
        status="complete",
        script_sha256=sha(__file__),
        formula="mean_k Phi((a-mu_k)/sigma_k)+mean_k Phi((mu_k-b)/sigma_k); upper survival computed directly.",
        meaning="The original transformed Gaussian mixture has nonzero mass beyond finite Yeo-Johnson image bounds even when all reported central intervals invert finitely; no conditioning or clipping performed.",
        supersedes="Numerically rounded mean_math_tail_mass/max_math_tail_mass fields in initial analysis metrics and CSVs for tiny-mass diagnostics only.",
    )
    (HERE / "stable_support_tail_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(pd.DataFrame(output).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
