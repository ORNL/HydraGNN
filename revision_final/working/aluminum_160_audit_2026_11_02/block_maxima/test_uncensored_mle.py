"""Matched uncensored Gumbel MLE and full-data censor invariance checks.

Same exact B=16/32/64 and seed101 partitions as test_censored_blocks.py.
The uncensored MLE uses all B numeric block maxima. The same unconditional
maximum map is used for both r0 and Type-II censored r1/r3 MLE variants.
The influence audit raises the original numerical global maximum while
keeping source ordering/ranks/partitions fixed, then verifies all retained
censored fitting arrays unchanged and checks whether target ties remain.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r

from test_block_maxima import CACHE, HERE, metrics
from test_censored_blocks import fit_censored


def main():
    with CACHE.open("rb") as f:
        source = pickle.load(f)
    saved = pd.read_csv(HERE / "censored_partition_predictions.csv").set_index(
        ["bucket", "sve_idx", "repeat", "B", "censored_top_blocks"]
    )
    rows, checks, refits = [], [], []
    for ordinal, (_, rec) in enumerate(source.iterrows()):
        bucket, sve_idx = int(rec.bucket), int(rec.sve_idx)
        x = np.sort(np.asarray(rec.mg, dtype=float))
        N = len(x)
        truth = float(x[-1])
        changed = x.copy()
        changed[-1] = truth * 2 + 1
        for rep in range(5):
            rng = np.random.default_rng(
                np.random.SeedSequence([101, bucket, sve_idx, rep])
            )
            order = rng.permutation(N)
            shuffled = x[order]
            changed_shuffled = changed[order]
            for B in (16, 32, 64):
                m = N // B
                maxima = np.sort(shuffled[: B * m].reshape(B, m).max(axis=1))
                changed_maxima = np.sort(
                    changed_shuffled[: B * m].reshape(B, m).max(axis=1)
                )
                shift = float(maxima.mean())
                unit = float(maxima.std(ddof=1))
                row = dict(
                    bucket=bucket,
                    sve_idx=sve_idx,
                    repeat=rep,
                    B=B,
                    N=N,
                    block_size=m,
                    remainder=N - B * m,
                    truth=truth,
                )
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        loc0, scale0 = gumbel_r.fit((maxima - shift) / unit)
                    loc, scale = shift + unit * loc0, unit * scale0
                    pred = float(loc + scale * (np.log(N / m) + np.euler_gamma))
                    row.update(
                        valid=bool(np.isfinite(pred) and scale > 0),
                        location=loc,
                        scale=scale,
                        prediction=pred,
                        warnings=" | ".join(str(w.message) for w in caught),
                    )
                except Exception as exc:
                    row.update(valid=False, prediction=np.nan, error=repr(exc))
                rows.append(row)
                for r in (1, 3):
                    kept = maxima[:-r]
                    altered = changed_maxima[:-r]
                    same = bool(np.array_equal(kept, altered))
                    key = (bucket, sve_idx, rep, B, r)
                    checks.append(
                        dict(
                            bucket=bucket,
                            sve_idx=sve_idx,
                            repeat=rep,
                            B=B,
                            censored_top_blocks=r,
                            retained_arrays_bitwise_identical=same,
                            original_globalmax_in_retained=bool(np.any(kept == truth)),
                            original_globalmax_grain_count=int(np.sum(x == truth)),
                        )
                    )
                    if rep == 0 and ordinal in (0, 200, 400, 556):
                        perturbed = fit_censored(altered, r)
                        old = saved.loc[key]
                        pred2 = perturbed["location"] + perturbed["scale"] * (
                            np.log(N / m) + np.euler_gamma
                        )
                        refits.append(
                            dict(
                                bucket=bucket,
                                sve_idx=sve_idx,
                                B=B,
                                censored_top_blocks=r,
                                modified_globalmax=truth * 2 + 1,
                                original_globalmax=truth,
                                retained_arrays_bitwise_identical=same,
                                censored_prediction_difference=float(
                                    pred2 - old.prediction
                                ),
                                location_difference=float(
                                    perturbed["location"] - old.location
                                ),
                                scale_difference=float(perturbed["scale"] - old.scale),
                            )
                        )
    partitions = pd.DataFrame(rows)
    predictions = []
    for (B, bucket, sve_idx), group in partitions.groupby(["B", "bucket", "sve_idx"]):
        group = group.sort_values("repeat")
        for nr in (1, 5):
            sub = group.iloc[:nr]
            good = bool(sub.valid.all())
            predictions.append(
                dict(
                    method=f"block_mle_B{B}_r0_avg{nr}",
                    B=B,
                    n_partitions=nr,
                    bucket=bucket,
                    sve_idx=sve_idx,
                    truth=float(sub.truth.iloc[0]),
                    prediction=float(sub.prediction.mean()) if good else np.nan,
                    valid=good,
                )
            )
    pred = pd.DataFrame(predictions)
    summary = []
    for method, frame in pred.groupby("method"):
        for bucket in [30, 45, 90, 160, "pooled"]:
            sub = frame if bucket == "pooled" else frame[frame.bucket == bucket]
            summary.append(dict(method=method, bucket=bucket, **metrics(sub)))
    summary = pd.DataFrame(summary)
    partitions.to_csv(HERE / "uncensored_mle_partition_predictions.csv", index=False)
    pred.to_csv(HERE / "uncensored_mle_predictions.csv", index=False)
    summary.to_csv(HERE / "uncensored_mle_metrics.csv", index=False)
    censor = pd.read_csv(HERE / "censored_metrics.csv")
    old = pd.read_csv(HERE / "metrics.csv")
    comparison = pd.concat(
        [old[old.method == "original_fitted_q"], summary, censor], ignore_index=True
    )
    comparison.to_csv(HERE / "mle_comparison_metrics.csv", index=False)
    check = pd.DataFrame(checks)
    check.to_csv(HERE / "censored_target_perturbation_checks.csv", index=False)
    audit = dict(
        n_uncensored_partition_fits=len(partitions),
        n_failures=int((~partitions.valid).sum()),
        n_warning_rows=int(partitions.warnings.fillna("").ne("").sum()),
        fitting="Uncensored gumbel_r MLE after block mean/SD standardization; exact same random equal-size blocks as censored fits.",
        shared_mean_formula="location+scale*(log(N/m)+EulerGamma)",
        n_censor_array_invariance_checks=len(check),
        n_changed_retained_arrays=int((~check.retained_arrays_bitwise_identical).sum()),
        n_retained_original_numeric_maximum=int(
            check.original_globalmax_in_retained.sum()
        ),
        max_original_globalmax_multiplicity=int(
            check.original_globalmax_grain_count.max()
        ),
        perturbation="Raise only original global maximum to 2*maximum+1, preserve all sorted positions and random block assignment.",
        direct_refit_checks=refits,
        max_direct_refit_prediction_difference=float(
            max(abs(x["censored_prediction_difference"]) for x in refits)
        ),
        source_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Uncensored estimates contain the target; censored top1/top3 block fits do not contain its numeric value when no tied maximum remains. This is an observed-response diagnostic, not input-only prediction.",
    )
    (HERE / "uncensored_mle_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.pivot(index="method", columns="bucket", values="R2").to_string())
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "direct_refit_checks"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
