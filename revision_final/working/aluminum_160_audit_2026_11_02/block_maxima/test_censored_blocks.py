"""Type-II right-censored Gumbel likelihood on within-SVE block maxima.

Largest r=1/3 block maxima are identified by rank and their numeric values
discarded. The largest retained block maximum c right-censors r unknown
values. Fit sum(logpdf(uncensored))+r*logsf(c); then use the SAME
unconditional max-stability mean loc+scale*(log(N/m)+EulerGamma).
All actual source maximum values appear only in score targets, not fit.
This is a post-hoc observed-response influence diagnostic, not a new test.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import CensoredData, gumbel_r

from test_block_maxima import CACHE, HERE, SOURCE, metrics

BS = (16, 32, 64)
RS = (1, 3)
REPS = 5
SEED = 101
GAMMA = float(np.euler_gamma)


def manual_nll_and_gradient(theta, x, c, r):
    """Independent normalized likelihood; parameterization (loc,logscale)."""
    loc, logscale = theta
    scale = np.exp(logscale)
    z = (x - loc) / scale
    a = np.exp(np.minimum(-z, 700.0))
    zc = (c - loc) / scale
    ac = np.exp(np.minimum(-zc, 700.0))
    if ac < 1e-8:
        logsf = np.log(ac) - ac / 2
        h = 1 - ac / 2 + ac * ac / 12
    elif ac > 50:
        logsf = 0.0
        h = ac * np.exp(-ac)
    else:
        logsf = np.log(-np.expm1(-ac))
        h = ac / np.expm1(ac)
    nll = float(len(x) * logscale + np.sum(z + a) - r * logsf)
    grad = np.array(
        [(np.sum(a - 1) - r * h) / scale, len(x) + np.sum(z * (a - 1)) - r * h * zc],
        dtype=float,
    )
    return nll, grad


def fit_censored(retained, r, verify=False):
    # Only retained numeric values reach this function.
    shift = float(np.mean(retained))
    unit = float(np.std(retained, ddof=1))
    if not (np.isfinite(unit) and unit > 0):
        return dict(valid=False, error="zero or invalid retained spread")
    x = (retained - shift) / unit
    c = float(x[-1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loc0, scale0 = gumbel_r.fit(CensoredData(uncensored=x, right=[c] * r))
    theta = np.array([loc0, np.log(scale0)])
    nll, grad = manual_nll_and_gradient(theta, x, c, r)
    out = dict(
        valid=bool(np.isfinite(loc0) and np.isfinite(scale0) and scale0 > 0),
        location=shift + unit * loc0,
        scale=unit * scale0,
        normalized_nll=nll,
        gradient_norm=float(np.linalg.norm(grad)),
        warnings=" | ".join(str(w.message) for w in caught),
    )
    if verify:
        reference = float(
            -np.sum(gumbel_r.logpdf(x, loc=loc0, scale=scale0))
            - r * gumbel_r.logsf(c, loc=loc0, scale=scale0)
        )
        numeric_grad = np.array(
            [
                (
                    manual_nll_and_gradient(theta + np.eye(2)[j] * 1e-5, x, c, r)[0]
                    - manual_nll_and_gradient(theta - np.eye(2)[j] * 1e-5, x, c, r)[0]
                )
                / 2e-5
                for j in range(2)
            ]
        )
        independent = minimize(
            lambda t: manual_nll_and_gradient(t, x, c, r),
            theta,
            jac=True,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 1000},
        )
        out.update(
            manual_vs_scipy_nll_error=abs(nll - reference),
            analytical_vs_finite_diff_gradient_error=float(
                np.max(abs(grad - numeric_grad))
            ),
            independent_nll_improvement=nll - float(independent.fun),
            independent_location_change=unit * (independent.x[0] - loc0),
            independent_scale_change=unit * (np.exp(independent.x[1]) - scale0),
            independent_success=bool(independent.success),
            independent_message=str(independent.message),
        )
    return out


def save_metrics(records):
    partitions = pd.DataFrame(records)
    rows = []
    for (B, r, bucket, sve_idx), group in partitions.groupby(
        ["B", "censored_top_blocks", "bucket", "sve_idx"]
    ):
        group = group.sort_values("repeat")
        for nrep in (1, REPS):
            if len(group) < nrep:
                continue
            sub = group.iloc[:nrep]
            # Complete failure visibility: no survivor-only average.
            valid = bool(sub.valid.all())
            first = sub.iloc[0]
            rows.append(
                dict(
                    method=f"block_censored_B{B}_r{r}_avg{nrep}",
                    B=B,
                    censored_top_blocks=r,
                    n_partitions=nrep,
                    bucket=bucket,
                    sve_idx=sve_idx,
                    truth=first.truth,
                    prediction=float(sub.prediction.mean()) if valid else np.nan,
                    valid=valid,
                    N=int(first.N),
                    block_size=int(first.block_size),
                    remainder=int(first.remainder),
                    successful_partitions=int(sub.valid.sum()),
                )
            )
    pred = pd.DataFrame(rows)
    summaries = []
    for method, frame in pred.groupby("method", sort=False):
        for bucket in [30, 45, 90, 160, "pooled"]:
            sub = frame if bucket == "pooled" else frame[frame.bucket == bucket]
            summaries.append(dict(method=method, bucket=bucket, **metrics(sub)))
    summary = pd.DataFrame(summaries)
    partitions.to_csv(HERE / "censored_partition_predictions.csv", index=False)
    pred.to_csv(HERE / "censored_predictions.csv", index=False)
    summary.to_csv(HERE / "censored_metrics.csv", index=False)
    return partitions, summary


def synthetic_check():
    # This test is fixed independently of aluminum outcomes.
    rng = np.random.default_rng(11701)
    mu, beta = 2.0, 0.7
    checks = []
    for B in BS:
        true_mean = mu + beta * (np.log(B) + GAMMA)
        for r in RS:
            values = []
            naive = []
            for trial in range(100):
                sample = np.sort(rng.gumbel(mu, beta, size=B))
                fit = fit_censored(sample[:-r], r, verify=trial == 0)
                values.append(fit["location"] + fit["scale"] * (np.log(B) + GAMMA))
                kept = sample[:-r]
                naive.append(
                    float(
                        kept.mean() + kept.std(ddof=1) * np.sqrt(6) / np.pi * np.log(B)
                    )
                )
            checks.append(
                dict(
                    B=B,
                    censored_top_blocks=r,
                    replicates=100,
                    known_expected_maximum=true_mean,
                    censored_mean_estimate=float(np.mean(values)),
                    censored_mean_bias=float(np.mean(values) - true_mean),
                    naive_deleted_MoM_mean_bias=float(np.mean(naive) - true_mean),
                    censored_estimate_standard_error=float(np.std(values, ddof=1) / 10),
                )
            )
    # A large exact-Gumbel sample with substantial censoring checks
    # recovery of known parent parameters, not a small-B Monte Carlo claim.
    sample = np.sort(rng.gumbel(mu, beta, size=10000))
    large = fit_censored(sample[:-2000], 2000, verify=True)
    return dict(
        seed=11701,
        parent_location=mu,
        parent_scale=beta,
        fixed_small_sample_checks=checks,
        large_sample=dict(n=10000, right_censored=2000, **large),
    )


def main():
    start = time.time()
    with CACHE.open("rb") as f:
        source = pickle.load(f)
    records = []
    for rep in range(REPS):
        for ordinal, (_, rec) in enumerate(source.iterrows()):
            bucket, sve_idx = int(rec.bucket), int(rec.sve_idx)
            x = np.asarray(rec.mg, dtype=float)
            x = np.sort(x[np.isfinite(x) & (x > 0)])
            N = len(x)
            truth = float(x[-1])
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, bucket, sve_idx, rep])
            )
            shuffled = x[rng.permutation(N)]
            for B in BS:
                m = N // B
                # All B blocks have exactly m grains. Discard random remainder.
                bm = np.sort(shuffled[: B * m].reshape(B, m).max(axis=1))
                for r in RS:
                    retained = bm[:-r].copy()
                    verify = rep == 0 and ordinal in (0, 200, 400, 556)
                    record = dict(
                        bucket=bucket,
                        sve_idx=sve_idx,
                        repeat=rep,
                        B=B,
                        censored_top_blocks=r,
                        N=N,
                        block_size=m,
                        remainder=N - B * m,
                        censor_threshold=float(retained[-1]),
                        truth=truth,
                        original_maximum_excluded_from_fit=True,
                    )
                    try:
                        fit = fit_censored(retained, r, verify=verify)
                        record.update(fit)
                        record["prediction"] = (
                            (fit["location"] + fit["scale"] * (np.log(N / m) + GAMMA))
                            if fit["valid"]
                            else np.nan
                        )
                    except Exception as e:
                        record.update(valid=False, prediction=np.nan, error=repr(e))
                    records.append(record)
            if (ordinal + 1) % 200 == 0:
                print(
                    f"repeat {rep+1}/{REPS}: {ordinal+1}/{len(source)}; elapsed {time.time()-start:.1f}s",
                    flush=True,
                )
        partitions, summary = save_metrics(records)
        print(f"Completed repeat{rep+1}:", flush=True)
        print(
            summary.pivot(index="method", columns="bucket", values="R2").to_string(),
            flush=True,
        )
    simulations = synthetic_check()
    verified = partitions[
        partitions.get(
            "manual_vs_scipy_nll_error", pd.Series(index=partitions.index, dtype=float)
        ).notna()
    ]
    audit = dict(
        scipy_version=scipy.__version__,
        cache=str(CACHE),
        cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        B=list(BS),
        right_censored_top_block_counts=list(RS),
        partition_counts=[1, REPS],
        seed=SEED,
        formula="mu_B+beta_B*(log(N/m)+EulerGamma)",
        likelihood="sum(log(gumbel_pdf(retained_block_maxima)))+r*log(gumbel_survival(largest_retained_block_maximum))",
        scaling="Fit in units standardized by mean and sample SD of retained block maxima only; invert units after fitting.",
        hidden_values="Top r block values are discarded numerically after rank selection. Neither their values nor the target enter parameters, initialization, censor threshold, or transformations.",
        assumptions="Gumbel iid block maxima; seeded random nonspatial blocks; equal m=floor(N/B), discard random remainder and use N/m count correction.",
        limits="Exploratory on previously examined SVEs. Not input-only prediction. Fitted retained peaks still are responses from the evaluated SVE. No calibration across SVEs and no tuning B/r/seed on160.",
        failure_policy="All partition failures retained; a five-partition mean is invalid if any component failed. No fallback.",
        n_partition_fits=len(partitions),
        n_failures=int((~partitions.valid).sum()),
        n_warning_rows=int(partitions.warnings.fillna("").ne("").sum()),
        max_gradient_norm=float(partitions.gradient_norm.max()),
        n_independent_likelihood_checks=len(verified),
        max_manual_vs_scipy_nll_error=float(verified.manual_vs_scipy_nll_error.max()),
        max_analytical_vs_finite_difference_gradient_error=float(
            verified.analytical_vs_finite_diff_gradient_error.max()
        ),
        max_independent_nll_improvement=float(
            verified.independent_nll_improvement.max()
        ),
        synthetic_gumbel_checks=simulations,
        elapsed_seconds=time.time() - start,
    )
    (HERE / "censored_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "synthetic_gumbel_checks"}, indent=2
        ),
        flush=True,
    )
    print(json.dumps(simulations, indent=2), flush=True)


if __name__ == "__main__":
    main()
