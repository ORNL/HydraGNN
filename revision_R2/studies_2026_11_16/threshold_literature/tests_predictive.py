"""Numerical verification and bounded runtime audit, without cohort scoring."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import genpareto
import predictive as p
import core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def direct_deletion_score(values, u, v, seed=901, power=13):
    x = np.sort(values)
    y = x[x > u] - u
    unit = y.mean()
    y = y / unit
    delta = (v - u) / unit
    n, k = len(x), len(y)
    scores = []
    for i in range(n):
        if x[i] <= u:
            yy = y
            rate = (k + 0.5) / n
        else:
            j = i - (n - k)
            yy = np.delete(y, j)
            rate = (k - 0.5) / n
        shape, scale, lw, _ = p._posterior_nodes(yy, seed + i, power, power)
        if x[i] <= v:
            surv = np.exp(logsumexp(lw + p._logsurvival(delta, shape, scale)))
            scores.append(np.log1p(-rate * surv))
        else:
            ld = p._logpdf([(x[i] - u) / unit], shape, scale)[:, 0]
            scores.append(np.log(rate) + logsumexp(lw + ld) - np.log(unit))
    return float(np.sum(scores)), scores


def main():
    start = time.time()
    core.build_native()
    report = {}
    rng = np.random.default_rng(20261116)
    # Distribution implementation matches independent scipy across both signs.
    err = []
    for shape in [-0.49, -0.2, 0.0, 0.3, 0.95]:
        values = genpareto.ppf(np.linspace(0.001, 0.999, 100), shape, scale=1.7)
        expected = genpareto.logpdf(values, shape, scale=1.7)
        err.append(
            float(np.max(np.abs(p._logpdf(values, [shape], [1.7])[0] - expected)))
        )
    assert max(err) < 1.0e-9
    report["maximum_scipy_logdensity_difference"] = max(err)

    # Independent deletion integrations include below-u, between-u-and-v,
    # above-v and sample-maximum cases. This directly checks the LOO algebra.
    sample = genpareto.rvs(-0.3, scale=2, size=30, random_state=rng)
    u = float(np.quantile(sample, 0.5))
    v = float(np.quantile(sample, 0.7))
    score, detail = p.bayesian_cv_score(
        sample, u, v, seed=200, power=14, max_power=14, return_details=True
    )
    brute, brute_rows = direct_deletion_score(sample, u, v, power=14)
    differences = np.array(detail["log_predictive_densities"]) - brute_rows
    assert abs(score - brute) < 0.025, (score, brute, differences)
    assert np.max(np.abs(differences)) < 0.012
    report["independent_loo_integrations"] = {
        "n": 30,
        "importance_score": score,
        "separate_integral_score": brute,
        "difference": score - brute,
        "max_pointwise_difference": float(np.abs(differences).max()),
    }
    permuted = p.select_predictive(sample[::-1], seed=320, power=12, max_power=12)
    original = p.select_predictive(sample, seed=320, power=12, max_power=12)
    scaled = p.select_predictive(sample * 17 + 4, seed=320, power=12, max_power=12)
    for name in original:
        assert original[name]["raw_maximum"] == permuted[name]["raw_maximum"]
        expected = 17 * original[name]["raw_maximum"] + 4
        assert np.isclose(scaled[name]["raw_maximum"], expected, rtol=2.0e-5), (
            name,
            scaled[name]["raw_maximum"],
            expected,
        )
    report["permutation_invariance"] = True
    report["positive_affine_equivariance"] = True
    for field in [np.ones(20), np.arange(8, dtype=float)]:
        result = p.select_predictive(field)
        assert all(r["raw_maximum"] == field.max() for r in result.values())
    report["constant_and_sparse_fallback"] = True

    # Accuracy is checked on inputs, without reading any SVE-maximum labels.
    archive = (
        ROOT / "revision_R2/studies_2026_11_15/data/grain_predictions_repeat0_fold0.npz"
    )
    z = np.load(archive)
    fields = []
    for size in [125, 250, 500]:
        j = int(np.flatnonzero(z["size_um"] == size)[0])
        a, b = z["offsets"][j : j + 2]
        for case in [0, 3]:
            fields.append((int(size), int(case), z["pred_peak"][a:b, case]))
    benchmark = []
    for index, (size, case, field) in enumerate(fields):
        runs = {}
        for power in [10, 12, 14]:
            t = time.time()
            result = p.select_predictive(
                field, seed=500 + index, power=power, max_power=power
            )
            runs[power] = {
                "seconds": time.time() - t,
                "selected_q": result["predictive_select"]["quantile"],
                "selected_max": result["predictive_select"]["raw_maximum"],
                "average_max": result["predictive_average"]["raw_maximum"],
                "weights": [
                    c.get("weight", 0)
                    for c in result["predictive_select"]["diagnostics"]["candidates"]
                ],
                "scores": [
                    c.get("score")
                    for c in result["predictive_select"]["diagnostics"]["candidates"]
                ],
            }
        benchmark.append(
            {"size_um": size, "case_index": case, "n_grains": len(field), "runs": runs}
        )
        print(
            "benchmark",
            size,
            case,
            {
                k: (v["selected_q"], v["average_max"], v["seconds"])
                for k, v in runs.items()
            },
            flush=True,
        )
    report["real_input_convergence"] = benchmark
    report["seconds"] = time.time() - start
    report["status"] = "passed"
    (HERE / "predictive_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "real_input_convergence"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
