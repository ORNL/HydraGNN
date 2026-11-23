"""Independent numerical and selection checks for literature diagnostics."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import genpareto
import core
import diagnostics as d

HERE = Path(__file__).resolve().parent


def run_checks():
    core.build_native()
    rng = np.random.default_rng(981733)
    report = {}
    # Explicit sequences test count/index convention, including all/no rejections.
    patterns = [
        ([0.5, 0.6, 0.8], 0, 0),
        ([0.001, 0.001, 0.8], 2, 2),
        ([0.001, 0.001, 0.001], 3, 2),
        ([1e-6, 1e-6, 1e-6], 3, 3),
        ([0, 0, 0], 3, 3),
        ([1, 1, 1], 0, 0),
        ([0.01, 0.8, 0.001], 1, 1),
    ]
    actual = []
    for p, forward, strong in patterns:
        a = d.ordered_rejection_count(p, "forward")
        b = d.ordered_rejection_count(p, "strong")
        assert (a, b) == (forward, strong), (p, a, b, forward, strong)
        actual.append({"pvalues": p, "forward": a, "strong": b})
    report["ordered_rules"] = actual
    # Compare the analytic score with centered finite differences of the original
    # multiple-threshold likelihood, without differentiating test code.
    gradient = []
    information = []
    cdfcheck = []
    for xi in [-0.48, -0.25, 0.0, 0.2, 0.8]:
        v = core.gpd_ppf(np.array([0, 0.25, 0.5, 0.75]), xi, 1.0)
        y = core.gpd_ppf(rng.uniform(0.001, 0.995, 150), xi, 1.0)
        score = d.piecewise_null_scores(y, v, xi)
        theta = np.r_[0.0, np.repeat(xi, len(v))]
        numerical = np.empty_like(score)
        h = 2e-6
        for k in range(len(theta)):
            plus = theta.copy()
            minus = theta.copy()
            plus[k] += h
            minus[k] -= h
            numerical[:, k] = (
                d.piecewise_logpdf(y, v, plus[0], plus[1:])
                - d.piecewise_logpdf(y, v, minus[0], minus[1:])
            ) / (2 * h)
        error = float(np.max(np.abs(score - numerical) / (1 + np.abs(numerical))))
        assert error < 2e-6, (xi, error)
        gradient.append({"shape": xi, "max_scaled_error": error})
        info, mean = d.expected_score_information(v, xi, 48)
        info2, mean2 = d.expected_score_information(v, xi, 96)
        error2 = float(np.max(np.abs(info - info2) / (1 + np.abs(info2))))
        assert error2 < 1e-8, (xi, error2)
        assert np.abs(mean).max() < 1e-9, (xi, mean)
        assert np.linalg.eigvalsh(info).min() > 0
        # Independent adaptive integration of original score functions on every
        # finite interval and the final interval in exponential probability.
        independent = np.empty_like(info)
        log_surv = -np.log1p(xi * v) / xi if xi else -v
        zbreak = -log_surv
        # Near -1/2 the exact tail moments are particularly valuable; the
        # ordinary adaptive integral needs a long finite cutoff plus a tail.
        if xi > -0.4:
            for a in range(len(v) + 1):
                for b in range(a, len(v) + 1):

                    def f(z):
                        if z > 350:
                            return 0.0
                        if xi < 0 and z > 70:
                            return 0.0
                        yy = np.expm1(xi * z) / xi if xi else z
                        ss = d.piecewise_null_scores(np.array([yy]), v, xi)[0]
                        return float(ss[a] * ss[b] * np.exp(-z))

                    val = sum(
                        quad(
                            f, float(lo), float(hi), epsabs=1e-9, epsrel=1e-8, limit=200
                        )[0]
                        for lo, hi in zip(zbreak, np.r_[zbreak[1:], np.inf])
                    )
                    independent[a, b] = independent[b, a] = val
            integral_error = float(
                np.max(np.abs(info - independent) / (1 + np.abs(independent)))
            )
            assert integral_error < 2e-7, (xi, integral_error)
        else:
            integral_error = None
        information.append(
            {
                "shape": xi,
                "quadrature_48_vs_96": error2,
                "expected_score_max_abs": float(np.abs(mean).max()),
                "adaptive_integral_max_scaled_error": integral_error,
            }
        )
        gp = np.sort(core.gpd_ppf(rng.random(100), xi, 2.3))
        zz = genpareto.cdf(gp, xi, scale=2.3)
        direct = (
            -100
            - np.sum((2 * np.arange(1, 101) - 1) * (np.log(zz) + np.log1p(-zz[::-1])))
            / 100
        )
        stable = d.anderson_darling(gp, xi, 2.3)
        assert abs(stable - direct) < 1e-8
        cdfcheck.append({"shape": xi, "AD_absolute_difference": abs(stable - direct)})
    report["analytic_score_finite_difference"] = gradient
    report["expected_information"] = information
    report["AD_independent_CDF_check"] = cdfcheck
    # Scalar GPD null for m=1 has known Fisher information in (log scale,shape).
    one = []
    for xi in [-0.49, -0.2, 0.0, 0.4, 0.95]:
        info, mean = d.expected_score_information(np.array([0.0]), xi)
        expected = np.array(
            [
                [1 / (1 + 2 * xi), 1 / ((1 + xi) * (1 + 2 * xi))],
                [1 / ((1 + xi) * (1 + 2 * xi)), 2 / ((1 + xi) * (1 + 2 * xi))],
            ]
        )
        assert np.allclose(info, expected, rtol=1e-12, atol=1e-12), (xi, info, expected)
        one.append({"shape": xi, "max_error": float(np.max(abs(info - expected)))})
    report["single_GPD_information_identity"] = one
    if d.DEFAULT_TABLE.exists():
        selections = []
        for n in [5, 12, 25, 50, 200, 1000]:
            field = rng.exponential(size=n)
            result = d.select_diagnostics(field, seed=1)
            assert len(result) == 3
            assert all(np.isfinite(r["raw_maximum"]) for r in result.values())
            assert result == d.select_diagnostics(field, seed=999)
            selections.append(
                {
                    "n": n,
                    "results": {
                        name: {k: v for k, v in rec.items() if k != "tests"}
                        for name, rec in result.items()
                    },
                }
            )
        constant = d.select_diagnostics(np.ones(100))
        assert all(
            r["raw_maximum"] == 1 and r["status"] == "no_admissible_threshold"
            for r in constant.values()
        )
        report["finite_deterministic_selection"] = selections
    # Null and deliberately piecewise alternative: fixed theoretical thresholds,
    # outside the selection problem. No assertion that p-values control spatial FIP.
    sim = []
    for label in ["exponential_null", "uniform_gp_hybrid"]:
        rejections = []
        valid = 0
        v = np.array([0.0, 0.25, 0.5, 1.0, 1.5])
        for r in range(250):
            if label == "exponential_null":
                x = rng.exponential(size=400)
            else:
                u = rng.random(400)
                x = np.where(
                    u <= 0.75,
                    u / 0.75,
                    1 + core.gpd_ppf(np.clip((u - 0.75) / 0.25, 0, 1), 0.1, 1 / 3),
                )
            fit = core.fit_gpd(x)
            t = d.shape_stability_test(x, v, fit)
            if t["valid"]:
                valid += 1
                rejections.append(t["pvalue"] < 0.05)
        sim.append(
            {
                "distribution": label,
                "replicates": 250,
                "valid": valid,
                "rejection_rate": float(np.mean(rejections)),
            }
        )
    assert sim[0]["rejection_rate"] < 0.12, sim
    assert sim[1]["rejection_rate"] > 0.4, sim
    report["score_simulation"] = sim
    report["status"] = "passed"
    return report


def validate_ad_table(path=d.DEFAULT_TABLE, replicates=700):
    """Independent off-grid null simulations assess interpolation/calibration."""
    calibration = d.ADCalibration(path)
    cases = [
        (-0.445, 12),
        (-0.35, 20),
        (-0.15, 32),
        (0.05, 80),
        (0.25, 200),
        (0.55, 500),
        (0.85, 1200),
    ]
    rows = []
    for i, (xi, n) in enumerate(cases):
        rng = np.random.default_rng(882000 + i)
        x = core.gpd_ppf(rng.random((replicates, n)), xi, 1.0)
        fits = core.fit_gpd_batch(x)
        stats = np.array(
            [
                d.anderson_darling(xx, fit["shape"], fit["scale"])
                for xx, fit in zip(x, fits)
            ]
        )
        # Interpolation at true parameters isolates lookup error. Plug-in estimate
        # p-values additionally expose finite-sample parameter uncertainty.
        known = np.array([calibration.pvalue(a, xi, n)[0] for a in stats])
        plug = np.array(
            [calibration.pvalue(a, fit["shape"], n)[0] for a, fit in zip(stats, fits)]
        )
        rows.append(
            {
                "shape": xi,
                "count": n,
                "replicates": replicates,
                "known_shape_rejection_05": float(np.mean(known <= 0.05)),
                "fitted_shape_rejection_05": float(np.mean(plug <= 0.05)),
                "known_shape_rejection_10": float(np.mean(known <= 0.10)),
                "fitted_shape_rejection_10": float(np.mean(plug <= 0.10)),
                "boundary_fraction": float(np.mean([f["boundary"] for f in fits])),
            }
        )
    return rows


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--validate-table", action="store_true")
    args = p.parse_args()
    started = time.time()
    if args.validate_table:
        report = {"AD_off_grid_validation": validate_ad_table(), "status": "completed"}
        dest = HERE / "diagnostic_reference/ad_interpolation_validation.json"
    else:
        report = run_checks()
        dest = HERE / "diagnostic_reference/diagnostics_verification.json"
    report["seconds"] = time.time() - started
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
