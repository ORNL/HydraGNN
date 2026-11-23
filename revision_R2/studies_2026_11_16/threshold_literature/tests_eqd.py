"""Independent numerical checks for the threshold-study numerical core."""
from pathlib import Path
import json, time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
from scipy.stats import genpareto
import core, eqd

HERE = Path(__file__).resolve().parent


def oracle_nll(par, y):
    xi, ls = par
    s = np.exp(ls)
    if xi < -0.49 or xi > 0.95 or np.any(1 + xi * y / s <= 0):
        return 1e100
    out = -genpareto.logpdf(y, c=xi, loc=0, scale=s).sum()
    return float(out) if np.isfinite(out) else 1e100


def main():
    core.build_native()
    rng = np.random.default_rng(2026111601)
    checks = []
    started = time.time()
    for xi in [-0.8, -0.4, 0, 0.2, 0.8, 1.2]:
        for n in [10, 35, 200, 1500]:
            y = genpareto.rvs(c=xi, scale=1.7, size=n, random_state=rng)
            fit = core.fit_gpd(y)
            assert fit["success"], (xi, n, fit)
            assert np.all(1 + fit["shape"] * y / fit["scale"] > 0)
            norm = y / y.mean()
            opt = differential_evolution(
                lambda par: oracle_nll(par, norm),
                [(-0.49, 0.95), (-7, 6)],
                seed=1000 + n,
                tol=1e-10,
                popsize=12,
                maxiter=400,
                polish=True,
            )
            ref = float(opt.fun) + n * np.log(y.mean())
            difference = fit["nll"] - ref
            assert difference < 2e-5, (xi, n, fit, ref, opt.x)
            replay = core.fit_gpd(y * 1e-5)
            assert abs(replay["shape"] - fit["shape"]) < 1e-6
            assert abs(replay["scale"] / 1e-5 - fit["scale"]) < 1e-5 * fit["scale"]
            checks.append(
                dict(true_shape=xi, n=n, fit=fit, oracle_nll=ref, nll_gap=difference)
            )
    quadrature = []
    for xi in [-0.49, 0, 1e-8, 0.2, 0.8]:
        for k in [1, 5, 100]:
            closed = core.expected_excess_maximum(1.7, xi, k)
            numeric = quad(
                lambda p: float(genpareto.ppf(p, c=xi, scale=1.7)) * k * p ** (k - 1),
                0,
                1,
                epsabs=1e-8,
                epsrel=1e-9,
                limit=500,
            )[0]
            rel = abs(closed - numeric) / max(1, abs(numeric))
            assert rel < 1e-6, (xi, k, closed, numeric)
            quadrature.append(dict(shape=xi, k=k, relative_error=rel))
    p = np.linspace(0.00001, 0.99999, 100)
    for xi in [-0.49, 0, 0.2, 0.95]:
        q = core.gpd_ppf(p, xi, 1.7)
        np.testing.assert_allclose(q, genpareto.ppf(p, c=xi, scale=1.7), rtol=1e-12)
        np.testing.assert_allclose(core.gpd_cdf(q, xi, 1.7), p, atol=1e-12)
    # Independent transparent implementation of equation (2), including type7.
    y = rng.exponential(size=23)
    B = 17
    m = 91
    seed = 9834
    actual = eqd.discrepancy(y, seed, B, m)
    samples = np.random.default_rng(seed).choice(y, size=(B, len(y)), replace=True)
    probs = np.arange(1, m + 1) / (m + 1)
    manual = []
    for sample in samples:
        fit = core.fit_gpd(sample)
        empirical = np.quantile(sample, probs, method="linear")
        theory = genpareto.ppf(probs, c=fit["shape"], scale=fit["scale"])
        manual.append(np.mean(abs(theory - empirical)))
    np.testing.assert_allclose(actual["score"], np.mean(manual), rtol=1e-12)
    fallback = eqd.select_eqd(np.ones(20), 3)
    assert fallback["raw_maximum"] == 1 and fallback["status"] == "maximum_fallback"
    sample = rng.exponential(size=500)
    t = time.time()
    result = eqd.select_eqd(sample, 43)
    seconds = time.time() - t
    assert np.isfinite(result["raw_maximum"])
    report = dict(
        status="passed",
        mle_checks=checks,
        quadrature=quadrature,
        eqd_equation2_score=actual["score"],
        manual_score=float(np.mean(manual)),
        benchmark_n500_seconds=seconds,
        total_seconds=time.time() - started,
        selected_quantile=result["selected_quantile"],
        bootstrap_failures=result["bootstrap_failures"],
    )
    (HERE / "eqd_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["mle_checks", "quadrature"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
