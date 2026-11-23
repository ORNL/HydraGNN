"""Independent expanded-bound likelihood checks, before field computation."""
from pathlib import Path
import importlib.util, json, time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import genpareto

HERE = Path(__file__).resolve().parent


def nll(par, y):
    xi, ls = par
    scale = np.exp(ls)
    if not -0.9 <= xi <= 0.95 or np.any(1 + xi * y / scale <= 0):
        return 1e100
    value = -genpareto.logpdf(y, c=xi, scale=scale).sum()
    return float(value) if np.isfinite(value) else 1e100


def main():
    spec = importlib.util.spec_from_file_location(
        "sensitivity_core", HERE / "lo090_n4/core.py"
    )
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    assert core.MIN_EXCESSES == 4 and core.SHAPE_BOUNDS == (-0.9, 0.95)
    rng = np.random.default_rng(2026111609)
    rows = []
    start = time.time()
    for xi in [-1.2, -0.9, -0.6, 0.0, 0.5, 1.2]:
        for n in [4, 10, 35, 200, 1500]:
            y = genpareto.rvs(xi, scale=1.3, size=n, random_state=rng)
            fit = core.fit_gpd(y)
            assert fit["success"], (xi, n, fit)
            assert np.all(1 + fit["shape"] * y / fit["scale"] > 0)
            yn = y / y.mean()
            reference = differential_evolution(
                lambda z: nll(z, yn),
                [(-0.9, 0.95), (-8, 7)],
                seed=n + 50,
                tol=1e-10,
                popsize=15,
                maxiter=500,
                polish=True,
            )
            gap = fit["nll"] - (reference.fun + n * np.log(y.mean()))
            assert gap < 2e-5, (xi, n, gap, fit, reference.x)
            transformed = core.fit_gpd(y * 1e-5)
            assert abs(fit["shape"] - transformed["shape"]) < 1e-6
            assert abs(fit["scale"] - transformed["scale"] / 1e-5) < 1e-5 * fit["scale"]
            rows.append(
                dict(
                    true_shape=xi,
                    n=n,
                    fit=fit,
                    independent_nll=float(reference.fun + n * np.log(y.mean())),
                    nll_gap=float(gap),
                )
            )
    # Bootstrap resamples can contain only one distinct value; a finite fit
    # and explicit maximum-map fallback are required for that case.
    for n in [4, 10]:
        fit = core.fit_gpd(np.ones(n))
        assert fit["success"]
        c = dict(threshold=0.0, excess=np.ones(n), fit=fit, maximum=1.0)
        assert core.finite_maximum(c) == (1.0, "maximum_fallback")
    result = dict(
        status="passed",
        seconds=time.time() - start,
        checks=rows,
        maximum_nll_gap=max(r["nll_gap"] for r in rows),
        constant_sample_checks=True,
        regularity="The -.9 lower bound is nonregular; no asymptotic score or AD law is used.",
    )
    (HERE / "numerical_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
