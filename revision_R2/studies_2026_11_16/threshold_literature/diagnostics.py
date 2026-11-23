"""Literature-based threshold diagnostics with explicit finite-sample adaptations.

Northrop--Coleman (2014): the multiple-threshold shape score statistic, using
expected information. Bader--Yan--Zhang (2018): fitted Anderson--Darling
statistics and the exact ForwardStop/StrongStop rejection formulas. The AD
null distributions are calibrated by a finite-size parametric bootstrap of
the common constrained MLE, not by the authors' asymptotic lookup table.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.special import roots_legendre
from scipy.stats import chi2

import core

HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "diagnostic_reference" / "ad_bootstrap_calibration.npz"
AD_SHAPES = np.array(
    [
        -0.49,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
)
AD_COUNTS = np.array([10, 15, 25, 40, 65, 100, 160, 250, 400, 650, 1000, 1600])
AD_REPLICATES = 1999
_TABLES = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_rejection_count(pvalues, rule="forward", alpha=0.05):
    """Return k in 0..m: hypotheses 1,...,k are rejected.

    The selected threshold has zero-based index k. k==m means every candidate
    was rejected. Values 0 and 1 are allowed and handled by exact limits.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a finite vector in [0,1]")
    if not len(p):
        return 0
    k = np.arange(1, len(p) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if rule == "forward":
            passed = -np.cumsum(np.log1p(-p)) / k <= alpha
        elif rule == "strong":
            log_value = np.cumsum((np.log(p) / k)[::-1])[::-1]
            passed = log_value <= np.log(alpha * k / len(p))
        else:
            raise ValueError(rule)
    return int(k[passed][-1]) if passed.any() else 0


def anderson_darling(excess, shape, scale):
    x = np.sort(np.asarray(excess, dtype=float))
    n = len(x)
    if n < 1 or scale <= 0 or not np.isfinite(x).all():
        return np.inf
    z = x / scale
    support = 1 + shape * z
    if np.any(support <= 0):
        return np.inf
    log_survival = -z if abs(shape) < 1e-10 else -np.log1p(shape * z) / shape
    # Stable both near zero CDF and near unit CDF; avoid rounding 1-S to 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_cdf = np.log(-np.expm1(log_survival))
    if not np.isfinite(log_cdf).all():
        return np.inf
    weight = 2 * np.arange(1, n + 1) - 1
    return float(-n - np.sum(weight * (log_cdf + log_survival[::-1])) / n)


def _log_survival_scores(z, shape, scale):
    """Derivatives of log GP survival in scale and shape."""
    z = np.asarray(z, dtype=float)
    t = 1 + shape * z
    ds = z / (scale * t)
    if abs(shape) < 1e-5:
        # Expansion through xi^3; |xi*z| is small in this branch in practice.
        dx = (
            z * z / 2
            - 2 * shape * z ** 3 / 3
            + 3 * shape ** 2 * z ** 4 / 4
            - 4 * shape ** 3 * z ** 5 / 5
        )
    else:
        dx = (np.log1p(shape * z) - shape * z / t) / (shape * shape)
    return ds, dx


def _density_scores(z, shape, scale):
    z = np.asarray(z, dtype=float)
    t = 1 + shape * z
    ds = (z - 1) / (scale * t)
    if abs(shape) < 1e-5:
        dx = (
            (z * z / 2 - z)
            + shape * (z * z - 2 * z ** 3 / 3)
            + shape ** 2 * (3 * z ** 4 / 4 - z ** 3)
            + shape ** 3 * (z ** 4 - 4 * z ** 5 / 5)
        )
    else:
        dx = np.log1p(shape * z) / (shape * shape) - (1 + shape) * z / (shape * t)
    return ds, dx


def piecewise_logpdf(y, thresholds, log_scale, shapes):
    """Equation (5), in (log sigma_1, xi_1,...,xi_m) coordinates."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(thresholds, dtype=float)
    xi = np.asarray(shapes, dtype=float)
    m = len(v)
    if len(xi) != m or v[0] != 0 or np.any(np.diff(v) <= 0):
        raise ValueError("threshold offsets must start at zero and increase")
    width = np.diff(v)
    sigma = np.r_[math.exp(log_scale), math.exp(log_scale) + np.cumsum(xi[:-1] * width)]
    if np.any(sigma <= 0):
        return np.full_like(y, -np.inf)
    logp = np.zeros(m)
    for j in range(m - 1):
        t = xi[j] * width[j] / sigma[j]
        if t <= -1:
            return np.full_like(y, -np.inf)
        logp[j + 1] = logp[j] - (
            width[j] / sigma[j] if abs(xi[j]) < 1e-10 else np.log1p(t) / xi[j]
        )
    bins = np.searchsorted(v, y, side="right") - 1
    out = np.full_like(y, -np.inf)
    for j in range(m):
        take = bins == j
        z = (y[take] - v[j]) / sigma[j]
        t = xi[j] * z
        if np.any(t <= -1):
            continue
        out[take] = (
            logp[j]
            - np.log(sigma[j])
            - (z if abs(xi[j]) < 1e-10 else (1 + 1 / xi[j]) * np.log1p(t))
        )
    return out


def piecewise_null_scores(y, thresholds, shape):
    """Analytic individual score vectors at sigma_1=1, all xi_j=shape."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(thresholds, dtype=float)
    m = len(v)
    width = np.diff(v)
    sigma = 1 + shape * v
    if np.any(sigma <= 0):
        raise ValueError("threshold outside null support")
    ds, dx = _log_survival_scores(width / sigma[:-1], shape, sigma[:-1])
    bins = np.searchsorted(v, y, side="right") - 1
    score = np.zeros((len(y), m + 1))
    for j in range(m):
        take = np.flatnonzero(bins == j)
        if not len(take):
            continue
        z = (y[take] - v[j]) / sigma[j]
        ls, lx = _density_scores(z, shape, sigma[j])
        score[take, 0] = ds[:j].sum() + ls
        for k in range(j):
            score[take, k + 1] = dx[k] + width[k] * (ds[k + 1 : j].sum() + ls)
        score[take, j + 1] = lx
    return score


def expected_score_information(thresholds, shape, quadrature_order=48):
    """Expected outer score product under the null; scale normalized to one.

    Finite probability intervals use Gauss--Legendre quadrature. The final
    unbounded probability interval is integrated exactly, avoiding the slow
    endpoint convergence as xi approaches -1/2. In that interval, scores are
    linear in 1, h1=(1-exp(-xi Q))/xi, h2=(Q-h1)/xi, Q~Exp(1).
    """
    v = np.asarray(thresholds, dtype=float)
    m = len(v)
    if shape <= -0.5:
        raise ValueError("expected information requires shape > -1/2")
    width = np.diff(v)
    sigma = 1 + shape * v
    if np.any(sigma <= 0):
        raise ValueError("threshold outside support")
    logsurv = -v if abs(shape) < 1e-10 else -np.log1p(shape * v) / shape
    survival = np.exp(logsurv)
    info = np.zeros((m + 1, m + 1))
    average = np.zeros(m + 1)
    nodes, weights = roots_legendre(quadrature_order)
    for j in range(m - 1):
        # Integrate over survival probability: no subtraction of nearly unit CDFs.
        lo, hi = survival[j + 1], survival[j]
        ss = lo + (nodes + 1) * (hi - lo) / 2
        y = -np.log(ss) if abs(shape) < 1e-10 else np.expm1(-shape * np.log(ss)) / shape
        sc = piecewise_null_scores(y, v, shape)
        ww = weights * (hi - lo) / 2
        info += (sc.T * ww) @ sc
        average += ww @ sc
    # Exact final interval conditional moments.
    j = m - 1
    ds, dx = _log_survival_scores(width / sigma[:-1], shape, sigma[:-1])
    coef = np.zeros((m + 1, 3))
    coef[0] = [ds.sum() - 1 / sigma[j], (1 + shape) / sigma[j], 0]
    for k in range(j):
        coef[k + 1] = [
            dx[k] + width[k] * (ds[k + 1 : j].sum() - 1 / sigma[j]),
            width[k] * (1 + shape) / sigma[j],
            0,
        ]
    coef[j + 1] = [0, -1, 1]
    a = 1 + shape
    b = 1 + 2 * shape
    moments = np.array(
        [
            [1, 1 / a, 1 / a],
            [1 / a, 2 / (a * b), (3 + 2 * shape) / (a * a * b)],
            [1 / a, (3 + 2 * shape) / (a * a * b), 2 * (3 + 2 * shape) / (a * a * b)],
        ]
    )
    info += survival[-1] * (coef @ moments @ coef.T)
    average += survival[-1] * (coef @ np.array([1, 1 / a, 1 / a]))
    return (info + info.T) / 2, average


def shape_stability_test(excess, threshold_offsets, fit, quadrature_order=48):
    """Northrop--Coleman score test of a common shape at all thresholds."""
    if not fit["success"]:
        return {"valid": False, "status": "gpd_fit_failed"}
    xi, scale = float(fit["shape"]), float(fit["scale"])
    v = np.asarray(threshold_offsets, dtype=float) / scale
    y = np.asarray(excess, dtype=float) / scale
    m = len(v)
    if m < 2:
        return {"valid": False, "status": "fewer_than_two_thresholds"}
    # A constrained-boundary optimum does not have the paper's ordinary null law.
    if xi <= -0.49 + 1e-5 or xi >= 0.95 - 1e-5:
        return {"valid": False, "status": "boundary_null_fit"}
    try:
        score = piecewise_null_scores(y, v, xi).sum(axis=0)
        per_info, average = expected_score_information(v, xi, quadrature_order)
        info = len(y) * per_info
        eig = np.linalg.eigvalsh(info)
        if not np.isfinite(info).all() or eig[0] <= 0 or eig[-1] / eig[0] > 1e12:
            return {"valid": False, "status": "ill_conditioned_information"}
        stat = float(score @ np.linalg.solve(info, score))
        if not np.isfinite(stat) or stat < -1e-10:
            return {"valid": False, "status": "invalid_score_statistic"}
        stat = max(stat, 0.0)
        return {
            "valid": True,
            "status": "ok",
            "statistic": stat,
            "df": m - 1,
            "pvalue": float(chi2.sf(stat, m - 1)),
            "information_condition": float(eig[-1] / eig[0]),
            "expected_score_max_abs": float(np.abs(average).max()),
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {"valid": False, "status": "information_evaluation_failed"}


def _table_cell(args):
    i, j, shape, n, replicates, seed = args
    rng = np.random.default_rng(seed)
    values = core.gpd_ppf(rng.random((replicates, n)), shape, 1.0)
    fits = core.fit_gpd_batch(values)
    stats = np.array(
        [
            anderson_darling(x, fit["shape"], fit["scale"])
            if fit["success"]
            else np.inf
            for x, fit in zip(values, fits)
        ]
    )
    failed = int((~np.isfinite(stats)).sum())
    if failed:
        raise RuntimeError(
            f"AD calibration fit failures: shape={shape}, n={n}, failed={failed}"
        )
    boundary = int(sum(bool(fit.get("boundary", False)) for fit in fits))
    return i, j, np.sort(stats), boundary


def build_ad_calibration(path=DEFAULT_TABLE, replicates=AD_REPLICATES, workers=12):
    """Build once from synthetic data only; no CPFEM or prediction labels used."""
    from concurrent.futures import ProcessPoolExecutor
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = 2026111605
    started = time.time()
    jobs = [
        (i, j, float(xi), int(n), int(replicates), seed + 10000 * i + j)
        for i, xi in enumerate(AD_SHAPES)
        for j, n in enumerate(AD_COUNTS)
    ]
    statistics = np.empty((len(AD_SHAPES), len(AD_COUNTS), replicates))
    boundary = np.zeros((len(AD_SHAPES), len(AD_COUNTS)), dtype=int)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (i, j, stats, nb) in enumerate(pool.map(_table_cell, jobs)):
            statistics[i, j] = stats
            boundary[i, j] = nb
            if (k + 1) % 12 == 0:
                print(
                    f"AD bootstrap table {k+1}/{len(jobs)}, seconds={time.time()-started:.1f}",
                    flush=True,
                )
    meta = {
        "replicates": int(replicates),
        "seed": seed,
        "core_sha256": _sha(core.__file__),
        "native_source_sha256": _sha(HERE / "core_native.cpp"),
        "calibration": "parametric bootstrap with same constrained MLE refitted",
        "pvalue": "plus-one empirical survival; bilinear interpolation in shape/log-count",
        "count_outside_grid": "clamped and explicitly flagged",
        "seconds": time.time() - started,
        "failed_fits": 0,
        "boundary_fit_counts": boundary.tolist(),
    }
    np.savez_compressed(
        path,
        shapes=AD_SHAPES,
        counts=AD_COUNTS,
        statistics=statistics,
        metadata=np.array(json.dumps(meta)),
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class ADCalibration:
    def __init__(self, path=DEFAULT_TABLE):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.shapes = z["shapes"]
            self.counts = z["counts"]
            self.statistics = z["statistics"]
            self.metadata = json.loads(str(z["metadata"]))
        if self.metadata["core_sha256"] != _sha(core.__file__):
            raise ValueError(
                "AD calibration uses a different GPD core; regenerate table"
            )
        if self.metadata.get("native_source_sha256") != _sha(HERE / "core_native.cpp"):
            raise ValueError(
                "AD calibration uses a different native fitter; regenerate table"
            )
        self.replicates = self.statistics.shape[-1]

    @staticmethod
    def _bracket(grid, value):
        j = int(np.clip(np.searchsorted(grid, value, side="right"), 1, len(grid) - 1))
        weight = float(np.clip((value - grid[j - 1]) / (grid[j] - grid[j - 1]), 0, 1))
        return j - 1, j, weight

    def pvalue(self, statistic, shape, count):
        i0, i1, a = self._bracket(self.shapes, shape)
        j0, j1, b = self._bracket(np.log(self.counts), np.log(count))

        def p(i, j):
            rank = np.searchsorted(self.statistics[i, j], statistic, side="left")
            return (self.replicates - rank + 1) / (self.replicates + 1)

        value = (1 - a) * ((1 - b) * p(i0, j0) + b * p(i0, j1)) + a * (
            (1 - b) * p(i1, j0) + b * p(i1, j1)
        )
        return float(value), {
            "shape_clamped": bool(shape < self.shapes[0] or shape > self.shapes[-1]),
            "count_clamped": bool(count < self.counts[0] or count > self.counts[-1]),
            "pvalue_resolution": 1 / (self.replicates + 1),
        }


def _result(candidate, method, status, **metadata):
    maximum, map_status = core.finite_maximum(candidate)
    return {
        "method": method,
        "selected_index": int(candidate["index"]),
        "selected_quantile": float(candidate["quantile"]),
        "threshold": float(candidate["threshold"]),
        "tail_count": int(candidate["count"]),
        "raw_maximum": float(maximum),
        "status": status,
        "map_status": map_status,
        **metadata,
    }


def _fallback(values, method, status, **metadata):
    return {
        "method": method,
        "selected_index": -1,
        "selected_quantile": None,
        "threshold": None,
        "tail_count": 0,
        "raw_maximum": float(np.max(values)),
        "status": status,
        "map_status": "predicted_maximum_fallback",
        **metadata,
    }


def select_diagnostics(
    values, seed=0, candidate_cache=None, calibration_path=DEFAULT_TABLE, alpha=0.05
):
    """Return AD ForwardStop, AD StrongStop, and shape-stability selections.

    Seed is accepted for common harness compatibility. The precomputed AD
    bootstrap table and score quadrature make this operation deterministic.
    """
    values = np.asarray(values, dtype=float)
    candidates = core.candidates(values) if candidate_cache is None else candidate_cache
    names = ["ad_forwardstop", "ad_strongstop", "shape_stability"]
    if not candidates:
        return {
            name: _fallback(values, name, "no_admissible_threshold") for name in names
        }
    path = str(Path(calibration_path).resolve())
    if path not in _TABLES:
        _TABLES[path] = ADCalibration(path)
    calibration = _TABLES[path]
    ad = []
    pvalues = []
    for cand in candidates:
        fit = cand["fit"]
        if not fit["success"]:
            ad.append(
                {
                    "index": int(cand["index"]),
                    "valid": False,
                    "status": "gpd_fit_failed",
                }
            )
            pvalues.append(None)
            continue
        statistic = anderson_darling(cand["excess"], fit["shape"], fit["scale"])
        if not np.isfinite(statistic):
            ad.append(
                {
                    "index": int(cand["index"]),
                    "valid": False,
                    "status": "invalid_ad_statistic",
                }
            )
            pvalues.append(None)
            continue
        p, info = calibration.pvalue(statistic, fit["shape"], cand["count"])
        ad.append(
            {
                "index": int(cand["index"]),
                "valid": True,
                "statistic": statistic,
                "pvalue": p,
                **info,
            }
        )
        pvalues.append(p)
    result = {}
    for name, rule in [("ad_forwardstop", "forward"), ("ad_strongstop", "strong")]:
        if any(p is None for p in pvalues):
            result[name] = _fallback(values, name, "gof_computation_failed", tests=ad)
            continue
        k = ordered_rejection_count(pvalues, rule, alpha)
        result[name] = (
            _fallback(
                values, name, "all_candidates_rejected", tests=ad, rejection_count=k
            )
            if k == len(candidates)
            else _result(candidates[k], name, "selected", tests=ad, rejection_count=k)
        )
    stability = []
    selected = None
    for i, cand in enumerate(candidates[:-1]):
        offsets = np.array([c["threshold"] - cand["threshold"] for c in candidates[i:]])
        test = shape_stability_test(cand["excess"], offsets, cand["fit"])
        stability.append({"index": int(cand["index"]), **test})
        if selected is None and test["valid"] and test["pvalue"] >= alpha:
            selected = cand
    if selected is not None:
        result["shape_stability"] = _result(
            selected, "shape_stability", "selected", tests=stability
        )
    elif len(candidates) == 1 and candidates[0]["fit"]["success"]:
        result["shape_stability"] = _result(
            candidates[0],
            "shape_stability",
            "single_candidate_no_stability_test",
            tests=stability,
        )
    else:
        result["shape_stability"] = _fallback(
            values, "shape_stability", "no_accepted_stability_test", tests=stability
        )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-table", action="store_true")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--replicates", type=int, default=AD_REPLICATES)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.build_table:
        print(
            json.dumps(
                build_ad_calibration(args.table, args.replicates, args.workers),
                indent=2,
            )
        )
