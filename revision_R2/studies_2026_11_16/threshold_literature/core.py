"""Shared numerical rules for the controlled literature threshold comparison.

The bounded likelihood is a declared finite-sample adaptation shared by every
selector. It is not an unrestricted reproduction of any cited estimator.
"""
from pathlib import Path
import ctypes
import hashlib
import subprocess
import numpy as np
from scipy.special import digamma, gammaln

HERE = Path(__file__).resolve().parent
QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95)
MIN_EXCESSES = 10
SHAPE_BOUNDS = (-0.49, 0.95)
_LIB = None


def build_native():
    """Compile once explicitly before process-pool use; source-hashed library."""
    src = HERE / "core_native.cpp"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    lib = HERE / f"core_native_{digest}.so"
    if not lib.exists():
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(lib)],
            check=True,
        )
    return lib


def _native():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build_native()))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        _LIB.fit_batch.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
        _LIB.fit_batch.restype = None
    return _LIB


def fit_gpd_array(samples):
    """Return columns xi, sigma, negative log likelihood, success, boundary, iters."""
    a = np.ascontiguousarray(samples, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("samples must be a vector or a matrix")
    result = np.empty((len(a), 6), dtype=np.float64)
    if a.shape[1] < 2:
        result[:] = [0, 1, 1e100, 0, 0, 0]
    else:
        _native().fit_batch(a, len(a), a.shape[1], result)
    return result


def _record(z):
    good = bool(z[3])
    return dict(
        shape=float(z[0]),
        scale=float(z[1]),
        nll=float(z[2]),
        success=good,
        status="ok" if good else "fit_failed",
        boundary=bool(z[4]),
        iterations=int(z[5]),
    )


def fit_gpd(excess):
    """Zero-location GPD MLE with xi in [-.49,.95] and positive support."""
    return _record(fit_gpd_array(np.asarray(excess, dtype=float))[0])


def fit_gpd_batch(samples):
    return [_record(z) for z in fit_gpd_array(samples)]


def gpd_cdf(y, shape, scale):
    y = np.asarray(y, dtype=float)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if abs(shape) < 1e-10:
        ans = -np.expm1(-np.maximum(y, 0) / scale)
    else:
        support = 1 + shape * np.maximum(y, 0) / scale
        with np.errstate(divide="ignore", invalid="ignore"):
            ans = -np.expm1(-np.log(np.maximum(support, np.finfo(float).tiny)) / shape)
        ans = np.where(support > 0, ans, 1.0)
    return np.where(y < 0, 0.0, np.clip(ans, 0, 1))


def gpd_ppf(p, shape, scale):
    p = np.asarray(p, dtype=float)
    if scale <= 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid GPD parameters or probabilities")
    with np.errstate(divide="ignore"):
        a = -np.log1p(-p)
        return scale * a if abs(shape) < 1e-10 else scale * np.expm1(shape * a) / shape


def candidates(values, quantiles=QUANTILES, min_excesses=MIN_EXCESSES):
    x = np.sort(np.asarray(values, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty field required")
    result = []
    previous = None
    for q in quantiles:
        u = float(np.quantile(x, q, method="linear"))
        if previous is not None and u == previous:
            continue
        previous = u
        y = np.ascontiguousarray(x[x > u] - u)
        if len(y) < min_excesses:
            continue
        fit = fit_gpd(y)
        result.append(
            dict(
                index=len(result),
                quantile=float(q),
                threshold=u,
                excess=y,
                count=len(y),
                fit=fit,
                maximum=float(x[-1]),
                n_total=len(x),
            )
        )
    return result


def candidate_at_threshold(
    values, threshold, index=0, quantile=None, min_excesses=MIN_EXCESSES
):
    """Construct a fixed-threshold comparator; sparse records retain fallbacks."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty field required")
    u = float(threshold)
    if not np.isfinite(u):
        raise ValueError("finite threshold required")
    y = np.ascontiguousarray(x[x > u] - u)
    fit = (
        fit_gpd(y)
        if len(y) >= min_excesses
        else dict(
            shape=0.0,
            scale=1.0,
            nll=0.0,
            success=False,
            status="sparse",
            boundary=False,
            iterations=0,
        )
    )
    return dict(
        index=int(index),
        quantile=quantile,
        threshold=u,
        excess=y,
        count=len(y),
        fit=fit,
        maximum=float(x[-1]),
        n_total=len(x),
    )


def expected_excess_maximum(scale, shape, k):
    """Exact iid conditional-count GPD expected maximum, with xi<1."""
    if scale <= 0 or k < 1 or shape >= 1:
        raise ValueError("finite maximum requires sigma>0,k>=1,xi<1")
    if abs(shape) < 1e-7:
        return float(scale * (digamma(k + 1) + np.euler_gamma))
    a = gammaln(k + 1) + gammaln(1 - shape) - gammaln(k + 1 - shape)
    return float(scale * np.expm1(a) / shape)


def finite_maximum(candidate):
    """All-case scalar: fitted GPD, then exponential excesses, then field max."""
    fit = candidate["fit"]
    u = candidate["threshold"]
    y = np.asarray(candidate["excess"])
    k = len(y)
    if k < MIN_EXCESSES or (k and np.ptp(y) == 0):
        return float(candidate["maximum"]), "maximum_fallback"
    if (
        fit["success"]
        and k
        and fit["scale"] > 0
        and fit["shape"] < 1
        and np.all(1 + fit["shape"] * y / fit["scale"] > 0)
    ):
        pred = u + expected_excess_maximum(fit["scale"], fit["shape"], k)
        if np.isfinite(pred):
            return float(pred), "gpd"
    if k and np.isfinite(y).all() and np.mean(y) > 0:
        pred = u + expected_excess_maximum(float(np.mean(y)), 0.0, k)
        if np.isfinite(pred):
            return float(pred), "exponential_fallback"
    return float(candidate["maximum"]), "maximum_fallback"
