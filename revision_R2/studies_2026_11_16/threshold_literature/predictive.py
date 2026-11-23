"""Bayesian common-threshold predictive selection, after Northrop et al. (2017).

The scoring structure follows their equations (2)--(8), including a separate
posterior for the maximum-deleted sample.  The MDI(0.6) prior is restricted to
the benchmark's common shape interval.  Seeded Sobol importance quadrature
replaces posterior MCMC.  Selection/averaging acts on the shared finite-count
MLE maximum maps to isolate threshold choice from the final map estimator.
"""
from functools import lru_cache
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp, ndtri, gammaln
from scipy.stats import chi2, qmc

try:
    from . import core
except ImportError:
    import core

SHAPE_LOW = -0.49
SHAPE_HIGH = 0.95
MDI_A = 0.6
DEFAULT_POWER = 12
MAX_POWER = 14
PROPOSAL_DF = 5.0


def _logpdf(y, shape, scale):
    """Rows are parameter points, columns are nonnegative excesses."""
    y = np.atleast_1d(np.asarray(y, float))
    shape = np.atleast_1d(np.asarray(shape, float))
    scale = np.atleast_1d(np.asarray(scale, float))
    z = y[None, :] / scale[:, None]
    a = shape[:, None]
    valid = (1 + a * z) > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = -np.log(scale[:, None]) - (1 + 1 / a) * np.log1p(a * z)
        close = np.abs(shape) < 1.0e-6
        if close.any():
            ac, zc = a[close], z[close]
            result[close] = (
                -np.log(scale[close, None])
                - zc
                + ac * (zc * zc / 2 - zc)
                + ac * ac * (zc * zc / 2 - zc ** 3 / 3)
            )
    return np.where(valid, result, -np.inf)


def _logsurvival(y, shape, scale):
    shape, scale = np.asarray(shape), np.asarray(scale)
    z = float(y) / scale
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = -np.log1p(shape * z) / shape
    close = np.abs(shape) < 1.0e-6
    result[close] = -z[close] + shape[close] * z[close] ** 2 / 2
    return np.where(1 + shape * z > 0, result, -np.inf)


def _parameters(z):
    z = np.asarray(z)
    with np.errstate(over="ignore"):
        scale = np.exp(z[..., 0])
    shape = SHAPE_LOW + (SHAPE_HIGH - SHAPE_LOW) * expit(z[..., 1])
    return shape, scale


def _posterior_log(z, y):
    z = np.atleast_2d(z)
    shape, scale = _parameters(z)
    likelihood = _logpdf(y, shape, scale).sum(axis=1)
    # The log-scale Jacobian cancels the scale^-1 part of the MDI prior.
    prior = (
        -MDI_A * shape
        + math.log(SHAPE_HIGH - SHAPE_LOW)
        + log_expit(z[:, 1])
        + log_expit(-z[:, 1])
    )
    result = likelihood + prior
    return np.where(np.isfinite(result), result, -np.inf)


def _mode_and_covariance(y, initial_fit=None):
    fit = initial_fit if initial_fit is not None else core.fit_gpd(y)
    shape = float(fit.get("shape", 0.0)) if fit.get("success", False) else 0.0
    scale = (
        float(fit.get("scale", np.mean(y)))
        if fit.get("success", False)
        else float(np.mean(y))
    )
    shape = float(np.clip(shape, SHAPE_LOW + 0.025, SHAPE_HIGH - 0.025))
    scale = max(scale, -shape * float(np.max(y)) * 1.02, 1.0e-12)
    probability = (shape - SHAPE_LOW) / (SHAPE_HIGH - SHAPE_LOW)
    initial = np.array([np.log(scale), np.log(probability / (1 - probability))])

    def objective(z):
        result = _posterior_log(z, y)[0]
        return -float(result) if np.isfinite(result) else 1.0e50

    answer = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 160, "xatol": 2.0e-6, "fatol": 2.0e-7},
    )
    mode = answer.x if np.isfinite(answer.fun) and answer.fun < 1.0e40 else initial
    # A numerical two-dimensional Hessian uses the posterior density in the
    # unbounded integration coordinates, including both prior Jacobians.
    step = 1.0e-3
    eye = np.eye(2) * step
    f0 = objective(mode)
    hessian = np.empty((2, 2))
    for i in range(2):
        hessian[i, i] = (
            objective(mode + eye[i]) - 2 * f0 + objective(mode - eye[i])
        ) / step ** 2
    hessian[0, 1] = hessian[1, 0] = (
        objective(mode + eye[0] + eye[1])
        - objective(mode + eye[0] - eye[1])
        - objective(mode - eye[0] + eye[1])
        + objective(mode - eye[0] - eye[1])
    ) / (4 * step ** 2)
    eig, vectors = np.linalg.eigh(hessian)
    hessian_regularized = bool(np.any(eig < 0.05))
    eig = np.clip(eig, 0.05, 1.0e8)
    covariance = (vectors * (1 / eig)) @ vectors.T
    # Inflate the local scale, while Student-t tails cover non-Gaussian tails.
    covariance *= 1.4
    return (
        mode,
        covariance,
        {
            "mode_converged": bool(answer.success),
            "hessian_regularized": hessian_regularized,
        },
    )


@lru_cache(maxsize=128)
def _standard_points(power, seed):
    engine = qmc.Sobol(3, scramble=True, seed=int(seed) % 2 ** 32)
    uniform = engine.random_base2(int(power))
    normals = ndtri(uniform[:, :2])
    radius = np.sqrt(chi2.ppf(uniform[:, 2], PROPOSAL_DF) / PROPOSAL_DF)
    return normals / radius[:, None]


def _posterior_nodes(
    y, seed, power=DEFAULT_POWER, max_power=MAX_POWER, initial_fit=None
):
    mode, covariance, diagnostics = _mode_and_covariance(y, initial_fit)
    factor = np.linalg.cholesky(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    norm = (
        gammaln((PROPOSAL_DF + 2) / 2)
        - gammaln(PROPOSAL_DF / 2)
        - np.log(PROPOSAL_DF * np.pi)
        - 0.5 * logdet
    )
    for actual_power in range(power, max_power + 1):
        standard = _standard_points(actual_power, seed)
        points = mode + standard @ factor.T
        logq = norm - ((PROPOSAL_DF + 2) / 2) * np.log1p(
            np.sum(standard ** 2, axis=1) / PROPOSAL_DF
        )
        logweight = _posterior_log(points, y) - logq
        finite = np.isfinite(logweight)
        if not finite.any():
            continue
        logweight = logweight[finite]
        logweight -= logsumexp(logweight)
        ess = float(np.exp(-logsumexp(2 * logweight)))
        if ess >= max(160, 2 ** actual_power * 0.12) or actual_power == max_power:
            shape, scale = _parameters(points[finite])
            diagnostics.update(
                nodes=2 ** actual_power,
                finite_nodes=int(finite.sum()),
                effective_sample_size=ess,
                max_normalized_weight=float(np.exp(logweight.max())),
                power=actual_power,
            )
            return shape, scale, logweight, diagnostics
    raise ArithmeticError("No posterior quadrature node has finite density.")


def bayesian_cv_score(
    values,
    threshold,
    validation_threshold,
    seed=0,
    power=DEFAULT_POWER,
    max_power=MAX_POWER,
    return_details=False,
):
    """LOO BGP predictive log-score at a common validation threshold.

    Thresholds remain fixed during deletion, as in the cited formulation.
    A truncated MDI(.6) shape/scale prior and independent Beta(.5,.5) prior
    are integrated.  No observed SVE maximum is supplied to this function.
    """
    x = np.sort(np.asarray(values, float))
    n = len(x)
    y_raw = x[x > threshold] - threshold
    k = len(y_raw)
    if k < 4 or validation_threshold < threshold:
        raise ValueError("Need at least four excesses and v >= u.")
    unit = float(np.mean(y_raw))
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("Positive, finite excess mean required.")
    y = y_raw / unit
    delta = float((validation_threshold - threshold) / unit)
    shape, scale, logweight, full_diagnostics = _posterior_nodes(
        y, seed, power, max_power
    )
    logg = _logpdf(y, shape, scale)
    logsurvival = _logsurvival(delta, shape, scale)
    p_after_body_deletion = (k + 0.5) / n
    p_after_tail_deletion = (k - 0.5) / n
    logpredictions = []
    if n > k:
        survival = float(np.exp(logsumexp(logweight + logsurvival)))
        body_logp = float(np.log1p(-p_after_body_deletion * survival))
        score = (n - k) * body_logp
        logpredictions.extend([body_logp] * (n - k))
    else:
        score = 0.0
    loo_ess = []
    for j in range(k - 1):
        inverse = logweight - logg[:, j]
        denominator = logsumexp(inverse)
        loo_ess.append(float(np.exp(2 * denominator - logsumexp(2 * inverse))))
        if y[j] <= delta:
            survival = float(np.exp(logsumexp(inverse + logsurvival) - denominator))
            logp = float(np.log1p(-p_after_tail_deletion * min(survival, 1.0)))
        else:
            logp = float(np.log(p_after_tail_deletion) - denominator - np.log(unit))
        score += logp
        logpredictions.append(logp)
    # For a negative shape the maximum-deleted posterior has additional
    # support.  Reweighting only the full-data posterior would miss it.
    shape_minus, scale_minus, logweight_minus, minus_diagnostics = _posterior_nodes(
        y[:-1], int(seed) + 104729, power, max_power
    )
    if y[-1] <= delta:
        logsurv_minus = _logsurvival(delta, shape_minus, scale_minus)
        survival = float(np.exp(logsumexp(logweight_minus + logsurv_minus)))
        max_logp = float(np.log1p(-p_after_tail_deletion * min(survival, 1.0)))
    else:
        logg_max = _logpdf([y[-1]], shape_minus, scale_minus)[:, 0]
        max_logp = float(
            np.log(p_after_tail_deletion)
            + logsumexp(logweight_minus + logg_max)
            - np.log(unit)
        )
    score += max_logp
    logpredictions.append(max_logp)
    if not np.isfinite(score):
        raise ArithmeticError(
            "Predictive quadrature did not resolve a positive density."
        )
    details = {
        "score": float(score),
        "threshold": float(threshold),
        "validation_threshold": float(validation_threshold),
        "count": k,
        "validation_count": int(np.sum(y > delta)),
        "full_posterior": full_diagnostics,
        "maximum_deleted_posterior": minus_diagnostics,
        "minimum_nonmaximum_loo_ess": min(loo_ess) if loo_ess else None,
        "maximum_log_predictive_density": max_logp,
    }
    if return_details:
        details["log_predictive_densities"] = logpredictions
    return float(score), details


def select_predictive(
    values, seed=0, candidate_cache=None, power=DEFAULT_POWER, max_power=MAX_POWER
):
    """Return selection and averaging using common-v Bayesian LOO weights."""
    x = np.sort(np.asarray(values, float))
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("A nonempty, finite predicted grain field is required.")
    available = core.candidates(x) if candidate_cache is None else candidate_cache
    available = [c for c in available if c["fit"]["success"]]
    fallback = {
        "selected_index": None,
        "threshold": None,
        "quantile": None,
        "raw_maximum": float(x[-1]),
        "status": "no_valid_candidate_predicted_maximum",
        "diagnostics": {"candidate_count": 0},
    }
    if not available:
        return {
            "predictive_select": fallback.copy(),
            "predictive_average": fallback.copy(),
        }
    validation_threshold = max(c["threshold"] for c in available)
    records = []
    for i, candidate in enumerate(available):
        try:
            score, detail = bayesian_cv_score(
                x,
                candidate["threshold"],
                validation_threshold,
                seed=int(seed) + 1009 * int(candidate["index"]),
                power=power,
                max_power=max_power,
            )
            raw, status = core.finite_maximum(candidate)
            records.append(
                {
                    "index": int(candidate["index"]),
                    "quantile": float(candidate["quantile"]),
                    "threshold": float(candidate["threshold"]),
                    "raw_maximum": float(raw),
                    "score": score,
                    "map_status": status,
                    "quadrature": detail,
                }
            )
        except (
            ArithmeticError,
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as error:
            records.append(
                {
                    "index": int(candidate["index"]),
                    "quantile": float(candidate["quantile"]),
                    "threshold": float(candidate["threshold"]),
                    "score": None,
                    "status": "quadrature_failed",
                    "error": str(error),
                }
            )
    resolved = [r for r in records if r["score"] is not None]
    if not resolved:
        fallback["status"] = "all_quadratures_failed_predicted_maximum"
        fallback["diagnostics"] = {
            "candidate_count": len(available),
            "candidates": records,
        }
        return {
            "predictive_select": fallback.copy(),
            "predictive_average": fallback.copy(),
        }
    logscores = np.array([r["score"] for r in resolved])
    weights = np.exp(logscores - logsumexp(logscores))
    for r, w in zip(resolved, weights):
        r["weight"] = float(w)
    winner = resolved[int(np.argmax(logscores))]
    diagnostics = {
        "candidate_count": len(available),
        "resolved_count": len(resolved),
        "validation_threshold": float(validation_threshold),
        "candidates": records,
        "prior": "Beta(.5,.5) rate; MDI(.6) restricted shape[-.49,.95]",
        "final_map": "shared GPD MLE finite-count expected maximum",
        "integration": "seeded Sobol Student-t importance quadrature",
        "maximum_deleted_support_handled_separately": True,
    }
    selected = {
        "selected_index": winner["index"],
        "threshold": winner["threshold"],
        "quantile": winner["quantile"],
        "raw_maximum": winner["raw_maximum"],
        "status": winner["map_status"],
        "diagnostics": diagnostics,
    }
    averaged = {
        "selected_index": None,
        "threshold": float(np.dot(weights, [r["threshold"] for r in resolved])),
        "quantile": float(np.dot(weights, [r["quantile"] for r in resolved])),
        "raw_maximum": float(np.dot(weights, [r["raw_maximum"] for r in resolved])),
        "status": "predictive_threshold_average",
        "diagnostics": diagnostics,
    }
    return {"predictive_select": selected, "predictive_average": averaged}
