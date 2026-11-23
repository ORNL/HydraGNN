"""
_evt_mixture.py — Two-component mixture bridge: Dirac(0) + Gumbel parent.

Naming convention: A grain is "plastic" if it has detectable per-grain
MAX FIP (m_g above a within-SVE threshold); "elastic" otherwise. The
within-SVE m_g population is modelled as

    m_g ~  q_elastic * delta(0) + (1 - q_elastic) * Gumbel(mu_g, beta_g)

with p_plastic = 1 - q_elastic the plastic-grain fraction.

Max of N i.i.d. samples from this mixture is asymptotically dominated by
the plastic subpopulation. With N_eff = N * p_plastic plastic samples in
expectation, and N_eff >> 1:

    E[M_S | mu_g, beta_g, N, q_elastic] = mu_g + beta_g * (ln(N_eff) + gamma_E)

This is the standard Gumbel-parent bridge with grain count replaced by
the *effective* count of plastic grains.

Two estimators of (mu_g, beta_g, p_plastic) per SVE are provided:
  - threshold_quantile  : split at a fixed quantile of m_g; truncated-MoM
                          on the above-threshold (plastic) subset
  - gmm_two_component   : fit a 2-component Gaussian mixture on m_g;
                          identify the higher-mean component as "plastic"

A third, physically grounded variant fits q from the data via
  p_plastic_data = fraction(m_g > mu_M + sigma_M)  →  q_elastic = 1 - p_plastic_data

The threshold variant is the cheap first pass; GMM is the more honest
bimodality detector. Both are tested.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA_E = 0.5772156649015329
SQRT6_OVER_PI = math.sqrt(6.0) / math.pi


def _gumbel_mom(mean: float, std: float) -> tuple[float, float]:
    if std <= 0:
        return float("nan"), float("nan")
    beta = std * SQRT6_OVER_PI
    mu = mean - GAMMA_E * beta
    return float(mu), float(beta)


def evt_mixture_threshold(mg: np.ndarray, N: int, q_elastic: float = 0.5) -> dict:
    """
    Threshold-based mixture bridge.

    Args:
        mg          per-grain MAX FIP sample for one SVE
        N           grain count
        q_elastic   quantile cutoff; m_g <= quantile(mg, q_elastic) is
                      treated as "elastic" (effectively zero contribution
                      to the max). Default 0.5 (median split).

    Returns dict with keys: p_elastic, mu_g, beta_g, N_eff, M_S_pred.
    """
    mg = np.asarray(mg, dtype=float)
    mg = mg[np.isfinite(mg)]
    if mg.size < 6:
        return {
            "p_elastic": float("nan"),
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    thresh = np.quantile(mg, q_elastic)
    active = mg[mg > thresh]
    if active.size < 4:
        return {
            "p_elastic": float("nan"),
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    p_elastic = 1.0 - active.size / mg.size
    mean_act = float(active.mean())
    std_act = float(active.std(ddof=1))
    mu_g, beta_g = _gumbel_mom(mean_act, std_act)
    if not np.isfinite(beta_g):
        return {
            "p_elastic": p_elastic,
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    N_eff = N * (1.0 - p_elastic)
    if N_eff < 2:
        return {
            "p_elastic": p_elastic,
            "mu_g": mu_g,
            "beta_g": beta_g,
            "N_eff": N_eff,
            "M_S_pred": float("nan"),
        }
    pred = mu_g + beta_g * (math.log(N_eff) + GAMMA_E)
    return {
        "p_elastic": p_elastic,
        "mu_g": mu_g,
        "beta_g": beta_g,
        "N_eff": N_eff,
        "M_S_pred": pred,
    }


def evt_mixture_fitted_q(mg: np.ndarray, N: int, method: str = "plastic_frac") -> dict:
    """
    Mixture bridge with q fit from data, not chosen as a hyperparameter.

    Naming convention:
      A grain is "plastic" if it has detectable per-grain MAX FIP above
      the within-SVE bulk; "elastic" otherwise. The within-SVE m_g
      population is modelled as q_elastic*delta(0) + (1-q_elastic)*Gumbel.
      The plastic fraction p_plastic = 1 - q_elastic.

    method = "plastic_frac": p_plastic = fraction(m_g > mu + sigma).
                              The physical plastic-grain-fraction
                              definition. Mu and sigma are computed on the
                              full m_g sample.
    method = "plastic_frac_pos": same but uses fraction(m_g > 0) — useful
                                   when many grains have exactly zero m_g.
    method = "median_split":   q_elastic = 0.5 (median split).

    Returns dict with q_fitted and predictor output.
    """
    mg = np.asarray(mg, dtype=float)
    mg = mg[np.isfinite(mg)]
    if mg.size < 6:
        return {
            "q_fitted": float("nan"),
            "p_elastic": float("nan"),
            "p_plastic": float("nan"),
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    mu_full = float(mg.mean())
    sd_full = float(mg.std(ddof=1))
    if method == "plastic_frac":
        if sd_full <= 0:
            p_plastic = 0.0
        else:
            p_plastic = float((mg > (mu_full + sd_full)).mean())
        q = max(0.0, min(0.99, 1.0 - p_plastic))
    elif method == "plastic_frac_pos":
        p_plastic = float((mg > 0).mean())
        q = max(0.0, min(0.99, 1.0 - p_plastic))
    else:
        q = 0.5
        p_plastic = float("nan")
    # Threshold the data at the q-quantile and apply the mixture bridge.
    thresh = np.quantile(mg, q)
    plastic = mg[mg > thresh]
    if plastic.size < 4:
        return {
            "q_fitted": q,
            "p_elastic": q,
            "p_plastic": 1.0 - q,
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    p_elastic = 1.0 - plastic.size / mg.size
    p_plastic_emp = plastic.size / mg.size
    mu_plas = float(plastic.mean())
    sd_plas = float(plastic.std(ddof=1))
    mu_g, beta_g = _gumbel_mom(mu_plas, sd_plas)
    if not np.isfinite(beta_g):
        return {
            "q_fitted": q,
            "p_elastic": p_elastic,
            "p_plastic": p_plastic_emp,
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    N_eff = N * p_plastic_emp
    if N_eff < 2:
        return {
            "q_fitted": q,
            "p_elastic": p_elastic,
            "p_plastic": p_plastic_emp,
            "mu_g": mu_g,
            "beta_g": beta_g,
            "N_eff": N_eff,
            "M_S_pred": float("nan"),
        }
    pred = mu_g + beta_g * (math.log(N_eff) + GAMMA_E)
    return {
        "q_fitted": q,
        "p_elastic": p_elastic,
        "p_plastic": p_plastic_emp,
        "mu_g": mu_g,
        "beta_g": beta_g,
        "N_eff": N_eff,
        "M_S_pred": pred,
    }


def evt_mixture_gmm(mg: np.ndarray, N: int) -> dict:
    """
    GMM-based mixture bridge. Fit a 2-component Gaussian mixture on m_g
    and use the higher-mean component as the plastic subpopulation.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return evt_mixture_threshold(mg, N, q_elastic=0.5)
    mg = np.asarray(mg, dtype=float)
    mg = mg[np.isfinite(mg)]
    if mg.size < 12:
        return evt_mixture_threshold(mg, N, q_elastic=0.5)
    try:
        gmm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=0, max_iter=200
        )
        gmm.fit(mg.reshape(-1, 1))
    except Exception:
        return evt_mixture_threshold(mg, N, q_elastic=0.5)
    means = gmm.means_.ravel()
    weights = gmm.weights_.ravel()
    covs = gmm.covariances_.ravel()
    hi = int(np.argmax(means))  # plastic component = higher mean
    lo = 1 - hi
    p_plastic = float(weights[hi])
    p_elastic = float(weights[lo])
    mu_plastic = float(means[hi])
    sigma_plastic = float(math.sqrt(max(covs[hi], 0.0)))
    mu_g, beta_g = _gumbel_mom(mu_plastic, sigma_plastic)
    if not np.isfinite(beta_g):
        return {
            "p_elastic": p_elastic,
            "mu_g": float("nan"),
            "beta_g": float("nan"),
            "N_eff": float("nan"),
            "M_S_pred": float("nan"),
        }
    N_eff = N * p_plastic
    if N_eff < 2:
        return {
            "p_elastic": p_elastic,
            "mu_g": mu_g,
            "beta_g": beta_g,
            "N_eff": N_eff,
            "M_S_pred": float("nan"),
        }
    pred = mu_g + beta_g * (math.log(N_eff) + GAMMA_E)
    return {
        "p_elastic": p_elastic,
        "mu_g": mu_g,
        "beta_g": beta_g,
        "N_eff": N_eff,
        "M_S_pred": pred,
    }


def r2(y: np.ndarray, yh: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    yh = np.asarray(yh, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    if y.size < 2:
        return float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - ((y - yh) ** 2).sum() / sst)


def rmse(y, yh):
    y, yh = np.asarray(y, float).ravel(), np.asarray(yh, float).ravel()
    m = np.isfinite(y) & np.isfinite(yh)
    if m.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yh[m]) ** 2)))


def mae(y, yh):
    y, yh = np.asarray(y, float).ravel(), np.asarray(yh, float).ravel()
    m = np.isfinite(y) & np.isfinite(yh)
    if m.sum() < 2:
        return float("nan")
    return float(np.mean(np.abs(y[m] - yh[m])))


def mape_pct(y, yh):
    y, yh = np.asarray(y, float).ravel(), np.asarray(yh, float).ravel()
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    if y.size < 2:
        return float("nan")
    floor = 1e-12 * max(1.0, float(np.max(np.abs(y))))
    msk = np.abs(y) > floor
    if msk.sum() < 2:
        return float("nan")
    return float(100.0 * np.mean(np.abs((y[msk] - yh[msk]) / y[msk])))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Synthetic test: mixture with p_elastic = 0.5, plastic Gumbel(0.003, 0.0013)
    n = 300
    elastic_mass = rng.uniform(0, 1e-5, size=n // 2)  # near-zero
    plastic_mass = rng.gumbel(loc=0.003, scale=0.0013, size=n // 2)
    mg = np.concatenate([elastic_mass, plastic_mass])
    rng.shuffle(mg)
    print(
        f"synthetic mixture: n={n}, true p_elastic=0.5, "
        f"true (mu_g, beta_g) = (0.003, 0.0013)"
    )
    print(f"  empirical max = {mg.max():.4e}")
    out_thr = evt_mixture_threshold(mg, N=n, q_elastic=0.5)
    print(
        f"  threshold(q=0.5): p_e={out_thr['p_elastic']:.3f}  "
        f"mu_g={out_thr['mu_g']:.4e}  beta_g={out_thr['beta_g']:.4e}  "
        f"N_eff={out_thr['N_eff']:.0f}  pred={out_thr['M_S_pred']:.4e}"
    )
    out_gmm = evt_mixture_gmm(mg, N=n)
    print(
        f"  GMM:               p_e={out_gmm['p_elastic']:.3f}  "
        f"mu_g={out_gmm['mu_g']:.4e}  beta_g={out_gmm['beta_g']:.4e}  "
        f"N_eff={out_gmm['N_eff']:.0f}  pred={out_gmm['M_S_pred']:.4e}"
    )
