"""Clean PyG dataset wrapper for examples/cp3d_v2/full_data_v2.pt.

Responsibilities:
  - Load the v2 .pt produced by data/build_dataset.py.
  - Optionally filter to a single SVE-size bucket.
  - Build deterministic seeded train/val/test splits via stratified random
    shuffle (or plain shuffle when filtered to one bucket).
  - Apply per-split feature/target scaling (StandardScaler fitted on TRAIN only).
  - Expose the result as a list[Data] (HydraGNN's expected format).

Design rules:
  - No mutation of the input .pt.
  - Splits saved to results/{tag}_seed{seed}/split_indices.json for reproducibility.
  - Target transform (log1p / identity / sqrt) is a parameter, not assumed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

HERE = Path(__file__).parent
# Canonical location: examples/data/full_data_v3.pt. Walk up from this file
# (.../examples/.../code/dataset.py) to the first ancestor named "examples"
# and look for data/full_data_v3.pt under it. Works for both the local
# (examples/fip_unified/isaac_v3_study/code) and cluster
# (examples/isaac_v3_study/code) layouts.
def _find_dataset_pt() -> Path:
    candidates = []
    # 1) study-local data dir (back-compat: study_root/data/full_data_v3.pt;
    #    typically a symlink to examples/data/full_data_v3.pt).
    candidates.append(HERE.parent / "data" / "full_data_v3.pt")
    # 2) walk up to the first "examples/" ancestor, then "examples/data/...".
    for p in HERE.parents:
        if p.name == "examples":
            candidates.append(p / "data" / "full_data_v3.pt")
            break
    # 3) backwards-compat fallback to v2 next to dataset.py.
    candidates.append(HERE / "full_data_v2.pt")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DEFAULT_PT = _find_dataset_pt()

EDGE_ATTR_NAMES = [
    "misorientation_angle",
    "edge_length",
    "relpos_crystal_x",
    "relpos_crystal_y",
    "relpos_crystal_z",
    "slip_transfer_mprime",
    "elastic_mismatch",
    "is_periodic",
]
EDGE_ATTR_INDEX = {n: i for i, n in enumerate(EDGE_ATTR_NAMES)}


# Physics-aware edge feature transforms (2026-05-04 PM, Phase 3 of PNA-v4 study).
# Each transform takes the RAW 8-d edge_attr tensor (N_edges, 8) and returns a
# (N_edges, 1) column. Names from this registry are valid in `use_edge_cols`
# alongside raw EDGE_ATTR_NAMES.
#
# Rationale:
# - inv_edge_length: shortest distance dominates fracture-style problems; the
#   linear distance can't express the proximity prior cleanly.
# - edge_length_rbf_2um: smooth Gaussian falloff with characteristic 2-um
#   scale (typical grain neighbor spacing).
# - cos_misor / sin_misor: separates low-angle GB (cos≈1) from high-angle
#   GB (sin large) physics; orthogonal pair.
def _xform_inv_edge_length(ea: torch.Tensor) -> torch.Tensor:
    return (1.0 / ea[:, EDGE_ATTR_INDEX["edge_length"]].clamp(min=1e-6)).unsqueeze(-1)


def _xform_edge_length_rbf(scale_um: float):
    def f(ea: torch.Tensor) -> torch.Tensor:
        return torch.exp(-ea[:, EDGE_ATTR_INDEX["edge_length"]] / scale_um).unsqueeze(
            -1
        )

    return f


def _xform_cos_misor(ea: torch.Tensor) -> torch.Tensor:
    return torch.cos(ea[:, EDGE_ATTR_INDEX["misorientation_angle"]]).unsqueeze(-1)


def _xform_sin_misor(ea: torch.Tensor) -> torch.Tensor:
    return torch.sin(ea[:, EDGE_ATTR_INDEX["misorientation_angle"]]).unsqueeze(-1)


EDGE_XFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "inv_edge_length": _xform_inv_edge_length,
    "edge_length_rbf_2um": _xform_edge_length_rbf(2.0),
    "edge_length_rbf_5um": _xform_edge_length_rbf(5.0),
    "cos_misor": _xform_cos_misor,
    "sin_misor": _xform_sin_misor,
}

# x[N, 22] layout from build_dataset.py:create_pyg_data
X_COL_LAYOUT = {
    "euler": slice(0, 3),
    "quat": slice(3, 7),
    "schmid": slice(7, 8),
    "volume": slice(8, 9),
    "surface_area": slice(9, 10),
    "perimeter": slice(10, 11),
    "degree": slice(11, 12),
    "Iuu": slice(12, 13),
    "Ivv": slice(13, 14),
    "Iuv": slice(14, 15),
    "J_polar": slice(15, 16),
    "I1": slice(16, 17),
    "I2": slice(17, 18),
    "theta": slice(18, 19),
    "Ixx": slice(19, 20),
    "Iyy": slice(20, 21),
    "Ixy": slice(21, 22),
    # New for v3.pt: per-grain voxel-derived geometric features (cols 22..27).
    # Only present in full_data_v3.pt (built by examples/fip_unified/data/build_voxel_dataset.py).
    # Loading v2.pt with "voxel_geom" in use_x_blocks will fail at _select_x.
    "voxel_geom": slice(22, 28),
    # Computed at load time from data.x[:, 0:3] (Bunge Euler radians) when
    # "schmid12" appears in use_x_blocks. 12 sorted-descending Schmid factors
    # per grain. Subsumes "schmid" (col 7 = the max == schmid12[0]).
    # Doubled per-grain R² on cp3d_v2 in the V100 sweep (memory: cp3d_v2_v100_sweep_findings_2026-04-28.md).
    "schmid12": slice(28, 40),
    # Phase 4 PNA-v4 derived blocks (2026-05-04 PM). Each is computed at
    # load time from raw cols. Standardised order: log_volume first, then
    # inv_volume, surface_to_vol, schmid_asymmetry, schmid_std.
    "log_volume": slice(
        40, 41
    ),  # log(volume) — fatigue scales log-linearly with grain size
    "inv_volume": slice(41, 42),  # 1/volume — small-grain stress concentration
    "surface_to_vol": slice(
        42, 43
    ),  # surface_area / volume^(2/3) — normalised boundary fraction
    "schmid_asymmetry": slice(43, 44),  # max(schmid12) - mean(schmid12)
    "schmid_std": slice(44, 45),  # std(schmid12)
    # Phase 5 PNA-v4 cross-variant blocks (read from data.{lcf_wp,hcf_fs,hcf_wp}_fip_dist
    # tensors set by examples/cpfem_fip_data builder; 10 cols per variant).
    # When any of these blocks is in use_x_blocks, _load() drops graphs
    # with NaN in the corresponding tensor (per user 2026-05-04 PM:
    # no imputation, drop entirely).
    "lcf_wp_fip_dist": slice(45, 55),
    "hcf_fs_fip_dist": slice(55, 65),
    "hcf_wp_fip_dist": slice(65, 75),
    # v9 — binary per-grain flag read from data.is_sve_boundary (exact, set
    # by the cpfem_fip_data v5 builder from voxel feature_ids; 1 iff at least
    # one voxel of the grain lies on one of the 6 outer faces of the SVE
    # cube). Requires full_data_v5.pt or later; not available in v4.pt
    # (which has no is_sve_boundary attribute). _load() raises if requested
    # against a pre-v5 .pt.
    "sve_boundary": slice(75, 76),
}


# Phase 4 derived block computers — each takes the current d.x and returns
# the new (N, k) tensor to be appended. Read from earlier columns that
# the caller guarantees are present (schmid12 must precede asymmetry/std).
def _compute_log_volume(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x[:, 8:9].clamp(min=1e-30))


def _compute_inv_volume(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / x[:, 8:9].clamp(min=1e-30)


def _compute_surface_to_vol(x: torch.Tensor) -> torch.Tensor:
    return x[:, 9:10] / x[:, 8:9].clamp(min=1e-30).pow(2.0 / 3.0)


def _compute_schmid_asymmetry(x: torch.Tensor) -> torch.Tensor:
    s12 = x[:, 28:40]  # caller guarantees schmid12 is computed
    return (s12.max(dim=1).values - s12.mean(dim=1)).unsqueeze(-1)


def _compute_schmid_std(x: torch.Tensor) -> torch.Tensor:
    s12 = x[:, 28:40]
    return s12.std(dim=1).unsqueeze(-1)


# Order matters: schmid12 must be computed BEFORE schmid_asymmetry/std.
DERIVED_BLOCKS_ORDER = [
    "schmid12",  # cols 28:40 — already handled separately in _load
    "log_volume",  # col 40
    "inv_volume",  # col 41
    "surface_to_vol",  # col 42
    "schmid_asymmetry",  # col 43 (depends on schmid12)
    "schmid_std",  # col 44 (depends on schmid12)
]
DERIVED_COMPUTERS = {
    "log_volume": _compute_log_volume,
    "inv_volume": _compute_inv_volume,
    "surface_to_vol": _compute_surface_to_vol,
    "schmid_asymmetry": _compute_schmid_asymmetry,
    "schmid_std": _compute_schmid_std,
}


# 12 FCC slip systems (4 unique {111} planes × 3 <110> directions). Pre-normalized
# so |n| = |b| = 1 → Schmid = |n·s| · |b·s|. This table matches the orthogonal
# version in examples/fip-gnn-sim/unified/schmid.py (audited at import there).
_FCC_SLIP_N = (
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, -1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, -1],
            [1, 1, -1],
        ],
        dtype=np.float64,
    )
    / np.sqrt(3.0)
)
_FCC_SLIP_B = (
    np.array(
        [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
            [0, -1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, -1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [-1, 1, 0],
        ],
        dtype=np.float64,
    )
    / np.sqrt(2.0)
)


def _schmid12_from_eulers(
    eul_rad: np.ndarray, load_sample=(1.0, 0.0, 0.0)
) -> np.ndarray:
    """Vectorized 12-sorted-Schmid computation from Bunge ZXZ Euler radians.

    Mirrors `fcc_schmid12_batch` in examples/fip-gnn-sim/unified/schmid.py;
    inlined here to keep the v3 study self-contained. Inputs in CORRECT
    radians (not double-deg2rad'd — see CRITICAL_dataset_orientation_bugs.md).
    """
    phi1, Phi, phi2 = eul_rad[:, 0], eul_rad[:, 1], eul_rad[:, 2]
    c1, c2, cP = np.cos(phi1), np.cos(phi2), np.cos(Phi)
    s1, s2, sP = np.sin(phi1), np.sin(phi2), np.sin(Phi)
    R = np.empty((eul_rad.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = c1 * c2 - s1 * s2 * cP
    R[:, 0, 1] = -c1 * s2 - s1 * c2 * cP
    R[:, 0, 2] = s1 * sP
    R[:, 1, 0] = s1 * c2 + c1 * s2 * cP
    R[:, 1, 1] = -s1 * s2 + c1 * c2 * cP
    R[:, 1, 2] = -c1 * sP
    R[:, 2, 0] = s2 * sP
    R[:, 2, 1] = c2 * sP
    R[:, 2, 2] = cP
    s = np.asarray(load_sample, dtype=np.float64)
    s = s / (np.linalg.norm(s) + 1e-12)
    n_s = np.einsum("nij,kj->nki", R, _FCC_SLIP_N)
    b_s = np.einsum("nij,kj->nki", R, _FCC_SLIP_B)
    m = np.abs(n_s @ s) * np.abs(b_s @ s)
    return np.sort(m, axis=1)[:, ::-1]


# =============================================================================
# Target transforms
# =============================================================================
# In addition to the original 4 (identity / log1p / log / sqrt), the v3 study
# adds 6 new transforms motivated by the empirical analysis in
# `examples/fip_unified/09_normalization_deep_dive.md`. Stateful transforms
# (zscore, zscore_per_bucket, yeo_johnson, quantile_normal, gumbel_*) need
# fit state cached at training time and reused at eval time. The state is
# carried on the FIPDataset instance via `_target_fit_state`.

_STATELESS_TRANSFORMS = ("identity", "log1p", "log", "sqrt", "gumbel_paper")


def _apply_target_transform(
    y: np.ndarray,
    name: str,
    *,
    fit_state: dict | None = None,
    bucket_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Forward (fit-time-or-eval-time) target transform.

    For stateful transforms (zscore, yeo_johnson, quantile_normal, gumbel_*
    except gumbel_paper), pass `fit_state`:
      - if `fit_state` is empty/None and we're at training time → fit, mutate fit_state.
      - if `fit_state` already populated → apply previously-fit parameters.

    `bucket_ids` (same length as y) is required for `zscore_per_bucket`.
    `gumbel_paper` is computed from the empirical CDF on each call; it is
    stateless but rank-dependent (pure diagnostic; not recommended for
    inference because the inverse cannot recover an unseen rank).
    """
    if name == "identity":
        return y
    if name == "log1p":
        return np.log1p(y)
    if name == "log":
        return np.log(np.maximum(y, 1e-12))
    if name == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))

    if name == "zscore":
        if fit_state is None:
            raise ValueError("zscore requires a fit_state dict")
        if "mean" not in fit_state:
            fit_state["mean"] = float(y.mean())
            fit_state["std"] = float(y.std()) or 1.0
        return (y - fit_state["mean"]) / fit_state["std"]

    if name == "zscore_per_bucket":
        if fit_state is None or bucket_ids is None:
            raise ValueError("zscore_per_bucket requires fit_state and bucket_ids")
        if "per_bucket" not in fit_state:
            per_b = {}
            for b in np.unique(bucket_ids):
                m = bucket_ids == b
                per_b[int(b)] = (float(y[m].mean()), float(y[m].std()) or 1.0)
            fit_state["per_bucket"] = per_b
        out = np.empty_like(y, dtype=np.float64)
        for b in np.unique(bucket_ids):
            m = bucket_ids == b
            mean, std = fit_state["per_bucket"].get(
                int(b), (float(y[m].mean()), float(y[m].std()) or 1.0)
            )
            out[m] = (y[m] - mean) / std
        return out

    if name == "yeo_johnson":
        from sklearn.preprocessing import PowerTransformer

        if fit_state is None:
            raise ValueError("yeo_johnson requires a fit_state dict")
        if "power_transformer" not in fit_state:
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            pt.fit(y.reshape(-1, 1))
            fit_state["power_transformer"] = pt
            # Save the train-side mean for a fallback if the inverse NaNs.
            fit_state["y_mean"] = float(np.nanmean(y))
            # Auto-detect the asymmetric safe inverse range. The fitted
            # lambda can drive the inverse to NaN at moderate |z| (esp. on
            # the upper side for positive-skew data). Bisect outward from
            # 0 in each direction until inverse hits NaN; record the last
            # finite z.  Used by `_invert_target_transform` to clip
            # predictions before inverse so we never produce NaN AND we
            # don't pile up at a too-tight ±2 boundary that creates the
            # vertical-line bands on parity plots.
            def _safe_z(direction: int, hard_cap: float = 8.0) -> float:
                lo, hi = 0.0, hard_cap if direction > 0 else hard_cap
                # binary search for first NaN
                for _ in range(40):
                    mid = (lo + hi) / 2
                    z = direction * mid
                    out = pt.inverse_transform(np.array([[z]]))[0, 0]
                    if np.isfinite(out):
                        lo = mid
                    else:
                        hi = mid
                return float(direction * lo * 0.999)  # 0.1% safety margin

            fit_state["z_min_safe"] = _safe_z(-1)
            fit_state["z_max_safe"] = _safe_z(+1)
        return fit_state["power_transformer"].transform(y.reshape(-1, 1))[:, 0]

    if name == "quantile_normal":
        from sklearn.preprocessing import QuantileTransformer

        if fit_state is None:
            raise ValueError("quantile_normal requires a fit_state dict")
        if "quantile_transformer" not in fit_state:
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(10000, max(10, len(y))),
                subsample=200_000,
            )
            qt.fit(y.reshape(-1, 1))
            fit_state["quantile_transformer"] = qt
        return fit_state["quantile_transformer"].transform(y.reshape(-1, 1))[:, 0]

    if name == "gumbel_paper":
        # Empirical -ln(-ln((rank-0.5)/N)). Stateless / rank-only.
        # Diagnostic transform — see `09_normalization_deep_dive.md`. Inverse is
        # NOT well-defined for unseen ranks; do not use for production.
        from scipy.stats import rankdata

        ranks = rankdata(y)
        F = (ranks - 0.5) / max(len(y), 1)
        F = np.clip(F, 1e-9, 1 - 1e-9)
        return -np.log(-np.log(F))

    if name == "gumbel_uniform":
        # y → F_Gumbel(y) ∈ (0,1). Fit μ, β on training data.
        from scipy.stats import gumbel_r

        if fit_state is None:
            raise ValueError("gumbel_uniform requires a fit_state dict")
        if "gumbel_params" not in fit_state:
            fit_state["gumbel_params"] = tuple(float(p) for p in gumbel_r.fit(y))
        mu, beta = fit_state["gumbel_params"]
        return gumbel_r.cdf(y, loc=mu, scale=beta)

    if name == "gumbel_probit":
        # y → Φ⁻¹(F_Gumbel(y)). Fit μ, β on training data.
        from scipy.stats import gumbel_r, norm

        if fit_state is None:
            raise ValueError("gumbel_probit requires a fit_state dict")
        if "gumbel_params" not in fit_state:
            fit_state["gumbel_params"] = tuple(float(p) for p in gumbel_r.fit(y))
        mu, beta = fit_state["gumbel_params"]
        F = np.clip(gumbel_r.cdf(y, loc=mu, scale=beta), 1e-9, 1 - 1e-9)
        return norm.ppf(F)

    raise ValueError(f"Unknown target transform: {name}")


def _invert_target_transform(
    y: np.ndarray,
    name: str,
    *,
    fit_state: dict | None = None,
    bucket_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of `_apply_target_transform`.

    Note: `gumbel_paper` does not have a stable inverse for unseen samples
    (rank-based, no parametric form). Use it only as a diagnostic on the
    training distribution; for eval/test inverse use `gumbel_uniform` or
    `gumbel_probit` which are parametric and invertible.
    """
    if name == "identity":
        return y
    if name == "log1p":
        return np.expm1(y)
    if name == "log":
        return np.exp(y)
    if name == "sqrt":
        return np.maximum(y, 0.0) ** 2

    if name == "zscore":
        s = fit_state or {}
        return y * s.get("std", 1.0) + s.get("mean", 0.0)

    if name == "zscore_per_bucket":
        if fit_state is None or bucket_ids is None:
            raise ValueError(
                "zscore_per_bucket inverse requires fit_state + bucket_ids"
            )
        out = np.empty_like(y, dtype=np.float64)
        for b in np.unique(bucket_ids):
            m = bucket_ids == b
            mean, std = fit_state["per_bucket"].get(int(b), (0.0, 1.0))
            out[m] = y[m] * std + mean
        return out

    if name == "yeo_johnson":
        if fit_state is None or "power_transformer" not in fit_state:
            raise ValueError("yeo_johnson inverse requires fitted power_transformer")
        # Clip predicted z-scores to the asymmetric safe range determined
        # at fit time. The fitted Yeo–Johnson lambda can produce NaN at
        # moderate |z| (esp. on the upper side for positive-skew data),
        # so we precompute z_min_safe / z_max_safe via bisection and clip
        # there rather than at a fixed ±2 (which created visible vertical-
        # band pile-ups on parity plots — see fs_within_mean test plot).
        z_lo = float(fit_state.get("z_min_safe", -2.0))
        z_hi = float(fit_state.get("z_max_safe", +2.0))
        y_clipped = np.clip(y, z_lo, z_hi)
        out = fit_state["power_transformer"].inverse_transform(
            y_clipped.reshape(-1, 1)
        )[:, 0]
        if not np.all(np.isfinite(out)):
            mean_y = float(fit_state.get("y_mean", np.nan))
            if not np.isnan(mean_y):
                out = np.where(np.isfinite(out), out, mean_y)
        return out

    if name == "quantile_normal":
        if fit_state is None or "quantile_transformer" not in fit_state:
            raise ValueError(
                "quantile_normal inverse requires fitted quantile_transformer"
            )
        return fit_state["quantile_transformer"].inverse_transform(y.reshape(-1, 1))[
            :, 0
        ]

    if name == "gumbel_paper":
        # Approximate inverse via the gumbel_paper relation:
        # η = -ln(-ln(F))  →  F = exp(-exp(-η))
        # On unseen samples we cannot recover original y from F without the
        # empirical distribution. Return F as a placeholder and warn.
        import warnings

        warnings.warn(
            "gumbel_paper inverse is not well-defined; returning F = exp(-exp(-y))"
        )
        return np.exp(-np.exp(-y))

    if name == "gumbel_uniform":
        from scipy.stats import gumbel_r

        if fit_state is None or "gumbel_params" not in fit_state:
            raise ValueError("gumbel_uniform inverse requires fitted gumbel_params")
        mu, beta = fit_state["gumbel_params"]
        F = np.clip(y, 1e-9, 1 - 1e-9)
        return gumbel_r.ppf(F, loc=mu, scale=beta)

    if name == "gumbel_probit":
        from scipy.stats import gumbel_r, norm

        if fit_state is None or "gumbel_params" not in fit_state:
            raise ValueError("gumbel_probit inverse requires fitted gumbel_params")
        mu, beta = fit_state["gumbel_params"]
        F = np.clip(norm.cdf(y), 1e-9, 1 - 1e-9)
        return gumbel_r.ppf(F, loc=mu, scale=beta)

    raise ValueError(f"Unknown target transform: {name}")


@dataclass
class FIPSplits:
    train: List[int]
    val: List[int]
    test: List[int]


@dataclass
class FIPDataset:
    """Wraps full_data_v2.pt with split + scale.

    All knobs that control DATA processing live here. Model-side knobs (mpnn,
    hidden_dim, ...) are in train.py.
    """

    pt_path: str = str(DEFAULT_PT)
    size_bucket: Optional[int] = None  # None = use all buckets; else 0..4
    seed: int = 42  # used only for SPLIT randomness (train/val/test).
    # Model init seed is set in train.py; pass `split_seed`
    # in train.py to make these decoupled.
    split_fracs: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    stratify_by_bucket: bool = True  # global mode: stratify by SVE bucket
    # Extrapolation mode: train on these buckets, test on others. When set,
    # `split_fracs` and stratify_by_bucket are ignored. Val takes 5% from
    # train_buckets (random split within those graphs); test = all graphs in
    # the buckets NOT in train_buckets. The standard GNN extrapolation test
    # for size generalization.
    train_buckets: Optional[List[int]] = None
    test_buckets: Optional[List[int]] = None

    # graph_attr conditioning. None = no graph_attr emitted; otherwise pick a
    # source column. Standardized on TRAIN.
    #   "L_SVE_um"          : 1-d physical size in microns
    #   "log_n_nodes"       : log(grain count) — directly targets Anto LS scaling
    #   "size_bucket_onehot": 5-d one-hot of bucket index
    graph_attr_source: Optional[str] = None

    # Target residualization — subtract a per-graph baseline from the target
    # BEFORE scaling. Cross-bucket variance is removed; the model trains only
    # on within-bucket residuals. At eval time the baseline is added back.
    # This is the Anto/heavyreg "B5 add-back" approach — exposes whether the
    # model has any within-bucket signal beyond the per-bucket / LS mean.
    #   None              : no residualization (default)
    #   "per_bucket_mean" : subtract per-bucket mean (5 floors)
    #   "log_n_ols"       : Anto Eq.13: fit y = a + b·log(N) on TRAIN, subtract prediction
    target_residual_baseline: Optional[str] = None

    # Per-graph target aggregator. The default "max" (per-grain max FIP) is the
    # extreme-value target; it has the most Gumbel noise and is the hardest to
    # learn within-bucket. Calmer alternatives reduce noise and may be easier
    # for the model to fit.
    #   "max"   : np.max(per_grain_fip)        — extreme value (default)
    #   "P99"   : 99th percentile              — less extreme
    #   "P95"   : 95th percentile              — even less
    #   "P90"   : 90th percentile              — robust
    #   "mean"  : np.mean(per_grain_fip)       — averages out Gumbel noise
    per_graph_target_aggr: str = "max"

    # Feature selection (defaults match Phase-1 floor: orientation + Schmid + 9 geom).
    use_x_blocks: List[str] = field(
        default_factory=lambda: [
            "quat",
            "schmid",
            "volume",
            "surface_area",
            "perimeter",
            "degree",
            "Iuu",
            "Ivv",
            "Iuv",
            "J_polar",
            "I1",
            "I2",
        ]
    )
    # Edge-attr columns to keep (None = all 8). Filter periodic edges via
    # `drop_periodic_edges=True`.
    use_edge_cols: Optional[List[str]] = field(
        default_factory=lambda: [
            "misorientation_angle",
            "edge_length",
            "relpos_crystal_x",
            "relpos_crystal_y",
            "relpos_crystal_z",
            "is_periodic",
        ]
    )
    drop_periodic_edges: bool = False

    # Targets
    use_per_grain_target: bool = True  # FS_FIP_COLS=[0]
    use_per_graph_target: bool = True  # GRAPH_FIP_TARGETS=["max"]
    per_grain_transform: str = "log1p"  # Phase-1 ablation knob
    per_graph_transform: str = "log1p"
    # Which per-grain column to predict.  fs_fip is [N, 2] where:
    #   col 0 = per-grain MAX FIP (the physically correct target — fatigue
    #           crack initiates at the worst point in each grain)
    #   col 1 = per-grain MEAN FIP (Sim et al. 2024's target — physically
    #           wrong but useful for direct comparison)
    per_grain_target_col: int = 0
    # Which Data attribute to read the per-grain target from. Defaults to
    # "fs_fip" (Fatemi-Socie); set "wp_fip" to predict the Walker-Phelps
    # FIP variant from the same dataset.
    per_grain_target_field: str = "fs_fip"

    # Per-graph target source override. Default "default" → use the existing
    # FIP-aggregate path (per_grain_target_field aggregated by per_graph_target_aggr).
    # Set to one of the mech-QoI keys below to predict a per-SVE mechanical
    # response instead of FIP at the per-graph head:
    #   "yield_stress"        : monotonic_plastic[2]   (~yield stress)
    #   "hardening_modulus"   : monotonic_plastic[3]
    #   "cyc_plastic_strain"  : cyclic_plastic[1:][2]  (Coffin-Manson governing)
    #   "cyc_stress_amp"      : cyclic_plastic[1:][3]  (Basquin governing)
    #   "aniso_E"             : std/mean of {E_xx, E_yy, E_zz} from monotonic_elastic[13,20,27]
    #   "aniso_G"             : std/mean of {G_xy, G_xz, G_yz} from monotonic_elastic[34,41,48]
    per_graph_target_source: str = "default"

    # Auxiliary per-graph targets for multi-task learning.
    # Each dict: {"name": str, "source": str, "transform": str}.
    # Each `source` must be a value accepted by `_pergraph_target_raw` (e.g.
    # "yield_stress", "cyc_plastic_strain", ..., or "default" to read another
    # FIP aggregate). Each gets its own per-target transform fit-state and
    # StandardScaler. Output: data.y_pergraph_aux_raw is a (n_aux,) tensor in
    # raw units; the per-graph head emits (1 + n_aux) outputs.
    aux_pergraph_targets: List[Dict[str, str]] = field(default_factory=list)

    # Auxiliary per-grain (per-node) targets — multi-task at the node level.
    # Each dict: {"name": str, "field": str, "col": int, "transform": str, "scale"?: float}.
    # Reads from `getattr(d, field)[:, col]` (e.g. fs_fip[:, 1] = per-grain MEAN,
    # lcf_fs_fip_dist[:, 4] = within-grain p99, hcf_fs_fip_dist[:, 0] = HCF/FS
    # per-grain MAX). Per-grain head output_dim becomes 1 + n_per_grain_aux;
    # flat y is row-major (per-node, per-channel). NaN cells in source are
    # zero-filled at load BUT the original NaN positions are tracked on each
    # Data object as `per_grain_aux_nan_mask` (N_grains, n_aux) so eval +
    # plotting can exclude them and avoid zero-fill artefacts.
    per_grain_aux_targets: List[Dict[str, Any]] = field(default_factory=list)

    # v9b deslot (2026-05-06): unified symmetric per-grain target list.
    # When non-empty, supersedes the legacy `per_grain_target_field` +
    # `per_grain_aux_targets` split (which is then auto-derived in
    # `__post_init__` from the unified list — slot 0 = element 0).
    # Each entry: {"name": str, "field": str, "col": int, "transform": str, "scale"?: float}.
    # The 8 entries are first-class equals — there is no "primary" slot 0 in
    # the user-facing API. The slot-0 plumbing is kept internally because
    # HydraGNN's head architecture is one head per K channels with a single
    # primary read; this is hidden from the recipe.
    per_grain_targets: List[Dict[str, Any]] = field(default_factory=list)

    # v9b deslot: unified symmetric per-graph target list. Same convention
    # as `per_grain_targets`. Each entry: {"name": str, "source": str, "transform": str}.
    # When non-empty, supersedes `per_graph_target_source` + `aux_pergraph_targets`.
    pergraph_targets: List[Dict[str, Any]] = field(default_factory=list)

    # Phase L5 heteroscedastic head (per-grain): when True, dataset y is
    # padded with 4 zero columns per grain (one per cell) to match the
    # head's 4 σ output channels. σ slots are unsupervised — `het_nll_*` loss.
    use_heteroscedastic_max: bool = False
    # Phase L6 heteroscedastic head (per-graph): same idea at SVE level.
    # When True, pg y is padded with 4 zero columns per graph for the σ
    # slots paired with the 4 SVE-MAX channels at slots 0..3.
    use_heteroscedastic_pergraph_max: bool = False

    # Scalers (fitted at .build())
    _scaler_x: Optional[StandardScaler] = None
    _scaler_pergrain_y: Optional[StandardScaler] = None
    _scaler_pergraph_y: Optional[StandardScaler] = None
    _scaler_graph_attr: Optional[StandardScaler] = None
    _scaler_edge_attr: Optional[StandardScaler] = None
    _graph_attr_dim: int = 0
    # Per-aux-target machinery (parallel lists; index i corresponds to aux_pergraph_targets[i]).
    _scaler_aux_pergraph_y: Optional[List[StandardScaler]] = None
    _aux_pergraph_target_states: Optional[List[Dict]] = None
    # Mirror lists for per-grain aux targets.
    _scaler_per_grain_aux_y: Optional[List[StandardScaler]] = None
    _per_grain_aux_target_states: Optional[List[Dict]] = None
    # Whether to standardize edge_attr columns (zscore per column, fit on train).
    normalize_edges: bool = True

    # Residualization state (fit on train; applied identically across splits).
    _residual_per_bucket_mean: Optional[
        Dict[int, float]
    ] = None  # in *transformed* per_graph_y space
    _residual_log_n_coef: Optional[
        Tuple[float, float]
    ] = None  # (a, b) in *transformed* per_graph_y space

    splits: Optional[FIPSplits] = None

    def __post_init__(self):
        self._raw: Optional[List[Data]] = None
        self._processed: Optional[List[Data]] = None
        # v9b deslot translation: if the user passed `per_grain_targets`, derive
        # the legacy slot-0 + aux split from it. The slot-0 mechanism is a
        # HydraGNN implementation detail; the recipe-facing API is symmetric.
        if self.per_grain_targets:
            head = self.per_grain_targets[0]
            self.per_grain_target_field = str(
                head.get("field", self.per_grain_target_field)
            )
            self.per_grain_target_col = int(head.get("col", self.per_grain_target_col))
            # Slots 1.. become the aux list. Custom transform per spec.
            self.per_grain_aux_targets = [dict(s) for s in self.per_grain_targets[1:]]
            # Adopt slot-0's transform as the primary transform (so scalers fit
            # the same way as the symmetric API expects).
            tf = head.get("transform")
            if tf is not None:
                self.per_grain_transform = str(tf)
        if self.pergraph_targets:
            head = self.pergraph_targets[0]
            # Slot 0 of the per-graph head reads from
            # `per_graph_target_source` (default: aggregate the per-grain
            # target via `per_graph_target_aggr`). For the deslot we expose
            # slot-0 as a named pg source instead.
            self.per_graph_target_source = str(
                head.get("source", self.per_graph_target_source)
            )
            # Slot-0 transform applies to the pg primary head.
            tf = head.get("transform")
            if tf is not None:
                self.per_graph_transform = str(tf)
            self.aux_pergraph_targets = [dict(s) for s in self.pergraph_targets[1:]]

    # --- public API -------------------------------------------------------

    def build(self) -> Tuple[List[Data], List[Data], List[Data]]:
        """Load + filter + split + scale. Returns (train, val, test)."""
        self._load()
        self._make_splits()
        self._fit_scalers()
        train = [self._transform(self._raw[i]) for i in self.splits.train]
        val = [self._transform(self._raw[i]) for i in self.splits.val]
        test = [self._transform(self._raw[i]) for i in self.splits.test]
        self._processed = train + val + test  # convenience reference
        return train, val, test

    @property
    def n_input_features(self) -> int:
        return sum(
            (X_COL_LAYOUT[b].stop - X_COL_LAYOUT[b].start) for b in self.use_x_blocks
        )

    @property
    def n_edge_features(self) -> int:
        if self.use_edge_cols is None:
            return len(EDGE_ATTR_NAMES)
        return len(self.use_edge_cols)

    def save_split_indices(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "train": self.splits.train,
                    "val": self.splits.val,
                    "test": self.splits.test,
                    "size_bucket": self.size_bucket,
                    "seed": self.seed,
                },
                f,
                indent=2,
            )

    def inverse_pergrain_target(
        self, y_scaled: np.ndarray, bucket_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """y_scaled (StandardScaler space) -> raw FIP.

        For `zscore_per_bucket`, callers must pass `bucket_ids` per element.
        """
        y = self._scaler_pergrain_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return _invert_target_transform(
            y,
            self.per_grain_transform,
            fit_state=getattr(self, "_pergrain_target_state", None),
            bucket_ids=bucket_ids,
        )

    def inverse_aux_pergraph_target(
        self, idx: int, y_scaled: np.ndarray, bucket_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Invert the i-th aux per-graph target back to raw units.

        Used at evaluation time to compute per-target metrics from the
        multi-output per-graph head. If the spec has a `scale` field, divide
        the recovered value by it (the spec's `scale` is applied at training
        time so the model trains in scaled-then-transformed space). If the
        spec uses `zscore_per_bucket`, callers must pass `bucket_ids` per
        SVE.
        """
        spec = self.aux_pergraph_targets[idx]
        scaler = self._scaler_aux_pergraph_y[idx]
        state = self._aux_pergraph_target_states[idx]
        # Undo loss_weight (post-zscore amplification) before StandardScaler
        # inverse — see _transform for the matching forward step.
        lw = float(spec.get("loss_weight", 1.0))
        if lw != 1.0:
            y_scaled = y_scaled / lw
        y_t = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        kwargs: Dict = {"fit_state": state}
        if spec.get("transform") == "zscore_per_bucket":
            if bucket_ids is None:
                raise ValueError(
                    f"aux target '{spec['name']}' uses zscore_per_bucket; "
                    f"caller must pass bucket_ids"
                )
            kwargs["bucket_ids"] = bucket_ids
        y_inv = _invert_target_transform(
            y_t, spec.get("transform", "identity"), **kwargs
        )
        scale = float(spec.get("scale", 1.0))
        if scale != 1.0:
            y_inv = y_inv / scale
        return y_inv

    def inverse_per_grain_aux_target(
        self, idx: int, y_scaled: np.ndarray, bucket_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Invert the i-th per-grain aux target back to raw units. Mirror of
        inverse_aux_pergraph_target but reads per-grain aux scaler/state."""
        spec = self.per_grain_aux_targets[idx]
        scaler = self._scaler_per_grain_aux_y[idx]
        state = self._per_grain_aux_target_states[idx]
        y_t = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        kwargs: Dict = {"fit_state": state}
        if spec.get("transform") == "zscore_per_bucket":
            if bucket_ids is None:
                raise ValueError(
                    f"per-grain aux '{spec['name']}' uses zscore_per_bucket; "
                    f"caller must pass bucket_ids"
                )
            kwargs["bucket_ids"] = bucket_ids
        y_inv = _invert_target_transform(
            y_t, spec.get("transform", "identity"), **kwargs
        )
        scale = float(spec.get("scale", 1.0))
        if scale != 1.0:
            y_inv = y_inv / scale
        return y_inv

    def inverse_pergraph_target(
        self, y_scaled: np.ndarray, baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """y_scaled (StandardScaler space, residual if baseline provided) -> raw FIP.

        When `target_residual_baseline` is set, callers must pass per-graph
        `baseline` (in transformed-target space, same units as the scaler input).
        """
        y_resid = self._scaler_pergraph_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()
        if self.target_residual_baseline is None:
            y_trans = y_resid
        else:
            assert (
                baseline is not None
            ), "baseline required when target_residual_baseline is set"
            y_trans = y_resid + np.asarray(baseline).flatten()
        return _invert_target_transform(
            y_trans,
            self.per_graph_transform,
            fit_state=getattr(self, "_pergraph_target_state", None),
        )

    # --- internals --------------------------------------------------------

    def _load(self) -> None:
        if self._raw is not None:
            return
        print(f"[FIPDataset] loading {self.pt_path}")
        from fip_release_paths import rebase_path

        self.pt_path = rebase_path(self.pt_path)
        data_list: List[Data] = torch.load(self.pt_path, weights_only=False)
        # Augment x with 12-sorted Schmid factors when requested. Computed
        # from cols 0..3 (Bunge Euler radians) using the audited orthogonal
        # FCC slip table. Done once at load (negligible cost); keeps the .pt
        # untouched on disk.
        # Phase 4 derived blocks: compute & append in canonical order.
        # schmid12 needed if any of {schmid12, schmid_asymmetry, schmid_std}
        # is in use_x_blocks.
        blocks_set = set(self.use_x_blocks)
        needs_s12 = bool(blocks_set & {"schmid12", "schmid_asymmetry", "schmid_std"})
        if needs_s12:
            need_cols = X_COL_LAYOUT["schmid12"].stop  # 40
            for d in data_list:
                if d.x.shape[1] < need_cols:
                    eul = d.x[:, 0:3].numpy().astype(np.float64)
                    s12 = _schmid12_from_eulers(eul).astype(np.float32)
                    d.x = torch.cat([d.x, torch.from_numpy(s12)], dim=1)

        # Append remaining derived blocks (log_volume, etc.) in canonical order.
        for name in DERIVED_BLOCKS_ORDER:
            if name == "schmid12" or name not in blocks_set:
                continue
            sl = X_COL_LAYOUT[name]
            for d in data_list:
                if d.x.shape[1] >= sl.stop:
                    continue  # already computed
                new_col = DERIVED_COMPUTERS[name](d.x).float()
                d.x = torch.cat([d.x, new_col], dim=1)

        # Phase 5 cross-variant blocks: read from data.{variant}_fip_dist
        # tensors (set by examples/cpfem_fip_data builder). Drop graphs
        # with NaN in any enabled cross-variant tensor (per user 2026-05-04
        # PM: no imputation; if HCF data is missing, drop the graph entirely).
        cross_blocks_in_use = [
            b
            for b in ["lcf_wp_fip_dist", "hcf_fs_fip_dist", "hcf_wp_fip_dist"]
            if b in blocks_set
        ]
        # Per-grain aux targets that read from cross-variant tensors also need
        # the all-NaN-graph drop. Mirror cross_blocks_in_use semantics for
        # target-side fields. Same applies to per-graph aux SOURCES that
        # read from cross-variant tensors (e.g. `hcf_fs_pergraph_max` reads
        # `hcf_fs_fip_dist[:, 0].max()` — if the graph is NaN-HCF the max is
        # NaN, the loss diverges, the model NaN-poisons. Bug seen in
        # AB_plus_per_graph_distr; this fix prevents it.)
        cross_target_fields = ("hcf_fs_fip_dist", "lcf_wp_fip_dist", "hcf_wp_fip_dist")
        cross_fields_in_use: List[str] = []
        for spec in self.per_grain_aux_targets or []:
            fld = spec.get("field", "fs_fip")
            if (
                fld in cross_target_fields
                and fld not in cross_blocks_in_use
                and fld not in cross_fields_in_use
            ):
                cross_fields_in_use.append(fld)
        # Per-graph aux sources that internally read from cross-variant
        # `*_fip_dist` tensors. These are SOURCE NAMES (not field names);
        # map each to the field it reads (see `_pergraph_target_raw_by_source`).
        _CROSS_PG_SOURCES = {
            "hcf_fs_pergraph_max": "hcf_fs_fip_dist",
            "hcf_wp_pergraph_max": "hcf_wp_fip_dist",
            # `lcf_wp_pergraph_max` reads `wp_fip[:, 0]` which is never
            # all-NaN on this dataset, so no filter needed for it.
        }
        for spec in self.aux_pergraph_targets or []:
            src = spec.get("source")
            fld = _CROSS_PG_SOURCES.get(src)
            if (
                fld is not None
                and fld not in cross_blocks_in_use
                and fld not in cross_fields_in_use
            ):
                cross_fields_in_use.append(fld)
        if cross_blocks_in_use or cross_fields_in_use:
            # Filter SVEs where any enabled cross-variant tensor is ALL-NaN
            # (genuinely missing data, e.g. 58 SVEs without HCF coverage).
            # Do NOT drop SVEs that have partial NaN in shape stat cols
            # (logn_sigma/skew/kurt for grains with <=3 voxels) — those are
            # degenerate-statistic cases handled by per-grain NaN MASK below.
            cross_attrs_to_check = list(cross_blocks_in_use) + list(cross_fields_in_use)
            n_before = len(data_list)
            data_list = [
                d
                for d in data_list
                if all(
                    hasattr(d, b) and not torch.isnan(getattr(d, b)).all()
                    for b in cross_attrs_to_check
                )
            ]
            n_after = len(data_list)
            if n_after < n_before:
                print(
                    f"[FIPDataset] cross-variant filter: kept "
                    f"{n_after}/{n_before} graphs (dropped "
                    f"{n_before - n_after} with all-NaN in "
                    f"{cross_attrs_to_check})"
                )
            # Zero-fill remaining NaN cells in cross-variant INPUT blocks
            # (degenerate shape stats for grains with <=3 voxels, ~0.35% of
            # grains). This is NOT imputation of missing data — it's a 'no
            # information' prior for stats that aren't estimable.
            for d in data_list:
                for b in cross_blocks_in_use:
                    t = getattr(d, b)
                    if torch.isnan(t).any():
                        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)
                        setattr(d, b, t)
            # Append cross-variant tensors to data.x in canonical order.
            for name in ["lcf_wp_fip_dist", "hcf_fs_fip_dist", "hcf_wp_fip_dist"]:
                if name not in blocks_set:
                    continue
                sl = X_COL_LAYOUT[name]
                for d in data_list:
                    if d.x.shape[1] >= sl.stop:
                        continue  # already appended
                    cv = getattr(d, name).float()
                    # Pad d.x up to sl.start if needed (e.g. user requested
                    # cross-variant without intermediate derived blocks).
                    if d.x.shape[1] < sl.start:
                        pad = torch.zeros(
                            (d.x.shape[0], sl.start - d.x.shape[1]), dtype=d.x.dtype
                        )
                        d.x = torch.cat([d.x, pad], dim=1)
                    d.x = torch.cat([d.x, cv], dim=1)

        # v9 sve_boundary block (slice 75:76). Read exact per-grain flag from
        # data.is_sve_boundary (set by the cpfem_fip_data v5 builder; uint8
        # per-grain tensor where 1 = grain touches an SVE cube outer face).
        if "sve_boundary" in blocks_set:
            sl_b = X_COL_LAYOUT["sve_boundary"]
            missing = [
                i for i, d in enumerate(data_list) if not hasattr(d, "is_sve_boundary")
            ]
            if missing:
                raise RuntimeError(
                    f"sve_boundary requested but data.is_sve_boundary is missing "
                    f"on {len(missing)} graphs (e.g. graph {missing[0]}). "
                    f"Use full_data_v5.pt (built by examples/cpfem_fip_data/"
                    f"scripts/build_v5.py); pre-v5 .pt files do not have this field."
                )
            for d in data_list:
                if d.x.shape[1] >= sl_b.stop:
                    continue  # already appended
                if d.x.shape[1] < sl_b.start:
                    pad = torch.zeros(
                        (d.x.shape[0], sl_b.start - d.x.shape[1]), dtype=d.x.dtype
                    )
                    d.x = torch.cat([d.x, pad], dim=1)
                flag = d.is_sve_boundary.to(dtype=d.x.dtype).unsqueeze(-1)
                d.x = torch.cat([d.x, flag], dim=1)

        # Per-grain aux NaN handling (2026-05-05): source tensors used as
        # per-grain aux TARGETS may carry NaN at degenerate small grains
        # (e.g. lognormal_sigma when a grain has <=3 voxels). We:
        #   1. Build `data.per_grain_aux_nan_mask` of shape (N_grains, n_aux)
        #      BEFORE any in-place fill so eval/plotting can EXCLUDE those
        #      grains from per-channel R^2 + parity plots (otherwise zero-
        #      fill creates vertical-line bands).
        #   2. Zero-fill in-place so the StandardScaler fit and the model
        #      loss receive finite values; the mask lets us undo the zero-
        #      fill artefact at metrics + plotting (model still trains on
        #      zero-filled cells; we just don't report on them).
        if self.per_grain_aux_targets:
            n_aux = len(self.per_grain_aux_targets)
            for d in data_list:
                N = int(d.x.shape[0])
                mask = torch.zeros((N, n_aux), dtype=torch.bool)
                for k, spec in enumerate(self.per_grain_aux_targets):
                    fld = spec.get("field", "fs_fip")
                    col = int(spec["col"])
                    mask[:, k] = torch.isnan(getattr(d, fld)[:, col])
                d.per_grain_aux_nan_mask = mask
            cols_per_field: Dict[str, List[int]] = {}
            for spec in self.per_grain_aux_targets:
                cols_per_field.setdefault(spec.get("field", "fs_fip"), []).append(
                    int(spec["col"])
                )
            for fld, cols in cols_per_field.items():
                cols = sorted(set(cols))
                for d in data_list:
                    t = getattr(d, fld)
                    sub = t[:, cols]
                    if torch.isnan(sub).any():
                        sub = torch.where(torch.isnan(sub), torch.zeros_like(sub), sub)
                        t = t.clone()
                        t[:, cols] = sub
                        setattr(d, fld, t)

        if self.size_bucket is not None:
            data_list = [
                d for d in data_list if int(d.size_bucket[0]) == self.size_bucket
            ]
            print(
                f"[FIPDataset] filtered to bucket {self.size_bucket}: {len(data_list)} graphs"
            )
        self._raw = data_list

    def _make_splits(self) -> None:
        n = len(self._raw)
        rng = np.random.default_rng(self.seed)
        ftr, fva, fte = self.split_fracs
        assert abs(ftr + fva + fte - 1.0) < 1e-6, "split fractions must sum to 1"

        if self.train_buckets is not None:
            # Extrapolation mode. train+val drawn from train_buckets; test from
            # test_buckets (or "everything else" if test_buckets is None).
            buckets = np.array([int(d.size_bucket[0]) for d in self._raw])
            train_set = set(int(b) for b in self.train_buckets)
            if self.test_buckets is None:
                test_set = set(int(b) for b in np.unique(buckets)) - train_set
            else:
                test_set = set(int(b) for b in self.test_buckets)
            assert not (
                train_set & test_set
            ), "train_buckets and test_buckets must be disjoint"
            tr_idx = np.where(np.isin(buckets, list(train_set)))[0]
            te_idx = np.where(np.isin(buckets, list(test_set)))[0]
            rng.shuffle(tr_idx)
            # 95/5 split inside train_buckets for val.
            n_va = max(1, int(round(len(tr_idx) * 0.05)))
            val = tr_idx[:n_va].tolist()
            train = tr_idx[n_va:].tolist()
            test = te_idx.tolist()
            print(
                f"[FIPDataset] EXTRAPOLATION split: train_buckets={sorted(train_set)} -> {len(train)}, "
                f"val=5% of train_buckets -> {len(val)}, test_buckets={sorted(test_set)} -> {len(test)} (seed={self.seed})"
            )
        elif self.size_bucket is not None or not self.stratify_by_bucket:
            # Single-bucket OR explicit no-stratify: plain shuffle.
            idx = np.arange(n)
            rng.shuffle(idx)
            n_tr = int(round(n * ftr))
            n_va = int(round(n * fva))
            train = idx[:n_tr].tolist()
            val = idx[n_tr : n_tr + n_va].tolist()
            test = idx[n_tr + n_va :].tolist()
            print(
                f"[FIPDataset] split: train={len(train)} val={len(val)} test={len(test)} (seed={self.seed}, stratified=False)"
            )
        else:
            # Global mode: stratify by SVE size bucket so each split sees all sizes.
            buckets = np.array([int(d.size_bucket[0]) for d in self._raw])
            train, val, test = [], [], []
            for b in np.unique(buckets):
                bidx = np.where(buckets == b)[0]
                rng.shuffle(bidx)
                nb = len(bidx)
                n_tr = int(round(nb * ftr))
                n_va = int(round(nb * fva))
                train.extend(bidx[:n_tr].tolist())
                val.extend(bidx[n_tr : n_tr + n_va].tolist())
                test.extend(bidx[n_tr + n_va :].tolist())
            print(
                f"[FIPDataset] split: train={len(train)} val={len(val)} test={len(test)} (seed={self.seed}, stratified=True)"
            )

        self.splits = FIPSplits(train=train, val=val, test=test)

    def _pergraph_target_raw(self, d_raw: Data) -> float:
        """Per-graph target value in raw units, dispatching on
        `self.per_graph_target_source` (default = FIP aggregate)."""
        return self._pergraph_target_raw_by_source(d_raw, self.per_graph_target_source)

    def _pergraph_target_raw_by_source(self, d_raw: Data, src: str) -> float:
        """Per-graph target value for a specific source name. Used both for
        the primary per-graph target (FIP) and for any aux multi-task targets.
        """
        if src in (None, "", "default", "fs_fip_max"):
            v = getattr(d_raw, self.per_grain_target_field)[
                :, self.per_grain_target_col
            ].numpy()
            agg = self.per_graph_target_aggr
            if agg == "max":
                return float(v.max())
            if agg == "P99":
                return float(np.percentile(v, 99))
            if agg == "P95":
                return float(np.percentile(v, 95))
            if agg == "P90":
                return float(np.percentile(v, 90))
            if agg == "mean":
                return float(v.mean())
            raise ValueError(f"unknown per_graph_target_aggr: {agg}")
        # Mechanical QoI sources (per-SVE scalars from CPFEM responses).
        if src == "yield_stress":
            return float(d_raw.monotonic_plastic[2].item())
        if src == "hardening_modulus":
            return float(d_raw.monotonic_plastic[3].item())
        if src == "cyc_plastic_strain":
            return float(d_raw.cyclic_plastic[1:][2].item())
        if src == "cyc_stress_amp":
            return float(d_raw.cyclic_plastic[1:][3].item())
        if src == "aniso_E":
            me = d_raw.monotonic_elastic.numpy()
            Es = np.array([me[13], me[20], me[27]], dtype=np.float64)
            return float(Es.std() / Es.mean())
        if src == "aniso_G":
            me = d_raw.monotonic_elastic.numpy()
            Gs = np.array([me[34], me[41], me[48]], dtype=np.float64)
            return float(Gs.std() / Gs.mean())
        # Cyclic-elastic (Δε=0.001 pre-yield) responses, per Anto 2026:
        if src == "cyc_low_stress_amp":
            return float(d_raw.cyclic_elastic[1:][3].item())  # σ_amp at low Δε
        if src == "cyc_low_plastic_strain":
            return float(d_raw.cyclic_elastic[1:][0].item())  # ε_p amp at low Δε
        # v9b mech QoI redesign (2026-05-06): plastic energy Wp and stress
        # amplitude σ_amp evaluated at cycle 3 (the LAST cycle = cycleN
        # mode in the per-cycle summary) for both LCF and HCF regimes.
        # Source: cyclic_qoi_{lcf,hcf}_summary, packed as
        # METRIC × MODE; we use cycleN mode to match the user's
        # "from cycle_3" specification (data has 3 cycles per SVE).
        # Index math (per cyclic_qoi.py: METRIC × 4 + MODE):
        #   energy_plastic_given × cycleN = 2 × 4 + 1 = 9
        #   amplitude_stress     × cycleN = 8 × 4 + 1 = 33
        if src == "Wp_lcf":
            return float(d_raw.cyclic_qoi_lcf_summary[9].item())
        if src == "Wp_hcf":
            return float(d_raw.cyclic_qoi_hcf_summary[9].item())
        if src == "sigma_amp_lcf":
            return float(d_raw.cyclic_qoi_lcf_summary[33].item())
        if src == "sigma_amp_hcf":
            return float(d_raw.cyclic_qoi_hcf_summary[33].item())
        # Per-SVE NORMAL parameters of the within-SVE per-grain MAX FIP
        # distribution. Per the within-VE empirical analysis (KS-confirmed
        # 2026-04-28: per-grain MAX is near-Normal, NOT Gumbel; Anto's Gumbel
        # applies cross-SVE), we fit Normal moments here:
        #   ve_mu_pergraph    = mean of per-grain MAX FIPs inside this SVE
        #   ve_sigma_pergraph = std  of per-grain MAX FIPs inside this SVE
        # The earlier `gumbel_mu_pergraph` / `gumbel_beta_pergraph` keys
        # remain accepted as ALIASES that redirect to the Normal mean / std
        # respectively, but new cells should use the `ve_*` names. The v5
        # paper's UQ uses train-set σ_pop at inference (per
        # sigma_pop_construction_2026-05-04.md); these per-SVE targets give
        # the architecture a chance to learn σ̂ instead, which P6 onwards
        # tests as an aleatoric-UQ supervisory signal.
        if src in (
            "ve_mu_pergraph",
            "ve_sigma_pergraph",
            "gumbel_mu_pergraph",
            "gumbel_beta_pergraph",
        ):
            grain_max = (
                getattr(d_raw, self.per_grain_target_field)[
                    :, self.per_grain_target_col
                ]
                .numpy()
                .astype(np.float64)
            )
            if src in ("ve_sigma_pergraph", "gumbel_beta_pergraph"):
                return float(grain_max.std(ddof=0))
            return float(grain_max.mean())
        # Cross-FIP-variant per-graph MAX. LCF/WP from data.wp_fip[:, 0]
        # (per-grain MAX of WP variant), HCF/* from cross-variant *_fip_dist
        # tensors. Symmetric counterparts to the primary LCF/FS per-graph MAX
        # so the model can predict all four variants' SVE-level extremes.
        # NaN-HCF graphs are dropped at load when HCF sources are used.
        # v9b deslot: LCF/FS pg-MAX is now an explicit named source — it
        # used to be implicit (FIP-aggregate path on slot 0). Mapping is
        # exact to data.fs_fip[:, 0].max() (= per-grain MAX of LCF/FS FIP).
        if src == "lcf_fs_pergraph_max":
            return float(d_raw.fs_fip[:, 0].numpy().max())
        if src in ("wp_pergraph_max", "lcf_wp_pergraph_max"):
            return float(d_raw.wp_fip[:, 0].numpy().max())
        if src == "hcf_fs_pergraph_max":
            return float(d_raw.hcf_fs_fip_dist[:, 0].numpy().max())
        if src == "hcf_wp_pergraph_max":
            return float(d_raw.hcf_wp_fip_dist[:, 0].numpy().max())
        # VE-level Normal parameters (mean, std) of the within-SVE per-grain
        # MAX FIP distribution, for each of the 4 variants. The primary
        # variant's params are already exposed via `ve_{mu,sigma}_pergraph`
        # above; here we add per-variant aliases so all 4 variants have
        # symmetric (μ, σ) targets.
        _ve_dist_map = {
            "lcf_fs_ve_mu_pergraph": ("lcf_fs", "mean"),
            "lcf_fs_ve_sigma_pergraph": ("lcf_fs", "std"),
            "lcf_wp_ve_mu_pergraph": ("lcf_wp", "mean"),
            "lcf_wp_ve_sigma_pergraph": ("lcf_wp", "std"),
            "hcf_fs_ve_mu_pergraph": ("hcf_fs", "mean"),
            "hcf_fs_ve_sigma_pergraph": ("hcf_fs", "std"),
            "hcf_wp_ve_mu_pergraph": ("hcf_wp", "mean"),
            "hcf_wp_ve_sigma_pergraph": ("hcf_wp", "std"),
        }
        if src in _ve_dist_map:
            variant, stat = _ve_dist_map[src]
            if variant == "lcf_fs":
                vec = d_raw.fs_fip[:, 0].numpy().astype(np.float64)
            elif variant == "lcf_wp":
                vec = d_raw.wp_fip[:, 0].numpy().astype(np.float64)
            elif variant == "hcf_fs":
                vec = d_raw.hcf_fs_fip_dist[:, 0].numpy().astype(np.float64)
            elif variant == "hcf_wp":
                vec = d_raw.hcf_wp_fip_dist[:, 0].numpy().astype(np.float64)
            else:
                raise ValueError(f"unknown variant: {variant}")
            return float(vec.std(ddof=0)) if stat == "std" else float(vec.mean())
        raise ValueError(f"unknown per_graph_target_source: {src}")

    def _pergraph_baseline_transformed(self, d_raw: Data) -> float:
        """Per-graph baseline in transformed-target space (same as what the
        scaler sees). Returns 0.0 if no residualization is configured."""
        if self.target_residual_baseline is None:
            return 0.0
        if self.target_residual_baseline == "per_bucket_mean":
            assert self._residual_per_bucket_mean is not None, "fit residual first"
            return float(
                self._residual_per_bucket_mean.get(int(d_raw.size_bucket[0]), 0.0)
            )
        if self.target_residual_baseline == "log_n_ols":
            assert self._residual_log_n_coef is not None, "fit residual first"
            a, b = self._residual_log_n_coef
            return float(a + b * float(np.log(int(d_raw.x.shape[0]))))
        raise ValueError(
            f"unknown target_residual_baseline: {self.target_residual_baseline}"
        )

    def _graph_attr_raw(self, d_raw: Data) -> np.ndarray:
        """Raw (unscaled) graph_attr for one graph based on `graph_attr_source`."""
        src = self.graph_attr_source
        if src is None:
            return np.zeros((0,), dtype=np.float32)
        if src == "L_SVE_um":
            return np.array([float(d_raw.L_SVE_um[0])], dtype=np.float32)
        if src == "log_n_nodes":
            return np.array([float(np.log(int(d_raw.x.shape[0])))], dtype=np.float32)
        if src == "size_bucket_onehot":
            v = np.zeros(5, dtype=np.float32)
            v[int(d_raw.size_bucket[0])] = 1.0
            return v
        if src == "agg_microstructure":
            # Multi-d summary of within-bucket microstructure variation:
            # log(n_grains) + Schmid (mean/max/std) + volume (mean/max/std,log)
            # + surface_area (mean/max,log) + edge stats (mean miso/length).
            n = int(d_raw.x.shape[0])
            x = d_raw.x.numpy()  # (N, 22)
            schmid = x[:, 7]  # col 7 per X_COL_LAYOUT
            vol = x[:, 8]  # col 8 (positive)
            sa = x[:, 9]  # col 9
            ea = (
                d_raw.edge_attr.numpy() if d_raw.edge_attr.numel() else np.zeros((1, 8))
            )
            miso = ea[:, 0]  # misorientation
            elen = ea[:, 1]  # edge length
            feats = [
                float(np.log(n)),  # 1
                float(schmid.mean()),
                float(schmid.max()),
                float(schmid.std()),  # 3
                float(np.log(vol.mean() + 1e-12)),  # 1
                float(np.log(vol.max() + 1e-12)),  # 1
                float(np.log(vol.std() + 1e-12)),  # 1
                float(np.log(sa.mean() + 1e-12)),  # 1
                float(np.log(sa.max() + 1e-12)),  # 1
                float(miso.mean()),
                float(miso.std()),  # 2
                float(np.log(elen.mean() + 1e-12)),  # 1
            ]
            return np.array(feats, dtype=np.float32)
        if src == "agg_microstructure_v2":
            # Richer graph-level summary that adds elastic-mismatch and
            # slip-transfer stats. Per user 2026-04-27: try elastic labels in
            # graph attr to get within-bucket signal.
            n = int(d_raw.x.shape[0])
            x = d_raw.x.numpy()
            schmid = x[:, 7]
            vol = x[:, 8]
            sa = x[:, 9]
            ea = (
                d_raw.edge_attr.numpy() if d_raw.edge_attr.numel() else np.zeros((1, 8))
            )
            miso = ea[:, 0]
            elen = ea[:, 1]
            mprime = ea[:, 5]  # slip-transfer m'
            emis = ea[:, 6]  # elastic mismatch
            feats = [
                float(np.log(n)),  # 1
                # Schmid distribution
                float(schmid.mean()),
                float(schmid.max()),
                float(schmid.std()),  # 3
                # Volume (log)
                float(np.log(vol.mean() + 1e-12)),  # 1
                float(np.log(vol.max() + 1e-12)),  # 1
                float(np.log(vol.std() + 1e-12)),  # 1
                # Surface area (log)
                float(np.log(sa.mean() + 1e-12)),  # 1
                float(np.log(sa.max() + 1e-12)),  # 1
                # Misorientation
                float(miso.mean()),
                float(miso.std()),  # 2
                float(np.log(elen.mean() + 1e-12)),  # 1
                # NEW: elastic mismatch stats (driver of inter-grain stress concentration)
                float(emis.mean()),
                float(emis.max()),
                float(emis.std()),  # 3
                # NEW: slip-transfer m' stats (Luster–Morris)
                float(mprime.mean()),
                float(mprime.min()),
                float(mprime.std()),  # 3
            ]
            return np.array(feats, dtype=np.float32)
        if src == "agg_elastic_only":
            # Minimal elastic-focused summary — test if elastic alone helps.
            n = int(d_raw.x.shape[0])
            ea = (
                d_raw.edge_attr.numpy() if d_raw.edge_attr.numel() else np.zeros((1, 8))
            )
            mprime = ea[:, 5]
            emis = ea[:, 6]
            feats = [
                float(np.log(n)),
                float(emis.mean()),
                float(emis.max()),
                float(emis.std()),
                float(mprime.mean()),
                float(mprime.min()),
                float(mprime.std()),
            ]
            return np.array(feats, dtype=np.float32)
        # === Per-SVE CPFEM mechanical response fields ===
        # These are macroscopic/homogenized results from the same SVEs the FIPs
        # came from. They capture each SVE's intrinsic mechanical behavior and
        # should provide strong within-bucket signal that's currently absent.
        if src == "monotonic_plastic":
            return d_raw.monotonic_plastic.numpy().astype(np.float32)  # (6,)
        if src == "cyclic_plastic":
            # First component is always N=3.0 (constant), drop it
            return d_raw.cyclic_plastic.numpy()[1:].astype(np.float32)  # (7,)
        if src == "monotonic_elastic":
            return d_raw.monotonic_elastic.numpy().astype(np.float32)  # (49,)
        if src == "cyclic_elastic":
            # First component is always N=3.0 (constant), drop it
            return d_raw.cyclic_elastic.numpy()[1:].astype(np.float32)  # (7,)
        if src == "mech_compact":
            # Compact mechanical summary: yield stress, hardening, yield strain,
            # cyclic stress amplitudes, cyclic strain amplitudes — 14 numbers.
            # NOTE: includes plastic responses, which are FIP-coupled and so
            # may be considered leakage for paper-grade results. Use
            # `mech_elastic` for the publication-safe variant.
            mp = d_raw.monotonic_plastic.numpy()  # 6
            cp = d_raw.cyclic_plastic.numpy()  # 8 (first is N const)
            ce = d_raw.cyclic_elastic.numpy()  # 8 (first is N const)
            return np.concatenate([mp, cp[1:], ce[1:]]).astype(np.float32)  # 6+7+7=20
        if src == "mech_elastic":
            # FULL ELASTIC: monotonic_elastic (49) + cyclic_elastic[1:] (7) = 56-d.
            # Most of the 49 monotonic dims are dataset-wide constants (e.g.
            # stiffness components shared by all FCC realizations); use only
            # for completeness/sanity — not recommended as a default.
            me = d_raw.monotonic_elastic.numpy()  # 49
            ce = d_raw.cyclic_elastic.numpy()  # 8 (first is N const)
            return np.concatenate([me, ce[1:]]).astype(np.float32)  # 49+7=56
        if src == "mech_compact_elastic":
            # COMPACT ELASTIC analog of mech_compact: keep only dims with
            # meaningful inter-SVE variance. Identified empirically on
            # full_data_v3.pt (998 SVEs):
            #
            # Of monotonic_elastic's 49 dims, only 9 have std > 0.2 across
            # the dataset; the rest are dataset-wide elastic constants.
            # Of cyclic_elastic's 8 dims, only 3 vary meaningfully (dim 0
            # = N=3 const; dims 1, 2, 5, 7 have std < 1e-3 across SVEs).
            # See `feedback`/audit notes; re-derive if dataset is rebuilt.
            #
            #   monotonic_elastic[[6, 8, 9, 13, 20, 21, 27, 34]]   = 8 dims
            #   cyclic_elastic[[3, 4, 6]]                           = 3 dims
            #   total = 11-d   (vs. 20-d for mech_compact, 56-d for mech_elastic)
            #
            # PUBLICATION-SAFE: no plastic components, so no FIP coupling risk.
            ME_VAR_DIMS = [6, 8, 9, 13, 20, 21, 27, 34]
            CE_VAR_DIMS = [3, 4, 6]
            me = d_raw.monotonic_elastic.numpy()
            ce = d_raw.cyclic_elastic.numpy()
            return np.concatenate([me[ME_VAR_DIMS], ce[CE_VAR_DIMS]]).astype(
                np.float32
            )  # 11
        if src == "mech_full":
            # All mechanical responses — 6 + 7 + 49 + 7 = 69 features.
            mp = d_raw.monotonic_plastic.numpy()
            cp = d_raw.cyclic_plastic.numpy()[1:]
            me = d_raw.monotonic_elastic.numpy()
            ce = d_raw.cyclic_elastic.numpy()[1:]
            return np.concatenate([mp, cp, me, ce]).astype(np.float32)  # 69
        if src == "agg_plus_mech":
            # 12-d microstructure summary + 20-d compact mechanical = 32-d
            n = int(d_raw.x.shape[0])
            x = d_raw.x.numpy()
            schmid = x[:, 7]
            vol = x[:, 8]
            sa = x[:, 9]
            ea = (
                d_raw.edge_attr.numpy() if d_raw.edge_attr.numel() else np.zeros((1, 8))
            )
            miso = ea[:, 0]
            elen = ea[:, 1]
            mprime = ea[:, 5]
            emis = ea[:, 6]
            mp = d_raw.monotonic_plastic.numpy()
            cp = d_raw.cyclic_plastic.numpy()[1:]
            ce = d_raw.cyclic_elastic.numpy()[1:]
            agg = np.array(
                [
                    np.log(n),
                    schmid.mean(),
                    schmid.max(),
                    schmid.std(),
                    np.log(vol.mean() + 1e-12),
                    np.log(vol.max() + 1e-12),
                    np.log(vol.std() + 1e-12),
                    np.log(sa.mean() + 1e-12),
                    np.log(sa.max() + 1e-12),
                    miso.mean(),
                    miso.std(),
                    np.log(elen.mean() + 1e-12),
                    emis.mean(),
                    emis.max(),
                    emis.std(),
                    mprime.mean(),
                    mprime.min(),
                    mprime.std(),
                ],
                dtype=np.float32,
            )
            return np.concatenate(
                [
                    agg,
                    mp.astype(np.float32),
                    cp.astype(np.float32),
                    ce.astype(np.float32),
                ]
            )  # 18+6+7+7=38
        raise ValueError(f"unknown graph_attr_source: {src}")

    def _fit_scalers(self) -> None:
        train_data = [self._raw[i] for i in self.splits.train]
        # x scaler
        x_blocks = []
        for d in train_data:
            x_blocks.append(self._select_x(d.x).numpy())
        x_all = np.concatenate(x_blocks, axis=0)
        self._scaler_x = StandardScaler().fit(x_all)

        # === target transform fit state (NEW v3-study extension) ===========
        # Stateful target transforms (zscore, yeo_johnson, quantile_normal,
        # gumbel_uniform, gumbel_probit, zscore_per_bucket) cache their fitted
        # parameters in self._pergrain_target_state and self._pergraph_target_state
        # so that train/val/test all share the same transform.
        self._pergrain_target_state: dict = {}
        self._pergraph_target_state: dict = {}
        # bucket_ids per node (for zscore_per_bucket on per-grain target)
        self._pergrain_train_bucket_ids: np.ndarray | None = None
        if (
            self.use_per_grain_target
            and self.per_grain_transform == "zscore_per_bucket"
        ):
            self._pergrain_train_bucket_ids = np.concatenate(
                [
                    np.full(int(d.x.shape[0]), int(d.size_bucket[0]), dtype=np.int64)
                    for d in train_data
                ]
            )

        # per-grain y scaler
        if self.use_per_grain_target:
            y_pg_raw = np.concatenate(
                [
                    getattr(d, self.per_grain_target_field)[
                        :, self.per_grain_target_col
                    ].numpy()
                    for d in train_data
                ]
            )
            y_pg_train_bucket_ids = self._pergrain_train_bucket_ids
            # Fit the target transform on training pool (mutates state dict).
            y_pg = _apply_target_transform(
                y_pg_raw,
                self.per_grain_transform,
                fit_state=self._pergrain_target_state,
                bucket_ids=y_pg_train_bucket_ids,
            ).reshape(-1, 1)
            self._scaler_pergrain_y = StandardScaler().fit(y_pg)

        # Per-grain aux targets (multi-task at the node level). Mirror of
        # aux_pergraph_targets but reads per-node values. Source NaN cells
        # were zero-filled at load — fit on the zero-filled pool here, mask
        # out NaN-source grains at eval time via per_grain_aux_nan_mask.
        self._per_grain_aux_target_states = []
        self._scaler_per_grain_aux_y = []
        if self.use_per_grain_target and self.per_grain_aux_targets:
            train_buckets_per_node = np.concatenate(
                [
                    np.full(int(d.x.shape[0]), int(d.size_bucket[0]), dtype=np.int64)
                    for d in train_data
                ]
            )
            for spec in self.per_grain_aux_targets:
                fld = spec.get("field", "fs_fip")
                col = int(spec["col"])
                tform = spec.get("transform", "identity")
                scale = float(spec.get("scale", 1.0))
                state: dict = {}
                raw_train = np.concatenate(
                    [
                        getattr(d, fld)[:, col].numpy().astype(np.float64) * scale
                        for d in train_data
                    ]
                )
                kw = {"fit_state": state}
                if tform == "zscore_per_bucket":
                    kw["bucket_ids"] = train_buckets_per_node
                trans_train = _apply_target_transform(raw_train, tform, **kw)
                scaler = StandardScaler().fit(trans_train.reshape(-1, 1))
                self._per_grain_aux_target_states.append(state)
                self._scaler_per_grain_aux_y.append(scaler)
                print(
                    f"[FIPDataset] per-grain aux '{spec['name']}' "
                    f"(field={fld} col={col} transform={tform} scale={scale}): "
                    f"raw*scale mean={raw_train.mean():+.4g} std={raw_train.std():.4g}; "
                    f"transformed mean={trans_train.mean():+.4g} std={trans_train.std():.4g}"
                )

        # per-graph y scaler — fitted on RESIDUALS when residualization is on.
        if self.use_per_graph_target:
            # Fit the per-graph target transform on the TRAIN raw values once;
            # `fit_state` is mutated in place and reused by every subsequent
            # call (val/test).  Required for stateful transforms (zscore,
            # yeo_johnson, quantile_normal, gumbel_*).  log1p / identity /
            # log / sqrt are stateless and ignore `fit_state`.
            y_pgr_raw = np.array(
                [self._pergraph_target_raw(d) for d in train_data], dtype=np.float64
            )
            y_pgr_T = _apply_target_transform(
                y_pgr_raw,
                self.per_graph_transform,
                fit_state=self._pergraph_target_state,
            )
            n_nodes_tr = np.array(
                [int(d.x.shape[0]) for d in train_data], dtype=np.float64
            )
            buckets_tr = np.array(
                [int(d.size_bucket[0]) for d in train_data], dtype=np.int64
            )

            # Fit residual baseline on TRAIN only.
            if self.target_residual_baseline == "per_bucket_mean":
                self._residual_per_bucket_mean = {
                    int(b): float(y_pgr_T[buckets_tr == b].mean())
                    for b in np.unique(buckets_tr)
                }
                baselines_tr = np.array(
                    [self._residual_per_bucket_mean[int(b)] for b in buckets_tr]
                )
                print(
                    f"[FIPDataset] residual=per_bucket_mean   per-bucket means: {self._residual_per_bucket_mean}"
                )
            elif self.target_residual_baseline == "log_n_ols":
                # Fit y_T = a + b·log(N) via OLS on train.
                X = np.column_stack([np.ones_like(n_nodes_tr), np.log(n_nodes_tr)])
                coef, *_ = np.linalg.lstsq(X, y_pgr_T, rcond=None)
                a, b = float(coef[0]), float(coef[1])
                self._residual_log_n_coef = (a, b)
                baselines_tr = a + b * np.log(n_nodes_tr)
                print(
                    f"[FIPDataset] residual=log_n_ols   y_T ≈ {a:+.4f} + {b:+.4f}·log(N)"
                )
            else:
                baselines_tr = np.zeros_like(y_pgr_T)

            y_residual = (y_pgr_T - baselines_tr).reshape(-1, 1)
            self._scaler_pergraph_y = StandardScaler().fit(y_residual)

        # Aux per-graph targets (multi-task learning). Each gets its own
        # transform fit-state and StandardScaler. Spec fields:
        #   - source: target source key (yield_stress, gumbel_mu_pergraph, ...)
        #   - transform: target transform name (zscore, yeo_johnson, zscore_per_bucket, ...)
        #   - scale: optional pre-multiplier for tiny-magnitude targets
        if self.aux_pergraph_targets:
            self._aux_pergraph_target_states = []
            self._scaler_aux_pergraph_y = []
            train_buckets = np.array(
                [int(d.size_bucket[0]) for d in train_data], dtype=np.int64
            )
            for spec in self.aux_pergraph_targets:
                src = spec["source"]
                trans = spec.get("transform", "identity")
                scale = float(spec.get("scale", 1.0))
                state: dict = {}
                raw_train = np.array(
                    [
                        self._pergraph_target_raw_by_source(d, src) * scale
                        for d in train_data
                    ],
                    dtype=np.float64,
                )
                # Per-bucket transforms need bucket_ids per element.
                if trans == "zscore_per_bucket":
                    trans_train = _apply_target_transform(
                        raw_train,
                        trans,
                        fit_state=state,
                        bucket_ids=train_buckets,
                    )
                else:
                    trans_train = _apply_target_transform(
                        raw_train, trans, fit_state=state
                    )
                scaler = StandardScaler().fit(trans_train.reshape(-1, 1))
                self._aux_pergraph_target_states.append(state)
                self._scaler_aux_pergraph_y.append(scaler)
                print(
                    f"[FIPDataset] aux pg target '{spec['name']}' (src={src}, "
                    f"transform={trans}, scale={scale}): "
                    f"raw*scale mean={raw_train.mean():+.4g} std={raw_train.std():.4g}; "
                    f"transformed mean={trans_train.mean():+.4g} std={trans_train.std():.4g}"
                )

        # edge_attr scaler — fit on TRAIN only, apply to all splits.
        # Skipped when normalize_edges=False or no edges are used.
        if self.normalize_edges:
            ea_blocks = []
            for d in train_data:
                if d.edge_attr is not None and d.edge_attr.numel():
                    ea_sel = self._select_edge_attr(d.edge_attr).numpy()
                    if ea_sel.shape[0] > 0:
                        ea_blocks.append(ea_sel)
            if ea_blocks and ea_blocks[0].shape[1] > 0:
                ea_all = np.concatenate(ea_blocks, axis=0)
                self._scaler_edge_attr = StandardScaler().fit(ea_all)
                print(
                    f"[FIPDataset] edge_attr scaler fit: dim={ea_all.shape[1]}, "
                    f"mean={self._scaler_edge_attr.mean_.round(3).tolist()}, "
                    f"std={np.sqrt(self._scaler_edge_attr.var_).round(3).tolist()}"
                )
            else:
                self._scaler_edge_attr = None
                print(f"[FIPDataset] edge_attr scaler skipped (use_edge_cols empty)")
        else:
            self._scaler_edge_attr = None

        # graph_attr scaler
        if self.graph_attr_source is not None:
            ga = np.stack([self._graph_attr_raw(d) for d in train_data], axis=0)
            self._graph_attr_dim = ga.shape[1]
            # one-hot doesn't need scaling
            if self.graph_attr_source == "size_bucket_onehot":
                self._scaler_graph_attr = None
            else:
                self._scaler_graph_attr = StandardScaler().fit(ga)

    def _select_x(self, x: torch.Tensor) -> torch.Tensor:
        cols = []
        for b in self.use_x_blocks:
            sl = X_COL_LAYOUT[b]
            cols.append(x[:, sl])
        return torch.cat(cols, dim=1)

    def _select_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Select / compute edge features by name. Names in EDGE_ATTR_NAMES
        are looked up directly; names in EDGE_XFORMS are computed from
        the raw edge_attr (e.g. inv_edge_length, cos_misor). Empty
        use_edge_cols returns a (n_edges, 0) tensor — used by SAGE-style
        cells that don't consume edge_attr."""
        if self.use_edge_cols is None:
            return edge_attr
        if not self.use_edge_cols:
            return edge_attr.new_zeros((edge_attr.shape[0], 0)).float()
        cols = []
        for n in self.use_edge_cols:
            if n in EDGE_ATTR_INDEX:
                cols.append(edge_attr[:, EDGE_ATTR_INDEX[n] : EDGE_ATTR_INDEX[n] + 1])
            elif n in EDGE_XFORMS:
                cols.append(EDGE_XFORMS[n](edge_attr))
            else:
                raise KeyError(
                    f"Unknown edge feature name: {n!r}. "
                    f"Valid: raw {sorted(EDGE_ATTR_INDEX)} "
                    f"or transform {sorted(EDGE_XFORMS)}."
                )
        return torch.cat(cols, dim=1).float()

    def _transform(self, d_raw: Data) -> Data:
        # 1. Select x blocks + standardize.
        x_sel = self._select_x(d_raw.x).numpy()
        x_sc = self._scaler_x.transform(x_sel).astype(np.float32)

        # 2. Select / filter edges
        if self.drop_periodic_edges:
            keep = d_raw.edge_attr[:, EDGE_ATTR_INDEX["is_periodic"]] < 0.5
            edge_index = d_raw.edge_index[:, keep]
            edge_attr_full = d_raw.edge_attr[keep]
        else:
            edge_index = d_raw.edge_index
            edge_attr_full = d_raw.edge_attr
        edge_attr = self._select_edge_attr(edge_attr_full).float()
        if self._scaler_edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = torch.from_numpy(
                self._scaler_edge_attr.transform(edge_attr.numpy()).astype(np.float32)
            )

        # 3. Build target tensor + y_loc.
        # HydraGNN convention: data.y is flat, data.y_loc[0] gives offsets per head.
        # Per-grain head (n_nodes elements) comes first if active; per-graph head (1 elem) follows.
        ys: List[torch.Tensor] = []
        offsets: List[int] = [0]
        if self.use_per_grain_target:
            n = int(d_raw.x.shape[0])
            bucket_id = int(d_raw.size_bucket[0])
            pg_bucket_ids = np.full(n, bucket_id, dtype=np.int64)
            y_pg_raw = _apply_target_transform(
                getattr(d_raw, self.per_grain_target_field)[
                    :, self.per_grain_target_col
                ].numpy(),
                self.per_grain_transform,
                fit_state=getattr(self, "_pergrain_target_state", None),
                bucket_ids=pg_bucket_ids,
            ).reshape(-1, 1)
            y_pg_primary = self._scaler_pergrain_y.transform(y_pg_raw).astype(
                np.float32
            )  # (N, 1)
            channels = [y_pg_primary]
            # Append per-grain aux targets as additional channels, in spec order.
            if self.per_grain_aux_targets:
                for spec, state, scaler in zip(
                    self.per_grain_aux_targets,
                    self._per_grain_aux_target_states,
                    self._scaler_per_grain_aux_y,
                ):
                    fld = spec.get("field", "fs_fip")
                    col = int(spec["col"])
                    tform = spec.get("transform", "identity")
                    scale = float(spec.get("scale", 1.0))
                    raw = getattr(d_raw, fld)[:, col].numpy().astype(np.float64) * scale
                    kw = {"fit_state": state}
                    if tform == "zscore_per_bucket":
                        kw["bucket_ids"] = pg_bucket_ids
                    trans = _apply_target_transform(raw, tform, **kw).reshape(-1, 1)
                    scaled = scaler.transform(trans).astype(np.float32)
                    channels.append(scaled)
            # Phase L5 heteroscedastic head extension: 4 per-cell σ outputs
            # are unsupervised nuisance parameters of the Gaussian-NLL loss.
            # HydraGNN strict-slices y by output_dim × n_nodes, so we pad y
            # with 4 zero columns per grain. The custom NLL loss class
            # (`_HeteroscedasticTopdecLoss` in losses.py) ignores these slots.
            if bool(getattr(self, "use_heteroscedastic_max", False)):
                channels.append(np.zeros((n, 4), dtype=np.float32))
            # Build (N, 1+K) and flatten ROW-MAJOR so HydraGNN's per-node head
            # with output_dim=1+K reads target channels correctly per node.
            y_pg_full = np.concatenate(channels, axis=1).astype(np.float32)
            ys.append(torch.from_numpy(y_pg_full.flatten()))
            offsets.append(offsets[-1] + ys[-1].numel())
        if self.use_per_graph_target:
            v_trans = _apply_target_transform(
                np.array([self._pergraph_target_raw(d_raw)]),
                self.per_graph_transform,
                fit_state=getattr(self, "_pergraph_target_state", None),
            )  # (1,)
            baseline_t = self._pergraph_baseline_transformed(d_raw)
            v_residual = (v_trans - baseline_t).reshape(-1, 1)
            y_pgr = self._scaler_pergraph_y.transform(v_residual).astype(np.float32)
            pg_block = [torch.from_numpy(y_pgr).flatten()]
            # Append aux multi-task targets (each scaled by its own scaler in
            # transformed space). They are stored INSIDE the per-graph head's
            # output block so HydraGNN's head dispatcher routes them to the
            # graph-level head (output_dim=1+n_aux).
            if self.aux_pergraph_targets:
                bucket_id = int(d_raw.size_bucket[0])
                for spec, state, scaler in zip(
                    self.aux_pergraph_targets,
                    self._aux_pergraph_target_states,
                    self._scaler_aux_pergraph_y,
                ):
                    raw = self._pergraph_target_raw_by_source(d_raw, spec["source"])
                    raw_x = raw * float(spec.get("scale", 1.0))
                    transform = spec.get("transform", "identity")
                    kwargs = {"fit_state": state}
                    if transform == "zscore_per_bucket":
                        kwargs["bucket_ids"] = np.array([bucket_id], dtype=np.int64)
                    trans = _apply_target_transform(
                        np.array([raw_x]),
                        transform,
                        **kwargs,
                    )
                    scaled = scaler.transform(trans.reshape(-1, 1)).astype(np.float32)
                    # Optional `loss_weight`: multiplies the standardized target
                    # by k. Since the StandardScaler-output has unit variance,
                    # scaling by k makes the post-zscore target have std=k, so
                    # mean-squared-error on this column gets weight k^2 in the
                    # head's averaged-over-output-dims MSE. Used by Stage-1.B5
                    # of the β-learnability study to amplify the β-MSE without
                    # changing HydraGNN's internal head loss aggregation.
                    lw = float(spec.get("loss_weight", 1.0))
                    if lw != 1.0:
                        scaled = scaled * lw
                    pg_block.append(torch.from_numpy(scaled).flatten())
            # Phase L6 heteroscedastic per-graph head: 4 zero-padded σ slots
            # at the end. Unsupervised nuisance parameters of `het_nll_pg`.
            if bool(getattr(self, "use_heteroscedastic_pergraph_max", False)):
                pg_block.append(torch.zeros(4, dtype=torch.float32))
            pg_concat = torch.cat(pg_block, dim=0)  # (1 + n_aux [+ 4 σ if pg het],)
            ys.append(pg_concat)
            offsets.append(offsets[-1] + pg_concat.numel())
        y = torch.cat(ys) if ys else torch.zeros((0,), dtype=torch.float32)
        y_loc = torch.tensor([offsets], dtype=torch.long)  # (1, num_heads + 1)

        d = Data(
            x=torch.from_numpy(x_sc),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_loc=y_loc,
            pos=d_raw.pos,
        )
        d.size_bucket = d_raw.size_bucket
        d.L_SVE_um = d_raw.L_SVE_um
        # Per-grain aux NaN mask (N, n_aux) — propagated from _load so eval
        # can EXCLUDE NaN-source grains from per-channel R^2 + parity plots.
        if self.per_grain_aux_targets and hasattr(d_raw, "per_grain_aux_nan_mask"):
            d.per_grain_aux_nan_mask = d_raw.per_grain_aux_nan_mask
        # Carry per-graph target in raw FIP units for eval (un-transformed).
        # Honors `per_graph_target_aggr` so eval metrics match the trained target.
        d.y_pergraph_raw = torch.tensor(
            [self._pergraph_target_raw(d_raw)], dtype=torch.float32
        )
        # Aux multi-task targets in raw units (one scalar per aux target),
        # for per-target eval metrics.
        if self.aux_pergraph_targets:
            d.y_pergraph_aux_raw = torch.tensor(
                [
                    self._pergraph_target_raw_by_source(d_raw, spec["source"])
                    for spec in self.aux_pergraph_targets
                ],
                dtype=torch.float32,
            )
        # Per-graph residual baseline (transformed-target space) — needed at
        # inversion time when residualization is active. Stored even when the
        # baseline is 0.0 so the eval code can be uniform.
        d.y_baseline_t = torch.tensor(
            [self._pergraph_baseline_transformed(d_raw)], dtype=torch.float32
        )

        # graph_attr (one row per graph, batched as (N_graphs, dim)).
        if self.graph_attr_source is not None:
            ga_raw = self._graph_attr_raw(d_raw).reshape(1, -1)
            if self._scaler_graph_attr is not None:
                ga = self._scaler_graph_attr.transform(ga_raw).astype(np.float32)
            else:
                ga = ga_raw.astype(np.float32)
            d.graph_attr = torch.from_numpy(ga)

        return d


def smoke_test():
    """Quick check: load + split + scale + verify shapes."""
    ds = FIPDataset(size_bucket=2, seed=42)
    train, val, test = ds.build()
    print(f"smoke: train={len(train)} val={len(val)} test={len(test)}")
    print(
        f"  n_input_features={ds.n_input_features}  n_edge_features={ds.n_edge_features}"
    )
    g = train[0]
    print(
        f"  graph[0]: x={tuple(g.x.shape)}  edge_index={tuple(g.edge_index.shape)}  edge_attr={tuple(g.edge_attr.shape)}  y={tuple(g.y.shape)}"
    )
    print(f"  y_pergraph_raw={g.y_pergraph_raw.item():.3e}")


if __name__ == "__main__":
    smoke_test()
