"""calibration.py — post-hoc quantile-mapping calibration for FIP predictions.

Why this exists.  The v3 ISAAC run produced models with pn_R²≈0.62 and
pn_spearman≈0.79 — usable rank quality — but per-graph max R² ≈ −3.6 because
MSE shrinks predictions toward the mean (std_pred = 0.79·std_true) and the
order statistic of shrunken predictions has 4× compressed variance plus a
17 % systematic bias on per-graph max.

A val-fit isotonic-style quantile mapping on the *grain-level* prediction
distribution recovers most of the lost variance:

  v3 F1 raw  : pn_R²=0.623  top10_slope=0.156  pgmax_R²=−3.45
  v3 F1 cal  : pn_R²=0.564  top10_slope=0.282  pgmax_R²=+0.15
  pred std   : 0.00120 → 0.00136 (matches true 0.00151)
  pgmax std  : 0.00024 → 0.00106 (matches true 0.00101)

The map is non-parametric (rank percentile → val-true at that percentile)
and preserves rank, so within-graph top-1 hit and Spearman are unchanged.

Two flavors are exposed:

  - `fit_global(val_pred, val_true)` — single map for all grains.
  - `fit_per_bucket(val_pred, val_true, bucket)` — separate map per
    SVE-size bucket. Slightly better when bucket-conditional means differ.

Both return an `apply(pred, bucket=None)` callable. The runner saves the
calibration parameters next to `run_metrics.json` so analysis is fully
reproducible.

There is no leakage: the map is fit on val only and applied to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Quantile mapping primitives
# ---------------------------------------------------------------------------
def _build_q_map(
    val_pred: np.ndarray, val_true: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build f such that f(test_pred) ≈ val_true at the same rank percentile."""
    val_pred = np.asarray(val_pred, dtype=np.float64).ravel()
    val_true = np.asarray(val_true, dtype=np.float64).ravel()
    pred_sorted = np.sort(val_pred)
    true_sorted = np.sort(val_true)
    n = len(pred_sorted)

    def apply(test_pred: np.ndarray) -> np.ndarray:
        test_pred = np.asarray(test_pred, dtype=np.float64).ravel()
        # rank in val_pred (left-side) divided by n → percentile in [0, 1]
        ranks = np.searchsorted(pred_sorted, test_pred, side="left") / max(1, n)
        ranks = np.clip(ranks, 0.0, 1.0)
        # interpolate val_true at those percentiles
        true_pcts = np.linspace(0.0, 1.0, n)
        return np.interp(ranks, true_pcts, true_sorted)

    return apply


# ---------------------------------------------------------------------------
# Calibration container
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """Holds either a global quantile map or one map per bucket."""

    mode: str  # "global" | "per_bucket"
    pred_sorted: np.ndarray  # used by global
    true_sorted: np.ndarray
    per_bucket_pred_sorted: Dict[int, np.ndarray] = field(default_factory=dict)
    per_bucket_true_sorted: Dict[int, np.ndarray] = field(default_factory=dict)

    def apply(
        self, pred: np.ndarray, bucket: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float64).ravel()
        if self.mode == "global" or bucket is None or not self.per_bucket_pred_sorted:
            return _interp_map(pred, self.pred_sorted, self.true_sorted)
        bucket = np.asarray(bucket).ravel()
        out = np.empty_like(pred)
        # Per-bucket apply with global fallback for unseen buckets.
        seen = set(self.per_bucket_pred_sorted.keys())
        for b in np.unique(bucket):
            m = bucket == b
            if int(b) in seen:
                ps = self.per_bucket_pred_sorted[int(b)]
                ts = self.per_bucket_true_sorted[int(b)]
                out[m] = _interp_map(pred[m], ps, ts)
            else:
                out[m] = _interp_map(pred[m], self.pred_sorted, self.true_sorted)
        return out

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pred_sorted": self.pred_sorted.tolist(),
            "true_sorted": self.true_sorted.tolist(),
            "per_bucket_pred_sorted": {
                int(k): v.tolist() for k, v in self.per_bucket_pred_sorted.items()
            },
            "per_bucket_true_sorted": {
                int(k): v.tolist() for k, v in self.per_bucket_true_sorted.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            mode=d["mode"],
            pred_sorted=np.asarray(d["pred_sorted"], dtype=np.float64),
            true_sorted=np.asarray(d["true_sorted"], dtype=np.float64),
            per_bucket_pred_sorted={
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in d.get("per_bucket_pred_sorted", {}).items()
            },
            per_bucket_true_sorted={
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in d.get("per_bucket_true_sorted", {}).items()
            },
        )


def _interp_map(
    pred: np.ndarray, pred_sorted: np.ndarray, true_sorted: np.ndarray
) -> np.ndarray:
    n = len(pred_sorted)
    if n == 0:
        return pred
    ranks = np.searchsorted(pred_sorted, pred, side="left") / max(1, n)
    ranks = np.clip(ranks, 0.0, 1.0)
    true_pcts = np.linspace(0.0, 1.0, n)
    return np.interp(ranks, true_pcts, true_sorted)


# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------
def fit_global(val_pred: np.ndarray, val_true: np.ndarray) -> Calibration:
    """Fit one quantile map across all grains."""
    return Calibration(
        mode="global",
        pred_sorted=np.sort(np.asarray(val_pred, dtype=np.float64).ravel()),
        true_sorted=np.sort(np.asarray(val_true, dtype=np.float64).ravel()),
    )


def fit_per_bucket(
    val_pred: np.ndarray,
    val_true: np.ndarray,
    val_bucket: np.ndarray,
    min_per_bucket: int = 50,
) -> Calibration:
    """Fit one quantile map per SVE size bucket, with a global fallback for
    any bucket that has < min_per_bucket val grains."""
    val_pred = np.asarray(val_pred, dtype=np.float64).ravel()
    val_true = np.asarray(val_true, dtype=np.float64).ravel()
    val_bucket = np.asarray(val_bucket).ravel()
    cal = Calibration(
        mode="per_bucket",
        pred_sorted=np.sort(val_pred),
        true_sorted=np.sort(val_true),
    )
    for b in np.unique(val_bucket):
        m = val_bucket == b
        if m.sum() < min_per_bucket:
            continue
        cal.per_bucket_pred_sorted[int(b)] = np.sort(val_pred[m])
        cal.per_bucket_true_sorted[int(b)] = np.sort(val_true[m])
    return cal


# ---------------------------------------------------------------------------
# CLI helper for offline post-hoc rescore
# ---------------------------------------------------------------------------
def calibrate_run_dir(run_dir: str, mode: str = "per_bucket") -> Dict[str, float]:
    """Per-grain calibration: fit on val_pergrain, apply to test_pergrain.
    Writes `test_pergrain_predictions_cal.csv` and `calibration.json`.
    Returns the new test-set per-grain tail metrics."""
    import csv
    from pathlib import Path
    import json
    from tail_metrics import compute_tail_metrics

    run = Path(run_dir)
    # Two formats supported: v9 wide (`pred_<split>_pn.csv` with cell-tagged
    # cols) and legacy long (`<split>_pergrain_predictions.csv`). v9 runs
    # write the wide form; v5/v6/v7/v8/L1-L3 dirs use the legacy form.
    val_csv_wide = run / "pred_val_pn.csv"
    test_csv_wide = run / "pred_test_pn.csv"
    val_csv_legacy = run / "val_pergrain_predictions.csv"
    test_csv_legacy = run / "test_pergrain_predictions.csv"
    use_wide = val_csv_wide.exists() and test_csv_wide.exists()
    if not use_wide and not (val_csv_legacy.exists() and test_csv_legacy.exists()):
        raise FileNotFoundError(f"missing predictions CSV in {run_dir}")

    def _load_legacy(p):
        rows = list(csv.DictReader(open(p)))
        return (
            np.asarray([float(r["y_true_raw"]) for r in rows]),
            np.asarray([float(r["y_pred_raw"]) for r in rows]),
            np.asarray([int(r["graph_idx"]) for r in rows]),
            np.asarray([int(r["size_bucket"]) for r in rows]),
        )

    def _load_wide(p):
        # Wide format: slot-0 LCF/FS lives in `true_lcf_fs_max`/`pred_lcf_fs_max`.
        rows = list(csv.DictReader(open(p)))
        rows = [r for r in rows if r.get("true_lcf_fs_max") not in (None, "")]
        return (
            np.asarray([float(r["true_lcf_fs_max"]) for r in rows]),
            np.asarray([float(r["pred_lcf_fs_max"]) for r in rows]),
            np.asarray([int(r["graph_idx"]) for r in rows]),
            np.asarray([int(r["size_bucket"]) for r in rows]),
        )

    if use_wide:
        vy, vp, _, vb = _load_wide(val_csv_wide)
        ty, tp, tg, tb = _load_wide(test_csv_wide)
    else:
        vy, vp, _, vb = _load_legacy(val_csv_legacy)
        ty, tp, tg, tb = _load_legacy(test_csv_legacy)

    if mode == "global":
        cal = fit_global(vp, vy)
    else:
        cal = fit_per_bucket(vp, vy, vb)
    tp_cal = cal.apply(tp, bucket=tb)

    json.dump(cal.to_dict(), open(run / "calibration.json", "w"))
    with open(run / "test_pergrain_predictions_cal.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(
            ["graph_idx", "size_bucket", "y_true_raw", "y_pred_raw", "y_pred_cal"]
        )
        for g, b, yt, yp, yc in zip(tg, tb, ty, tp, tp_cal):
            w.writerow([int(g), int(b), float(yt), float(yp), float(yc)])

    return compute_tail_metrics(ty, tp_cal, graph_idx=tg)


# ---------------------------------------------------------------------------
# σ_pop_within voxel-level prediction intervals (Phase D1b)
# ---------------------------------------------------------------------------
# σ_pop_within is a population dispersion estimator pooled across training-set
# grains in log space. For a grain whose predicted log-mean is μ̂, the voxel-
# level FIP distribution is approximated as Lognormal(μ̂, σ_pop_within²); the
# z-coverage interval at z (90 % nominal at z=1.65) is
#   FIP_voxel ∈ [exp(μ̂ − z·σ_pop_within), exp(μ̂ + z·σ_pop_within)]
# Most v9 surrogates predict ARITHMETIC voxel-mean rather than log-mean. The
# Jensen correction `μ̂_log ≈ ln(μ̂_lin) − σ²/2` recovers the equivalent log-
# mean — bias σ²/2 ≈ 0.022 for σ ≈ 0.21, well below model R² noise.
#
# Per-cell σ_pop_within (log space) computed in
# `results/D1b_sigma_pop_within/sigma_pop_within.json`:
#   LCF/FS = 0.1701   LCF/WP = 0.1603
#   HCF/FS = 0.2333   HCF/WP = 0.2325
# Per-bucket variation is ≤ 3 %, so the global value is a safe default; the
# per-bucket map is also exported for comparison if voxel-Pcov falls short.

# Default frozen σ_pop_within values (log space; Phase D1b, 2026-05-06).
_SIGMA_POP_WITHIN_DEFAULTS_LOG = {
    "lcf_fs": 0.1701,
    "lcf_wp": 0.1603,
    "hcf_fs": 0.2333,
    "hcf_wp": 0.2325,
}


def apply_sigma_pop_within(
    predicted_mean: np.ndarray,
    sigma_pop_within: float,
    *,
    z: float = 1.65,
    shape: str = "lognormal",
    mean_space: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-level prediction interval given the surrogate's per-grain mean.

    Parameters
    ----------
    predicted_mean
        Surrogate's per-grain output. Either arithmetic mean (default,
        `mean_space="linear"`) or log-mean (`mean_space="log"`).
    sigma_pop_within
        Population dispersion in log space (D1b output column).
    z
        Coverage half-width. 1.65 ≈ 90 % nominal coverage of the lognormal.
    shape
        "lognormal" (D1a winner, 12/20 cells) or "normal" (a fallback for
        cells where normal won D1a — currently 5/20, mostly LCF/FS).
    mean_space
        "linear" (default) applies Jensen correction to recover log-mean.
        "log" assumes `predicted_mean` is already log-space.

    Returns
    -------
    lower, upper : np.ndarray
        Per-grain (lower, upper) voxel-FIP prediction interval bounds in
        LINEAR space (i.e. real FIP values, not log values).
    """
    pm = np.asarray(predicted_mean, dtype=np.float64)
    s = float(sigma_pop_within)
    if shape == "lognormal":
        if mean_space == "linear":
            # Jensen: μ_log = ln(μ_lin) − σ²/2 (only valid for μ_lin > 0).
            with np.errstate(invalid="ignore", divide="ignore"):
                mu_log = np.log(np.clip(pm, 1e-30, None)) - 0.5 * s * s
        else:
            mu_log = pm
        lower = np.exp(mu_log - z * s)
        upper = np.exp(mu_log + z * s)
    elif shape == "normal":
        # Linear-space normal PI; σ_pop_within taken as linear std equivalent.
        if mean_space == "log":
            raise ValueError("normal shape with mean_space='log' is incoherent")
        lower = pm - z * s
        upper = pm + z * s
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return lower, upper


def get_sigma_pop_within(
    cell: str,
    *,
    sigma_json_path: Optional[str] = None,
    bucket: Optional[int] = None,
) -> float:
    """Lookup σ_pop_within for a cell.

    If `sigma_json_path` is None, return the frozen 2026-05-06 D1b global
    defaults (most callers want this). If a path is given, it is read on
    every call (cheap; a few KB of JSON), which lets analysis scripts swap
    in re-derived values without touching the calibration pipeline.

    `bucket` (0–4) returns the per-bucket value if json_path is supplied.
    The frozen defaults are global only.
    """
    cell = cell.lower().replace("/", "_")
    if sigma_json_path is None:
        if cell not in _SIGMA_POP_WITHIN_DEFAULTS_LOG:
            raise KeyError(
                f"unknown cell {cell!r}; expected one of "
                f"{list(_SIGMA_POP_WITHIN_DEFAULTS_LOG)}"
            )
        return _SIGMA_POP_WITHIN_DEFAULTS_LOG[cell]
    import json as _json

    blob = _json.load(open(sigma_json_path))
    cell_block = blob["sigma_pop_within"].get(cell.upper().replace("_", "/"))
    if cell_block is None:
        raise KeyError(f"cell {cell!r} not in {sigma_json_path}")
    if bucket is None:
        return float(cell_block["log_space"]["mean"])
    return float(cell_block["per_bucket"][str(int(bucket))]["log_space_mean"])


def calibrate_run_dir_pergraph(run_dir: str, mode: str = "per_bucket") -> bool:
    """Per-graph calibration: fit on val_predictions, apply to test_predictions.
    Writes `test_predictions_cal.csv` and `calibration_pergraph.json`.
    Per-graph predictions only exist when `use_per_graph_target=True`; returns
    False (no-op) otherwise."""
    import csv
    from pathlib import Path
    import json

    run = Path(run_dir)
    # v9 wide vs legacy long format detection (mirrors per-grain calibrator).
    val_csv_wide = run / "pred_val_pg.csv"
    test_csv_wide = run / "pred_test_pg.csv"
    val_csv_legacy = run / "val_predictions.csv"
    test_csv_legacy = run / "test_predictions.csv"
    use_wide = val_csv_wide.exists() and test_csv_wide.exists()
    if not use_wide and not (val_csv_legacy.exists() and test_csv_legacy.exists()):
        return False

    def _load_legacy(p):
        rows = list(csv.DictReader(open(p)))
        return (
            np.asarray([float(r["y_true_raw"]) for r in rows]),
            np.asarray([float(r["y_pred_raw"]) for r in rows]),
            np.asarray([int(r["size_bucket"]) for r in rows]),
        )

    def _load_wide(p):
        # Wide format: slot-0 LCF/FS pg-MAX lives in
        # `true_lcf_fs_max`/`pred_lcf_fs_max`.
        rows = list(csv.DictReader(open(p)))
        rows = [r for r in rows if r.get("true_lcf_fs_max") not in (None, "")]
        return (
            np.asarray([float(r["true_lcf_fs_max"]) for r in rows]),
            np.asarray([float(r["pred_lcf_fs_max"]) for r in rows]),
            np.asarray([int(r["size_bucket"]) for r in rows]),
        )

    if use_wide:
        vy, vp, vb = _load_wide(val_csv_wide)
        ty, tp, tb = _load_wide(test_csv_wide)
    else:
        vy, vp, vb = _load_legacy(val_csv_legacy)
        ty, tp, tb = _load_legacy(test_csv_legacy)

    if mode == "global" or len(np.unique(vb)) < 2:
        cal = fit_global(vp, vy)
    else:
        # Per-bucket can be thin at per-graph level (50 val graphs split across
        # 5 buckets ≈ 10 per bucket); the helper auto-falls-back to global for
        # any bucket with < min_per_bucket samples.
        cal = fit_per_bucket(vp, vy, vb, min_per_bucket=8)
    tp_cal = cal.apply(tp, bucket=tb)

    json.dump(cal.to_dict(), open(run / "calibration_pergraph.json", "w"))
    with open(run / "test_predictions_cal.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["size_bucket", "y_true_raw", "y_pred_raw", "y_pred_cal"])
        for b, yt, yp, yc in zip(tb, ty, tp, tp_cal):
            w.writerow([int(b), float(yt), float(yp), float(yc)])
    return True
