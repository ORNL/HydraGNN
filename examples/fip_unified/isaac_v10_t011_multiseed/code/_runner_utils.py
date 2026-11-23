"""_runner_utils.py — shared helpers for v4 Track A/L/promote/E runners.

run_cell()                — call train.train_one with a merged config
detect_track_a_winner()   — pick best Track A cell by pgmax-aware score
detect_track_l_winner()   — pick best Track L cell
qualifies_for_promotion() — does this cell beat the promotion thresholds?
load_metrics(), cell_score() — read run_metrics + recompute tail metrics
"""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent  # isaac_v4_study/code/
STUDY = HERE.parent  # isaac_v4_study/
RESULTS = STUDY / "results"
LOGS = STUDY / "logs"

sys.path.insert(0, str(HERE))


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def run_cell(
    tag: str,
    seed: int,
    base: dict,
    extra: dict,
    num_epoch: Optional[int] = None,
    skip_done: bool = True,
) -> Dict[str, Any]:
    """Run one (tag, seed) cell. Returns metrics dict."""
    res_path = RESULTS / f"{tag}_seed{seed}" / "run_metrics.json"
    if skip_done and res_path.exists():
        with open(res_path) as f:
            print(f"[skip] {tag}_seed{seed} already has metrics.json")
            return json.load(f)

    from train import train_one

    overrides = deep_merge(base, extra)
    print(f"[run ] tag={tag} seed={seed}", flush=True)
    metrics = train_one(
        seed=seed, exp_tag=tag, config_overrides=overrides, num_epoch=num_epoch
    )
    metrics.setdefault("exp_tag", tag)
    metrics.setdefault("seed", seed)
    return metrics


def load_metrics(tag: str, seed: int) -> Optional[Dict[str, Any]]:
    p = RESULTS / f"{tag}_seed{seed}" / "run_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def tail_metrics_for_run(
    tag: str, seed: int, *, calibrated: bool = False
) -> Optional[Dict[str, float]]:
    """Compute tail metrics from saved test predictions CSV if present.

    If `calibrated=True`, use `test_pergrain_predictions_cal.csv` (the
    column is `y_pred_cal`); else use the raw `y_pred_raw`.
    """
    if calibrated:
        csv_path = RESULTS / f"{tag}_seed{seed}" / "test_pergrain_predictions_cal.csv"
        pred_col = "y_pred_cal"
    else:
        csv_path = RESULTS / f"{tag}_seed{seed}" / "test_pergrain_predictions.csv"
        pred_col = "y_pred_raw"
    if not csv_path.exists():
        return None
    import csv as _csv
    import numpy as _np
    from tail_metrics import compute_tail_metrics

    y_true, y_pred, gi = [], [], []
    with open(csv_path) as f:
        r = _csv.DictReader(f)
        for row in r:
            y_true.append(float(row["y_true_raw"]))
            y_pred.append(float(row[pred_col]))
            gi.append(int(row["graph_idx"]))
    return compute_tail_metrics(
        _np.asarray(y_true),
        _np.asarray(y_pred),
        graph_idx=_np.asarray(gi, dtype=_np.int64),
    )


def cell_score(tag: str, seed: int, *, calibrated: bool = False) -> Dict[str, float]:
    """Combined headline + tail scoreboard for one cell.

    `calibrated=True` reads the calibrated CSV (post-quantile-mapping).

    2026-05-04: switched from pn_R²-centric ranking to a composite that
    weights per-graph head R² + within-bucket pg_R² + top10_slope. pn_R²
    is still reported but NOT optimised (per user feedback).
    """
    m = load_metrics(tag, seed) or {}
    out = {
        "tag": tag,
        "seed": seed,
        "pn_r2": float(m.get("test", {}).get("pergrain", {}).get("r2", float("nan"))),
        "pn_pearson": float(
            m.get("test", {}).get("pergrain", {}).get("pearson", float("nan"))
        ),
        "pn_spear": float(
            m.get("test", {}).get("pergrain", {}).get("spearman", float("nan"))
        ),
        "within_g_spear": float(
            m.get("test", {}).get("pergrain_spearman_per_graph_mean", float("nan"))
        ),
        # Per-graph HEAD direct R² (operationally important; the new headline)
        "pg_head_r2": float(
            m.get("test", {}).get("pergraph", {}).get("r2", float("nan"))
        ),
        "pg_head_pearson": float(
            m.get("test", {}).get("pergraph", {}).get("pearson", float("nan"))
        ),
    }
    tm = tail_metrics_for_run(tag, seed, calibrated=calibrated) or {}
    out["top10_r2"] = float(tm.get("top10_r2", float("nan")))
    out["top10_slope"] = float(tm.get("top10_slope", float("nan")))
    out["top10_mean_rel_err"] = float(tm.get("top10_mean_rel_err", float("nan")))
    out["pgmax_r2"] = float(tm.get("per_graph_max_r2", float("nan")))
    out["pgmax_slope"] = float(tm.get("per_graph_max_slope", float("nan")))
    out["pgmax_pearson"] = float(tm.get("per_graph_max_pearson", float("nan")))
    out["pgmax_spear"] = float(tm.get("per_graph_max_spearman", float("nan")))
    out["top1_hit"] = float(tm.get("within_graph_top1_hit", float("nan")))
    out["top10pct_recall"] = float(tm.get("within_graph_top10pct_recall", float("nan")))

    # Min per-bucket pg_R² — the "within-bucket evenness" metric the user
    # wants to optimise. Read from per_size_breakdown.csv.
    out["min_per_bucket_pg_r2"] = _min_per_bucket_pg_r2(tag, seed)
    return out


def _min_per_bucket_pg_r2(tag: str, seed: int) -> float:
    """Read per_size_breakdown.csv and return min pg_R² across L_SVE buckets
    on the test split. NaN if file missing or no admissible buckets."""
    p = RESULTS / f"{tag}_seed{seed}" / "per_size_breakdown.csv"
    if not p.exists():
        return float("nan")
    import csv as _csv

    r2s = []
    with open(p) as f:
        for row in _csv.DictReader(f):
            if row.get("level") == "pergraph" and row.get("split") == "test":
                try:
                    r = float(row.get("r2", "nan"))
                    if r == r:  # not NaN
                        r2s.append(r)
                except Exception:
                    pass
    return min(r2s) if r2s else float("nan")


def composite_score(s: Dict[str, float]) -> float:
    """Per-cell screening composite (per 2026-05-04 user pivot):

        0.5 * pg_head_r2 + 0.3 * min_per_bucket_pg_r2 + 0.2 * top10_slope

    No pn_R² admissibility floor. Returns NaN if the headline metric
    (pg_head_r2) is missing.
    """
    pg = s.get("pg_head_r2", float("nan"))
    mb = s.get("min_per_bucket_pg_r2", float("nan"))
    t10 = s.get("top10_slope", float("nan"))
    if pg != pg:
        return float("nan")
    if mb != mb:  # NaN: no per-bucket breakdown available
        mb = pg  # fall back to overall pg
    if t10 != t10:
        t10 = 0.0
    return 0.5 * pg + 0.3 * mb + 0.2 * t10


# ---------------------------------------------------------------------------
# Winner-detection helpers (pgmax-aware: v4 cares about per-graph max R²)
# ---------------------------------------------------------------------------
def _select_winner(track_cells, key="pgmax_pearson", default=None):
    """Pick the cell with the highest `key`. Falls back to `default` if no
    cells have completed (or `key` is NaN everywhere)."""
    candidates = []
    for c in track_cells:
        s = cell_score(c["tag"], 42)
        v = s.get(key, float("nan"))
        if v == v:  # not NaN
            candidates.append((v, s))
    if not candidates:
        return default
    candidates.sort(key=lambda cv: -cv[0])
    return candidates[0][1]["tag"]


def detect_track_a_winner(default: str = "A4_pg_mech") -> str:
    """Pick Track-A winner by per-graph max pearson (primary structural lift).
    Falls back to A4_pg_mech (memory says mech_compact gives largest lift)."""
    from manifest import TRACK_A

    return _select_winner(TRACK_A, key="pgmax_pearson", default=default) or default


def detect_track_l_winner(default: str = "L0_mse") -> str:
    """Pick Track-L winner by top10_slope (primary tail metric)."""
    from manifest import TRACK_L

    return _select_winner(TRACK_L, key="top10_slope", default=default) or default


def qualifies_for_promotion(tag: str, seed: int) -> bool:
    """Per `manifest.PROMOTE_THRESHOLDS`."""
    from manifest import PROMOTE_THRESHOLDS

    s = cell_score(tag, seed)
    return (
        s["top10_slope"] == s["top10_slope"]
        and s["top10_slope"] >= PROMOTE_THRESHOLDS["top10_slope_min"]
        and s["pn_r2"] >= PROMOTE_THRESHOLDS["pn_r2_min"]
        and s.get("pgmax_pearson", float("nan"))
        >= PROMOTE_THRESHOLDS["pgmax_pearson_min"]
    )
