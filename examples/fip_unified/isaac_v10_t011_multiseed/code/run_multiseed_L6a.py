"""run_multiseed_L6a.py — multi-init-seed run of L6a (per-graph heteroscedastic).

Runs the locked v9b config (het_nll_pg recipe + shallow head + residual +
max-pool + 4 σ outputs on per-graph head) at 5 init seeds. Each cell uses
the SAME split (split_seed=42) so the only thing changing is the model
init randomness — gives ± error bars on every metric.

5 seeds × ~10 min ≈ 50 min total. Uses the deslotted train.py so each
run dir natively has per-cell hotspot metrics (no post-hoc backfill).

Usage:
    python run_multiseed_L6a.py
    python run_multiseed_L6a.py --seeds 42,7,99
    python run_multiseed_L6a.py --post                # aggregate without re-running
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))


DEFAULT_SEEDS = [42, 7, 99, 137, 2024]
TAG_TEMPLATE = "L6a_het_nll_pg_multiseed"

CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
SHALLOW_NODE_HEAD = {"num_headlayers": 1, "dim_headlayers": [8]}


def _build_extras(base_extra: Dict[str, Any]) -> Dict[str, Any]:
    from manifest import _V9_PERGRAIN_TARGETS_8, _V9_PERGRAPH_TARGETS_14

    extras = copy.deepcopy(base_extra)
    extras["Recipe"]["loss_recipe"] = "het_nll_pg"
    extras["Recipe"]["node_head_dims"] = dict(SHALLOW_NODE_HEAD)
    extras["Recipe"]["use_residual_node_head"] = True
    extras["Recipe"]["per_grain_targets"] = list(_V9_PERGRAIN_TARGETS_8)
    extras["Recipe"]["pergraph_targets"] = list(_V9_PERGRAPH_TARGETS_14)
    extras["Recipe"].pop("per_grain_target_field", None)
    extras["Recipe"].pop("per_grain_target_col", None)
    extras["Recipe"].pop("per_grain_aux_targets", None)
    extras["Recipe"].pop("aux_pergraph_targets", None)
    extras["Recipe"]["use_heteroscedastic_max"] = False
    extras["Recipe"]["use_heteroscedastic_pergraph_max"] = True
    # Lock split_seed so all 5 init seeds use the same train/val/test split.
    extras["Recipe"]["split_seed"] = 42
    extras.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    extras["NeuralNetwork"]["Architecture"]["graph_pooling"] = "max"
    return extras


def _row_for_run(run_dir: Path, init_seed: int) -> Dict[str, Any]:
    rp = run_dir / "run_metrics.json"
    row: Dict[str, Any] = {"init_seed": init_seed, "run_dir": run_dir.name}
    if not rp.exists():
        return row
    m = json.load(open(rp))
    row["train_time_s"] = float(m.get("elapsed_sec", float("nan")))
    test = m.get("test") or {}
    # 4-cell mean hotspot metrics (native after deslot).
    for k in (
        "pergrain_top1_hit_rate_mean_4cell",
        "pergrain_top10_recall_mean_4cell",
        "pergrain_spearman_per_graph_mean_4cell",
        "pergrain_hotspot_loc_err_mean_4cell",
    ):
        row[k] = float(test.get(k, float("nan")))
    # Per-cell hotspot from native deslot output if present.
    pch = test.get("pergrain_per_cell_hotspot") or {}
    for cell in ("lcf_fs_max", "lcf_wp_max", "hcf_fs_max", "hcf_wp_max"):
        ent = pch.get(cell, {})
        row[f"{cell}_top1"] = float(ent.get("top1_hit_rate", float("nan")))
        row[f"{cell}_within_spearman"] = float(
            ent.get("spearman_per_graph_mean", float("nan"))
        )
        row[f"{cell}_loc_err"] = float(ent.get("hotspot_loc_err", float("nan")))
    # σ output stats per cell (val).
    val = m.get("val") or {}
    sig_stats = val.get("pergraph_logsigma_stats") or {}
    for cell, st in sig_stats.items():
        row[f"sigma_{cell}_mean_val"] = float(st.get("mean", float("nan")))
        row[f"sigma_{cell}_std_val"] = float(st.get("std", float("nan")))
    # Pull pg σ-residual Spearman + Pcov from wide test pg CSV (post-hoc compute).
    p = run_dir / "pred_test_pg.csv"
    if p.exists():
        import csv as _csv
        from sklearn.metrics import r2_score
        from scipy.stats import spearmanr

        rows = list(_csv.DictReader(open(p)))
        for cell in CELL_KEYS:
            col_t = f"true_{cell}_max"
            col_p = f"pred_{cell}_max"
            col_s = f"logsigma_{cell}_max_pergraph"
            rs = [r for r in rows if r.get(col_t) not in (None, "")]
            if not rs:
                continue
            yt = np.array([float(r[col_t]) for r in rs])
            yp = np.array([float(r[col_p]) for r in rs])
            finite = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[finite], yp[finite]
            if len(yt) < 5:
                continue
            row[f"{cell}_bulk_r2_pg"] = float(r2_score(yt, yp))
            if rs[0].get(col_s) not in (None, ""):
                sig = np.array(
                    [
                        float(r[col_s]) if r[col_s] not in (None, "") else float("nan")
                        for r in rs
                    ]
                )[finite]
                resid = np.abs(yp - yt)
                try:
                    cor, _ = spearmanr(sig, resid)
                    row[f"{cell}_sigma_residual_spearman_pg"] = (
                        float(cor) if cor == cor else float("nan")
                    )
                except Exception:
                    pass
                # Pcov via local Jacobian σ_raw approx.
                sigma_raw = np.exp(sig) * yt.std()
                row[f"{cell}_pcov_z1.65_pg"] = float(
                    ((yp - yt).__abs__() <= 1.65 * sigma_raw).mean()
                )
    return row


def _print_aggregate(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("[multiseed] no rows")
        return
    print()
    print(
        f"v9b L6a multi-seed aggregate ({len(rows)} reps, init_seeds = {[r['init_seed'] for r in rows]})"
    )
    print("=" * 130)
    metrics = [
        ("mean_4cell_top1", "pergrain_top1_hit_rate_mean_4cell"),
        ("mean_4cell_top10", "pergrain_top10_recall_mean_4cell"),
        ("mean_4cell_within_sp", "pergrain_spearman_per_graph_mean_4cell"),
        ("mean_4cell_loc_err", "pergrain_hotspot_loc_err_mean_4cell"),
    ]
    for label, key in metrics:
        vals = np.array([r.get(key, float("nan")) for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            print(
                f"  {label:<28s}  {finite.mean():>+8.3f} ± {finite.std(ddof=1) if finite.size > 1 else 0:.3f}   "
                f"(per-rep: {', '.join(f'{v:+.3f}' for v in vals)})"
            )

    print()
    print("Per-cell pg σ-residual Spearman (test):")
    for cell in CELL_KEYS:
        vals = np.array(
            [r.get(f"{cell}_sigma_residual_spearman_pg", float("nan")) for r in rows]
        )
        f = vals[np.isfinite(vals)]
        if f.size:
            print(
                f"  {cell:<8s}  {f.mean():>+7.3f} ± {f.std(ddof=1) if f.size > 1 else 0:.3f}   "
                f"(per-rep: {', '.join(f'{v:+.3f}' for v in vals)})"
            )
    print()
    print("Per-cell pg Pcov at z=1.65 (test):")
    for cell in CELL_KEYS:
        vals = np.array([r.get(f"{cell}_pcov_z1.65_pg", float("nan")) for r in rows])
        f = vals[np.isfinite(vals)]
        if f.size:
            print(
                f"  {cell:<8s}  {f.mean():>7.3f} ± {f.std(ddof=1) if f.size > 1 else 0:.3f}   "
                f"(per-rep: {', '.join(f'{v:.3f}' for v in vals)})"
            )
    print()
    print("Per-cell test top-1 hit rate (4-cell symmetric, post-deslot):")
    for cell in CELL_KEYS:
        key = f"{cell}_max_top1"
        vals = np.array([r.get(key, float("nan")) for r in rows])
        f = vals[np.isfinite(vals)]
        if f.size:
            print(
                f"  {cell:<8s}  {f.mean():>7.3f} ± {f.std(ddof=1) if f.size > 1 else 0:.3f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS)
    )
    ap.add_argument("--num_epoch", type=int, default=400)
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.post:
        from manifest import BASE_RECIPE, TRACK_V7
        from _runner_utils import run_cell

        base_extra = TRACK_V7[0]["extra"]
        for s in seeds:
            tag = f"L6a_het_nll_pg_init{s}"
            res_path = STUDY / "results" / f"{tag}_seed{s}" / "run_metrics.json"
            if res_path.exists():
                print(f"[multiseed] {tag} already has metrics — skip", flush=True)
                continue
            extras = _build_extras(base_extra)
            print(f"[multiseed] init_seed={s} starting", flush=True)
            t0 = time.time()
            try:
                run_cell(
                    tag,
                    seed=s,
                    base=BASE_RECIPE,
                    extra=extras,
                    num_epoch=args.num_epoch,
                    skip_done=True,
                )
                print(
                    f"[multiseed] init_seed={s} done ({time.time()-t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"[multiseed] init_seed={s} FAILED: {e}", flush=True)
                traceback.print_exc()

    out_dir = STUDY / "results" / "L6a_multiseed_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for s in seeds:
        tag = f"L6a_het_nll_pg_init{s}"
        rd = STUDY / "results" / f"{tag}_seed{s}"
        rows.append(_row_for_run(rd, init_seed=s))
    if rows:
        (out_dir / "aggregate.json").write_text(
            json.dumps(rows, indent=2, default=float)
        )
    _print_aggregate(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
