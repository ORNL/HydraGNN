"""run_track_M_distshape.py — Phase M distribution-shape per-grain targets.

Tests whether per-grain distribution-shape targets (lognormal_μ, lognormal_σ,
p90, p99) are LEARNABLE from microstructure features at the v9b base
config (L6b2 het_nll_both_50_15). Three proposals:

  M_A_distA  het_nll_both_50_15_distA   ← +4 lognormal_σ
  M_B_distB  het_nll_both_50_15_distB   ← +4 lognormal_μ + 4 lognormal_σ
  M_C_distC  het_nll_both_50_15_distC   ← +4 p90 + 4 p99 (quantile-style)

Decision criterion: per-cell R² on the new targets. R² > 0.50 ⇒ usable;
> 0.70 ⇒ strong (worth promoting to v9b base for §4.2 voxel-level
feature-conditional UQ).

Why this might work where v7 σ-aux failed (R² < 0.35):
  - Better features (ENRICHED_39 with voxel_geom_6 + schmid12 + sve_boundary)
  - Better architecture (SAGE + max-pool + residual node head + L6b2 base)
  - Multi-task synergy with related distributional targets
  - Heteroscedastic NLL on max channels does NOT compete for the σ_log
    target (σ_log is a separate supervised regression, σ_pn is a nuisance
    parameter on the MAX channel).

Usage:
  python run_track_M_distshape.py                # train all 3 proposals
  python run_track_M_distshape.py --post         # aggregate only
  python run_track_M_distshape.py --cells_only M_A
"""
from __future__ import annotations
import argparse
import copy
import csv
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


# Each cell: (tag, recipe_name, target_list_attr_name, n_extra)
CELLS = [
    (
        "M_A_logsigma",
        "het_nll_both_50_15_distA",
        "_V9b_PERGRAIN_DISTSHAPE_A_12",
        4,
        [
            "lcf_fs_logsigma_voxel",
            "lcf_wp_logsigma_voxel",
            "hcf_fs_logsigma_voxel",
            "hcf_wp_logsigma_voxel",
        ],
    ),
    (
        "M_B_logmu_logsigma",
        "het_nll_both_50_15_distB",
        "_V9b_PERGRAIN_DISTSHAPE_B_16",
        8,
        [
            "lcf_fs_logmu_voxel",
            "lcf_wp_logmu_voxel",
            "hcf_fs_logmu_voxel",
            "hcf_wp_logmu_voxel",
            "lcf_fs_logsigma_voxel",
            "lcf_wp_logsigma_voxel",
            "hcf_fs_logsigma_voxel",
            "hcf_wp_logsigma_voxel",
        ],
    ),
    (
        "M_C_p90_p99",
        "het_nll_both_50_15_distC",
        "_V9b_PERGRAIN_DISTSHAPE_C_16",
        8,
        [
            "lcf_fs_p90",
            "lcf_wp_p90",
            "hcf_fs_p90",
            "hcf_wp_p90",
            "lcf_fs_p99",
            "lcf_wp_p99",
            "hcf_fs_p99",
            "hcf_wp_p99",
        ],
    ),
]

SHALLOW_NODE_HEAD = {"num_headlayers": 1, "dim_headlayers": [8]}
CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
BASE_CELLS_FOR_REPORT = ("lcf_fs_max", "lcf_wp_max", "hcf_fs_max", "hcf_wp_max")


def _build_extras(
    loss_recipe: str, target_list: List[Dict[str, Any]], base_extra: Dict[str, Any]
) -> Dict[str, Any]:
    """Inject the locked v9b base + extended target list + het flags."""
    from manifest import _V9_PERGRAPH_TARGETS_14

    extras = copy.deepcopy(base_extra)
    extras["Recipe"]["loss_recipe"] = loss_recipe
    extras["Recipe"]["node_head_dims"] = dict(SHALLOW_NODE_HEAD)
    extras["Recipe"]["use_residual_node_head"] = True
    extras["Recipe"]["per_grain_targets"] = list(target_list)
    extras["Recipe"]["pergraph_targets"] = list(_V9_PERGRAPH_TARGETS_14)
    extras["Recipe"].pop("per_grain_target_field", None)
    extras["Recipe"].pop("per_grain_target_col", None)
    extras["Recipe"].pop("per_grain_aux_targets", None)
    extras["Recipe"].pop("aux_pergraph_targets", None)
    extras["Recipe"]["use_heteroscedastic_max"] = True
    extras["Recipe"]["use_heteroscedastic_pergraph_max"] = True
    extras.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    extras["NeuralNetwork"]["Architecture"]["graph_pooling"] = "max"
    return extras


def _aux_r2_per_target(run_dir: Path, target_names: List[str]) -> Dict[str, float]:
    """Pull per-channel R² from run_metrics.json's pergrain_aux block.
    Aux indices are 0-based positions in `per_grain_targets[1:]` (legacy
    deslot translation). target_names ⇒ slot positions."""
    rp = run_dir / "run_metrics.json"
    out: Dict[str, float] = {}
    if not rp.exists():
        return out
    m = json.load(open(rp))
    test = m.get("test") or {}
    aux = test.get("pergrain_aux") or {}
    # aux is keyed by aux index (0-based, where aux 0 = full per_grain_targets[1])
    # OR by name. Accept both.
    for nm in target_names:
        if nm in aux:
            out[nm] = float(aux[nm].get("r2", float("nan")))
        else:
            # fall back: look up by index from name patterns (best-effort)
            out[nm] = float("nan")
    return out


def _full_pergrain_r2(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Pull per-channel R² + Spearman for every per-grain target. The
    pergrain_aux entries have shape {name: {"metrics": {"r2", "spearman", ...}}}."""
    rp = run_dir / "run_metrics.json"
    if not rp.exists():
        return {}
    m = json.load(open(rp))
    test = m.get("test") or {}
    out: Dict[str, Dict[str, float]] = {}
    # Slot 0 — first per-grain target (after deslot translation = lcf_fs_max).
    pn0 = test.get("pergrain") or {}
    if isinstance(pn0, dict) and "r2" in pn0:
        out["lcf_fs_max"] = {
            "r2": float(pn0.get("r2", float("nan"))),
            "spearman": float(pn0.get("spearman", float("nan"))),
        }
    aux = test.get("pergrain_aux") or {}
    for nm, st in aux.items():
        if isinstance(st, dict):
            mt = st.get("metrics") or {}
            out[nm] = {
                "r2": float(mt.get("r2", float("nan"))),
                "spearman": float(mt.get("spearman", float("nan"))),
            }
    return out


def _row_for_run(
    run_dir: Path, tag: str, recipe: str, extra_target_names: List[str]
) -> Dict[str, Any]:
    """Aggregate per-channel R² for the BASE 4 max channels + the extras."""
    rp = run_dir / "run_metrics.json"
    row: Dict[str, Any] = {"cell": tag, "loss_recipe": recipe}
    if not rp.exists():
        return row
    m = json.load(open(rp))
    test = m.get("test") or {}
    row["train_time_s"] = float(m.get("elapsed_sec", float("nan")))

    # Base 4 max channel R² (from run_metrics test.pergrain_aux + .pergrain[slot 0])
    full = _full_pergrain_r2(run_dir)
    for nm in BASE_CELLS_FOR_REPORT:
        row[f"r2_{nm}"] = full.get(nm, {}).get("r2", float("nan"))
    # HCF top-1 to flag capacity-stealing
    h = test.get("pergrain_top1_hit_rate") or {}
    if isinstance(h, dict):
        row["hcf_fs_top1"] = float(
            (h.get("hcf_fs_max") or {}).get("rate", float("nan"))
        )
        row["hcf_wp_top1"] = float(
            (h.get("hcf_wp_max") or {}).get("rate", float("nan"))
        )
    row["mean_top1_4cell"] = float(
        test.get("pergrain_top1_hit_rate_mean_4cell", float("nan"))
    )

    # NEW: per-target R² for the extras
    for nm in extra_target_names:
        row[f"r2_{nm}"] = full.get(nm, {}).get("r2", float("nan"))

    # Headline summary across the extras
    extra_r2 = [row[f"r2_{nm}"] for nm in extra_target_names]
    extra_r2 = [r for r in extra_r2 if r == r and r is not None]
    row["mean_r2_extras"] = float(np.mean(extra_r2)) if extra_r2 else float("nan")
    row["min_r2_extras"] = float(np.min(extra_r2)) if extra_r2 else float("nan")
    row["max_r2_extras"] = float(np.max(extra_r2)) if extra_r2 else float("nan")
    return row


def _print_leaderboard(rows: List[Dict[str, Any]]) -> None:
    print()
    print("v9b Phase M — distribution-shape per-grain targets")
    print("=" * 130)
    hdr = (
        f"  {'cell':<22s} {'recipe':<28s}  "
        f"{'mean_r2_ext':>11s} {'min_r2_ext':>10s} {'max_r2_ext':>10s}  "
        f"{'mean_top1':>9s} {'time':>5s}"
    )
    print(hdr)
    print("-" * 130)
    for r in rows:
        print(
            f"  {r['cell']:<22s} {r.get('loss_recipe',''):<28s}  "
            f"{r.get('mean_r2_extras', float('nan')):>+11.3f} "
            f"{r.get('min_r2_extras', float('nan')):>+10.3f} "
            f"{r.get('max_r2_extras', float('nan')):>+10.3f}  "
            f"{r.get('mean_top1_4cell', float('nan')):>+9.3f} "
            f"{r.get('train_time_s', float('nan')):>5.0f}"
        )
    print("=" * 130)
    print()
    print("Per-target R² breakdown:")
    for r in rows:
        print(f"  {r['cell']}:")
        for k, v in sorted(r.items()):
            if k.startswith("r2_") and not k.endswith("_extras"):
                print(f"    {k[3:]:<28s} R² = {v:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_epoch", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--cells_only", type=str, default=None)
    args = ap.parse_args()

    cells = CELLS
    if args.cells_only:
        wanted = set(args.cells_only.split(","))
        cells = [c for c in CELLS if c[0] in wanted or c[0].split("_", 1)[1] in wanted]

    if not args.post:
        from manifest import BASE_RECIPE, TRACK_V7
        from _runner_utils import run_cell
        import manifest as M

        base_extra = TRACK_V7[0]["extra"]
        for tag, recipe, target_attr, n_extra, target_names in cells:
            res_path = STUDY / "results" / f"{tag}_seed{args.seed}" / "run_metrics.json"
            if res_path.exists():
                print(f"[M] {tag} already has metrics — skip", flush=True)
                continue
            target_list = getattr(M, target_attr)
            extras = _build_extras(recipe, target_list, base_extra)
            print(
                f"[M] {tag} starting (loss={recipe}, n_extra={n_extra}, "
                f"n_pn_targets={len(target_list)})",
                flush=True,
            )
            t0 = time.time()
            try:
                run_cell(
                    tag,
                    seed=args.seed,
                    base=BASE_RECIPE,
                    extra=extras,
                    num_epoch=args.num_epoch,
                    skip_done=True,
                )
                print(f"[M] {tag} done ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[M] {tag} FAILED: {e}", flush=True)
                traceback.print_exc()

    out_dir = STUDY / "results" / "M_distshape_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for tag, recipe, _, _, target_names in cells:
        rd = STUDY / "results" / f"{tag}_seed{args.seed}"
        rows.append(_row_for_run(rd, tag, recipe, target_names))
    if rows:
        (out_dir / "leaderboard.json").write_text(
            json.dumps(rows, indent=2, default=float)
        )
    _print_leaderboard(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
