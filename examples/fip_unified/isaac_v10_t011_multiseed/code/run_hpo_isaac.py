"""run_hpo_isaac.py — v10 HPO sweep, narrow & signal-rich follow-up to v9d.

Changes vs v9d (post-mortem of 37/200 v9d trials drove this design):

  Search space:
    - DROPPED: pg_loss_weight (silent no-op in v9d), batch_size, weight_decay,
               graph_pooling, lam_topdec, top_frac (all fixed at known-good)
    - DROPPED dropout=0 (incompatible with MCD in the locked pipeline)
    - DROPPED dropout=0.30 (top tier avoided it)
    - TIGHTENED hidden_dim, num_conv, node/graph head dims, lr, activation
    - ADDED graph_attr_source (concat_node vs film) — never searched in v9d

  Objectives (2, NSGA-II-friendly):
    O1 = mean(HCF tail R²)   — only metric with real HPO room in v9d
    O2 = -mean(HCF CRPS)     — proper scoring rule; subsumes Pcov + sharpness +
                                σ_log fit; not gameable by σ inflation

  Floors (admissibility — TrialPruned if violated):
    - min(HCF max R²)             >= 0.85   (existing)
    - min(HCF σ_pn Pcov z=1.65)   >= 0.85   (existing)
    - mean(HCF top-1)             >= 0.10   (existing)
    - mean(HCF mech-QoI R²)       >= 0.50   (NEW — would have screened M_B)
    - mean(HCF σ-residual Spearman) >= 0.50 (NEW — catches degenerate UQ)

  Per-trial: SINGLE seed (split=42, init=42). Multi-seed deferred to post-HPO
  winner selection on top 2-3 Pareto trials.

Locked from M_B:
  - SAGE backbone, ENRICHED_39 features, max-pool
  - residual node head ON, het-NLL on both heads (recipe `het_nll_both`)
  - target schema = _V9b_PERGRAIN_DISTSHAPE_B_16 + _V9_PERGRAPH_TARGETS_14
  - loss_recipe FIXED at `het_nll_both_50_10_distB` (lam=0.50, top_frac=0.10)
  - AdamW + ReduceLROnPlateau, BatchNorm, warmup_epochs=0, patience=50

Usage:
  # Pre-create study DB (single-process, before sbatch dispatch):
  python run_hpo_isaac.py --pre_create_study --n_trials 50
  # Submit via sbatch (per-slot worker):
  sbatch submit_isaac_v10_hpo.sbatch
  # Aggregate after completion:
  python run_hpo_isaac.py --post
"""
from __future__ import annotations
from fip_release_paths import release_root
import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
STUDY_NAME = "v10_M_B_HPO"
LOCKED_LOSS_RECIPE = "het_nll_both_50_10_distB"
HCF_CELLS = ("hcf_fs", "hcf_wp")
ALL_CELLS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
MECH_KEYS = (
    "yield_stress",
    "hardening_modulus",
    "Wp_lcf",
    "Wp_hcf",
    "sigma_amp_lcf",
    "sigma_amp_hcf",
)
HIDDEN_DIM_CHOICES = [96, 128, 192, 256]
NUM_CONV_CHOICES = [3, 5, 7]
NODE_HEAD_DIM_CHOICES = ["[32,16]", "[64,32]", "[64,64]"]
GRAPH_HEAD_DIM_CHOICES = ["[16,8]", "[32,16]", "[64,32]"]
DROPOUT_CHOICES = [0.05, 0.1, 0.15, 0.2]
ACTIVATIONS = ["lrelu_01", "prelu", "elu"]
GRAPH_ATTR_MODE_CHOICES = ["concat_node", "film"]


def _parse_dim_str(s: str) -> List[int]:
    """Parse '[64,32]' into [64, 32]."""
    return [int(x) for x in s.strip("[]").split(",")]


def suggest(trial) -> Dict[str, Any]:
    """Sample one HP point from the v10 narrow search space."""
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", HIDDEN_DIM_CHOICES),
        "num_conv_layers": trial.suggest_categorical(
            "num_conv_layers", NUM_CONV_CHOICES
        ),
        "node_head_dim_str": trial.suggest_categorical(
            "node_head_dim_str", NODE_HEAD_DIM_CHOICES
        ),
        "graph_head_dim_str": trial.suggest_categorical(
            "graph_head_dim_str", GRAPH_HEAD_DIM_CHOICES
        ),
        "dropout": trial.suggest_categorical("dropout", DROPOUT_CHOICES),
        "activation_function": trial.suggest_categorical(
            "activation_function", ACTIVATIONS
        ),
        "graph_attr_conditioning_mode": trial.suggest_categorical(
            "graph_attr_conditioning_mode", GRAPH_ATTR_MODE_CHOICES
        ),
        "lr": trial.suggest_float("lr", 5e-05, 0.0005, log=True),
    }


def overrides_to_extras(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Translate v10 HPO knobs → deep extras dict for run_cell."""
    from manifest import _V9_PERGRAPH_TARGETS_14, _V9b_PERGRAIN_DISTSHAPE_B_16

    pt_path = (
        os.environ.get("FIP_V5_DATA_PATH")
        or os.environ.get("FIP_V4_DATA_PATH")
        or os.environ.get("FIP_DATA_PATH")
        or str(release_root() / "examples/cpfem_fip_data/outputs/full_data_v5.pt")
    )
    node_dims = _parse_dim_str(overrides["node_head_dim_str"])
    graph_dims = _parse_dim_str(overrides["graph_head_dim_str"])
    arch = {
        "mpnn_type": "SAGE",
        "edge_features": [],
        "graph_pooling": "max",
        "dropout": overrides["dropout"],
        "hidden_dim": overrides["hidden_dim"],
        "num_conv_layers": overrides["num_conv_layers"],
        "activation_function": overrides["activation_function"],
    }
    train_cfg = {
        "Optimizer": {"learning_rate": overrides["lr"]},
        "weight_decay": 1e-05,
        "batch_size": 32,
        "patience": 30,
        "num_epoch": 200,
    }
    head_dims = {
        "node_head_dims": {
            "num_headlayers": len(node_dims),
            "dim_headlayers": node_dims,
        },
        "graph_head_dims": {
            "num_headlayers": len(graph_dims),
            "dim_headlayers": graph_dims,
        },
    }
    return {
        "NeuralNetwork": {"Architecture": arch, "Training": train_cfg},
        "Recipe": {
            "pt_path": pt_path,
            "use_per_graph_target": True,
            "graph_attr_source": "size_bucket_onehot",
            "graph_attr_conditioning_mode": overrides["graph_attr_conditioning_mode"],
            "loss_recipe": LOCKED_LOSS_RECIPE,
            "per_grain_targets": list(_V9b_PERGRAIN_DISTSHAPE_B_16),
            "pergraph_targets": list(_V9_PERGRAPH_TARGETS_14),
            "node_head_dims": head_dims["node_head_dims"],
            "graph_head_dims": head_dims["graph_head_dims"],
            "use_residual_node_head": True,
            "use_heteroscedastic_max": True,
            "use_heteroscedastic_pergraph_max": True,
            "split_seed": 42,
        },
    }


def _hcf_max_r2(metrics: Dict, split: str = "val") -> Tuple[List[float], float]:
    test = metrics.get(split) or {}
    aux = test.get("pergrain_aux") or {}
    rs = []
    for c in HCF_CELLS:
        v = aux.get(f"{c}_max", {})
        if isinstance(v, dict):
            r2 = float((v.get("metrics") or {}).get("r2", float("nan")))
            rs.append(r2)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_pg_r2(metrics: Dict, split: str = "val") -> Tuple[List[float], float]:
    test = metrics.get(split) or {}
    aux = test.get("aux_pergraph") or {}
    rs = []
    for c in HCF_CELLS:
        v = aux.get(f"{c}_pergraph_max", {})
        if isinstance(v, dict):
            r2 = float((v.get("metrics") or {}).get("r2", float("nan")))
            rs.append(r2)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_logsigma_r2(metrics: Dict, split: str = "val") -> Tuple[List[float], float]:
    """Diagnostic only (not an objective in v10)."""
    test = metrics.get(split) or {}
    aux = test.get("pergrain_aux") or {}
    rs = []
    for c in HCF_CELLS:
        v = aux.get(f"{c}_logsigma_voxel", {})
        if isinstance(v, dict):
            r2 = float((v.get("metrics") or {}).get("r2", float("nan")))
            rs.append(r2)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_top1(metrics: Dict, split: str = "val") -> Tuple[List[float], float]:
    test = metrics.get(split) or {}
    h = test.get("pergrain_per_cell_hotspot") or {}
    rs = []
    for c in HCF_CELLS:
        v = h.get(f"{c}_max", {})
        if isinstance(v, dict):
            rs.append(float(v.get("top1_hit_rate", float("nan"))))
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_mech_r2_mean(metrics: Dict, split: str = "val") -> Tuple[List[float], float]:
    """NEW v10 floor metric: mean R² across the 6 mech-QoI per-graph heads."""
    test = metrics.get(split) or {}
    apg = test.get("aux_pergraph") or {}
    rs = []
    for k in MECH_KEYS:
        v = apg.get(k, {})
        if isinstance(v, dict):
            r2 = float((v.get("metrics") or {}).get("r2", float("nan")))
            if np.isfinite(r2):
                rs.append(r2)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_sigma_spearman_mean(
    metrics: Dict, split: str = "val"
) -> Tuple[List[float], float]:
    """NEW v10 floor metric: σ-residual Spearman across HCF cells."""
    test = metrics.get(split) or {}
    sig = test.get("pergrain_sigma") or {}
    rs = []
    for c in HCF_CELLS:
        v = sig.get(f"{c}_max", {})
        if isinstance(v, dict):
            s = v.get("spearman_sigma")
            if s is not None and np.isfinite(float(s)):
                rs.append(float(s))
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_tail_r2_from_csv(
    res_dir: Path, split: str = "val"
) -> Tuple[List[float], float]:
    """Mean R² on top-10% truth across HCF cells (v10 O1 objective)."""
    p = res_dir / f"pred_{split}_pn.csv"
    if not p.exists():
        return ([], float("nan"))
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return ([], float("nan"))
    from sklearn.metrics import r2_score

    rs = []
    for c in HCF_CELLS:
        col_t = f"true_{c}_max"
        col_p = f"pred_{c}_max"
        if col_t not in rows[0] or col_p not in rows[0]:
            continue
        yt = []
        yp = []
        for r in rows:
            t = r.get(col_t)
            pp = r.get(col_p)
            if t in ("", None) or pp in ("", None):
                continue
            try:
                yt.append(float(t))
                yp.append(float(pp))
            except:
                pass
        yt = np.array(yt)
        yp = np.array(yp)
        if len(yt) < 50:
            continue
        thr = np.quantile(yt, 0.9)
        mask = yt >= thr
        if mask.sum() < 10:
            continue
        try:
            r2 = float(r2_score(yt[mask], yp[mask]))
        except Exception:
            r2 = float("nan")
        rs.append(r2)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_sigma_pn_pcov(
    res_dir: Path, split: str = "val", z: float = 1.65
) -> Tuple[List[float], float]:
    """σ_pn Pcov(z) over HCF cells (existing floor)."""
    p = res_dir / f"pred_{split}_pn.csv"
    if not p.exists():
        return ([], float("nan"))
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return ([], float("nan"))
    rs = []
    for c in HCF_CELLS:
        col_t = f"true_{c}_max"
        col_p = f"pred_{c}_max"
        col_s = f"logsigma_{c}_max"
        if col_s not in rows[0]:
            continue
        yt = []
        yp = []
        sg = []
        for r in rows:
            try:
                yt.append(float(r[col_t]))
                yp.append(float(r[col_p]))
                sg.append(float(r[col_s]))
            except:
                pass
        yt = np.array(yt)
        yp = np.array(yp)
        sg = np.array(sg)
        mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(sg)
        yt, yp, sg = (yt[mask], yp[mask], sg[mask])
        if len(yt) < 50:
            continue
        sig_raw = np.exp(sg) * yt.std()
        cov = float(np.mean(np.abs(yt - yp) <= z * sig_raw))
        rs.append(cov)
    return (rs, float(np.mean(rs)) if rs else float("nan"))


def _hcf_crps_from_csv(res_dir: Path, split: str = "val") -> Tuple[List[float], float]:
    """v10 O2 objective: mean Gaussian CRPS over HCF cells (lower = better)."""
    from crps import gaussian_crps

    p = res_dir / f"pred_{split}_pn.csv"
    if not p.exists():
        return ([], float("nan"))
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return ([], float("nan"))
    crps_vals = []
    for c in HCF_CELLS:
        col_t = f"true_{c}_max"
        col_p = f"pred_{c}_max"
        col_s = f"logsigma_{c}_max"
        if col_s not in rows[0]:
            continue
        yt = []
        yp = []
        sg = []
        for r in rows:
            try:
                yt.append(float(r[col_t]))
                yp.append(float(r[col_p]))
                sg.append(float(r[col_s]))
            except:
                pass
        yt = np.array(yt)
        yp = np.array(yp)
        sg = np.array(sg)
        mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(sg)
        yt, yp, sg = (yt[mask], yp[mask], sg[mask])
        if len(yt) < 50:
            continue
        sig_raw = np.exp(sg) * yt.std()
        crps = gaussian_crps(yt, yp, sig_raw)
        if np.isfinite(crps):
            crps_vals.append(crps)
    return (crps_vals, float(np.mean(crps_vals)) if crps_vals else float("nan"))


def make_objective(args, base_recipe):
    from _runner_utils import run_cell, deep_merge

    def objective(trial):
        import optuna

        overrides = suggest(trial)
        tag = f"hpo_v10_t{trial.number:03d}"
        extras = overrides_to_extras(overrides)
        t0 = time.time()
        try:
            metrics = run_cell(
                tag,
                seed=42,
                base=base_recipe,
                extra=extras,
                num_epoch=args.num_epoch,
                skip_done=True,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[trial {trial.number}] FAILED ({elapsed:.1f}s): {e}", flush=True)
            traceback.print_exc()
            raise optuna.TrialPruned()
        elapsed = time.time() - t0
        res_dir = STUDY / "results" / f"{tag}_seed42"
        hcf_r2_list, hcf_r2_mean = _hcf_max_r2(metrics, split="val")
        if not hcf_r2_list or min(hcf_r2_list) < 0.85:
            print(
                f"[trial {trial.number}] PRUNED — HCF max R² floor ({hcf_r2_list}, elapsed={elapsed:.1f}s)",
                flush=True,
            )
            raise optuna.TrialPruned()
        pcov_list, _ = _hcf_sigma_pn_pcov(res_dir, split="val", z=1.65)
        if not pcov_list or min(pcov_list) < 0.85:
            print(
                f"[trial {trial.number}] PRUNED — HCF σ_pn Pcov floor ({pcov_list}, elapsed={elapsed:.1f}s)",
                flush=True,
            )
            raise optuna.TrialPruned()
        _, top1_mean = _hcf_top1(metrics, split="val")
        if not top1_mean == top1_mean or top1_mean < 0.1:
            print(
                f"[trial {trial.number}] PRUNED — HCF top-1 mean {top1_mean:.3f} < 0.10",
                flush=True,
            )
            raise optuna.TrialPruned()
        mech_list, mech_mean = _hcf_mech_r2_mean(metrics, split="val")
        if not mech_mean == mech_mean or mech_mean < 0.5:
            print(
                f"[trial {trial.number}] PRUNED — mech R² mean {mech_mean:.3f} < 0.50",
                flush=True,
            )
            raise optuna.TrialPruned()
        sspr_list, sspr_mean = _hcf_sigma_spearman_mean(metrics, split="val")
        if not sspr_mean == sspr_mean or sspr_mean < 0.5:
            print(
                f"[trial {trial.number}] PRUNED — σ-Spearman mean {sspr_mean:.3f} < 0.50",
                flush=True,
            )
            raise optuna.TrialPruned()
        _, o1_tail_r2 = _hcf_tail_r2_from_csv(res_dir, split="val")
        _, o2_crps = _hcf_crps_from_csv(res_dir, split="val")
        _, pg_r2 = _hcf_pg_r2(metrics, split="val")
        _, logsig_r2 = _hcf_logsigma_r2(metrics, split="val")
        for nm, v in (("o1_tail_r2", o1_tail_r2), ("o2_crps", o2_crps)):
            if not v == v:
                print(f"[trial {trial.number}] PRUNED — {nm} NaN", flush=True)
                raise optuna.TrialPruned()
        print(
            f"[trial {trial.number}] DONE ({elapsed:.1f}s)  O1_tail_R²={o1_tail_r2:+.3f}  O2_CRPS={o2_crps:.4e}  top1={top1_mean:.3f}  mech={mech_mean:+.3f}  σspr={sspr_mean:+.3f}  pgR²={pg_r2:+.3f}  σlogR²={logsig_r2:+.3f}",
            flush=True,
        )
        trial.set_user_attr("loss_recipe", LOCKED_LOSS_RECIPE)
        trial.set_user_attr("hcf_max_r2_min", float(min(hcf_r2_list)))
        trial.set_user_attr("hcf_max_r2_mean", float(hcf_r2_mean))
        trial.set_user_attr("hcf_pcov_min", float(min(pcov_list)))
        trial.set_user_attr("hcf_top1_mean", float(top1_mean))
        trial.set_user_attr("hcf_mech_r2_mean", float(mech_mean))
        trial.set_user_attr("hcf_sigspr_mean", float(sspr_mean))
        trial.set_user_attr("hcf_pg_r2_mean", float(pg_r2))
        trial.set_user_attr("hcf_logsigma_r2_mean", float(logsig_r2))
        trial.set_user_attr("elapsed_s", float(elapsed))
        return (o1_tail_r2, -o2_crps)

    return objective


def post_aggregate(args, study) -> None:
    """Write Pareto front + all trials to disk (v10 = 2 objectives)."""
    import optuna

    out_dir = STUDY / "results" / "hpo_v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append(
            {
                "trial": t.number,
                **t.params,
                "o1_tail_r2": t.values[0] if t.values else float("nan"),
                "o2_neg_crps": t.values[1] if t.values else float("nan"),
                "o2_crps": -t.values[1] if t.values else float("nan"),
                **{k: v for k, v in (t.user_attrs or {}).items()},
            }
        )
    print(f"[post] {len(rows)} complete trials")
    if rows:
        with open(out_dir / "all_trials.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(out_dir / "all_trials.json", "w") as f:
            json.dump(rows, f, indent=2)
        try:
            best_trials = study.best_trials
            pareto = [
                {
                    "trial": t.number,
                    **t.params,
                    "o1_tail_r2": t.values[0],
                    "o2_crps": -t.values[1],
                    **{k: v for k, v in (t.user_attrs or {}).items()},
                }
                for t in best_trials
            ]
            with open(out_dir / "pareto.json", "w") as f:
                json.dump(pareto, f, indent=2)
            print(f"[post] Pareto front size: {len(pareto)}")
        except Exception as e:
            print(f"[post] Pareto extraction failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--num_epoch", type=int, default=400)
    ap.add_argument(
        "--slot", type=int, default=0, help="ISAAC slot index for sampler seeding"
    )
    ap.add_argument(
        "--pre_create_study",
        action="store_true",
        help="Single-process pre-create the SQLite study DB to avoid race",
    )
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--sampler", type=str, default="nsga", choices=["nsga", "motpe"])
    args = ap.parse_args()
    import optuna
    from _runner_utils import deep_merge
    from manifest import BASE_RECIPE

    db_dir = STUDY / "results"
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir}/{STUDY_NAME}.db"
    if args.pre_create_study:
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
            print(
                f"[pre_create] study {STUDY_NAME} already exists ({len(study.trials)} trials)"
            )
        except Exception:
            study = optuna.create_study(
                study_name=STUDY_NAME,
                storage=storage,
                directions=["maximize", "maximize"],
                sampler=optuna.samplers.NSGAIISampler(seed=42, population_size=12),
                load_if_exists=True,
            )
            print(f"[pre_create] created study {STUDY_NAME} at {storage}")
        return 0
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    if args.post:
        post_aggregate(args, study)
        return 0
    if args.sampler == "nsga":
        sampler = optuna.samplers.NSGAIISampler(seed=42 + args.slot, population_size=12)
    else:
        sampler = optuna.samplers.MOTPESampler(seed=42 + args.slot, n_startup_trials=10)
    study.sampler = sampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(
        f"[v10 HPO] worker slot={args.slot} sampler={args.sampler} n_trials={args.n_trials} num_epoch={args.num_epoch}",
        flush=True,
    )
    objective = make_objective(args, BASE_RECIPE)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    print(f"[v10 HPO] worker slot={args.slot} done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
