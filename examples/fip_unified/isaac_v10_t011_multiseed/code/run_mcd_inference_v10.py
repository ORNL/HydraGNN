"""run_mcd_inference_v9c.py — MC-Dropout inference for v9c E-sweep reps.

For each E_M_B_split{S}_seed{I} run dir, load best-val checkpoint, set
dropout layers to train mode (BatchNorm stays eval), run K=30
stochastic forward passes on the test set, and write
mcd_predictions.csv with per-grain posterior mean and epistemic std
for the 4 max channels and 4 σ_log channels.

Combined with the heteroscedastic σ_pn output (aleatoric component),
this gives the full epistemic + aleatoric decomposition:
    σ_total² = σ_pn² + σ_e²
where σ_e is the std across MCD passes (epistemic), σ_pn is the model's
own NLL-trained heteroscedastic σ (aleatoric).

Usage:
  python run_mcd_inference_v9c.py --K 30
  python run_mcd_inference_v9c.py --K 30 --start_idx 0 --stride 2  # parallel
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

import hydragnn
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model.model import load_existing_model
from hydragnn.models.create import create_model_config
from dataset import FIPDataset
from train import _install_residual_node_head

CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
SPLIT_SEEDS = [7, 13, 21, 42, 99, 137, 271, 314, 666, 2024]
INIT_SEEDS = [42]
WINNER_TAG = "t011"


def _ensure_distributed():
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29543")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def _activate_dropout(model: torch.nn.Module) -> int:
    """Turn on dropout while keeping BatchNorm in eval mode."""
    n_drop = 0
    model.eval()
    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
            ),
        ):
            m.train()
            n_drop += 1
    return n_drop


def _rebuild_model(res_dir: Path, log_dir: Path):
    cfg_path = res_dir / "config.json"
    cfg = json.load(open(cfg_path))
    ds_cfg = cfg["Recipe"]
    split_seed = ds_cfg.get("split_seed") or ds_cfg.get("seed") or 42

    fipds = FIPDataset(
        size_bucket=ds_cfg.get("size_bucket"),
        seed=int(split_seed),
        split_fracs=tuple(ds_cfg.get("split_fracs", (0.80, 0.10, 0.10))),
        stratify_by_bucket=bool(ds_cfg.get("stratify_by_bucket", True)),
        per_grain_target_field=str(ds_cfg.get("per_grain_target_field", "fs_fip")),
        per_grain_target_col=int(ds_cfg.get("per_grain_target_col", 0)),
        per_grain_targets=ds_cfg.get("per_grain_targets") or [],
        per_grain_aux_targets=ds_cfg.get("per_grain_aux_targets") or [],
        pergraph_targets=ds_cfg.get("pergraph_targets") or [],
        aux_pergraph_targets=ds_cfg.get("aux_pergraph_targets") or [],
        use_x_blocks=ds_cfg["use_x_blocks"],
        use_edge_cols=ds_cfg["use_edge_cols"],
        drop_periodic_edges=ds_cfg["drop_periodic_edges"],
        use_per_grain_target=ds_cfg["use_per_grain_target"],
        use_per_graph_target=ds_cfg["use_per_graph_target"],
        per_grain_transform=ds_cfg["per_grain_transform"],
        per_graph_transform=ds_cfg["per_graph_transform"],
        graph_attr_source=ds_cfg.get("graph_attr_source"),
        target_residual_baseline=ds_cfg.get("target_residual_baseline"),
        per_graph_target_aggr=ds_cfg.get("per_graph_target_aggr", "max"),
        per_graph_target_source=ds_cfg.get("per_graph_target_source", "default"),
        use_heteroscedastic_max=bool(ds_cfg.get("use_heteroscedastic_max", False)),
        use_heteroscedastic_pergraph_max=bool(
            ds_cfg.get("use_heteroscedastic_pergraph_max", False)
        ),
        pt_path=ds_cfg.get("pt_path"),
    )
    train_data, val_data, test_data = fipds.build()

    if ds_cfg.get("graph_attr_source") is not None:
        cfg["NeuralNetwork"]["Architecture"]["use_graph_attr_conditioning"] = True
        cfg["NeuralNetwork"]["Architecture"][
            "graph_attr_conditioning_mode"
        ] = ds_cfg.get("graph_attr_conditioning_mode", "concat_node")

    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
    _ensure_distributed()
    cfg = update_config(cfg, train_loader, val_loader, test_loader)

    model = create_model_config(
        config=cfg["NeuralNetwork"], verbosity=cfg["Verbosity"]["level"]
    )
    model = hydragnn.utils.distributed.get_distributed_model(
        model, cfg["Verbosity"]["level"]
    )
    # Phase L4 residual head: locked into v9b/v9c base. The trained
    # checkpoint has `base_head.*` + `skip.weight` keys, so the rebuilt
    # model must be wrapped before load_state_dict.
    if bool(ds_cfg.get("use_residual_node_head", False)):
        _install_residual_node_head(model)
    load_existing_model(model, str(log_dir))
    return model, fipds, test_loader, cfg


def _one_pass(model, loader, device, n_super):
    """One forward pass on test loader. Returns (pn_arr, pg_arr) where:
      pn_arr: (sum_grains, n_pn_total)  — per-grain head, ordered by graph then grain
      pg_arr: (n_sves,    n_pg_total)  — per-graph head, ordered by graph
    pg_arr is None if the model has no second head.
    """
    pn_parts, pg_parts = [], []
    has_pg = None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            if isinstance(preds, (list, tuple)):
                pn_parts.append(preds[0].detach().cpu().numpy())
                if has_pg is None:
                    has_pg = len(preds) > 1
                if has_pg:
                    pg_parts.append(preds[1].detach().cpu().numpy())
            else:
                pn_parts.append(preds.detach().cpu().numpy())
                has_pg = False
    pn_arr = np.concatenate(pn_parts, axis=0) if pn_parts else np.array([])
    pg_arr = np.concatenate(pg_parts, axis=0) if pg_parts else None
    return pn_arr, pg_arr


def run_one(res_dir: Path, K: int = 30) -> Optional[Path]:
    log_dir = STUDY / "logs" / res_dir.name
    if not log_dir.exists():
        log_dir = STUDY / "logs" / f"v4_{res_dir.name}"
    if not log_dir.exists():
        # Many runs use exp_tag-prefixed name
        for cand in (STUDY / "logs").glob(f"*{res_dir.name}*"):
            log_dir = cand
            break
    if not log_dir.exists():
        print(f"  [skip] {res_dir.name}: no log_dir found")
        return None

    print(f"  [load] {res_dir.name} -> {log_dir.name}")
    model, fipds, test_loader, cfg = _rebuild_model(res_dir, log_dir)
    n_drop = _activate_dropout(model)
    print(f"  [mcd] {n_drop} dropout layers active; running K={K} passes")
    if n_drop == 0:
        print(
            f"  [warn] no dropout layers — MCD reduces to deterministic; running 1 pass"
        )
        K = 1

    # Determine n_super from per_grain_targets
    n_super = len(cfg["Recipe"].get("per_grain_targets") or [])
    if n_super == 0:
        n_super = 1 + len(cfg["Recipe"].get("per_grain_aux_targets") or [])
    n_sigma = 4 if cfg["Recipe"].get("use_heteroscedastic_max") else 0
    n_total = n_super + n_sigma
    print(f"  [head] n_super={n_super}, n_sigma={n_sigma}, n_total={n_total}")

    device = next(model.parameters()).device
    pn_passes, pg_passes = [], []
    for k in range(K):
        pn_arr, pg_arr = _one_pass(model, test_loader, device, n_super)
        pn_passes.append(pn_arr)
        if pg_arr is not None:
            pg_passes.append(pg_arr)
        if (k + 1) % 5 == 0:
            print(f"    pass {k+1}/{K}", flush=True)

    P = np.stack(pn_passes, axis=0)  # (K, sum_grains, n_pn_total)
    pred_mean = P.mean(axis=0)
    pred_std = P.std(axis=0)

    # Per-graph head MCD (NEW): same K passes, one row per SVE
    has_pg = len(pg_passes) > 0
    if has_pg:
        Q = np.stack(pg_passes, axis=0)  # (K, n_sves, n_pg_total)
        pg_pred_mean = Q.mean(axis=0)
        pg_pred_std = Q.std(axis=0)
        # Inspect pergraph_targets to decode pg slots
        n_pg_super = len(cfg["Recipe"].get("pergraph_targets") or [])
        n_pg_sigma = 4 if cfg["Recipe"].get("use_heteroscedastic_pergraph_max") else 0
        n_pg_total = n_pg_super + n_pg_sigma
        print(
            f"  [pg]   n_pg_super={n_pg_super}, n_pg_sigma={n_pg_sigma}, "
            f"shape={pg_pred_mean.shape}"
        )
    else:
        pg_pred_mean = pg_pred_std = np.array([])
        n_pg_super = n_pg_sigma = n_pg_total = 0

    # Save numpy artifact (now with both pn and pg ensembles)
    save_kwargs = {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "K": K,
        "n_super": n_super,
        "n_pg_super": n_pg_super,
        "n_pg_sigma": n_pg_sigma,
    }
    if has_pg:
        save_kwargs["pg_pred_mean"] = pg_pred_mean
        save_kwargs["pg_pred_std"] = pg_pred_std
    np.savez_compressed(res_dir / "mcd_predictions.npz", **save_kwargs)

    # Per-channel epistemic σ summary (max channels at slots 0..3)
    summary = {
        "K": K,
        "n_grains": int(pred_mean.shape[0]),
        "epistemic_std_mean_per_max_channel": {
            c: float(pred_std[:, i].mean()) for i, c in enumerate(CELL_KEYS)
        },
        "epistemic_std_mean_per_logsigma_channel": {},
    }
    if n_super == 16:
        for i, c in enumerate(CELL_KEYS):
            summary["epistemic_std_mean_per_logsigma_channel"][c] = float(
                pred_std[:, 12 + i].mean()
            )

    # NEW: per-graph head epistemic σ summary
    if has_pg:
        summary["pg_epistemic_std_mean_per_max_channel"] = {
            c: float(pg_pred_std[:, i].mean())
            for i, c in enumerate(CELL_KEYS)
            if pg_pred_std.shape[1] > i
        }
        # pg slot ordering for distB recipe = 4 max + 4 ve_mu + 6 mech (+ 4 σ_pg)
        # mech slot offsets at 8..13 if n_pg_super=14
        if n_pg_super == 14:
            MECH_KEYS = [
                "yield_stress",
                "hardening_modulus",
                "Wp_lcf",
                "Wp_hcf",
                "sigma_amp_lcf",
                "sigma_amp_hcf",
            ]
            summary["pg_epistemic_std_mean_per_mech_channel"] = {
                k: float(pg_pred_std[:, 8 + j].mean())
                for j, k in enumerate(MECH_KEYS)
                if pg_pred_std.shape[1] > 8 + j
            }
            summary["pg_epistemic_std_mean_per_ve_mu_channel"] = {
                c: float(pg_pred_std[:, 4 + i].mean())
                for i, c in enumerate(CELL_KEYS)
                if pg_pred_std.shape[1] > 4 + i
            }

    with open(res_dir / "mcd_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [done] {res_dir.name}: K={K} passes complete (pn + pg)")
    return res_dir / "mcd_predictions.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    pairs = [(s, i) for s in SPLIT_SEEDS for i in INIT_SEEDS]
    pairs = pairs[args.start_idx :: args.stride]

    n_ok = 0
    for split_seed, init_seed in pairs:
        tag = f"{WINNER_TAG}_split{split_seed}"
        res_dir = STUDY / "results" / f"{tag}_seed{init_seed}"
        if not res_dir.exists():
            print(f"  [skip] {res_dir.name}: dir missing")
            continue
        if (res_dir / "mcd_predictions.npz").exists():
            print(f"  [skip] {res_dir.name}: mcd_predictions.npz already exists")
            n_ok += 1
            continue
        try:
            run_one(res_dir, K=args.K)
            n_ok += 1
        except Exception as e:
            import traceback

            print(f"  [fail] {res_dir.name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n[mcd] done — {n_ok}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
