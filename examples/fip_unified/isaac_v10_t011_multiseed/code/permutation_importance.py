"""permutation_importance.py - feature attribution for the trained C1 GNN.

Loads the best-val checkpoint of the chosen cell, then permutes one feature
at a time across the test set and measures the drop in test per-grain R^2.
Outputs a JSON + a horizontal bar chart.

Granularity:
  * Permute each NODE feature dim individually (e.g. quat[0..3], schmid12[0..11], ...)
  * Permute each EDGE feature column individually (e.g. edge_length, ...)
  * Permute groups of related dims together (quat, schmid12, voxel_geom, ...)

Permutation is per-graph (within an SVE) so global statistics are preserved.
This isolates the *informational* content of a feature without contaminating
the test distribution shift.

Usage:
  python permutation_importance.py --tag C1_a5_stdmatch_l75 --seed 42
  python permutation_importance.py --tag C1_a5_stdmatch_l75 --seed 42 --groups
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hydragnn  # noqa: F401
from hydragnn.models.create import create_model_config
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.model.model import load_existing_model
from hydragnn.utils.distributed import setup_ddp
from dataset import FIPDataset, X_COL_LAYOUT
from train import build_var_config_and_heads, _evaluate
from losses import install_batch_stash_hook, override_model_loss


def build_x_groups_from_blocks(use_x_blocks):
    """Map each use_x_blocks name to its slice in the *selected* (concatenated) X tensor.

    X_COL_LAYOUT slices are positions in the *raw* x[:, :40] tensor. After
    _select_x(x), columns are concatenated in the order of `use_x_blocks`.
    Compute the resulting per-block slice in the concatenated tensor.
    """
    out = {}
    cursor = 0
    for b in use_x_blocks:
        sl = X_COL_LAYOUT[b]
        width = sl.stop - sl.start
        out[b] = slice(cursor, cursor + width)
        cursor += width
    return out


def build_edge_cols_from_use(use_edge_cols):
    """Map each use_edge_cols name to its position in the *selected* edge_attr tensor."""
    return {name: idx for idx, name in enumerate(use_edge_cols)}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def predict_pergrain(model, loader, fipds, device):
    """Use train._evaluate to get raw per-grain (true, pred) arrays + buckets.

    _evaluate handles the data.y → per-grain/per-graph slicing correctly via
    batch.to_data_list(); rolling our own confused them.
    """
    metrics_dict = _evaluate(model, loader, fipds, device)
    if "pergrain_pred" not in metrics_dict:
        return np.array([]), np.array([]), np.array([])
    y_p = np.concatenate(metrics_dict["pergrain_pred"])
    y_t = np.concatenate(metrics_dict["pergrain_true"])
    # Reconstruct bucket-per-grain from the loader by mirroring _evaluate's loop.
    buckets = []
    for batch in loader:
        for d in batch.to_data_list():
            n = int(d.x.shape[0])
            b = int(d.size_bucket[0]) if hasattr(d, "size_bucket") else 0
            buckets.extend([b] * n)
    return y_t, y_p, np.asarray(buckets, dtype=np.int64)


def permute_node_dim_inplace(
    test_data: list, dim_slice: slice, rng: np.random.Generator
):
    """Permute the values of x[:, dim_slice] within each graph (preserves
    per-graph distribution, breaks per-feature signal). Modifies copies."""
    out = []
    for g in test_data:
        g2 = copy.copy(g)
        x = g.x.clone()
        n = x.size(0)
        if n > 1:
            perm = torch.tensor(rng.permutation(n), dtype=torch.long)
            x[:, dim_slice] = x[perm][:, dim_slice]
        g2.x = x
        out.append(g2)
    return out


def permute_edge_col_inplace(test_data: list, col: int, rng: np.random.Generator):
    """Permute one column of edge_attr within each graph."""
    out = []
    for g in test_data:
        g2 = copy.copy(g)
        if g.edge_attr is None or g.edge_attr.numel() == 0:
            out.append(g2)
            continue
        ea = g.edge_attr.clone()
        n_edges = ea.size(0)
        if n_edges > 1:
            perm = torch.tensor(rng.permutation(n_edges), dtype=torch.long)
            ea[:, col] = ea[perm, col]
        g2.edge_attr = ea
        out.append(g2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="C1_a5_stdmatch_l75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of permutation trials per feature (averaged).",
    )
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    res_dir = HERE.parent / "results" / f"{args.tag}_seed{args.seed}"
    log_dir = HERE.parent / "logs" / f"v4_{args.tag}_seed{args.seed}"
    if not (res_dir / "config.json").exists():
        raise FileNotFoundError(f"no config for {res_dir}")

    cfg = json.load(open(res_dir / "config.json"))

    # update_config calls assert torch.distributed.is_initialized() — set up
    # a 1-process DDP world so the assertion passes.
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else None
        )
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=device_id,
        )

    # Build dataset (same params as the trained run).
    ds_cfg = cfg["Recipe"]
    split_seed = ds_cfg.get("split_seed") or args.seed
    fipds_kwargs = dict(
        size_bucket=ds_cfg["size_bucket"],
        seed=split_seed,
        split_fracs=tuple(ds_cfg.get("split_fracs", (0.90, 0.05, 0.05))),
        stratify_by_bucket=bool(ds_cfg.get("stratify_by_bucket", True)),
        graph_attr_source=ds_cfg.get("graph_attr_source"),
        per_graph_target_aggr=ds_cfg.get("per_graph_target_aggr", "max"),
        per_grain_target_col=int(ds_cfg.get("per_grain_target_col", 0)),
        per_grain_target_field=str(ds_cfg.get("per_grain_target_field", "fs_fip")),
        use_x_blocks=ds_cfg["use_x_blocks"],
        use_edge_cols=ds_cfg["use_edge_cols"],
        drop_periodic_edges=ds_cfg["drop_periodic_edges"],
        use_per_grain_target=ds_cfg["use_per_grain_target"],
        use_per_graph_target=ds_cfg["use_per_graph_target"],
        per_grain_transform=ds_cfg["per_grain_transform"],
        per_graph_transform=ds_cfg["per_graph_transform"],
    )
    fipds = FIPDataset(**fipds_kwargs)
    train_data, val_data, test_data = fipds.build()
    print(
        f"[perm] dataset built: train={len(train_data)} val={len(val_data)} test={len(test_data)}"
    )

    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

    # Build model — needs update_config to fill input_dim/pna_deg.
    cfg2 = build_var_config_and_heads(cfg)
    cfg2 = update_config(cfg2, train_loader, val_loader, test_loader)
    model = create_model_config(
        config=cfg2["NeuralNetwork"], verbosity=cfg2["Verbosity"]["level"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Custom loss override (needed for batch-stash hook to be installed,
    # so the model's forward hooks fire). Match the trained config.
    loss_recipe = ds_cfg.get("loss_recipe")
    if loss_recipe and loss_recipe != "default":
        install_batch_stash_hook(model)
        override_model_loss(model, loss_recipe)

    # Load checkpoint.
    load_existing_model(model, str(log_dir))
    print(f"[perm] loaded checkpoint from {log_dir}")

    # Baseline metric.
    yt, yp, _ = predict_pergrain(model, test_loader, fipds, device)
    base_r2 = r2_score(yt, yp)
    print(f"[perm] baseline test pn_R^2 = {base_r2:+.4f}  (n={yt.size})")

    rng = np.random.default_rng(args.seed)
    results = {
        "baseline_pn_r2": base_r2,
        "node_groups": {},
        "node_dims": {},
        "edge_cols": {},
    }

    # Compute X_GROUPS and EDGE_COLS from the actual config (selected indices).
    X_GROUPS = build_x_groups_from_blocks(ds_cfg["use_x_blocks"])
    EDGE_COLS = build_edge_cols_from_use(ds_cfg["use_edge_cols"])
    print(f"[perm] node groups (selected positions): {X_GROUPS}")
    print(f"[perm] edge cols (selected positions): {EDGE_COLS}")

    # Per-group node permutations.
    for name, sl in X_GROUPS.items():
        deltas = []
        for trial in range(args.n_trials):
            test_perm = permute_node_dim_inplace(test_data, sl, rng)
            loader = DataLoader(test_perm, batch_size=bs, shuffle=False)
            yt2, yp2, _ = predict_pergrain(model, loader, fipds, device)
            r2 = r2_score(yt2, yp2)
            deltas.append(base_r2 - r2)
        d_mean = float(np.mean(deltas))
        d_std = float(np.std(deltas))
        results["node_groups"][name] = {
            "delta_r2_mean": d_mean,
            "delta_r2_std": d_std,
            "n_dims": sl.stop - sl.start,
        }
        print(
            f"  [node group] {name:<14s} (dims {sl.start}..{sl.stop-1})  ΔR^2 = {d_mean:+.4f} ± {d_std:.4f}"
        )

    # Per-column edge permutations.
    for name, col in EDGE_COLS.items():
        deltas = []
        for trial in range(args.n_trials):
            test_perm = permute_edge_col_inplace(test_data, col, rng)
            loader = DataLoader(test_perm, batch_size=bs, shuffle=False)
            yt2, yp2, _ = predict_pergrain(model, loader, fipds, device)
            r2 = r2_score(yt2, yp2)
            deltas.append(base_r2 - r2)
        d_mean = float(np.mean(deltas))
        d_std = float(np.std(deltas))
        results["edge_cols"][name] = {
            "delta_r2_mean": d_mean,
            "delta_r2_std": d_std,
            "col": col,
        }
        print(
            f"  [edge col] {name:<22s} (col {col})  ΔR^2 = {d_mean:+.4f} ± {d_std:.4f}"
        )

    # Save.
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (HERE.parent / "results" / "permutation_importance")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"perm_imp_{args.tag}_seed{args.seed}.json"
    with open(out_json, "w") as f:
        json.dump(
            {"tag": args.tag, "seed": args.seed, "n_trials": args.n_trials, **results},
            f,
            indent=2,
        )
    print(f"[perm] wrote {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
