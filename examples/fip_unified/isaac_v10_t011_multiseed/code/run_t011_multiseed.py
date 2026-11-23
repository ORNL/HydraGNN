"""run_t011_multiseed.py — multi-split-seed training of v10 HPO winner t011.

10 split seeds × 1 init seed = 10 reps. Train each to 400 epochs with
patience=50. Checkpoint=True so MCD inference can load best-val later.

Locked t011 hyperparameters (from v10_hpo_local hpo_v10_t011_seed42):
  hidden_dim          = 256
  num_conv_layers     = 3
  graph_pooling       = max
  activation_function = elu
  dropout             = 0.1
  lr                  = 1.5867e-04
  weight_decay        = 1.0e-05
  batch_size          = 32
  graph_attr_cond_mode= concat_node
  loss_recipe         = het_nll_both_50_10_distB
  node_head_dims      = [64, 32]
  graph_head_dims     = [32, 16] (shared 2x16)

Per-trial layout:
  results/t011_split{S}_seed42/   ← run_metrics.json, pred_test_*.csv, ...

After all training: run run_mcd_inference_v10.py (K=30 stochastic
passes per rep), then aggregate_mcd_v10.py.

Usage:
  python run_t011_multiseed.py            # train all 10 reps
  python run_t011_multiseed.py --skip_train  # only post-aggregate metrics
"""
from __future__ import annotations
from fip_release_paths import release_root
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
SPLIT_SEEDS = [7, 13, 21, 42, 99, 137, 271, 314, 666, 2024]
INIT_SEED = 42
WINNER_TAG = "t011"
NUM_EPOCH = 400
T011_ARCH = {
    "mpnn_type": "SAGE",
    "edge_features": [],
    "graph_pooling": "max",
    "dropout": 0.1,
    "hidden_dim": 256,
    "num_conv_layers": 3,
    "activation_function": "elu",
}
T011_TRAINING = {
    "Optimizer": {"learning_rate": 0.0001586668862464421},
    "weight_decay": 1e-05,
    "batch_size": 32,
    "patience": 50,
    "num_epoch": NUM_EPOCH,
    "Checkpoint": True,
    "EarlyStopping": True,
    "checkpoint_warmup": 5,
}
T011_HEAD_DIMS = {
    "node_head_dims": {"num_headlayers": 2, "dim_headlayers": [64, 32]},
    "graph_head_dims": {
        "num_sharedlayers": 2,
        "dim_sharedlayers": 16,
        "num_headlayers": 2,
        "dim_headlayers": [32, 16],
    },
}


def _build_extras(split_seed: int) -> dict:
    """Build a deep-mergeable extras dict for one (split_seed, init=42) rep."""
    from manifest import _V9_PERGRAPH_TARGETS_14, _V9b_PERGRAIN_DISTSHAPE_B_16
    import os

    pt_path = (
        os.environ.get("FIP_V5_DATA_PATH")
        or os.environ.get("FIP_V4_DATA_PATH")
        or str(release_root() / "examples/cpfem_fip_data/outputs/full_data_v5.pt")
    )
    return {
        "NeuralNetwork": {
            "Architecture": copy.deepcopy(T011_ARCH),
            "Training": copy.deepcopy(T011_TRAINING),
        },
        "Recipe": {
            "pt_path": pt_path,
            "use_per_graph_target": True,
            "graph_attr_source": "size_bucket_onehot",
            "graph_attr_conditioning_mode": "concat_node",
            "loss_recipe": "het_nll_both_50_10_distB",
            "per_grain_targets": list(_V9b_PERGRAIN_DISTSHAPE_B_16),
            "pergraph_targets": list(_V9_PERGRAPH_TARGETS_14),
            "node_head_dims": dict(T011_HEAD_DIMS["node_head_dims"]),
            "graph_head_dims": dict(T011_HEAD_DIMS["graph_head_dims"]),
            "use_residual_node_head": True,
            "use_heteroscedastic_max": True,
            "use_heteroscedastic_pergraph_max": True,
            "split_seed": int(split_seed),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip_train", action="store_true", help="Don't train; just print summary"
    )
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    from _runner_utils import run_cell
    from manifest import BASE_RECIPE

    seeds = SPLIT_SEEDS[args.start_idx :: args.stride]
    print(f"[t011 multi-seed] {len(seeds)} reps to train")
    print(f"  split seeds: {seeds}")
    print(f"  init seed:   {INIT_SEED} (fixed)")
    print(f"  num_epoch:   {NUM_EPOCH}, patience: {T011_TRAINING['patience']}")
    print()
    if args.skip_train:
        print("[skip] --skip_train flag set; not running training")
        return 0
    n_ok = 0
    n_fail = 0
    for k, split_seed in enumerate(seeds):
        tag = f"{WINNER_TAG}_split{split_seed}"
        res_path = STUDY / "results" / f"{tag}_seed{INIT_SEED}" / "run_metrics.json"
        if res_path.exists():
            print(
                f"[skip] rep {k + 1}/{len(seeds)}: {tag}_seed{INIT_SEED} already done"
            )
            n_ok += 1
            continue
        extras = _build_extras(split_seed)
        print(
            f"[rep {k + 1}/{len(seeds)}] {tag}_seed{INIT_SEED} starting...", flush=True
        )
        t0 = time.time()
        try:
            run_cell(
                tag, INIT_SEED, BASE_RECIPE, extras, num_epoch=NUM_EPOCH, skip_done=True
            )
            elapsed = time.time() - t0
            print(
                f"[rep {k + 1}/{len(seeds)}] {tag}_seed{INIT_SEED} DONE ({elapsed:.1f}s)",
                flush=True,
            )
            n_ok += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(
                f"[rep {k + 1}/{len(seeds)}] {tag}_seed{INIT_SEED} FAILED ({elapsed:.1f}s): {e}",
                flush=True,
            )
            traceback.print_exc()
            n_fail += 1
    print()
    print(f"[t011 multi-seed] {n_ok}/{len(seeds)} OK, {n_fail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
