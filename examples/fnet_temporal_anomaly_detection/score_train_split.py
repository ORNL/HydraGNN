#!/usr/bin/env python3
"""
score_train_split.py
====================

Re-load a trained FNET multi-feature T-GCN checkpoint and dump
``preds_train.npy`` / ``ys_train.npy`` next to the existing
``preds_val.npy`` / ``preds_test.npy`` so the diagnostics script can
also be run on the training split.

Usage
-----
    python score_train_split.py \\
        --cache_dir examples/fnet_temporal_anomaly_detection/dataset \\
        --date 2024-06-01 \\
        --log fnet_full_2024-06-01 \\
        --out_dir outputs_fnet_full \\
        --cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make sibling helpers importable when run from the repo root.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import hydragnn  # noqa: E402

# Re-use private helpers from the trainer.
from fnet_temporal_anomaly_detection import (  # noqa: E402
    _score_split,
    read_cache,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score the trainset of an existing FNET T-GCN run "
        "and dump preds_train / ys_train npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=str(THIS_DIR / "dataset"),
        help="Pickle/ADIOS cache directory used at training time",
    )
    p.add_argument(
        "--date",
        type=str,
        default="2024-06-01",
        help="Date subfolder used at training time",
    )
    p.add_argument(
        "--format", type=str, default="pickle", choices=["pickle", "adios"]
    )
    p.add_argument(
        "--log",
        type=str,
        required=True,
        help="Log name (subdir under ./logs/) used at training time",
    )
    p.add_argument(
        "--logs_root",
        type=str,
        default="logs",
        help="Root directory containing the log subdir",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where preds_train.npy / ys_train.npy will be written "
        "(usually the same as the trainer's --out_dir)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override scoring batch size (defaults to the value in config.json)",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA exists")
    return p


def main():
    args = build_argparser().parse_args()
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training-time config.
    cfg_path = Path(args.logs_root) / args.log / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = int(args.batch_size)
    verbosity = config["Verbosity"]["level"]

    # DDP setup is required by HydraGNN dataloaders / model factories.
    hydragnn.utils.distributed.setup_ddp()

    print(f"[cache] reading splits ({args.format}) from {cache_dir}")
    trainset, valset, testset, predset, meta = read_cache(
        cache_dir, args.date, args.format
    )
    print(
        f"[cache] train/val/test = {len(trainset)}/{len(valset)}/{len(testset)}  "
        f"| sites={len(meta['sites'])}, Tin={meta['Tin']}, H={meta['horizon']}, "
        f"F_dyn={meta['F_dyn']}, F_static={meta['F_static']}, F_out={meta['F_out']}"
    )

    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity
    )
    model, _ = hydragnn.utils.distributed.distributed_model_wrapper(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3), verbosity
    )

    # Load the final-epoch checkpoint written by hydragnn.train.train_validate_test.
    ckpt_path = Path(args.logs_root) / args.log / f"{args.log}.pk"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    print(f"[ckpt] loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model_state_dict"]
    target = model.module if hasattr(model, "module") else model
    # Strip 'module.' prefix if present and target is unwrapped, or add it back.
    first_key = next(iter(state_dict))
    target_first = next(iter(target.state_dict()))
    if first_key.startswith("module.") and not target_first.startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    target.load_state_dict(state_dict, strict=False)

    raw_model = (model.module if hasattr(model, "module") else model).to(device)
    print("[score] Scoring train split ...")
    preds_train, ys_train = _score_split(raw_model, train_loader, meta, device)
    train_mse = float(np.mean((preds_train - ys_train) ** 2)) if preds_train.size else float("nan")
    print(f"[score] train MSE = {train_mse:.6e}  shape={preds_train.shape}")

    np.save(out_dir / "preds_train.npy", preds_train)
    np.save(out_dir / "ys_train.npy", ys_train)
    print(f"[save]  wrote preds_train.npy / ys_train.npy to {out_dir}/")


if __name__ == "__main__":
    main()
