"""Fine-tuning script for FT1: OPF feasibility binary classification.

Loads a pretrained HeteroSAGE or HeteroHEAT checkpoint and fine-tunes it for
graph-level binary feasibility prediction.

The dataset must first be generated with generate_infeasible_samples.py, which
produces a balanced HDF5 dataset where:
  - feasible samples: data.y = [1.0]  (original OPF solutions)
  - infeasible samples: data.y = [0.0] (load features scaled by overload_factor)

The model uses a graph-level output head (pool all node embeddings → MLP → 1
logit) with binary cross-entropy with logits loss.  Inference: sigmoid(logit)
> 0.5 → feasible.

Example::

    python train_opf_ft1_classify.py \\
        --inputfile FT1_feasibility_classification/config_HeteroSAGE_full.json \\
        --data_root ../dataset \\
        --pretrained_model_dir ../pretrained_models \\
        --pretrained_model_name HeteroSAGE_best \\
        --finetune_regime full \\
        --num_epoch 50 \\
        --learning_rate 1e-4
"""

import os
import sys
import json
import argparse

# Make examples/opf importable
_OPF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _OPF_DIR)

import torch
import torch.distributed as dist
from collections import OrderedDict
from mpi4py import MPI

import hydragnn
import hydragnn.utils.model as model_utils
from hydragnn.utils.model import print_model
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.input_config_parsing import save_config
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset

from ft_utils import EpochCSVWriter, evaluate_ft1, save_run_results

from opf_solution_utils import (
    EdgeAttrDatasetAdapter,
    OPFEnhancedModelWrapper,
    info,
)


# ---------------------------------------------------------------------------
# Utility helpers (duplicated from train_opf_finetune.py for independence)
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    import numpy as np
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _resolve_edge_dim(config):
    arch = config.get("NeuralNetwork", {}).get("Architecture", {})
    return arch.get("edge_dim", {"ac_line": 9, "transformer": 11})


def apply_freeze_regime(model, regime: str):
    """Freeze model parameters according to the fine-tuning regime.

    Called after loading pretrained weights, before creating the optimizer.

    Parameters
    ----------
    model : nn.Module  (not yet DDP-wrapped)
    regime : str — 'full', 'partial', or 'head_only'
    """
    if regime == "full":
        return

    inner = model
    if isinstance(inner, OPFEnhancedModelWrapper):
        inner = inner.model

    if regime == "head_only":
        for name, param in inner.named_parameters():
            if "heads_NN" not in name and "graph_shared" not in name:
                param.requires_grad_(False)
        n = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        info(f"[FT1] head_only: {n:,} trainable parameters (graph head only)")
        return

    if regime == "partial":
        for name, param in inner.named_parameters():
            if "input_projectors" in name or "node_embedders" in name:
                param.requires_grad_(False)
        n_conv = len(inner.graph_convs)
        for i, (conv, feat) in enumerate(zip(inner.graph_convs, inner.feature_layers)):
            if i < n_conv - 1:
                for p in conv.parameters():
                    p.requires_grad_(False)
                for p in feat.parameters():
                    p.requires_grad_(False)
        n = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        info(
            f"[FT1] partial: {n:,} trainable parameters "
            f"(last conv layer + graph head; {n_conv-1} of {n_conv} conv layers frozen)"
        )
        return

    raise ValueError(
        f"Unknown finetune_regime '{regime}'. Choose from: full, partial, head_only."
    )


def load_pretrained_weights(model, pretrained_model_name: str, pretrained_model_dir: str):
    """Load conv-layer weights from a pretrained node-level regression checkpoint.

    The prediction head is NOT loaded (strict=False), so the pretrained encoder
    is reused while the graph-level classification head starts from scratch.

    Checkpoint expected at:
        <pretrained_model_dir>/<pretrained_model_name>/<pretrained_model_name>.pk
    """
    path = os.path.join(
        pretrained_model_dir,
        pretrained_model_name,
        pretrained_model_name + ".pk",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: '{path}'. "
            "Check --pretrained_model_dir and --pretrained_model_name."
        )

    map_loc = {"cuda:%d" % 0: str(get_device())}
    info(f"[FT1] Loading pretrained weights from: {path}")
    ckpt = torch.load(path, map_location=map_loc)
    state_dict = ckpt["model_state_dict"]

    target = model.module if hasattr(model, "module") else model
    own_keys = set(target.state_dict().keys())
    if own_keys and not next(iter(own_keys)).startswith("module"):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        state_dict = new_sd

    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    # Graph head keys will be missing (new head); report for transparency
    head_keys   = [k for k in missing    if "heads_NN" in k or "graph_shared" in k]
    other_miss  = [k for k in missing    if k not in head_keys]
    info(
        f"[FT1] Loaded weights: {len(missing)} missing keys "
        f"({len(head_keys)} graph-head keys expected), "
        f"{len(unexpected)} unexpected keys."
    )
    if other_miss:
        info(f"[FT1]  Non-head missing keys: {other_miss[:5]}"
             f"{'...' if len(other_miss) > 5 else ''}")
    info("[FT1] Pretrained encoder loaded successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="FT1: fine-tune pretrained HydraGNN OPF model for feasibility classification.",
    )
    parser.add_argument(
        "--inputfile",
        type=str,
        default="FT1_feasibility_classification/config_HeteroSAGE_full.json",
        help="Fine-tuning config JSON (relative to this script's directory).",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default=None,
        help="Log/checkpoint name.  Defaults to a derived tag.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../dataset",
        help="Root directory containing the FT1 HDF5 dataset.",
    )
    parser.add_argument("--num_epoch",     type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="../pretrained_models",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        required=True,
        choices=["HeteroSAGE_best", "HeteroHEAT_best"],
    )
    parser.add_argument(
        "--finetune_regime",
        type=str,
        default="full",
        choices=["full", "partial", "head_only"],
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=False,
        help=(
            "Skip loading pretrained weights — train from random initialisation "
            "(baseline comparison). Implies --finetune_regime full."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "Truncate the training split to this many samples before training. "
            "Val/test splits are unchanged.  Used for data-efficiency sweeps."
        ),
    )

    args = parser.parse_args()
    if args.no_pretrained:
        args.finetune_regime = "full"  # no freezing for baseline

    # ── Resolve paths ──────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filename        = os.path.join(script_dir, args.inputfile)
    pretrained_model_dir  = os.path.join(script_dir, args.pretrained_model_dir)
    data_root             = os.path.join(script_dir, args.data_root)

    with open(input_filename) as f:
        config = json.load(f)

    # ── CLI overrides ──────────────────────────────────────────────────────
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = args.learning_rate

    arch_config = config["NeuralNetwork"]["Architecture"]
    edge_dim    = _resolve_edge_dim(config)

    # ── Distributed init ───────────────────────────────────────────────────
    hydragnn.utils.distributed.setup_ddp()

    # ── Log name ───────────────────────────────────────────────────────────
    ft_tag = f"{args.pretrained_model_name}_{args.finetune_regime}"
    log_name = args.modelname if args.modelname is not None else f"FT1_classify_{ft_tag}"

    # ── Load HDF5 dataset ──────────────────────────────────────────────────
    data_modelname = config.get("ft_data_modelname", "FT1_feasibility_data")
    basedir = os.path.join(data_root, f"{data_modelname}.h5")
    if not os.path.isdir(basedir):
        raise RuntimeError(
            f"FT1 dataset not found at '{basedir}'.\n"
            "Generate it first with:\n"
            "  python generate_infeasible_samples.py "
            "--src_dir ../dataset/<source>.h5 "
            "--out_dir ../dataset/FT1_feasibility_data.h5"
        )

    trainset = HDF5Dataset(basedir, "trainset")
    valset   = HDF5Dataset(basedir, "valset")
    testset  = HDF5Dataset(basedir, "testset")

    # Validate edge feature shapes (no node-level target injection needed —
    # data.y is already the graph-level label set by generate_infeasible_samples.py)
    trainset = EdgeAttrDatasetAdapter(trainset, edge_dim=edge_dim)
    valset   = EdgeAttrDatasetAdapter(valset,   edge_dim=edge_dim)
    testset  = EdgeAttrDatasetAdapter(testset,  edge_dim=edge_dim)

    # Optionally truncate trainset for data-efficiency sweep
    if args.max_train_samples is not None and len(trainset) > args.max_train_samples:
        from torch.utils.data import Subset

        trainset = Subset(trainset, list(range(args.max_train_samples)))
        info(
            f"Truncated trainset to {args.max_train_samples} samples "
            "for data-efficiency sweep."
        )

    info(
        "FT1 dataset sizes: train=%d  val=%d  test=%d"
        % (len(trainset), len(valset), len(testset))
    )
    sample0 = trainset[0]
    info(
        f"  Sample node types: {sample0.node_types}  |  "
        f"y = {sample0.y.tolist()}"
    )

    # ── DataLoaders ────────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )

    # ── Update config from data ────────────────────────────────────────────
    # update_config handles graph-level outputs without y_loc: it reads
    # output_dim directly from config["Variables_of_interest"]["output_dim"].
    config = update_config(config, train_loader, val_loader, test_loader)
    arch_config = config["NeuralNetwork"]["Architecture"]

    config = _to_jsonable(config)
    save_config(config, log_name)

    # ── Create model ───────────────────────────────────────────────────────
    node_input_dims = arch_config.get("node_input_dims")
    if node_input_dims is None:
        raise RuntimeError(
            "node_input_dims not found in config after update_config. "
            "Ensure the dataset contains HeteroData with node-type features."
        )

    try:
        metadata = trainset[0].metadata()
    except Exception:
        metadata = None

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
        metadata=metadata,
        node_input_dims=node_input_dims,
    )

    # ── Optionally load pretrained encoder weights ────────────────────────
    if not args.no_pretrained:
        load_pretrained_weights(model, args.pretrained_model_name, pretrained_model_dir)
    else:
        info("[FT1] --no_pretrained: starting from random initialisation (baseline).")

    # ── Apply freeze regime ────────────────────────────────────────────────
    apply_freeze_regime(model, args.finetune_regime)

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    info(
        f"[FT1] Regime '{args.finetune_regime}': "
        f"{n_trainable:,} / {n_total:,} parameters trainable "
        f"({100.0 * n_trainable / max(n_total, 1):.1f}%)"
    )

    lr = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── DDP wrap ───────────────────────────────────────────────────────────
    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer,
        config["Verbosity"]["level"],
        find_unused_parameters=(args.finetune_regime != "full"),
    )

    print_model(model)

    # ── TensorBoard + CSV training-curve writer ───────────────────────────
    _tb_writer = model_utils.get_summary_writer(log_name)
    _csv_path  = os.path.join("logs", log_name, "training_curve.csv")
    writer = EpochCSVWriter(_tb_writer, _csv_path)

    precision = config["NeuralNetwork"]["Training"].get("precision", "fp32")

    # ── Train ──────────────────────────────────────────────────────────────
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        config["Verbosity"]["level"],
        create_plots=False,
        precision=precision,
    )

    # ── Save checkpoint ────────────────────────────────────────────────────
    model_utils.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(config["Verbosity"]["level"])

    # ── Post-training evaluation ───────────────────────────────────────────
    device = get_device()
    info("[FT1] Running post-training classification evaluation on test set...")
    test_metrics = evaluate_ft1(model, test_loader, device, comm)

    if rank == 0 and test_metrics is not None:
        info(
            f"[FT1] Test metrics — "
            f"BCE={test_metrics['bce']:.4f}  "
            f"Acc={test_metrics['accuracy']:.4f}  "
            f"F1={test_metrics['f1']:.4f}  "
            f"AUC={test_metrics['auc_roc']}"
        )
        run_meta = {
            "ft_strategy": "FT1_feasibility_classification",
            "arch":         arch_config["mpnn_type"],
            "regime":       args.finetune_regime,
            "pretrained":   not args.no_pretrained,
            "pretrained_model": args.pretrained_model_name if not args.no_pretrained else "none",
            "num_epoch":    config["NeuralNetwork"]["Training"]["num_epoch"],
            "learning_rate": config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
            "config_file":  args.inputfile,
        }
        save_run_results(log_name, run_meta, test_metrics)

    comm.Barrier()

    writer.close()

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
