"""Fine-tuning script for HydraGNN OPF heterogeneous models.

Loads a pretrained HeteroSAGE or HeteroHEAT checkpoint and fine-tunes it on
a target OPF dataset.  Requires that the target dataset has already been
serialised to HDF5 using train_opf_solution_heterogeneous.py --preonly --hdf5.

Three fine-tuning regimes are supported:
  head_only  -- freeze all input projectors and all conv/feature layers;
                only the MLP prediction head is updated.
  partial    -- freeze input projectors and the first (num_conv_layers-1)
                conv/feature layers; the last conv layer and the prediction
                head are updated.
  full       -- all parameters are updated.

Example (FT1, topology-specific, full fine-tuning with HeteroSAGE):
    python train_opf_finetune.py \\
        --inputfile FT1_topology/config_HeteroSAGE_full.json \\
        --hdf5 \\
        --modelname FT1_case118_HeteroSAGE_full \\
        --pretrained_model_dir ../pretrained_models \\
        --pretrained_model_name HeteroSAGE_best \\
        --finetune_regime full \\
        --num_epoch 50 \\
        --learning_rate 5e-4
"""

import os
import sys
import json
import argparse
import shutil

# Make examples/opf importable
_OPF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _OPF_DIR)

import torch
import torch.distributed as dist
from mpi4py import MPI

import hydragnn
import hydragnn.utils.model as model_utils
from hydragnn.utils.model.model import load_existing_model
from hydragnn.utils.model import print_model
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.input_config_parsing import save_config

from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset

from ft_utils import EpochCSVWriter, evaluate_ft3, save_run_results

from opf_solution_utils import (
    EdgeAttrDatasetAdapter,
    OPFEnhancedModelWrapper,
    NodeBatchAdapter,
    NodeTargetDatasetAdapter,
    OPFDomainLoss,
    compute_pna_deg_for_hetero_dataset,
    validate_voi_node_features,
    info,
    resolve_node_target_type as _resolve_node_target_type,
)


# ---------------------------------------------------------------------------
# Utility helpers
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
    """Freeze model parameters according to the requested fine-tuning regime.

    Must be called *after* loading pretrained weights but *before* creating the
    optimizer so that frozen parameters are excluded from the parameter groups.

    Parameters
    ----------
    model : nn.Module
        The (not yet DDP-wrapped) model.
    regime : str
        One of 'full', 'partial', 'head_only'.
    """
    if regime == "full":
        return  # nothing to freeze

    # Unwrap OPFEnhancedModelWrapper
    inner = model
    if isinstance(inner, OPFEnhancedModelWrapper):
        inner = inner.model

    if regime == "head_only":
        # Freeze everything except the prediction heads (heads_NN).
        for name, param in inner.named_parameters():
            if "heads_NN" not in name:
                param.requires_grad_(False)
        n_trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        info(f"[FT] head_only: {n_trainable:,} trainable parameters (prediction heads only)")
        return

    if regime == "partial":
        # Freeze input projectors and all but the last conv/feature layer pair.
        for name, param in inner.named_parameters():
            if "input_projectors" in name or "node_embedders" in name:
                param.requires_grad_(False)

        n_conv = len(inner.graph_convs)
        for i, (conv, feat) in enumerate(
            zip(inner.graph_convs, inner.feature_layers)
        ):
            if i < n_conv - 1:  # freeze all but the last conv layer
                for p in conv.parameters():
                    p.requires_grad_(False)
                for p in feat.parameters():
                    p.requires_grad_(False)

        n_trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        info(
            f"[FT] partial: {n_trainable:,} trainable parameters "
            f"(last conv layer + heads; {n_conv - 1} of {n_conv} conv layers frozen)"
        )
        return

    raise ValueError(
        f"Unknown finetune_regime '{regime}'. Choose from: full, partial, head_only."
    )


def load_pretrained_weights(model, pretrained_model_name: str, pretrained_model_dir: str):
    """Load model weights from a pretrained checkpoint, discarding optimizer state.

    Expects the checkpoint at:
        <pretrained_model_dir>/<pretrained_model_name>/<pretrained_model_name>.pk
    """
    from collections import OrderedDict

    path_name = os.path.join(
        pretrained_model_dir, pretrained_model_name, pretrained_model_name + ".pk"
    )
    if not os.path.isfile(path_name):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{path_name}'. "
            "Verify --pretrained_model_dir and --pretrained_model_name."
        )

    map_location = {"cuda:%d" % 0: str(get_device())}
    info(f"[FT] Loading pretrained weights from: {path_name}")
    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]

    # Unwrap DDP prefix if needed
    target = model.module if hasattr(model, "module") else model
    own_keys = set(target.state_dict().keys())
    if own_keys and not next(iter(own_keys)).startswith("module"):
        # Remove 'module.' prefix coming from a DDP-saved state dict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        state_dict = new_sd

    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    if missing:
        info(f"[FT] WARNING: missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        info(f"[FT] WARNING: unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    info("[FT] Pretrained weights loaded successfully.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fine-tune a pretrained HydraGNN OPF model.",
    )
    # ── Dataset arguments ──────────────────────────────────────────────────
    parser.add_argument(
        "--inputfile",
        type=str,
        default="FT1_topology/config_HeteroSAGE_full.json",
        help="Path to the fine-tuning JSON config (relative to this script's directory).",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default=None,
        help="Log/checkpoint name for this fine-tuning run. Defaults to the config key.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../dataset",
        help="Root directory containing pre-serialised HDF5 data.",
    )
    parser.add_argument(
        "--node_target_type",
        type=str,
        default="bus",
        choices=["bus", "generator"],
    )
    # ── Data format ────────────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hdf5", action="store_const", dest="format", const="hdf5")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="hdf5")

    # ── Training overrides ─────────────────────────────────────────────────
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    # ── Fine-tuning specific ───────────────────────────────────────────────
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="../pretrained_models",
        help="Directory containing pretrained model subdirectories.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        required=True,
        choices=["HeteroSAGE_best", "HeteroHEAT_best"],
        help="Name of the pretrained model (must match a subdirectory in pretrained_model_dir).",
    )
    parser.add_argument(
        "--finetune_regime",
        type=str,
        default="full",
        choices=["full", "partial", "head_only"],
        help=(
            "Freeze regime: "
            "'full' trains all parameters, "
            "'partial' trains last conv layer + head, "
            "'head_only' trains only the prediction head."
        ),
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
        args.finetune_regime = "full"

    # ── Resolve paths ──────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(script_dir, args.inputfile)
    pretrained_model_dir = os.path.join(script_dir, args.pretrained_model_dir)
    data_root = os.path.join(script_dir, args.data_root)

    with open(input_filename) as f:
        config = json.load(f)

    # ── Apply CLI overrides ────────────────────────────────────────────────
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = args.learning_rate

    arch_config = config["NeuralNetwork"]["Architecture"]

    edge_dim = _resolve_edge_dim(config)

    # ── Distributed initialisation ─────────────────────────────────────────
    hydragnn.utils.distributed.setup_ddp()

    # ── Log name ───────────────────────────────────────────────────────────
    ft_tag = f"{args.pretrained_model_name}_{args.finetune_regime}"
    if args.modelname is not None:
        log_name = args.modelname
    else:
        log_name = f"finetune_{ft_tag}"

    # ── Load serialised datasets ───────────────────────────────────────────
    # The dataset is expected to have been serialised by the preprocessing
    # step of train_opf_solution_heterogeneous.py --preonly --hdf5.
    # The modelname used during preprocessing is stored in the config as
    # 'ft_data_modelname' or falls back to 'OPF_Solution_Hetero'.
    data_modelname = config.get("ft_data_modelname", "OPF_Solution_Hetero")
    basedir = os.path.join(data_root, f"{data_modelname}.h5")
    if not os.path.isdir(basedir):
        raise RuntimeError(
            f"Pre-serialised HDF5 dataset not found at '{basedir}'. "
            "Run the preprocessing step first:\n"
            "  python ../train_opf_solution_heterogeneous.py "
            "--preonly --hdf5 --case_name <case> [--max_samples N]"
        )
    trainset = HDF5Dataset(basedir, "trainset")
    valset = HDF5Dataset(basedir, "valset")
    testset = HDF5Dataset(basedir, "testset")

    # ── Adapt datasets ─────────────────────────────────────────────────────
    resolved_node_target_type = _resolve_node_target_type(
        trainset[0], args.node_target_type
    )
    args.node_target_type = resolved_node_target_type
    config["NeuralNetwork"]["Architecture"]["node_target_type"] = args.node_target_type
    validate_voi_node_features(config, args.node_target_type)

    trainset = EdgeAttrDatasetAdapter(trainset, edge_dim=edge_dim)
    valset   = EdgeAttrDatasetAdapter(valset,   edge_dim=edge_dim)
    testset  = EdgeAttrDatasetAdapter(testset,  edge_dim=edge_dim)

    trainset = NodeTargetDatasetAdapter(trainset, args.node_target_type, edge_dim=edge_dim)
    valset   = NodeTargetDatasetAdapter(valset,   args.node_target_type, edge_dim=edge_dim)
    testset  = NodeTargetDatasetAdapter(testset,  args.node_target_type, edge_dim=edge_dim)

    # Optionally truncate trainset for data-efficiency sweep
    if args.max_train_samples is not None and len(trainset) > args.max_train_samples:
        from torch.utils.data import Subset

        trainset = Subset(trainset, list(range(args.max_train_samples)))
        info(
            f"Truncated trainset to {args.max_train_samples} samples "
            "for data-efficiency sweep."
        )

    info(
        "trainset / valset / testset sizes: %d / %d / %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )
    train_loader = NodeBatchAdapter(train_loader, args.node_target_type, edge_dim=edge_dim)
    val_loader   = NodeBatchAdapter(val_loader,   args.node_target_type, edge_dim=edge_dim)
    test_loader  = NodeBatchAdapter(test_loader,  args.node_target_type, edge_dim=edge_dim)

    config = update_config(config, train_loader, val_loader, test_loader)
    arch_config = config["NeuralNetwork"]["Architecture"]

    if arch_config.get("mpnn_type") == "HeteroPNA" and not arch_config.get("pna_deg"):
        pna_deg = compute_pna_deg_for_hetero_dataset(trainset, verbosity=2)
        arch_config["pna_deg"] = pna_deg
        arch_config["max_neighbours"] = max(0, len(pna_deg) - 1)

    config = _to_jsonable(config)
    save_config(config, log_name)

    # ── Create model ───────────────────────────────────────────────────────
    node_input_dims = arch_config.get("node_input_dims")
    if node_input_dims is None:
        raise RuntimeError("Missing NeuralNetwork.Architecture.node_input_dims in config.")

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

    # ── Optionally wrap with domain-loss ───────────────────────────────────
    domain_loss_config = config["NeuralNetwork"]["Training"].get("DomainLoss")
    if domain_loss_config is not None and domain_loss_config.get("enabled", False):
        model = OPFEnhancedModelWrapper(
            model,
            OPFDomainLoss(domain_loss_config, node_target_type=args.node_target_type),
        )

    # ── Load pretrained weights (before freezing, before optimizer) ────────
    if not args.no_pretrained:
        load_pretrained_weights(model, args.pretrained_model_name, pretrained_model_dir)
    else:
        info("[FT] --no_pretrained: starting from random initialisation (baseline).")

    # ── Apply freeze regime ────────────────────────────────────────────────
    apply_freeze_regime(model, args.finetune_regime)

    # ── Create optimizer over trainable parameters only ────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    info(
        f"[FT] Regime '{args.finetune_regime}': "
        f"{n_trainable:,} / {n_total:,} parameters trainable "
        f"({100.0 * n_trainable / max(n_total, 1):.1f}%)"
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Wrap in DDP ────────────────────────────────────────────────────────
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

    # ── Flush domain-loss log ──────────────────────────────────────────────
    _inner = model.module if hasattr(model, "module") else model
    if isinstance(_inner, OPFEnhancedModelWrapper):
        _inner._flush_epoch_log(_inner._last_seen_epoch)

    # ── Save final checkpoint ──────────────────────────────────────────────
    model_utils.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(config["Verbosity"]["level"])

    # ── Post-training regression evaluation ───────────────────────────────
    device = get_device()
    _cfg_voi = config["NeuralNetwork"]["Variables_of_interest"]
    _output_names = _cfg_voi.get("output_names", None)
    if _output_names and len(_output_names) == 1:
        _out_dim = arch_config.get("output_dim", [2])[0]
        if args.node_target_type == "bus":
            _out_names = ["Va", "Vm"][:_out_dim]
        else:
            _out_names = ["Pg", "Qg"][:_out_dim]
    else:
        _out_names = _output_names or ["dim_0", "dim_1"]

    info(f"[FT] Post-training regression eval on test set (dims: {_out_names})...")
    test_metrics = evaluate_ft3(model, test_loader, device, comm,
                                output_names=_out_names)

    if rank == 0 and test_metrics is not None:
        _mse_strs = ", ".join(
            f"{n}_MSE={test_metrics.get(n + '_mse', float('nan')):.5f}"
            for n in _out_names
        )
        info(f"[FT] Test metrics — {_mse_strs}  overall_MSE={test_metrics['overall_mse']:.5f}")
        run_meta = {
            "ft_strategy":      config.get("_ft_strategy", "unknown"),
            "arch":             arch_config["mpnn_type"],
            "regime":           args.finetune_regime,
            "pretrained":       not args.no_pretrained,
            "pretrained_model": args.pretrained_model_name if not args.no_pretrained else "none",
            "node_target_type": args.node_target_type,
            "num_epoch":        config["NeuralNetwork"]["Training"]["num_epoch"],
            "learning_rate":    config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
            "config_file":      args.inputfile,
        }
        save_run_results(log_name, run_meta, test_metrics)

    comm.Barrier()

    writer.close()

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
