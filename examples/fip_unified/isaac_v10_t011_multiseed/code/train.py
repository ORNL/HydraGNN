"""Clean train entry point for cp3d_v2.

Usage:
    from train import train_one
    metrics = train_one(seed=42, exp_tag="floor_b2", size_bucket=2,
                        config_overrides={...})

What this owns:
  - Build a HydraGNN config dict from a minimal recipe + overrides.
  - Construct the dataset (via FIPDataset), splits, scalers.
  - Build the model via hydragnn.models.create_model_config.
  - Build the optimizer + LR scheduler (cosine_warmup default).
  - Run training via hydragnn.train.train_validate_test (handles best-val
    checkpointing + early stopping + per-epoch LR step).
  - After training: load best-val checkpoint, evaluate on raw-FIP metrics,
    save results/{tag}_seed{seed}/{run_metrics.json, parity_*.png,
    loss_history.png, per_size_breakdown.csv, test_predictions.csv,
    split_indices.json}.

Hyperparameters NOT assumed without justification:
  - Pool, weight_decay, target transform, slip_transfer, etc. are all knobs
    in `config_overrides`. The DEFAULTS below are the simplest defensible
    values per the plan's Phase-1 floor.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import train_validate_test
from hydragnn.utils.model.model import (
    save_model,
    load_existing_model,
    get_summary_writer,
)
from hydragnn.utils.input_config_parsing.config_utils import update_config

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# Study writes logs/ and results/ at the STUDY root, NOT under code/.
# Matches the layout expected by analyze_v4_study.py and submit_isaac.sbatch.
_STUDY_ROOT = HERE.parent
from dataset import FIPDataset, X_COL_LAYOUT  # noqa: E402
from plots import (
    parity_plot,
    loss_history_plot,
    loss_history_per_head_plot,
    pred_vs_true_dist_overlay_plot,
    per_grain_diagnostic_plot,
    parity_grid_plot,
    dist_overlay_grid_plot,
    hotspot_diagnostic_grid_plot,
)  # noqa: E402
from augment import FCCSymAugmentDataset, SO3AugmentDataset  # noqa: E402
from losses import override_model_loss, install_batch_stash_hook  # noqa: E402

BUCKET_TO_LSVE_LABEL = {0: "125 um", 1: "177 um", 2: "250 um", 3: "354 um", 4: "500 um"}

# v9 result-naming convention (refactor 2026-05-06): symmetric 4-cell layout,
# drop "_aux" suffix, group plots into multi-panel grids. See
# `feedback_v9_result_naming_2026-05-06.md` for the contract.
_V9_CELL_KEYS = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
_V9_SPLITS = ("train", "val", "test")

# Mapping from per-grain aux head name → (cell, stat). Slot 0 is implicit
# (lcf_fs_max). Aux entries come from `_V9_PERGRAIN_QOIS_7` in manifest.
_V9_PN_AUX_TO_CELL_STAT: Dict[str, tuple] = {
    "lcf_wp_max": ("lcf_wp", "max"),
    "hcf_fs_max": ("hcf_fs", "max"),
    "hcf_wp_max": ("hcf_wp", "max"),
    "lcf_fs_within_mean": ("lcf_fs", "mean"),
    "lcf_wp_within_mean": ("lcf_wp", "mean"),
    "hcf_fs_within_mean": ("hcf_fs", "mean"),
    "hcf_wp_within_mean": ("hcf_wp", "mean"),
    # Phase M (2026-05-06) — distribution-shape per-grain targets.
    "lcf_fs_logmu_voxel": ("lcf_fs", "logmu"),
    "lcf_wp_logmu_voxel": ("lcf_wp", "logmu"),
    "hcf_fs_logmu_voxel": ("hcf_fs", "logmu"),
    "hcf_wp_logmu_voxel": ("hcf_wp", "logmu"),
    "lcf_fs_logsigma_voxel": ("lcf_fs", "logsigma"),
    "lcf_wp_logsigma_voxel": ("lcf_wp", "logsigma"),
    "hcf_fs_logsigma_voxel": ("hcf_fs", "logsigma"),
    "hcf_wp_logsigma_voxel": ("hcf_wp", "logsigma"),
    "lcf_fs_p90": ("lcf_fs", "p90"),
    "lcf_wp_p90": ("lcf_wp", "p90"),
    "hcf_fs_p90": ("hcf_fs", "p90"),
    "hcf_wp_p90": ("hcf_wp", "p90"),
    "lcf_fs_p99": ("lcf_fs", "p99"),
    "lcf_wp_p99": ("lcf_wp", "p99"),
    "hcf_fs_p99": ("hcf_fs", "p99"),
    "hcf_wp_p99": ("hcf_wp", "p99"),
}

# Distribution-shape stat keys (used by wide CSV writer + plot helpers).
_V9_DISTSHAPE_STATS = ("logmu", "logsigma", "p90", "p99")

# Mapping from per-graph aux head name → (cell, stat). Slot 0 is implicit
# (lcf_fs pg-max). Aux entries from `_V8_PERGRAPH_QOIS_12`.
_V9_PG_AUX_TO_CELL_STAT: Dict[str, tuple] = {
    "lcf_wp_pergraph_max": ("lcf_wp", "max"),
    "hcf_fs_pergraph_max": ("hcf_fs", "max"),
    "hcf_wp_pergraph_max": ("hcf_wp", "max"),
    "lcf_fs_ve_mu_pergraph": ("lcf_fs", "ve_mu"),
    "lcf_wp_ve_mu_pergraph": ("lcf_wp", "ve_mu"),
    "hcf_fs_ve_mu_pergraph": ("hcf_fs", "ve_mu"),
    "hcf_wp_ve_mu_pergraph": ("hcf_wp", "ve_mu"),
}

# v9b mech QoI redesign (user 2026-05-06). Six entries, paper-clean:
#   monotonic: yield_stress (σ_y), hardening_modulus
#   LCF cycle3: Wp_lcf (plastic energy), sigma_amp_lcf (stress amplitude)
#   HCF cycle3: Wp_hcf (plastic energy), sigma_amp_hcf (stress amplitude)
_V9_MECH_QOI_NAMES = (
    "yield_stress",
    "hardening_modulus",
    "Wp_lcf",
    "Wp_hcf",
    "sigma_amp_lcf",
    "sigma_amp_hcf",
)


# ---------------------------------------------------------------------------
# Phase L4 — learnable residual / skip-connection on the per-grain MLP head.
# ---------------------------------------------------------------------------
# Diagnosis. v9 Phase L1/L3 parity plots show the per-grain head spreads its
# bulk predictions reasonably but caps the upper tail at ~0.85 × true p99.
# Hypothesis: gradient flow through the deep MLP head bottleneck attenuates
# tail signal. A linear skip (encoder hidden → output) gives the head a free
# "identity-equivalent at init" path: the wrapper learns RESIDUAL corrections
# rather than the full mapping.
#
# Initialised so the wrapper is exactly the base MLPNode at epoch 0 (skip
# weight = zeros). Becomes nonzero only as training proceeds. No effect on
# bulk parameters; the only new params are `hidden_dim × output_dim`
# weights for the skip projection (e.g. 48 × 8 = 384 floats).
#
# Usage. Set `Recipe.use_residual_node_head = True`. v9 Phase L4 cells use
# this combined with the L3 winner loss (mc_huber or asym_huber, etc).
import torch as _torch_for_residual


class _ResidualMLPNodeWrapper(_torch_for_residual.nn.Module):
    """Wrap an MLPNode with a learnable linear skip from input → output.

    Module-dict registration: assigning the wrapper into the parent
    `head_dict[branchtype]` slot makes torch register both the base head
    and the new skip weights as model parameters. The optimiser must be
    built AFTER wrapping or the skip weights won't be optimised — see
    `_install_residual_node_head` for the install sequence.
    """

    def __init__(self, base_head, input_dim: int, output_dim: int):
        super().__init__()
        self.base_head = base_head
        self.skip = _torch_for_residual.nn.Linear(input_dim, output_dim, bias=False)
        _torch_for_residual.nn.init.zeros_(self.skip.weight)

    def forward(self, x, batch):
        return self.base_head(x, batch) + self.skip(x)


def _install_residual_node_head(model) -> int:
    """Walk model.heads_NN, wrap every MLPNode with a linear skip. Return
    the number of heads wrapped (0 if model has no MLPNode heads).

    Must be called BEFORE `build_optimizer_and_scheduler` so the new skip
    parameters are registered with the optimiser. The wrapper's freshly-
    constructed Linear layer lives on CPU by default; we move it onto the
    base head's device so a CUDA-trained model doesn't hit a cross-device
    mat-mul error on the first forward pass.
    """
    inner = model.module if hasattr(model, "module") else model
    if not hasattr(inner, "heads_NN"):
        return 0
    try:
        from hydragnn.models.Base import MLPNode
    except ImportError:
        return 0
    n_wrapped = 0
    for head_dict in inner.heads_NN:
        # head_dict is a ModuleDict (branchtype → module).
        for branchtype, head_module in list(head_dict.items()):
            if isinstance(head_module, MLPNode):
                # Detect base head's device + dtype.
                try:
                    base_param = next(head_module.parameters())
                    device = base_param.device
                    dtype = base_param.dtype
                except StopIteration:
                    device = _torch_for_residual.device("cpu")
                    dtype = _torch_for_residual.float32
                wrapped = _ResidualMLPNodeWrapper(
                    head_module,
                    int(head_module.input_dim),
                    int(head_module.output_dim),
                ).to(device=device, dtype=dtype)
                # ModuleDict supports item assignment; this re-registers params.
                head_dict[branchtype] = wrapped
                n_wrapped += 1
    return n_wrapped


# ---------------------------------------------------------------------------
# Loss-history capture — wraps SummaryWriter so we don't depend on HydraGNN's
# brittle "./logs/{name}/history_loss.pckl" path (which is only written when
# create_plots=True and uses cwd-relative resolution).
# ---------------------------------------------------------------------------
class LossHistoryWriter:
    """Captures train/validate/test error per epoch, then forwards everything
    to the underlying TensorBoard SummaryWriter.

    Also captures HydraGNN per-task losses (`train error of task<N>` tags) for
    multi-head loss diagnostics. Per-task losses are train-side only because
    HydraGNN does not emit val/test per-task to the writer (only printed)."""

    KNOWN_TAGS = {"train error": "train", "validate error": "val", "test error": "test"}
    PER_TASK_PREFIX = "train error of task"

    def __init__(self, summary_writer):
        self._w = summary_writer
        self.history: Dict[str, Dict[int, float]] = {"train": {}, "val": {}, "test": {}}
        self.task_history: Dict[int, Dict[int, float]] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        if step is not None:
            if tag in self.KNOWN_TAGS:
                try:
                    self.history[self.KNOWN_TAGS[tag]][int(step)] = float(value)
                except Exception:
                    pass
            elif tag.startswith(self.PER_TASK_PREFIX):
                try:
                    task_id = int(tag[len(self.PER_TASK_PREFIX) :])
                    self.task_history.setdefault(task_id, {})[int(step)] = float(value)
                except Exception:
                    pass
        return self._w.add_scalar(tag, value, step, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._w, name)

    def flat(self, key: str) -> List[float]:
        d = self.history.get(key, {})
        if not d:
            return []
        n = max(d) + 1
        out = [0.0] * n
        for k, v in d.items():
            out[k] = v
        return out

    def task_flat(self, task_id: int) -> List[float]:
        d = self.task_history.get(task_id, {})
        if not d:
            return []
        n = max(d) + 1
        out = [float("nan")] * n
        for k, v in d.items():
            out[k] = v
        return out

    def task_ids(self) -> List[int]:
        return sorted(self.task_history.keys())


# ---------------------------------------------------------------------------
# Default config (the Phase-1 floor — every choice has a justification)
# ---------------------------------------------------------------------------
def default_config() -> Dict[str, Any]:
    return {
        "Verbosity": {"level": 1},
        "NeuralNetwork": {
            "Architecture": {
                "mpnn_type": "PNA",  # user pref: keep mpnn fixed
                "radius": 7,  # cp_config.json default
                "max_neighbours": 30,
                "periodic_boundary_conditions": False,
                "hidden_dim": 32,  # plan: smaller default for per-size
                "num_conv_layers": 3,  # plan: smaller default
                "num_radial": 6,
                "num_gaussians": 50,
                "envelope_exponent": 5,
                "activation_function": "relu",
                "edge_features": [
                    "misorientation_angle",
                    "edge_length",
                    "relpos_crystal_x",
                    "relpos_crystal_y",
                    "relpos_crystal_z",
                    "is_periodic",
                ],
                "graph_pooling": "max",  # pool=sum is known-bad (Phase 1.0); ABLATION knob
                "global_attn_engine": "",
                "global_attn_type": "",
                "pe_dim": 0,
                "global_attn_heads": 1,
            },
            "Training": {
                "num_epoch": 200,  # 200 enough for per-size 160-train
                "perc_train": 0.8,
                "loss_function_type": "huber",
                "batch_size": 16,
                "EarlyStopping": True,
                "patience": 100,  # cp_config.json default; ABLATION knob
                "Checkpoint": False,  # disabled for HPO: we don't load .pk back; prevents per-epoch *.pk accumulation
                "checkpoint_warmup": 5,
                "Optimizer": {
                    "type": "AdamW",
                    "learning_rate": 1.0e-3,
                },
                "weight_decay": 1.0e-4,  # ABLATION knob
                "lr_scheduler": "reduce_on_plateau",  # "reduce_on_plateau" | "cosine_warmup"
                "lr_scheduler_factor": 0.5,
                "lr_scheduler_patience": 20,
                "lr_scheduler_min_lr": 1.0e-6,
            },
            "Variables_of_interest": {},  # filled in build_model_config
        },
        "Visualization": {
            "plot_init_solution": False,
            "plot_hist_solution": False,
            "create_plots": False,  # we make our own plots
        },
        # cp3d_v2-specific knobs (NOT passed to HydraGNN):
        "Recipe": {
            "size_bucket": 2,  # 250 um (Phase 1.0 default)
            "use_per_grain_target": True,
            "use_per_graph_target": True,
            "node_task_weight": 0.5,  # ABLATION knob
            "per_grain_transform": "log1p",  # ABLATION knob
            "per_graph_transform": "log1p",
            "split_fracs": [
                0.80,
                0.10,
                0.10,
            ],  # ABLATION knob (Phase 1.2: try 0.90/0.05/0.05)
            "dropout": 0.0,  # ABLATION knob (Phase 1.2: try 0.2)
            "graph_head_dims": {  # ABLATION knob — override the output_heads.graph spec
                "num_sharedlayers": 2,
                "dim_sharedlayers": 16,
                "num_headlayers": 2,
                "dim_headlayers": [16, 8],
            },
            "use_x_blocks": [
                "quat",
                "schmid",
                "volume",
                "surface_area",
                "perimeter",
                "degree",
                "Iuu",
                "Ivv",
                "Iuv",
                "J_polar",
                "I1",
                "I2",
            ],
            "use_edge_cols": [
                "misorientation_angle",
                "edge_length",
                "relpos_crystal_x",
                "relpos_crystal_y",
                "relpos_crystal_z",
                "is_periodic",
            ],
            "drop_periodic_edges": False,
            # Phase 2 (global model): condition on SVE size as a graph_attr.
            "graph_attr_source": None,  # "L_SVE_um" | "log_n_nodes" | "size_bucket_onehot" | None
            "graph_attr_conditioning_mode": "concat_node",  # "concat_node" | "fuse_pool" | "film"
            "stratify_by_bucket": True,  # global mode: stratify split by SVE bucket
            "split_seed": None,  # if set, override `seed` for splits — enables true ensembling
            "train_buckets": None,  # extrapolation: list of bucket ids to use for train+val
            "test_buckets": None,  # extrapolation: list of bucket ids to use for test (None = the rest)
            "target_residual_baseline": None,  # None | "per_bucket_mean" | "log_n_ols"
            "per_graph_target_aggr": "max",  # "max" | "P99" | "P95" | "P90" | "mean"
            "use_fcc_sym_augment": False,  # train-time FCC cubic-sym augmentation
            "use_so3_augment": False,  # train-time random SO(3) sample-frame rotation
            "per_grain_target_col": 0,  # 0 = per-grain MAX (ours), 1 = per-grain MEAN (Sim et al)
            "loss_recipe": None,  # None|"rank_huber_05"|"peak_huber_topk_05_15"|...
            # see examples/cp3d_v2/losses.py
        },
    }


def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge b into a (b wins). Returns merged copy."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Output heads spec — built from dataset config
# ---------------------------------------------------------------------------
def build_var_config_and_heads(cfg: dict) -> dict:
    """Populate Variables_of_interest + output_heads + task_weights from
    dataset config. Mirrors what cp3d/train_pre.py:build_var_config_and_heads
    does, but minimal."""
    arch = cfg["NeuralNetwork"]["Architecture"]
    ds = cfg["Recipe"]

    # Input node feature size
    n_input = sum(
        X_COL_LAYOUT[b].stop - X_COL_LAYOUT[b].start for b in ds["use_x_blocks"]
    )

    # Build target-aware head labels so the saved metrics, configs, and
    # output-name registries use a name that reflects the actual target.
    target_field = ds.get("per_grain_target_field", "fs_fip")
    # Map per_grain_target_field to a fully-qualified prefix that includes
    # both loading regime (lcf / hcf) and FIP variant (fs / wp). v4's
    # `fs_fip` and `wp_fip` are LCF tensors carried over from v3 naming;
    # the *_fip_dist tensors are explicit. Renaming the primary head from
    # `fs_max_*` to `lcf_fs_max_*` (etc.) prevents confusion with the
    # cross-variant per-graph aux targets `lcf_wp_pergraph_max`,
    # `hcf_fs_pergraph_max`, `hcf_wp_pergraph_max`.
    _TARGET_PREFIX = {
        "fs_fip": "lcf_fs",
        "wp_fip": "lcf_wp",
        "lcf_fs_fip_dist": "lcf_fs",
        "lcf_wp_fip_dist": "lcf_wp",
        "hcf_fs_fip_dist": "hcf_fs",
        "hcf_wp_fip_dist": "hcf_wp",
    }
    target_short = _TARGET_PREFIX.get(
        target_field, target_field.replace("_fip_dist", "").replace("_fip", "")
    )
    # v9b deslot: head names are CELL-NEUTRAL. Slot ordering inside each head
    # is communicated via `per_grain_targets` / `pergraph_targets` lists in
    # the recipe, NOT via the head name. (Pre-deslot: `lcf_fs_max_pernode`
    # implied that slot 0 was the privileged primary; symmetric framing
    # rejects that.)
    pernode_name = "pn_targets"
    pergraph_name = "pg_targets"
    common_name = "pn_targets"

    # Output head spec: one node head (per-grain MAX) + one graph head
    # (per-graph MAX); names are derived from the per-grain target field.
    output_names: List[str] = []
    output_dim: List[int] = []
    output_index: List[int] = []
    output_type: List[str] = []
    task_weights: List[float] = []
    output_heads: Dict[str, dict] = {}

    if ds["use_per_grain_target"]:
        # v9b deslot: per-grain head emits len(per_grain_targets) channels —
        # ALL first-class. Slot ordering inside the head is recorded by the
        # `per_grain_targets` list (the recipe owns the ordering); the
        # internal "slot 0 from per_grain_target_field + slots 1+ from
        # per_grain_aux_targets" mechanism is hidden in FIPDataset.__post_init__.
        # Phase L5 het: 4 extra σ output channels paired with the first 4
        # supervised channels (which by deslot convention are the 4 SVE-MAX
        # cells in lcf_fs/lcf_wp/hcf_fs/hcf_wp order). σ slots are
        # unsupervised — `het_nll_*` loss handles them.
        n_per_grain_targets = len(ds.get("per_grain_targets") or [])
        if n_per_grain_targets == 0:
            # Legacy slotted API fallback (slot 0 + aux list).
            n_per_grain_targets = 1 + len(ds.get("per_grain_aux_targets") or [])
        het = bool(ds.get("use_heteroscedastic_max", False))
        n_sigma = 4 if het else 0
        pn_head_dim = n_per_grain_targets + n_sigma
        output_names.append(pernode_name)
        output_dim.append(pn_head_dim)
        output_index.append(0)
        output_type.append("node")
        task_weights.append(float(ds["node_task_weight"]))
        nhd = ds.get("node_head_dims", {}) or {}
        output_heads["node"] = {
            "type": "mlp",
            "num_headlayers": int(nhd.get("num_headlayers", 2)),
            "dim_headlayers": list(nhd.get("dim_headlayers", [16, 8])),
        }

    if ds["use_per_graph_target"]:
        # v9b deslot: per-graph head emits len(pergraph_targets) channels.
        # Phase L6 het: 4 extra σ output channels paired with the first 4
        # supervised pg channels (the 4 SVE-MAX cells per the deslot ordering).
        n_pergraph_targets = len(ds.get("pergraph_targets") or [])
        if n_pergraph_targets == 0:
            n_pergraph_targets = 1 + len(ds.get("aux_pergraph_targets") or [])
        het_pg = bool(ds.get("use_heteroscedastic_pergraph_max", False))
        n_sigma_pg = 4 if het_pg else 0
        pg_head_dim = n_pergraph_targets + n_sigma_pg
        output_names.append(pergraph_name)
        output_dim.append(pg_head_dim)
        output_index.append(0)
        output_type.append("graph")
        task_weights.append(1.0)
        ghd = ds.get("graph_head_dims", {})
        output_heads["graph"] = {
            "num_sharedlayers": int(ghd.get("num_sharedlayers", 2)),
            "dim_sharedlayers": int(ghd.get("dim_sharedlayers", 16)),
            "num_headlayers": int(ghd.get("num_headlayers", 2)),
            "dim_headlayers": list(ghd.get("dim_headlayers", [16, 8])),
        }

    # HydraGNN's parse_feature_config (newer versions) requires either
    # `node_features`/`graph_features` (new dict format) or `*_feature_names`
    # at the Variables_of_interest level. Write the new-format keys here so
    # both the strict check and the legacy `output_names`/`output_dim` path
    # are satisfied.
    var_node_features: Dict[str, dict] = {
        "x_input": {"role": "input", "dim": int(n_input), "output_type": "node"},
    }
    var_graph_features: Dict[str, dict] = {}
    for nm, dim, idx, typ in zip(output_names, output_dim, output_index, output_type):
        if typ == "node":
            var_node_features[nm] = {
                "role": "output",
                "dim": int(dim),
                "output_type": "node",
            }
        else:
            var_graph_features[nm] = {
                "role": "output",
                "dim": int(dim),
                "output_type": "graph",
            }
    cfg["NeuralNetwork"]["Variables_of_interest"] = {
        "input_node_features": list(range(n_input)),
        "output_names": output_names,
        "output_dim": output_dim,
        "output_index": output_index,
        "type": output_type,
        "node_features": var_node_features,
        "graph_features": var_graph_features,
    }
    arch["input_dim"] = n_input
    arch["output_heads"] = output_heads
    arch["task_weights"] = task_weights
    arch["edge_dim"] = len(ds["use_edge_cols"]) if ds["use_edge_cols"] else 8

    # HydraGNN's update_config consults cfg["Dataset"]["{node,graph}_features"]
    # for output-dim consistency checks. v9b deslot: derive dims from the
    # symmetric target lists when present; otherwise fall back to the legacy
    # slot-0 + aux split for older configs.
    n_per_grain = len(ds.get("per_grain_targets") or [])
    if n_per_grain == 0:
        n_per_grain = 1 + len(ds.get("per_grain_aux_targets") or [])
    n_pergraph = len(ds.get("pergraph_targets") or [])
    if n_pergraph == 0:
        n_pergraph = 1 + len(ds.get("aux_pergraph_targets") or [])
    n_sigma_pn = 4 if bool(ds.get("use_heteroscedastic_max", False)) else 0
    n_sigma_pg = 4 if bool(ds.get("use_heteroscedastic_pergraph_max", False)) else 0
    cfg["Dataset"] = {
        "node_features": {"name": [common_name], "dim": [n_per_grain + n_sigma_pn]},
        "graph_features": {"name": [pergraph_name], "dim": [n_pergraph + n_sigma_pg]},
    }
    return cfg


# ---------------------------------------------------------------------------
# LR scheduler factory (cosine_warmup or reduce_on_plateau)
# ---------------------------------------------------------------------------
class _LRStepShim:
    """Uniform .step(val_loss) interface so hydragnn's train loop works."""

    def __init__(self, sched, optimizer, recipe):
        self._s = sched
        self._opt = optimizer
        self._recipe = recipe

    def step(self, val_loss=None):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(self._s, ReduceLROnPlateau):
            self._s.step(val_loss)
        else:
            self._s.step()

    def state_dict(self):
        return self._s.state_dict()

    def load_state_dict(self, sd):
        self._s.load_state_dict(sd)


def build_optimizer_and_scheduler(model, cfg):
    train = cfg["NeuralNetwork"]["Training"]
    lr = float(train["Optimizer"]["learning_rate"])
    wd = float(train.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    recipe = train.get("lr_scheduler", "reduce_on_plateau")
    n_epoch = int(train["num_epoch"])

    if recipe == "cosine_warmup":
        warmup_n = max(1, int(0.05 * n_epoch))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_n,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_epoch - warmup_n),
            eta_min=1e-6,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_n],
        )
    else:  # reduce_on_plateau (default)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train.get("lr_scheduler_factor", 0.5)),
            patience=int(train.get("lr_scheduler_patience", 20)),
            min_lr=float(train.get("lr_scheduler_min_lr", 1e-6)),
        )
    return optimizer, _LRStepShim(sched, optimizer, recipe)


# ---------------------------------------------------------------------------
# Eval / metrics
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    # NaN/Inf-tolerant: drop predictions that are NaN or Inf and report the
    # count separately. yeo_johnson inverse on out-of-distribution
    # auxiliary-head predictions can produce NaN; rather than crash the
    # entire eval (and lose run_metrics.json for that rep), we drop those
    # samples from the metric computation and record n_dropped so it can
    # be inspected later.
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    n_total = int(y_true.size)
    n_dropped = int(n_total - finite.sum())
    if n_dropped == n_total:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mape_pct": float("nan"),
            "n": n_total,
            "n_dropped": n_dropped,
        }
    y_true_f = y_true[finite]
    y_pred_f = y_pred[finite]
    mae = float(mean_absolute_error(y_true_f, y_pred_f))
    rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
    if y_true_f.std() < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true_f, y_pred_f))
    pear = (
        float(pearsonr(y_true_f, y_pred_f)[0]) if y_true_f.std() > 0 else float("nan")
    )
    spear = (
        float(spearmanr(y_true_f, y_pred_f)[0]) if y_true_f.std() > 0 else float("nan")
    )
    nz = np.abs(y_true_f) > 1e-12
    mape = (
        float(np.mean(np.abs((y_true_f[nz] - y_pred_f[nz]) / y_true_f[nz])) * 100)
        if nz.any()
        else float("nan")
    )
    out = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson": pear,
        "spearman": spear,
        "mape_pct": mape,
        "n": n_total,
    }
    if n_dropped > 0:
        out["n_dropped"] = n_dropped
    return out


def _sigma_spearman(
    pred_list: List[np.ndarray], true_list: List[np.ndarray], min_grains: int = 5
) -> Dict[str, float]:
    """Per-SVE within-graph std correlation across SVEs.

    For each SVE g, compute σ_g(pred) = std of within-SVE per-grain MAX
    predictions, σ_g(true) = std of true. Then Spearman the two σ
    vectors across SVEs. Tests whether the model captures within-SVE
    FIP-dispersion variation across SVEs — the central question for the
    Gaussian-W₂² loss design (it aims to learn that σ).

    Returns {"spearman_sigma": float, "pearson_sigma": float, "n": int}.
    SVEs with fewer than `min_grains` valid grains are dropped.
    """
    sig_p, sig_t = [], []
    for p, t in zip(pred_list, true_list):
        p = np.asarray(p, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < min_grains:
            continue
        p_v, t_v = p[m], t[m]
        sig_p.append(float(np.std(p_v)))
        sig_t.append(float(np.std(t_v)))
    n_valid = len(sig_p)
    if n_valid < 3:
        return {
            "spearman_sigma": float("nan"),
            "pearson_sigma": float("nan"),
            "n": n_valid,
        }
    sp = np.asarray(sig_p, dtype=np.float64)
    st = np.asarray(sig_t, dtype=np.float64)
    if sp.std() < 1e-12 or st.std() < 1e-12:
        return {
            "spearman_sigma": float("nan"),
            "pearson_sigma": float("nan"),
            "n": n_valid,
        }
    return {
        "spearman_sigma": float(spearmanr(sp, st).statistic),
        "pearson_sigma": float(pearsonr(sp, st)[0]),
        "n": n_valid,
    }


def _evaluate(model, loader, fipds: FIPDataset, device) -> Dict[str, Dict[str, float]]:
    """Forward-pass; return dict with per-grain and per-graph raw-FIP metrics."""
    model.eval()
    pg_pred_raw: List[float] = []
    pg_true_raw: List[float] = []
    pn_pred_raw: List[float] = []
    pn_true_raw: List[float] = []
    aux_buf: Dict[str, Dict[str, List[float]]] = {
        spec["name"]: {"pred": [], "true": []}
        for spec in (fipds.aux_pergraph_targets or [])
    }
    # Per-grain aux running buffer (one per channel). Each entry mirrors the
    # per-graph list-of-arrays layout of `pn_pred_raw` so downstream code can
    # reuse the per-grain pipeline. NaN-source grains are EXCLUDED at append
    # time using `data.per_grain_aux_nan_mask` to avoid zero-fill artefacts
    # in plots/metrics.
    pn_aux_running: List[Dict[str, List[np.ndarray]]] = [
        {"pred": [], "true": []} for _ in (fipds.per_grain_aux_targets or [])
    ]
    # Phase L5 σ outputs (one entry per graph, in batch order). Each entry:
    # {"n_grains", "sigma_logspace": (n_grains, 4)}. Per-cell column order
    # is [lcf_fs, lcf_wp, hcf_fs, hcf_wp] mirroring the head's MAX-channel
    # ordering. Empty when use_heteroscedastic_max is False.
    pn_sigma_running: List[Dict[str, Any]] = []
    # Phase L6 per-graph σ outputs (one (n_graphs_in_batch, 4) array per
    # batch). Concatenated across batches at output-time. Empty when
    # use_heteroscedastic_pergraph_max is False.
    pg_sigma_running: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            # Output layout follows build_var_config_and_heads ordering.
            i = 0
            if fipds.use_per_grain_target:
                # Multi-channel per-grain head. Channels:
                #   [0]            primary FIP MAX (e.g. lcf_fs)
                #   [1..1+n_aux]   per-grain aux outputs (3 cross-variant max + 4 within-mean)
                #   [1+n_aux..]    Phase L5 σ outputs (4, only if use_heteroscedastic_max)
                # We slice the supervised section to K_super for downstream
                # per-channel handling; the σ slice is captured separately.
                K_super = 1 + len(fipds.per_grain_aux_targets or [])
                K = K_super + (
                    4 if getattr(fipds, "use_heteroscedastic_max", False) else 0
                )
                pn_full = pred[i].cpu().numpy()
                if pn_full.ndim == 1:
                    pn_full = pn_full.reshape(-1, 1)
                data_list = batch.to_data_list()
                offset = 0
                for d in data_list:
                    nn = int(d.x.shape[0])
                    bucket_id = (
                        int(d.size_bucket[0]) if hasattr(d, "size_bucket") else 0
                    )
                    pg_bucket_ids = np.full(nn, bucket_id, dtype=np.int64)
                    pn_g_full = pn_full[offset : offset + nn]  # (nn, K)
                    truth_full = d.y[: nn * K].cpu().numpy().reshape(nn, K)
                    pn_pred_raw.append(
                        fipds.inverse_pergrain_target(
                            pn_g_full[:, 0], bucket_ids=pg_bucket_ids
                        )
                    )
                    pn_true_raw.append(
                        fipds.inverse_pergrain_target(
                            truth_full[:, 0], bucket_ids=pg_bucket_ids
                        )
                    )
                    # Per-grain aux channels — exclude NaN-source grains per
                    # channel using d.per_grain_aux_nan_mask (nn, K_super-1).
                    # Only iterate the SUPERVISED aux range; σ slots (Phase
                    # L5) sit at indices [K_super..K) and are NOT aux targets.
                    if K_super > 1 and fipds.per_grain_aux_targets:
                        aux_mask = getattr(d, "per_grain_aux_nan_mask", None)
                        aux_mask_np = (
                            aux_mask.cpu().numpy() if aux_mask is not None else None
                        )
                        for k in range(K_super - 1):
                            pred_raw = fipds.inverse_per_grain_aux_target(
                                k, pn_g_full[:, 1 + k], bucket_ids=pg_bucket_ids
                            )
                            true_raw = fipds.inverse_per_grain_aux_target(
                                k, truth_full[:, 1 + k], bucket_ids=pg_bucket_ids
                            )
                            if aux_mask_np is not None:
                                valid = ~aux_mask_np[:, k]
                                pred_raw = pred_raw[valid]
                                true_raw = true_raw[valid]
                            pn_aux_running[k]["pred"].append(pred_raw)
                            pn_aux_running[k]["true"].append(true_raw)
                    # Phase L5: capture log_σ outputs (4 channels per grain)
                    # in TRANSFORMED (yeo-johnson normalized) space. We do NOT
                    # inverse-transform σ — the per-cell yeo-johnson scaling
                    # is monotonic but its inverse on σ alone is ill-posed
                    # (σ is the std of a normalized distribution). For paper
                    # purposes the σ ranking within a cell is the useful
                    # signal; absolute scale is a secondary calibration step.
                    if K > K_super:
                        sigma_block = pn_g_full[:, K_super:K]  # (nn, 4)
                        pn_sigma_running.append(
                            {
                                "graph_idx_offset": offset,
                                "n_grains": nn,
                                "sigma_logspace": sigma_block.copy(),
                            }
                        )
                    offset += nn
                i += 1
            if fipds.use_per_graph_target:
                # Multi-task per-graph head. Channel layout:
                #   [0]                   primary pg-MAX (slot 0 source)
                #   [1..1+n_aux]          aux per-graph outputs
                #   [1+n_aux..]           Phase L6 σ outputs (4, only if use_heteroscedastic_pergraph_max)
                pg_full = pred[i].cpu().numpy()
                if pg_full.ndim == 1:
                    pg_full = pg_full.reshape(-1, 1)
                K_super_pg = 1 + len(fipds.aux_pergraph_targets or [])
                K_pg = K_super_pg + (
                    4
                    if getattr(fipds, "use_heteroscedastic_pergraph_max", False)
                    else 0
                )
                pg = pg_full[:, 0]  # FIP column
                baselines = np.array(
                    [float(d.y_baseline_t.item()) for d in batch.to_data_list()]
                )
                pg_pred_raw.extend(
                    fipds.inverse_pergraph_target(pg, baseline=baselines).tolist()
                )
                pg_true_raw.extend(
                    [float(d.y_pergraph_raw.item()) for d in batch.to_data_list()]
                )
                # Aux multi-task targets (only iterate the SUPERVISED aux
                # range — σ slots at [K_super_pg..K_pg) are nuisance params).
                if fipds.aux_pergraph_targets and pg_full.shape[1] > 1:
                    bucket_per_sve = np.array(
                        [int(d.size_bucket[0]) for d in batch.to_data_list()],
                        dtype=np.int64,
                    )
                    for k, spec in enumerate(fipds.aux_pergraph_targets):
                        col_pred = pg_full[:, 1 + k]
                        pred_raw = fipds.inverse_aux_pergraph_target(
                            k,
                            col_pred,
                            bucket_ids=bucket_per_sve,
                        ).tolist()
                        true_raw = [
                            float(d.y_pergraph_aux_raw[k].item())
                            for d in batch.to_data_list()
                        ]
                        aux_buf[spec["name"]]["pred"].extend(pred_raw)
                        aux_buf[spec["name"]]["true"].extend(true_raw)
                # Phase L6: capture per-graph log_σ outputs (4 channels per
                # graph) in transformed (yeo-johnson) space. As with the
                # per-grain σ block, we do NOT inverse-transform σ — the
                # per-cell σ ranking is the useful signal. Cell order:
                # lcf_fs, lcf_wp, hcf_fs, hcf_wp (paired with the 4 SVE-MAX
                # channels at slots [0..3] — but slot 0 is from primary pg
                # head, slots 1..3 are from aux_pergraph_targets entries).
                if K_pg > K_super_pg:
                    sigma_block_pg = pg_full[
                        :, K_super_pg:K_pg
                    ]  # (n_graphs_in_batch, 4)
                    pg_sigma_running.append(sigma_block_pg.copy())
                i += 1
    out: Dict[str, Any] = {}
    if pg_pred_raw:
        out["pergraph"] = _metrics(pg_true_raw, pg_pred_raw)
        out["pergraph_pred"] = pg_pred_raw
        out["pergraph_true"] = pg_true_raw
    # Per-aux-target metrics + buffered raw predictions.
    if aux_buf:
        out["aux_pergraph"] = {}
        for name, bufd in aux_buf.items():
            if not bufd["pred"]:
                continue
            out["aux_pergraph"][name] = {
                "metrics": _metrics(bufd["true"], bufd["pred"]),
                "pred": bufd["pred"],
                "true": bufd["true"],
            }
    # Per-grain aux metrics (NaN-source grains already excluded at append).
    if pn_aux_running and any(b["pred"] for b in pn_aux_running):
        out["pergrain_aux"] = {}
        for k, spec in enumerate(fipds.per_grain_aux_targets):
            pred_list = pn_aux_running[k]["pred"]
            true_list = pn_aux_running[k]["true"]
            if not pred_list:
                continue
            flat_p = np.concatenate(pred_list)
            flat_t = np.concatenate(true_list)
            out["pergrain_aux"][spec["name"]] = {
                "metrics": _metrics(flat_t, flat_p),
                "pred": pred_list,
                "true": true_list,
            }
    # Phase L5 heteroscedastic σ outputs: per-grain log σ for the 4 MAX
    # channels. Store the per-graph list-of-arrays layout so downstream
    # CSV writers can align with pn_pred_raw. Cell order: lcf_fs, lcf_wp,
    # hcf_fs, hcf_wp (matches head MAX-channel ordering).
    if pn_sigma_running:
        per_graph_sigma = [entry["sigma_logspace"] for entry in pn_sigma_running]
        out["pergrain_logsigma"] = per_graph_sigma
        # Aggregate stats per cell (transformed-space log σ).
        flat = (
            np.concatenate(per_graph_sigma, axis=0)
            if per_graph_sigma
            else np.zeros((0, 4))
        )
        out["pergrain_logsigma_stats"] = {
            cell: {
                "mean": float(flat[:, k].mean()) if flat.size else float("nan"),
                "std": float(flat[:, k].std()) if flat.size else float("nan"),
                "p10": float(np.quantile(flat[:, k], 0.10))
                if flat.size
                else float("nan"),
                "p90": float(np.quantile(flat[:, k], 0.90))
                if flat.size
                else float("nan"),
            }
            for k, cell in enumerate(("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"))
        }
    # Phase L6 per-graph σ outputs (4 channels per SVE) — concat across
    # batches into one (n_graphs_in_split, 4) array. Cell order:
    # lcf_fs, lcf_wp, hcf_fs, hcf_wp (paired with the 4 SVE-MAX channels
    # at slots 0..3 of the per-graph head per the deslot ordering).
    if pg_sigma_running:
        flat_pg = np.concatenate(pg_sigma_running, axis=0)
        out["pergraph_logsigma"] = flat_pg
        out["pergraph_logsigma_stats"] = {
            cell: {
                "mean": float(flat_pg[:, k].mean()) if flat_pg.size else float("nan"),
                "std": float(flat_pg[:, k].std()) if flat_pg.size else float("nan"),
                "p10": float(np.quantile(flat_pg[:, k], 0.10))
                if flat_pg.size
                else float("nan"),
                "p90": float(np.quantile(flat_pg[:, k], 0.90))
                if flat_pg.size
                else float("nan"),
            }
            for k, cell in enumerate(("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"))
        }
    if pn_pred_raw:
        flat_p = np.concatenate(pn_pred_raw)
        flat_t = np.concatenate(pn_true_raw)
        out["pergrain"] = _metrics(flat_t, flat_p)
        out["pergrain_pred"] = pn_pred_raw
        out["pergrain_true"] = pn_true_raw
        # Order-statistic per-graph: max(per-grain pred) per graph.
        os_pred = np.array([float(p.max()) for p in pn_pred_raw])
        os_true = np.array([float(t.max()) for t in pn_true_raw])
        out["pergraph_orderstat"] = _metrics(os_true, os_pred)
        out["pergraph_orderstat_pred"] = os_pred.tolist()
        out["pergraph_orderstat_true"] = os_true.tolist()
        # σ-Spearman: per-SVE within-graph std correlation, for the 4
        # within-SVE per-grain MAX channels (slot 0 + first 3 aux entries
        # under v9 naming convention `*_max`). Tests Gaussian-W₂² target.
        sigma_block: Dict[str, Dict[str, float]] = {
            "lcf_fs_max": _sigma_spearman(pn_pred_raw, pn_true_raw),
        }
        if pn_aux_running and fipds.per_grain_aux_targets:
            for k, spec in enumerate(fipds.per_grain_aux_targets):
                if not spec["name"].endswith("_max"):
                    continue
                pred_list = pn_aux_running[k]["pred"]
                true_list = pn_aux_running[k]["true"]
                if not pred_list:
                    continue
                sigma_block[spec["name"]] = _sigma_spearman(pred_list, true_list)
        out["pergrain_sigma"] = sigma_block
        # Hot-spot identification metrics — computed PER CELL across all 4
        # SVE-MAX channels (LCF/FS, LCF/WP, HCF/FS, HCF/WP). v9b deslot:
        # the slot-0-only path is replaced by a uniform per-cell loop. Each
        # cell's "top hot grain" is its OWN argmax, which generally differs
        # from the other cells' (LCF and HCF have Spearman 0.07–0.97 across
        # cell pairs; D2 cross-corr).
        from scipy.stats import spearmanr as _spr

        def _cell_hotspot_metrics(pred_per_graph, true_per_graph):
            """Compute (within-SVE Spearman, top-1 hit rate, top-10% recall,
            hotspot loc err) PER GRAPH then averaged. Returns dict of scalars."""
            spears = []
            top1_hits = 0
            top10_recall = []
            hotspot_loc_err = []
            n_graphs_used = 0
            for pred, true in zip(pred_per_graph, true_per_graph):
                pred = np.asarray(pred)
                true = np.asarray(true)
                # Drop NaN-source grains (HCF cells on missing-HCF SVEs).
                m = np.isfinite(pred) & np.isfinite(true)
                pred, true = pred[m], true[m]
                n = len(pred)
                if n < 2:
                    continue
                n_graphs_used += 1
                try:
                    s = _spr(true, pred).statistic
                    if s == s:
                        spears.append(float(s))
                except Exception:
                    pass
                if int(np.argmax(pred)) == int(np.argmax(true)):
                    top1_hits += 1
                k = max(1, int(np.ceil(0.10 * n)))
                top_true = set(np.argsort(true)[-k:].tolist())
                top_pred = set(np.argsort(pred)[-k:].tolist())
                top10_recall.append(len(top_true & top_pred) / k)
                true_max_idx = int(np.argmax(true))
                pred_rank_of_true_max = int(np.sum(pred >= pred[true_max_idx]))
                hotspot_loc_err.append((pred_rank_of_true_max - 1) / max(n - 1, 1))
            return {
                "n_graphs_used": n_graphs_used,
                "spearman_per_graph_mean": float(np.mean(spears))
                if spears
                else float("nan"),
                "top1_hit_rate": (top1_hits / n_graphs_used)
                if n_graphs_used
                else float("nan"),
                "top10_recall": float(np.mean(top10_recall))
                if top10_recall
                else float("nan"),
                "hotspot_loc_err": float(np.mean(hotspot_loc_err))
                if hotspot_loc_err
                else float("nan"),
            }

        # 4-cell symmetric per-cell hotspot metrics. Order (lcf_fs, lcf_wp,
        # hcf_fs, hcf_wp) matches the head's MAX-channel slots.
        per_cell_hotspot: Dict[str, Dict[str, float]] = {}
        # Slot 0 = lcf_fs from `pn_pred_raw`/`pn_true_raw`.
        per_cell_hotspot["lcf_fs_max"] = _cell_hotspot_metrics(pn_pred_raw, pn_true_raw)
        # Slots 1+ = aux *_max channels. Read from pergrain_aux entries.
        if "pergrain_aux" in out:
            for aux_name in ("lcf_wp_max", "hcf_fs_max", "hcf_wp_max"):
                ent = out["pergrain_aux"].get(aux_name)
                if ent is None:
                    continue
                per_cell_hotspot[aux_name] = _cell_hotspot_metrics(
                    ent["pred"], ent["true"]
                )
        out["pergrain_per_cell_hotspot"] = per_cell_hotspot

        # 4-cell symmetric means (paper-grade headline metrics).
        def _mean_over_cells(key: str) -> float:
            vals = [
                per_cell_hotspot[c][key]
                for c in per_cell_hotspot
                if per_cell_hotspot[c][key] == per_cell_hotspot[c][key]
            ]
            return float(np.mean(vals)) if vals else float("nan")

        out["pergrain_top1_hit_rate_mean_4cell"] = _mean_over_cells("top1_hit_rate")
        out["pergrain_top10_recall_mean_4cell"] = _mean_over_cells("top10_recall")
        out["pergrain_spearman_per_graph_mean_4cell"] = _mean_over_cells(
            "spearman_per_graph_mean"
        )
        out["pergrain_hotspot_loc_err_mean_4cell"] = _mean_over_cells("hotspot_loc_err")

        # Legacy slot-0-only keys (kept for backward-compat with run_track_*
        # scripts and HPO objective parsing that still reads slot-0 metrics).
        s0 = per_cell_hotspot.get("lcf_fs_max", {})
        if s0:
            out["pergrain_spearman_per_graph_mean"] = s0["spearman_per_graph_mean"]
            out["pergrain_top1_hit_rate"] = s0["top1_hit_rate"]
            out["pergrain_top10_recall"] = s0["top10_recall"]
            out["pergrain_hotspot_loc_err"] = s0["hotspot_loc_err"]
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _prune_epoch_checkpoints(log_dir: Path) -> None:
    """Replace {log_dir}.pk symlink with the actual best-val file and delete
    the rest of the per-epoch checkpoints.

    HydraGNN's `save_model` writes `{log_dir}_epoch_N.pk` for every improving
    epoch and only retargets the `{log_dir}.pk` symlink — old files persist.
    After training, this collapses dozens of per-epoch files down to a single
    `{log_dir}.pk` so disk stays tidy and downstream loads still work.
    """
    parent = log_dir.parent
    base = log_dir.name
    link = parent / f"{base}.pk"
    if not link.exists() and not link.is_symlink():
        return
    # Resolve the symlink to find the winning checkpoint file.
    target = link.resolve() if link.is_symlink() else link
    # Replace the symlink with the actual file (rename is atomic on POSIX).
    if link.is_symlink():
        link.unlink()
        if target.exists():
            target.rename(link)
    # Delete every remaining per-epoch checkpoint.
    n_removed = 0
    for f in parent.glob(f"{base}_epoch_*.pk"):
        try:
            f.unlink()
            n_removed += 1
        except FileNotFoundError:
            pass
    print(f"[train_one] pruned {n_removed} per-epoch checkpoints; kept {link.name}")


def train_one(
    seed: int = 42,
    exp_tag: str = "floor",
    config_overrides: Optional[dict] = None,
    num_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one training. Returns metrics dict; also writes results/{tag}_seed{seed}/."""

    cfg = default_config()
    if config_overrides:
        cfg = deep_merge(cfg, config_overrides)
    if num_epoch is not None:
        cfg["NeuralNetwork"]["Training"]["num_epoch"] = num_epoch

    # Determinism (data + weight init).
    torch.manual_seed(seed)
    np.random.seed(seed)

    # HydraGNN expects a torch.distributed group to exist (it's used for
    # cross-rank checks even on single-rank runs). Idempotent setup.
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Use a per-process port so concurrent train_one invocations don't collide.
        os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 500)))
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Pass explicit device_id so PyTorch ≥2.4 doesn't emit the
        # "No device id provided" warning. Each slot sees one GPU under
        # CUDA_VISIBLE_DEVICES, so cuda:0 is unambiguous.
        device_id = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else None
        )
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", device_id=device_id
        )
        # Tear down the process group at process exit so PyTorch ≥2.4
        # doesn't emit the "destroy_process_group() was not called" leak
        # warning. Using atexit (rather than per-call destroy) keeps the
        # group alive across repeated train_one() calls in one process.
        import atexit as _atexit

        def _cleanup_dist():
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass

        _atexit.register(_cleanup_dist)

    # Build dataset.
    ds_cfg = cfg["Recipe"]
    # Decouple split seed from model init seed when caller wants a fixed split
    # across multiple init seeds (for honest seed-ensembling).
    split_seed = ds_cfg.get("split_seed")
    if split_seed is None:
        split_seed = seed
    _fipds_kwargs = dict(
        size_bucket=ds_cfg["size_bucket"],
        seed=split_seed,
        split_fracs=tuple(ds_cfg.get("split_fracs", (0.80, 0.10, 0.10))),
        stratify_by_bucket=bool(ds_cfg.get("stratify_by_bucket", True)),
        train_buckets=ds_cfg.get("train_buckets"),
        test_buckets=ds_cfg.get("test_buckets"),
        graph_attr_source=ds_cfg.get("graph_attr_source"),
        target_residual_baseline=ds_cfg.get("target_residual_baseline"),
        per_graph_target_aggr=ds_cfg.get("per_graph_target_aggr", "max"),
        per_graph_target_source=ds_cfg.get("per_graph_target_source", "default"),
        aux_pergraph_targets=ds_cfg.get("aux_pergraph_targets") or [],
        per_grain_aux_targets=ds_cfg.get("per_grain_aux_targets") or [],
        # Defensive: handle the case where BASE_RECIPE or HPO extras set
        # these keys to None explicitly (deslot path); fall back to defaults.
        per_grain_target_col=int(ds_cfg.get("per_grain_target_col") or 0),
        per_grain_target_field=str(ds_cfg.get("per_grain_target_field") or "fs_fip"),
        use_x_blocks=ds_cfg["use_x_blocks"],
        use_edge_cols=ds_cfg["use_edge_cols"],
        drop_periodic_edges=ds_cfg["drop_periodic_edges"],
        use_per_grain_target=ds_cfg["use_per_grain_target"],
        use_per_graph_target=ds_cfg["use_per_graph_target"],
        per_grain_transform=ds_cfg["per_grain_transform"],
        per_graph_transform=ds_cfg["per_graph_transform"],
        use_heteroscedastic_max=bool(ds_cfg.get("use_heteroscedastic_max", False)),
        use_heteroscedastic_pergraph_max=bool(
            ds_cfg.get("use_heteroscedastic_pergraph_max", False)
        ),
        # v9b deslot: prefer the unified symmetric target lists when provided.
        # When non-empty, FIPDataset.__post_init__ derives the legacy slot-0 +
        # aux split internally so the recipe-facing API stays symmetric.
        per_grain_targets=list(ds_cfg.get("per_grain_targets") or []),
        pergraph_targets=list(ds_cfg.get("pergraph_targets") or []),
    )
    # Optional override: point at a different .pt (e.g. full_data_v3.pt).
    if ds_cfg.get("pt_path") is not None:
        _fipds_kwargs["pt_path"] = ds_cfg["pt_path"]
    fipds = FIPDataset(**_fipds_kwargs)
    train_data, val_data, test_data = fipds.build()

    # If graph_attr is active, tell HydraGNN.
    if ds_cfg.get("graph_attr_source") is not None:
        cfg["NeuralNetwork"]["Architecture"]["use_graph_attr_conditioning"] = True
        cfg["NeuralNetwork"]["Architecture"][
            "graph_attr_conditioning_mode"
        ] = ds_cfg.get("graph_attr_conditioning_mode", "concat_node")

    # Build var_config + output_heads now that we know feature sizes.
    cfg = build_var_config_and_heads(cfg)

    # Build loaders. The sbatch launcher can give each HPO slot a per-process
    # CPU budget via HYDRAGNN_NUM_WORKERS so batch collation does not bottleneck
    # the GPU worker.
    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    num_workers = max(0, int(os.environ.get("HYDRAGNN_NUM_WORKERS", "0")))
    loader_kwargs = {
        "batch_size": bs,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    print(
        "[train_one] DataLoader config: "
        f"num_workers={num_workers} "
        f"pin_memory={loader_kwargs['pin_memory']} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}"
    )
    if ds_cfg.get("use_fcc_sym_augment", False):
        train_aug = FCCSymAugmentDataset(train_data, p=1.0, seed=seed)
        train_loader = DataLoader(train_aug, shuffle=True, **loader_kwargs)
        print(f"[train_one] FCC cubic-sym augmentation ENABLED (p=1.0)")
    elif ds_cfg.get("use_so3_augment", False):
        train_aug = SO3AugmentDataset(train_data, p=1.0, seed=seed)
        train_loader = DataLoader(train_aug, shuffle=True, **loader_kwargs)
        print(f"[train_one] SO(3) sample-frame augmentation ENABLED (p=1.0)")
    else:
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    # Build model.
    log_name = f"v4_{exp_tag}_seed{seed}"
    log_dir = _STUDY_ROOT / "logs" / log_name
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir = _STUDY_ROOT / "results" / f"{exp_tag}_seed{seed}"
    res_dir.mkdir(parents=True, exist_ok=True)

    # update_config fills defaults like input_dim / pna_deg from the dataloader.
    cfg = update_config(cfg, train_loader, val_loader, test_loader)

    model = create_model_config(
        config=cfg["NeuralNetwork"], verbosity=cfg["Verbosity"]["level"]
    )
    model = hydragnn.utils.distributed.get_distributed_model(
        model, cfg["Verbosity"]["level"]
    )

    # Custom loss support: if recipe specified, install batch-stash hook (so
    # rank/topk losses can read data.batch) and override model.loss_function.
    # v3 BUG: previously also excluded "huber" (a HydraGNN-native loss).
    # That caused T1_huber cells to silently train MSE. v4 routes huber via
    # `loss_function_type` in NeuralNetwork.Training instead — only "default"
    # short-circuits the override here.
    loss_recipe = ds_cfg.get("loss_recipe")
    if loss_recipe and loss_recipe != "default":
        install_batch_stash_hook(model)
        override_model_loss(model, loss_recipe)

    # Phase L4 residual head: wrap every MLPNode with a learnable linear
    # skip (encoder hidden → output_dim, bias=False, init=zeros). Must run
    # BEFORE the optimiser is constructed so the skip weights are tracked.
    if bool(ds_cfg.get("use_residual_node_head", False)):
        n_wrapped = _install_residual_node_head(model)
        print(
            f"[residual_head] wrapped {n_wrapped} per-node MLP head(s) "
            f"with learnable linear skip"
        )

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    writer = LossHistoryWriter(get_summary_writer(str(log_dir)))

    # Save config used for this run.
    with open(res_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    fipds.save_split_indices(str(res_dir / "split_indices.json"))

    # ---- Train ----
    t0 = time.time()
    train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        cfg["NeuralNetwork"],
        str(log_dir),
        cfg["Verbosity"]["level"],
        plot_init_solution=False,
        plot_hist_solution=False,
        create_plots=False,
    )
    elapsed = time.time() - t0
    print(f"[train_one] {exp_tag} seed={seed} elapsed={elapsed:.1f}s")

    # ---- Load best-val checkpoint ----
    try:
        load_existing_model(model, str(log_dir))
    except Exception as e:
        print(f"[train_one] WARNING: load_existing_model failed: {e}")

    # ---- Evaluate ----
    device = next(model.parameters()).device
    train_metrics = _evaluate(model, train_loader, fipds, device)
    val_metrics = _evaluate(model, val_loader, fipds, device)
    test_metrics = _evaluate(model, test_loader, fipds, device)

    # ---- Prune per-epoch checkpoints, keep only the best-val one ----
    # HydraGNN's save_model writes a new {log_dir}_epoch_N.pk every time val
    # improves and only retargets the {log_dir}.pk symlink — old per-epoch
    # files are never cleaned up. After eval completes, replace the symlink
    # with the actual best-val file and delete the rest.
    try:
        _prune_epoch_checkpoints(log_dir)
    except Exception as e:
        print(f"[train_one] WARNING: checkpoint prune failed: {e}")

    # ---- Save results ----
    def _strip_pergrain_aux(d: dict) -> dict:
        """Drop pred/true lists from pergrain_aux for JSON output (per-grain
        arrays would balloon run_metrics.json). Keep metrics summary only —
        full predictions are in per-channel CSVs."""
        if "pergrain_aux" not in d:
            return d
        slim = dict(d)
        slim["pergrain_aux"] = {
            name: {"metrics": ent["metrics"]} for name, ent in d["pergrain_aux"].items()
        }
        return slim

    keep = (
        "pergraph",
        "pergrain",
        "pergraph_orderstat",
        "aux_pergraph",
        "pergrain_aux",
        "pergrain_sigma",
        "pergrain_spearman_per_graph_mean",
        "pergrain_top1_hit_rate",
        "pergrain_top10_recall",
        "pergrain_hotspot_loc_err",
        "pergrain_logsigma_stats",  # Phase L5 het-NLL σ aggregate (per-cell mean/std/p10/p90)
        "pergraph_logsigma_stats",  # Phase L6 het-NLL σ aggregate (per-graph)
        # v9b deslot: 4-cell symmetric hotspot metrics (per-cell + 4-cell means)
        "pergrain_per_cell_hotspot",
        "pergrain_top1_hit_rate_mean_4cell",
        "pergrain_top10_recall_mean_4cell",
        "pergrain_spearman_per_graph_mean_4cell",
        "pergrain_hotspot_loc_err_mean_4cell",
    )
    metrics = {
        "method": "GNN",
        "seed": seed,
        "exp_tag": exp_tag,
        "log_name": log_name,
        "elapsed_sec": elapsed,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "train": _strip_pergrain_aux(
            {k: v for k, v in train_metrics.items() if k in keep}
        ),
        "val": _strip_pergrain_aux({k: v for k, v in val_metrics.items() if k in keep}),
        "test": _strip_pergrain_aux(
            {k: v for k, v in test_metrics.items() if k in keep}
        ),
    }
    with open(res_dir / "run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # ===== v9 result-naming refactor (2026-05-06) ==========================
    # Replaces the old per-(split × output × plot-type) file explosion with
    # wide CSVs (one per split×level) + multi-panel grid plots (one per
    # stat-family). ~12× file reduction, symmetric 4-cell naming, no "_aux".
    # See `feedback_v9_result_naming_2026-05-06.md` for the contract.
    split_dataset_indices = {
        "train": fipds.splits.train,
        "val": fipds.splits.val,
        "test": fipds.splits.test,
    }
    splits_iter = [
        ("train", train_data, train_metrics),
        ("val", val_data, val_metrics),
        ("test", test_data, test_metrics),
    ]
    # Cache split metadata for re-use across CSV + plot writers.
    split_meta: Dict[str, Dict] = {}
    for split_name, split_data, split_eval in splits_iter:
        split_meta[split_name] = {
            "split_data": split_data,
            "split_eval": split_eval,
            "sb_arr": np.array([int(d.size_bucket[0]) for d in split_data]),
            "n_arr": np.array([int(d.x.shape[0]) for d in split_data]),
            "ds_idx": split_dataset_indices[split_name],
        }

    # ---- Wide pn CSVs (one per split): 8 outputs × 2 (true,pred) columns ----
    for split_name in _V9_SPLITS:
        meta = split_meta[split_name]
        ev = meta["split_eval"]
        if "pergrain_pred" not in ev:
            continue
        pn_pred_s0 = ev["pergrain_pred"]  # slot 0 = lcf_fs_max
        pn_true_s0 = ev["pergrain_true"]
        aux = ev.get("pergrain_aux", {}) or {}
        # Flatten per-graph lists into per-grain rows ordered by (graph_idx,grain_idx).
        sb_arr = meta["sb_arr"]
        ds_idx = meta["ds_idx"]
        # Collect per-cell tensors (true, pred) at the SAME (graph_idx, grain_idx).
        # Slot 0 (lcf_fs_max) is the index basis; aux entries are aligned 1:1.
        pn_max_data: Dict[str, tuple] = {"lcf_fs": (pn_true_s0, pn_pred_s0)}
        pn_mean_data: Dict[str, tuple] = {}
        # Distribution-shape extras (Phase M): logmu, logsigma, p90, p99.
        pn_distshape_data: Dict[str, Dict[str, tuple]] = {
            stat: {} for stat in _V9_DISTSHAPE_STATS
        }
        for aname, ent in aux.items():
            cs = _V9_PN_AUX_TO_CELL_STAT.get(aname)
            if cs is None:
                continue
            cell, stat = cs
            if stat == "max":
                pn_max_data[cell] = (ent["true"], ent["pred"])
            elif stat == "mean":
                pn_mean_data[cell] = (ent["true"], ent["pred"])
            elif stat in _V9_DISTSHAPE_STATS:
                pn_distshape_data[stat][cell] = (ent["true"], ent["pred"])
        # Phase L5 σ outputs (one per cell, transformed-space log σ; only
        # present when use_heteroscedastic_max=True). Aligned to pn_pred_s0
        # graph order; same shape (per-graph list of (n_grains, 4) arrays).
        pn_logsigma_per_graph = ev.get("pergrain_logsigma")
        cols = ["dataset_idx", "graph_idx", "grain_idx", "size_bucket"]
        for cell in _V9_CELL_KEYS:
            cols += [f"true_{cell}_max", f"pred_{cell}_max"]
        for cell in _V9_CELL_KEYS:
            cols += [f"true_{cell}_mean", f"pred_{cell}_mean"]
        if pn_logsigma_per_graph is not None:
            for cell in _V9_CELL_KEYS:
                cols.append(f"logsigma_{cell}_max")
        # Phase M distribution-shape extras (only emit columns for stats
        # actually present in this run).
        active_distshape_stats = [
            stat for stat in _V9_DISTSHAPE_STATS if pn_distshape_data[stat]
        ]
        for stat in active_distshape_stats:
            for cell in _V9_CELL_KEYS:
                cols += [f"true_{cell}_{stat}", f"pred_{cell}_{stat}"]
        with open(res_dir / f"pred_{split_name}_pn.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for gi in range(len(pn_pred_s0)):
                sb = int(sb_arr[gi])
                di = int(ds_idx[gi])
                ng = len(pn_pred_s0[gi])
                for ki in range(ng):
                    row = [str(di), str(gi), str(ki), str(sb)]
                    for cell in _V9_CELL_KEYS:
                        if cell in pn_max_data:
                            tlist, plist = pn_max_data[cell]
                            if gi < len(tlist) and ki < len(tlist[gi]):
                                row += [
                                    f"{float(tlist[gi][ki]):.6e}",
                                    f"{float(plist[gi][ki]):.6e}",
                                ]
                            else:
                                row += ["", ""]
                        else:
                            row += ["", ""]
                    for cell in _V9_CELL_KEYS:
                        if cell in pn_mean_data:
                            tlist, plist = pn_mean_data[cell]
                            if gi < len(tlist) and ki < len(tlist[gi]):
                                row += [
                                    f"{float(tlist[gi][ki]):.6e}",
                                    f"{float(plist[gi][ki]):.6e}",
                                ]
                            else:
                                row += ["", ""]
                        else:
                            row += ["", ""]
                    if pn_logsigma_per_graph is not None and gi < len(
                        pn_logsigma_per_graph
                    ):
                        sig = pn_logsigma_per_graph[gi]
                        if ki < sig.shape[0]:
                            for k in range(4):
                                row.append(f"{float(sig[ki, k]):.6e}")
                        else:
                            row.extend([""] * 4)
                    # Phase M distribution-shape columns (one per stat × cell).
                    for stat in active_distshape_stats:
                        for cell in _V9_CELL_KEYS:
                            d = pn_distshape_data[stat]
                            if cell in d:
                                tlist, plist = d[cell]
                                if gi < len(tlist) and ki < len(tlist[gi]):
                                    row += [
                                        f"{float(tlist[gi][ki]):.6e}",
                                        f"{float(plist[gi][ki]):.6e}",
                                    ]
                                else:
                                    row += ["", ""]
                            else:
                                row += ["", ""]
                    f.write(",".join(row) + "\n")

    # ---- Wide pg CSVs (one per split): 13 outputs × 2 columns ----
    for split_name in _V9_SPLITS:
        meta = split_meta[split_name]
        ev = meta["split_eval"]
        if "pergraph_pred" not in ev:
            continue
        pg_pred_s0 = np.asarray(ev["pergraph_pred"])
        pg_true_s0 = np.asarray(ev["pergraph_true"])
        aux = ev.get("aux_pergraph", {}) or {}
        sb_arr = meta["sb_arr"]
        ds_idx = meta["ds_idx"]
        n_arr = meta["n_arr"]
        pg_max_data: Dict[str, tuple] = {"lcf_fs": (pg_true_s0, pg_pred_s0)}
        pg_ve_mu_data: Dict[str, tuple] = {}
        pg_mech_data: Dict[str, tuple] = {}
        for aname, ent in aux.items():
            cs = _V9_PG_AUX_TO_CELL_STAT.get(aname)
            if cs is not None:
                cell, stat = cs
                if stat == "max":
                    pg_max_data[cell] = (
                        np.asarray(ent["true"]),
                        np.asarray(ent["pred"]),
                    )
                elif stat == "ve_mu":
                    pg_ve_mu_data[cell] = (
                        np.asarray(ent["true"]),
                        np.asarray(ent["pred"]),
                    )
            elif aname in _V9_MECH_QOI_NAMES:
                pg_mech_data[aname] = (np.asarray(ent["true"]), np.asarray(ent["pred"]))
        # Phase L6 σ outputs at the per-graph level (4 cells, 1 row per graph,
        # transformed-space log_σ). Same shape as pg_pred_s0 in graph axis.
        pg_logsigma = ev.get("pergraph_logsigma")
        cols = ["dataset_idx", "graph_idx", "n_nodes", "size_bucket"]
        for cell in _V9_CELL_KEYS:
            cols += [f"true_{cell}_max", f"pred_{cell}_max"]
        for cell in _V9_CELL_KEYS:
            cols += [f"true_{cell}_ve_mu", f"pred_{cell}_ve_mu"]
        for q in _V9_MECH_QOI_NAMES:
            cols += [f"true_{q}", f"pred_{q}"]
        if pg_logsigma is not None:
            for cell in _V9_CELL_KEYS:
                cols.append(f"logsigma_{cell}_max_pergraph")
        with open(res_dir / f"pred_{split_name}_pg.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for k in range(len(pg_pred_s0)):
                sb = int(sb_arr[k])
                di = int(ds_idx[k])
                nn = int(n_arr[k])
                row = [str(di), str(k), str(nn), str(sb)]
                for cell in _V9_CELL_KEYS:
                    if cell in pg_max_data:
                        t, p = pg_max_data[cell]
                        row += [f"{float(t[k]):.6e}", f"{float(p[k]):.6e}"]
                    else:
                        row += ["", ""]
                for cell in _V9_CELL_KEYS:
                    if cell in pg_ve_mu_data:
                        t, p = pg_ve_mu_data[cell]
                        row += [f"{float(t[k]):.6e}", f"{float(p[k]):.6e}"]
                    else:
                        row += ["", ""]
                for q in _V9_MECH_QOI_NAMES:
                    if q in pg_mech_data:
                        t, p = pg_mech_data[q]
                        row += [f"{float(t[k]):.6e}", f"{float(p[k]):.6e}"]
                    else:
                        row += ["", ""]
                if pg_logsigma is not None and k < pg_logsigma.shape[0]:
                    for kk in range(4):
                        row.append(f"{float(pg_logsigma[k, kk]):.6e}")
                f.write(",".join(row) + "\n")

    # ---- Grid plot helpers: pull (yt, yp, sb) per (level, stat, cell, split) ----
    def _pn_panel_data(split_name: str, cell: str, stat: str):
        """Return (yt_flat, yp_flat, sb_per_grain) for a per-grain cell×stat
        on a given split. stat ∈ {"max","mean"}. Returns Nones if absent."""
        meta = split_meta[split_name]
        ev = meta["split_eval"]
        sb_arr = meta["sb_arr"]
        if stat == "max" and cell == "lcf_fs":
            tlist, plist = ev.get("pergrain_true"), ev.get("pergrain_pred")
        else:
            aux_name = f"{cell}_max" if stat == "max" else f"{cell}_within_mean"
            entry = (ev.get("pergrain_aux", {}) or {}).get(aux_name)
            if entry is None:
                return None, None, None
            tlist, plist = entry["true"], entry["pred"]
        if not tlist:
            return None, None, None
        yt = np.concatenate([np.asarray(t).ravel() for t in tlist])
        yp = np.concatenate([np.asarray(p).ravel() for p in plist])
        sb = np.concatenate(
            [
                np.full(len(tlist[gi]), int(sb_arr[gi]), dtype=int)
                for gi in range(len(tlist))
            ]
        )
        return yt, yp, sb.tolist()

    def _pg_panel_data(split_name: str, cell: str, stat: str):
        """Per-graph cell×stat. stat ∈ {"max","ve_mu"}."""
        meta = split_meta[split_name]
        ev = meta["split_eval"]
        sb_arr = meta["sb_arr"]
        if stat == "max" and cell == "lcf_fs":
            t, p = ev.get("pergraph_true"), ev.get("pergraph_pred")
            if t is None or p is None:
                return None, None, None
            return np.asarray(t), np.asarray(p), sb_arr.tolist()
        aux_name = f"{cell}_pergraph_max" if stat == "max" else f"{cell}_ve_mu_pergraph"
        entry = (ev.get("aux_pergraph", {}) or {}).get(aux_name)
        if entry is None:
            return None, None, None
        return np.asarray(entry["true"]), np.asarray(entry["pred"]), sb_arr.tolist()

    def _mech_panel_data(split_name: str, qoi: str):
        meta = split_meta[split_name]
        ev = meta["split_eval"]
        sb_arr = meta["sb_arr"]
        entry = (ev.get("aux_pergraph", {}) or {}).get(qoi)
        if entry is None:
            return None, None, None
        return np.asarray(entry["true"]), np.asarray(entry["pred"]), sb_arr.tolist()

    def _build_panels_4cells(panel_fn, stat: str):
        """Build a list of 12 panels (4 cells × 3 splits), row-major: cell-as-row."""
        panels = []
        for cell in _V9_CELL_KEYS:
            for split_name in _V9_SPLITS:
                yt, yp, sb = panel_fn(split_name, cell, stat)
                title = f"{cell} / {split_name}"
                panels.append(
                    {
                        "title": title,
                        "y_true": yt if yt is not None else np.array([]),
                        "y_pred": yp if yp is not None else np.array([]),
                        "size_buckets": sb if sb is not None else None,
                    }
                )
        return panels

    # 2 × 3 mech-QoI parity grid (test split only). Physics-grouped layout:
    #   row 1 (stress)  = [yield_stress, sigma_amp_lcf, sigma_amp_hcf]
    #   row 2 (other)   = [hardening_modulus, Wp_lcf, Wp_hcf]
    # Columns = regime (monotonic / LCF / HCF). Compact, paper-clean.
    _MECH_PARITY_LAYOUT = [
        # row 1: stress-like quantities
        ("yield_stress", "monotonic"),
        ("sigma_amp_lcf", "LCF"),
        ("sigma_amp_hcf", "HCF"),
        # row 2: energy / hardening
        ("hardening_modulus", "monotonic"),
        ("Wp_lcf", "LCF"),
        ("Wp_hcf", "HCF"),
    ]
    _MECH_PARITY_NROWS = 2
    _MECH_PARITY_NCOLS = 3
    _MECH_PARITY_ROWLABELS = ["stress", "energy / hardening"]
    _MECH_PARITY_COLLABELS = ["monotonic", "LCF", "HCF"]

    def _build_panels_mech(split_name: str = "test"):
        """Build the 6-panel 2×3 mech grid for a single split (default test)."""
        panels = []
        for q, regime_label in _MECH_PARITY_LAYOUT:
            yt, yp, sb = _mech_panel_data(split_name, q)
            panels.append(
                {
                    "title": f"{q}",
                    "y_true": yt if yt is not None else np.array([]),
                    "y_pred": yp if yp is not None else np.array([]),
                    "size_buckets": sb if sb is not None else None,
                }
            )
        return panels

    # ---- Parity grids (5 PNGs total instead of 21 outputs × 3 splits = 63) ----
    pn_max_panels = _build_panels_4cells(_pn_panel_data, "max")
    pn_mean_panels = _build_panels_4cells(_pn_panel_data, "mean")
    pg_max_panels = _build_panels_4cells(_pg_panel_data, "max")
    pg_ve_mu_panels = _build_panels_4cells(_pg_panel_data, "ve_mu")
    mech_panels = _build_panels_mech()
    base_title = f"{exp_tag} seed={seed}"
    parity_grid_plot(
        res_dir / "parity_pn_max.png",
        pn_max_panels,
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-grain MAX parity",
        bucket_to_label=BUCKET_TO_LSVE_LABEL,
    )
    parity_grid_plot(
        res_dir / "parity_pn_within_mean.png",
        pn_mean_panels,
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-grain within-mean parity",
        bucket_to_label=BUCKET_TO_LSVE_LABEL,
    )
    parity_grid_plot(
        res_dir / "parity_pg_max.png",
        pg_max_panels,
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-graph MAX parity",
        bucket_to_label=BUCKET_TO_LSVE_LABEL,
    )
    parity_grid_plot(
        res_dir / "parity_pg_ve_mu.png",
        pg_ve_mu_panels,
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-graph within-SVE μ parity",
        bucket_to_label=BUCKET_TO_LSVE_LABEL,
    )
    # parity_mech.png — 2 × 3 layout, test split only. Rows = stress vs
    # energy/hardening; columns = regime (monotonic / LCF / HCF).
    parity_grid_plot(
        res_dir / "parity_mech.png",
        mech_panels,
        nrows=_MECH_PARITY_NROWS,
        ncols=_MECH_PARITY_NCOLS,
        row_labels=_MECH_PARITY_ROWLABELS,
        col_labels=_MECH_PARITY_COLLABELS,
        suptitle=f"{base_title} — mech QoI parity (test split, 2×3 physics-grouped)",
        bucket_to_label=BUCKET_TO_LSVE_LABEL,
    )

    # ---- Hotspot diagnostic: ONE plot per cell (4 cells × 3 splits × 4 panels) ----
    # v9b deslot: hotspot diagnostic was previously slot-0 LCF/FS only. Each
    # of the 4 (criterion × regime) cells has DIFFERENT top-hot grains within
    # an SVE (cross-cell within-SVE Spearman varies 0.07–0.97 per D2). To be
    # honest about "where is the hot grain in cell X", we need a separate
    # diagnostic per cell. One grid PNG per cell × all-splits.
    def _per_cell_pn_pairs(cell: str):
        """Return per-graph (pred, true) lists for `cell`. Reads slot 0
        from `pergrain_pred`/`pergrain_true` for lcf_fs; aux entries
        elsewhere."""
        out_per_split = {}
        for split_name in _V9_SPLITS:
            ev = split_meta[split_name]["split_eval"]
            if cell == "lcf_fs":
                pred_list = ev.get("pergrain_pred", []) or []
                true_list = ev.get("pergrain_true", []) or []
            else:
                aux_name = f"{cell}_max"
                ent = (ev.get("pergrain_aux", {}) or {}).get(aux_name)
                if ent is None:
                    pred_list, true_list = [], []
                else:
                    pred_list = ent["pred"]
                    true_list = ent["true"]
            out_per_split[split_name] = (pred_list, true_list)
        return out_per_split

    for cell in _V9_CELL_KEYS:
        per_split = _per_cell_pn_pairs(cell)
        hotspot_splits = []
        for split_name in _V9_SPLITS:
            pred_list, true_list = per_split[split_name]
            hotspot_splits.append(
                {
                    "split": split_name,
                    "pred_per_graph": pred_list,
                    "true_per_graph": true_list,
                }
            )
        hotspot_diagnostic_grid_plot(
            res_dir / f"hotspot_diagnostic_{cell}_max.png",
            hotspot_splits,
            suptitle=f"{base_title} — hot-spot diagnostic (per-grain MAX, cell={cell})",
        )

    # Loss-history plot from our captured writer history.
    loss_history_plot(
        res_dir / "loss_history.png",
        train_loss=writer.flat("train"),
        val_loss=writer.flat("val"),
        test_loss=writer.flat("test"),
    )

    # Per-head train-loss curves and CSV. HydraGNN emits one curve per output
    # head (typically task0=per-grain, task1=per-graph with 1+n_aux dims).
    head_names = list(
        cfg.get("NeuralNetwork", {})
        .get("Variables_of_interest", {})
        .get("output_names", [])
    )
    task_history = {t: writer.task_flat(t) for t in writer.task_ids()}
    loss_history_per_head_plot(
        res_dir / "loss_history_per_head.png",
        task_history=task_history,
        head_names=head_names,
    )
    if task_history:
        n_eps = max(len(v) for v in task_history.values())
        with open(res_dir / "loss_history_per_head.csv", "w") as fh:
            cols = sorted(task_history.keys())
            hdr = ["epoch"] + [
                head_names[t] if t < len(head_names) else f"task{t}" for t in cols
            ]
            fh.write(",".join(hdr) + "\n")
            for ep in range(n_eps):
                row = [str(ep + 1)]
                for t in cols:
                    arr = task_history[t]
                    row.append(
                        f"{arr[ep]:.6e}" if ep < len(arr) and arr[ep] == arr[ep] else ""
                    )
                fh.write(",".join(row) + "\n")

    # ---- Distribution overlay grids (5 PNGs total) ----
    # Tail-compression diagnostic: each panel overlays predicted vs true
    # marginal histograms + KDE for one (cell, split). Reuses the same
    # panel-builders as parity grids.
    dist_overlay_grid_plot(
        res_dir / "dist_pn_max.png",
        _build_panels_4cells(_pn_panel_data, "max"),
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-grain MAX dist overlay",
    )
    dist_overlay_grid_plot(
        res_dir / "dist_pn_within_mean.png",
        _build_panels_4cells(_pn_panel_data, "mean"),
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-grain within-mean dist overlay",
    )
    dist_overlay_grid_plot(
        res_dir / "dist_pg_max.png",
        _build_panels_4cells(_pg_panel_data, "max"),
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-graph MAX dist overlay",
    )
    dist_overlay_grid_plot(
        res_dir / "dist_pg_ve_mu.png",
        _build_panels_4cells(_pg_panel_data, "ve_mu"),
        nrows=4,
        ncols=3,
        row_labels=list(_V9_CELL_KEYS),
        col_labels=list(_V9_SPLITS),
        suptitle=f"{base_title} — per-graph within-SVE μ dist overlay",
    )
    dist_overlay_grid_plot(
        res_dir / "dist_mech.png",
        _build_panels_mech("test"),
        nrows=_MECH_PARITY_NROWS,
        ncols=_MECH_PARITY_NCOLS,
        row_labels=_MECH_PARITY_ROWLABELS,
        col_labels=_MECH_PARITY_COLLABELS,
        suptitle=f"{base_title} — mech QoI dist overlay (test split, 2×3 physics-grouped)",
    )

    # Per-size breakdown CSV (per-graph + per-grain, when each is present).
    with open(res_dir / "per_size_breakdown.csv", "w") as f:
        f.write("level,bucket,L_SVE_um,split,n,r2,mae\n")
        for split_name, split_data, split_eval in [
            ("train", train_data, train_metrics),
            ("val", val_data, val_metrics),
            ("test", test_data, test_metrics),
        ]:
            sb_arr = np.array([int(d.size_bucket[0]) for d in split_data])
            # per-graph
            if "pergraph_pred" in split_eval:
                pred_arr = np.array(split_eval["pergraph_pred"])
                true_arr = np.array(split_eval["pergraph_true"])
                for b in sorted(set(sb_arr.tolist())):
                    m = sb_arr == b
                    if m.sum() < 2:
                        continue
                    yt, yp = true_arr[m], pred_arr[m]
                    ss_tot = float(((yt - yt.mean()) ** 2).sum())
                    ss_res = float(((yt - yp) ** 2).sum())
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                    mae = float(np.mean(np.abs(yt - yp)))
                    lsve = BUCKET_TO_LSVE_LABEL.get(b, f"bucket_{b}")
                    f.write(
                        f"pg_lcf_fs_max,{b},{lsve},{split_name},{int(m.sum())},{r2:+.4f},{mae:.4e}\n"
                    )
            # per-grain (slot 0 = lcf_fs_max)
            if "pergrain_pred" in split_eval:
                pn_pred_list = split_eval["pergrain_pred"]
                pn_true_list = split_eval["pergrain_true"]
                for b in sorted(set(sb_arr.tolist())):
                    parts_p, parts_t = [], []
                    n_graphs_in_bucket = 0
                    for gi, (pred, true) in enumerate(zip(pn_pred_list, pn_true_list)):
                        if int(sb_arr[gi]) != b:
                            continue
                        parts_p.append(pred)
                        parts_t.append(true)
                        n_graphs_in_bucket += 1
                    if n_graphs_in_bucket < 2:
                        continue
                    yt = np.concatenate(parts_t)
                    yp = np.concatenate(parts_p)
                    ss_tot = float(((yt - yt.mean()) ** 2).sum())
                    ss_res = float(((yt - yp) ** 2).sum())
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                    mae = float(np.mean(np.abs(yt - yp)))
                    lsve = BUCKET_TO_LSVE_LABEL.get(b, f"bucket_{b}")
                    f.write(
                        f"pn_lcf_fs_max,{b},{lsve},{split_name},{int(yt.size)},{r2:+.4f},{mae:.4e}\n"
                    )
            # per-grain aux: row per (channel, bucket, split)
            if "pergrain_aux" in split_eval and split_eval["pergrain_aux"]:
                for aux_name, aux_ent in split_eval["pergrain_aux"].items():
                    pn_pred_list = aux_ent["pred"]
                    pn_true_list = aux_ent["true"]
                    for b in sorted(set(sb_arr.tolist())):
                        parts_p, parts_t = [], []
                        n_in = 0
                        for gi, (pred, true) in enumerate(
                            zip(pn_pred_list, pn_true_list)
                        ):
                            if int(sb_arr[gi]) != b:
                                continue
                            parts_p.append(pred)
                            parts_t.append(true)
                            n_in += 1
                        if n_in < 2:
                            continue
                        yt = np.concatenate(parts_t)
                        yp = np.concatenate(parts_p)
                        ss_tot = float(((yt - yt.mean()) ** 2).sum())
                        ss_res = float(((yt - yp) ** 2).sum())
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                        mae = float(np.mean(np.abs(yt - yp)))
                        lsve = BUCKET_TO_LSVE_LABEL.get(b, f"bucket_{b}")
                        # v9 naming: emit cell-coded label when the aux name maps
                        # cleanly to (cell, stat); otherwise keep aux_name as-is.
                        cs = _V9_PN_AUX_TO_CELL_STAT.get(aux_name)
                        label = f"pn_{cs[0]}_{cs[1]}" if cs else f"pn_{aux_name}"
                        f.write(
                            f"{label},{b},{lsve},{split_name},{int(yt.size)},{r2:+.4f},{mae:.4e}\n"
                        )
            # per-graph aux: row per (channel, bucket, split)
            if "aux_pergraph" in split_eval and split_eval["aux_pergraph"]:
                for aux_name, aux_ent in split_eval["aux_pergraph"].items():
                    pred_arr = np.array(aux_ent["pred"])
                    true_arr = np.array(aux_ent["true"])
                    for b in sorted(set(sb_arr.tolist())):
                        m = sb_arr == b
                        if m.sum() < 2:
                            continue
                        yt, yp = true_arr[m], pred_arr[m]
                        ss_tot = float(((yt - yt.mean()) ** 2).sum())
                        ss_res = float(((yt - yp) ** 2).sum())
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                        mae = float(np.mean(np.abs(yt - yp)))
                        lsve = BUCKET_TO_LSVE_LABEL.get(b, f"bucket_{b}")
                        # v9 naming: cell-coded when possible; mech QoIs keep
                        # their canonical name (no cell coding).
                        cs = _V9_PG_AUX_TO_CELL_STAT.get(aux_name)
                        if cs:
                            label = f"pg_{cs[0]}_{cs[1]}"
                        elif aux_name in _V9_MECH_QOI_NAMES:
                            label = f"mech_{aux_name}"
                        else:
                            label = f"pg_{aux_name}"
                        f.write(
                            f"{label},{b},{lsve},{split_name},{int(m.sum())},{r2:+.4f},{mae:.4e}\n"
                        )

    # ===========================================================================
    # MANDATORY UQ + diagnostic plots — call at the END of every run.
    # ---------------------------------------------------------------------------
    # User instruction (reinforced multiple times 2026-05-06): every training run
    # MUST emit the full UQ plot set automatically. Each helper below is robust to
    # missing inputs (no σ output → falls back gracefully). Failures here are
    # NEVER allowed to kill the run — they're wrapped in try/except so the
    # main training/eval results are preserved even if plotting blows up.
    # See `feedback_always_emit_uq_plots_2026-05-06.md` in memory.
    # ---------------------------------------------------------------------------
    print("[train_one] generating UQ + diagnostic plot set...", flush=True)
    plot_helpers = [
        ("per-grain σ (pn) UQ plots", "make_l5_sigma_plots", "make_plots_for_run"),
        ("per-graph σ (pg) UQ plots", "make_l6_sigma_plots", "make_plots_for_run"),
        ("pg-MAX direct vs orderstat", "make_pg_max_method_comparison", "make_for_run"),
        (
            "per-cell hotspot diagnostic",
            "make_hotspot_per_cell_plots",
            "make_hotspot_for_run",
        ),
        ("mech-QoI 2×3 physics-grouped", "make_mech_2x3_plot", "make_for_run"),
        ("σ_pn vs σ_voxel diagnostic", "make_sigma_pn_vs_sigma_voxel", "make_for_run"),
        ("Phase M μ_log/σ_log/p90/p99 parity", "make_distshape_plots", "make_for_run"),
    ]
    for label, mod_name, fn_name in plot_helpers:
        try:
            mod = __import__(mod_name)
            fn = getattr(mod, fn_name)
            fn(res_dir)
            print(f"  [uq_plots] {label}: OK", flush=True)
        except Exception as _e:
            print(
                f"  [uq_plots] {label}: SKIPPED ({type(_e).__name__}: {_e})", flush=True
            )

    return metrics


if __name__ == "__main__":
    # Smoke test: floor config, 250um bucket, 30 epochs.
    metrics = train_one(seed=42, exp_tag="smoke_floor", num_epoch=30)
    print("\n=== SMOKE METRICS ===")
    print(json.dumps(metrics, indent=2, default=float))
