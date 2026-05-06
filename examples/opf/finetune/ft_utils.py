"""Shared evaluation and statistics utilities for the FT1 and FT3 fine-tuning pipeline.

Provides:
  EpochCSVWriter     — wraps a TensorBoard SummaryWriter, also writes scalars to CSV
  evaluate_ft1       — classification metrics (accuracy, F1, AUC-ROC) gathered across ranks
  evaluate_ft3       — regression metrics (per-dim MSE, MAE, R²) gathered across ranks
  save_run_results   — writes results.json to a run's log directory
  load_best_or_last_checkpoint — loads the best available checkpoint for post-training eval
"""

import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# TensorBoard + CSV writer wrapper
# ──────────────────────────────────────────────────────────────────────────────

class EpochCSVWriter:
    """Wraps a TensorBoard SummaryWriter and *simultaneously* writes every
    ``add_scalar`` call to a CSV file.

    The TB writer may be ``None`` (non-rank-0 processes) — CSV writing is
    skipped in that case to keep files on rank 0 only.

    Usage::

        tb = model_utils.get_summary_writer(log_name)   # None on non-rank-0
        writer = EpochCSVWriter(tb, f"logs/{log_name}/training_curve.csv")
        # Pass writer to train_validate_test as usual.
        # At the end of the script call writer.close() — CSV is flushed then.
    """

    def __init__(self, tb_writer, csv_path: str):
        self._tb = tb_writer
        self._csv_path = csv_path
        self._rows: list[dict] = []  # [{step, tag, value}]

    # ── intercept add_scalar ──────────────────────────────────────────────

    def add_scalar(self, tag, value, global_step=None, *args, **kwargs):
        if self._tb is not None:
            self._tb.add_scalar(tag, value, global_step, *args, **kwargs)
        if self._tb is not None:  # only rank-0 writes CSV
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = None
            self._rows.append({"step": global_step, "tag": tag, "value": v})

    # ── forward everything else to the underlying writer ──────────────────

    def __getattr__(self, name):
        # Called only for attributes not found on EpochCSVWriter itself
        return getattr(self._tb, name)

    # ── close / flush ─────────────────────────────────────────────────────

    def close(self):
        if self._tb is not None:
            self._flush_csv()
            self._tb.close()

    def _flush_csv(self):
        if not self._rows:
            return
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
        with open(self._csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", "tag", "value"])
            w.writeheader()
            w.writerows(self._rows)


# ──────────────────────────────────────────────────────────────────────────────
# Distributed evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gather_tensors(comm, local_tensor):
    """MPI-gather a 2-D tensor from all ranks to rank 0.

    Returns the concatenated tensor on rank 0, None on all other ranks.
    """
    all_tensors = comm.gather(local_tensor, root=0)
    if comm.Get_rank() != 0:
        return None
    return torch.cat([t for t in all_tensors if t is not None and t.numel() > 0], dim=0)


def evaluate_ft1(model, test_loader, device, comm):
    """Evaluate FT1 binary feasibility classification on the full test set.

    All ranks participate; metrics are computed and returned only on rank 0
    (other ranks receive ``None``).

    Parameters
    ----------
    model : nn.Module (DDP-wrapped is fine)
    test_loader : DataLoader yielding HeteroData with data.y ∈ {0.0, 1.0}
    device : torch.device
    comm : MPI communicator

    Returns (rank 0 only)
    -------
    dict with keys: bce, accuracy, precision, recall, f1, auc_roc,
                    n_samples, n_feasible, n_infeasible,
                    probs (list), labels (list)
    """
    model.eval()
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            # Handle (pred, pred_var) tuple from var_output models
            if isinstance(pred, tuple):
                pred = pred[0]
            logits = pred[0]  # [n_graphs, 1]
            labels = data.y.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            local_logits.append(logits.cpu().float())
            local_labels.append(labels.cpu().float())

    if local_logits:
        lc = torch.cat(local_logits, dim=0)  # [local_N, 1]
        ll = torch.cat(local_labels, dim=0)
    else:
        lc = torch.zeros(0, 1)
        ll = torch.zeros(0, 1)

    logits = _gather_tensors(comm, lc)
    labels = _gather_tensors(comm, ll)

    if comm.Get_rank() != 0:
        return None

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    bce = F.binary_cross_entropy_with_logits(logits, labels).item()
    probs_np = torch.sigmoid(logits).numpy().flatten()
    labels_np = labels.numpy().flatten()
    preds_bin = (probs_np >= 0.5).astype(int)
    labels_int = labels_np.astype(int)

    has_both_classes = len(set(labels_int)) > 1

    return {
        "bce":         bce,
        "accuracy":    float(accuracy_score(labels_int, preds_bin)),
        "precision":   float(precision_score(labels_int, preds_bin, zero_division=0)),
        "recall":      float(recall_score(labels_int, preds_bin, zero_division=0)),
        "f1":          float(f1_score(labels_int, preds_bin, zero_division=0)),
        "auc_roc":     float(roc_auc_score(labels_np, probs_np)) if has_both_classes else None,
        "n_samples":   int(len(labels_int)),
        "n_feasible":  int(labels_int.sum()),
        "n_infeasible": int((1 - labels_int).sum()),
        # Store raw arrays for ROC / confusion matrix plots
        "probs":       probs_np.tolist(),
        "labels":      labels_int.tolist(),
    }


def evaluate_ft3(model, test_loader, device, comm, output_names=None):
    """Evaluate FT3 node-level regression on the full test set.

    All ranks participate; metrics returned on rank 0 only (others get None).

    Parameters
    ----------
    model : nn.Module
    test_loader : NodeBatchAdapter-wrapped DataLoader
                  (each batch has data.y = [total_bus_nodes, out_dim])
    device : torch.device
    comm : MPI communicator
    output_names : list[str], default ["Va", "Vm"]

    Returns (rank 0 only)
    -------
    dict with per-dimension keys: <name>_mse, <name>_mae, <name>_r2,
    plus overall_mse, n_nodes, preds (list), targets (list).
    """
    if output_names is None:
        output_names = ["Va", "Vm"]

    model.eval()
    local_preds: list[torch.Tensor] = []
    local_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]
            node_pred = pred[0].cpu().float()   # [total_nodes, out_dim]
            node_true = data.y.cpu().float()
            if node_true.dim() == 1:
                node_true = node_true.unsqueeze(1)
            # Guard against shape mismatches from padding
            n = min(node_pred.shape[0], node_true.shape[0])
            local_preds.append(node_pred[:n])
            local_targets.append(node_true[:n])

    if local_preds:
        lp = torch.cat(local_preds, dim=0)
        lt = torch.cat(local_targets, dim=0)
    else:
        out_dim = len(output_names)
        lp = torch.zeros(0, out_dim)
        lt = torch.zeros(0, out_dim)

    preds   = _gather_tensors(comm, lp)
    targets = _gather_tensors(comm, lt)

    if comm.Get_rank() != 0:
        return None

    out_dim = min(preds.shape[1], len(output_names))
    metrics = {
        "overall_mse": float(F.mse_loss(preds, targets).item()),
        "n_nodes":     int(preds.shape[0]),
    }
    for i, name in enumerate(output_names[:out_dim]):
        p = preds[:, i]
        t = targets[:, i]
        mse = float(F.mse_loss(p, t).item())
        mae = float(F.l1_loss(p, t).item())
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        metrics[f"{name}_mse"] = mse
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_r2"]  = float(r2)

    # Store a subsample for scatter plots (max 2000 points per rank-0 data)
    sample_idx = torch.randperm(preds.shape[0])[:2000]
    metrics["preds_sample"]   = preds[sample_idx].tolist()
    metrics["targets_sample"] = targets[sample_idx].tolist()

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_run_results(log_name: str, run_meta: dict, test_metrics: dict,
                     logs_root: str = "./logs"):
    """Write ``results.json`` for a single run into its log directory.

    Parameters
    ----------
    log_name : str  — matches the training log name (``logs/<log_name>/``)
    run_meta : dict — any descriptive fields: strategy, arch, regime, etc.
    test_metrics : dict — output of evaluate_ft1 or evaluate_ft3
    logs_root : str — path to the logs directory (default ``"./logs"``)
    """
    out_dir  = os.path.join(logs_root, log_name)
    os.makedirs(out_dir, exist_ok=True)
    payload = {"log_name": log_name, "meta": run_meta, "test_metrics": test_metrics}
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"[results] Saved: {path}")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_best_or_last_checkpoint(log_name: str, model, logs_root: str = "./logs"):
    """Load the best checkpoint from a run's log directory.

    HydraGNN's ``Checkpoint`` class saves checkpoints as ``<log_name>.pk`` (a
    symlink pointing to the epoch file with the best val_loss).  The final
    ``save_model`` call in the training script overwrites the symlink with the
    last-epoch weights.

    This function preferentially loads the EPOCH checkpoint with the lowest
    index under the log dir (the best checkpoint written by the Checkpoint
    class before the final save overwrites the symlink).  Falls back to
    ``<log_name>.pk`` if no epoch files exist.

    Returns True if a checkpoint was loaded, False otherwise.
    """
    log_dir = os.path.join(logs_root, log_name)
    if not os.path.isdir(log_dir):
        return False

    # Prefer the best-epoch checkpoint (lowest val_loss, saved by Checkpoint)
    # The symlink <log_name>.pk always points to the LAST saved file.
    # We can't reliably know the best epoch without re-reading the CSV, so we
    # just load the symlink target (which IS the best if SaveCheckpoint fires
    # and the final save_model hasn't been called yet — but after the script
    # the symlink points to the last epoch).
    # Best practical option: use the file with the SMALLEST val_loss based on
    # training_curve.csv if available.

    csv_path = os.path.join(log_dir, "training_curve.csv")
    best_epoch = _find_best_epoch_from_csv(csv_path)

    if best_epoch is not None:
        cand = os.path.join(log_dir, f"{log_name}_epoch_{best_epoch}.pk")
        if os.path.isfile(cand):
            return _load_ckpt(cand, model)

    # Fallback: symlink / main checkpoint
    main_ckpt = os.path.join(log_dir, f"{log_name}.pk")
    if os.path.exists(main_ckpt):
        return _load_ckpt(main_ckpt, model)

    # Fallback: any .pk file
    pk_files = sorted(glob.glob(os.path.join(log_dir, "*.pk")))
    if pk_files:
        return _load_ckpt(pk_files[-1], model)

    return False


def _find_best_epoch_from_csv(csv_path: str):
    """Return epoch with lowest 'validate error' from training_curve.csv."""
    if not os.path.isfile(csv_path):
        return None
    try:
        best_val, best_epoch = float("inf"), None
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("tag") == "validate error":
                    v = float(row["value"]) if row["value"] else float("inf")
                    if v < best_val:
                        best_val = v
                        best_epoch = row["step"]
        return best_epoch
    except Exception:
        return None


def _load_ckpt(path: str, model) -> bool:
    from collections import OrderedDict
    try:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        target = model.module if hasattr(model, "module") else model
        own_keys = set(target.state_dict().keys())
        if own_keys and not next(iter(own_keys)).startswith("module"):
            new_sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
            sd = new_sd
        target.load_state_dict(sd, strict=False)
        print(f"[ft_utils] Loaded checkpoint: {path}")
        return True
    except Exception as exc:
        print(f"[ft_utils] WARNING: could not load checkpoint {path}: {exc}")
        return False
