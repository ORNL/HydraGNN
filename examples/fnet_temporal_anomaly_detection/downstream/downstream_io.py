"""Downstream I/O adapter: forecaster ``--out_dir`` -> regional-detector inputs.

The T-GCN downstream tasks read a different artifact layout than the
integrated fnet example writes. This module is the single bridge: ``load_split``
returns arrays + graph + timestamps in the shape / units / order the detectors
expect, so the ported detector logic stays identical.

Reconciled here:

  * axis order  ``preds_{split}.npy`` ``[S, N, H, F_out]`` -> ``[S, H, N, F_out]``
  * units       standardized preds/ys -> physical (``feat_mean/std`` restricted
                to ``out_idx``); the fnet scorer already re-adds the last level
                for ``predict_delta`` runs, so no delta handling is needed here
  * mask        ``masks_{split}.npy`` returned as-is (imputed / missing targets
                are 0 and get excluded from residuals downstream)
  * graph       raw ``A_geo.npy`` (``exp(-d/sigma)``); ``A_hat`` is NOT used for
                detection (its D^-1/2 normalization can't be inverted back)
  * timestamps  from the strided ``tvec.npy`` via the contiguous slice-then-window
                mapping (no parquet re-read): sample ``w`` of a split maps to
                ``tvec[lo + Tin + w]``

Only ``val`` / ``test`` / ``pred`` are available — the fnet example scores and
saves those three splits, not ``train``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Dynamic-feature channel indices from the fnet example's
# compute_dynamic_features: 0=freq_dev, 1=rocof, 2=angle_delta, 3=volt_dev.
FREQ_CHANNEL = 0
ROCOF_CHANNEL = 1

_SCORED_SPLITS = ("val", "test", "pred")


@dataclass
class DownstreamData:
    """Everything a regional detector needs, in standalone-compatible form."""

    pred: np.ndarray        # [S, H, N, F_out]  physical units (unless physical=False)
    true: np.ndarray        # [S, H, N, F_out]  recorded signal
    mask: np.ndarray        # [S, H, N, F_out]  1.0 = observed target, 0.0 = imputed/missing
    A_geo: np.ndarray       # [N, N]            raw exp(-d/sigma) similarity, symmetric, 0 diag
    sensor_ids: np.ndarray  # [N]               fdr_id in node order (from sites.csv)
    timestamps: np.ndarray  # [S]               seconds from series start (float64)
    out_idx: np.ndarray     # [F_out]           dynamic channels present in the outputs
    freq_pos: int           # column of freq_dev within out_idx, or -1 if absent
    rocof_pos: int          # column of rocof within out_idx, or -1 (-1 -> estimate from freq)
    split: str
    meta: dict              # raw downstream_meta.json contents


def _split_lo(split: str, n_train: int, n_val: int, n_test: int) -> int:
    """Start index (strided X/tvec timeline) of a split's contiguous segment.

    Mirrors preprocess_stage's seg_bounds: train [0, n_train), val
    [n_train, +n_val), test [.., +n_test), pred [.., T).
    """
    return {
        "train": 0,
        "val": n_train,
        "test": n_train + n_val,
        "pred": n_train + n_val + n_test,
    }[split]


def _denormalize(
    arr: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray, out_idx: np.ndarray
) -> np.ndarray:
    """``[S, H, N, F_out]`` standardized -> physical, using the per-node train
    scaler restricted to the output channels. Safe on empty arrays."""
    if arr.size == 0:
        return arr
    center = feat_mean[:, out_idx]  # [N, F_out]
    scale = feat_std[:, out_idx]  # [N, F_out]
    return arr * scale[None, None, :, :] + center[None, None, :, :]


def load_split(
    out_dir: Union[str, Path],
    split: str,
    *,
    physical: bool = True,
    horizon_offset: str = "first",
) -> DownstreamData:
    """Load one scored split from a forecaster ``out_dir`` as a ``DownstreamData``.

    Parameters
    ----------
    out_dir : path to the fnet example's ``--out_dir``.
    split : one of ``val`` / ``test`` / ``pred``.
    physical : if True, denormalize preds/true to physical units; if False, leave
        them in the model's standardized space.
    horizon_offset : which forecast step a window's single timestamp refers to —
        ``first`` (step ``s+1``, matches horizon_reduce mean/first) or ``last``
        (step ``s+H``, matches horizon_reduce last).
    """
    out_dir = Path(out_dir)
    if split not in _SCORED_SPLITS:
        raise ValueError(
            f"split must be one of {_SCORED_SPLITS} (got {split!r}); the fnet "
            "example only scores those three."
        )
    if horizon_offset not in ("first", "last"):
        raise ValueError("horizon_offset must be 'first' or 'last'")

    meta = json.loads((out_dir / "downstream_meta.json").read_text())

    # [S, N, H, F_out] -> [S, H, N, F_out]
    def _load_shft(name: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.transpose(np.load(out_dir / name), (0, 2, 1, 3))
        ).astype(np.float32)

    pred = _load_shft(f"preds_{split}.npy")
    true = _load_shft(f"ys_{split}.npy")
    mask = _load_shft(f"masks_{split}.npy")

    out_idx = np.load(out_dir / "out_idx.npy").astype(np.int64)

    if physical:
        feat_mean = np.load(out_dir / "feat_mean.npy")
        feat_std = np.load(out_dir / "feat_std.npy")
        pred = _denormalize(pred, feat_mean, feat_std, out_idx)
        true = _denormalize(true, feat_mean, feat_std, out_idx)

    A_geo = np.load(out_dir / "A_geo.npy").astype(np.float64)
    A_geo = np.maximum(A_geo, A_geo.T)  # defensive; already symmetric
    np.fill_diagonal(A_geo, 0.0)

    sensor_ids = pd.read_csv(out_dir / "sites.csv")["fdr_id"].to_numpy()

    # --- timestamps: strided tvec + contiguous slice-then-window mapping --------
    tvec = np.load(out_dir / "tvec.npy").astype(np.float64)
    T = len(tvec)
    n_train = int(T * float(meta["train_frac"]))
    n_val = int(T * float(meta["val_frac"]))
    n_test = int(T * float(meta["test_frac"]))
    lo = _split_lo(split, n_train, n_val, n_test)
    Tin = int(meta["Tin"])
    H = int(meta["horizon"])
    S = pred.shape[0]
    # Forecast window for sample w spans strided steps [lo+Tin+w, lo+Tin+w+H-1].
    base = lo + Tin + (H - 1 if horizon_offset == "last" else 0)
    idx = np.clip(base + np.arange(S), 0, T - 1)
    timestamps = tvec[idx]

    freq_pos = (
        int(np.where(out_idx == FREQ_CHANNEL)[0][0]) if FREQ_CHANNEL in out_idx else -1
    )
    rocof_pos = (
        int(np.where(out_idx == ROCOF_CHANNEL)[0][0]) if ROCOF_CHANNEL in out_idx else -1
    )

    return DownstreamData(
        pred=pred,
        true=true,
        mask=mask,
        A_geo=A_geo,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        out_idx=out_idx,
        freq_pos=freq_pos,
        rocof_pos=rocof_pos,
        split=split,
        meta=meta,
    )
