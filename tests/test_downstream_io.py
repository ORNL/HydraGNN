"""Parquet-free unit tests for the downstream adapter.

Builds a tiny synthetic forecaster ``out_dir`` on disk (random arrays, a 3-row
sites.csv, a strided tvec, and downstream_meta.json) and checks the four transforms
the adapter is responsible for:

  * axis swap        preds_{split} [S,N,H,F_out] -> [S,H,N,F_out]
  * denormalize      standardized -> physical via feat_mean/std restricted to out_idx
  * timestamps       strided-tvec window mapping  tvec[lo + Tin + w]
  * out_idx branch   freq_pos / rocof_pos (DB4: default [0,2,3] drops rocof)

The adapter imports only numpy/pandas (no hydragnn, no torch), so it is loaded
directly from its file path.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ADAPTER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "fnet_temporal_anomaly_detection",
    "downstream",
    "downstream_io.py",
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("downstream_io", _ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module in sys.modules
    # (dataclasses looks up cls.__module__ there during class processing).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_adapter()


# --------------------------------------------------------------------------- #
# Synthetic out_dir builder
# --------------------------------------------------------------------------- #

_F_DYN = 4  # freq_dev, rocof, angle_delta, volt_dev
_T = 200
_TRAIN, _VAL, _TEST = 0.8, 0.05, 0.05  # -> n_train=160, n_val=10, n_test=10
_TIN, _H = 5, 3


def _write_out_dir(out_dir, split, out_idx, S, N):
    """Write a minimal forecaster out_dir; return the raw (pre-transform) arrays."""
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    F_out = len(out_idx)
    out_idx = np.asarray(out_idx, dtype=np.int64)

    # preds/ys stored as [S, N, H, F_out]; encode indices so the transpose is
    # unambiguously checkable: value = s*1000 + n*100 + h*10 + f.
    s, n, h, f = np.meshgrid(
        np.arange(S), np.arange(N), np.arange(_H), np.arange(F_out), indexing="ij"
    )
    preds_raw = (s * 1000 + n * 100 + h * 10 + f).astype(np.float32)
    ys_raw = preds_raw + 0.5
    masks_raw = (((s + n + h + f) % 2) == 0).astype(np.float32)  # 0/1 checkerboard

    np.save(os.path.join(out_dir, f"preds_{split}.npy"), preds_raw)
    np.save(os.path.join(out_dir, f"ys_{split}.npy"), ys_raw)
    np.save(os.path.join(out_dir, f"masks_{split}.npy"), masks_raw)
    np.save(os.path.join(out_dir, "out_idx.npy"), out_idx)

    # Non-trivial per-node scaler over all F_dyn channels.
    feat_mean = (np.arange(N * _F_DYN).reshape(N, _F_DYN) + 1.0).astype(np.float32)
    feat_std = (np.arange(N * _F_DYN).reshape(N, _F_DYN) * 0.1 + 2.0).astype(np.float32)
    np.save(os.path.join(out_dir, "feat_mean.npy"), feat_mean)
    np.save(os.path.join(out_dir, "feat_std.npy"), feat_std)

    # Asymmetric raw graph with a diagonal -> adapter must symmetrize + zero diag.
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N - 1):
        A[i, i + 1] = 0.3 + 0.1 * i
    np.fill_diagonal(A, 5.0)
    np.save(os.path.join(out_dir, "A_geo.npy"), A)

    pd.DataFrame(
        {
            "site": [f"S{i}" for i in range(N)],
            "fdr_id": [1000 + i for i in range(N)],
            "grid_name": [f"grid{i}" for i in range(N)],
        }
    ).to_csv(os.path.join(out_dir, "sites.csv"), index=False)

    tvec = (np.arange(_T) * 0.5).astype(np.float32)  # 2 Hz
    np.save(os.path.join(out_dir, "tvec.npy"), tvec)

    meta = {
        "Tin": _TIN,
        "horizon": _H,
        "train_frac": _TRAIN,
        "val_frac": _VAL,
        "test_frac": _TEST,
        "stride": 1,
        "dt": 0.5,
        "scaling": "per_node",
        "predict_delta": False,
        "n_nodes": N,
        "F_out": F_out,
    }
    with open(os.path.join(out_dir, "downstream_meta.json"), "w") as fp:
        json.dump(meta, fp)

    return preds_raw, ys_raw, masks_raw, feat_mean, feat_std, tvec


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_downstream_axis_swap_and_mask(tmp_path):
    """preds [S,N,H,F] -> [S,H,N,F]; mask same shape, values in {0,1}."""
    S, N, out_idx = 8, 4, [0, 2, 3]
    preds_raw, _, masks_raw, _, _, _ = _write_out_dir(tmp_path, "test", out_idx, S, N)

    d = tb.load_split(tmp_path, "test", physical=False)

    assert d.pred.shape == (S, _H, N, len(out_idx))
    assert d.mask.shape == d.pred.shape
    # Spot-check the transpose: loaded[s,h,n,f] == raw[s,n,h,f].
    for (si, hi, ni, fi) in [(0, 0, 0, 0), (7, 2, 3, 2), (3, 1, 2, 1)]:
        assert d.pred[si, hi, ni, fi] == preds_raw[si, ni, hi, fi]
        assert d.mask[si, hi, ni, fi] == masks_raw[si, ni, hi, fi]
    assert set(np.unique(d.mask).tolist()).issubset({0.0, 1.0})


@pytest.mark.mpi_skip()
def pytest_downstream_denormalize(tmp_path):
    """physical=True applies feat_mean/std restricted to out_idx, per node."""
    S, N, out_idx = 8, 4, [0, 2, 3]
    _, _, _, feat_mean, feat_std, _ = _write_out_dir(tmp_path, "test", out_idx, S, N)

    std = tb.load_split(tmp_path, "test", physical=False)
    phys = tb.load_split(tmp_path, "test", physical=True)

    oi = np.asarray(out_idx)
    expected = std.pred * feat_std[:, oi][None, None, :, :] + feat_mean[:, oi][
        None, None, :, :
    ]
    assert np.allclose(phys.pred, expected, atol=1e-4)
    # Sanity: denormalization actually changed the values.
    assert not np.allclose(phys.pred, std.pred)


@pytest.mark.mpi_skip()
def pytest_downstream_timestamps(tmp_path):
    """Window w of a split maps to tvec[lo + Tin + w] (strided-index space)."""
    S, N, out_idx = 8, 4, [0, 2, 3]
    _, _, _, _, _, tvec = _write_out_dir(tmp_path, "test", out_idx, S, N)

    d = tb.load_split(tmp_path, "test", horizon_offset="first")

    n_train = int(_T * _TRAIN)  # 160
    n_val = int(_T * _VAL)  # 10
    lo = n_train + n_val  # test segment start = 170
    base = lo + _TIN  # 175
    expected = tvec[base + np.arange(S)]
    assert np.allclose(d.timestamps, expected)

    # horizon_offset='last' shifts by H-1.
    d_last = tb.load_split(tmp_path, "test", horizon_offset="last")
    assert np.allclose(d_last.timestamps, tvec[base + (_H - 1) + np.arange(S)])


@pytest.mark.mpi_skip()
def pytest_downstream_out_idx_branch(tmp_path):
    """DB4: default outputs [0,2,3] drop rocof (rocof_pos=-1); [0,1,2,3] keep it."""
    d_default = tb.load_split(
        _fresh(tmp_path, "a", [0, 2, 3]), "test", physical=False
    )
    assert d_default.freq_pos == 0
    assert d_default.rocof_pos == -1

    d_full = tb.load_split(_fresh(tmp_path, "b", [0, 1, 2, 3]), "test", physical=False)
    assert d_full.freq_pos == 0
    assert d_full.rocof_pos == 1


@pytest.mark.mpi_skip()
def pytest_downstream_graph_symmetrized(tmp_path):
    """A_geo is symmetrized with a zeroed diagonal even from an asymmetric save."""
    d = tb.load_split(_fresh(tmp_path, "g", [0, 2, 3]), "test", physical=False)
    assert np.allclose(d.A_geo, d.A_geo.T)
    assert np.allclose(np.diag(d.A_geo), 0.0)


@pytest.mark.mpi_skip()
def pytest_downstream_rejects_train_split(tmp_path):
    """train is never scored/saved by the fnet example -> explicit error."""
    _write_out_dir(tmp_path, "test", [0, 2, 3], 8, 4)
    with pytest.raises(ValueError):
        tb.load_split(tmp_path, "train")


def _fresh(tmp_path, name, out_idx):
    d = tmp_path / name
    _write_out_dir(d, "test", out_idx, 8, 4)
    return d
