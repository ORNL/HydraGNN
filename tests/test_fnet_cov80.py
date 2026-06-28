"""
Parquet-free unit tests for the FNET cov80 forecaster transforms.

Exercises the data-side cov80 logic on tiny in-memory arrays (no parquet, no
DDP, no real PMU files) so it runs in well under a second:

  * keep-and-mask alignment     (stack_node_features)
  * gap imputation              (robust_impute_feature_matrix)
  * train-only observed scaling (scale_features)
  * windowing                   (make_window_dataset): the observed-mask input
    channel, the forecast-window loss mask aligned with y, and configurable
    out_idx.

The example module imports hydragnn at top level, so it is loaded via importlib
(the examples directory is not a package).
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import torch

_EXAMPLE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "fnet_temporal_anomaly_detection",
    "fnet_temporal_anomaly_detection.py",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("fnet_cov80_example", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fnet = _load_example()


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_cov80_impute_fills_gaps():
    """robust_impute interpolates internal gaps and leaves no NaN."""
    X = np.array([[1.0], [np.nan], [3.0], [np.nan], [np.nan]], dtype=np.float32)
    mask = np.array([[1.0], [0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    out = fnet.robust_impute_feature_matrix(
        X.copy(), mask, train_end=5, short_gap_steps=10, medium_gap_steps=10
    )
    assert not np.isnan(out).any()
    assert out[1, 0] == pytest.approx(2.0, abs=1e-5)  # linear interp 1 -> 3


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_cov80_scale_features_observed_only_and_methods():
    """Scaling uses observed values only; method shapes/behaviors are correct."""
    T, N, F = 10, 2, 1
    X = np.ones((T, N, F), dtype=np.float32)
    X[:, 0, 0] = np.arange(T)  # node 0 varies 0..9
    obsm = np.ones((T, N), dtype=np.float32)
    obsm[0, 0] = 0.0  # node 0's value 0 at t=0 is "imputed" -> excluded from stats

    # none = identity
    Xs, c, s = fnet.scale_features(X.copy(), obsm, train_end=T, method="none")
    assert np.allclose(Xs, X) and np.allclose(c, 0.0) and np.allclose(s, 1.0)

    # global = one center/scale per channel, tiled across nodes, observed-only
    _, cg, sg = fnet.scale_features(X.copy(), obsm, train_end=T, method="global")
    assert cg.shape == (N, F)
    assert np.allclose(cg[:, 0], cg[0, 0])  # same value across nodes
    expected = X[:, :, 0][obsm > 0.5].mean()
    assert cg[0, 0] == pytest.approx(float(expected), abs=1e-4)

    # per_node = different center per node
    _, cp, sp = fnet.scale_features(X.copy(), obsm, train_end=T, method="per_node")
    assert cp.shape == (N, F)
    assert cp[0, 0] != cp[1, 0]


# --------------------------------------------------------------------------- #
# Windowing: mask input channel + forecast-window loss mask + out_idx
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_cov80_window_mask_channel_and_loss_mask():
    N, T, Fstat, Tin, H = 4, 12, 4, 3, 2
    Fdyn = fnet.F_DYN
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, N, Fdyn)).astype(np.float32)
    obsm = (rng.random((T, N)) > 0.1).astype(np.float32)
    grid = rng.standard_normal((N, Fstat)).astype(np.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out_idx = np.array([0, 3], dtype=np.int64)  # 2 outputs

    ds = fnet.make_window_dataset(
        X, grid, edge_index, Tin=Tin, H=H, observed_mask=obsm, out_idx=out_idx
    )
    assert len(ds) == (T - H) - (Tin - 1)
    d = ds[0]
    # x_seq = dynamic + 1 mask + static channels
    assert tuple(d.x_seq.shape) == (N, Tin, Fdyn + fnet.F_MASK + Fstat)
    # y and the loss mask are H * len(out_idx) wide and identically shaped
    assert tuple(d.y.shape) == (N, H * len(out_idx))
    assert tuple(d.observed_mask.shape) == tuple(d.y.shape)
    # loss mask is 0/1
    uniq = set(np.unique(d.observed_mask.numpy()).tolist())
    assert uniq.issubset({0.0, 1.0})


# --------------------------------------------------------------------------- #
# Keep-and-mask alignment
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_cov80_stack_keep_and_mask():
    feats = ["freq_dev", "rocof", "angle_delta", "volt_dev"]
    base = pd.Timestamp("2024-01-01")
    grid = pd.date_range(base, periods=10, freq="1s")

    def dev(times):
        df = pd.DataFrame({"timestamp": times})
        for c in feats:
            df[c] = np.arange(len(times), dtype=float)
        return df

    data = {
        1: dev(grid),               # full coverage
        2: dev(grid.delete(5)),     # one internal gap (missing t=5)
        3: dev(grid[[0, 4, 9]]),    # spans full range but only 30% covered
    }
    tvec, X, obsm, keep = fnet.stack_node_features(
        [1, 2, 3], data, dt=1.0, min_device_coverage=0.5, min_step_coverage=0.5
    )

    assert set(keep) == {1, 2}          # device 3 dropped (< 50% coverage)
    assert X.shape[1] == 2               # N = 2 kept nodes
    assert obsm.shape == (X.shape[0], 2)
    assert obsm.min() == 0.0             # device 2's gap is recorded as unobserved
    assert obsm.max() == 1.0
