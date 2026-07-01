"""Unit tests for downstream regional cluster detection (regional_detector).

No parquet, no model. Building blocks (connected components, mask-aware horizon
reduce, rocof branch, adjacency thresholding) are tested directly; the two
detectors are run end-to-end on tiny hand-built DownstreamData where the cell
counts are fixed so the internal quantile thresholds land at known values.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_DOWNSTREAM = os.path.join(
    _HERE, "..", "examples", "fnet_temporal_anomaly_detection", "downstream"
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_DOWNSTREAM, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so dataclass / absolute import resolve
    spec.loader.exec_module(mod)
    return mod


tb = _load("downstream_io", "downstream_io.py")  # load first (regional imports it)
rd = _load("regional_detector", "regional_detector.py")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _path_graph(n):
    """Undirected path 0-1-...-(n-1), unit weights."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def _mk_data(pred, true, mask, A, out_idx, timestamps=None):
    out_idx = np.asarray(out_idx, dtype=np.int64)
    N = pred.shape[2]
    S = pred.shape[0]
    freq_pos = int(np.where(out_idx == 0)[0][0]) if 0 in out_idx else -1
    rocof_pos = int(np.where(out_idx == 1)[0][0]) if 1 in out_idx else -1
    return tb.DownstreamData(
        pred=pred.astype(np.float32),
        true=true.astype(np.float32),
        mask=mask.astype(np.float32),
        A_geo=A,
        sensor_ids=np.array([100 + i for i in range(N)]),
        timestamps=(np.arange(S, dtype=float) if timestamps is None else timestamps),
        out_idx=out_idx,
        freq_pos=freq_pos,
        rocof_pos=rocof_pos,
        split="test",
        meta={},
    )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_connected_components_path():
    A = _path_graph(4) > 0
    comps = rd.connected_components(A)
    assert len(comps) == 1 and sorted(comps[0]) == [0, 1, 2, 3]
    # Break the middle edge -> two components.
    A2 = A.copy()
    A2[1, 2] = A2[2, 1] = False
    comps2 = rd.connected_components(A2)
    assert sorted(sorted(c) for c in comps2) == [[0, 1], [2, 3]]


@pytest.mark.mpi_skip()
def pytest_masked_horizon_reduce_excludes_unobserved():
    """Masked steps are excluded from the mean; all-masked -> invalid."""
    # [S=1, H=2, N=2, F=1]; node0 has 2nd step masked, node1 fully masked.
    arr = np.array([[[[2.0], [9.0]], [[6.0], [9.0]]]])  # [1,2,2,1]
    mask = np.array([[[[1.0], [0.0]], [[1.0], [0.0]]]])
    red, valid = rd.masked_horizon_reduce(arr, mask, "mean")
    assert red.shape == (1, 2, 1) and valid.shape == (1, 2, 1)
    assert red[0, 0, 0] == pytest.approx(4.0)  # (2 + 6) / 2
    assert valid[0, 0, 0]
    assert red[0, 1, 0] == pytest.approx(0.0)  # fully masked -> 0
    assert not valid[0, 1, 0]


@pytest.mark.mpi_skip()
def pytest_thresholded_adjacency_modes():
    A = _path_graph(3)
    A[0, 0] = 5.0  # self-loop that must be zeroed
    cfg_map = rd.DetectConfig(detector="map_hot", edge_threshold=0.0)
    mask_map, thr_map = rd._thresholded_adjacency(A, cfg_map)
    assert thr_map == 0.0
    assert mask_map[0, 1] and not mask_map[0, 0]  # diagonal removed
    cfg_str = rd.DetectConfig(detector="strict", edge_quantile=0.5)
    mask_str, thr_str = rd._thresholded_adjacency(A, cfg_str)
    assert thr_str == 1.0  # all positive weights equal 1.0
    assert mask_str[1, 2]


@pytest.mark.mpi_skip()
def pytest_rocof_source_channel_vs_estimated():
    S, N = 4, 3
    z = np.zeros((S, 1, N, 2), dtype=np.float32)
    m = np.ones((S, 1, N, 2), dtype=np.float32)
    data_ch = _mk_data(z.copy(), z.copy(), m, _path_graph(N), [0, 1])
    res_ch = rd.compute_residuals(data_ch, rd.DetectConfig(detector="strict"))
    assert res_ch.rocof_source == "channel"

    z1 = np.zeros((S, 1, N, 1), dtype=np.float32)
    m1 = np.ones((S, 1, N, 1), dtype=np.float32)
    data_est = _mk_data(z1.copy(), z1.copy(), m1, _path_graph(N), [0])
    res_est = rd.compute_residuals(data_est, rd.DetectConfig(detector="strict"))
    assert res_est.rocof_source == "estimated"


# --------------------------------------------------------------------------- #
# End-to-end detection (deterministic thresholds)
# --------------------------------------------------------------------------- #


@pytest.mark.mpi_skip()
def pytest_map_hot_detects_injected_cluster():
    """9 zeros + [10,12,14] at t=2; q=0.75 -> thr=2.5 -> one size-3 cluster."""
    S, N = 4, 3
    pred = np.zeros((S, 1, N, 1), dtype=np.float32)
    true = np.zeros((S, 1, N, 1), dtype=np.float32)
    pred[2, 0, :, 0] = [10.0, 12.0, 14.0]
    mask = np.ones((S, 1, N, 1), dtype=np.float32)
    data = _mk_data(pred, true, mask, _path_graph(N), [0])

    cfg = rd.DetectConfig(
        detector="map_hot", color_max_quantile=0.75, min_cluster_size=2
    )
    clusters_all, clusters_top, thr = rd.detect_clusters(data, cfg)

    assert len(clusters_all) == 1
    row = clusters_all.iloc[0]
    assert row["time_idx"] == 2
    assert row["cluster_size"] == 3
    assert row["sensor_indices"] == "0;1;2"
    assert row["sensor_ids"] == "100;101;102"
    assert thr["rocof_source"] == "none"
    assert abs(thr["freq_threshold"] - 2.5) < 1e-9


@pytest.mark.mpi_skip()
def pytest_strict_requires_freq_and_rocof():
    """Sustained step at t>=2: freq active t=2,t=3 but estimated rocof only t=2,
    so strict fires exactly once (t=2)."""
    S, N = 4, 3
    pred = np.zeros((S, 1, N, 1), dtype=np.float32)
    true = np.zeros((S, 1, N, 1), dtype=np.float32)
    true[2, 0, :, 0] = [10.0, 12.0, 14.0]
    true[3, 0, :, 0] = [10.0, 12.0, 14.0]
    mask = np.ones((S, 1, N, 1), dtype=np.float32)
    data = _mk_data(pred, true, mask, _path_graph(N), [0])

    cfg = rd.DetectConfig(
        detector="strict",
        freq_quantile=0.5,  # -> thr 5.0 (freq active t=2,t=3)
        rocof_quantile=0.8,  # -> thr 8.0 (rocof active t=2 only)
        edge_quantile=0.5,
        min_cluster_size=2,
        true_rocof_min_abs=1e-6,
    )
    clusters_all, _, thr = rd.detect_clusters(data, cfg)

    assert thr["rocof_source"] == "estimated"
    assert len(clusters_all) == 1
    row = clusters_all.iloc[0]
    assert row["time_idx"] == 2
    assert row["cluster_size"] == 3
    assert "mean_rocof_dev" in clusters_all.columns


@pytest.mark.mpi_skip()
def pytest_strict_spread_guard_drops_flat_cluster():
    """If every node in the cluster has the SAME true value (no cross-sensor
    spread), strict's guard drops it -> no clusters."""
    S, N = 4, 3
    pred = np.zeros((S, 1, N, 1), dtype=np.float32)
    true = np.zeros((S, 1, N, 1), dtype=np.float32)
    true[2, 0, :, 0] = [12.0, 12.0, 12.0]  # identical -> zero spread
    mask = np.ones((S, 1, N, 1), dtype=np.float32)
    data = _mk_data(pred, true, mask, _path_graph(N), [0])

    cfg = rd.DetectConfig(
        detector="strict",
        freq_quantile=0.5,
        rocof_quantile=0.5,
        edge_quantile=0.5,
        min_cluster_size=2,
        true_rocof_min_abs=1e-6,
    )
    with pytest.raises(ValueError):
        rd.detect_clusters(data, cfg)


@pytest.mark.mpi_skip()
def pytest_out_names_per_detector():
    assert rd._out_names("map_hot")[0] == "map_freq_hot_clusters_all.csv"
    assert rd._out_names("strict")[0] == "clusters_all.csv"
