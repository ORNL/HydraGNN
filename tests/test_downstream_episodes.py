"""Unit tests for downstream episode collapse (collapse_episodes).

Pure pandas, no model, no parquet. Exercises the gap-based collapse, the
auto-gap selection, the peak-window suppression, and the end-to-end collapse().
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(__file__)
_MODULE = os.path.join(
    _HERE,
    "..",
    "examples",
    "fnet_temporal_anomaly_detection",
    "downstream",
    "collapse_episodes.py",
)


def _load():
    spec = importlib.util.spec_from_file_location("collapse_episodes", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ce = _load()


def _clusters(rows):
    df = pd.DataFrame(rows)
    for col, default in (
        ("cluster_size", 3),
        ("max_abs_err", 0.0),
        ("sensor_names", ""),
        ("sensor_indices", ""),
        ("timestamp", ""),
    ):
        if col not in df.columns:
            df[col] = default
    return df


@pytest.mark.mpi_skip()
def pytest_canonical_sensor_set_is_order_independent():
    assert ce.canonical_sensor_set("2;0;1") == "0;1;2"
    assert ce.canonical_sensor_set("0;1;2") == ce.canonical_sensor_set("1;2;0")


@pytest.mark.mpi_skip()
def pytest_collapse_splits_on_time_gap():
    """Same sensor set, a 38-step gap -> two episodes; peak = max mean_abs_err."""
    df = _clusters(
        [
            {"time_idx": 10, "sensor_ids": "0;1;2", "mean_abs_err": 0.5},
            {"time_idx": 11, "sensor_ids": "0;1;2", "mean_abs_err": 0.9},  # peak 1
            {"time_idx": 12, "sensor_ids": "0;1;2", "mean_abs_err": 0.4},
            {"time_idx": 50, "sensor_ids": "0;1;2", "mean_abs_err": 0.7},  # peak 2
            {"time_idx": 51, "sensor_ids": "0;1;2", "mean_abs_err": 0.3},
        ]
    )
    df["sensor_set"] = df["sensor_ids"].map(ce.canonical_sensor_set)
    eps = ce.collapse_to_episodes(df, episode_gap=10)

    assert len(eps) == 2
    e0 = eps[eps["start_time_idx"] == 10].iloc[0]
    assert e0["end_time_idx"] == 12 and e0["peak_time_idx"] == 11 and e0["n_frames"] == 3
    assert e0["peak_mean_abs_err"] == pytest.approx(0.9)
    e1 = eps[eps["start_time_idx"] == 50].iloc[0]
    assert e1["peak_time_idx"] == 50 and e1["n_frames"] == 2


@pytest.mark.mpi_skip()
def pytest_choose_gap_picks_smallest_under_target():
    """A 15-step separation: gap=10 -> 2 episodes, gap=20 -> 1; target 1 -> gap 20."""
    df = _clusters(
        [
            {"time_idx": 0, "sensor_ids": "0;1", "mean_abs_err": 0.5},
            {"time_idx": 1, "sensor_ids": "0;1", "mean_abs_err": 0.6},
            {"time_idx": 16, "sensor_ids": "0;1", "mean_abs_err": 0.7},
            {"time_idx": 17, "sensor_ids": "0;1", "mean_abs_err": 0.4},
        ]
    )
    df["sensor_set"] = df["sensor_ids"].map(ce.canonical_sensor_set)
    used, eps = ce.choose_episode_gap(df, 10, [10, 20], target_max_episodes=1)
    assert used == 20 and len(eps) == 1


@pytest.mark.mpi_skip()
def pytest_peak_window_overlap_vs_global():
    """Within the window: sensor_overlap keeps a non-overlapping episode, global
    suppresses it regardless."""
    eps = pd.DataFrame(
        [
            {"peak_time_idx": 100, "sensor_ids": "0;1"},
            {"peak_time_idx": 110, "sensor_ids": "3;4"},  # within window, no overlap
            {"peak_time_idx": 300, "sensor_ids": "5;6"},  # far away
        ]
    )
    sel_so = ce.select_with_peak_window(
        eps, max_selected=10, window=50, overlap_mode="sensor_overlap"
    )
    assert len(sel_so) == 3  # non-overlapping neighbor kept
    sel_g = ce.select_with_peak_window(
        eps, max_selected=10, window=50, overlap_mode="global"
    )
    assert len(sel_g) == 2  # peak 110 suppressed by 100


@pytest.mark.mpi_skip()
def pytest_collapse_end_to_end_filters_small_clusters():
    clusters = _clusters(
        [
            {"time_idx": 10, "sensor_ids": "0;1;2", "mean_abs_err": 0.9, "cluster_size": 3},
            {"time_idx": 11, "sensor_ids": "0;1;2", "mean_abs_err": 0.5, "cluster_size": 3},
            {"time_idx": 40, "sensor_ids": "7;8", "mean_abs_err": 0.8, "cluster_size": 2},
        ]
    )
    episodes, selected, summary = ce.collapse(
        clusters, episode_gap=10, auto_increase_cutoff=False, min_cluster_size=3
    )
    assert len(episodes) == 1  # the size-2 "7;8" cluster is filtered out
    assert summary["episodes_after_collapse"] == 1
    assert not selected.empty
    assert episodes.iloc[0]["sensor_set"] == "0;1;2"
