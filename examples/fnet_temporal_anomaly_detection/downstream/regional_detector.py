"""Regional-anomaly cluster detection on the fnet forecaster residuals.

Unifies the two T-GCN detectors behind ``--detector``:

  * ``map_hot`` (default) — high freq_dev error only, connected by geographic
    adjacency above a raw weight threshold. Loose filter -> many raw clusters,
    intended to be reduced to episodes by the collapse stage.
  * ``strict`` — high freq_dev AND RoCoF error simultaneously, connected by
    strong (edge-quantile) adjacency, with extra guards against degenerate
    clusters. Tight filter -> ~tens of clusters.

Rewired onto the downstream adapter (``downstream_io.load_split``), so it reads a
forecaster ``--out_dir`` and inherits: mask-aware residuals (imputed targets
excluded), timestamps from ``tvec`` (no parquet re-read), and the RoCoF channel
branch (use a real rocof output if present, else estimate from freq).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:  # package import (e.g. from tests) vs script import (same-dir on sys.path)
    from .downstream_io import DownstreamData, load_split
except ImportError:  # pragma: no cover - exercised only when run as a script
    from downstream_io import DownstreamData, load_split


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class DetectConfig:
    detector: str = "map_hot"  # "map_hot" | "strict"
    horizon_reduce: str = "mean"  # "mean" | "first" | "last"
    min_cluster_size: int = 2
    max_rows: int = 200
    # map_hot
    color_max_quantile: float = 0.90
    edge_threshold: float = 0.0
    # strict
    freq_quantile: float = 0.98
    rocof_quantile: float = 0.98
    edge_quantile: float = 0.85
    true_rocof_min_abs: float = 5e-4


# --------------------------------------------------------------------------- #
# Core numerics
# --------------------------------------------------------------------------- #


def connected_components(adj_mask: np.ndarray) -> list[list[int]]:
    """Connected components of a boolean adjacency (iterative DFS)."""
    n = adj_mask.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(adj_mask[u]).tolist():
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def masked_horizon_reduce(
    arr: np.ndarray, mask: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the H axis of ``[S, H, N, F]`` to ``[S, N, F]``, observed-only.

    Returns ``(reduced, valid)`` where ``valid`` ``[S, N, F]`` is True where at
    least one observed step contributed. ``mean`` averages over observed steps;
    ``first`` / ``last`` take that horizon step (valid iff it is observed).
    """
    if mode == "mean":
        num = (arr * mask).sum(axis=1)
        den = mask.sum(axis=1)
        reduced = num / np.clip(den, 1.0, None)
        valid = den > 0.5
        return reduced, valid
    step = 0 if mode == "first" else -1
    return arr[:, step], mask[:, step] > 0.5


@dataclass
class _Residuals:
    pf: np.ndarray  # [S, N] pred freq (reduced, physical)
    tf: np.ndarray  # [S, N] true freq
    freq_dev: np.ndarray  # [S, N] |pf - tf| where valid, else 0
    valid: np.ndarray  # [S, N] bool, freq observed
    pr: np.ndarray | None  # [S, N] pred rocof (real channel or first-diff estimate)
    tr: np.ndarray | None  # [S, N] true rocof
    rocof_dev: np.ndarray | None  # [S, N] |pr - tr|
    rocof_source: str  # "channel" | "estimated" | "none"


def compute_residuals(data: DownstreamData, cfg: DetectConfig) -> _Residuals:
    if data.freq_pos < 0:
        raise ValueError(
            "freq_dev (channel 0) is not among the forecaster outputs (out_idx="
            f"{data.out_idx.tolist()}); regional detection needs it."
        )
    red_p, valid_p = masked_horizon_reduce(data.pred, data.mask, cfg.horizon_reduce)
    red_t, _ = masked_horizon_reduce(data.true, data.mask, cfg.horizon_reduce)

    fp = data.freq_pos
    pf, tf = red_p[:, :, fp], red_t[:, :, fp]
    valid = valid_p[:, :, fp]
    freq_dev = np.abs(pf - tf) * valid

    pr = tr = rocof_dev = None
    rocof_source = "none"
    need_rocof = cfg.detector == "strict"
    if need_rocof:
        if data.rocof_pos >= 0:
            rp = data.rocof_pos
            pr, tr = red_p[:, :, rp], red_t[:, :, rp]
            rocof_source = "channel"
        else:
            # Fallback: estimate rocof as first difference over samples of the
            # reduced freq (identical to the forecaster's own rocof =
            # np.diff(freq_dev)).
            pr = np.zeros_like(pf)
            tr = np.zeros_like(tf)
            pr[1:] = pf[1:] - pf[:-1]
            tr[1:] = tf[1:] - tf[:-1]
            rocof_source = "estimated"
        rocof_dev = np.abs(pr - tr) * valid

    return _Residuals(pf, tf, freq_dev, valid, pr, tr, rocof_dev, rocof_source)


def _thresholded_adjacency(A_geo: np.ndarray, cfg: DetectConfig) -> tuple[np.ndarray, float]:
    """Boolean edge mask + the threshold used, per detector."""
    A = np.maximum(A_geo, A_geo.T).astype(float)
    np.fill_diagonal(A, 0.0)
    if cfg.detector == "strict":
        pos = A[A > 0]
        thr = (
            float(np.quantile(pos, min(max(cfg.edge_quantile, 0.01), 0.9999)))
            if pos.size
            else 0.0
        )
        return A >= thr, thr
    return A > float(cfg.edge_threshold), float(cfg.edge_threshold)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def detect_clusters(
    data: DownstreamData, cfg: DetectConfig, name_map: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(clusters_all, clusters_top, thresholds)``.

    Same connected-components core for both detectors; the activation criterion,
    adjacency threshold, ranking, and degeneracy guards are detector-specific.
    ``name_map`` optionally maps fdr_id -> human name (else ``Sensor-<id>``).
    """
    res = compute_residuals(data, cfg)
    N = data.sensor_ids.shape[0]
    edge_mask, edge_thr = _thresholded_adjacency(data.A_geo, cfg)

    # Activation thresholds over observed cells only.
    obs = res.valid
    freq_obs = res.freq_dev[obs]
    if freq_obs.size == 0:
        raise ValueError("No observed forecast cells in this split; nothing to detect.")
    freq_thr = float(
        np.quantile(freq_obs, min(max(cfg.color_max_quantile if cfg.detector == "map_hot" else cfg.freq_quantile, 0.01), 0.9999))
    )
    if cfg.detector == "strict":
        rocof_obs = res.rocof_dev[obs]
        rocof_thr = float(
            np.quantile(rocof_obs, min(max(cfg.rocof_quantile, 0.5), 0.9999))
        )
        active = (res.freq_dev >= freq_thr) & (res.rocof_dev >= rocof_thr) & obs
    else:
        rocof_thr = None
        active = (res.freq_dev >= freq_thr) & obs

    name_map = name_map or {}

    rows: list[dict] = []
    cluster_id = 0
    for t in range(active.shape[0]):
        active_nodes = np.flatnonzero(active[t])
        if active_nodes.size < cfg.min_cluster_size:
            continue
        induced = edge_mask[np.ix_(active_nodes, active_nodes)]
        for comp in connected_components(induced):
            if len(comp) < cfg.min_cluster_size:
                continue
            nodes = active_nodes[np.array(comp, dtype=int)]

            if cfg.detector == "strict":
                tr_vals = res.tr[t, nodes]
                if float(np.max(np.abs(tr_vals))) <= float(cfg.true_rocof_min_abs):
                    continue
                tf_vals = res.tf[t, nodes]
                if (
                    float(np.max(tf_vals) - np.min(tf_vals)) <= 1e-12
                    and float(np.max(tr_vals) - np.min(tr_vals)) <= 1e-12
                ):
                    continue
                freq_mean = float(np.mean(res.freq_dev[t, nodes]))
                rocof_mean = float(np.mean(res.rocof_dev[t, nodes]))
                if freq_mean <= 1e-12 or rocof_mean <= 1e-12:
                    continue
                magnitude = freq_mean + rocof_mean
                score = len(nodes) * magnitude
            else:
                freq_mean = float(np.mean(res.freq_dev[t, nodes]))
                rocof_mean = None
                magnitude = freq_mean
                score = len(nodes) * magnitude

            ids = [int(data.sensor_ids[i]) for i in nodes.tolist()]
            row = {
                "cluster_id": cluster_id,
                "time_idx": int(t),
                "timestamp": f"{float(data.timestamps[t]):.3f}s",
                "cluster_size": int(len(nodes)),
                "mean_abs_err": freq_mean,
                "max_abs_err": float(np.max(res.freq_dev[t, nodes])),
                "score": float(score),
                "magnitude": float(magnitude),
                "sensor_ids": ";".join(str(x) for x in ids),
                "sensor_names": ";".join(name_map.get(x, f"Sensor-{x}") for x in ids),
                "sensor_indices": ";".join(str(int(x)) for x in nodes.tolist()),
            }
            if cfg.detector == "strict":
                row["mean_freq_dev"] = freq_mean
                row["mean_rocof_dev"] = rocof_mean
            rows.append(row)
            cluster_id += 1

    if not rows:
        raise ValueError(
            "No clusters detected. Lower the quantile / min_cluster_size, or check "
            "that the split has real disturbances."
        )

    clusters_all = pd.DataFrame(rows)
    if cfg.detector == "strict":
        clusters_all = clusters_all.sort_values(
            ["score", "cluster_size", "magnitude"], ascending=False
        ).reset_index(drop=True)
    else:
        clusters_all = clusters_all.sort_values(
            ["cluster_size", "mean_abs_err", "max_abs_err"], ascending=False
        ).reset_index(drop=True)
    clusters_top = clusters_all.head(max(1, cfg.max_rows)).copy()

    thresholds = {
        "detector": cfg.detector,
        "freq_threshold": freq_thr,
        "rocof_threshold": rocof_thr,
        "edge_threshold": edge_thr,
        "rocof_source": res.rocof_source,
        "horizon_reduce": cfg.horizon_reduce,
        "num_samples": int(active.shape[0]),
        "num_sensors": int(N),
        "clusters_detected": int(len(clusters_all)),
    }
    return clusters_all, clusters_top, thresholds


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _out_names(detector: str) -> tuple[str, str]:
    if detector == "map_hot":
        return "map_freq_hot_clusters_all.csv", "map_freq_hot_clusters_top.csv"
    return "clusters_all.csv", "clusters_top.csv"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regional-anomaly cluster detection on forecaster residuals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", required=True, help="forecaster output dir (from the fnet example)")
    p.add_argument("--split", default="pred", choices=["val", "test", "pred"])
    p.add_argument("--detector", default="map_hot", choices=["map_hot", "strict"])
    p.add_argument("--results_dir", default=None, help="where to write cluster CSVs (default: <out_dir>/downstream)")
    p.add_argument("--horizon_reduce", default="mean", choices=["mean", "first", "last"])
    p.add_argument("--min_cluster_size", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=200)
    # map_hot
    p.add_argument("--color_max_quantile", type=float, default=0.90)
    p.add_argument("--edge_threshold", type=float, default=0.0)
    # strict
    p.add_argument("--freq_quantile", type=float, default=0.98)
    p.add_argument("--rocof_quantile", type=float, default=0.98)
    p.add_argument("--edge_quantile", type=float, default=0.85)
    p.add_argument("--true_rocof_min_abs", type=float, default=5e-4)
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = DetectConfig(
        detector=args.detector,
        horizon_reduce=args.horizon_reduce,
        min_cluster_size=args.min_cluster_size,
        max_rows=args.max_rows,
        color_max_quantile=args.color_max_quantile,
        edge_threshold=args.edge_threshold,
        freq_quantile=args.freq_quantile,
        rocof_quantile=args.rocof_quantile,
        edge_quantile=args.edge_quantile,
        true_rocof_min_abs=args.true_rocof_min_abs,
    )
    data = load_split(args.out_dir, args.split, horizon_offset="first")

    # Human-readable node names from sites.csv (grid_name), aligned by fdr_id.
    name_map = {}
    sites_csv = Path(args.out_dir) / "sites.csv"
    if sites_csv.exists():
        sites = pd.read_csv(sites_csv)
        if "grid_name" in sites.columns:
            name_map = {
                int(fid): str(nm)
                for fid, nm in zip(sites["fdr_id"], sites["grid_name"])
            }

    clusters_all, clusters_top, thresholds = detect_clusters(data, cfg, name_map)

    results_dir = Path(args.results_dir) if args.results_dir else Path(args.out_dir) / "downstream"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_name, top_name = _out_names(cfg.detector)
    clusters_all.to_csv(results_dir / all_name, index=False)
    clusters_top.to_csv(results_dir / top_name, index=False)
    with open(results_dir / f"{cfg.detector}_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"[detect] detector={cfg.detector} split={args.split} rocof={thresholds['rocof_source']}")
    print(
        f"[detect] freq_thr={thresholds['freq_threshold']:.5f} "
        + (f"rocof_thr={thresholds['rocof_threshold']:.5f} " if thresholds["rocof_threshold"] else "")
        + f"edge_thr={thresholds['edge_threshold']:.5f}"
    )
    print(f"[detect] clusters: all={len(clusters_all)} top={len(clusters_top)}")
    print(f"[save]   {results_dir / all_name}")
    print(f"[save]   {results_dir / top_name}")


if __name__ == "__main__":
    main()
