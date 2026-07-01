"""Collapse repeated regional clusters into episodes.

The map_hot detector emits one cluster row per timestamp, so a single physical
disturbance appears as many near-identical rows (same sensor set, adjacent
times). This stage collapses them: group by sensor set, split each group on a
time gap, and keep the peak frame per run -> one row per episode. A peak-window
pass then suppresses near-duplicate episodes that share sensors.

Pure pandas over the detector's ``map_freq_hot_clusters_all.csv`` — no model, no
arrays, no plotting. Consumes/produces CSVs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Collapse
# --------------------------------------------------------------------------- #


def parse_sensor_ids(sensor_ids_text: str) -> list[int]:
    return [int(x) for x in str(sensor_ids_text).split(";") if str(x) != ""]


def canonical_sensor_set(sensor_ids_text: str) -> str:
    """Order-independent key for a cluster's sensor set."""
    return ";".join(str(x) for x in sorted(parse_sensor_ids(sensor_ids_text)))


def collapse_to_episodes(df: pd.DataFrame, episode_gap: int) -> pd.DataFrame:
    """Collapse per-timestamp clusters into episodes.

    Within each ``sensor_set`` group, consecutive rows whose ``time_idx`` gap is
    <= ``episode_gap`` belong to the same episode; the episode's representative
    frame is the one with the largest ``mean_abs_err``.
    """
    rows: list[dict] = []
    episode_id = 0
    for sensor_set, group in df.sort_values(["sensor_set", "time_idx"]).groupby(
        "sensor_set", sort=False
    ):
        g = group.sort_values("time_idx").reset_index(drop=True)
        t = g["time_idx"].to_numpy(dtype=int)
        start = 0
        for i in range(1, len(g) + 1):
            boundary = i == len(g) or (t[i] - t[i - 1]) > episode_gap
            if not boundary:
                continue
            seg = g.iloc[start:i]
            peak_row = seg.loc[int(seg["mean_abs_err"].idxmax())]
            rows.append(
                {
                    "episode_id": episode_id,
                    "sensor_set": sensor_set,
                    "cluster_size": int(peak_row["cluster_size"]),
                    "start_time_idx": int(seg["time_idx"].min()),
                    "end_time_idx": int(seg["time_idx"].max()),
                    "peak_time_idx": int(peak_row["time_idx"]),
                    "n_frames": int(len(seg)),
                    "duration_steps": int(
                        seg["time_idx"].max() - seg["time_idx"].min() + 1
                    ),
                    "peak_mean_abs_err": float(seg["mean_abs_err"].max()),
                    "peak_max_abs_err": float(seg["max_abs_err"].max()),
                    "sensor_ids": str(peak_row["sensor_ids"]),
                    "sensor_names": str(peak_row.get("sensor_names", "")),
                    "sensor_indices": str(peak_row["sensor_indices"]),
                    "peak_timestamp": str(peak_row.get("timestamp", "")),
                }
            )
            episode_id += 1
            start = i
    return pd.DataFrame(rows)


def choose_episode_gap(
    df: pd.DataFrame,
    initial_gap: int,
    candidate_gaps: list[int],
    target_max_episodes: int,
) -> tuple[int, pd.DataFrame]:
    """Pick the smallest gap (>= initial) whose collapse yields <= target
    episodes. Larger gaps merge more, so episode count falls as the gap grows."""
    ordered = sorted(set([g for g in candidate_gaps if g >= initial_gap] + [initial_gap]))
    chosen_gap, chosen_df = ordered[-1], collapse_to_episodes(df, ordered[-1])
    for gap in ordered:
        eps = collapse_to_episodes(df, gap)
        chosen_gap, chosen_df = gap, eps
        if len(eps) <= target_max_episodes:
            break
    return chosen_gap, chosen_df


def select_with_peak_window(
    episodes: pd.DataFrame, max_selected: int, window: int, overlap_mode: str
) -> pd.DataFrame:
    """Greedily keep top episodes, suppressing a candidate whose peak is within
    ``window`` steps of an already-selected one (``global``: any; ``sensor_overlap``:
    only if their sensor sets intersect). Assumes ``episodes`` is pre-ranked."""
    if episodes.empty or max_selected <= 0:
        return episodes.head(0).copy()
    selected_rows: list[dict] = []
    selected_peaks: list[int] = []
    selected_sets: list[set] = []
    for row in episodes.itertuples(index=False):
        candidate_peak = int(row.peak_time_idx)
        candidate_set = set(parse_sensor_ids(str(row.sensor_ids)))
        blocked = False
        for prev_peak, prev_set in zip(selected_peaks, selected_sets):
            if abs(candidate_peak - prev_peak) > window:
                continue
            if overlap_mode == "global" or candidate_set.intersection(prev_set):
                blocked = True
                break
        if blocked:
            continue
        selected_rows.append(row._asdict())
        selected_peaks.append(candidate_peak)
        selected_sets.append(candidate_set)
        if len(selected_rows) >= max_selected:
            break
    return pd.DataFrame(selected_rows)


def rank_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Rank by error magnitude with a light size bonus, then reindex episode_id."""
    episodes = episodes.copy()
    episodes["rank_score"] = episodes["peak_mean_abs_err"] * (
        1.0 + 0.15 * np.maximum(episodes["cluster_size"] - 2, 0)
    )
    episodes = episodes.sort_values(
        ["rank_score", "peak_max_abs_err", "n_frames"], ascending=False
    ).reset_index(drop=True)
    episodes["episode_id"] = np.arange(len(episodes), dtype=int)
    return episodes


def collapse(
    clusters: pd.DataFrame,
    *,
    episode_gap: int = 10,
    auto_increase_cutoff: bool = True,
    candidate_gaps: list[int] | None = None,
    target_max_episodes: int = 800,
    min_cluster_size: int = 2,
    min_peak_mean_abs_err: float = 0.0,
    max_selected: int = 120,
    peak_exclusion_window: int = 120,
    window_overlap_mode: str = "sensor_overlap",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full collapse: raw clusters -> (episodes_all, episodes_selected, summary)."""
    if clusters.empty:
        raise ValueError("Input clusters are empty; nothing to collapse.")
    clusters = clusters.copy()
    clusters["sensor_set"] = clusters["sensor_ids"].map(canonical_sensor_set)

    gaps = candidate_gaps if candidate_gaps is not None else [10, 20, 30, 60, 120]
    if auto_increase_cutoff:
        used_gap, episodes = choose_episode_gap(
            clusters, episode_gap, gaps, target_max_episodes
        )
    else:
        used_gap, episodes = episode_gap, collapse_to_episodes(clusters, episode_gap)

    episodes = episodes[
        (episodes["cluster_size"] >= min_cluster_size)
        & (episodes["peak_mean_abs_err"] >= min_peak_mean_abs_err)
    ]
    if episodes.empty:
        raise ValueError(
            "No episodes after filtering; lower --min_cluster_size / "
            "--min_peak_mean_abs_err."
        )
    episodes = rank_episodes(episodes)

    selected = select_with_peak_window(
        episodes,
        max_selected=max_selected,
        window=max(0, peak_exclusion_window),
        overlap_mode=window_overlap_mode,
    ).reset_index(drop=True)
    if not selected.empty:
        selected["episode_id"] = np.arange(len(selected), dtype=int)

    summary = {
        "input_rows": int(len(clusters)),
        "episodes_after_collapse": int(len(episodes)),
        "selected": int(len(selected)),
        "used_episode_gap": int(used_gap),
        "peak_exclusion_window": int(peak_exclusion_window),
        "window_overlap_mode": str(window_overlap_mode),
        "auto_increase_cutoff": bool(auto_increase_cutoff),
        "target_max_episodes": int(target_max_episodes),
        "min_cluster_size": int(min_cluster_size),
        "min_peak_mean_abs_err": float(min_peak_mean_abs_err),
    }
    return episodes, selected, summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collapse map_hot clusters into episodes (no plotting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=None, help="forecaster out_dir; used to derive default paths")
    p.add_argument(
        "--input_csv",
        default=None,
        help="clusters CSV (default: <out_dir>/downstream/map_freq_hot_clusters_all.csv)",
    )
    p.add_argument(
        "--results_dir",
        default=None,
        help="where to write episode CSVs (default: <out_dir>/downstream)",
    )
    p.add_argument("--episode_gap", type=int, default=10)
    p.add_argument("--no_auto_increase_cutoff", action="store_true")
    p.add_argument("--candidate_gaps", default="10,20,30,60,120")
    p.add_argument("--target_max_episodes", type=int, default=800)
    p.add_argument("--min_cluster_size", type=int, default=2)
    p.add_argument("--min_peak_mean_abs_err", type=float, default=0.0)
    p.add_argument("--max_selected", type=int, default=120)
    p.add_argument("--peak_exclusion_window", type=int, default=120)
    p.add_argument(
        "--window_overlap_mode",
        default="sensor_overlap",
        choices=["sensor_overlap", "global"],
    )
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    if args.input_csv:
        input_csv = Path(args.input_csv)
    elif args.out_dir:
        input_csv = Path(args.out_dir) / "downstream" / "map_freq_hot_clusters_all.csv"
    else:
        raise SystemExit("provide --input_csv or --out_dir")
    if not input_csv.exists():
        raise SystemExit(f"missing input clusters CSV: {input_csv}")

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else (Path(args.out_dir) / "downstream" if args.out_dir else input_csv.parent)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.read_csv(input_csv)
    episodes, selected, summary = collapse(
        clusters,
        episode_gap=args.episode_gap,
        auto_increase_cutoff=not args.no_auto_increase_cutoff,
        candidate_gaps=[int(x) for x in args.candidate_gaps.split(",") if x.strip()],
        target_max_episodes=args.target_max_episodes,
        min_cluster_size=args.min_cluster_size,
        min_peak_mean_abs_err=args.min_peak_mean_abs_err,
        max_selected=args.max_selected,
        peak_exclusion_window=args.peak_exclusion_window,
        window_overlap_mode=args.window_overlap_mode,
    )

    all_path = results_dir / "map_freq_hot_cluster_episodes_all.csv"
    sel_path = results_dir / "map_freq_hot_cluster_episodes_selected.csv"
    episodes.to_csv(all_path, index=False)
    selected.to_csv(sel_path, index=False)
    with open(results_dir / "episodes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[collapse] rows={summary['input_rows']} -> episodes="
        f"{summary['episodes_after_collapse']} (gap={summary['used_episode_gap']}), "
        f"selected={summary['selected']}"
    )
    print(f"[save]   {all_path}")
    print(f"[save]   {sel_path}")


if __name__ == "__main__":
    main()
