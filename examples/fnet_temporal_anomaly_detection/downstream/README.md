# Downstream regional-anomaly detection

Post-hoc disturbance detection on the cov80 forecaster's residuals. **No
retraining** — everything here reads a forecaster `--out_dir` (produced by
`fnet_temporal_anomaly_detection.py`) and writes CSVs under `<out_dir>/downstream/`.

```
fnet_temporal_anomaly_detection.py  --out_dir OUT
        │  (preds/ys/masks, A_geo.npy, tvec.npy, sites.csv, downstream_meta.json)
        ▼
regional_detector.py  --out_dir OUT --detector map_hot   →  map_freq_hot_clusters_{all,top}.csv
        ▼
collapse_episodes.py  --out_dir OUT                       →  map_freq_hot_cluster_episodes_{all,selected}.csv
```

## Components

| File | Role |
|---|---|
| `downstream_io.py` | Adapter: maps the forecaster `out_dir` artifacts into `DownstreamData` (arrays `[S,H,N,F_out]` in physical units, raw `A_geo`, per-window timestamps, mask). The single bridge; the detectors never read the model or raw parquet. |
| `regional_detector.py` | Flags connected sensor clusters with high residual, per timestamp (`--detector strict\|map_hot`). |
| `collapse_episodes.py` | Collapses the many per-timestamp `map_hot` clusters into distinct episodes (framework only; no plotting). |

## Detectors (`--detector`)

| | `map_hot` (default) | `strict` |
|---|---|---|
| Activation | high `freq_dev` error only | high `freq_dev` **and** RoCoF error |
| Adjacency | raw weight `> --edge_threshold` | `>= --edge_quantile` of positive weights |
| Guards | — | drops clusters with no true RoCoF / no cross-sensor spread |
| Volume | many raw clusters → feed to `collapse_episodes` | ~tens of clusters, self-contained |
| Output CSV | `map_freq_hot_clusters_*.csv` | `clusters_*.csv` |

## Usage

```bash
# 1) detect (map_hot is the default; strict via --detector strict)
python downstream/regional_detector.py --out_dir OUT --split pred --detector map_hot

# 2) collapse the map_hot clusters into episodes
python downstream/collapse_episodes.py --out_dir OUT
```

`--split` is one of `val` / `test` / `pred` (the scored splits). Key detector
flags: `--color_max_quantile` / `--freq_quantile` / `--rocof_quantile` (activation),
`--edge_threshold` / `--edge_quantile` (adjacency), `--min_cluster_size`. Collapse
flags: `--episode_gap`, `--target_max_episodes`, `--peak_exclusion_window`,
`--window_overlap_mode`. Run either script with `-h` for the full list.

## What it inherits from the forecaster

- **Mask-aware residuals** — imputed / missing forecast targets (`masks_*.npy` = 0)
  are excluded from the residual, the activation quantiles, and the cluster stats.
- **Timestamps from `tvec`** — each window's time comes from the saved `tvec.npy`
  via the contiguous slice-then-window mapping (no parquet re-read).
- **RoCoF channel branch** — `strict` uses a real rocof output channel if the run
  produced one (`--out_features … 1 …`), else estimates it as the first difference
  of `freq_dev` (identical to the forecaster's own rocof).

## Scope

This is the detection + episode-collapse pipeline. The failure-vs-event
**discriminator** and the synthetic **fault-injection / scoring** harness are not
included here.
