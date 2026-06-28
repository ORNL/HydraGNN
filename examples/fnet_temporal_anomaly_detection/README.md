# FNET temporal anomaly detection (cov80 T-GCN forecaster)

Multi-feature spatiotemporal forecasting on FNET PMU data, built on HydraGNN's
`TemporalGCN`. This is the HydraGNN port of the standalone T-GCN "cov80"
forecaster: it tolerates missing PMU samples by **keeping and masking** gaps
(rather than dropping them), trains with a **masked loss** so imputed targets do
not contribute, and reports **masked metrics**.

## Pipeline

1. Load one day of FNET parquet (`[FDRID]-[device]-[date].parquet`) + `FDRLocation.xlsx`.
2. Per-device dynamic features: `freq_dev, rocof, angle_delta, volt_dev`.
3. **Keep-and-mask alignment** onto a uniform `dt` grid: gaps are kept as NaN and
   recorded in `observed_mask`; devices below `--min_device_coverage` and timesteps
   below `--min_step_coverage` are dropped.
4. **Impute** the remaining gaps (interpolate → ffill/bfill → train-only median).
5. **Train-only feature scaling** over observed values (`--scaling`).
6. Geographic k-NN graph (`exp(-d/σ)` haversine) with edge weights.
7. Sliding-window `Data` objects:
   - `x_seq = [N, Tin, F_dyn + 1 (observed-mask) + F_static]`
   - `y     = [N, H * len(out_features)]`
   - `observed_mask = [N, H * len(out_features)]` (forecast-window loss mask)
8. **Slice-then-window 4-way split** (train/val/test/pred) — each segment is
   windowed independently so no window straddles a boundary (no leakage).
9. Train `TemporalGCN`; score val/test/pred with masked MSE/MAE/RMSE/R²/MAPE.

## Usage

```bash
# Pre-process + cache only
python fnet_temporal_anomaly_detection.py --preonly \
    --data_root <grid-data> --date <date>

# Train from an existing cache
python fnet_temporal_anomaly_detection.py --date <date>

# End to end (preprocess -> cache -> train)
python fnet_temporal_anomaly_detection.py --do_all \
    --data_root <grid-data> --date <date> --num_epoch 30
```

> **Regenerate the cache after any preprocessing change.** The cache stores the
> windowed `Data` objects (mask channel, loss mask, 4-way split, scaling). An old
> cache will silently train on stale tensors — delete
> `dataset/fnet_<date>.pickle/` + `dataset/fnet_<date>_meta.pkl` and re-run
> `--preonly`/`--do_all`.

## cov80 flags

| Flag | Default | Meaning |
|---|---|---|
| `--min_device_coverage` | 0.5 | Drop a device covering less than this fraction of the grid |
| `--min_step_coverage` | 0.8 | Keep a timestep only if ≥ this fraction of nodes are observed |
| `--short_gap_steps` | 10 | Max gap (steps) filled by linear interpolation |
| `--medium_gap_steps` | 300 | Max gap (steps) filled by ffill/bfill |
| `--scaling` | `global` | `none` / `per_node` / `global` / `robust` (train-only, observed-only) |
| `--scaled_clip_value` | 10.0 | For `--scaling robust`: clip to ± this (≤0 disables) |
| `--out_features` | `0 2 3` | Output channels into `[0=freq_dev,1=rocof,2=angle_delta,3=volt_dev]`; use `0 3` for the 2-output parity set |
| `--train_frac/--val_frac/--test_frac/--pred_frac` | 0.8/0.05/0.05/0.1 | 4-way time split |

Model knobs are in `fnet_temporal_anomaly_detection.json`; `loss_function_type`
must be `mse` (the masked-loss path is MSE-specific).

## Outputs (`--out_dir`, default `outputs_fnet_temporal/`)

- `metrics.json` — masked MSE/MAE/RMSE/R²/MAPE per split, in scaled + original units
- `preds_{val,test,pred}.npy`, `ys_*.npy`, `masks_*.npy` — `[W, N, H, F_out]`
- `feat_mean/feat_std/out_idx.npy`, `X.npy`, `tvec.npy`, `A_hat.npy`, `sites.csv`

## Notes

- The masked loss requires the `mask=` support in `Base.loss`;
  without it the run still trains, just unmasked.
