# Case-4661 AC/DC summary and RPE experiments

This suite compares five models with the same four-layer HeteroSAGE local
branch and bus-only, four-head GPS attention branch. Only the positional input
or attention bias changes.

| Array index | Run name | Active structural signal |
|---:|---|---|
| 0 | `case4661_bus_attention_control` | None |
| 1 | `case4661_dc_summary` | Five diagonal-free effective-resistance statistics |
| 2 | `case4661_ac_summary` | Ten diagonal-free effective-impedance statistics: five each for real and imaginary parts |
| 3 | `case4661_effective_resistance_rpe` | Raw effective resistance passed to a per-head bias MLP |
| 4 | `case4661_effective_impedance_rpe` | Raw real and imaginary effective impedance passed to a per-head bias MLP |

The statistics are ordered `[min, max, std, median, mean]`. The diagonal is
removed before every summary reduction. Direct RPE retains ordinary
self-attention but produces zero structural bias for self-pairs.

Effective impedance is computed from the complete complex AC `Ybus`, including
line resistance/reactance and charging, transformer taps and phase shifts, and
bus shunts. With `Zbus = pinv(Ybus)`, the pairwise value is
`Zeff[i,j] = Zbus[i,i] + Zbus[j,j] - Zbus[i,j] - Zbus[j,i]`.

## Shared preprocessing

All configurations declare the same preprocessing superset and therefore use
one HDF5 dataset:

```bash
sbatch job-frontier-case4661-acdc-rpe-preonly.sh
```

The output is `dataset/case4661_acdc_rpe.h5`. Node summaries are stored in each
sample. Dense pairwise matrices are stored once in `dataset/spectral_pe_cache`;
samples contain only the corresponding cache path so an O(N^2) tensor is not
duplicated for every operating point.

## Training and resuming

After preprocessing succeeds, submit all five fresh runs with:

```bash
sbatch job-frontier-case4661-acdc-rpe-train.sh
```

If a two-hour allocation ends before training completes, resume all five with:

```bash
sbatch job-frontier-case4661-acdc-rpe-resume.sh
```

Each array task requests eight Frontier nodes, uses batch size one, and writes
to its own existing TensorBoard run directory when resumed.
