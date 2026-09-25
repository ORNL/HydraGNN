# Named training-loss reporting

HydraGNN writes one `LossComponents` record per epoch and dataset split to
`run.log`. Each record contains the final objective and named diagnostics for
every supervised quantity and active constraint:

```text
LossComponents epoch=04  split=validation  total=0.85202000  \
supervised.energy.raw=0.01200000  supervised.energy.weight=1  \
supervised.energy.weighted=0.01200000  constraints.voltage_bound.raw=0.00300000  \
constraints.voltage_bound.weight=0.01  constraints.voltage_bound.weighted=0.00002000
```

- `raw` is the metric or residual before task weights, constraint scales, or
  augmented-Lagrangian transformations.
- `weight` is the effective task weight or constraint scale.
- `weighted` is the diagnostic after weighting. For constraints it is the
  actual fixed-penalty or augmented-Lagrangian contribution, including the
  curriculum factor.
- `total` is the complete objective used by that split.

Train, validation, and test values are accumulated independently and reduced
across distributed ranks. Values are averaged with the same per-graph sampling
convention as the existing epoch losses.

For a head that predicts several quantities, name its columns with
`Variables.outputs[].components`:

```json
{
    "name": "bus_va_vm",
    "level": "node",
    "node_type": "bus",
    "dim": 2,
    "components": ["bus_voltage_angle", "bus_voltage_magnitude"]
}
```

Without explicit component names, HydraGNN reports `name[0]`, `name[1]`, and so
on. The original per-head `Tasks ... Loss` lines remain available, but
`LossComponents` is the unambiguous interface for machine-readable diagnostics.
