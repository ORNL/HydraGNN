# OPF constraint optimization

The `optimal_power_flow` training-loss provider enforces enabled OPF inequality
constraints with

\[
L_{AL} = \sum_i \lambda_i c_i + \frac{\rho}{2}c_i^2,
\qquad
\lambda_i \leftarrow \max(0, \lambda_i + \rho \bar c_i),
\]

where `c_i` is the mean positive violation for one constraint family. The
implemented families are bus-voltage bounds, AC-line and transformer angle
bounds, and AC-line and transformer apparent-power limits.

Each constraint `scale` is a physical scaling factor applied to its residual before the
augmented-Lagrangian calculation. A zero scale disables that family. Duals are
updated once per completed training epoch; validation and test batches never
modify them. The duals, `rho`, and the preceding residual norm are registered
PyTorch buffers and therefore survive normal model checkpoints.

Configuration lives under `NeuralNetwork.Training.loss`. The `supervised`
section defines the data loss, `constraints` defines named residuals, and
`constraint_optimizer` applies one of two generic strategies:

- `fixed_penalty`: stateless `scale * residual^2`.
- `augmented_lagrangian`: stateful dual ascent using the equation above.

The complete layout is:

```json
"loss": {
    "enabled": true,
    "provider": "optimal_power_flow",
    "supervised": {
        "default_metric": "mse",
        "terms": [{"variable": "bus_va_vm", "weight": 1.0}]
    },
    "constraints": [
        {
            "name": "voltage_limits",
            "operator": "bounded",
            "value": "bus_voltage_magnitude",
            "lower": "bus_vmin",
            "upper": "bus_vmax",
            "scale": 0.01
        },
        {
            "name": "angle_limits",
            "operator": "edge_difference_bounded",
            "value": "bus_voltage_angle",
            "relations": ["ac_line", "transformer"],
            "lower": "angle_minimum",
            "upper": "angle_maximum",
            "scale": 0.001
        },
        {
            "name": "thermal_limits",
            "operator": "ac_thermal_limit",
            "relations": ["ac_line", "transformer"],
            "scale": 0.001,
            "slack": 0.0001
        }
    ],
    "constraint_optimizer": {
        "type": "augmented_lagrangian",
        "rho": 1.0,
        "rho_growth": 2.0,
        "rho_max": 10000.0,
        "required_reduction": 0.9,
        "update_every": 1,
        "warmup_epochs": 3,
        "ramp_epochs": 3
    }
}
```

For `augmented_lagrangian`, the options are:

- `rho`: initial quadratic coefficient; must be positive.
- `rho_growth`: multiplier used when constraint reduction stalls; at least 1.
- `rho_max`: upper bound for `rho`.
- `required_reduction`: required ratio of the new residual norm to the old
  norm. If the new norm exceeds this threshold, `rho` grows.
- `update_every`: number of completed epochs between dual updates.
- `warmup_epochs` and `ramp_epochs`: delay and then phase in the augmented term.

Constraint fields such as `value`, `lower`, and `upper` refer to names in
`Variables`. `Variables.edge_attributes` names every ordered relation column,
so feature indices are not exposed in loss configuration.

The canonical example is `opf_solution_heterogeneous.json`. Command-line
overrides include `--constraint_voltage_scale`, `--constraint_thermal_slack`,
`--constraint_optimizer_rho`, and `--constraint_optimizer_update_every`.
Raw and weighted values for every supervised property and constraint are written
separately for train, validation, and test as documented in
[Named training-loss reporting](../../docs/training_loss_reporting.md).
