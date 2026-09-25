# OPF constraint optimization

`OPFDomainLoss` enforces the enabled OPF inequality constraints with

\[
L_{AL} = \sum_i \lambda_i c_i + \frac{\rho}{2}c_i^2,
\qquad
\lambda_i \leftarrow \max(0, \lambda_i + \rho \bar c_i),
\]

where `c_i` is the mean positive violation for one constraint family. The
implemented families are bus-voltage bounds, AC-line and transformer angle
bounds, and AC-line and transformer apparent-power limits.

Each constraint `scale` is a physical scaling factor applied to its residual before the
augmented-Lagrangian calculation. A zero weight disables that family. Duals are
updated once per completed training epoch; validation and test batches never
modify them. The duals, `rho`, and the preceding residual norm are registered
PyTorch buffers and therefore survive normal model checkpoints.

Configuration lives under `NeuralNetwork.Training.loss`. The `supervised`
section defines the data loss, `constraints` defines named residuals, and
`constraint_optimizer` applies one of two generic strategies:

- `fixed_penalty`: stateless `scale * residual^2`.
- `augmented_lagrangian`: stateful dual ascent using the equation above.

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
