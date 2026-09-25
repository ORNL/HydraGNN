# OPF augmented-Lagrangian training

`OPFDomainLoss` enforces the enabled OPF inequality constraints with

\[
L_{AL} = \sum_i \lambda_i c_i + \frac{\rho}{2}c_i^2,
\qquad
\lambda_i \leftarrow \max(0, \lambda_i + \rho \bar c_i),
\]

where `c_i` is the mean positive violation for one constraint family. The
implemented families are bus-voltage bounds, AC-line and transformer angle
bounds, and AC-line and transformer apparent-power limits.

Each `*_weight` is a physical scaling factor applied to its residual before the
augmented-Lagrangian calculation. A zero weight disables that family. Duals are
updated once per completed training epoch; validation and test batches never
modify them. The duals, `rho`, and the preceding residual norm are registered
PyTorch buffers and therefore survive normal model checkpoints.

Configuration keys under `NeuralNetwork.Training.DomainLoss`:

- `rho`: initial quadratic coefficient; must be positive.
- `rho_growth`: multiplier used when constraint reduction stalls; at least 1.
- `rho_max`: upper bound for `rho`.
- `constraint_reduction`: required ratio of the new residual norm to the old
  norm. If the new norm exceeds this threshold, `rho` grows.
- `dual_update_interval`: number of completed epochs between dual updates.
- `warmup_epochs` and `ramp_epochs`: delay and then phase in the augmented term.

The canonical example is `opf_solution_heterogeneous.json`. Command-line
overrides are available as `--domain_loss_rho`, `--domain_loss_rho_growth`,
`--domain_loss_rho_max`, and `--domain_loss_constraint_reduction`.
The dual-update cadence can be overridden with
`--domain_loss_dual_update_interval`.
