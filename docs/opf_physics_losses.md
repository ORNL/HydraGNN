# OPF physics-informed losses

`NeuralNetwork.Training.DomainLoss` adds or monitors OPF feasibility penalties
for bus-level predictions. The supervised target convention is
`[Va, Vm]`: voltage angle followed by voltage magnitude.

## Modes

| `mode` | Behavior |
| --- | --- |
| `static` | Adds weighted, EMA-normalized quadratic penalties. This is the default. |
| `augmented_lagrangian` or `al` | Adds augmented-Lagrangian terms and updates constraint multipliers during training. |
| `monitor` or `monitor_only` | Computes and logs violations but adds no physics loss. |

Set `enabled: false` to disable both penalties and monitoring.

## Configuration reference

| Key | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `false` | Enable domain-loss evaluation. |
| `mode` | `static` | Select the behavior above. |
| `voltage_bound_weight` | `0` | Penalize predicted `Vm` outside bus `[v_min, v_max]`. |
| `angle_diff_weight` | `0` | Penalize branch angle differences outside configured limits. |
| `line_flow_weight` | `0` | Penalize DC-approximate thermal-limit violations. |
| `ac_line_flow_weight` | `0` | Penalize AC apparent-flow violations. |
| `include_transformer_ac_flow` | `false` | Include transformer apparent-flow constraints. |
| `line_flow_slack` | `1e-4` | Tolerance above `rate_a` for DC-linearization residuals. |
| `voltage_bound_feature_indices` | unset | Indices of `v_min` and `v_max` in bus inputs. |
| `voltage_output_index` | `1` | `Vm` column in the bus prediction. |
| `va_output_index` | `0` | `Va` column in the bus prediction. |
| `circular_angle_loss` | `true` | Use wrapped angular residuals for `Va`. |
| `ema_momentum` | `0.1` | Update coefficient for per-term scale tracking. |
| `warmup_epochs` | `0` | Initial epochs with zero physics contribution. |
| `ramp_epochs` | `0` | Epochs used to linearly reach full weight. |
| `al_rho` | `1e-3` | Augmented-Lagrangian quadratic coefficient. |
| `al_mu_max` | `100` | Maximum learned constraint multiplier. |

## EMA normalization

For each penalty `L_k`, the implementation tracks an exponential moving scale
and divides the current penalty by that detached scale before applying its
configured weight. This makes weights more comparable across quantities with
different units and raw magnitudes. It does not normalize data or predictions.
Raw violations are logged separately and should be used to assess physical
quality.

The EMA state updates only during training. In distributed execution, the
domain-loss wrapper uses the model's batched data and training lifecycle; do
not infer feasibility from the normalized loss alone.

## Curriculum and augmented Lagrangian

During `warmup_epochs`, the physics contribution is zero. Over
`ramp_epochs`, it increases linearly to full strength. In augmented-Lagrangian
mode, multipliers update only on the training split; validation and test calls
measure constraints without changing them.

## Geometry and units

Angles are radians and are compared with wrapped differences. Power-system
features and limits must use the same per-unit convention as the raw OPF data.
The DC flow term is an approximation; `line_flow_slack` prevents small
linearization residuals in an AC-feasible solution from being penalized.

The CLI exposes overrides such as `--enable_domain_loss`,
`--domain_loss_voltage_bound_weight`, `--domain_loss_angle_diff_weight`,
`--domain_loss_line_flow_weight`, `--domain_loss_line_flow_slack`,
`--domain_loss_ema_momentum`, `--domain_loss_warmup_epochs`, and
`--domain_loss_ramp_epochs`.

