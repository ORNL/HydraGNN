# Interatomic-potential training loss

Machine-learning interatomic potentials declare their complete objective under
`NeuralNetwork.Training.loss`. Architecture configuration contains no loss
weights or interatomic-mode switch.

```json
"loss": {
    "enabled": true,
    "provider": "interatomic_potential",
    "supervised": {
        "default_metric": "mse",
        "terms": [
            {"variable": "energy", "weight": 1.0, "normalization": "per_structure"},
            {"variable": "energy_per_atom", "weight": 1.0, "normalization": "per_atom"},
            {
                "variable": "forces", "weight": 10.0,
                "prediction": {"operator": "negative_gradient", "of": "energy", "with_respect_to": "positions"}
            }
        ]
    },
    "constraints": [],
    "constraint_optimizer": {"type": "fixed_penalty"}
}
```

Every positive-weight term is active. A term may specify `metric`; otherwise it
inherits `supervised.default_metric`. Forces are conservative by construction and are
computed as the negative gradient of the predicted total energy with respect to
atomic positions. At least one term must be enabled with a positive weight.

The provider validates variable, normalization, and prediction semantics when the
model is created. Training and evaluation report only the active terms, in JSON
order.

Each term is reported by name for train, validation, and test. See
[Named training-loss reporting](training_loss_reporting.md) for the `.raw`,
`.weight`, `.weighted`, and `total` fields written to `run.log`.
