# Interatomic-potential domain loss

Machine-learning interatomic potentials declare their complete training objective
under `NeuralNetwork.Training.DomainLoss`. Model architecture no longer contains
loss weights or an interatomic-mode switch.

```json
"DomainLoss": {
    "enabled": true,
    "provider": "interatomic_potential",
    "terms": {
        "energy": {
            "enabled": true,
            "weight": 1.0,
            "target": "energy",
            "normalization": "structure"
        },
        "energy_per_atom": {
            "enabled": false,
            "weight": 1.0,
            "target": "energy",
            "normalization": "atom"
        },
        "forces": {
            "enabled": true,
            "weight": 10.0,
            "target": "forces",
            "prediction": "negative_energy_gradient"
        }
    }
}
```

Each active term may specify `loss`; otherwise it inherits
`Training.loss_function_type`. Forces are conservative by construction and are
computed as the negative gradient of the predicted total energy with respect to
atomic positions. At least one term must be enabled with a positive weight.

The provider validates target, normalization, and prediction semantics when the
model is created. Training and evaluation report only the active terms, in JSON
order.
