# Materials preprocessing utilities

HydraGNN normalizes material stresses to full symmetric `3 x 3` tensors in
eV/Å³ with tensile stress positive. Dataset adapters must state their source
unit and sign convention when calling `normalize_stress`; this prevents unit
or sign assumptions from being hidden in individual example loaders.

`validate_materials_sample` checks the common atomistic schema before scalable
serialization: positions, atomic numbers, forces, optional cell and stress,
finite values, consistent atom counts, and self-loop-free graph connectivity.
It raises a field-specific `ValueError` so distributed preprocessors can count
and report rejected records.

MPTrj and OMat24 are the first consumers. MPTrj converts VASP stress from kbar
with compression positive. OMat24 receives ASE stress already expressed in
eV/Å³ with tension positive.

## Diagnosing an unknown sign convention

Use `diagnose_stress_sign` when an energy evaluator is available for strained
versions of a source structure:

```python
from hydragnn.utils.materials import diagnose_stress_sign

diagnostic = diagnose_stress_sign(
    data.pos,
    data.cell,
    source_stress,  # symmetric 3 x 3, eV/Å³
    lambda positions, cell: calculator.energy(positions, cell),
)
print(diagnostic.inferred_source_sign)
print(diagnostic.tension_positive_rmse)
print(diagnostic.compression_positive_rmse)
```

The utility applies positive and negative perturbations for all six independent
symmetric strain components, estimates `dE/dstrain / volume` with central
finite differences, and compares that tensor with both the reported stress and
its negation. It returns `ambiguous` for configurations where the two errors are
too similar, as commonly happens near zero stress. It does not modify the
reported tensor; pass the inferred convention explicitly to `normalize_stress`.
