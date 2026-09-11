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

## Checking stress against reference energy–strain derivatives

Use `check_stress_against_energy_strain` when the original reference method or
labeled strained energies are available for a source structure:

```python
from hydragnn.utils.materials import check_stress_against_energy_strain

check = check_stress_against_energy_strain(
    data.pos,
    data.cell,
    source_stress,  # symmetric 3 x 3, eV/Å³
    reference_energy_fn=lambda positions, cell: reference_calculator.energy(
        positions, cell
    ),
)
print(check.preferred_sign_convention)
print(check.tension_positive_rmse)
print(check.compression_positive_rmse)
```

The utility applies positive and negative perturbations for all six independent
symmetric strain components, estimates `dE/dstrain / volume` with central
finite differences, and compares that tensor with both the reported stress and
its negation. It returns `ambiguous` for configurations where the two errors are
too similar, as commonly happens near zero stress. It does not modify the
reported tensor.

The energy callback is deliberately named `reference_energy_fn`: use the same
first-principles method that produced the labels, actual labeled strained
energies, or another independent and sufficiently accurate calculator. Using
the learned model under evaluation cannot independently establish the label
convention; it checks only that model's energy–stress consistency, and energy
errors can reverse the numerical derivative. The preferred convention is
evidence for one structure rather than proof of dataset-wide metadata. Check
multiple structures with appreciable stress and multiple strain steps before
changing a dataset conversion, then pass the verified convention explicitly to
`normalize_stress`.
