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
