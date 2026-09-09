# PubChem Gaussian force and Hessian example

This example trains a scalar molecular potential, `E_theta(R)`, on a small
subset of the VIBRANT PubChem Gaussian data. Forces and the complete Cartesian
Hessian are not independent output heads. HydraGNN obtains them by automatic
differentiation: `F = -dE_theta/dR` and `H = -dF/dR = d²E_theta/dR²`.

The energy and force labels are read from the same optimization-step record in
`Opt_Trj.txt`; its coordinates must match the optimized `Structure.txt`
geometry. The scalar energy is stored as `data.energy` and participates in the
loss together with the force and Hessian labels.

`Structure.txt` positions are converted from Angstrom to Bohr. Gaussian forces
from the matching optimized `Opt_Trj.txt` geometry remain in Hartree/Bohr, and
`Hessian.txt` remains in Hartree/Bohr². The configured cutoff is 11.33835675
Bohr, equivalent to 6 Angstrom.

Place the distributed `*.tar.zst` archives in `dataset/raw/data`, then
preprocess a small subset:

```bash
python train.py --preonly --num-molecules 10
```

Train using the preprocessed data:

```bash
python train.py
```

Hessian-loss training differentiates through second coordinate derivatives and
is expensive. This correctness-first implementation requires batch size 1,
FP32, and no graph parallelism or FSDP. Start with small molecules and subsets.
Malformed records and geometry or dimension mismatches are skipped with a
CID-specific warning. Raw and processed dataset files are ignored by Git.
