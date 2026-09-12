# PubChem Gaussian Hessian example

This example preprocesses a small subset of the VIBRANT PubChem dataset and
trains a HydraGNN model from the serialized graphs. Each distributed
`*.tar.zst` archive contains one nested tar archive per PubChem CID, so
preprocessing first expands the requested number of molecular records under
`dataset/raw/extracted`.

The parser aligns input-orientation coordinates, SCF energy, and Cartesian
forces from the Gaussian log, then retains the record whose coordinates match
the optimized `Structure.txt` geometry. Positions are converted from Angstrom
to Bohr; energies remain in Hartree and forces remain in Hartree/Bohr.

The isolated-atom calculations in `dataset/raw/atomization.tar.gz` use
UB3LYP/6-311++G(d,p) with the ground-state multiplicity of H, C, N, O, P, and
S. For each molecule, preprocessing computes the reference-shifted energy

$$
E_{\mathrm{form}} = E_{\mathrm{SCF}} - \sum_A n_A E_A^{\mathrm{atom}}.
$$

This is a formation energy relative to isolated gas-phase atoms, and its
negative is the atomization energy. It is not a standard-state formation
enthalpy. Each graph stores the raw `data.total_energy`, summed
`data.atomic_reference_energy`, `data.formation_energy`, and positive
`data.atomization_energy`. `data.energy` aliases `data.formation_energy` for
training. Because the atomic reference depends only on composition, forces and
Hessians are unchanged by this energy shift.

The analytical Hessian is available only for the optimized structure used by
the frequency calculation. Every retained graph therefore has a dense
`data.hessian` attribute of shape $3N\times3N$ in Hartree/Bohr$^2$.

For `pos` and forces shaped `(N, 3)`, PyTorch returns the force Jacobian with
axes `(output_atom, output_xyz, input_atom, input_xyz)`. Its element
`[a, alpha, b, beta]` is $\partial F_{a\alpha}/\partial R_{b\beta}$. Since
$F=-\partial E/\partial R$, the energy Hessian is the negative force Jacobian.
Gaussian and a contiguous PyTorch tensor both use atom-major Cartesian order,
`(0x, 0y, 0z, 1x, 1y, 1z, ...)`, so the negated Jacobian can be reshaped
directly to `(3N, 3N)`. Matrix element `[3*a + alpha, 3*b + beta]` then equals
$\partial^2 E/(\partial R_{a\alpha}\partial R_{b\beta})$.

The configuration trains a scalar additive formation-energy model. HydraGNN
differentiates that energy once for force loss and again for Hessian loss. Batch
size is one because dense Hessians vary with molecular size.

Each epoch's text log reports the energy, energy-per-atom, force, and (when
enabled) Hessian train, validation, and test losses using the same named format.

## Hyperparameter optimization

`pubchem_gaussian_hpo.py` follows the SC26 DeepHyper queued-worker pattern and
searches over the message-passing implementation, optional GPS global attention,
energy/force/Hessian weights, message-passing depth and width, attention type,
conditional multi-head count, and graph-transformer feed-forward depth and
width. GPS is used for the optional graph transformer because it supports both
`multihead` and `performer`; the equivariant local MPNN channels remain separate
from its global attention over invariant channels. `global_attn_heads` is used
only for `multihead`; Performer and transformer-disabled trials set it to one.

The HPO objective is the negative unweighted mean of the named energy, force,
and Hessian validation losses from the latest completed epoch. Keeping the
objective unweighted prevents trials with smaller loss weights from winning
solely because their aggregate loss has been rescaled.

First create the shared processed dataset, then submit the launcher after
adapting its allocation and environment setup to the target system:

```bash
srun -N1 -n1 python examples/pubchem_gaussian/train.py --preonly
sbatch examples/pubchem_gaussian/job-hpo-frontier.sh
```

Atomic numbers are categorical node inputs encoded by a learned 128-dimensional
embedding. The 118 categories cover atomic numbers 1 through 118; `min_value: 1`
maps hydrogen to embedding index zero without reserving a category for atomic
number zero.

Preprocess 100 molecules into pickle datasets:

```bash
python train.py --preonly --num-molecules 100
```

Train using the preprocessed data:

```bash
python train.py
```

Both commands support distributed execution through the same launcher and
environment used by the other HydraGNN examples. The raw archives, extracted
records, and generated pickle files are intentionally ignored by Git.
