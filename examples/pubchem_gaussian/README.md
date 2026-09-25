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

The configuration trains a multitask model whose first head is the scalar,
additive formation energy. HydraGNN differentiates that energy once for force
loss and again for Hessian loss. Total molecular charge and spin multiplicity
are graph-level conditioning inputs, rather than prediction targets.

The auxiliary labels are per-atom Mulliken charges, dipole magnitude, sorted
eigenvalues of the traceless quadrupole and polarizability tensors, HOMO/LUMO
energies and their gap, rotational constants, and thermochemistry. Magnitudes
and tensor eigenvalues are used instead of Cartesian components so invariant
scalar heads do not learn an orientation-dependent target. Frequencies and
normal modes should instead be derived from the predicted Hessian; IR and Raman
intensities require dipole and polarizability derivatives, respectively, and
INS intensities require the normal modes plus neutron-scattering weights. None
of these spectra is trained as an independent unconstrained head. Labels are
read only from the final frequency calculation; records missing any configured
label are skipped with a warning. Batch size is one because dense Hessians vary
with molecular size.

Each epoch's text log reports energy, energy-per-atom, force, Hessian, and all
configured auxiliary train, validation, and test losses using the same named
format.

## Hyperparameter optimization

`pubchem_gaussian_hpo.py` follows the SC26 DeepHyper queued-worker pattern and
searches over the message-passing implementation, the optional equivariant
all-to-all graph Transformer, energy/force/Hessian weights, message-passing
depth and width, conditional attention-head count, and equivariant feed-forward
depth and irrep multiplicity. The engine is `EquivariantTransformer`, which
uses exact equivariant multi-head softmax attention. It does not use GPS or
Performer because HydraGNN has no equivariant Performer implementation.
The default MPNN set is `EGNN`, `SchNet`, `DimeNet`, `MACE`, `PAINN`, `PNAEq`,
`AllScAIP`, and `UMA`. AllScAIP and UMA are monolithic transformer backbones,
and EGNN has no compatible irrep adapter, so those three types do not receive
an additional `EquivariantTransformer`; its sampled parameters are
conditionally ignored for those model types.
MACE and UMA trials fix the tensor order at $\ell_{\max}=2$. MACE sets
`max_ell`, `node_max_ell`, and the optional transformer's
`equivariant_attn_lmax` to two; UMA sets `max_ell=2` and `uma_mmax=2`.

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

The Frontier launcher defaults to the shared PubChem Gaussian ADIOS2 dataset.
Override `PUBCHEM_DATASET` to select another cache, `HPO_MPNN_TYPES` to select
model families, and `HPO_MAX_EVALS` to set the search budget. For example, a
MACE/UMA-only search uses:

```bash
HPO_MPNN_TYPES=MACE,UMA HPO_MAX_EVALS=100 \
	sbatch examples/pubchem_gaussian/job-hpo-frontier.sh
```

The spherical-harmonic order is fixed at $\ell_{\max}=2$ for both families; it
is not sampled as a hyperparameter.

Two directly comparable campaigns can be submitted from the same search space
and dataset:

```bash
HPO_CAMPAIGN=primary sbatch examples/pubchem_gaussian/job-hpo-frontier.sh
HPO_CAMPAIGN=multitask sbatch examples/pubchem_gaussian/job-hpo-frontier.sh
```

The `primary` campaign exposes only the scalar energy output head; forces and
Hessians are still obtained by differentiating that energy. The `multitask`
campaign additionally trains the Mulliken-charge, dipole, quadrupole,
polarizability, frontier-orbital, rotational-constant, and thermochemistry
heads. Both campaigns use exactly the same HPO objective: the mean validation
loss over energy, forces, and Hessian. Auxiliary losses therefore affect the
learned representation but never directly affect trial selection. Logs and
DeepHyper search state are written to campaign-specific directories.

For the three-million-sample search, use the resumable successive-halving
driver and its checked-in stage schedule:

```bash
sbatch examples/pubchem_gaussian/job-multistage-hpo-frontier.sh
```

By default the job first preprocesses up to 3,000,000 molecules across 56 CPU
MPI ranks per allocated Frontier node, one per physical CPU core. Outer
archives are divided among ranks, and completed molecule directories are
installed atomically so extraction can resume after a wall-time interruption.
Set `PREPROCESS_TASKS_PER_NODE` to override the CPU rank count, or set
`PREPROCESS_DATASET=0` when resuming with an already complete shared cache. The
Frontier launcher defaults to the ADIOS2 backend. Set `DATASET_FORMAT=pickle`
to use the pickle backend instead. ADIOS2 reads can additionally use DDStore with
`USE_DDSTORE=1` and an optional `DDSTORE_WIDTH`, or node-local shared memory
with `USE_SHMEM=1`; DDStore and shared memory are mutually exclusive.

`pubchem_hpo_stages.json` screens 512 balanced candidates on 50,000 training
samples, then promotes 128, 32, and 8 candidates through 300,000, 1,000,000,
and 3,000,000 requested samples. The final stage retains three configurations.
All stages reuse one processed dataset and deterministic nested subsets; graph
records are not copied per trial. Validation and test subsets remain fixed at
the sizes in the schedule.

The screen-stage median of each raw validation metric defines fixed scales.
The mean normalized energy, force, and Hessian loss identifies the primary
anchor model. A candidate enters the comparable cohort only when each of its
three primary losses is within the stage's `primary_tolerance` of that anchor;
the default schedule uses 2% initially and 1% in the last two stages. Auxiliary
losses rank models only within this cohort and can never compensate for a model
outside all three primary error gates. Remaining promotion slots are filled by
primary score. The CLI `--primary-tolerance` supplies the default for schedules
that do not set it. Each trial writes its configuration, log, and result
separately. Existing result files are reused on restart, and every stage writes
a CSV ranking with primary score, auxiliary score, and comparability status;
`finalists.json` contains the last promoted configurations.

The launcher defaults to 16 allocated nodes, four nodes per trial, and eight DDP
ranks (one per GPU) on every node. It therefore trains four configurations
concurrently, each with 32 DDP ranks. Change `NNODES_PER_TRIAL` to control the
distributed-training scale; the total allocation must be divisible by it.
`TASKS_PER_NODE` controls participating GPUs per node. When running outside
Slurm, use one process unless distributed process launch is managed externally.

Atomic numbers are categorical node inputs encoded by a learned 128-dimensional
embedding. The 118 categories cover atomic numbers 1 through 118; `min_value: 1`
maps hydrogen to embedding index zero without reserving a category for atomic
number zero.

Preprocess 100 molecules into the default ADIOS2 dataset:

```bash
python train.py --preonly --num-molecules 100
```

Train using the preprocessed data:

```bash
python train.py
```

Use `--pickle` on both commands to select the pickle backend. The mutually
exclusive `--adios` and `--pickle` flags follow the convention used by the
other HydraGNN dataset examples; ADIOS2 is the default. Training also accepts
`--ddstore`, `--ddstore-width`, and `--shmem` for the standard ADIOS2 caching
options.

Both commands support distributed execution through the same launcher and
environment used by the other HydraGNN examples. The raw archives, extracted
records, and generated ADIOS2 or pickle files are intentionally ignored by Git.
