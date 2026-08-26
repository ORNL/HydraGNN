# Equivariant all-to-all graph Transformer

## Scope

HydraGNN's `GPS` global-attention engine applies scalar attention to invariant
node channels. The `EquivariantTransformer` engine is an alternative
GraphGPS-style hybrid: every layer combines a local equivariant MPNN update
with an explicitly SE(3)-equivariant, all-to-all attention update.

This is a new HydraGNN architecture. It is not EquiformerV2, whose attention is
restricted to a local neighbor graph, and it is not AllScAIP, whose rotational
equivariance is learned rather than encoded with irreducible tensors.

The end-to-end model integration supports PaiNN, PNAEq, and MACE with
tensor-valued coupling. SchNet and DimeNet are supported only in the explicitly
acknowledged scalar-only mode described below. MACE applies global attention
after its tensor-valued hidden layers but not after its final scalar-only
readout layer, and consequently requires at least two convolution layers.

## Equivariance contract

Node features are represented as e3nn irreducible representations (irreps).
For every ordered atom pair `(i, j)` within the same graph, the global branch
uses the relative displacement `r_ij = r_j - r_i`, its radial expansion, and
spherical harmonics of its direction. It must never use absolute positions.

Queries and keys are equivariant linear projections of the node irreps. Their
contraction to degree zero produces invariant attention logits. Softmax is
applied over all source atoms `j` belonging to the target atom's graph.

Values are equivariant tensor-product messages coupling source-node irreps to
spherical harmonics of `r_ij`. Multiplication by invariant attention weights
and summation over source atoms preserve equivariance. The output projection,
normalization, residual path, and feed-forward network must also preserve the
declared irreps.

Consequently, under a rotation `R`, scalar outputs remain unchanged and every
degree-`l` output transforms with the corresponding representation `D_l(R)`.
Translation invariance follows from using only relative displacements.

## Local-model adapters

HydraGNN's current equivariant models do not share one tensor layout, so the
global engine must use explicit adapters instead of guessing from tensor
shapes.

- `PAINN` and `PNAEq` expose scalar channels plus vector channels shaped
  `[num_nodes, 3, channels]`. Their adapter maps these to
  `channels x 0e + channels x 1o` and restores the original layout afterward.
- `MACE` exposes flattened e3nn irreps. Its adapter receives the exact output
  irreps from each wrapped MACE layer and preserves all configured degrees.
- `SchNet` and `DimeNet` may be used in an explicitly acknowledged scalar-only
  mode. Their hidden node features map to `channels x 0e`; consequently, their
  local and global branches exchange invariant features only. This mode does
  not provide tensor-valued local/global feature exchange and must not be
  described as equivalent to the PaiNN/PNAEq or MACE integrations.
- `EGNN` uses atomic coordinates as its equivariant state rather than latent
  irreducible tensor features. It is initially unsupported; treating positions
  as ordinary vector channels would break translation invariance.
- Other scalar-only MPNNs remain unsupported until their feature and geometry
  contracts have been reviewed explicitly.

Adapters must validate dimensions and parity and fail with actionable errors.

## Complete-graph construction

All ordered non-self pairs are constructed independently for each graph in a
batch. Cross-graph attention is forbidden. Self-attention is handled by the
residual path rather than by zero-length geometric edges.

### Periodic convention

Periodic attention is opt-in and follows an explicit finite-supercell
convention. This is a collection of modeling choices rather than an incidental
implementation detail:

1. Atoms in the supplied cell are the only queries and the only outputs.
   Replicated atoms are keys and values only. Graph pooling therefore remains
   over central atoms and does not multiply an extensive prediction by the
   number of images.
2. The default replication count is one in each periodic lattice direction.
   For a fully periodic cell this produces a `3 x 3 x 3` source supercell.
   Non-periodic directions are never replicated. The count is configurable so
   users can measure convergence rather than assuming this default is exact.
3. Every source in the selected supercell participates in attention. There is
   no radius cutoff: introducing one would turn the global branch into another
   local neighborhood operation. The finite supercell itself is the explicit
   approximation to the infinite crystal.
4. Image positions are constructed as `r_j + n @ cell`, where `n` is an
   integer lattice shift. The resulting displacement is used directly. It is
   **never minimum-image wrapped**. Minimum-image wrapping would collapse
   different explicit images onto the same displacement, duplicate equivalent
   keys, and alter the softmax denominator.
5. One softmax per query and attention head spans all selected images in that
   query's graph. This preserves the Transformer defined by the non-periodic
   implementation, but it also means changing the replication extent changes
   both the source set and its normalization. Supercell convergence must be
   evaluated for the application.
6. Replicas are differentiable functions of the original positions and cell;
   they are not independent coordinates. Energy gradients therefore accumulate
   image contributions onto central atoms, and cell gradients remain available
   for a separately defined stress conversion.

The implementation deliberately does not silently wrap input coordinates into
a canonical cell. Such wrapping uses discontinuous integer choices at cell
boundaries and can obscure errors in prepared data. Callers must provide the
intended central-cell representation explicitly.

A finite explicit supercell cannot be exactly invariant to every equivalent
choice of unit-cell origin or basis. Rewrapping an atom can shift which boundary
images lie inside a finite image range. Exact invariance would require an
infinite image sum or a pair-centered/minimum-image convention; the latter is
explicitly rejected here. Increasing the replication count supplies a
systematic convergence check, not a proof of infinite-lattice equivalence.

The dense reference path materializes the complete edge set for correctness
tests. Production execution chunks target nodes and constructs only the pairs
needed by the current target chunk. Every target still attends to every source
in its graph in one softmax; chunks never split a target's source set. Thus
chunking bounds pair-message memory without approximating or truncating
attention. Dense/chunked output and gradient parity are regression-tested.

## Configuration

The engine is selected with:

```json
{
  "NeuralNetwork": {
    "Architecture": {
      "global_attn_engine": "EquivariantTransformer",
      "global_attn_heads": 4,
      "equivariant_attn_lmax": 1,
      "equivariant_attn_num_radial": 32,
      "equivariant_attn_chunk_size": 512,
      "equivariant_attn_coupling_mode": "parallel",
      "equivariant_attn_periodic": true,
      "equivariant_attn_periodic_replication": [1, 1, 1],
      "equivariant_attn_allow_scalar_only": false,
      "equivariant_attn_require_tensor_coupling": true
    }
  }
}
```

`equivariant_attn_lmax` controls the spherical-harmonic degrees available to
the value tensor product. In scalar-only mode, non-scalar harmonics cannot
couple a scalar input back to a scalar output, so increasing `lmax` does not
create tensor-valued latent channels. Chunking changes execution and memory
use, not numerical semantics.

`equivariant_attn_coupling_mode` controls how the local MPNN and global
Transformer are combined. The default, `"parallel"`, applies both branches to
the same input and adds their outputs in the shared irrep representation. This
matches GraphGPS's local/global organization while preserving equivariance.
`"sequential"` retains the earlier local-MPNN-then-global-Transformer flow for
experiments and compatibility with models trained using that architecture.

`equivariant_attn_periodic` defaults to `false`. When enabled, every sample
must provide `data.cell` and `data.pbc`; batched forms must contain one `3 x 3`
cell and one length-three PBC mask per graph. The replication option accepts one
nonnegative integer for all axes or a length-three list. Counts on axes whose
PBC flag is false are ignored. A value of zero selects only the central image
on that axis. Nonzero local `edge_shifts` are rejected unless periodic global
attention is explicitly enabled, preventing an accidental change of physical
semantics.

`equivariant_attn_allow_scalar_only` must be set to `true` for SchNet or
DimeNet, and `equivariant_attn_require_tensor_coupling` must be set to `false`.
If tensor coupling remains requested, model construction fails because these
MPNNs expose no non-scalar latent irreps. The Transformer layer independently
performs the same irrep-level validation so the restriction cannot be bypassed
by constructing it directly. HydraGNN emits a warning when the limited mode is
selected. For SchNet it also rejects coordinate-update mode
(`Architecture.equivariance=true`) because raw coordinates are not
translation-invariant latent tensor features.

## Verification coverage

The regression suite verifies:

1. rotation equivariance for every output irrep;
2. translation invariance;
3. permutation equivariance;
4. absence of cross-graph attention in mixed-size batches;
5. dense-versus-chunked numerical and gradient parity;
6. PaiNN/PNAEq and MACE adapter round trips;
7. forward and backward integration with each supported local MPNN;
8. invariant energy and equivariant energy-gradient forces; and
9. explicit-image construction without minimum-image wrapping;
10. central-query output cardinality, configurable image extent, and cell
    gradients; and
11. clear rejection of scalar-only misuse, EGNN, and periodic inputs without
    explicit opt-in or required metadata.

Periodic full-model forward/backward tests cover PaiNN, PNAEq, and MACE tensor
coupling as well as the explicitly restricted SchNet and DimeNet scalar-only
adapters. EGNN is not included because its equivariant state consists of
updated coordinates rather than translation-invariant latent irreps. Passing
those absolute coordinates through the current tensor adapter would violate
the global Transformer's translation-invariance contract; EGNN requires a
separately designed coordinate-state adapter before integration can be enabled.

All symmetry tests must use multiple random transformations and nontrivial
features; shape-only smoke tests are insufficient.
