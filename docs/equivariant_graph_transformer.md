# Equivariant all-to-all graph Transformer

## Scope

HydraGNN's `GPS` global-attention engine applies scalar attention to invariant
node channels. The proposed `EquivariantTransformer` engine is an alternative
GraphGPS-style hybrid: every layer combines a local equivariant MPNN update
with an explicitly SE(3)-equivariant, all-to-all attention update.

This is a new HydraGNN architecture. It is not EquiformerV2, whose attention is
restricted to a local neighbor graph, and it is not AllScAIP, whose rotational
equivariance is learned rather than encoded with irreducible tensors.

The current end-to-end model integration is enabled for PaiNN and PNAEq.
SchNet and DimeNet have representation adapters, but model construction keeps
them disabled until each path has its own full-model invariance and gradient
tests. This distinction prevents an adapter-level test from being mistaken for
a supported training configuration.

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
- `MACE` exposes flattened e3nn irreps. Its adapter obtains the exact irreps
  from the stack configuration and preserves all configured degrees.
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

The first implementation targets finite, non-periodic systems. Periodic
all-to-all attention requires an explicit image convention or a lattice-aware
long-range formulation; silently applying a minimum-image displacement is not
equivalent to an infinite periodic interaction.

The reference implementation may materialize the complete edge set for
correctness tests. The production path must support query chunking so memory
does not require storing every pairwise message simultaneously.

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

`equivariant_attn_allow_scalar_only` must be set to `true` for SchNet or
DimeNet, and `equivariant_attn_require_tensor_coupling` must be set to `false`.
If tensor coupling remains requested, model construction fails because these
MPNNs expose no non-scalar latent irreps. The Transformer layer independently
performs the same irrep-level validation so the restriction cannot be bypassed
by constructing it directly. HydraGNN emits a warning when the limited mode is
selected. For SchNet it also rejects coordinate-update mode
(`Architecture.equivariance=true`) because raw coordinates are not
translation-invariant latent tensor features.

## Required tests

The engine is not complete until tests cover:

1. rotation equivariance for every output irrep;
2. translation invariance;
3. permutation equivariance;
4. absence of cross-graph attention in mixed-size batches;
5. dense-versus-chunked numerical and gradient parity;
6. PaiNN/PNAEq and MACE adapter round trips;
7. forward and backward integration with each supported local MPNN;
8. invariant energy and equivariant energy-gradient forces; and
9. clear rejection of scalar-only, EGNN, and unsupported periodic inputs.

All symmetry tests must use multiple random transformations and nontrivial
features; shape-only smoke tests are insufficient.
