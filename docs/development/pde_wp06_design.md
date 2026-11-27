# PDE extension: WP06 design (FNO, GINO, GSNO)

Status: implemented, in review (PR #585). Base: `main` at `c32cae8c` (WP04, PR #584).

| Item | Spec acceptance criterion | Where |
|---|---|---|
| FNO | Complex spectral weights, FFT modes, padding, pointwise paths, 1D/2D/3D | `tests/test_fno.py`, `examples/pde/verify_wp06.py`, `examples/pde/verify_neuralop_upstream.py`, `poisson_fno.json`, §5 |
| GINO | Non-matching query grids; 3D geometry test; comparison with a spectral reference; memory profiling | `tests/test_gino.py`, `verify_wp06.py`, `verify_neuralop_upstream.py`, `poisson_gino.json`, §5 |
| GSNO | Eigenbasis normalization, degenerate eigenspaces, temporal modes, reconstruction | `tests/test_gsno.py`, `verify_wp06.py`, `poisson_gsno.json`, §5 |

"Match upstream settings before extending": FNO and GINO are ports of
neuraloperator `a6648eeeb60fe0837f2d7d021517a945a159a0c8` (neuralop 2.0.0,
MIT, `LICENSES/NEURALOP-MIT.txt`), with neuralop's defaults, and match it
numerically (§5). GSNO has no public reference code. It follows the paper's
layer and defaults. Its extensions are off by default. This WP does not
substitute spherical GSNO or the spectral generator SGNO.

## 1. Components

| File | Content |
|---|---|
| `hydragnn/models/fno.py` | `SpectralConv`, `DomainPadding`, `FNOBlocks`, `ChannelMLP`, `FNO(OperatorBase)`, registered as `"FNO"` (output level `node`). |
| `hydragnn/models/gino.py` | `SinusoidalEmbedding`, `radius_neighbors`, `GNOBlock`, `GINO(OperatorBase)`, registered as `"GINO"` (output levels `node`, `query`). |
| `hydragnn/models/gsno.py` | `GraphSpectralConv`, `GSNO(OperatorBase)`, registered as `"GSNO"` (output level `node`). |
| `hydragnn/data/meshes/graph_spectral.py` | `normalized_laplacian`, `graph_spectral_basis`, `attach_graph_spectral_basis` (offline eigenbasis, sign fixing, degenerate groups). |
| `hydragnn/model_settings.py` | `FNOSettings` (`fno_*`), `GINOSettings` (`gino_*`), `GSNOSettings` (`gsno_*`). |
| `examples/pde/poisson_manufactured.py` | Adds `grid_shape` for FNO and the eigenbasis for GSNO, chosen by `mpnn_type`. |
| `examples/pde/poisson_{fno,gino,gsno}.json` | Presets. |
| `examples/pde/verify_wp06.py` | FP64 property checks, 3D GINO test against a spectral reference, memory/time profile, eigensolve timing. |
| `examples/pde/verify_neuralop_upstream.py` | Forward parity with neuralop (same weights, same inputs). |

None of the three models supports global attention or `pe_dim`; the registry
rejects them.

## 2. FNO

Port of the dense, real-input, post-activation path of `neuralop.models.FNO`:

* grid embedding: `linspace(0, 1, res + 1)[:-1]` per dimension;
* lifting `ChannelMLP(in + d → 2·hidden → hidden)`;
* optional symmetric zero `DomainPadding`;
* `num_conv_layers` blocks, each `x = act(K x + W x); x = act(MLP(x) + g ⊙ x)`,
  with no activation after the last block;
* one projection `ChannelMLP(hidden → 2·hidden → out)` per head.

`SpectralConv` matches upstream:

* `rfftn(norm="forward")`;
* in every dimension but the last, it keeps `n_modes` modes centred on zero
  (via `fftshift`); in the last it keeps `n_modes[-1] // 2 + 1`;
* Hermitian symmetry is enforced before `irfftn`;
* bias added.

Complex weights are stored as real `(..., 2)` tensors through
`torch.view_as_complex`. As a result, `.double()` gives complex128 weights,
and the FFT dtype follows the input dtype. Upstream hard-codes `cfloat`;
§5 covers what that does to parity.

**HydraGNN interface:**

* Node features live on a tensor-product grid with graph-level
  `data.grid_shape [1, d]` and C-ordered nodes (`meshgrid(indexing="ij")`).
* A batch can mix grid shapes. Graphs with equal shapes share one FFT batch.
* Mismatched shapes or node counts raise an error.

## 3. GINO

Port of `neuralop.models.GINO`, with the default `linear` transforms,
`transformer` sinusoidal embeddings and no weighting function. The forward
pass has four stages:

1. GNO-in: mean over input points within `in_radius` of each latent grid
   point of `k(e(y), e(g)) · x(y)`.
2. Lifting, then `num_conv_layers` FNO blocks on the latent grid (no grid
   embedding, no padding, as upstream).
3. GNO-out: sum over grid points within `out_radius` of each query.
4. One projection per head.

Neighbour search is a dense `cdist` with `dist <= r`. Upstream uses the same
native search when open3d is unavailable.

**Extension (batching):** upstream shares one geometry across a batch. Here:

* every graph has its own point cloud and its own query points;
* neighbour pairs are restricted to the same graph;
* all graphs share the latent grid `linspace(box_min, box_max, latent_resolution)`.

Node heads query at `pos`. Query heads query at `query_pos`, which can be
any set of points that does not match the nodes. Graph ids come from
`query_cell` or `query_graph`.

## 4. GSNO

The layer from the GSNO paper:

$$v_{l+1} = \sigma\big(W v_l + \Phi\, F_t^{-1}(R \cdot F_t(\Phi^\top v_l))\big)$$

* $\Phi$ holds the first `gsno_spatial_modes` eigenvectors.
* $R$ is complex `[k_s, k_t, C, C]`. In the steady case (`gsno_temporal_modes = 0`) $F_t$ is dropped and $R$ is real.
* Activation is GELU, and the projection is `width → 128 → out`.

**Basis** (`graph_spectral_basis`, offline, once per geometry):

* Laplacian: $L = I - D^{-1/2} A D^{-1/2}$ with $A_{ij} = \exp(-|x_i - x_j|^2 / s^2)$ on mesh edges.
* Eigensolver: dense `eigh` for small graphs, `scipy` LOBPCG for large ones.
* Normalization: eigenvectors are orthonormal ($\Phi^\top\Phi = I$).
* Sign fixing: the largest-magnitude entry of each eigenvector is made positive.
* Degenerate groups: modes whose eigenvalues agree within a relative tolerance form a group. `split_degenerate` reports whether the cut at `k` splits a group.

**Invariances:**

* The layer is invariant to sign flips, since each mode enters through $\phi_k\phi_k^\top$.
* Rotations inside a degenerate eigenspace leave it invariant only when the modes of that space share $R$.
* `gsno_tie_degenerate=True` (an extension, off by default) shares $R$ within each group.

**Temporal I/O (assumption):** the paper gives only `T_in = 5 → T_out = 46`.

* Inputs are `T_in` snapshots, flattened time-major.
* Every output time sees all inputs plus `t / T_out`, the FNO-3D convention.
* The FFT runs along the `T_out` axis.

## 5. Verification and results

Environment for every result below:

* CPU: AMD EPYC 7763 login node.
* Software: torch 2.9.1+rocm7.13.0, PyG 2.8.0, scipy 1.16.3, Python 3.11.17.
* Seed: 0.

**Unit tests** (`tests/test_{fno,gino,gsno}.py`, 53 tests):

* FNO:
  * spectral identity under mode truncation in 1D, 2D and 3D, for even and odd sizes;
  * low-pass behaviour;
  * the padding round trip;
  * mixed-resolution batches;
  * gradcheck.
* GINO:
  * non-matching queries in 2D and 3D;
  * queries at the node positions equal the node head;
  * batched vs single evaluation;
  * locality: points outside the radius have no effect.
* GSNO:
  * orthonormality and eigen-residual;
  * LOBPCG vs dense;
  * degenerate groups on a ring graph;
  * sign-flip invariance;
  * rotation invariance only when modes are tied;
  * full-basis identity;
  * temporal low-pass;
  * batches of different meshes.
* All three: a tiny training run and the registry/`create_model` path.

**Upstream parity** (`verify_neuralop_upstream.py`, CPU).

Command:

```
PYTHONPATH=$PWD:<neuralop deps>:<neuralop checkout> \
    python examples/pde/verify_neuralop_upstream.py
```

Method: the script copies the weights into `neuralop.models.FNO`/`GINO` and
compares the outputs on the same inputs. Maximum relative error:

| Case | FP64 | FP32 |
|---|---|---|
| FNO 1D even, 1D odd/truncated, 2D, 2D odd/truncated, 2D padded, 3D | 0 | 0 |
| GINO 2D | 1.1e-15 | 3.0e-7 |
| GINO 3D | 1.0e-15 | 4.0e-7 |

The FP64 numbers need a patch to upstream's dtypes:

* Upstream allocates its spectral output as `torch.cfloat` whatever the input
  dtype, which caps FP64 parity near 1e-9.
* The script therefore patches that dtype to complex128 inside upstream for
  the FP64 runs only. HydraGNN's own code needs no patch.
* Upstream GINO returns `(B, N_q, out)`.

The parity script is not part of the test suite because it needs a neuralop
checkout.

**Property checks** (`python examples/pde/verify_wp06.py`, FP64, 300 Adam
steps, wall time 410 s, CPU):

| Check | Result |
|---|---|
| FNO spectral identity (16, 15, 8×10, 9×7, 6×5×8, 5×7×4) | ≤ 1.8e-15 |
| FNO discretization invariance (same weights, 16 vs 48 grid, band-limited input) | 8.9e-16 |
| FNO padding round trip / gradcheck | 0 / pass |
| GINO 3D: 600 random points → 250 held-out non-matching queries, $u = \sin\pi x \cos\pi y + z^2$ | rel. L2 0.078 |
| Spectral reference: FNO on the 8³ grid, interpolated trilinearly to the same queries | rel. L2 0.029 |
| GSNO orthonormality / eigen-residual | 1.4e-15 / 4.8e-16 |
| GSNO LOBPCG vs dense (eigenvalues / subspace) | 6.2e-16 / 1.4e-12 |
| GSNO sign-flip invariance | 0 |
| GSNO degenerate rotation, untied / tied (groups `[0,1,1,3,3]`) | 3.5e-5 / 6.9e-18 |
| GSNO full-basis reconstruction | 3.0e-15 |
| GSNO projection error with 4 / 16 / 64 modes | 0.59 / 0.14 / 0.031 |
| GSNO temporal low-pass | 2.5e-16 |

GINO is trained on the 600 node values and on 400 query points that are
separate from the evaluation queries. Its error is 2.7× that of the grid FNO
reference, which sees the field on a regular grid. GINO has to integrate
from scattered points through the radius graph.

**GINO memory and time** (FP64, one forward + backward pass on CPU, latent
grid 8³, radius 0.25):

| Points | Neighbour pairs (in = out) | Time |
|---|---|---|
| 500 | 10,027 | 1.4 s |
| 2,000 | 40,583 | 2.3 s |
| 8,000 | 161,672 | 3.8 s |

The number of neighbour pairs grows linearly with the point count. The dense
`cdist` neighbour search allocates an `N × M` distance matrix
(8,000 × 512 here). §5.1 has the GPU peak memory.

**GSNO eigensolve** (CPU, `unit_square_mesh`):

| Nodes | Dense `eigh` | LOBPCG |
|---|---|---|
| 289 | 1.31 s | 0.65 s |
| 1089 | 8.72 s | 1.21 s |
| 4225 | 3.34 s | 1.28 s |

The dense timings are noisy on the shared login node. At these sizes, LOBPCG
(`tol=1e-10`, `maxiter=2000`) warns that some residuals stay near 1e-7, so
its eigenvectors are accurate to about that level. The default `method="auto"`
uses the dense solver up to 2000 nodes.

**Manufactured Poisson** (`examples/pde/poisson_manufactured.py`):

* Data: 240 samples on resolutions n ∈ {6, 8, 10}, with a held-out n = 14 mesh.
* Training: 60 epochs, FP32, 1 rank, CPU.
* Command: `/usr/bin/time -v python poisson_manufactured.py --config <preset>`.

| Model | Config | Test rel. L2 | Held-out 14×14 | Wall time | Peak RSS |
|---|---|---|---|---|---|
| MeshGraphNet (WP03) | `poisson_mgn.json` | 0.147 | 0.188 | – | – |
| FNO | `poisson_fno.json` | 0.024 | 0.025 | 96 s | 3.1 GB |
| GINO | `poisson_gino.json` | 0.050 | 0.067 | 84 s | 3.5 GB |
| GSNO | `poisson_gsno.json` | 0.044 | 0.039 | 45 s | 1.9 GB |

All three generalize to the unseen resolution. The FNO result relies on the
structured unit-square grid. GINO and GSNO make no use of grid structure.

### 5.1 Frontier MI250X

Job: `sbatch ~/fe_runs/wp06_gpu.sbatch` (1 node). It runs `verify_wp06.py
--device cuda --sizes 16 32 64 128` on 1 GCD, then trains each preset on
1 and 8 GCDs. Job 6006158: AMD Instinct MI250X, torch 2.9.1+rocm7.13.0,
Python 3.11.17, seed 0, code identical to `dba901f4`. Every step exited 0.

**`verify_wp06.py`** (FP64, `failed: []`, 243 s inside the script, 436 s
including startup):

* FNO spectral identity ≤ 2.0e-15 (1D/2D/3D); discretization invariance
  8.9e-16; padding round trip 0; gradcheck passes.
* GSNO orthonormality 1.6e-15, eigen-residual 4.8e-16, sign invariance
  2.8e-17, tied degenerate rotation 0 (untied 3.5e-5), full-basis
  reconstruction 3.3e-15, temporal low-pass 2.2e-16.
* GINO 3D held-out query rel. L2 0.078 (FNO grid reference 0.029), same as
  CPU.

**GINO memory** (one forward + backward pass on the GPU):

| Points | Neighbour pairs | Peak GPU memory | Time |
|---|---|---|---|
| 500 | 10,027 | 49 MB | 0.011 s |
| 2,000 | 40,583 | 197 MB | 0.021 s |
| 8,000 | 161,672 | 780 MB | 0.077 s |

**GSNO eigensolve.** This runs in SciPy on the compute node's CPU, not on the
GPU: 289 / 1089 / 4225 nodes take 0.10 / 37.6 / 3.6 s dense and
0.88 / 20.2 / 30.1 s LOBPCG; 16,641 nodes take 78 s LOBPCG. The timings are
noisy (the dense 1089-node solve is slower than the 4225-node one). LOBPCG
emits the same tolerance warning as on the CPU.

**Manufactured Poisson** (FP32, 60 epochs, same data as above):

| Model | GCDs | Test rel. L2 | Held-out 14×14 | Wall time |
|---|---|---|---|---|
| FNO | 1 | 0.024 | 0.025 | 167 s |
| FNO | 8 | 0.020 | 0.021 | 72 s |
| GINO | 1 | 0.050 | 0.067 | 106 s |
| GINO | 8 | 0.051 | 0.068 | 59 s |
| GSNO | 1 | 0.044 | 0.039 | 41 s |
| GSNO | 8 | 0.045 | 0.038 | 32 s |

Wall time includes process startup and data generation. The 1-GCD errors
match the CPU runs to the reported digits.

### 5.2 Full regression suite

Command: `python -m pytest -q -W error -W ignore::DeprecationWarning -W ignore::UserWarning`
on a Frontier login node (serial; `mpi`-marked tests are skipped by `pytest-mpi`
and run separately with `srun -n 2 python -m pytest --with-mpi -m mpi`).

Failures found on this host and their fixes:

| Failures | Cause | Fix |
|---|---|---|
| 7 | `AugmentedLagrangian.update` all-reduced CPU tensors, which NCCL rejects | `_all_reduce_sum` in `domain_losses/constraint_optimizers.py` moves the buffer to the process device for the reduction |
| 2 | `test_heterogeneous_message_passing.py` mocked `move_data` left data on CPU while the model was on GPU | the mock moves data to the model's device |
| 340 | pyg-lib built against the wrong Python ABI and ROCm 6 libraries | environment: rebuild pyg-lib with the venv Python and `gcc` |
| 4 | `fairchem` import failed on missing dependencies | environment: install `fairchem-core` dependencies with torch pinned |
| 3 | `mpi`-marked tests ran in a serial session | environment: install `pytest-mpi`, as in CI |

The first two are code fixes in this PR. The last three are local
environment issues; CI was not affected.

## 6. Limitations

* FNO:
  * tensor-product grids only, with `grid_shape` and C ordering;
  * dense weights only (no tensor factorization, no separable weights, no
    mixed or half precision);
  * no resolution scaling inside blocks.
* GINO:
  * dense `cdist` neighbour search, with O(N·M) memory per graph;
  * one latent grid shared by every graph in a batch;
  * 2D and 3D only;
  * no open3d, no weighting functions, no AdaIN;
  * the batching of separate point clouds goes beyond upstream.
* GSNO:
  * no reference implementation, so parity is with the paper's equations
    only;
  * the edge-weight scale $s$ (mean edge length) and the temporal I/O layout
    are assumptions;
  * the eigenbasis is computed offline on the CPU, once per geometry;
  * LOBPCG on large meshes may stop at residuals of about 1e-7;
  * `tie_degenerate` is an extension;
  * no spherical GSNO and no SGNO.
* Parity script: needs a neuralop checkout, and upstream's FP64 path needs the
  documented dtype patch.
