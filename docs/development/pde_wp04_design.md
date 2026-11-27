# PDE extension: WP04 design (Multiscale MeshGraphNet and RIGNO)

Status: implemented. Base: branch `pde-wp02-complete` (PR #583).

| Item | Spec acceptance criterion | Where |
|---|---|---|
| MS-MGN | Constant preservation; correct hierarchy indices; Poisson/Darcy error and scaling vs flat MGN | `tests/test_multiscale_meshgraphnet.py`, `examples/pde/verify_wp04.py`, `poisson_msmgn.json`, `darcy_msmgn.json`, §4 |
| RIGNO | Different input/query resolutions; masking reproducibility; irregular-domain benchmark; comparison with upstream | `tests/test_rigno.py`, `examples/pde/verify_wp04.py`, `examples/pde/verify_rigno_upstream.py`, `poisson_rigno.json`, `darcy_rigno.json`, §4 |

## 1. Components

| File | Content |
|---|---|
| `hydragnn/data/meshes/hierarchy.py` | `build_hierarchy(..., node_transfer=True)` adds `down_index`/`down_weight` (fine node -> containing coarse triangle corners) and `up_index`/`up_weight` (the reverse), with barycentric weights. `prolong_nodes`, `restrict_nodes` interpolate node fields. |
| `hydragnn/models/multiscale_meshgraphnet.py` | `MultiscaleMeshGraphNet(MeshGraphNet)`, registered as `"MultiscaleMeshGraphNet"`. |
| `hydragnn/data/meshes/region_graph.py` | `RegionGraph(MeshData)` (batching offsets for the bipartite indices) and `build_rigno_graph`. |
| `hydragnn/models/rigno.py` | `RIGNO(OperatorBase)`, registered as `"RIGNO"` (output levels `node`, `query`). |
| `hydragnn/model_settings.py` | `MultiscaleMGNSettings` (`msmgn_schedule`) and `RIGNOSettings` (`rigno_mlp_layers`, `rigno_edge_masking`, `rigno_overlap_p2r`, `rigno_overlap_r2p`, `rigno_concat_time`, `rigno_time_conditioning`, `rigno_spatial_dim`). |
| `examples/pde/poisson_manufactured.py` | Builds hierarchies (MS-MGN) or region graphs (RIGNO) from `mpnn_type`; Darcy problem when the inputs include `kappa`. |
| `examples/pde/verify_wp04.py` | FP64 property checks for both models and the time/memory scaling study. |
| `examples/pde/verify_rigno_upstream.py` | Numeric parity with the JAX upstream (graphs and forward pass). |

## 2. Multiscale MeshGraphNet

Port of Fortunato et al. 2022 (arXiv:2210.00612), Sec. 2 and App. A.1. The
fine graph is encoded as in MGN. The coarse graph encodes geometry only
(relative edge coordinates, their norm, one-hot boundary flag). The processor
follows a schedule string such as `"1H 11L 1H"`: `H`/`L` are unshared MGN
blocks on the fine/coarse graph, and a down (`D`) or up (`U`) block is
inserted at every level change. As in the paper, `D` and `U` count towards
the message-passing steps, so `"1H 11L 1H"` is 15 steps. Decoding happens on
fine nodes only and reuses the MGN heads, so face/cell/query heads work.

## 3. RIGNO

PyTorch port of camlab-ethz/rigno (`rigno/graph/builder.py`,
`rigno/models/rigno.py`; Mousavi et al. 2025, arXiv:2501.19205).

**Graphs** (`build_rigno_graph`, preprocessing, NumPy/SciPy):

* Coordinates map to `[-1, 1]`. Upstream scales each axis separately; here
  one isotropic scale keeps the aspect ratio, so distances stay Euclidean.
* Regional nodes are a random `int(N / subsample_factor)` subset drawn from
  a `torch.Generator`. Upstream resamples every epoch; here rebuild with a
  new seed to do the same.
* Support radius per regional node: the largest `0.67 × median` over its
  adjacent Delaunay triangles, so the discs cover the hull of the regional
  nodes (checked in `test_support_radius_covers_hull`). Radii `>= 0.5`
  become `0.2` (upstream guard, configurable through `radius_guard`).
* p2r: physical nodes within `overlap_p2r × r`. r2p/r2q: within
  `overlap_r2p × r`. Both are clipped at `max r`.
* r2r: Delaunay neighbours over `rmesh_levels` nested prefixes of the
  regional nodes (`int(R / factor^l)`, skipped below 4), symmetric, unique.
* **Deviation:** upstream leaves points outside every support disc (hull
  corners, or radii cut by the guard) without a decoder message. With
  `ensure_coverage=True` (default) they link to their nearest regional
  node.

**Model**: edge features `[z, |z|] / (2√d)`. The encoder is one typed
interaction step p2r. Regional nodes embed zeros plus `[x_r, f_p2r r]`, and
physical latents also get a self-update. The processor is `num_conv_layers`
unshared steps on r2r. The decoder is one step r2p onto the encoder's
physical latents, followed by one MLP per head. Aggregations are means;
edges and receivers have residuals. During training each stage keeps a
random `1 - rigno_edge_masking` fraction of its edges (global torch RNG,
reproducible with `torch.manual_seed`); evaluation uses all edges. With
`rigno_time_conditioning`, block MLPs end in
`ConditionedNorm(x, τ) = x (1 + τ s(τ)) + τ b(τ)` on the graph-level `dt`.
With `rigno_concat_time`, graph-level `time` and `dt` are appended to the
physical inputs.

**Query points** (**deviation**): upstream decodes onto the input-node
latents, so outputs must sit on input nodes. Here query points get a
coordinate embedder and their own r2q decoder, so the query set can differ
from the input mesh at every sample (`test_input_and_query_resolutions`,
`test_query_outputs_are_pointwise`).

**Face outputs**: the registry declares only `node` and `query`, so a face
head is rejected (spec: a regional latent must be projected to the fine
mesh before a face-flux decoder).

## 4. Verification and results

* `tests/test_multiscale_meshgraphnet.py`: schedule parsing, node-transfer
  partition of unity and linear exactness, batching, batched vs single,
  translation invariance, the coarse level extending the receptive field,
  face/cell heads, gradients, checkpointing, freezing.
* `tests/test_rigno.py`: graph ranges and symmetry, hull coverage of the
  support radii, decoder coverage, seeded resampling, isotropic
  normalization, batching offsets, batched vs single, input/query
  resolutions, query locality, masking reproducibility and train-only
  behaviour, time conditioning (identity at `dt = 0`, per-graph), gradients,
  checkpointing, freezing, registry. It also runs an L-shaped-domain
  benchmark: train on 8×8/10×10 meshes and test on an unseen 14×14 mesh.
  The relative L2 must stay below 0.5 (a constant predictor gives about 1.0).
* `tests/test_poisson_mgn_train.py`: end-to-end runs for both models through
  the standard data/config/training path.

Manufactured Poisson (`examples/pde/poisson_manufactured.py`, 240 samples,
60 epochs, CPU, defaults per config), relative L2 for `u`:

| Model | Config | Test | Held-out 14×14 |
|---|---|---|---|
| MeshGraphNet | `poisson_mgn.json` | 0.147 | 0.188 |
| MultiscaleMeshGraphNet | `poisson_msmgn.json` (`2H 2L 2H`) | 0.153 | 0.198 |
| RIGNO | `poisson_rigno.json` | 0.186 | 0.262 |

Darcy, `-∇·(κ∇u) = f` with `κ = exp(β cos(pπx) cos(qπy))`, `β ∈ [-1, 1]`,
`p, q ∈ {1, 2}` (contrast up to e²) and the same manufactured `u`; κ is an
extra node input (`darcy_*.json`, otherwise identical to the Poisson
presets). Same protocol, CPU (AMD EPYC 7763, 4 threads), FP32, seed 0:

| Model | Config | Test | Held-out 14×14 | Wall time | Peak RSS |
|---|---|---|---|---|---|
| MeshGraphNet | `darcy_mgn.json` | 0.235 | 0.248 | 47 s | 2.1 GB |
| MultiscaleMeshGraphNet | `darcy_msmgn.json` | 0.213 | 0.215 | 55 s | 2.1 GB |
| RIGNO | `darcy_rigno.json` | 0.239 | 0.214 | 52 s | 2.2 GB |

On Darcy both models beat flat MGN on the held-out resolution (by about
13%). On Poisson they do not.

At these small resolutions, neither model beats single-level MGN on Poisson. MS-MGN's
coarse level has only 3–5 cells per side. RIGNO uses about 1/4 of the
nodes as regional nodes and samples them at random, so it loses detail on
meshes this small. Both papers report their gains on much larger meshes,
where long-range propagation matters; this benchmark tests correctness and
the shared training path, not those gains.

**Property checks** (`python examples/pde/verify_wp04.py`, FP64, random
weights, tolerance 1e-11; CPU, torch 2.9.1+rocm7.13.0, PyG 2.8.0):

| Check | Max error |
|---|---|
| MS-MGN down/up weights: partition of unity | 0 |
| MS-MGN hierarchy indices (`Σ w p_corner = p`, `w >= 0`) | 0 |
| MS-MGN constant preservation (prolong and restrict) | 0 |
| MS-MGN linear exactness | 1.1e-16 |
| MS-MGN batched vs single | 1.0e-15 |
| RIGNO masking under the same seed (train mode) | 0 (different seed: 0.098) |
| RIGNO eval determinism | 0 |
| RIGNO query subset vs full query set | 1.7e-16 |
| RIGNO batched vs single | 7.2e-16 |
| RIGNO input n ∈ {8, 12, 20} × queries ∈ {5, 50, 500} | finite, correct shapes |

**Upstream parity** (`verify_rigno_upstream.py`): this compares against
camlab-ethz/rigno at commit `3e4b307c90f34237d0c1e5e497d4301116e9c3db` (JAX
0.10.2, Flax 0.12.8, jraph 0.0.6.dev0, Python 3.11, CPU, FP64). Two stages:

```
python examples/pde/verify_rigno_upstream.py --stage export --npz rigno_parity.npz
<jax-python> examples/pde/verify_rigno_upstream.py --stage compare \
    --npz rigno_parity.npz --upstream <rigno checkout>
```

The export stage builds L-shaped meshes (n = 12, 20) and runs three
configurations in HydraGNN: plain, time-conditioned, and 3-layer MLPs. It
saves the graphs, weights, latents and outputs. The compare stage rebuilds
the graphs with upstream's `_compute_minimum_support_radius`,
`_get_supported_pnodes_by_rnodes` and `_get_r2r_edges`. It then maps the
weights into upstream's `RIGNO` parameter tree and runs `model.apply` on the
same graphs. Results:

* support radii agree to 1.1e-16;
* p2r, r2r and r2p (before coverage edges) edge sets are identical;
* the parameter trees match;
* encoder, processor and final outputs agree to ≤ 2.7e-15.

The compare stage sets two upstream options to match PyTorch:

* LayerNorm epsilon 1e-5 (the PyTorch default; Flax uses 1e-6);
* `f32_aggregation=False` (upstream casts the aggregation to FP32).

`ensure_coverage` added 18 r2p edges at n = 12 and none at n = 20.

**Scaling** (`verify_wp04.py --scaling`, FP32, presets' hyperparameters,
median of 3 forward + backward passes, CPU, 4 threads):

| n | nodes | MGN | MS-MGN | RIGNO |
|---|---|---|---|---|
| 16 | 289 | 29 ms | 20 ms | 13 ms |
| 32 | 1089 | 95 ms | 71 ms | 31 ms |
| 64 | 4225 | 341 ms | 182 ms | 95 ms |

MS-MGN runs 4 of its 8 steps on the coarse level (1/4 of the nodes), and
RIGNO processes on N/4 regional nodes, so both are cheaper per step than
flat MGN. Graph construction (CPU preprocessing) at n = 64 took 0.11 s for
MGN, 0.94 s for MS-MGN (point location is O(N·C)) and 0.15 s for RIGNO.
GPU numbers: §4.1.

### 4.1 Frontier MI250X

Job `sbatch ~/fe_runs/wp04_gpu.sbatch` (1 node, one MI250X GCD per rank,
torch 2.9.1+rocm7.13.0, PyG 2.8.0). It runs the property checks and
scaling on 1 GCD, then trains each preset on 1 and 8 GCDs:

```
srun -n1 -c7 --gpus-per-task=1 python examples/pde/verify_wp04.py \
    --device cuda --scaling --sizes 16 32 64 128
srun -n$N -c7 --gpus-per-task=1 --gpu-bind=closest \
    python examples/pde/poisson_manufactured.py --config <preset>.json
```

All property checks pass on the GPU in FP64; the largest error is 1.1e-15
(MS-MGN batched vs single). With a different seed, the RIGNO masking
output differs by 0.138. Wall time: 29 s.

Scaling (FP32, median forward + backward, peak allocated memory):

| n | nodes | MGN | MS-MGN | RIGNO |
|---|---|---|---|---|
| 16 | 289 | 11.6 ms / 210 MiB | 13.4 ms / 205 MiB | 14.6 ms / 196 MiB |
| 32 | 1089 | 11.9 ms / 279 MiB | 13.7 ms / 256 MiB | 15.2 ms / 228 MiB |
| 64 | 4225 | 13.3 ms / 557 MiB | 14.1 ms / 456 MiB | 15.8 ms / 354 MiB |
| 128 | 16641 | 35.7 ms / 1592 MiB | 28.3 ms / 1226 MiB | 21.3 ms / 824 MiB |

Up to n = 64 the GPU is latency-bound: the deeper MS-MGN and RIGNO
pipelines launch more kernels. At n = 128, MS-MGN is 21% faster than MGN
and RIGNO is 40% faster, and they use 23% and 48% less memory. At n = 128,
graph construction on the CPU takes 0.40 s (MGN), 6.55 s (MS-MGN) and
0.54 s (RIGNO). MS-MGN construction is dominated by the O(N·C)
fine-to-coarse point location.

Training (same protocol as §4, relative L2 for `u`, test / held-out 14×14):

| Preset | 1 GCD | 8 GCDs |
|---|---|---|
| `poisson_mgn.json` | 0.147 / 0.181 | 0.192 / 0.192 |
| `poisson_msmgn.json` | 0.153 / 0.200 | 0.129 / 0.203 |
| `poisson_rigno.json` | 0.186 / 0.262 | 0.183 / 0.274 |
| `darcy_mgn.json` | 0.233 / 0.244 | 0.255 / 0.285 |
| `darcy_msmgn.json` | 0.212 / 0.213 | 0.214 / 0.222 |
| `darcy_rigno.json` | 0.239 / 0.214 | 0.238 / 0.213 |

All twelve runs exit 0, each in 39–56 s. The 1-GCD results match the CPU
runs to within 0.01. On Darcy at 8 GCDs, MS-MGN and RIGNO
still beat flat MGN on the held-out mesh (0.222 and 0.213 vs 0.285).

## 5. Limitations

* MS-MGN: two levels, nested triangle hierarchies only, coarse mesh
  supplied by the user (no bi-stride pooling).
* MS-MGN: `build_hierarchy` locates points by brute force (O(N·C) memory),
  which dominates preprocessing on large meshes.
* RIGNO: 2D, non-periodic domains only. Upstream parity holds with LayerNorm
  epsilon 1e-5 and FP64 aggregation. Upstream defaults (epsilon 1e-6, FP32
  aggregation) give differences of about 1e-3. The parity script is not part
  of the test suite because it needs a separate JAX environment. Per-epoch
  regional resampling is left to the data pipeline. Graph construction runs
  on the CPU.
