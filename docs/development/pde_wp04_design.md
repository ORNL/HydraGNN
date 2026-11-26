# PDE extension: WP04 design (Multiscale MeshGraphNet and RIGNO)

Status: implemented. Base: branch `pde-wp02-complete` (PR #583).

| Item | Spec acceptance criterion | Where |
|---|---|---|
| MS-MGN | Two-level V-cycle with restriction/prolongation edges; compare with single-level MGN | `tests/test_multiscale_meshgraphnet.py`, `examples/pde/poisson_msmgn.json`, §4 |
| RIGNO | Regional graph construction, p2r/r2p maps, regional processor, edge masking, time conditioning; arbitrary query points; irregular-domain benchmark | `tests/test_rigno.py`, `examples/pde/poisson_rigno.json`, §4 |

## 1. Components

| File | Content |
|---|---|
| `hydragnn/data/meshes/hierarchy.py` | `build_hierarchy(..., node_transfer=True)` adds `down_index`/`down_weight` (fine node -> containing coarse triangle corners) and `up_index`/`up_weight` (the reverse), with barycentric weights. `prolong_nodes`, `restrict_nodes` interpolate node fields. |
| `hydragnn/models/multiscale_meshgraphnet.py` | `MultiscaleMeshGraphNet(MeshGraphNet)`, registered as `"MultiscaleMeshGraphNet"`. |
| `hydragnn/data/meshes/region_graph.py` | `RegionGraph(MeshData)` (batching offsets for the bipartite indices) and `build_rigno_graph`. |
| `hydragnn/models/rigno.py` | `RIGNO(OperatorBase)`, registered as `"RIGNO"` (output levels `node`, `query`). |
| `hydragnn/model_settings.py` | `MultiscaleMGNSettings` (`msmgn_schedule`) and `RIGNOSettings` (`rigno_mlp_layers`, `rigno_edge_masking`, `rigno_overlap_p2r`, `rigno_overlap_r2p`, `rigno_concat_time`, `rigno_time_conditioning`, `rigno_spatial_dim`). |
| `examples/pde/poisson_manufactured.py` | Builds hierarchies (MS-MGN) or region graphs (RIGNO) from `mpnn_type`. |

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

Neither model beats single-level MGN at these small resolutions. MS-MGN's
coarse level has only 3–5 cells per side. RIGNO uses about 1/4 of the
nodes as regional nodes and samples them at random, so it loses detail on
meshes this small. Both papers report their gains on much larger meshes,
where long-range propagation matters; this benchmark tests correctness and
the shared training path, not those gains.

## 5. Limitations

* MS-MGN: two levels, nested triangle hierarchies only, coarse mesh
  supplied by the user (no bi-stride pooling).
* RIGNO: 2D, non-periodic domains only. No numeric parity with the JAX
  upstream (JAX is not installed in the test environment), so the
  comparison is structural (graph rules, feature definitions, block order)
  plus the benchmark above. Per-epoch regional resampling is left to the
  data pipeline. Graph construction runs on the CPU.
