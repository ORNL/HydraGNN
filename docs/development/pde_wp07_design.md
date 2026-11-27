# PDE extension: WP07 design (DeepONet and composite architecture)

Status: implemented, in review (PR #586). Base: `main` at `6ef13162` (WP06, PR #585).

| Item | Spec acceptance criterion | Where |
|---|---|---|
| DeepONet | Branch/trunk networks with sensor and query encoding, multi-output contractions; arbitrary queries; manufactured function-to-function map; gradient and sensor-shape tests; stacked/unstacked and multi-output variants; DeepXDE as optional reference | `tests/test_deeponet.py`, `examples/pde/verify_wp07.py`, `poisson_deeponet.json`, §2, §5 |
| Composite | Typed component pipeline, adapters, skips and configurable multi-head outputs; at least two distinct combinations trained; incompatible entities/dimensions rejected; checkpoint round-trip | `tests/test_composite.py`, `verify_wp07.py`, `poisson_composite_{mgn_gsno,regional_flux}.json`, §3, §5 |

DeepONet follows the DeepXDE reference (`deepxde/nn/pytorch/deeponet.py` and
`deeponet_strategy.py`, lululxvi/deepxde) and Lu et al., *Nat. Mach. Intell.*
3, 218 (2021), doi:10.1038/s42256-021-00302-5. DeepXDE is not a dependency.
Its contraction formulas are re-implemented in the tests and in the verify
script, and the model matches them to round-off.

`CompositeNeuralOperator` is a HydraGNN-native hybrid. It reuses the
MeshGraphNet, RIGNO and GSNO building blocks. No pipeline it builds is a
faithful published architecture, and the registry and docs do not label it
as one.

## 1. Components

| File | Content |
|---|---|
| `hydragnn/models/deeponet.py` | `GroupedFNN`, `DeepONet(OperatorBase)`, registered as `"DeepONet"` (output levels `node`, `query`). |
| `hydragnn/models/composite.py` | Components, `DECODERS`, `resolve_pipeline`, `CompositeNeuralOperator(OperatorBase)`, registered as `"CompositeNeuralOperator"` (output levels `node`, `cell`, `face`, `query`). |
| `hydragnn/models/meshgraphnet.py` | `cell_latent`, `query_latent`, `decode_face`, factored out of `MeshGraphNet.decode` so the composite decoders reuse them. |
| `hydragnn/models/gino.py` | `_query_graph_ids(data, owner)` names the calling model in its error. |
| `hydragnn/model_settings.py` | `DeepONetSettings` (`deeponet_*`), `CompositeSettings` (`composite_*`). |
| `examples/pde/poisson_manufactured.py` | Adds `deeponet_sensors` (an 8×8 sensor grid of `f`) and the RIGNO graphs and eigenbasis that a composite pipeline needs, chosen by `mpnn_type`. |
| `examples/pde/poisson_{deeponet,composite_mgn_gsno,composite_regional_flux}.json` | Presets. |
| `examples/pde/verify_wp07.py` | FP64 property checks, the antiderivative operator, and composite fits. |

Neither model supports global attention or `pe_dim`; the registry rejects
them, and it rejects `face` outputs for DeepONet.

## 2. DeepONet

$$G(u)(y)_k = \sum_{i=1}^{p} b_{k,i}\big(u(x_1),\dots,u(x_m)\big)\, t_{k,i}(y) + b_{0,k}$$

* **Branch:** an FNN on the flattened sensor values `[G, m·c]`, with a linear
  last layer.
* **Trunk:** an FNN on the first `coord_dim` coordinates. Its output goes
  through the activation (`act(trunk(y))`), as in DeepXDE.
* **Bias:** one scalar per output column (`bias [K]`).
* **Unstacked** (default): one branch net of width `p`. **Stacked**
  (`deeponet_stacked: true`): `p` independent scalar-output branch nets.
  `GroupedFNN` evaluates the groups in one `einsum`, and a test checks that it
  equals separate nets.
* **Multiple outputs.** `K` counts every head column, including GaussianNLL
  variance columns. The DeepXDE strategies are:

| `deeponet_multi_output_strategy` | Branch | Trunk | Contraction |
|---|---|---|---|
| `null` (K = 1 only) | width p | width p | $\sum_i b_i t_i$ |
| `independent` (default for K > 1) | K nets, width p | K nets, width p | per output |
| `split_both` (`p % K == 0`) | width p, K groups | width p, K groups | per group of p/K |
| `split_branch` | width K·p | width p, shared | $\sum_i b_{k,i} t_i$ |
| `split_trunk` | width p, shared | width K·p | $\sum_i b_i t_{k,i}$ |

  A strategy with K = 1 is an error.
* **Sensors.** The model reads `data.deeponet_sensors [G, m·c]`. This is a
  graph-level tensor, so meshes of any size share one sensor layout. Without
  it, the model reads `data.x` of graphs with exactly `m` nodes, in node
  order. Any other shape raises an error that names the expected shape.
* **Queries.** Node heads evaluate the trunk at `pos`; query heads evaluate it
  at `query_pos`. Query graph ids come from `query_graph` or `query_cell`.
  The output can therefore be evaluated at arbitrary points, independent of
  the mesh.
* `freeze_conv` freezes the branch and trunk and leaves the output biases
  trainable.

## 3. Composite architecture

A pipeline is an encoder, a list of processors and adapters, and one decoder
per output head. The latent state has two entities: `node` (fine mesh) and
`region` (RIGNO regional nodes from `build_rigno_graph`).

| Component | Kind | Reads → writes | Needs | Dims |
|---|---|---|---|---|
| `NodeEncoder` | encoder | – → node | `x` | 2, 3 |
| `RegionalGraphEncoder` | encoder | – → region | `x`, RIGNO p2r | 2 |
| `LocalMessagePassing` | processor | node → node | `pos`, `edge_index` | 2, 3 |
| `GraphSpectral` | processor | node → node | `gsno_basis` | 2, 3 |
| `RegionalMessagePassing` | processor | region → region | RIGNO r2r | 2 |
| `NodeToRegion` | adapter | node → region | RIGNO p2r | 2 |
| `RegionToNode` | adapter | region → node | RIGNO r2p | 2 |
| `Projection` | adapter | any → same, new width | – | 2, 3 |

| Decoder | Level | Notes |
|---|---|---|
| `NodeDecoder` | node | MLP |
| `CellDecoder` | cell | mean of the cell's node latents |
| `QueryDecoder` | query | barycentric interpolation in `query_cell` (2D) |
| `ConservativeFluxDecoder` | face | antisymmetric in the face orientation (default) |
| `FluxDecoder` | face | not antisymmetric |

`resolve_pipeline` checks the whole configuration before any module is built.
It rejects:

* an unknown component, a missing or repeated encoder, or a processor that
  reads the wrong entity;
* a component used in an unsupported spatial dimension (regional blocks are
  2D only);
* a width mismatch (insert a `Projection`), unknown options, and `skip` on
  adapters;
* any decoder on a pipeline that ends on `region` latents. The error tells
  the user to add `RegionToNode`, so the regional state is projected back to
  the fine mesh before a node, cell, face or query decoder;
* a decoder that does not match its level, unknown decoder levels, and query
  decoders outside 2D.

At run time, a missing data attribute is reported with the full list of
missing names. `skip: true` wraps a processor in a residual. Unknown
activation names are rejected; `activation_function_selection` would
otherwise return `None`. Checkpoints store the resolved config through
`checkpoint_metadata`, and a test round-trips weights and config.

## 4. Configuration

```json
"mpnn_type": "DeepONet",
"deeponet_num_sensors": 64,
"deeponet_basis_dim": 64,
"deeponet_stacked": false,
"deeponet_multi_output_strategy": null
```

```json
"mpnn_type": "CompositeNeuralOperator",
"composite_encoder": {"type": "RegionalGraphEncoder"},
"composite_processors": [
  {"type": "RegionalMessagePassing", "layers": 4},
  {"type": "RegionToNode"},
  {"type": "LocalMessagePassing", "layers": 2, "skip": true}
],
"composite_decoders": {"face": "ConservativeFluxDecoder"}
```

## 5. Verification

**Unit tests:**

* `tests/test_deeponet.py` (33 tests): `GroupedFNN` against separate nets;
  `GroupedFNN` weight gradients keep the parameter strides (needed by DDP);
  every strategy, stacked and unstacked, against the DeepXDE formulas;
  default strategy; constructor validation; independence of stacked branch
  nets; sensors from `x` and sensor-shape errors; batched equals single;
  query at node positions equals the node output; FP64 `gradcheck`;
  GaussianNLL variance columns; `freeze_conv`; loss reduction in training;
  registry and `create_model`; checkpoint round-trip.
* `tests/test_composite.py` (26 tests): capabilities; both pipelines train;
  batched equals single for node, face, cell and query heads; conservative
  antisymmetry; skip as a residual; `conv_checkpointing`; missing attributes;
  `Projection`; 13 configuration rejections; constructor rejections; default
  pipeline; checkpoint round-trip with config metadata.
* `tests/test_poisson_mgn_train.py`: end-to-end training of the three presets
  through `hydragnn.run_training`.
* `tests/test_model_registry.py`: both models added to `PDE_MODELS`.

**`verify_wp07.py`** (FP64):

Command: `python examples/pde/verify_wp07.py [--device cuda] > verify_wp07.json`
(seed 0, 3000 DeepONet steps, 300 composite steps). Both runs used the
uncommitted WP07 tree on top of `6ef13162`. All checks pass
(`failed: []`) on CPU (AMD EPYC 7763, 1 process; script wall time 158 s,
peak RSS 0.56 GB) and on one MI250X GCD (wall time 28 s).

| Check | CPU | MI250X |
|---|---|---|
| Antiderivative: initial → held-out rel L2 (64 functions, 50 sensors, 512 train functions, ReLU) | 1.020 → 0.0494 (train 0.0207) | 1.020 → 0.0487 (train 0.0212) |
| Strategies vs DeepXDE formulas, max abs diff | 2.2e-16 | 5.6e-17 |
| Query at node points vs node output; batched vs single | 0.0; 2.8e-17 | 0.0; 0.0 |
| Sensor-shape errors (4 cases) and FP64 `gradcheck` | pass | pass |
| Composite MGN + GSNO (node, cell, query): loss; batched vs single | 0.392 → 1.60e-4; 1.1e-16 | 0.392 → 1.60e-4; 5.6e-16 |
| Composite regional + flux (node, face): loss; batched vs single; face antisymmetry | 0.383 → 1.98e-4; 6.7e-16; 0.0 | 0.383 → 1.98e-4; 1.0e-15; 1.1e-15 |
| Rejections (6 cases) | pass | pass |

The held-out antiderivative error plateaus near 0.05 for this budget (tanh:
0.057–0.088; ReLU with 256/512/1024 functions: 0.060/0.049/0.051), so the
pass threshold is `held-out < 0.08` and `held-out < 0.1 × initial`.

**Manufactured Poisson** (`examples/pde/poisson_manufactured.py`; 240 samples
on n ∈ {6, 8, 10}, held-out n = 14; 60 epochs, FP32, seed 0):

Command (from `examples/pde`):
`OMP_NUM_THREADS=8 python poisson_manufactured.py --config <preset>.json`
(CPU, 1 process, AMD EPYC 7763).

| Preset | Test rel L2 | Held-out n = 14 rel L2 | Wall | Peak RSS |
|---|---|---|---|---|
| `poisson_deeponet.json` | 0.0734 | 0.0797 | 47.5 s | 2.36 GB |
| `poisson_composite_mgn_gsno.json` | 0.0187 | 0.0272 | 61.3 s | 2.28 GB |
| `poisson_composite_regional_flux.json` | u 0.0993 / q 0.1129 | u 0.1769 / q 0.2498 | 80.3 s | 2.20 GB |

The DeepONet preset uses 64 sensors on an 8 × 8 grid; it is less accurate
than the mesh models because its input side is a fixed sensor layout.

### 5.1 Frontier MI250X

Slurm job 6007497 (1 node, `srun -n {1,8} --gpus-per-task=1`, NCCL
backend, torch 2.9.1+rocm7.13.0, FP32, seed 0). All runs exited 0. Wall
time is the whole `srun` step, including start-up.

| Preset | GCDs | Wall | Test rel L2 | Held-out n = 14 rel L2 |
|---|---|---|---|---|
| `poisson_deeponet.json` | 1 | 33 s | 0.0898 | 0.1052 |
| `poisson_deeponet.json` | 8 | 28 s | 0.1174 | 0.1640 |
| `poisson_composite_mgn_gsno.json` | 1 | 51 s | 0.0187 | 0.0272 |
| `poisson_composite_mgn_gsno.json` | 8 | 37 s | 0.0182 | 0.0292 |
| `poisson_composite_regional_flux.json` | 1 | 66 s | u 0.0993 / q 0.1129 | u 0.1769 / q 0.2498 |
| `poisson_composite_regional_flux.json` | 8 | 49 s | u 0.0827 / q 0.1032 | u 0.1591 / q 0.2334 |

With 8 GCDs each rank sees 1/8 of the 240 samples per epoch, so each run
takes 8 times fewer optimizer steps at the same batch size. The DeepONet
preset is the most sensitive to this; its 8-GCD validation loss was still
fluctuating at epoch 60 (0.003–0.007). The problem is too small to show
speed-up; these runs only check that the models train under DDP. The
DeepONet difference between CPU and 1 GCD (0.073 vs 0.090) comes from
non-deterministic GPU kernels in FP32 over 60 epochs. The composite
presets match to 4 digits.

### 5.2 Full regression suite

Serial suite on a Frontier login node (`python -m pytest -q -W error -W
ignore::DeprecationWarning -W ignore::UserWarning -m "not mpi and not
fairchem" tests`): 1177 passed, 3 failed, 28 skipped, 358 deselected
(66 min). The three failures were
`test_examples.py::test_examples_energy[md17-{GAT,MFC,PNA}-multihead-GPS]`,
all with `ncclUnhandledCudaError` in the example subprocess while other
GPU jobs shared the login node. A re-run of those three on their own
passed (3 passed, 124 s). They do not touch WP07 code.

MPI suites (2 ranks, gloo, CPU, compute node): running (jobs 6007688 and 6007689); results will
be added here.

## 6. Limitations

* DeepONet:
  * one fixed sensor layout per model; it is not resolution independent on
    the input side;
  * the trunk sees coordinates only (no extra trunk features, no Fourier
    feature encoding);
  * no POD-DeepONet and no MIONet (multiple input functions).
* Composite:
  * regional components and the query decoder are 2D only;
  * activations are limited to the `activation_function_selection` names;
  * no multiscale MGN, attention or FiLM components yet;
  * the regional mesh is fixed per sample; per-epoch resampling is left to
    the data pipeline, as in WP04.
