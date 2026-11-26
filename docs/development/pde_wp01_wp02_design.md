# PDE extension: WP01/WP02 design

Status: partially implemented. Slice 1 (MeshData, OperatorBase, MeshGraphNet, Poisson example) and slice 2
(face/cell outputs, `Variables.spatial_dim`) and slice 3 (`FieldNormalizer`) are done; the `query` level, checkpoint storage of the normalizer, and dataset storage are not. Base commit `d751d688` (ORNL/HydraGNN `main`, 2026-11-25).
Scope: WP01 (registry, configuration, compatibility baseline) and WP02 (mesh/field data contracts),
plus the minimum needed for the first vertical slice (MeshGraphNets on a manufactured Poisson problem).

## 1. What exists today (verified in the checkout)

| Area | Current behaviour | Location |
|---|---|---|
| Model registry | Flat `name -> builder(p)`; `p` is a `SimpleNamespace` of every `create_model` argument. No capability metadata. | `hydragnn/models/registry.py` |
| Factory | `create_model_config` reads `config["Architecture"]["mpnn_type"]` and ~60 flat keys; applies precision, graph conditioning, domain loss, checkpointing. | `hydragnn/models/create.py` |
| Model body | `Base` = embedding -> `graph_convs`/`feature_layers` stack -> multi-head decoder. Node-centric by construction. | `hydragnn/models/base.py` |
| Output heads | Only `"graph"` and `"node"`; anything else raises. | `Base._multihead`, `Base.forward` |
| Named variables | `Variables` section with levels `node`/`edge`/`graph`; validated and compiled into `x`, `edge_attr`, `graph_attr`, `y`, `y_loc`. | `hydragnn/config/variable_schema.py` |
| Edge outputs | Accepted by the schema but explicitly rejected: "HydraGNN does not yet provide an edge prediction head". | `hydragnn/config/config_utils.py` |
| Target indexing | `get_head_indices_node_or_mixed` slices `data.y` using `y_loc`; it is level-agnostic. | `hydragnn/train/head_indices.py` |
| Loss | `MultiHeadLossMixin.loss(pred, value, head_index)`; weighted per-head sum. | `hydragnn/models/common/losses.py` |
| Domain losses | `create_domain_loss` wraps the model per `Training.loss.provider` (`interatomic_potential`, `optimal_power_flow`). | `hydragnn/domain_losses/__init__.py` |
| Normalization | Dataset-global `minmax_node_feature` / `minmax_graph_feature`; per-head `output_denormalize`. | `hydragnn/data/*dataset.py`, `hydragnn/postprocess/postprocess.py` |
| Checkpoint | `{"model_state_dict", "optimizer_state_dict"}` only; no config or schema version. | `hydragnn/models/utils/model.py::save_model` |
| Registry tests | Per-architecture forward/gradient/state-dict/determinism/permutation contracts. | `tests/test_model_registry.py` |

Consequences for the PDE work:

1. A new head level does not require changes to target indexing or loss; `y_loc` already carries per-variable boundaries.
2. PyG batching will not offset custom index tensors (faces, cells, boundary faces) unless the `Data` subclass declares `__inc__`.
3. Operator architectures (FNO, DeepONet, RIGNO, Transolver) do not fit `Base`'s conv-stack template and need a parallel base class that satisfies the same training contract.
4. Checkpoints cannot be validated against configuration today; WP01 adds that without breaking existing files.

## 2. WP01: registry and configuration

### 2.1 Registry extension (backward compatible)

Keep `MODEL_REGISTRY` and `register_model(name)` unchanged. Add an optional capability declaration:

```python
@dataclass(frozen=True)
class Capabilities:
    input_levels: frozenset[str] = frozenset({"node"})          # node | edge | face | cell | graph | query
    output_levels: frozenset[str] = frozenset({"node", "graph"})
    spatial_dims: frozenset[int] = frozenset({1, 2, 3})
    requires: frozenset[str] = frozenset()                      # e.g. {"pos"}, {"face_index"}, {"hierarchy"}
    supports_query_mesh: bool = False
    supports_domain_parallel: bool = False
    dtypes: frozenset[str] = frozenset({"bf16", "fp32", "fp64"})
    reference: str | None = None                                # paper id + pinned upstream SHA, or "inspired"

def register_model(name, capabilities: Capabilities | None = None): ...
def get_capabilities(name) -> Capabilities: ...
```

Existing registrations get the default (`node`/`graph`, all dims), which matches current behaviour.
`validate_composition(capabilities, schema, config)` runs in `update_config` and raises actionable errors,
e.g. `"FNO requires a structured grid (requires={'grid_shape'}); dataset provides an unstructured mesh"`.

A second, independent registry is added only when WP07 (composite operators) starts:
`COMPONENT_REGISTRY[kind][name]` with `kind in {encoder, processor, decoder, constraint, integrator}`.
Not part of the first slice.

### 2.2 Operator base class

New `hydragnn/models/operator_base.py`:

```python
class OperatorBase(MultiHeadLossMixin, torch.nn.Module):
    # Same public contract the trainer uses on Base:
    #   head_type, head_dims, num_heads, var_output, loss(pred, value, head_index)
    #   forward(data) -> list[Tensor]   (or (list, list) when var_output)
    def encode(self, data) -> LatentState: ...
    def process(self, state, data) -> LatentState: ...
    def decode(self, state, data) -> list[Tensor]: ...
```

`LatentState` holds named latent tensors per entity level (`node`, `face`, `cell`, `query`, `region`), so
encoder/processor/decoder pairs can map between levels explicitly. MeshGraphNets is implemented on
`OperatorBase`, not `Base`, because its edge latent update and residual scheme differ from `Base`'s conv loop.

Before implementation, confirm every attribute the trainer reads from `model.module`
(`train_validate_test.py`, `head_indices.py`, `loss_reporting.py`) and add a contract test that
`OperatorBase` subclasses satisfy it.

### 2.3 Configuration

- Keep `mpnn_type` as the selector. PDE models register under new names (`MeshGraphNet`, `FluxGNN`, ...).
- Architecture-specific options go in a prefixed settings dataclass, following `model_settings.py`:
  `MeshGraphNetSettings(prefix="mgn_")` with `latent_dim`, `num_processor_steps`, `mlp_layers`, `edge_features`, `noise_std`.
- New top-level optional section `"PDE"`: equation metadata, boundary tags, time stepping, normalization policy.
  It is ignored when absent, so existing configs are untouched.
- `save_model` writes two extra keys, `"config"` (resolved) and `"schema_version"`. `load_existing_model` ignores
  them when missing. A new `load_and_validate` warns or raises on mismatch.

### 2.4 Compatibility baseline

1. Run the existing test suite on the clean checkout and record the pass/skip/fail lists (`tests/` on CPU; MPI-tagged tests separately).
2. Round-trip a checkpoint written by the base commit through the new loader.

## 3. WP02: mesh and field data contracts

### 3.1 Entity model

| Entity | Count | Tensor(s) | Notes |
|---|---|---|---|
| node (vertex) | N | `pos [N,d]` | Existing `pos`; `d in {1,2,3}`. Today `pos` must be dim 3; relax to `d` for PDE schemas only. |
| edge (message graph) | E | `edge_index [2,E]` | Bidirectional message graph, unchanged semantics. |
| face | F | `face_index [2,F]`, `face_normal [F,d]`, `face_area [F,1]` | One row per interior face, oriented from `face_index[0]` to `face_index[1]`. Endpoints are cells (FV) or nodes (vertex-centred). |
| boundary face | Fb | `bface_owner [Fb]`, `bface_normal [Fb,d]`, `bface_area [Fb,1]`, `bface_tag [Fb]` | Outward normals; tags map to BC definitions. |
| cell | C | `cell_conn [nnz]` + `cell_size [C]` (sizes, not CSR pointers, so PyG concatenates them without fix-up), `cell_volume [C,1]`, `cell_centroid [C,d]` | Needed for FEM/FVM; optional for MGN. |
| query | Q | `query_pos [Q,d]` | For operators with a different output mesh (WP04/WP06/WP07). |
| graph | 1 | `graph_attr`, `time`, `dt`, `fidelity` | Existing graph-level path. |

Interior faces are stored once, not twice. That is what makes `F_ij = -F_ji` hold by construction rather than by training.

### 3.2 `MeshData(torch_geometric.data.Data)`

```python
class MeshData(Data):
    _NODE_INDEXED = {"edge_index", "face_index_node", "cell_nodes", "bface_owner_node"}
    _CELL_INDEXED = {"face_index", "bface_owner"}   # when cells are the FV unknowns
    def __inc__(self, key, value, *args, **kwargs):
        if key in self._CELL_INDEXED: return self.num_cells
        if key == "cell_nodes": return self.num_nodes   # -1 padding must be preserved; see test
        return super().__inc__(key, value, *args, **kwargs)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"face_index", "edge_index"}: return -1
        return super().__cat_dim__(key, value, *args, **kwargs)
```

Batching invariants (tested):

- `Batch.from_data_list([a, b])` followed by `to_data_list()` is the identity for every entity tensor.
- Cell connectivity is `cell_conn` (node-offset) + `cell_size` (not offset); padded arrays and CSR pointers both break default collation. Offsets are recovered with `cell_offsets()`.
- Per-graph `face_batch`, `cell_batch` and `query_batch` vectors are created, so pooling and FiLM never mix samples.
- Batched inference equals per-sample inference to FP64 round-off.

### 3.3 Schema changes

Extend `VariableLevel` to `node | edge | face | cell | query | graph`. `_expected_rows` uses `face_index.shape[1]`,
`num_cells` and `query_pos.shape[0]`. `prepare_data_from_schema` already flattens each output into `y`/`y_loc`,
so target indexing works without changes. Add `face_output`, `cell_output` and `query_output` to `_DERIVED_TENSOR_NAMES`.

Implemented: `face` and `cell` levels exist and are accepted **for outputs only** (inputs raise). `query` is not
implemented yet. The `pos` dimension is controlled by `Variables.spatial_dim` (default 3, allowed 1/2/3): the
position input must have dim `spatial_dim`, so atomistic configs keep the dim-3 check unchanged. For MeshGraphNet,
`update_config` copies `spatial_dim` into `Architecture.mgn_spatial_dim` unless it is already set. `create_model`
rejects output levels outside the model's registered `Capabilities.output_levels` (legacy models: node/graph).

### 3.4 Output heads for new levels

Remove the edge-output rejection in `config_utils.py` once the head exists. Head behaviour per level:

| Level | Head input | Head | Output rows |
|---|---|---|---|
| node | node latent | existing `MLPNode` | N |
| face | `concat(h_i, h_j, face_geom)` for oriented faces, antisymmetrized: `f(h_i,h_j) - f(h_j,h_i)` when `conservative=true` | `FaceHead` | F |
| cell | cell latent (gathered from nodes or native) | MLP | C |
| query | latent sampled at `query_pos` (decoder-specific) | MLP | Q |

Implemented in MeshGraphNet: the face head reads the owner/neighbour cell latents from `face_index`
(cell latent = mean node latent of the cell), the face normal and area, and the centroid distance:
$g(c_o, c_n, n_f, A_f, d_{on})$. With `mgn_conservative_faces=true` (default) the mean is
$g(c_o, c_n, n_f) - g(c_n, c_o, -n_f)$, which is exactly antisymmetric under face orientation flip; Gaussian
variance columns use the sum before the softplus so they stay orientation-invariant. A first version used the
two nodes *on* the face (`face_nodes`) with endpoint-symmetric features; it could not learn a normal flux
(loss stayed at the target variance) because those nodes carry no information across the face. The cell head is an MLP on the mean node
latent over the cell's vertices.

The antisymmetric face head is what WP03 FluxGNN uses. The conservative update
$u_i^{n+1} = u_i^n - \frac{\Delta t}{V_i}\left(\sum_{f \in \partial i} \pm A_f F_f + \sum_{b \in \partial i} A_b F_b\right) + \Delta t\, s_i$
is assembled by a `scatter_add` with signs from `face_index` orientation in `physics/conservation.py` (WP08).
WP02 only provides the data needed for it.

### 3.5 Normalization

Today min-max scaling is dataset-global per head. Add a `FieldNormalizer` stored with the dataset and the checkpoint:

- Per named variable, policy in `{none, minmax, standard, scale_only}`. `scale_only` is the default for fluxes and residual targets, so zero and antisymmetry are preserved.
- Statistics are fitted on the training split only.
- Vector and tensor fields use one scale per field, not per component, to preserve rotation equivariance.
- Inverse transforms are applied before any physics residual is computed.

Implemented in `hydragnn/preprocess/field_normalizer.py`. Each variable entry takes an optional
`"normalize"` key (default `"none"`; positions and encoded inputs are rejected). Fitting happens in
`create_dataloaders` on the training split, with statistics reduced across MPI ranks, and the normalizer
is applied in `SchemaPreparedDataset` before `x`/`y` are compiled. The fitted normalizer is reachable as
`loader.dataset.normalizer` and serializes through `state_dict()`. It is not yet written to checkpoints.
No default policy is inferred, so fluxes must opt in to `scale_only` explicitly. The Poisson example
now uses `scale_only` for `f` and `q`, replacing the hand-written scale factors.

### 3.6 Data storage

- Small fixtures and manufactured problems are generated on the fly in `examples/pde/` (no files).
- For large datasets, reuse the ADIOS2 writer with the extra entity tensors registered as variable-length arrays.
  Verify that `adiosdataset` handles arrays whose first dimension is not `num_nodes` (faces, cells) before relying on it.
- Single meshes too large for one device are out of scope until WP11. Record this as a known limitation.

## 4. First vertical slice

Goal: `mpnn_type="MeshGraphNet"` trains on a 2D manufactured Poisson problem using the existing `train_validate_test`, and passes the tests below.

1. `examples/pde/poisson_manufactured.py` generates triangular meshes of the unit square at several resolutions using a pure-PyTorch/NumPy structured triangulation (no external mesher), with $u = \sin(\pi x)\sin(\pi y)$, $f = 2\pi^2 u$. Inputs: `f`, boundary mask, `pos`. Output: `u` at nodes.
2. `hydragnn/data/meshes/mesh_data.py` contains `MeshData`, size-encoded cells, faces and boundary faces built from triangles.
3. `hydragnn/models/operator_base.py` and `hydragnn/models/meshgraphnet.py` implement encode–process–decode with edge and node latent updates, residual connections and LayerNorm, following Pfaff et al. 2021. Record the upstream SHA.
4. Register `MeshGraphNet` with `Capabilities(output_levels={"node"}, requires={"pos", "edge_index"})`.
   The first slice implements only `output_levels`, `requires` and `reference`; the remaining fields above are added when a model needs them.
5. Add `examples/pde/poisson_mgn.json` using the `Variables` section.

### 4.1 Tests

| Test | Check | Tolerance |
|---|---|---|
| `test_mesh_data_batching` | collate/split identity; offsets for `face_index`, `bface_owner`, `cell_conn` | exact |
| `test_mesh_geometry` | sum of cell volumes = domain area; sum over closed cell of outward normal * area = 0 | FP64 1e-12 |
| `test_variable_levels_pde` | face/cell/query shape validation; atomistic `pos` rule unchanged | exact |
| `test_meshgraphnet_contract` | added to `tests/test_model_registry.py` parametrization | existing |
| `test_meshgraphnet_permutation` | node relabelling equivariance | FP64 1e-10 |
| `test_batched_vs_single` | batched vs per-sample predictions | FP64 1e-12 |
| `test_poisson_mgn_train` | loss decreases; relative L2 on held-out resolution reported (no fixed threshold until baseline measured) | report |

## 5. Open decisions

1. FV unknowns on cells vs vertices. Proposal: support both, with `face_index` endpoints declared by `PDE.unknown_location`.
2. Whether `MeshData` lives in `hydragnn/data/meshes/` (spec layout) or `hydragnn/preprocess/`. Proposal: `data/meshes/`.
3. Whether to relax `pos` dim globally or only under `PDE`. Proposal: only under `PDE`, to avoid touching atomistic validation.
4. Optional dependencies (`meshio`, `scipy.sparse`) are allowed in examples and tests only, not in core imports.
