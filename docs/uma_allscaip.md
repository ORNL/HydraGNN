# UMA and AllScAIP integration

HydraGNN provides `UMA` and `AllScAIP` as monolithic atomistic backbones. They
run their own complete block stacks inside HydraGNN and then expose node
representations to HydraGNN's task heads; they are not ordinary local MPNN
layers combined with the `GPS` engine.

## Input data contract

Both backbones require `data.pos` with shape `[num_nodes, 3]`. Prefer an integer
`data.atomic_numbers` tensor with one value per node. For compatibility, the
current integration falls back to `data.x[:, 0]` when `atomic_numbers` is not
present. Periodic samples may provide `data.cell` and `data.pbc`; non-periodic
samples default to a zero cell with periodicity disabled. Optional `charge` and
`spin` values are graph-level scalars.

Set `enable_interatomic_potential` to `true` when training an energy-conserving
potential. HydraGNN then obtains conservative forces by differentiating the
predicted invariant energy with respect to `data.pos`.

## UMA

Select UMA with `"mpnn_type": "UMA"` and set `"equivariance": true`. UMA
maintains SO(3) irreducible features internally and is genuinely equivariant.
The default HydraGNN decoder reads its invariant L=0 channels. Set
`uma_equivariant_vector_head` to expose an L=1 per-node vector head and use
`uma_vector_head_index` when more than one node head could match.

Standard HydraGNN keys configure the cutoff, neighbor cap, width, depth, radial
basis size, and periodic behavior. UMA-specific controls include `uma_variant`
(`S`, `M`, or `L`), `uma_mmax`, `uma_grid_resolution`, `uma_edge_channels`,
`uma_hidden_channels`, `uma_norm_type`, `uma_ff_type`, `uma_use_chg_spin`,
`uma_num_experts`, `uma_moe_dropout`, and `uma_use_composition_embedding`.

HydraGNN vendors the required FAIR-Chem UMA implementation and its preserved
MIT notice, so normal UMA execution does not import the external
`fairchem-core` package. HydraGNN does not yet provide FAIR-Chem's graph-parallel
UMA runtime; large-graph support therefore does not imply distributed graph
partitioning parity.

## AllScAIP

Select AllScAIP with `"mpnn_type": "AllScAIP"`. It constructs its own
differentiable radius/kNN graph and uses scalar hidden representations. Its
energy prediction is invariant and its energy-gradient forces are equivariant,
but its latent representation is not an e3nn-style equivariant tensor field.

The standard `radius`, `max_neighbours`, `hidden_dim`, and `num_conv_layers`
keys configure its principal dimensions. AllScAIP-specific controls include
`allscaip_num_heads`, `allscaip_freq_list`, `allscaip_atten_name`,
`allscaip_use_node_path`, `allscaip_use_sincx_mask`,
`allscaip_use_freq_mask`, `allscaip_max_num_elements`, `allscaip_knn_soft`,
`allscaip_distance_function`, normalization and dropout settings, stress
regression, and dataset routing.

For large graphs, `allscaip_use_chunked_graph` bounds graph-construction memory
by processing target nodes in chunks of `allscaip_graph_chunk_size` while
retaining the same graph semantics. `allscaip_knn_use_low_mem` selects the
lower-memory kNN implementation.

Keep `allscaip_knn_soft` enabled for energy-gradient force training. The hard
neighbor selection is not differentiable at membership changes. The `math`
attention backend is the safe default when double backward is required.

## Dependency and provenance

Vendored source retains its upstream MIT notices; HydraGNN additions remain
under the repository's BSD-3-Clause terms. `requirements-specific-models.txt`
provides auxiliary dependencies required by the vendored UMA backbone, while
`requirements-dev.txt` freezes `fairchem-core==2.22.0` for native parity tests.
The native test dependency is distinct from the vendored runtime used by
HydraGNN's adapters. See the stack module docstrings and vendoring provenance
files for the complete key mapping and upstream revision details.
