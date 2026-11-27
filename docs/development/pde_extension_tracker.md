# PDE extension tracker

Evidence record required by the specification (§1.2). The machine-readable equivalent is
`pde_extension_tracker.json`; keep both in sync. "Verified" means implementation, configuration,
training, inference, gradients, numerical checks, regressions and documentation were all demonstrated;
measured values and environments are in the linked design documents.

| ID | Deliverable | Depends on | Status | PR / commit | Design doc |
|---|---|---|---|---|---|
| WP01 | Registry, configuration, compatibility baseline | – | verified | #581 / `e1564c39` | `pde_wp01_wp02_design.md` |
| WP02 | Mesh, fields, trajectory, hierarchy contracts | WP01 | verified | #581 / `e1564c39`, #583 | `pde_wp01_wp02_design.md` |
| WP03 | MeshGraphNets and FluxGNN | WP02 | verified | #582 | `pde_wp03_design.md` |
| WP04 | Multiscale MGN and RIGNO | WP02 | in review | #584 | `pde_wp04_design.md` |
| WP05 | Transolver, ++, -3 and GAOT | WP02, WP11 | not started | | |
| WP06 | FNO, GINO, GSNO | WP02 | in review | | `pde_wp06_design.md` |
| WP07 | DeepONet and composite architecture | WP01, WP02 | not started | | |
| WP08 | Physics losses, constraints and residuals | WP02 | not started | | |
| WP09 | Time integration and adaptivity | WP08 | not started | | |
| WP10 | Hybrid solvers and preconditioning | WP08 | not started | | |
| WP11 | Spatial and hybrid parallelism | WP02 | not started | | |
| WP12 | Multi-PDE, symbolic, fidelity, UQ, AL, HPO | WP03, WP08 | not started | | |
| WP13 | Verification, benchmark suite, documentation | – | in progress | | this file |

## WP01

* Sources: `models/registry.py` (capabilities), `models/operator_base.py`, `models/create.py`,
  `config/config_utils.py`, `models/utils/model.py` (checkpoint metadata, `load_and_validate`), `cli.py`.
* Tests: `test_model_registry.py`, `test_create_model.py`, `test_config.py`, `test_checkpoint_metadata.py`,
  `test_cli.py`. 276 regression tests passed at merge.
* Limitations: legacy architectures declare only node/graph capabilities.

## WP02

* Sources: `data/meshes/mesh_data.py`, `data/meshes/trajectory.py`, `data/meshes/hierarchy.py`,
  `config/variable_schema.py` (face/cell/query levels, `spatial_dim`),
  `preprocess/field_normalizer.py`, ADIOS2/DDStore `MeshData` storage, `models/meshgraphnet.py`,
  `examples/pde/poisson_manufactured.py`.
* Tests: `test_mesh_data.py`, `test_mesh_contracts.py`, `test_named_variable_schema.py`, `test_variable_levels_pde.py`,
  `test_field_normalizer.py`, `test_mesh_adios.py`, `test_meshgraphnet.py`, `test_poisson_mgn_train.py`.
* Results: Poisson MGN on Frontier (1 and 8 GCDs, FP64), held-out relative L2 0.19–0.23.
* Limitations: hierarchy supports nested meshes only (conservative non-nested transfer deferred); the normal
  orientation check assumes convex cells.

## WP03

* Sources: `models/fluxgnn.py`, `models/rollout.py`, `physics/conservation.py`, cell inputs in
  `config/variable_schema.py` and `config/config_utils.py`, `examples/pde/fv_transport.py`,
  `examples/pde/verify_fluxgnn.py`.
* Tests: `test_fluxgnn.py`, `test_fv_transport_train.py`, `test_meshgraphnet.py`,
  `test_variable_levels_pde.py`. 316 regression tests pass.
* Results: FP64 property checks at round-off (interior cancellation, global balance with sources and
  boundaries, orientation, rotation/reflection, batching, 20-step rollout balance). On held-out transport,
  FluxGNN step-10 rollout relative L2 0.013 vs 0.533 for MGN with 100× fewer parameters. Reproduced on
  Frontier MI250X with 1 and 8 GCDs (DDP); property checks pass on GPU. Details in `pde_wp03_design.md` §4.
* Limitations: no positivity or CFL enforcement; 2D triangles only; boundary-condition values are not
  inputs; MGN not compared with the TF1 DeepMind reference (blocked: no TF1 environment).

## WP04

* Sources: `models/multiscale_meshgraphnet.py`, node transfer in `data/meshes/hierarchy.py`,
  `data/meshes/region_graph.py`, `models/rigno.py`, `MultiscaleMGNSettings`/`RIGNOSettings` in
  `model_settings.py`, `examples/pde/poisson_manufactured.py`, `poisson_{msmgn,rigno}.json`,
  `darcy_{mgn,msmgn,rigno}.json`, `examples/pde/verify_wp04.py`, `examples/pde/verify_rigno_upstream.py`.
* Tests: `test_multiscale_meshgraphnet.py`, `test_rigno.py`, `test_poisson_mgn_train.py`,
  `test_model_registry.py`.
* Results: FP64 property checks (constant preservation, hierarchy indices, masking reproducibility,
  input/query resolutions, batching) are at round-off on CPU and on Frontier MI250X.
  RIGNO matches the JAX upstream (commit `3e4b307`) to 2.7e-15: identical graphs, same forward pass.
  Held-out (n = 14) relative L2 on CPU:
  * Poisson: MGN 0.188, MS-MGN 0.198, RIGNO 0.262.
  * Darcy: MGN 0.248, MS-MGN 0.215, RIGNO 0.214.

  All six presets train on MI250X at 1 and 8 GCDs. On Darcy at 8 GCDs the held-out relative L2 is
  MGN 0.285, MS-MGN 0.222 and RIGNO 0.213.

  At n = 128 on one MI250X GCD, forward + backward takes 36 ms for MGN, 28 ms for MS-MGN and 21 ms for RIGNO.
  RIGNO reaches 0.215 on a held-out L-shaped domain. Details in `pde_wp04_design.md` §4.
* Limitations: MS-MGN supports two nested triangle levels only, and its hierarchy point location is O(N·C).
  RIGNO is 2D non-periodic only. Upstream parity requires LayerNorm epsilon 1e-5 and FP64 aggregation.

## WP06

* Sources: `models/fno.py`, `models/gino.py`, `models/gsno.py`, `data/meshes/graph_spectral.py`,
  `FNOSettings`/`GINOSettings`/`GSNOSettings` in `model_settings.py`, `examples/pde/poisson_manufactured.py`,
  `poisson_{fno,gino,gsno}.json`, `examples/pde/verify_wp06.py`, `examples/pde/verify_neuralop_upstream.py`,
  `LICENSES/NEURALOP-MIT.txt`.
* Tests: `test_fno.py`, `test_gino.py`, `test_gsno.py`, `test_model_registry.py`.
* Results (CPU):
  * FNO and GINO match neuralop 2.0.0 (`a6648ee`). FNO agrees exactly; GINO agrees to 1.1e-15 in FP64 and 4e-7 in FP32.
  * FP64 property checks pass at round-off. They cover the spectral identity in 1D/2D/3D, discretization
    invariance, the GSNO eigenbasis and degenerate tying, and reconstruction.
  * 3D GINO test on 250 held-out non-matching queries: relative L2 0.078. The FNO grid reference reaches 0.029.
  * Poisson, held-out n = 14 relative L2: FNO 0.025, GINO 0.067, GSNO 0.039 (MGN 0.188).
  * GPU run pending. Details in `pde_wp06_design.md` §5.
* Limitations: FNO needs tensor-product grids. GINO uses a dense O(N·M) neighbour search with one latent grid shared
  by every graph. For GSNO, the edge-weight scale and the temporal I/O layout are assumptions (no reference code).
  No spherical GSNO and no SGNO.
