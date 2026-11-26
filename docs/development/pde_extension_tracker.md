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
| WP04 | Multiscale MGN and RIGNO | WP02 | implemented | | `pde_wp04_design.md` |
| WP05 | Transolver, ++, -3 and GAOT | WP02, WP11 | not started | | |
| WP06 | FNO, GINO, GSNO | WP02 | not started | | |
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
  `model_settings.py`, `examples/pde/poisson_manufactured.py`, `poisson_msmgn.json`, `poisson_rigno.json`.
* Tests: `test_multiscale_meshgraphnet.py`, `test_rigno.py`, `test_poisson_mgn_train.py`,
  `test_model_registry.py`. 303 targeted regression tests pass.
* Results: Poisson (CPU, 1 rank) held-out (n = 14) relative L2: MGN 0.188, MS-MGN 0.198, RIGNO 0.262.
  RIGNO reaches 0.215 on a held-out L-shaped irregular domain at a finer resolution than training.
  Details in `pde_wp04_design.md` §4.
* Limitations: MS-MGN supports two nested triangle levels only; RIGNO is 2D non-periodic, and has no numeric
  parity check against the JAX upstream (blocked: no JAX environment).
