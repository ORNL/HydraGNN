# PDE extension tracker

Evidence record required by the specification (§1.2). The machine-readable equivalent is
`pde_extension_tracker.json`; keep both in sync. "Verified" means implementation, configuration,
training, inference, gradients, numerical checks, regressions and documentation were all demonstrated;
measured values and environments are in the linked design documents.

| ID | Deliverable | Depends on | Status | PR / commit | Design doc |
|---|---|---|---|---|---|
| WP01 | Registry, configuration, compatibility baseline | – | verified | #581 / `e1564c39` | `pde_wp01_wp02_design.md` |
| WP02 | Mesh, fields, trajectory, hierarchy contracts | WP01 | partial | #581 / `e1564c39` | `pde_wp01_wp02_design.md` |
| WP03 | MeshGraphNets and FluxGNN | WP02 | verified | #582 | `pde_wp03_design.md` |
| WP04 | Multiscale MGN and RIGNO | WP02 | not started | | |
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

* Sources: `data/meshes/mesh_data.py`, `config/variable_schema.py` (face/cell/query levels, `spatial_dim`),
  `preprocess/field_normalizer.py`, ADIOS2/DDStore `MeshData` storage, `models/meshgraphnet.py`,
  `examples/pde/poisson_manufactured.py`.
* Tests: `test_mesh_data.py`, `test_named_variable_schema.py`, `test_variable_levels_pde.py`,
  `test_field_normalizer.py`, `test_mesh_adios.py`, `test_meshgraphnet.py`, `test_poisson_mgn_train.py`.
* Results: Poisson MGN on Frontier (1 and 8 GCDs, FP64), held-out relative L2 0.19–0.23.
* Missing: trajectory and hierarchy containers (needed by WP04 and WP09). Only 2D triangle meshes tested.

## WP03

* Sources: `models/fluxgnn.py`, `models/rollout.py`, `physics/conservation.py`, cell inputs in
  `config/variable_schema.py` and `config/config_utils.py`, `examples/pde/fv_transport.py`,
  `examples/pde/verify_fluxgnn.py`.
* Tests: `test_fluxgnn.py`, `test_fv_transport_train.py`, `test_meshgraphnet.py`,
  `test_variable_levels_pde.py`. 316 regression tests pass.
* Results: FP64 property checks at round-off (interior cancellation, global balance with sources and
  boundaries, orientation, rotation/reflection, batching, 20-step rollout balance). On held-out transport,
  FluxGNN step-10 rollout relative L2 0.013 vs 0.533 for MGN with 100× fewer parameters. CPU and
  Frontier GPU/DDP numbers are in `pde_wp03_design.md` §4.
* Limitations: no positivity or CFL enforcement; 2D triangles only; boundary-condition values are not
  inputs; MGN not compared with the TF1 DeepMind reference (blocked: no TF1 environment).
