# PDE extension: WP03 design (MeshGraphNet baseline and FluxGNN)

Status: implemented. Base: branch `pde-wp01-wp02` at `20d217f7` (PR #581).
Gate from the specification: "trained baseline and flux verification".

| Item | Spec acceptance criterion | Where |
|---|---|---|
| MGN | Permutation equivariance; train and roll out on transport; compare under matched settings | `tests/test_meshgraphnet.py`, `examples/pde/fv_transport_mgn.json`, §4 |
| FluxGNN | Internal flux cancellation; conservation with sources and boundaries; symmetry and rollout tests | `tests/test_fluxgnn.py`, `examples/pde/verify_fluxgnn.py`, §4 |

## 1. Components

| File | Content |
|---|---|
| `hydragnn/physics/conservation.py` | `net_outflow`, `fv_increment`, `balance_residual`: the signed scatter of face fluxes onto cells and the per-graph discrete balance check. Shared by models, examples and tests. |
| `hydragnn/models/fluxgnn.py` | `FluxGNN(OperatorBase)`, registered as `"FluxGNN"` (output levels `cell`, `face`; requires `cell_x`, `face_index`, `cell_volume`). |
| `hydragnn/models/rollout.py` | `rollout(model, data, num_steps, columns, level, head, residual)`: autoregressive rollout that writes a head back into `x` (node) or `cell_x` (cell) columns. |
| `hydragnn/model_settings.py` | `FluxGNNSettings` (`fluxgnn_*`): `dt`, `spatial_dim`, `mlp_layers`, `advection`, `diffusion`, `boundary`, `num_bface_tags`, `source`, `layout`. |
| `hydragnn/config/variable_schema.py` | `cell` is now accepted as an **input** level; cell inputs are concatenated into `data.cell_x`. `face` and `query` stay output-only. At least one node *or* cell feature is required. |
| `hydragnn/config/config_utils.py` | `configure_fluxgnn`: builds `fluxgnn_layout` and `input_dim` from the cell inputs (`fluxgnn_vector_inputs`, `fluxgnn_scalar_inputs`; the rest are conserved state). Cell inputs with another model, or FluxGNN without cell inputs, raise. |
| `examples/pde/fv_transport.py` | Reference FV advection-diffusion-source data, training, held-out evaluation for both models. |
| `examples/pde/verify_fluxgnn.py` | Numerical verification report (exit code 1 on failure). |

## 2. FluxGNN formulation

Cell-centred unknowns $u_i \in \mathbb{R}^k$. One forward pass advances one step of length `fluxgnn_dt` with
`num_conv_layers` explicit substeps $\Delta t_s$:

$$
z \leftarrow z - \frac{\Delta t_s}{V_i}\Big(\sum_{f \ni i} s_{if} A_f F_f + \sum_{b \ni i} A_b F^b_b\Big) + \Delta t_s\, q_i,
\qquad z = E u .
$$

* **Encoder/decoder.** $E$ is a bias-free linear map ($k \to$ `hidden_dim`, orthogonal init); the decoder is
  $E^{+}$. The output is $u^0 + E^{+}(z - z^0)$, so the physical update is a linear image of a conservative
  latent update and keeps the same balance.
* **Interior flux** $F_f$ (owner $\to$ neighbour):
  upwind advection with the face-averaged normal velocity of the first vector input,
  $-\mathrm{softplus}(\kappa)\,(z_n - z_o)/d_{on}$ with a learned per-channel diffusivity, and the correction
  $g(z_o, z_n, \cdot_{+n}) - g(z_n, z_o, \cdot_{-n})$. Every part changes sign with the face orientation, so
  $F_f$ is antisymmetric and interior contributions cancel in $\sum_i V_i \Delta u_i$ exactly.
* **Boundary flux** (`fluxgnn_boundary="learned"`): upwind outflow plus an MLP of the owner latent, scalars,
  boundary-normal velocity invariants and an embedding of `bface_tag` (`fluxgnn_num_bface_tags`).
  `"closed"` sets $F^b = 0$ (no-flux walls).
* **Source** (`fluxgnn_source=true`): MLP of the cell latent, scalars and squared vector norms.
* **Invariant features.** Vectors enter only as normal components and squared tangential magnitudes relative
  to the face normal, plus centroid distances. Outputs are invariant under rotation, reflection and
  translation (vector inputs transformed with the mesh).
* **Initialisation.** All correction MLPs end in zero-initialised layers, so an untrained model is a plain
  upwind/two-point-diffusion FV scheme (tested in `test_untrained_model_is_upwind_diffusion_scheme`).
* **Heads.** A `cell` head returns $u^{n+1}$; a `face` head returns the step-averaged interior flux
  $E^{+}\bar F$ (physical space). `model.fluxes(data)` returns `u_new`, `face_flux`, `bface_flux`, `source`
  so that the update can be checked with `balance_residual`.

### Configuration

```json
"Architecture": {"mpnn_type": "FluxGNN", "hidden_dim": 16, "num_conv_layers": 4,
  "fluxgnn_dt": 0.02, "fluxgnn_source": true,
  "fluxgnn_vector_inputs": ["v"], "fluxgnn_scalar_inputs": ["s"]},
"Variables": {"inputs": [
  {"name": "pos", "role": "position", "level": "node", "dim": 2},
  {"name": "u", "level": "cell", "dim": 1},
  {"name": "v", "level": "cell", "dim": 2},
  {"name": "s", "level": "cell", "dim": 1, "normalize": "scale_only"}],
 "outputs": [{"name": "u_next", "level": "cell", "dim": 1}], "spatial_dim": 2}
```

State, vector and output variables must use `"normalize": "none"`: the update $u + \Delta t(\dots)$ is only
consistent in data units. Scalars may be normalized.

## 3. Relation to the reference implementations

| Source | Pinned commit | License | Use |
|---|---|---|---|
| Horie & Mitsume, ICML 2024, `yellowshippo/fluxgnn-icml2024` | `d4d44497a8ada90c9f39d04a19f805730a887ef4` | Apache-2.0 | Design reference (inspired, no code copied) |
| `ricosjp/siml` (FluxGNN dependency) | `25af1590ac117e1fdcbe5429d6dc178c1cf80040` | Apache-2.0 | Read for FV operator conventions only |
| Pfaff et al., ICLR 2021, `google-deepmind/deepmind-research/meshgraphnets` | — | Apache-2.0 | MGN design reference (WP01/WP02) |

Deliberate differences from the reference FluxGNN:

* Linear encoder/pseudo-inverse decoder instead of the reference's nonlinear encoder with a separate decoder,
  so conservation of the *physical* field holds by construction rather than only in latent space.
* Similarity equivariance (scale) is not enforced; only rotation, reflection, translation and cell permutation.
* No temporal bundling or neural-ODE integrator; explicit forward Euler substeps.
* One advecting vector (the first in `fluxgnn_vector_inputs`); others enter only through invariants.

## 4. Verification and results

Environment: Frontier login node, CPU, single process; Python 3.11.17, torch 2.9.1+rocm7.13.0, PyG 2.8.0;
FP64; seeds fixed in code (`torch.manual_seed(0)`, dataset seed 1, held-out seed 7).

### 4.1 Property checks (`python examples/pde/verify_fluxgnn.py`, random weights, 3-mesh batch)

| Check | Value | Tolerance |
|---|---|---|
| interior cancellation (relative) | 8.2e-18 | 1e-11 |
| balance, learned boundary + source (relative) | 4.6e-16 | 1e-11 |
| balance, closed boundary (relative) | 1.9e-15 | 1e-11 |
| face orientation flip (abs) | 0 | 1e-11 |
| rotation / reflection + translation (abs) | 6.4e-15 / 8.9e-15 | 1e-11 |
| batched vs single (abs) | 0 | 1e-11 |
| balance over a 20-step rollout (relative) | 2.6e-17 | 1e-11 |

### 4.2 Matched training on transport

Command (from any scratch directory):
`python examples/pde/fv_transport.py --config fv_transport_fluxgnn.json` and
`python examples/pde/fv_transport.py --config fv_transport_mgn.json`.

Same 360 samples (60 trajectories × 6 steps on 8×8, 10×10, 12×12 meshes), same 80/10/10 split, 30 epochs,
batch 16, AdamW lr 2e-3 with ReduceLROnPlateau, FP64. MGN runs on the dual graph (node per cell centroid, edge pair per interior
face) with the same inputs. Held-out: a 16×16 trajectory (unseen resolution), 10-step rollout.

| Model | Params | Test one-step rel. L2 | Rollout rel. L2, step 1 / 5 / 10 | Mass drift after 10 steps | Max balance residual |
|---|---|---|---|---|---|
| FluxGNN (hidden 16, 4 substeps) | 2,248 | 3.1e-3 | 0.018 / 0.084 / 0.163 | 0.117 | 1.8e-17 |
| MeshGraphNet (hidden 64, 8 layers) | 246,849 | 7.8e-2 | 0.082 / 0.326 / 0.527 | 0.130 | n/a |

"Mass drift" is $|\sum V(u_{10} - u^{ref}_{10})| / \sum V |u^{ref}_{10}|$. FluxGNN balances its *own* fluxes
exactly; its drift from the reference comes from boundary outflow and source errors, not from interior
non-conservation. Rollout error grows roughly linearly for both models; at the held-out resolution FluxGNN
stays about 3× more accurate at step 10.

Tests: `tests/test_fluxgnn.py` (properties, construction, config, rollout),
`tests/test_fv_transport_train.py` (end-to-end training of both configs, `update_config` errors),
`tests/test_meshgraphnet.py::test_rollout_feeds_node_head_back`, `tests/test_variable_levels_pde.py` (cell inputs).

## 5. Limitations

* Conservation is exact up to round-off; it does **not** imply positivity, a maximum principle or entropy
  stability. Explicit substeps are subject to a CFL-type limit; nothing enforces it.
* Tested on 2D triangle meshes only. The code is dimension-agnostic, but 3D and mixed-element meshes are untested.
* Fixed `dt` per model; variable time steps would need `dt` as an input.
* One `cell` and one `face` head at most, both with the state dimension; graph/node heads, Gaussian NLL,
  global attention, positional encodings and `graph_attr` conditioning are rejected.
* `bface_tag` must lie in `[0, fluxgnn_num_bface_tags)`; boundary-condition *values* (e.g. inflow data) are
  not yet inputs, so the example uses zero inflow.
* Cell inputs are FluxGNN-only; MGN consumes cell data through the dual graph conversion in the example.
* No DDStore/ADIOS-specific test for `cell_x`; it is derived from stored cell variables at load time.
