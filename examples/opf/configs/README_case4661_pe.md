# Case-4661 positional-encoding experiments

The five configurations in this directory compare bus-only shared-input
positional encodings while keeping the same HeteroSAGE + standard multihead
GPS architecture:

| Configuration | Active input PE | Sign flip |
|---|---|---|
| `opf_heterosage_case4661_lpe.json` | Laplacian eigenpairs | No |
| `opf_heterosage_case4661_lpe_signflip.json` | Laplacian eigenpairs | Yes |
| `opf_heterosage_case4661_effective_resistance.json` | Effective-resistance summary | N/A |
| `opf_heterosage_case4661_lpe_effective_resistance.json` | Both | No |
| `opf_heterosage_case4661_lpe_signflip_effective_resistance.json` | Both | Yes |

All five configs precompute the same superset (Laplacian eigenpairs plus the
five effective-resistance statistics), so they reuse one serialized dataset:

```bash
sbatch job-frontier-case4661-pe-preonly.sh
```

The resulting dataset is `dataset/case4661_lpe_er.h5`. To train a particular
variant from an allocation, select its config but keep that shared model name:

```bash
python -u train_opf_solution_heterogeneous.py \
  --hdf5 \
  --inputfile configs/opf_heterosage_case4661_lpe.json \
  --modelname case4661_lpe_er \
  --log case4661_lpe
```

`Architecture.positional_encodings.precompute` controls which tensors are
written into the serialized samples, while `Architecture.positional_encodings.use`
controls which of those tensors the model actually consumes. Valid source names
are `laplacian`, `effective_resistance`, `effective_impedance`,
`effective_resistance_rpe`, `effective_impedance_rpe`, and `ybus_svd`; compatible
input PEs can be selected independently or combined. The first three are shared
input PEs. The final three are mutually exclusive relative biases on the
multihead-attention logits and require `attn_node_types: ["bus"]`.

The Laplacian encoding stores the eight smallest nonzero eigenvectors and
their eigenvalues. Training-time sign flipping samples one sign per graph and
eigenmode. Effective resistance uses the fixed-width approximation
`[min, max, std, median, mean]` after removing each row's diagonal entry. Both
encodings are attached only to buses and are fused into the initial bus
representation before the local and global GPS branches.

The configs deliberately use regular multihead attention as requested. Its
quadratic memory cost is why their batch size is set to one.
