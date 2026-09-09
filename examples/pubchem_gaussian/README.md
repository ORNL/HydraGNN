# PubChem Gaussian Hessian example

This example preprocesses a small subset of the VIBRANT PubChem dataset and
trains a HydraGNN model from the serialized graphs. Each distributed
`*.tar.zst` archive contains one nested tar archive per PubChem CID, so
preprocessing first expands the requested number of molecular records under
`dataset/raw/extracted`.

The initial baseline predicts the three eigenvalues of each atom's on-site
$3\times3$ Hessian block in Hartree/Bohr$^2$. This target is invariant to global
rotation and is compatible with the SchNet encoder. Each serialized graph also
retains the complete symmetric $3N\times3N$ Cartesian Hessian in batchable
lower-triangular form as `data.hessian_lower_triangle`, with its dimension in
`data.hessian_dimension`, for development of a future covariant, all-atom-pair
Hessian decoder.

Preprocess 100 molecules into pickle datasets:

```bash
python train.py --preonly --num-molecules 100
```

Train using the preprocessed data:

```bash
python train.py
```

Both commands support distributed execution through the same launcher and
environment used by the other HydraGNN examples. The raw archives, extracted
records, and generated pickle files are intentionally ignored by Git.