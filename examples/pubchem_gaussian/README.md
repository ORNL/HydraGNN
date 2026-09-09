# PubChem Gaussian Hessian example

This example preprocesses a small subset of the VIBRANT PubChem dataset and
trains a HydraGNN model from the serialized graphs. Each distributed
`*.tar.zst` archive contains one nested tar archive per PubChem CID, so
preprocessing first expands the requested number of molecular records under
`dataset/raw/extracted`.

Each Gaussian force evaluation in the optimization trajectory becomes one graph.
The parser aligns its input-orientation coordinates, SCF energy, and Cartesian
forces from the Gaussian log. Positions are converted from Angstrom to Bohr;
energies remain in Hartree and forces remain in Hartree/Bohr.

The analytical Hessian is available only for the optimized structure used by
the frequency calculation. Every graph has a dense `data.hessian` attribute of
shape $3N\times3N$. It contains the Hessian in Hartree/Bohr$^2$ when `data.pos`
corresponds to `Structure.txt`; all other trajectory graphs contain an all-NaN
tensor of the same shape. `data.hessian_available` records the same condition
explicitly. Hessian losses must mask unavailable values before reduction.

For `pos` and forces shaped `(N, 3)`, PyTorch returns the force Jacobian with
axes `(output_atom, output_xyz, input_atom, input_xyz)`. Its element
`[a, alpha, b, beta]` is $\partial F_{a\alpha}/\partial R_{b\beta}$. Since
$F=-\partial E/\partial R$, the energy Hessian is the negative force Jacobian.
Gaussian and a contiguous PyTorch tensor both use atom-major Cartesian order,
`(0x, 0y, 0z, 1x, 1y, 1z, ...)`, so the negated Jacobian can be reshaped
directly to `(3N, 3N)`. Matrix element `[3*a + alpha, 3*b + beta]` then equals
$\partial^2 E/(\partial R_{a\alpha}\partial R_{b\beta})$.

The configuration trains a scalar additive energy model. HydraGNN differentiates
that energy once for force loss and, only when finite Hessian labels are present,
again for a masked Hessian loss. Non-optimized trajectory structures therefore
contribute energy and force supervision without triggering Hessian computation.
Batch size is one because dense Hessians vary with molecular size.

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