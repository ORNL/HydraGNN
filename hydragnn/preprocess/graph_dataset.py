##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Load training-ready graph datasets without modifying their samples."""

import pickle
from pathlib import Path

import torch


def load_pickled_graphs(dataset_path):
    """Load a trusted, training-ready HydraGNN pickle container as-is.

    The container stores node/graph normalization metadata followed by the
    graph collection. Loading deliberately does not rebuild connectivity,
    normalize edge attributes, regenerate descriptors, or compile variables.
    Those operations belong to the preprocessing workflow that created the
    artifact.
    """
    with open(dataset_path, "rb") as stream:
        pickle.load(stream)
        pickle.load(stream)
        return pickle.load(stream)


def load_prepared_graph_dataset(dataset_path):
    """Load a trusted prepared pickle or directory of PyG ``.pt`` samples.

    Every returned sample is exactly the object stored in the prepared
    artifact. PyTorch and pickle deserialization can execute code, so only
    artifacts produced by a trusted workflow may be loaded.
    """
    path = Path(dataset_path)
    if path.is_file():
        if path.suffix != ".pkl":
            raise ValueError(f"Prepared dataset file must end in .pkl: {path}")
        return load_pickled_graphs(path)
    if path.is_dir():
        samples = sorted(path.glob("*.pt"))
        if not samples:
            raise ValueError(f"No serialized PyG .pt samples found in {path}")
        return [torch.load(sample, weights_only=False) for sample in samples]
    raise ValueError(f"Prepared dataset does not exist: {path}")
