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
"""Generate prepared named PyG samples for the Ising example."""

import math

import numpy as np
import scipy.special
import torch
from sympy.utilities.iterables import multiset_permutations
from torch_geometric.data import Data

try:
    from .create_configurations import E_dimensionless
except ImportError:  # Direct execution from the example directory.
    from create_configurations import E_dimensionless
from hydragnn.preprocess import get_radius_graph
from hydragnn.utils.input_config_parsing.variable_schema import (
    parse_variable_schema,
    prepare_data_from_schema,
)


def _configurations(length, cutoff, rng):
    # Include both fully polarized endpoint classes: zero and N down spins.
    for num_downs in range(length**3 + 1):
        primal = np.ones(length**3)
        primal[:num_downs] = -1.0
        if scipy.special.binom(length**3, num_downs) > cutoff:
            for _ in range(cutoff):
                yield rng.permutation(primal).reshape(length, length, length)
        else:
            for values in multiset_permutations(primal):
                yield np.asarray(values).reshape(length, length, length)


def prepare_ising_dataset(length, cutoff, config, seed=43, sampling=None):
    """Generate Ising configurations and compile named model tensors."""
    if length <= 0 or cutoff <= 0:
        raise ValueError("length and cutoff must be positive")
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    architecture = config["NeuralNetwork"]["Architecture"]
    add_edges = get_radius_graph(
        radius=architecture["radius"],
        max_neighbours=architecture["max_neighbours"],
        loop=False,
    )
    schema = parse_variable_schema(config["Variables"])
    dataset = []
    for lattice in _configurations(length, cutoff, rng):
        energy, features = E_dimensionless(
            lattice,
            length,
            lambda value: math.sin(math.pi * value / 2),
            True,
        )
        sample = Data(
            node_features=torch.as_tensor(features[:, [0]]).float(),
            spin=torch.as_tensor(features[:, [4]]).float(),
            total_energy=torch.tensor([[float(energy)]], dtype=torch.float32),
            pos=torch.as_tensor(features[:, 1:4]).float(),
        )
        dataset.append(prepare_data_from_schema(add_edges(sample), schema))
    if sampling is not None:
        if not 0 < sampling <= 1:
            raise ValueError("sampling must be in (0, 1]")
        dataset = dataset[: max(1, int(len(dataset) * sampling))]
    return dataset
