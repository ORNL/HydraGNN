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
from hydragnn.data.abstractbasedataset import AbstractBaseDataset

from .dataset import AdiosDataset


class AdiosMultiDataset(AbstractBaseDataset):
    """
    AdiosMultiDataset is a wrapper of multiple AdiosDataset. It will concatenate multiple AdiosDataset together.
    """

    def __init__(self, filenames, label, comm, **kwargs):
        datasets = list()
        for filename in filenames:
            dataset = AdiosDataset(filename, label, comm, **kwargs)
            datasets.append(dataset)
        self.datasets = datasets

    def len(self):
        return sum([dataset.len() for dataset in self.datasets])

    def get(self, i):
        # find which dataset the index belongs to
        for dataset in self.datasets:
            if i < dataset.len():
                return dataset.get(i)
            else:
                i -= dataset.len()
        raise IndexError("Index out of range")
