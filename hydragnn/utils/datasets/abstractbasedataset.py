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
from abc import ABC, abstractmethod

import torch


class AbstractBaseDataset(torch.utils.data.Dataset, ABC):
    """
    HydraGNN's base datasets. This is abstract class.
    """

    def __init__(self):
        super().__init__()
        self.dataset = list()
        self.dataset_name = None

    @abstractmethod
    def get(self, idx):
        """
        Return a datasets at idx
        """
        pass

    @abstractmethod
    def len(self):
        """
        Total number of datasets.
        If data is distributed, it should be the global total size.
        """
        pass

    def apply(self, func):
        for data in self.dataset:
            func(data)

    def map(self, func):
        for data in self.dataset:
            yield func(data)

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        obj = self.get(idx)
        if hasattr(self, "dataset_name"):
            if self.dataset_name is not None:
                if not hasattr(self, "dataset_name_dict"):
                    ## dataset_name_dict is a dict to convert dataset_name to index
                    ## It needs an explicit dimension: 1-by-1
                    self.dataset_name_dict = {
                        "ani1x": torch.tensor([[0]]),
                        "qm7x": torch.tensor([[1]]),
                        "mptrj": torch.tensor([[2]]),
                        "alexandria": torch.tensor([[3]]),
                        "transition1x": torch.tensor([[4]]),
                        "omat24": torch.tensor([[5]]),
                        "oc2020_all": torch.tensor([[6]]),
                        "oc2022": torch.tensor([[7]]),
                        "omol25": torch.tensor([[8]]),
                        "qcml": torch.tensor([[9]]),
                        "odac23": torch.tensor([[10]]),
                        "nabla2dft": torch.tensor([[11]]),
                        "oc2025": torch.tensor([[12]]),
                        "opoly2026": torch.tensor([[13]]),
                    }
                obj.dataset_name = self.dataset_name_dict.get(
                    self.dataset_name, torch.tensor([[-1]])
                )
        return obj

    def __iter__(self):
        for idx in range(self.len()):
            yield self.get(idx)
