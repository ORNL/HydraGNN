##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

"""
Unit tests for the optional masked-loss path in Base.loss_hpweighted.

This is the core-framework change of the cov80 T-GCN port (PR #2): an optional
``mask=`` kwarg that lets imputed / missing targets be excluded from the loss.
The contract this guards:

  * mask=None         -> bit-identical to the existing (unmasked) MSE  [no-op]
  * all-ones mask     -> equals plain MSE (mean over every element)
  * partial mask      -> mean of squared error over the KEPT elements only
  * non-"mse" + mask  -> rejected (masked path needs an elementwise loss)

The method is exercised directly through a lightweight stub so the test stays
fast and model-free: loss_hpweighted only reads num_heads, loss_weights,
loss_function, and loss_function_type off ``self``.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from hydragnn.models.Base import Base
from hydragnn.utils.model.model import loss_function_selection


def _stub(loss_function_type="mse"):
    s = types.SimpleNamespace()
    s.num_heads = 1
    s.loss_weights = [1.0]
    s.loss_function_type = loss_function_type
    s.loss_function = loss_function_selection(loss_function_type)
    return s


# A single node-level head: pred[0] and value are [N, D]; head 0 spans all rows.
_PRED = [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])]
_VALUE = torch.tensor([[1.5, 2.0], [2.0, 5.0], [5.0, 5.0]])
_HEAD_INDEX = [torch.arange(3)]


@pytest.mark.mpi_skip()
def pytest_masked_loss_none_is_plain_mse():
    """mask=None must reproduce the existing unmasked MSE exactly (no-op)."""
    tot, _ = Base.loss_hpweighted(_stub(), _PRED, _VALUE, _HEAD_INDEX, mask=None)
    expected = F.mse_loss(_PRED[0], _VALUE)
    assert torch.allclose(tot, expected)


@pytest.mark.mpi_skip()
def pytest_masked_loss_all_ones_equals_mse():
    """An all-ones mask == plain MSE (mean over every element)."""
    mask = torch.ones_like(_VALUE)
    tot, _ = Base.loss_hpweighted(_stub(), _PRED, _VALUE, _HEAD_INDEX, mask=mask)
    expected = F.mse_loss(_PRED[0], _VALUE)
    assert torch.allclose(tot, expected)


@pytest.mark.mpi_skip()
def pytest_masked_loss_partial_equals_mean_over_kept():
    """A partial mask averages squared error over the kept entries only."""
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tot, _ = Base.loss_hpweighted(_stub(), _PRED, _VALUE, _HEAD_INDEX, mask=mask)
    sq = (_PRED[0] - _VALUE) ** 2
    expected = (sq * mask).sum() / mask.sum()
    assert torch.allclose(tot, expected)
    # Sanity: with these values the masked result differs from the plain MSE,
    # so the mask is genuinely changing the objective.
    assert not torch.allclose(tot, F.mse_loss(_PRED[0], _VALUE))


@pytest.mark.mpi_skip()
def pytest_masked_loss_rejects_non_mse():
    """The masked path is MSE-only; a non-mse loss with a mask must error."""
    mask = torch.ones_like(_VALUE)
    with pytest.raises(AssertionError):
        Base.loss_hpweighted(
            _stub("smooth_l1"), _PRED, _VALUE, _HEAD_INDEX, mask=mask
        )
