##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class StructuralAttentionContext:
    """Domain-neutral structural inputs consumed by a global attention block.

    Providers may supply one pairwise representation and/or structural
    coordinates. Tensor types are intentionally expressed as ``Any`` so this
    transport object does not impose a storage backend on preprocessors.
    """

    factorized_pairwise_features: Optional[Mapping[str, Mapping[str, Any]]] = None
    pairwise_features: Optional[Sequence[Any]] = None
    qk_coordinates: Optional[Any] = None
    qk_coefficient: Optional[Any] = None
