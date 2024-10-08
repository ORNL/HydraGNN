###########################################################################################
# __init__ file for Modules
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
# Taken From:
# GitHub: https://github.com/ACEsuit/mace
# ArXiV: https://arxiv.org/pdf/2206.07697
# Date: August 27, 2024  |  12:37 (EST)
###########################################################################################

from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    ScaleShiftBlock,
)

from .radial import BesselBasis, GaussianBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    "SymmetricContraction",
    "interaction_classes",
]
