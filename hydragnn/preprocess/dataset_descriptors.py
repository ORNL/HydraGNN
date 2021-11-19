##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

from enum import Enum


class AtomFeatures(Enum):
    """Class is an enum that represents features of an atom. Values paired with names of features represent column
    indexes for each feature that are used in referencing them throughout the project.
    """

    NUM_OF_PROTONS = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2


class StructureFeatures(Enum):
    """Class is an enum that represents features of a structure. Values paired with names of features represent column
    indexes for each feature that are used in referencing them throughout the project.
    """

    FREE_ENERGY = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2
