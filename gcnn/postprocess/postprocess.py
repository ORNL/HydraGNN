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


def output_denormalize(y_minmax, true_values, predicted_values):
    # Fixme, should be improved later
    for ihead in range(len(y_minmax)):
        for isamp in range(len(predicted_values[0])):
            for iatom in range(len(predicted_values[ihead][0])):
                ymin = y_minmax[ihead][0][iatom]
                ymax = y_minmax[ihead][1][iatom]

                predicted_values[ihead][isamp][iatom] = (
                    predicted_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )
                true_values[ihead][isamp][iatom] = (
                    true_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )

    return true_values, predicted_values
