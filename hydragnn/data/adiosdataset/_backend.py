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
"""Optional ADIOS2 / DDStore imports and shared low-level helpers."""

import numpy as np

try:
    import adios2

    adios2_version = [int(x) for x in adios2.__version__.split(".")]
    if adios2_version[0] <= 2 and adios2_version[1] < 10:
        import adios2 as ad2
    else:
        import adios2.bindings as ad2
except ImportError:
    adios2 = None
    ad2 = None

try:
    import pyddstore as dds
except ImportError:
    dds = None

# ADIOS2 type names -> numpy dtypes
DTYPE_MAP = {
    "double": np.float64,
    "float": np.float32,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "uint8_t": np.uint8,
}


def adios_dtype(type_name):
    """Return the numpy dtype for an ADIOS2 variable type name."""
    try:
        return DTYPE_MAP[type_name]
    except KeyError:
        raise ValueError(type_name) from None


def parse_shape(shape):
    """Parse an ADIOS2 ``Shape`` string such as ``"10, 3"`` into a list of ints."""
    return [int(x.strip(",")) for x in shape.strip().split()]


def _comm_args(comm):
    """Return the communicator argument for ADIOS2, or none for serial builds."""
    if getattr(ad2, "is_built_with_mpi", True):
        return (comm,)
    if comm.Get_size() > 1:
        raise RuntimeError(
            "ADIOS2 was built without MPI support; it cannot be used with "
            f"{comm.Get_size()} MPI ranks."
        )
    return ()


def adios2_open(filename, mode, comm):
    adios2_version = [int(x) for x in ad2.__version__.split(".")]
    if adios2_version[0] <= 2 and adios2_version[1] < 10:
        f_adios2_open = ad2.open
    else:
        f_adios2_open = adios2.Stream
    return f_adios2_open(filename, mode, *_comm_args(comm))
