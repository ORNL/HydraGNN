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
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def assert_is_instance(obj: object, cls: type[_T]) -> _T:
    if obj and not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj


def none_throws(x: _T | None, msg: str | None = None) -> _T:
    if x is None:
        if msg:
            raise ValueError(msg)
        raise ValueError("x cannot be None")
    return x
