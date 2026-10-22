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
"""Keep pre-restructuring ``hydragnn.utils.*`` import paths working."""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

# Longest prefixes first so the most specific match wins.
MOVED_MODULES = {
    "hydragnn.utils.model.uma._vendored": "hydragnn.third_party.uma",
    "hydragnn.utils.model.uma": "hydragnn.third_party",
    "hydragnn.utils.model.allscaip": "hydragnn.models.backends.allscaip",
    "hydragnn.utils.model.escaip": "hydragnn.models.backends.escaip",
    "hydragnn.utils.model.mace_utils": "hydragnn.models.backends.mace_utils",
    "hydragnn.utils.descriptors_and_embeddings": "hydragnn.data.descriptors_and_embeddings",
    "hydragnn.utils.materials": "hydragnn.data.materials",
    "hydragnn.utils.lsms": "hydragnn.data.lsms",
    "hydragnn.utils.datasets": "hydragnn.data",
    "hydragnn.utils.distributed": "hydragnn.distributed",
    "hydragnn.utils.input_config_parsing": "hydragnn.config",
}


def new_module_name(fullname):
    for old, new in MOVED_MODULES.items():
        if fullname == old or fullname.startswith(old + "."):
            return new + fullname[len(old) :]
    return None


class _MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if new_module_name(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        new = new_module_name(module.__name__)
        warnings.warn(
            f"{module.__name__} has moved to {new}; the old path will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        # The import system re-reads sys.modules after exec, so this aliases the
        # old name to the real module without touching the real module's __spec__.
        sys.modules[module.__name__] = importlib.import_module(new)


def install():
    if not any(isinstance(f, _MovedModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MovedModuleFinder())
