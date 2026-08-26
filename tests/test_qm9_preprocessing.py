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
from rdkit import Chem

from examples.qm9 import qm9


def test_qm9_raw_builder_limits_supplier_before_pyg_processing(monkeypatch):
    original_supplier = Chem.SDMolSupplier
    monkeypatch.setattr(Chem, "SDMolSupplier", lambda *args, **kwargs: range(5000))

    observed = {}

    def fake_qm9(**kwargs):
        observed["records"] = list(Chem.SDMolSupplier("raw.sdf"))
        return kwargs

    monkeypatch.setattr(qm9.torch_geometric.datasets, "QM9", fake_qm9)
    result = qm9.build_qm9_from_raw("cache", pre_transform=lambda data: data)

    assert len(observed["records"]) == qm9.num_samples
    assert result["root"] == "cache"
    assert Chem.SDMolSupplier is not original_supplier
