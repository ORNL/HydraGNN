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

import pytest
import torch
from e3nn import o3

from hydragnn.globalAtt.equivariant_features import (
    IrrepsFeatureAdapter,
    ScalarIrrepsAdapter,
    ScalarVectorIrrepsAdapter,
    create_local_feature_adapter,
)


def test_scalar_vector_adapter_round_trip():
    adapter = ScalarVectorIrrepsAdapter(channels=4)
    scalars = torch.randn(5, 4)
    vectors = torch.randn(5, 3, 4)

    encoded = adapter(scalars, vectors)
    decoded_scalars, decoded_vectors = adapter.decode(encoded)

    assert adapter.irreps == o3.Irreps("4x0e + 4x1o")
    assert encoded.shape == (5, adapter.irreps.dim)
    assert torch.equal(decoded_scalars, scalars)
    assert torch.equal(decoded_vectors, vectors)


def test_scalar_vector_adapter_is_rotation_equivariant():
    adapter = ScalarVectorIrrepsAdapter(channels=3).double()
    scalars = torch.randn(7, 3, dtype=torch.float64)
    vectors = torch.randn(7, 3, 3, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)

    rotated_vectors = torch.einsum("ij,njc->nic", rotation, vectors)
    encoded_after_rotation = adapter(scalars, rotated_vectors)

    representation = adapter.irreps.D_from_matrix(rotation)
    rotated_after_encoding = adapter(scalars, vectors) @ representation.T

    torch.testing.assert_close(
        encoded_after_rotation,
        rotated_after_encoding,
        # e3nn obtains D matrices through an angle decomposition whose
        # numerical error is larger than the feature-layout conversion.
        rtol=1.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    ("scalars", "vectors", "message"),
    [
        (torch.randn(2, 3, 1), torch.randn(2, 3, 3), "inv_node_feat"),
        (torch.randn(2, 3), torch.randn(2, 3, 2), "equiv_node_feat"),
        (torch.randn(2, 3), torch.randn(3, 3, 3), "equiv_node_feat"),
    ],
)
def test_scalar_vector_adapter_rejects_invalid_shapes(scalars, vectors, message):
    adapter = ScalarVectorIrrepsAdapter(channels=3)

    with pytest.raises(ValueError, match=message):
        adapter(scalars, vectors)


def test_scalar_vector_adapter_rejects_mixed_dtypes():
    adapter = ScalarVectorIrrepsAdapter(channels=3)

    with pytest.raises(ValueError, match="same dtype"):
        adapter(torch.randn(2, 3), torch.randn(2, 3, 3, dtype=torch.float64))


def test_scalar_vector_adapter_rejects_invalid_encoded_width():
    adapter = ScalarVectorIrrepsAdapter(channels=3)

    with pytest.raises(ValueError, match="12 entries per node"):
        adapter.decode(torch.randn(2, 11))


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_local_models_require_explicit_limited_mode(mpnn_type):
    with pytest.raises(ValueError, match="cannot provide tensor-valued"):
        create_local_feature_adapter(mpnn_type, channels=4, allow_scalar_only=True)

    with pytest.raises(ValueError, match="allow_scalar_only"):
        create_local_feature_adapter(
            mpnn_type, channels=4, require_tensor_coupling=False
        )

    with pytest.warns(UserWarning, match="scalar-only"):
        adapter = create_local_feature_adapter(
            mpnn_type,
            channels=4,
            allow_scalar_only=True,
            require_tensor_coupling=False,
        )

    assert isinstance(adapter, ScalarIrrepsAdapter)
    assert adapter.irreps == o3.Irreps("4x0e")
    features = torch.randn(3, 4)
    assert adapter.decode(adapter(features)) is features


def test_schnet_scalar_mode_rejects_coordinate_updates():
    with pytest.raises(ValueError, match="coordinate updates"):
        create_local_feature_adapter(
            "SchNet",
            channels=4,
            allow_scalar_only=True,
            require_tensor_coupling=False,
            local_equivariance=True,
        )


def test_equivariant_local_models_use_scalar_vector_adapter_without_opt_in():
    for mpnn_type in ("PAINN", "PNAEq"):
        adapter = create_local_feature_adapter(mpnn_type, channels=4)
        assert isinstance(adapter, ScalarVectorIrrepsAdapter)


def test_mace_irreps_adapter_round_trip_and_rotation():
    irreps = o3.Irreps("3x0e + 3x1o + 2x2e")
    adapter = IrrepsFeatureAdapter(irreps)
    features = torch.randn(5, irreps.dim, dtype=torch.float64)
    scalars, tensors = adapter.decode(features)

    assert scalars.shape == (5, 3)
    assert tensors.shape == (5, irreps.dim - 3)
    assert torch.equal(adapter(scalars, tensors), features)

    rotation = o3.rand_matrix(dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)
    transformed = features @ representation.T
    transformed_scalars, transformed_tensors = adapter.decode(transformed)
    torch.testing.assert_close(transformed_scalars, scalars)
    torch.testing.assert_close(
        adapter(transformed_scalars, transformed_tensors), transformed
    )


def test_mace_irreps_adapter_rejects_nonleading_scalars():
    with pytest.raises(ValueError, match="precede tensor irreps"):
        IrrepsFeatureAdapter("1x1o + 1x0e")


def test_mace_parallel_input_zero_initializes_absent_tensor_irreps():
    adapter = IrrepsFeatureAdapter("2x0e + 2x1o")
    scalars = torch.randn(3, 2)
    features = adapter.encode_parallel_input(scalars, torch.empty(3, 0))

    torch.testing.assert_close(features[:, :2], scalars)
    torch.testing.assert_close(features[:, 2:], torch.zeros(3, 6))

    with pytest.raises(ValueError, match="match.*or be absent"):
        adapter.encode_parallel_input(scalars, torch.randn(3, 3))
