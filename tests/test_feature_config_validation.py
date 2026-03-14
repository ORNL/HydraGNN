import pytest
import torch
from hydragnn.utils.input_config_parsing.feature_config import (
    validate_node_feature_columns,
)


def test_validate_node_feature_columns_simple():
    # 2 features: dim=1 and dim=3, total columns=4
    node_feature_dims = [1, 3]
    data_x = torch.zeros((10, 4))
    # Should pass
    validate_node_feature_columns(data_x, node_feature_dims)
    # Should fail if data_x has extra columns

    data_x_extra = torch.zeros((10, 5))
    with pytest.raises(ValueError):
        validate_node_feature_columns(data_x_extra, node_feature_dims)


def test_validate_node_feature_columns_with_column_indices():
    # 2 features: dim=1 at col 0, dim=3 at col 2, total columns=5
    node_feature_dims = [1, 3]
    column_indices = [0, 2]
    data_x = torch.zeros((10, 5))
    # Should pass
    validate_node_feature_columns(data_x, node_feature_dims, column_indices)
    # Should fail if second feature overruns columns
    bad_column_indices = [0, 3]
    with pytest.raises(ValueError):
        validate_node_feature_columns(data_x, node_feature_dims, bad_column_indices)


def test_validate_node_feature_columns_edge_cases():
    # Single feature, dim=5
    node_feature_dims = [5]
    data_x = torch.zeros((10, 5))
    validate_node_feature_columns(data_x, node_feature_dims)
    # Should fail if not enough columns
    data_x_bad = torch.zeros((10, 4))
    with pytest.raises(ValueError):
        validate_node_feature_columns(data_x_bad, node_feature_dims)
