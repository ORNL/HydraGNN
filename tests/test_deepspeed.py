import os, json
import pytest

import torch
import random
import hydragnn
from .test_graphs import unittest_train_model


# Test vector output
@pytest.mark.parametrize("model_type", ["PNA"])
def pytest_train_model_vectoroutput_w_deepspeed(model_type, overwrite_data=False):
    unittest_train_model(
        model_type, "ci_vectoroutput.json", True, overwrite_data, use_deepspeed=True
    )


# # Test deepspeed zero
# @pytest.mark.parametrize("model_type", ["PNA"])
# @pytest.mark.parametrize("zero_stage", ["1", "2", "3"])
# def pytest_train_model_vectoroutput_w_deepspeed_zero(
#     model_type, zero_stage, overwrite_data=False
# ):
#     unittest_train_model(
#         model_type,
#         f"ci_vectoroutput_deepspeed_zero_{zero_stage}.json",
#         True,
#         overwrite_data,
#         use_deepspeed=True,
#     )
