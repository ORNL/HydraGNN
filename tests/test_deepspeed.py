import pytest

from tests.test_graphs import unittest_train_model

# Check if deepspeed is available
try:
    import deepspeed

    deepspeed_available = True
except ImportError:
    deepspeed_available = False


# Test vector output
@pytest.mark.parametrize("model_type", ["PNA"])
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.deepspeed
@pytest.mark.skipif(not deepspeed_available, reason="deepspeed package not installed")
def pytest_train_model_vectoroutput_w_deepspeed(model_type, overwrite_data=False):
    unittest_train_model(
        model_type,
        None,
        None,
        "ci_vectoroutput.json",
        True,
        overwrite_data,
        use_deepspeed=True,
    )


# Test vector output with global attention mechanisms
@pytest.mark.parametrize(
    "global_attn_engine",
    ["GPS"],
)
@pytest.mark.parametrize("global_attn_type", ["multihead"])
@pytest.mark.parametrize("model_type", ["PNA"])
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.deepspeed
@pytest.mark.skipif(not deepspeed_available, reason="deepspeed package not installed")
def pytest_train_model_vectoroutput_w_deepspeed_global_attention(
    model_type, global_attn_engine, global_attn_type, overwrite_data=False
):
    unittest_train_model(
        model_type,
        global_attn_engine,
        global_attn_type,
        "ci_vectoroutput.json",
        True,
        overwrite_data,
        use_deepspeed=True,
    )


# # Test deepspeed zero
# # cannot use deepspeed zero-optimizer with gloo backend on the server
# @pytest.mark.parametrize("model_type", ["PNA"])
# @pytest.mark.parametrize("zero_stage", [1, 2, 3])
# @pytest.mark.mpi
# def pytest_train_model_vectoroutput_w_deepspeed_zero(
#     model_type, zero_stage, overwrite_data=False
# ):
#     overwrite_config = {
#         "NeuralNetwork": {
#             "ds_config": {
#                 "zero_optimization": {
#                     "stage": zero_stage,
#                 }
#             }
#         }
#     }
#     unittest_train_model(
#         model_type,
#         f"ci_vectoroutput.json",
#         True,
#         overwrite_data,
#         use_deepspeed=True,
#         overwrite_config=overwrite_config,
#     )
