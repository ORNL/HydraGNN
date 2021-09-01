import sys, os, json
import pytest

from run_config_input import run_normal_config_file
from utils.deterministic_graph_data import deterministic_graph_data
from utils.utils import get_comm_size_and_rank
from test_trained_model import test_trained_model

import torch

torch.manual_seed(0)


@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNN"])
@pytest.mark.parametrize("ci_input", ["ci.json", "ci_multihead.json"])
def pytest_train_model(model_type, ci_input):

    _, rank = get_comm_size_and_rank()

    if rank == 0:
        deterministic_graph_data()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = "./examples/" + ci_input
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    tmp_file = "./tmp.json"
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    with open(tmp_file, "w") as f:
        json.dump(config, f)

    run_normal_config_file(tmp_file)

    (
        error,
        error_sumofnodes_task,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = test_trained_model(tmp_file, model_type)

    # Set RMSE and sample error thresholds
    thresholds = {
        "PNN": [0.02, 0.10],
        "MFC": [0.05, 0.20],
        "GIN": [0.08, 0.20],
        "GAT": [0.05, 0.20],
    }
    for ihead in range(len(true_values)):
        error_head_sum = error_sumofnodes_task[ihead] / len(true_values[ihead][0])
        assert error_head_sum < 0.05, (
            "RMSE checking failed for sum of head "
            + str(ihead)
            + "! "
            + str(error_head_sum)
        )
        error_head_rmse = error_rmse_task[ihead]
        assert error_head_rmse < 0.05, (
            "RMSE checking failed for components of head "
            + str(ihead)
            + "! "
            + str(error_head_rmse)
        )
        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        # Check individual samples
        for true_value, predicted_value in zip(head_true, head_pred):
            for idim in range(len(true_value)):
                assert (
                    abs(true_value[idim] - predicted_value[idim]) < thresholds[model_type][1]
                ), "Samples checking failed!" + str(
                    abs(true_value[idim] - predicted_value[idim])
                )
    # Check RMSE error
    assert error < thresholds[model_type][0], "RMSE checking failed!" + str(error)

if __name__ == "__main__":
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    pytest_train_model(sys.argv[1], "ci.json")
