import sys, os, json
import pytest

from run_config_input import run_normal_config_file
from utils.deterministic_graph_data import deterministic_graph_data
from utils.utils import get_comm_size_and_rank
from test_trained_model import test_trained_model

import torch

torch.manual_seed(0)


@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNN"])
def pytest_train_model(model_type):

    _, rank = get_comm_size_and_rank()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = "./examples/ci.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    if rank == 0:
        deterministic_graph_data(number_atoms=config["num_node"])


    tmp_file = "./tmp.json"
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    with open(tmp_file, "w") as f:
        json.dump(config, f)

    run_normal_config_file(tmp_file)

    error, true_values, predicted_values = test_trained_model(tmp_file, model_type)

    # Set RMSE and sample error thresholds
    thresholds = {
        "PNN": [0.02, 0.10],
        "MFC": [0.05, 0.20],
        "GIN": [0.08, 0.20],
        "GAT": [0.05, 0.20],
    }
    # Check RMSE error
    assert error < thresholds[model_type][0], "RMSE checking failed!" + str(error)
    # Check individual samples
    for true_value, predicted_value in zip(true_values, predicted_values):
        assert (
            abs(true_value[0] - predicted_value[0]) < thresholds[model_type][1]
        ), "Samples checking failed!" + str(abs(true_value[0] - predicted_value[0]))


if __name__ == "__main__":
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    pytest_train_model(sys.argv[1])
