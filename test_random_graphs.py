import os, json
import pytest

from run_config_input import run_normal_config_file
from utils.random_graph_data import random_graph_data
from utils.utils import get_comm_size_and_rank
from test_trained_model import test_trained_model


@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNN"])
def pytest_train_model(model_type):

    _, rank = get_comm_size_and_rank()

    if rank == 0:
        random_graph_data()

    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = "./examples/ci.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    tmp_file = "./tmp.json"
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    with open(tmp_file, "w") as f:
        json.dump(config, f)

    run_normal_config_file(tmp_file)

    error, true_values, predicted_values = test_trained_model(tmp_file, model_type)

    for true_value, predicted_value in zip(true_values, predicted_values):
        assert abs(true_value[0] - predicted_value[0]) < 1e-3
