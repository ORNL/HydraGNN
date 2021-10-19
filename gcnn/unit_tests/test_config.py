import os, json
import pytest


@pytest.mark.parametrize("config_file", ["configuration.json"])
@pytest.mark.mpi_skip()
def pytest_config(config_file):

    config_file = os.path.join("examples", config_file)
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    expected = {
        "Dataset": [
            "name",
            "path",
            "format",
            "num_nodes",
            "node_features",
            "graph_features",
        ],
        "NeuralNetwork": ["Architecture", "Variables_of_interest", "Training"],
    }

    for category in expected.keys():
        assert category in config, "Missing required input category"

        for input in category:
            assert input in category, "Missing required input"
