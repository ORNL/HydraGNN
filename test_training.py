import os, json
import pytest

from run_config_input import run_normal_config_file


@pytest.mark.mpi()
@pytest.mark.parametrize("model_type", ["GIN", "GAT", "MFC", "PNN"])
def pytest_train_model(model_type):
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Read in config settings and override model type.
    config_file = "./examples/ci.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    tmp_file = "./tmp.json"
    config["model_type"] = model_type
    with open(tmp_file, "w") as f:
        json.dump(config, f)

    run_normal_config_file(tmp_file)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)
