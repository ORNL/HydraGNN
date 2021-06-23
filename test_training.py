import os
import pytest

from run_config_input import run_normal_config_file


@pytest.mark.mpi()
def pytest_train_model():
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    run_normal_config_file("./examples/ci.json")
