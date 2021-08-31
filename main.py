import os, sys

from run_hyperparam_opt import run_with_hyperparameter_optimization
from run_config_input import run_normal_config_file


def main(choice):
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    type_of_run = {
        1: run_with_hyperparameter_optimization,
        2: run_normal_config_file,
    }
    type_of_run[int(choice)]()


if __name__ == "__main__":
    print(
        "Select the type of run: 1) hyperparameter optimization and 2) normal run with configuration file input"
    )
    if len(sys.argv) < 2:
        print("Error: must make a choice")
    else:
        main(sys.argv[1])
