import os

from run_hyperparam_opt import run_with_hyperparameter_optimization
from run_config_input import run_normal_config_file, run_normal_terminal_input


def main():
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    type_of_run = {
        1: run_with_hyperparameter_optimization,
        2: run_normal_terminal_input,
        3: run_normal_config_file,
    }

    choice = int(
        input(
            "Select the type of run between hyperparameter optimization, normal run with configuration input from terminal and normal run with configuration input from a file: 1)Hyperopt 2)Normal(terminal input) 3)Normal(config file) "
        )
    )
    type_of_run[choice]()


if __name__ == "__main__":
    main()
