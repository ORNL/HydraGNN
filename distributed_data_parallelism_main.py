import os
import argparse


from run_config_input import run_normal_config_file


def main():
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    type_of_run = {
        3: run_normal_config_file,
    }

    type_of_run[3]()


if __name__ == "__main__":
    main()
