##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Run Optuna over the shared robust QM9-HPO workflow."""

import argparse
import csv
import os

import torch.distributed as dist

import hydragnn

try:
    from .workflow import load_base_config, prepare_splits, train_trial
except ImportError:
    from workflow import load_base_config, prepare_splits, train_trial


def suggest_parameters(trial):
    num_headlayers = trial.suggest_int("num_headlayers", 1, 3)
    parameters = {
        "mpnn_type": trial.suggest_categorical("mpnn_type", ["EGNN", "PNA", "SchNet"]),
        "hidden_dim": trial.suggest_int("hidden_dim", 50, 300),
        "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 5),
        "num_headlayers": num_headlayers,
    }
    for index in range(num_headlayers):
        parameters[f"dim_headlayer_{index}"] = trial.suggest_int(
            f"dim_headlayer_{index}", 50, 300
        )
    return parameters


def write_results(study, destination):
    fieldnames = ["trial", "state", "validation_loss", "parameters"]
    with open(destination, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(
                {
                    "trial": trial.number,
                    "state": trial.state.name,
                    "validation_loss": trial.value,
                    "parameters": trial.params,
                }
            )


def main(num_trials=5):
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Install optuna to run qm9_optuna.py") from error

    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
    hydragnn.utils.distributed.setup_ddp()
    base_config = load_base_config()
    splits = prepare_splits(base_config)

    def objective(trial):
        return train_trial(
            base_config,
            splits,
            suggest_parameters(trial),
            f"qm9_optuna_{trial.number}",
        )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=num_trials)
    write_results(study, "hpo_results.csv")
    print("Best Hyperparameters:", study.best_params)
    print("Best Validation Loss:", study.best_value)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=5)
    args = parser.parse_args()
    main(args.num_trials)
