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
"""Run DeepHyper over the shared robust QM9-HPO workflow."""

import argparse
import os

import hydragnn

try:
    from .workflow import load_base_config, prepare_splits, train_trial
except ImportError:
    from workflow import load_base_config, prepare_splits, train_trial


_worker_context = None


def _context():
    global _worker_context
    if _worker_context is None:
        os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
        hydragnn.utils.distributed.setup_ddp()
        config = load_base_config()
        _worker_context = config, prepare_splits(config)
    return _worker_context


def run(trial):
    """DeepHyper objective; its maximization convention requires negation."""
    config, splits = _context()
    validation_loss = train_trial(
        config, splits, dict(trial.parameters), f"qm9_deephyper_{trial.id}"
    )
    return -validation_loss


def main(max_evals=10, timeout=1200):
    try:
        from deephyper.evaluator import Evaluator
        from deephyper.hpo import CBO, HpProblem
    except ImportError as error:
        raise ImportError("Install deephyper to run qm9_deephyper.py") from error

    problem = HpProblem()
    problem.add_hyperparameter((1, 2), "num_conv_layers")
    problem.add_hyperparameter((1, 100), "hidden_dim")
    problem.add_hyperparameter((1, 3), "num_headlayers")
    problem.add_hyperparameter((1, 3), "dim_headlayers")
    problem.add_hyperparameter([2, 4, 8], "global_attn_heads")
    problem.add_hyperparameter(["EGNN", "PNA", "SchNet", "DimeNet"], "mpnn_type")

    evaluator = Evaluator.create(
        run, method="process", method_kwargs={"num_workers": 1}
    )
    search = CBO(problem, evaluator, random_state=42, log_dir="qm9_deephyper")
    print(search.search(max_evals=max_evals, timeout=timeout))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    main(args.max_evals, args.timeout)
