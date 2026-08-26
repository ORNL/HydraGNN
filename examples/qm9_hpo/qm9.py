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
"""Train one QM9 configuration through the shared HPO workflow."""

import argparse
import os

import torch.distributed as dist

import hydragnn

try:
    from .workflow import load_base_config, prepare_splits, train_trial
except ImportError:
    from workflow import load_base_config, prepare_splits, train_trial


def main(parameters=None, log_name="qm9_hpo"):
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
    hydragnn.utils.distributed.setup_ddp()
    config = load_base_config()
    splits = prepare_splits(config)
    validation_loss = train_trial(config, splits, parameters or {}, log_name)
    print(f"Validation Loss: {validation_loss}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return validation_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpnn_type", "--model_type", dest="mpnn_type")
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_conv_layers", type=int)
    parser.add_argument("--num_headlayers", type=int)
    parser.add_argument("--dim_headlayers", type=int)
    parser.add_argument("--global_attn_heads", type=int)
    parser.add_argument("--log", default="qm9_hpo")
    arguments = vars(parser.parse_args())
    log_name = arguments.pop("log")
    main(
        {name: value for name, value in arguments.items() if value is not None},
        log_name,
    )
