##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import tempfile

import torch

import hydragnn


def _write_checkpoint(root: str, model_name: str, state_dict: dict):
    model_dir = os.path.join(root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, model_name + ".pk")
    torch.save({"model_state_dict": state_dict}, ckpt_path)
    return ckpt_path


def pytest_load_existing_model_strips_module_prefix():
    """
    Checkpoint saved from DDP/DataParallel has "module." prefix, but inference model is unwrapped.
    """
    base = torch.nn.Linear(4, 3)
    ddp_like_state = {"module." + k: v.clone() for k, v in base.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        model_name = "tmp_model"
        _write_checkpoint(tmp, model_name, ddp_like_state)

        inference_model = torch.nn.Linear(4, 3)
        hydragnn.utils.model.load_existing_model(inference_model, model_name, path=tmp)


def pytest_load_existing_model_adds_module_prefix():
    """
    Checkpoint saved from unwrapped model has no prefix, but current model expects "module." keys
    (simulating DDP/DataParallel wrapper behavior).
    """

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(4, 3)

    base = torch.nn.Linear(4, 3)
    unwrapped_state = {k: v.clone() for k, v in base.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        model_name = "tmp_model"
        _write_checkpoint(tmp, model_name, unwrapped_state)

        wrapped_model = Wrapper()
        hydragnn.utils.model.load_existing_model(wrapped_model, model_name, path=tmp)


