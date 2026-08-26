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
"""Shared, named-variable-aware QM9 workflow for HPO backends."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE

import hydragnn

try:
    from examples.qm9.qm9 import (
        QM9_CACHE_VERSION,
        build_qm9_from_raw,
        mark_qm9_cache_current,
        num_samples,
        prepare_qm9_cache,
        qm9_pre_filter,
        qm9_pre_transform,
        validate_named_cache,
    )
except ModuleNotFoundError:
    # Direct execution places examples/qm9_hpo, not the repository root, on
    # sys.path. Add the root so this adapter still reuses the primary workflow.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.qm9.qm9 import (
        QM9_CACHE_VERSION,
        build_qm9_from_raw,
        mark_qm9_cache_current,
        num_samples,
        prepare_qm9_cache,
        qm9_pre_filter,
        qm9_pre_transform,
        validate_named_cache,
    )

CONFIG_PATH = Path(__file__).with_name("qm9.json")
QM9_HPO_CACHE_VERSION = f"{QM9_CACHE_VERSION}:subset-{num_samples}"


def load_base_config():
    """Load a fresh configuration so trial mutations cannot leak."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def prepare_splits(config, cache_root="dataset/qm9"):
    """Prepare the shared raw-QM9 subset once and return deterministic splits."""
    transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )

    def build_dataset():
        prepare_qm9_cache(cache_root, QM9_HPO_CACHE_VERSION)
        result = build_qm9_from_raw(
            root=cache_root,
            pre_transform=lambda data: qm9_pre_transform(data, transform),
            pre_filter=qm9_pre_filter,
            max_records=num_samples,
            report_directory=Path(cache_root) / "preprocessing_report" / "subset-1000",
        )
        mark_qm9_cache_current(cache_root, QM9_HPO_CACHE_VERSION)
        return result

    dataset = hydragnn.preprocess.build_dataset_on_rank_zero(build_dataset)
    validate_named_cache(dataset, config["Variables"], cache_root)
    return hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )


def configure_trial(base_config, parameters):
    """Return an isolated config populated from backend-neutral parameters."""
    config = deepcopy(base_config)
    architecture = config["NeuralNetwork"]["Architecture"]
    mpnn_type = parameters.get("mpnn_type", parameters.get("model_type"))
    if mpnn_type is not None:
        architecture["mpnn_type"] = mpnn_type
    for name in ("num_conv_layers", "global_attn_heads"):
        if parameters.get(name) is not None:
            architecture[name] = int(parameters[name])

    hidden_dim = parameters.get("hidden_dim")
    if hidden_dim is not None:
        heads = parameters.get("global_attn_heads")
        architecture["hidden_dim"] = int(hidden_dim) * (int(heads) if heads else 1)

    num_headlayers = parameters.get("num_headlayers")
    if num_headlayers is not None:
        if "dim_headlayers" in parameters:
            dimensions = [int(parameters["dim_headlayers"])] * int(num_headlayers)
        else:
            dimensions = [
                int(parameters[f"dim_headlayer_{index}"])
                for index in range(int(num_headlayers))
            ]
        for head in architecture["output_heads"].values():
            head["num_headlayers"] = int(num_headlayers)
            head["dim_headlayers"] = dimensions
    return config


def create_trial_loaders(splits, config):
    """Build loaders from the shared splits using the current trial config."""
    return hydragnn.preprocess.create_dataloaders(
        *splits,
        config["NeuralNetwork"]["Training"]["batch_size"],
        variables=config["Variables"],
    )


def train_trial(base_config, splits, parameters, log_name):
    """Train one isolated trial and return its scalar validation loss."""
    config = configure_trial(base_config, parameters)
    loaders = create_trial_loaders(splits, config)
    train_loader, val_loader, test_loader = loaders
    config = hydragnn.utils.input_config_parsing.update_config(config, *loaders)
    verbosity = config["Verbosity"]["level"]
    hydragnn.utils.print.print_utils.setup_log(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity
    )
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    try:
        hydragnn.train.train_validate_test(
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            writer,
            scheduler,
            config,
            log_name,
            verbosity,
            create_plots=False,
        )
        validation_loss, _ = hydragnn.train.validate(
            val_loader,
            model,
            verbosity,
            num_tasks=model.module.num_heads,
            reduce_ranks=True,
        )
        return float(validation_loss.detach().cpu())
    finally:
        if writer is not None:
            writer.close()
