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
import os
import pdb
import json
import torch
import torch.distributed as dist
import torch_geometric
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import argparse
from pathlib import Path

try:
    from .qm9_raw_processor import RobustQM9
except ImportError:
    from qm9_raw_processor import RobustQM9

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr

num_samples = 1000


# charge and spin are constant across QM9 dataset
charge = 0.0
spin = 1.0
QM9_CACHE_VERSION = "named-schema-v3-robust-raw"
QM9_LEGACY_CACHE_DIRECTORIES = (
    "named-schema-v1",
    "named-schema-v2-raw-subset",
)


# Update each sample prior to loading.
def qm9_pre_transform(data, transform):
    # LPE
    data = transform(data)
    data.atomic_numbers = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.free_energy = data.y[:, 10:11] / data.num_nodes
    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data


def qm9_pre_filter(data):
    return data.idx < num_samples


def build_qm9_from_raw(
    root,
    pre_transform,
    pre_filter=qm9_pre_filter,
    *,
    max_records=num_samples,
    invalid_molecule_policy="report_and_skip",
    max_rejected_molecules=None,
    report_directory=None,
):
    """Build QM9 from raw records with explicit rejection reporting."""
    return RobustQM9(
        root=root,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        max_records=max_records,
        invalid_molecule_policy=invalid_molecule_policy,
        max_rejected_molecules=max_rejected_molecules,
        report_directory=report_directory,
    )


def prepare_qm9_cache(cache_root, expected_version=QM9_CACHE_VERSION):
    """Keep one raw cache and invalidate only incompatible processed data."""
    hydragnn.utils.datasets.prepare_pyg_cache(
        cache_root, expected_version, QM9_LEGACY_CACHE_DIRECTORIES
    )


def mark_qm9_cache_current(cache_root, version=QM9_CACHE_VERSION):
    """Record the schema version after PyG finishes processing successfully."""
    hydragnn.utils.datasets.mark_pyg_cache_current(cache_root, version)


def validate_named_cache(dataset, variables, cache_root):
    """Reject a processed cache that predates the declared named schema."""
    schema = hydragnn.utils.input_config_parsing.parse_variable_schema(variables)
    sample = dataset[0]
    try:
        for spec in (*schema.inputs, *schema.outputs):
            hydragnn.utils.input_config_parsing.validate_variable(sample, spec)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Incompatible QM9 processed cache at '{cache_root}'. "
            f"Expected the {QM9_CACHE_VERSION} named-variable format. "
            "The processed cache must be rebuilt from the retained raw files."
        ) from error


def main(
    mpnn_type=None,
    global_attn_engine=None,
    global_attn_type=None,
    qm9_preprocess_all=False,
    qm9_invalid_molecule_policy="report_and_skip",
    qm9_max_rejected_molecules=None,
):
    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)

    # If a model type is provided, update the configuration accordingly.
    if global_attn_engine:
        config["NeuralNetwork"]["Architecture"][
            "global_attn_engine"
        ] = global_attn_engine

    if global_attn_type:
        config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type

    if mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type

    verbosity = config["Verbosity"]["level"]
    var_config = config["Variables"]

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    log_name = f"qm9_test_{mpnn_type}" if mpnn_type else "qm9_test"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # LPE
    transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    try:
        import rdkit  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The QM9 example requires RDKit so it can process the raw QM9 "
            "dataset; fallback to PyG's preprocessed artifact is disabled."
        ) from error

    cache_root = os.path.join("dataset", "qm9")
    cache_mode = "full" if qm9_preprocess_all else f"subset-{num_samples}"
    cache_version = f"{QM9_CACHE_VERSION}:{cache_mode}"

    def build_dataset():
        prepare_qm9_cache(cache_root, cache_version)
        result = build_qm9_from_raw(
            root=cache_root,
            pre_transform=lambda data: qm9_pre_transform(data, transform),
            pre_filter=None if qm9_preprocess_all else qm9_pre_filter,
            max_records=None if qm9_preprocess_all else num_samples,
            invalid_molecule_policy=qm9_invalid_molecule_policy,
            max_rejected_molecules=qm9_max_rejected_molecules,
            report_directory=Path(cache_root) / "preprocessing_report" / cache_mode,
        )
        mark_qm9_cache_current(cache_root, cache_version)
        return result

    dataset = hydragnn.preprocess.build_dataset_on_rank_zero(build_dataset)
    validate_named_cache(dataset, config["Variables"], cache_root)
    if qm9_preprocess_all:
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        train,
        val,
        test,
        config["NeuralNetwork"]["Training"]["batch_size"],
        variables=config["Variables"],
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )

    # Run training with the given model and qm9 datasets.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    tr.initialize()
    tr.disable()

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
    )

    tr.save(log_name)
    if writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the QM9 example with optional model type."
    )
    parser.add_argument(
        "--mpnn_type",
        type=str,
        default=None,
        help="Specify the model type for training (default: None).",
    )
    parser.add_argument(
        "--global_attn_engine",
        type=str,
        default=None,
        help="Specify if global attention is being used (default: None).",
    )
    parser.add_argument(
        "--global_attn_type",
        type=str,
        default=None,
        help="Specify the global attention type (default: None).",
    )
    parser.add_argument(
        "--qm9-preprocess-all",
        action="store_true",
        help="convert the complete raw QM9 archive, write reports, and exit",
    )
    parser.add_argument(
        "--qm9-invalid-molecule-policy",
        choices=("report_and_skip", "error"),
        default="report_and_skip",
    )
    parser.add_argument("--qm9-max-rejected-molecules", type=int, default=None)
    args = parser.parse_args()

    main(
        mpnn_type=args.mpnn_type,
        global_attn_engine=args.global_attn_engine,
        global_attn_type=args.global_attn_type,
        qm9_preprocess_all=args.qm9_preprocess_all,
        qm9_invalid_molecule_policy=args.qm9_invalid_molecule_policy,
        qm9_max_rejected_molecules=args.qm9_max_rejected_molecules,
    )
