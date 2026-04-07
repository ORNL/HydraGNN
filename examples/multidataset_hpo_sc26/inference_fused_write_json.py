#!/usr/bin/env python3
"""Fused HydraGNN + BranchWeightMLP inference – per-GPU JSON to node-local NVMe.

Each GPU writes a JSON file with atom types, coordinates, weighted formation
energy, forces, and per-structure branch softmax weights. Output paths use the
local GPU id to avoid collisions on multi-GPU nodes (e.g. OLCF Frontier
``/mnt/bb/$USER``).

Usage:
    srun ... python inference_fused_write_json.py \\
        --logdir <path_to_training_log_dir> \\
        --num_structures 100

The --logdir should contain config.json, a .pk HydraGNN checkpoint, and
``mlp_weights/`` with .pt checkpoints.
"""

import json
import os

from hydragnn.utils.distributed import get_comm_size_and_rank, get_local_rank

from inference_random_structures import build_argument_parser

from inference_fused import (
    add_fused_cli_arguments,
    generate_structures,
    load_fused_stack,
    print_fused_results,
    run_fused_inference,
)

from inference_random_structures_write_json import structure_to_dict


def main():
    parser = build_argument_parser(
        description="Fused HydraGNN + MLP inference – JSON output to NVMe"
    )
    add_fused_cli_arguments(parser)
    parser.set_defaults(num_structures=100)
    parser.add_argument(
        "--nvme_dir",
        type=str,
        default=None,
        help="NVMe output directory. Defaults to /mnt/bb/$USER",
    )
    args = parser.parse_args()

    _, world_rank = get_comm_size_and_rank()
    local_gpu_id = get_local_rank()

    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        _g,
        _m,
    ) = load_fused_stack(
        args.logdir,
        args.checkpoint,
        args.mlp_checkpoint,
        args.precision,
        args.mlp_precision,
        args.mlp_device,
    )
    print(f"[rank {world_rank}] local GPU id: {local_gpu_id}")

    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)

    structures = generate_structures(
        args.num_structures,
        args.min_atoms,
        args.max_atoms,
        args.box_size,
        args.max_atomic_number,
        radius,
        max_neighbours,
        args.seed + world_rank,
    )
    print(
        f"[rank {world_rank}] Generated {len(structures)} random structures "
        f"(atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    (
        all_energies,
        all_forces,
        all_natoms,
        all_weights,
        batch_latencies_ms,
        total_timed_structures,
        stage_stats,
    ) = run_fused_inference(
        model,
        mlp,
        structures,
        args.batch_size,
        param_dtype,
        autocast_ctx,
        device,
        num_branches,
        args.num_warmup,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        False,
    )

    json_entries = []
    for i in range(len(all_energies)):
        entry = structure_to_dict(structures[i], all_energies[i], all_forces[i])
        entry["structure_index"] = i
        entry["branch_weights"] = all_weights[i].tolist()
        json_entries.append(entry)

    json_payload = json.dumps(
        {"num_structures": len(json_entries), "structures": json_entries},
        indent=2,
    )

    nvme_base = args.nvme_dir
    if nvme_base is None:
        user = os.environ.get("USER", "unknown")
        nvme_base = f"/mnt/bb/{user}"

    os.makedirs(nvme_base, exist_ok=True)

    filename = f"inference_fused_results_gpu{local_gpu_id}.json"
    nvme_path = os.path.join(nvme_base, filename)

    with open(nvme_path, "w") as f:
        f.write(json_payload)

    print(f"[rank {world_rank}] Wrote {len(json_entries)} structures to {nvme_path}")

    print_fused_results(
        all_energies,
        all_forces,
        all_natoms,
        all_weights,
        num_branches,
        batch_latencies_ms,
        total_timed_structures,
        stage_stats,
    )


if __name__ == "__main__":
    main()
