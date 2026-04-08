#!/usr/bin/env python3
"""Fused HydraGNN + BranchWeightMLP inference – per-GPU JSON to node-local NVMe.

Each GPU writes a JSON file with atom types, coordinates, weighted formation
energy, forces, and per-structure branch softmax weights. Output paths use the
local GPU id to avoid collisions on multi-GPU nodes (e.g. OLCF Frontier
``/mnt/bb/$USER``).

NVMe writes are **pipelined** with GPU inference: a single-threaded
``ThreadPoolExecutor`` serialises and flushes each batch's results to NVMe
while the next batch is already being processed on the GPU.

Usage:
    srun ... python inference_fused_write_json.py \\
        --logdir <path_to_training_log_dir> \\
        --num_structures 100

The --logdir should contain config.json, a .pk HydraGNN checkpoint, and
``mlp_weights/`` with .pt checkpoints.
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import IO, List

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


# ---------------------------------------------------------------------------
# Background NVMe writer
# ---------------------------------------------------------------------------


def _write_batch_entries(
    fh: IO[str],
    structs: list,
    energies: list,
    forces: list,
    weights: list,
    global_offset: int,
    need_leading_comma: bool,
) -> None:
    """Serialise one batch's results and stream them to the open file handle.

    Runs in a background thread so that NVMe I/O overlaps with the next
    batch's GPU inference.
    """
    for i in range(len(energies)):
        entry = structure_to_dict(structs[i], energies[i], forces[i])
        entry["structure_index"] = global_offset + i
        entry["branch_weights"] = weights[i].tolist()

        prefix = ",\n    " if (need_leading_comma or i > 0) else "    "
        fh.write(prefix)
        json.dump(entry, fh)
    fh.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # --- Open NVMe output and begin streaming JSON ---
    nvme_base = args.nvme_dir
    if nvme_base is None:
        user = os.environ.get("USER", "unknown")
        nvme_base = f"/mnt/bb/{user}"

    os.makedirs(nvme_base, exist_ok=True)
    filename = f"inference_fused_results_gpu{local_gpu_id}.json"
    nvme_path = os.path.join(nvme_base, filename)

    fh = open(nvme_path, "w")
    fh.write('{"structures": [\n')

    executor = ThreadPoolExecutor(max_workers=1)
    futures: List[Future] = []
    struct_offset = [0]

    def _on_batch(batch_idx, batch_energies, batch_forces, batch_natoms, batch_weights):
        """Called from run_fused_inference after each batch is extracted to CPU.

        Submits the serialisation + NVMe write to the background thread so
        that it overlaps with GPU processing of the next batch.
        """
        offset = struct_offset[0]
        n = len(batch_energies)
        structs_slice = structures[offset : offset + n]
        struct_offset[0] += n

        fut = executor.submit(
            _write_batch_entries,
            fh,
            structs_slice,
            batch_energies,
            batch_forces,
            batch_weights,
            offset,
            need_leading_comma=(offset > 0),
        )
        futures.append(fut)

    print(
        f"[rank {world_rank}] NVMe write pipeline enabled – "
        f"background thread will stream results to {nvme_path}"
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
        args.profile_stages,
        encoder_reuse=args.encoder_reuse,
        num_streams=args.num_streams,
        weight_threshold=args.weight_threshold,
        fused_energy_grad=args.fused_energy_grad,
        per_batch_callback=_on_batch,
    )

    # --- Wait for all background NVMe writes, close the JSON file ---
    for fut in futures:
        fut.result()
    executor.shutdown(wait=True)

    total_written = struct_offset[0]
    fh.write(f'\n], "num_structures": {total_written}}}\n')
    fh.close()

    print(
        f"[rank {world_rank}] Wrote {total_written} structures to {nvme_path} "
        f"(pipelined with GPU inference)"
    )

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
