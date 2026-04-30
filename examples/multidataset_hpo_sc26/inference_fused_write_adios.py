#!/usr/bin/env python3
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
"""Fused HydraGNN + BranchWeightMLP inference – per-GPU ADIOS2 BP5 to node-local NVMe.

Each GPU writes a BP5 file where every ADIOS step is one atomistic structure.
Variables written per step:

    atom_types         int32  [N]      – atomic numbers
    coordinates_x/y/z float64 [N]     – positions (Å)
    forces_x/y/z       float64 [N]     – weighted-average forces (eV/Å)
    formation_energy   float64 [1]     – weighted-average total energy (eV)
    branch_weights     float64 [16]    – MLP softmax weights for this structure

Output files are named ``inference_fused_results_gpu{local_gpu_id}.bp`` and
placed under ``--nvme_dir`` (default: /mnt/bb/$USER) so that each GPU on a
node writes to a separate file with no contention.

ADIOS2 writes are **pipelined** with GPU inference exactly as in the JSON
variant: a single-threaded ThreadPoolExecutor owns the ADIOS writer for its
lifetime and flushes each batch while the next batch runs on the GPU.

Usage:
    srun ... python inference_fused_write_adios.py \\
        --logdir <path_to_training_log_dir> \\
        --num_structures 100 \\
        --nvme_dir /mnt/bb/$USER
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List

import numpy as np
import adios2.bindings as adios2

from hydragnn.utils.distributed import get_comm_size_and_rank, get_local_rank

from inference_random_structures import build_argument_parser

from inference_fused import (
    add_fused_cli_arguments,
    generate_structures,
    load_fused_stack,
    print_fused_results,
    run_fused_inference,
)


# ---------------------------------------------------------------------------
# ADIOS2 setup
# ---------------------------------------------------------------------------

NUM_BRANCHES = 16  # must match MLP output dimension


def _open_adios_writer(nvme_path: str):
    """Open a BP5 writer and define all variables with dummy shape [1].

    Variables are resized per step via SetShape + SetSelection.
    Returns (adios_obj, io_obj, writer, vars_dict).
    """
    a = adios2.ADIOS()
    io = a.DeclareIO("writer")
    io.SetEngine("BP5")
    writer = io.Open(nvme_path, adios2.Mode.Write)

    dummy_f = np.zeros(1, dtype=np.float64)
    dummy_i = np.zeros(1, dtype=np.int32)

    vars = {
        "atom_types": io.DefineVariable("atom_types", dummy_i, [1], [0], [1]),
        "coordinates_x": io.DefineVariable("coordinates_x", dummy_f, [1], [0], [1]),
        "coordinates_y": io.DefineVariable("coordinates_y", dummy_f, [1], [0], [1]),
        "coordinates_z": io.DefineVariable("coordinates_z", dummy_f, [1], [0], [1]),
        "forces_x": io.DefineVariable("forces_x", dummy_f, [1], [0], [1]),
        "forces_y": io.DefineVariable("forces_y", dummy_f, [1], [0], [1]),
        "forces_z": io.DefineVariable("forces_z", dummy_f, [1], [0], [1]),
        "formation_energy": io.DefineVariable(
            "formation_energy", dummy_f, [1], [0], [1]
        ),
        "branch_weights": io.DefineVariable(
            "branch_weights",
            np.zeros(NUM_BRANCHES, dtype=np.float64),
            [NUM_BRANCHES],
            [0],
            [NUM_BRANCHES],
        ),
    }
    return a, io, writer, vars


def _write_structure_step(
    writer, vars, atom_types, coords, forces, energy, branch_weights
):
    """Write one structure as one ADIOS2 step.

    All per-atom arrays are resized to N via SetShape + SetSelection.
    branch_weights is fixed width (NUM_BRANCHES) and never resized.
    """
    N = atom_types.shape[0]

    # Resize per-atom variables to current atom count
    per_atom = [
        ("atom_types", atom_types),
        ("coordinates_x", coords[:, 0]),
        ("coordinates_y", coords[:, 1]),
        ("coordinates_z", coords[:, 2]),
        ("forces_x", forces[:, 0]),
        ("forces_y", forces[:, 1]),
        ("forces_z", forces[:, 2]),
    ]
    for name, arr in per_atom:
        vars[name].SetShape([N])
        vars[name].SetSelection([[0], [N]])

    writer.BeginStep()
    for name, arr in per_atom:
        writer.Put(vars[name], arr)
    writer.Put(vars["formation_energy"], energy)
    writer.Put(vars["branch_weights"], branch_weights)
    writer.EndStep()


# ---------------------------------------------------------------------------
# Background NVMe writer (runs in the thread pool)
# ---------------------------------------------------------------------------


def _write_batch_entries(
    writer,
    vars,
    structs: list,
    energies: list,
    forces: list,
    weights: list,
    flush: bool,
) -> None:
    """Serialise one batch's results into ADIOS steps.

    Runs in a background thread so NVMe I/O overlaps with GPU inference.

    Parameters
    ----------
    writer, vars    : ADIOS2 Engine and variable dict (owned by this thread).
    structs         : list of PyG Data objects for this batch.
    energies        : list of float, one per structure.
    forces          : list of CPU tensors [N_atoms, 3], one per structure.
    weights         : list of CPU tensors [NUM_BRANCHES], one per structure.
    flush           : if True, call PerformPuts() after writing all structures
                      in this batch to bound memory usage on large batches.
    """
    for i in range(len(energies)):
        data = structs[i]

        # atom_types: data.x is [N, 1] float (atomic numbers) from build_random_structure
        atom_types = data.x.squeeze(1).numpy().astype(np.int32)

        # coordinates: data.pos is [N, 3] float
        coords = data.pos.numpy().astype(np.float64)

        # forces: [N, 3] CPU tensor from run_fused_inference
        f = forces[i].numpy().astype(np.float64)

        # energy scalar
        energy = np.array([energies[i]], dtype=np.float64)

        # branch weights: [NUM_BRANCHES] CPU tensor
        bw = weights[i].numpy().astype(np.float64)

        _write_structure_step(writer, vars, atom_types, coords, f, energy, bw)

    if flush:
        writer.PerformPuts()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_argument_parser(
        description="Fused HydraGNN + MLP inference – ADIOS2 BP5 output to NVMe"
    )
    add_fused_cli_arguments(parser)
    parser.set_defaults(num_structures=100)
    parser.add_argument(
        "--nvme_dir",
        type=str,
        default=None,
        help="NVMe output directory. Defaults to /mnt/bb/$USER",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=1000,
        help="Call PerformPuts() every N structures to bound ADIOS buffer memory "
        "(default: 1000). Set to 0 to disable mid-batch flushing.",
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

    # --- Open NVMe output ---
    nvme_base = args.nvme_dir
    if nvme_base is None:
        user = os.environ.get("USER", "unknown")
        nvme_base = f"/mnt/bb/{user}"

    os.makedirs(nvme_base, exist_ok=True)
    filename = f"inference_fused_results_gpu{local_gpu_id}.bp"
    nvme_path = os.path.join(nvme_base, filename)

    # Open writer before the thread pool so the ADIOS handle is created on the
    # main thread, then exclusively owned by the single background thread.
    adios_obj, io_obj, writer, vars = _open_adios_writer(nvme_path)

    flush_every = args.flush_every if args.flush_every > 0 else None

    # --- Thread pool: max_workers=1 so writes are serialised ---
    executor = ThreadPoolExecutor(max_workers=1)
    futures: List[Future] = []
    struct_offset = [0]
    batch_counter = [0]

    def _on_batch(batch_idx, batch_energies, batch_forces, batch_natoms, batch_weights):
        """Called from run_fused_inference after each batch is on CPU.

        Submits the ADIOS write to the background thread so it overlaps
        with GPU inference of the next batch.
        """
        offset = struct_offset[0]
        n = len(batch_energies)
        structs_slice = structures[offset : offset + n]
        struct_offset[0] += n

        # Flush if this batch would push us past a flush_every boundary
        should_flush = flush_every is not None and (offset // flush_every) != (
            (offset + n - 1) // flush_every
        )

        fut = executor.submit(
            _write_batch_entries,
            writer,
            vars,
            structs_slice,
            batch_energies,
            batch_forces,
            batch_weights,
            should_flush,
        )
        futures.append(fut)
        batch_counter[0] += 1

    print(
        f"[rank {world_rank}] ADIOS2 write pipeline enabled – "
        f"background thread will stream results to {nvme_path}"
    )

    omnistat_fom_url = None
    if args.omnistat_fom:
        omnistat_fom_url = f"http://localhost:{args.omnistat_fom_port}/fom"

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
        omnistat_fom_url=omnistat_fom_url,
        omnistat_fom_gpu_id=local_gpu_id,
        disable_param_grad=args.disable_param_grad,
        batched_decoder=args.batched_decoder,
        compile_encoder=args.compile_encoder,
        compile_decoder=args.compile_decoder,
        compile_full=args.compile_full,
        compile_backend=args.compile_backend,
    )

    # --- Drain the thread pool before closing the writer ---
    for fut in futures:
        fut.result()  # re-raises any exception from the background thread
    executor.shutdown(wait=True)

    writer.Close()

    total_written = struct_offset[0]
    print(
        f"[rank {world_rank}] Wrote {total_written} structures "
        f"({batch_counter[0]} batches) to {nvme_path}"
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
