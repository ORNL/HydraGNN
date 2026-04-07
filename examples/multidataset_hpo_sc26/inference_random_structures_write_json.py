#!/usr/bin/env python3
"""Inference on randomly generated atomistic structures – writes per-GPU JSON results.

Extends ``inference_random_structures.py`` with two additional capabilities:

1.  Each GPU assembles a JSON file containing every structure it processed,
    together with the predicted energy and forces.
2.  The JSON file is first built in GPU-host memory and then flushed to the
    node-local NVMe on OLCF-Frontier (``/mnt/bb/$USER``).  File names include
    the local GPU id so that multiple GPUs on the same node never collide.

Usage:
    srun ... python inference_random_structures_write_json.py \
        --logdir <path_to_training_log_dir> \
        --num_structures 100 \
        --min_atoms 2 --max_atoms 20 \
        --box_size 10.0

The --logdir should contain a config.json and a .pk checkpoint file produced
by a prior HydraGNN training run.
"""

import json
import os

import numpy as np

from hydragnn.utils.distributed import get_comm_size_and_rank, get_local_rank

from inference_random_structures import (
    build_argument_parser,
    load_config_and_model,
    generate_structures,
    run_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def structure_to_dict(data, energy, forces):
    """Convert a single PyG Data object + predictions into a JSON-serialisable dict."""
    pos_np = data.pos.cpu().numpy()
    entry = {
        "atom_types": data.x.squeeze(1).cpu().numpy().astype(int).tolist(),
        "coordinates_x": pos_np[:, 0].tolist(),
        "coordinates_y": pos_np[:, 1].tolist(),
        "coordinates_z": pos_np[:, 2].tolist(),
        "formation energy": energy,
    }
    if forces is not None:
        forces_np = forces.cpu().numpy()
        entry["forces_x"] = forces_np[:, 0].tolist()
        entry["forces_y"] = forces_np[:, 1].tolist()
        entry["forces_z"] = forces_np[:, 2].tolist()
    else:
        n_atoms = data.pos.size(0)
        entry["forces_x"] = [0.0] * n_atoms
        entry["forces_y"] = [0.0] * n_atoms
        entry["forces_z"] = [0.0] * n_atoms
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_argument_parser(
        description="HydraGNN inference on random structures – JSON output to NVMe"
    )
    parser.add_argument(
        "--nvme_dir",
        type=str,
        default=None,
        help="NVMe output directory. Defaults to /mnt/bb/$USER",
    )
    args = parser.parse_args()

    _, world_rank = get_comm_size_and_rank()
    local_gpu_id = get_local_rank()

    model, config, device, autocast_ctx, param_dtype = load_config_and_model(
        args.logdir,
        args.checkpoint,
        args.precision,
    )
    print(f"[rank {world_rank}] local GPU id: {local_gpu_id}")

    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)
    enable_ip = arch.get("enable_interatomic_potential", False)

    # Use (seed + world_rank) so each GPU gets different random structures
    structures = generate_structures(
        args.num_structures,
        args.min_atoms,
        args.max_atoms,
        args.box_size,
        args.max_atomic_number,
        radius,
        max_neighbours,
        args.branch_id,
        args.seed + world_rank,
    )
    print(
        f"[rank {world_rank}] Generated {len(structures)} random structures "
        f"(atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    all_energies, all_forces, all_natoms = run_inference(
        model,
        structures,
        args.batch_size,
        param_dtype,
        autocast_ctx,
        enable_ip,
    )

    # ----- Build per-structure JSON entries -----
    json_entries = []
    for i in range(len(all_energies)):
        entry = structure_to_dict(structures[i], all_energies[i], all_forces[i])
        entry["structure_index"] = i
        json_entries.append(entry)

    # ----- Build complete JSON payload in memory -----
    json_payload = json.dumps(
        {"num_structures": len(json_entries), "structures": json_entries},
        indent=2,
    )

    # ----- Write to node-local NVMe -----
    nvme_base = args.nvme_dir
    if nvme_base is None:
        user = os.environ.get("USER", "unknown")
        nvme_base = f"/mnt/bb/{user}"

    os.makedirs(nvme_base, exist_ok=True)

    filename = f"inference_results_gpu{local_gpu_id}.json"
    nvme_path = os.path.join(nvme_base, filename)

    with open(nvme_path, "w") as f:
        f.write(json_payload)

    print(f"[rank {world_rank}] Wrote {len(json_entries)} structures to {nvme_path}")

    # ----- Brief console summary -----
    print(f"\n[rank {world_rank}] " + "=" * 60)
    print(
        f"[rank {world_rank}] {'Struct':>6} | {'Atoms':>5} | {'Energy':>16} | "
        f"{'Energy/atom':>16} | {'|F|_mean':>12}"
    )
    print(f"[rank {world_rank}] " + "-" * 60)
    for entry in json_entries:
        idx = entry["structure_index"]
        n = len(entry["atom_types"])
        e = entry["formation energy"]
        e_per_atom = e / n
        fx = np.array(entry["forces_x"])
        fy = np.array(entry["forces_y"])
        fz = np.array(entry["forces_z"])
        f_norms = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        f_mean = f_norms.mean()
        print(
            f"[rank {world_rank}] {idx:6d} | {n:5d} | {e:16.6f} | "
            f"{e_per_atom:16.6f} | {f_mean:12.6f}"
        )
    print(f"[rank {world_rank}] " + "=" * 60)


if __name__ == "__main__":
    main()
