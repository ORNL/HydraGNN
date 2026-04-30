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
"""ASE structure optimization with fused HydraGNN + BranchWeightMLP inference.

This replaces the obsolete single-model loading path with the same fused stack
used by ``inference_fused.py``. The calculator builds a single graph from an
ASE ``Atoms`` object, evaluates all HydraGNN branches, applies branch weights
from the auxiliary MLP, and returns weighted energy and forces to ASE.
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
from torch_geometric.data import Data

from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc

from inference_fused import load_fused_stack, run_fused_inference


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASE optimization using fused HydraGNN + BranchWeightMLP"
    )
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join(dirpwd, "multidataset_hpo-BEST6-fp64"),
        help="Directory containing config.json and HydraGNN .pk checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional HydraGNN checkpoint filename or absolute path",
    )
    parser.add_argument(
        "--mlp_checkpoint",
        type=str,
        default=os.path.join(dirpwd, "mlp_branch_weights.pt"),
        help="Path to the auxiliary MLP checkpoint",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="structures/mos2-B_Defect-Free_PBE.vasp",
        help="Structure file to optimize; relative paths resolve from this script directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Optional ASE input format override",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Optional HydraGNN precision override (fp32, fp64, bf16)",
    )
    parser.add_argument(
        "--mlp_precision",
        type=str,
        default=None,
        help="Optional MLP precision override (fp32, fp64, bf16)",
    )
    parser.add_argument(
        "--mlp_device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for the auxiliary MLP",
    )
    parser.add_argument(
        "--charge",
        type=float,
        default=0.0,
        help="Graph-level charge used for graph_attr conditioning",
    )
    parser.add_argument(
        "--spin",
        type=float,
        default=0.0,
        help="Graph-level spin used for graph_attr conditioning",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["FIRE", "BFGS", "BFGSLineSearch"],
        default="FIRE",
        help="ASE optimizer",
    )
    parser.add_argument(
        "--maxstep",
        type=float,
        default=1e-2,
        help="Maximum ASE optimizer step size",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum optimization steps",
    )
    parser.add_argument(
        "--relative_increase_threshold",
        type=float,
        default=0.05,
        help="Stop and revert if max force increases by more than this fraction",
    )
    parser.add_argument(
        "--random_displacement",
        action="store_true",
        help="Add uniform random displacement before optimization",
    )
    parser.add_argument(
        "--random_displacement_scale",
        type=float,
        default=0.1,
        help="Maximum absolute displacement in Angstrom for random perturbation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for optional initial displacement",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output filename for the optimized structure",
    )
    return parser


def _resolve_path(script_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)


def atoms_to_graph(atoms, graph_attr: torch.Tensor, radius: float, max_neighbours: int):
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)

    hist, _ = np.histogram(atomic_numbers, bins=range(1, 118 + 2))
    data = Data(
        x=torch.tensor(atomic_numbers, dtype=torch.get_default_dtype()).unsqueeze(1),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.get_default_dtype()),
        chemical_composition=torch.tensor(hist, dtype=torch.float32).unsqueeze(1),
        graph_attr=graph_attr.clone(),
        natoms=torch.tensor([len(atomic_numbers)], dtype=torch.long),
        cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
        pbc=torch.tensor(pbc, dtype=torch.bool),
    )
    add_edges_pbc = get_radius_graph_pbc(radius=radius, max_neighbours=max_neighbours)
    return add_edges_pbc(data)


class FusedHydraGNNCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model,
        mlp,
        radius,
        max_neighbours,
        param_dtype,
        autocast_ctx,
        device,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        charge,
        spin,
    ):
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.param_dtype = param_dtype
        self.autocast_ctx = autocast_ctx
        self.device = device
        self.num_branches = num_branches
        self.mlp_device = mlp_device
        self.mlp_autocast_ctx = mlp_autocast_ctx
        self.unified_mlp_gnn_stack = unified_mlp_gnn_stack
        self.graph_attr = torch.tensor([charge, spin], dtype=torch.float32)
        self.last_branch_weights = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        structure = atoms_to_graph(
            atoms,
            self.graph_attr,
            self.radius,
            self.max_neighbours,
        )
        (
            all_energies,
            all_forces,
            _all_natoms,
            all_weights,
            _batch_latencies_ms,
            _total_timed_structures,
            _stage_stats,
        ) = run_fused_inference(
            self.model,
            self.mlp,
            [structure],
            batch_size=1,
            param_dtype=self.param_dtype,
            autocast_ctx=self.autocast_ctx,
            device=self.device,
            num_branches=self.num_branches,
            num_warmup=0,
            mlp_device=self.mlp_device,
            mlp_autocast_ctx=self.mlp_autocast_ctx,
            unified_mlp_gnn_stack=self.unified_mlp_gnn_stack,
            profile_stages=False,
        )

        self.last_branch_weights = all_weights[0].numpy()
        self.results["energy"] = float(all_energies[0])
        self.results["forces"] = all_forces[0].numpy()


def build_optimizer(name: str, atoms, maxstep: float):
    if name == "FIRE":
        return FIRE(atoms, maxstep=maxstep)
    if name == "BFGS":
        return BFGS(atoms, maxstep=maxstep)
    return BFGSLineSearch(atoms, maxstep=maxstep)


def default_output_filename(structure_path: str, random_displacement: bool) -> str:
    root, ext = os.path.splitext(structure_path)
    suffix = "_optimized_structure"
    if random_displacement:
        suffix = "_optimized_structure_from_initial_randomly_perturbed_structure"
    return root + suffix + ext


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logdir = _resolve_path(script_dir, args.logdir)
    mlp_checkpoint = _resolve_path(script_dir, args.mlp_checkpoint)
    structure_path = _resolve_path(script_dir, args.structure)

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
        _gnn_prec,
        _mlp_prec,
    ) = load_fused_stack(
        logdir,
        args.checkpoint,
        mlp_checkpoint,
        args.precision,
        args.mlp_precision,
        args.mlp_device,
    )

    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)

    calculator = FusedHydraGNNCalculator(
        model,
        mlp,
        radius,
        max_neighbours,
        param_dtype,
        autocast_ctx,
        device,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        args.charge,
        args.spin,
    )

    atoms = read(structure_path, format=args.format)
    atoms.calc = calculator

    if args.random_displacement:
        rng = np.random.default_rng(args.seed)
        random_displacement = rng.uniform(
            -args.random_displacement_scale,
            args.random_displacement_scale,
            size=atoms.get_positions().shape,
        )
        atoms.set_positions(atoms.get_positions() + random_displacement)
        displaced_output = default_output_filename(structure_path, False)
        displaced_output = displaced_output.replace(
            "_optimized_structure", "_randomly_perturbed_structure"
        )
        write(displaced_output, atoms)
        print(f"Wrote perturbed structure to {displaced_output}")

    optimizer = build_optimizer(args.optimizer, atoms, args.maxstep)

    prev_max_force = None
    prev_positions = None
    for step in range(args.maxiter):
        optimizer.step()

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = float(np.sqrt((forces ** 2).sum(axis=1).max()))
        weights = calculator.last_branch_weights
        top_branch = int(np.argmax(weights)) if weights is not None else -1
        top_weight = float(weights[top_branch]) if weights is not None else float("nan")

        print(
            f"Step {step + 1}: Energy = {energy:.6f} eV, "
            f"Max Force = {max_force:.6f} eV/Å, "
            f"Top Branch = {top_branch}, Top Weight = {top_weight:.4f}"
        )

        if prev_max_force is not None and prev_max_force > 0.0:
            relative_increase = (max_force - prev_max_force) / prev_max_force
            if relative_increase > args.relative_increase_threshold:
                print(
                    f"Reverting to previous step at step {step + 1} due to "
                    f"a relative force increase of {relative_increase:.2%}."
                )
                atoms.set_positions(prev_positions)
                break

        prev_max_force = max_force
        prev_positions = deepcopy(atoms.get_positions())

    output_path = (
        _resolve_path(script_dir, args.output)
        if args.output is not None
        else default_output_filename(structure_path, args.random_displacement)
    )
    write(output_path, atoms)
    print(f"Wrote optimized structure to {output_path}")


if __name__ == "__main__":
    main()
