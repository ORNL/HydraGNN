import os, json
import sys
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.distributed import nsplit
from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
from tqdm import tqdm

# This import requires having installed the package mpi_list
try:
    from mpi_list import Context, DFM
except ImportError:
    print("mpi_list requires having installed: https://github.com/frobnitzem/mpi_list")


def solve_least_squares_svd(A, b):
    # Compute the SVD of A
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Compute the pseudo-inverse of S
    S_inv = np.diag(1 / S)
    # Compute the pseudo-inverse of A
    A_pinv = np.dot(Vt.T, np.dot(S_inv, U.T))
    # Solve for x using the pseudo-inverse
    x = np.dot(A_pinv, b)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("modelname", help="modelname", type=str, default="ANI1x")
    parser.add_argument(
        "--nsample_only",
        help="nsample only",
        type=int,
    )
    parser.add_argument(
        "--verbose",
        help="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--savenpz",
        help="save npz",
        action="store_true",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    fname = os.path.join(os.path.dirname(__file__), "./datasets/%s.bp" % args.modelname)
    print("fname:", fname)
    trainset = AdiosDataset(
        fname,
        "trainset",
        comm,
        enable_cache=True,
    )
    valset = AdiosDataset(
        fname,
        "valset",
        comm,
        enable_cache=True,
    )
    testset = AdiosDataset(
        fname,
        "testset",
        comm,
        enable_cache=True,
    )
    pna_deg = trainset.pna_deg

    ## Iterate over local datasets
    energy_per_atom_list = list()
    feature_list = list()
    for dataset in [trainset, valset, testset]:
        rx = list(nsplit(range(len(dataset)), comm_size))[comm_rank]
        upper = rx[-1] + 1 if args.nsample_only is None else rx[0] + args.nsample_only
        print(comm_rank, "Loading:", rx[0], upper)
        dataset.setsubset(rx[0], upper, preload=True)

        for data in tqdm(
            dataset, disable=comm_rank != 0, desc="Collecting node feature"
        ):
            energy_per_atom_list.append(data.energy_per_atom.item())
            atomic_number_list = data.x[:, 0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            hist = hist / data.num_nodes
            feature_list.append(hist)

    ## energy
    if comm_rank == 0:
        print("Collecting energy")
    _e = np.array(energy_per_atom_list)
    _X = np.array(feature_list)
    _N = len(_e)
    N = comm.allreduce(_N, op=MPI.SUM)
    _esum = _e.sum()

    ## A
    if comm_rank == 0:
        print("Collecting A")
    _A = _X.T @ _X
    A = comm.allreduce(_A, op=MPI.SUM)

    ## b
    if comm_rank == 0:
        print("Collecting b")
    _b = _X.T @ _e
    b = comm.allreduce(_b, op=MPI.SUM)

    ## Solve Ax=b
    # x = np.linalg.solve(A, b)
    x = solve_least_squares_svd(A, b)

    ## Re-calculate energy
    energy_per_atom_list = list()
    for dataset in [trainset, valset, testset]:
        for data in tqdm(dataset, disable=comm_rank != 0, desc="Update energy"):
            atomic_number_list = data.x[:, 0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            hist = hist / data.num_nodes
            if args.verbose:
                print(
                    comm_rank,
                    "current,new,diff:",
                    data.energy_per_atom.item(),
                    data.energy_per_atom.item() - np.dot(hist, x),
                    np.dot(hist, x),
                )
            data.energy_per_atom -= np.dot(hist, x)
            energy_per_atom_list.append((data.energy_per_atom.item(), -np.dot(hist, x)))
            if "y_loc" in data:
                del data.y_loc

            # We need to update the values of the energy in data.y
            # We assume that the energy is the first entry of data.y
            data.y[0] = data.energy_per_atom.detach().clone()

    if args.savenpz:
        if comm_size < 400:
            if comm_rank == 0:
                energy_per_atom_list_all = comm.gather(energy_per_atom_list, root=0)
                energy_arr = np.concatenate(energy_per_atom_list_all, axis=0)
                np.savez(f"{args.modelname}_energy_per_atom.npz", energy=energy_arr)
            else:
                comm.gather(energy_per_atom_list, root=0)
        else:
            energy_arr = np.concatenate(energy_per_atom_list, axis=0)
            np.savez(
                f"{args.modelname}_energy_per_atom_rank_{comm_rank}.npz",
                energy=energy_arr,
            )

    ## Writing
    fname = os.path.join(
        os.path.dirname(__file__), "./datasets/%s-v3.bp" % args.modelname
    )
    if comm_rank == 0:
        print("Saving:", fname)
    adwriter = AdiosWriter(fname, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    adwriter.add_global("pna_deg", pna_deg)
    adwriter.add_global("energy_linear_regression_coeff", x)
    adwriter.add_global("dataset_name", args.modelname.lower())
    adwriter.save()

    print("Done.")
