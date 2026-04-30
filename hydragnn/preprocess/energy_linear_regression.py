import os
from mpi4py import MPI
import argparse
import numpy as np

from hydragnn.utils.distributed import nsplit
from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
from tqdm import tqdm
import logging


logger = logging.getLogger("energy_linear_transform")
logging.basicConfig(
    format="%(levelname)s %(asctime)s %(message)s",
    level=os.environ.get("GPS_LOG_LEVEL", logging.DEBUG),
)


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


def __parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input adios file")
    parser.add_argument(
        "output_file", type=str, help="Path to the output adios file to be created"
    )
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
    return args


def _read_adios_input(adios_input, comm):
    valset = AdiosDataset(
        adios_input,
        "valset",
        comm,
        enable_cache=True,
    )
    trainset = AdiosDataset(
        adios_input,
        "trainset",
        comm,
        enable_cache=True,
    )
    testset = AdiosDataset(
        adios_input,
        "testset",
        comm,
        enable_cache=True,
    )

    return trainset, valset, testset


def _write_adios_output(
    adios_output, trainset, valset, testset, pna_deg, x, modelname, comm
):
    adwriter = AdiosWriter(adios_output, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    adwriter.add_global("pna_deg", pna_deg)
    adwriter.add_global("energy_linear_regression_coeff", x)
    adwriter.add_global("dataset_name", modelname)
    adwriter.save()


def main():
    args = __parse_args()

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    trainset, valset, testset = _read_adios_input(args.input_file, comm)
    pna_deg = trainset.pna_deg
    modelname = trainset.dataset_name

    ## Iterate over local datasets
    energy_list = list()
    feature_list = list()
    for dataset in [trainset, valset, testset]:
        rx = list(nsplit(range(len(dataset)), comm_size))[comm_rank]
        upper = rx[-1] + 1 if args.nsample_only is None else rx[0] + args.nsample_only
        print(comm_rank, "Loading:", rx[0], upper)
        dataset.setsubset(rx[0], upper, preload=True)

        for data in tqdm(
            dataset, disable=comm_rank != 0, desc="Collecting node feature"
        ):
            energy_list.append(data.energy.item())
            atomic_number_list = data.x[:, 0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            # hist = hist / data.num_nodes
            feature_list.append(hist)

    ## energy
    if comm_rank == 0:
        print("Collecting energy")
    _e = np.array(energy_list)
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
    energy_list = list()
    for dataset in [trainset, valset, testset]:
        for data in tqdm(dataset, disable=comm_rank != 0, desc="Update energy"):
            atomic_number_list = data.x[:, 0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            # hist = hist / data.num_nodes
            if args.verbose:
                print(
                    comm_rank,
                    "current,new,diff:",
                    data.energy.item(),
                    data.energy.item() - np.dot(hist, x),
                    np.dot(hist, x),
                )
            data.energy -= np.dot(hist, x)
            energy_list.append((data.energy.item(), -np.dot(hist, x)))
            if "y_loc" in data:
                del data.y_loc

            # We need to update the values of the energy in data.y
            # We assume that the energy is the first entry of data.y
            data.y[0] = data.energy.detach().clone()

    if args.savenpz:
        if comm_size < 400:
            if comm_rank == 0:
                energy_list_all = comm.gather(energy_list, root=0)
                energy_arr = np.concatenate(energy_list_all, axis=0)
                np.savez(f"{modelname}_energy.npz", energy=energy_arr)
            else:
                comm.gather(energy_list, root=0)
        else:
            energy_arr = np.concatenate(energy_list, axis=0)
            np.savez(f"{modelname}_energy_rank_{comm_rank}.npz", energy=energy_arr)

    ## Writing
    if comm_rank == 0:
        print(f"Writing output to {args.output_file}")
    _write_adios_output(
        args.output_file, trainset, valset, testset, pna_deg, x, modelname.lower(), comm
    )

    print("Done.")


if __name__ == "__main__":
    main()
