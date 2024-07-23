import os, json
import sys
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils import nsplit
from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
from tqdm import tqdm
from mpi_list import Context, DFM

def subset(i):
    #sz = len(dataset)
    #chunk = sz // C.procs
    #left  = sz % C.procs
    #a = i*chunk     + min(i, left)
    #b = (i+1)*chunk + min(i+1, left)
    #print(f"Rank {i}/{C.procs} converting subset [{a},{b})")
    #return np.array([np.array(x) for x in dataset[a:b]["image"]])
    return np.random.random( (100,4) )

# form the correlation matrix
def covar(x):
    return np.tensordot(x, x, axes=[(),()])

def summarize(x):
    N = len(x)
    m = x.sum(0)/N
    y = x-m[None,...]
    V = np.tensordot(y, y, [0,0])/N
    return {'N':N, 'm':m, 'V':V}

def merge_est(a, b):
    if not isinstance(b, dict):
        b = summarize(b)

    x = a['N']/(a['N']+b['N'])
    y = b['N']/(a['N']+b['N'])

    m = x*a['m'] + y*b['m']
    a['N'] += b['N']
    a['V'] = x*(a['V'] + covar(m - a['m'])) \
           + y*(b['V'] + covar(m - b['m']))
    a['m'] = m
    return a

def test():
    C = Context() # calls MPI_Init via mpi4py

    dfm = C . iterates(C.procs) \
            . map( subset )

    ans = {'N': 0, 'm': 0, 'V': 0}
    ans = dfm.reduce(merge_est, ans, False)
    if C.rank == 0:
        print(ans)
        print(f"theoretical: m = 0.5, v = {0.25/3}")

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
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="gfm_multitasking.json"
    )
    parser.add_argument(
        "--modelname", help="modelname", type=str, default="ANI1x"
    )
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "force"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % args.modelname)
    print("fname:", fname)
    trainset = AdiosDataset(
        fname,
        "trainset",
        comm,
        var_config=var_config,
    )
    valset = AdiosDataset(
        fname,
        "valset",
        comm,
        var_config=var_config,
    )
    testset = AdiosDataset(
        fname,
        "testset",
        comm,
        var_config=var_config,
    )
    pna_deg = trainset.pna_deg

    ## Iterate over local dataset
    energy_list = list()
    feature_list = list()
    for dataset in [trainset, valset, testset]:
        rx = list(nsplit(range(len(dataset)), comm_size))[comm_rank]
        print(comm_rank, "Loading:", rx[0], rx[-1] + 1)
        dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

        for data in tqdm(dataset, disable=comm_rank != 0, desc="Collecting node feature"):
            energy_list.append(data.energy.item()/data.num_nodes)
            atomic_number_list = data.x[:,0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118+2))
            hist = hist/data.num_nodes
            feature_list.append(hist)
            import pdb; pdb.set_trace()
    
    ## energy
    if comm_rank == 0:
        print("Collecting energy")
    _e = np.array(energy_list)
    _X = np.array(feature_list)
    _n = len(_e)
    n = comm.allreduce(_n, op=MPI.SUM)
    _esum = _e.sum()
    emean = comm.allreduce(_esum, op=MPI.SUM)/n
    ## e = e - e_mean
    _e = _e - emean

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
    for dataset in [trainset, valset, testset]:
        for data in tqdm(dataset, disable=comm_rank != 0, desc="Update energy"):
            atomic_number_list = data.x[:,0].tolist()
            assert len(atomic_number_list) == data.num_nodes
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118+2))
            hist = hist/data.num_nodes
            data.energy = data.energy/data.num_nodes - np.dot(hist, x)
            if "y_loc" in data:
                del data.y_loc

    ## Writing
    fname = os.path.join(os.path.dirname(__file__), "./dataset/%s-v2.bp" % args.modelname)
    if comm_rank == 0:
        print("Saving:", fname)
    adwriter = AdiosWriter(fname, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    adwriter.add_global("pna_deg", pna_deg)
    adwriter.save()

    print("Done.")
