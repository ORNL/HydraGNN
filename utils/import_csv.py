#!/usr/bin/env python3

""" Import a CSV file into ADIOS2 format
    so that it can be read by HydraGNN
    as an AdiosDataset.
"""
import os, json
import pickle, csv
from pathlib import Path
import random

import logging
import sys
import argparse
import time

import mpi4py
mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from tqdm import tqdm
import pandas as pd
import yaml

import hydragnn
from hydragnn.utils.pickledataset import SimplePickleWriter #, SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch

node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5, "Hg": 6, "Cl": 7}

info = logging.info

def random_splits(N, x, y):
    """ Shuffle and sample the data indices.

        Args:
        
          N: number of data elements
          x: training fraction
          y: validation fraction

        Notes:
          0 <= x <= 1.0
          0 <= y <= 1.0
          0 <= (1-x-y) <= 1.0 is the testing fraction
          so x+y <= 1.0
    """
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0-x
    a = list(range(N))
    a = random.sample(a, N)
    return np.split( a, [int(x * N), int((x+y) * N)] )

def load_columns(datafile, descr):
    df = pd.read_csv(datafile)
    smiles_all = df[ descr["smiles"] ].to_list()
    names = [ val["name"] for val in descr["graph_tasks"] ]
    values_all = df[ names ].values #.tolist()

    N = len(smiles_all)
    assert len(values_all) == N
    print("Total:", N)

    idxs = random_splits(N, 0.8, 0.1)
    assert len(idxs) == 3
    smiles = [ [smiles_all[i] for i in ix] for ix in idxs ]
    values = [ torch.tensor([values_all[i] for i in ix]) for ix in idxs ]
    return smiles, values

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        help="input CSV/PQ file",
        required=True,
    )
    parser.add_argument(
        "--descr",
        help="yaml description",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="output directory",
        required=True,
    )
    args = parser.parse_args()
    if args.output.endswith(".pkl"):
        output_format = "pickle"
    elif args.output.endswith(".bp"):
        output_format = "adios"
    else:
        raise "Invalid output format. --output must end with .pkl (pickle) or .bp (adios) suffix."
    basedir = args.output
    # create output path and ensure it doesn't yet exist
    Path(basedir).mkdir(parents=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.descr, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)
    smiles_sets, values_sets = load_columns(args.input, descr)
    info(
        "trainset,valset,testset size: %d %d %d",
        *map(len, smiles_sets)
    )

    verbosity = 2

    setnames = ["trainset", "valset", "testset"]
    dataset = dict((k,[]) for k in setnames)
    for name, smileset, valueset in zip(setnames, smiles_sets, values_sets):
        rx = list(nsplit(range(len(smileset)), comm_size))[rank]
        info("subset range (%s): %d %d %d", name, len(smileset), rx.start, rx.stop)
        ## local portion
        _smileset = smileset[rx.start : rx.stop]
        _valueset = valueset[rx.start : rx.stop]

        for smilestr, ytarget in tqdm(
            zip(_smileset, _valueset),
            disable=rank != 0,
            desc="Featurizing",
            total=len(_smileset)
        ):
            try:
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, node_types
                )

                assert isinstance(data, torch_geometric.data.Data)
                dataset[name].append(data)
            except Exception as e:
                print(
                    f"Exception in call to generate_graphdata_from_smilestr."
                    f" {e} for {smilestr}. Ignoring molecule and proceeding .."
                )

    # pre-compute PNA degrees
    pna_deg = gather_deg(dataset["trainset"])
    # this is stored with the trainset in pickle format
    # and as a global in adios format
    #config["pna_deg"] = pna_deg

    if output_format == 'pickle':
        for name, data in dataset.items():
            attrs = dict()
            if name == "trainset":
                attrs["pna_deg"] = pna_deg
            SimplePickleWriter(
                data,
                basedir,
                name,
                use_subdir=True,
                attrs=attrs,
            )
    elif output_format == 'adios':
        adwriter = AdiosWriter(basedir, comm)
        for name, data in dataset.items():
            adwriter.add(name, data)
        adwriter.add_global("pna_deg", pna_deg)
        adwriter.save()
    else:
        raise "Invalid output format. Must be one of pickle/adios"

if __name__ == "__main__":
    main()
