#!/usr/bin/env python3

""" Import a CSV file into ADIOS2 format
    so that it can be read by HydraGNN
    as an AdiosDataset.
"""
import os, json
import pickle, csv
from pathlib import Path
import random
from typing import List

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
import numpy as np

import hydragnn
from hydragnn.utils.pickledataset import SimplePickleWriter  # , SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    get_edge_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit

from models import DataDescriptor, number_categories


try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch

info = logging.info


def random_splits(N, x, y):
    """Shuffle and sample the data indices.

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
    assert 0.0 <= y <= 1.0 - x
    a = list(range(N))
    a = random.sample(a, N)
    return np.split(a, [int(x * N), int((x + y) * N)])


def validate_data(x: np.ndarray, ncat: int, tol=0.001) -> None:
    """Validate that the data elements (x)
    are correctly described by the type label ncat.
    """
    if ncat == 0:  # floating point data
        return
    assert ncat != 1, "Invalid number of categories."
    # cast -1 to NaN
    neg_one = np.abs(x + 1.0) < tol  # within tol of -1
    x[neg_one] = np.nan

    x = x[~np.isnan(x)]
    y = x.astype(int)
    assert np.allclose(x, y, atol=tol)
    assert np.all(x >= 0), "Negative categorical values are not allowed."
    assert np.all(x < ncat), "Categorical values out of range."


def validate_split_names(split):
    names = ["train", "val", "test", "excl"]
    test = [split.startswith(name) for name in names]
    test1 = test[0] | test[1] | test[2] | test[3]

    vals = split.decode(encoding="ascii")
    if test1.sum() == len(vals):
        return
    mismatch = vals[~test1]
    print("mismatches:")
    print(mismatch)


def load_columns(datafile: str, descr: DataDescriptor):
    if datafile.endswith("csv"):
        df = pd.read_csv(datafile)
    else:
        df = pd.read_parquet(datafile)
    smiles_all = df[descr.smiles].to_list()
    names = [val.name for val in descr.graph_tasks]
    values_all = df[names].values.astype(float)

    N = len(smiles_all)
    assert len(values_all) == N
    print("    total records:", N)
    for i, task in enumerate(descr.graph_tasks):
        ncat = number_categories(task.type)
        validate_data(values_all[:, i], ncat)

    if descr.split is None:  # no labels - generate an 80/10/10 split.
        idxs = random_splits(N, 0.8, 0.1)
    else:
        split = df[descr.split].str  # use string functions on the "split" col.
        validate_split_names(split)
        idxs = [
            np.flatnonzero(split.startswith(name)) for name in ["train", "val", "test"]
        ]
    smiles = [[smiles_all[i] for i in ix] for ix in idxs]
    values = [torch.tensor([values_all[i] for i in ix]) for ix in idxs]
    return smiles, values, names


def calc_offsets(counts: List[int]) -> List[int]:
    """Calculate the starting offsets for each
    range -- where counts is a list of lengths of each range.
    """
    k = 0
    off = []
    for d in counts:
        off.append(k)
        k += d
    return off


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
    basedir = Path(args.output)
    # create output path and ensure it doesn't yet exist
    try:  # only succeeds if dir is empty
        basedir.rmdir()
    except FileNotFoundError:
        pass
    basedir.mkdir(parents=True)

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
        descr = DataDescriptor.model_validate(yaml.safe_load(f))
    smiles_sets, values_sets, task_names = load_columns(args.input, descr)
    info("trainset,valset,testset size: %d %d %d", *map(len, smiles_sets))

    setnames = ["trainset", "valset", "testset"]
    dataset = dict((k, []) for k in setnames)
    for name, smileset, valueset in zip(setnames, smiles_sets, values_sets):
        rx = list(nsplit(range(len(smileset)), comm_size))[rank]
        # info("subset range (%s): %d %d %d", name, len(smileset), rx.start, rx.stop)
        ## local portion
        _smileset = smileset[rx.start : rx.stop]
        _valueset = valueset[rx.start : rx.stop]
        missed_count = 0
        for smilestr, ytarget in tqdm(
            zip(_smileset, _valueset),
            disable=rank != 0,
            desc="Featurizing",
            total=len(_smileset),
        ):
            try:
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, get_positions=True
                )
                # hack to make edge_attr as the models expect.
                data.edge_attr = (
                    torch.Tensor([1]).repeat(data.edge_index.shape[1]).unsqueeze(1)
                )
                # TODO: ensure data.pos is populated (e.g. call rdkit)
                # TODO: should we energy minimize these coordinates

                assert isinstance(data, torch_geometric.data.Data)
                dataset[name].append(data)
            except Exception as e:
                print(
                    f"Exception in call to generate_graphdata_from_smilestr."
                    f" {e} for {smilestr}. Ignoring molecule and proceeding .."
                )
                missed_count += 1
    print(f"missed {missed_count} molecules")
    # pre-compute PNA degrees
    pna_deg = gather_deg(dataset["trainset"])
    # this is stored with the trainset in pickle format
    # and as a global in adios format
    # config["pna_deg"] = pna_deg

    node_names, node_dims = get_node_attribute_name()
    node_names = [x.replace("atomicnumber", "atomic_number") for x in node_names]
    edge_names, edge_dims = get_edge_attribute_name()
    task_dims = [1] * len(task_names)
    attrs = {
        "x_name": node_names,
        "x_name/feature_count": np.array(node_dims),
        "x_name/feature_offset": np.array(calc_offsets(node_dims)),
        "y_name": task_names,
        "y_name/feature_count": np.array(task_dims),
        "y_name/feature_offset": np.array(calc_offsets(task_dims)),
        "edge_attr_name": edge_names,
        "edge_attr_name/feature_count": np.array(edge_dims),
        "edge_attr_name/feature_offset": np.array(calc_offsets(edge_dims)),
    }
    # attrs = dict( (k, np.array(v)) for k,v in attrs.items() )

    if output_format == "pickle":
        for name, data in dataset.items():
            if name == "trainset":
                attrs["pna_deg"] = pna_deg
            SimplePickleWriter(
                data,
                str(basedir),
                name,
                use_subdir=True,
                attrs=attrs,
            )
            if name == "trainset":
                del attrs["pna_deg"]
    elif output_format == "adios":
        adwriter = AdiosWriter(str(basedir), comm)
        for name, data in dataset.items():
            adwriter.add(name, data)
        for k, v in attrs.items():
            adwriter.add_global(k, v)
        adwriter.add_global("pna_deg", pna_deg)
        adwriter.save()
    else:
        raise "Invalid output format. Must be one of pickle/adios"


if __name__ == "__main__":
    main()
