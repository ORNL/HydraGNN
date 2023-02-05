import os
import glob
import pickle

import torch
from mpi4py import MPI

from .print_utils import print_distributed, log, iterate_tqdm

from hydragnn.utils.basedataset import BaseDataset
from hydragnn.utils.rawdataset import RawDataset


class SimplePickleDataset(BaseDataset):
    """Simple Pickle Dataset"""

    def __init__(self, basedir, label, subset=None, preload=False):
        """
        Parameters
        ----------
        basedir: basedir
        label: label
        subset: a list of index to subset
        """
        super().__init__()

        self.basedir = basedir
        self.label = label
        self.subset = subset
        self.preload = preload

        fname = os.path.join(basedir, "%s-meta.pk" % label)
        with open(fname, "rb") as f:
            self.minmax_node_feature = pickle.load(f)
            self.minmax_graph_feature = pickle.load(f)
            self.ntotal = pickle.load(f)
            self.use_subdir = pickle.load(f)
            self.nmax_persubdir = pickle.load(f)

        log("Pickle files:", self.label, self.ntotal)

        if self.subset is None:
            self.subset = list(range(self.ntotal))

        if self.preload:
            for i in range(self.ntotal):
                self.dataset.append(self.read(i))

    def len(self):
        return len(self.subset)

    def get(self, i):
        k = self.subset[i]
        if self.preload:
            return self.dataset[k]
        else:
            return self.read(k)

    def setsubset(self, subset):
        self.subset = subset

    def read(self, k):
        """
        Read from disk
        """
        fname = "%s-%d.pk" % (self.label, k)
        dirfname = os.path.join(self.basedir, fname)
        if self.use_subdir:
            subdir = str(k // self.nmax_persubdir)
            dirfname = os.path.join(self.basedir, subdir, fname)
        with open(dirfname, "rb") as f:
            data_object = pickle.load(f)
        return data_object


class SimplePickleWriter:
    """SimplePickleWriter class to write Torch Geometric graph data"""

    def __init__(
        self,
        dataset,
        basedir,
        label="total",
        minmax_node_feature=None,
        minmax_graph_feature=None,
        use_subdir=False,
        nmax_persubdir=10_000,
        comm=MPI.COMM_WORLD,
    ):
        """
        Parameters
        ----------
        dataset: locally owned dataset (should be iterable)
        basedir: basedir
        label: label
        nmax: nmax in case of subdir
        minmax_node_feature: minmax_node_feature
        minmax_graph_feature: minmax_graph_feature
        comm: MPI communicator
        """

        self.dataset = dataset
        if not isinstance(dataset, list):
            raise Exception("Unsuppored data type yet.")

        self.basedir = basedir
        self.label = label
        self.use_subdir = use_subdir
        self.nmax_persubdir = nmax_persubdir
        self.comm = comm
        self.rank = comm.Get_rank()

        self.minmax_node_feature = minmax_node_feature
        self.minmax_graph_feature = minmax_graph_feature

        ns = self.comm.allgather(len(self.dataset))
        noffset = sum(ns[: self.rank])
        ntotal = sum(ns)

        if self.rank == 0:
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            fname = os.path.join(basedir, "%s-meta.pk" % (label))
            with open(fname, "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(ntotal, f)
                pickle.dump(use_subdir, f)
                pickle.dump(nmax_persubdir, f)
        comm.Barrier()

        if use_subdir:
            ## Create subdirs first
            subdirs = set()
            for i in range(len(self.dataset)):
                subdirs.add(str((noffset + i) // nmax_persubdir))
            for k in subdirs:
                subdir = os.path.join(basedir, k)
                os.makedirs(subdir, exist_ok=True)

        for i, data in enumerate(self.dataset):
            fname = "%s-%d.pk" % (label, noffset + i)
            dirfname = os.path.join(basedir, fname)
            if use_subdir:
                subdir = str((noffset + i) // nmax_persubdir)
                dirfname = os.path.join(basedir, subdir, fname)
            with open(dirfname, "wb") as f:
                pickle.dump(data, f)
