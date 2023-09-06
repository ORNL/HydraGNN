import os
import pickle

import torch
from mpi4py import MPI

from .print_utils import print_distributed, log, iterate_tqdm

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess import update_predicted_values, update_atom_features

import hydragnn.utils.tracer as tr


class SimplePickleDataset(AbstractBaseDataset):
    """Simple Pickle Dataset"""

    def __init__(self, basedir, label, subset=None, preload=False, var_config=None):
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
        self.var_config = var_config
        self.input_node_features = var_config["input_node_features"]

        if self.var_config is not None:
            self.variables_type = self.var_config["type"]
            self.output_index = self.var_config["output_index"]
            self.graph_feature_dim = self.var_config["graph_feature_dims"]
            self.node_feature_dim = self.var_config["node_feature_dims"]

        fname = os.path.join(basedir, "%s-meta.pkl" % label)
        with open(fname, "rb") as f:
            self.minmax_node_feature = pickle.load(f)
            self.minmax_graph_feature = pickle.load(f)
            self.ntotal = pickle.load(f)
            self.use_subdir = pickle.load(f)
            self.nmax_persubdir = pickle.load(f)
            self.attrs = pickle.load(f)
        log("Pickle files:", self.label, self.ntotal)
        if self.attrs is None:
            self.attrs = dict()
        for k in self.attrs:
            setattr(self, k, self.attrs[k])

        if self.subset is None:
            self.subset = list(range(self.ntotal))

        if self.preload:
            for i in range(self.ntotal):
                data = self.read(i)
                self.update_data_object(data)
                self.dataset.append(data)

    def len(self):
        return len(self.subset)

    @tr.profile("get")
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
        fname = "%s-%d.pkl" % (self.label, k)
        dirfname = os.path.join(self.basedir, fname)
        if self.use_subdir:
            subdir = str(k // self.nmax_persubdir)
            dirfname = os.path.join(self.basedir, subdir, fname)
        with open(dirfname, "rb") as f:
            data_object = pickle.load(f)
            self.update_data_object(data_object)
        return data_object

    def setsubset(self, subset):
        self.subset = subset

    def update_data_object(self, data_object):
        if self.var_config is not None:
            update_predicted_values(
                self.variables_type,
                self.output_index,
                self.graph_feature_dim,
                self.node_feature_dim,
                data_object,
            )
            update_atom_features(self.input_node_features, data_object)


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
        attrs=dict(),
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
            fname = os.path.join(basedir, "%s-meta.pkl" % (label))
            with open(fname, "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(ntotal, f)
                pickle.dump(use_subdir, f)
                pickle.dump(nmax_persubdir, f)
                pickle.dump(attrs, f)
        comm.Barrier()

        if use_subdir:
            ## Create subdirs first
            subdirs = set()
            for i in range(len(self.dataset)):
                subdirs.add(str((noffset + i) // nmax_persubdir))
            for k in subdirs:
                subdir = os.path.join(basedir, k)
                os.makedirs(subdir, exist_ok=True)

        for i, data in iterate_tqdm(
            enumerate(self.dataset),
            2,
            total=len(self.dataset),
            desc="Pickle write %s" % self.label,
        ):
            fname = "%s-%d.pkl" % (label, noffset + i)
            dirfname = os.path.join(basedir, fname)
            if use_subdir:
                subdir = str((noffset + i) // nmax_persubdir)
                dirfname = os.path.join(basedir, subdir, fname)
            with open(dirfname, "wb") as f:
                pickle.dump(data, f)
