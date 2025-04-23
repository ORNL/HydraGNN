import os
import pickle

from hydragnn.utils.print.print_utils import log

from hydragnn.utils.distributed import get_comm_size_and_rank
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset


class SerializedDataset(AbstractBaseDataset):
    """Serialized Dataset"""

    def __init__(self, basedir, datasetname, label, dist=False):
        """
        Parameters
        ----------
        basedir: basedir
        datasetname: datasets name
        label: label
        """
        super().__init__()

        self.basedir = basedir
        self.datasetname = datasetname
        self.label = label
        self.dist = dist

        if dist:
            _, rank = get_comm_size_and_rank()
            basename = "%s-%s-%d.pkl" % (datasetname, label, rank)
        else:
            basename = "%s-%s.pkl" % (datasetname, label)

        fname = os.path.join(basedir, basename)
        with open(fname, "rb") as f:
            self.minmax_node_feature = pickle.load(f)
            self.minmax_graph_feature = pickle.load(f)
            self.dataset = pickle.load(f)

        log("Pickle files:", self.label, len(self.dataset))

    def len(self):
        return len(self.dataset)

    def get(self, i):
        return self.dataset[i]


class SerializedWriter:
    """Serialized Dataset Writer"""

    def __init__(
        self,
        dataset,
        basedir,
        datasetname,
        label="total",
        minmax_node_feature=None,
        minmax_graph_feature=None,
        dist=False,
    ):
        """
        Parameters
        ----------
        dataset: locally owned datasets (should be iterable)
        basedir: basedir
        datasetname: datasets name
        label: label
        nmax: nmax in case of subdir
        minmax_node_feature: minmax_node_feature
        minmax_graph_feature: minmax_graph_feature
        dist: distributed or not
        """
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        if dist:
            _, rank = get_comm_size_and_rank()
            basename = "%s-%s-%d.pkl" % (datasetname, label, rank)
        else:
            basename = "%s-%s.pkl" % (datasetname, label)

        fname = os.path.join(basedir, basename)
        with open(fname, "wb") as f:
            pickle.dump(minmax_node_feature, f)
            pickle.dump(minmax_graph_feature, f)
            pickle.dump(dataset, f)
