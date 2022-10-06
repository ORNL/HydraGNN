import os
import glob
import pickle

import torch

from .print_utils import print_distributed, log, iterate_tqdm


class SimplePickleDataset(torch.utils.data.Dataset):
    """Simple Pickle Dataset"""

    def __init__(self, basedir, prefix, label):
        self.basedir = basedir
        self.prefix = prefix
        self.label = label

        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
        with open("%s/%s.meta" % (self.basedir, self.label)) as f:
            self.ndata = int(f.read())

        log("Pickle files:", self.label, self.ndata)

    def __len__(self):
        return self.ndata

    def __getitem__(self, idx):
        fname = "%s/%s-%s-%d.pk" % (self.basedir, self.prefix, self.label, idx)
        with open(fname, "rb") as f:
            data_object = pickle.load(f)
        return data_object

    def __iter__(self):
        for idx in range(self.ndata):
            yield self.__getitem__(idx)
