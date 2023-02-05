from mpi4py import MPI
import numpy as np

import torch
import torch_geometric.data

from hydragnn.utils.basedataset import BaseDataset

try:
    import pyddstore as dds
except ImportError:
    pass

from hydragnn.utils.print_utils import log


class DistDataset(BaseDataset):
    """Distributed dataset class"""

    def __init__(self, data, label, comm=MPI.COMM_WORLD):
        super().__init__()

        if isinstance(data, list):
            self.dataset.extend(data)
        else:
            self.dataset.extend(list(data))

        self.label = label
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.ddstore = dds.PyDDStore(comm)

        ns = self.comm.allgather(len(self.dataset))
        ns_offset = sum(ns[: self.rank])
        self.total_ns = sum(ns)

        self.keys = sorted(self.dataset[0].keys)
        self.variable_shape = dict()
        self.variable_dim = dict()
        self.variable_dtype = dict()
        self.variable_count = dict()
        self.variable_offset = dict()
        self.data = dict()
        for k in self.keys:
            arr_list = [data[k].cpu().numpy() for data in self.dataset]
            m0 = np.min([x.shape for x in arr_list], axis=0)
            m1 = np.max([x.shape for x in arr_list], axis=0)
            wh = np.where(m0 != m1)[0]
            assert len(wh) < 2
            vdim = wh[0] if len(wh) == 1 else 1
            val = np.concatenate(arr_list, axis=vdim).copy()

            self.variable_shape[k] = val.shape
            self.variable_dim[k] = vdim
            self.variable_dtype[k] = val.dtype

            vcount = np.array([x.shape[vdim] for x in arr_list])
            assert len(vcount) == len(self.dataset)
            vcount_list = self.comm.allgather(vcount)
            vcount = np.hstack(vcount_list)
            self.variable_count[k] = vcount

            offset_arr = np.zeros_like(vcount)
            offset_arr[1:] = np.cumsum(vcount)[:-1]
            self.variable_offset[k] = offset_arr

            if vdim > 0:
                val = np.moveaxis(val, vdim, 0)
                val = np.ascontiguousarray(val)
            assert val.data.contiguous
            self.data[k] = val

            vname = "%s/%s" % (label, k)
            self.ddstore.add(vname, val)
            log(
                "DDStore add:",
                (
                    vname,
                    vdim,
                    val.dtype,
                    val.shape,
                    val.sum(),
                ),
            )

    def len(self):
        return self.total_ns

    def get(self, idx):
        data_object = torch_geometric.data.Data()
        for k in self.keys:
            count = list(self.variable_shape[k])
            vdim = self.variable_dim[k]
            dtype = self.variable_dtype[k]
            offset = self.variable_offset[k][idx]
            count[vdim] = self.variable_count[k][idx]
            val = np.zeros(count, dtype=dtype)
            ## vdim should be the first dim for DDStore
            if vdim > 0:
                val = np.moveaxis(val, vdim, 0)
                val = np.ascontiguousarray(val)
                assert val.data.contiguous
            vname = "%s/%s" % (self.label, k)
            self.ddstore.get(vname, val, offset)
            if vdim > 0:
                val = np.moveaxis(val, 0, vdim)
                val = np.ascontiguousarray(val)
            v = torch.tensor(val)
            exec("data_object.%s = v" % (k))
        return data_object
