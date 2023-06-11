from mpi4py import MPI
import numpy as np

import torch
import torch_geometric.data

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

try:
    import pyddstore as dds
except ImportError:
    pass

from hydragnn.utils.print_utils import log
from hydragnn.utils import nsplit

import hydragnn.utils.tracer as tr


class DistDataset(AbstractBaseDataset):
    """Distributed dataset class"""

    def __init__(self, data, label, comm=MPI.COMM_WORLD, ddstore_width=None):
        super().__init__()

        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        self.ddstore = dds.PyDDStore(self.ddstore_comm)

        ## set total before set subset
        self.total_ns = len(data)
        rx = list(nsplit(range(len(data)), self.ddstore_comm_size))[
            self.ddstore_comm_rank
        ]
        for i in rx:
            self.dataset.append(data[i])

        self.keys = sorted(self.dataset[0].keys)
        self.variable_shape = dict()
        self.variable_dim = dict()
        self.variable_dtype = dict()
        self.variable_count = dict()
        self.variable_offset = dict()
        self.data = dict()
        nbytes = 0
        for k in self.keys:
            arr_list = [data[k].cpu().numpy() for data in self.dataset]
            m0 = np.min([x.shape for x in arr_list], axis=0)
            m1 = np.max([x.shape for x in arr_list], axis=0)
            vdims = list()
            for i in range(len(m0)):
                if m0[i] != m1[i]:
                    vdims.append(i)
            ## We can handle only single variable dimension.
            assert len(vdims) < 2
            vdim = 0
            if len(vdims) > 0:
                vdim = vdims[0]
            val = np.concatenate(arr_list, axis=vdim)
            assert val.data.contiguous

            self.variable_shape[k] = val.shape
            self.variable_dim[k] = vdim
            self.variable_dtype[k] = val.dtype

            vcount = np.array([x.shape[vdim] for x in arr_list])
            assert len(vcount) == len(self.dataset)
            vcount_list = self.ddstore_comm.allgather(vcount)
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
                    (val.size * val.itemsize) / 1024 / 1024 / 1024,
                ),
            )
            nbytes += val.size * val.itemsize
        log("DDStore total (GB):", nbytes / 1024 / 1024 / 1024)

    def len(self):
        return self.total_ns

    @tr.profile("get")
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
