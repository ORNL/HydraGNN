from mpi4py import MPI
import numpy as np

import torch
import torch_geometric.data

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset

try:
    import pyddstore as dds
except ImportError:
    pass

from hydragnn.utils.print.print_utils import log0
from hydragnn.utils.distributed import nsplit
from hydragnn.preprocess import update_predicted_values, update_atom_features

import hydragnn.utils.profiling_and_tracing.tracer as tr
from tqdm import tqdm


class DistDataset(AbstractBaseDataset):
    """Distributed datasets class"""

    def __init__(
        self,
        data,
        label,
        comm=MPI.COMM_WORLD,
        ddstore_width=None,
        local=False,
        var_config=None,
    ):
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
        if local:
            local_ns = len(data)
            local_ns_list = comm.allgather(local_ns)
            maxrank = np.argmax(local_ns_list).item()
            for i in tqdm(
                range(local_ns), desc="Loading", disable=(self.rank != maxrank)
            ):
                self.dataset.append(data[i])
            self.total_ns = comm.allreduce(local_ns, op=MPI.SUM)
        else:
            self.total_ns = len(data)
            rx = list(nsplit(range(len(data)), self.ddstore_comm_size))[
                self.ddstore_comm_rank
            ]
            for i in rx:
                self.dataset.append(data[i])

        self.keys = (
            self.dataset[0].keys()
            if callable(self.dataset[0].keys)
            else self.dataset[0].keys
        )
        self.keys = sorted(self.keys)
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
            ## vdim should be globally equal
            vdim = self.comm.allreduce(vdim, op=MPI.MAX)
            val = np.concatenate(arr_list, axis=vdim)
            if not val.flags["C_CONTIGUOUS"]:
                val = np.ascontiguousarray(val)
            assert val.data.c_contiguous

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
            log0(
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
        log0("DDStore total (GB):", nbytes / 1024 / 1024 / 1024)

        ## FIXME: Using the same routine in SimplePickleDataset. We need to make as a common function
        self.var_config = var_config

        if self.var_config is not None:
            self.input_node_features = self.var_config["input_node_features"]
            self.variables_type = self.var_config["type"]
            self.output_index = self.var_config["output_index"]
            self.graph_feature_dim = self.var_config["graph_feature_dims"]
            self.node_feature_dim = self.var_config["node_feature_dims"]

    def len(self):
        return self.total_ns

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

        self.update_data_object(data_object)
        return data_object
