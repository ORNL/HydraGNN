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
from io import BytesIO
import os


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
        ddstore_store_per_sample=True,  ## True for per-sample saving. False for feature-first saving
    ):
        super().__init__()

        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )

        ## FIXME
        ddstore_method = int(os.getenv("HYDRAGNN_DDSTORE_METHOD", "0"))
        print("DDStore method:", ddstore_method)
        if ddstore_method == 1:
            # Using libfabric. Need a map each rank to a network interface
            iface = system = os.getenv("FABRIC_IFACE", None)
            if iface is None:
                system = os.getenv("LMOD_SYSTEM_NAME", "none")
                if system == "frontier":
                    gpu_id = int(os.getenv("SLURM_LOCALID", "0"))
                    os.environ["FABRIC_IFACE"] = f"hsn{gpu_id//2}"
                elif system == "perlmutter":
                    gpu_id = int(os.getenv("SLURM_LOCALID", "0"))
                    os.environ["FABRIC_IFACE"] = f"hsn{gpu_id}"
                elif system == "aurora":
                    ## FIMXE
                    pass

            print("FABRIC_IFACE:", os.environ["FABRIC_IFACE"])

        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        self.ddstore = dds.PyDDStore(self.ddstore_comm, method=ddstore_method)
        self.ddstore_store_per_sample = ddstore_store_per_sample

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
            arr_list = list()
            for data in self.dataset:
                if isinstance(data[k], torch.Tensor):
                    arr_list.append(data[k].cpu().numpy())
                elif isinstance(data[k], np.ndarray):
                    arr_list.append(data[k])
                elif isinstance(data[k], (np.floating, np.integer)):
                    arr_list.append(np.array((data[k],)))
                elif isinstance(data[k], str):
                    arr = np.frombuffer(data[k].encode("utf-8"), dtype=np.uint8)
                    arr_list.append(arr)
                else:
                    print("Error: type(data[k]):", label, k, type(data[k]))
                    raise NotImplementedError(
                        "Not supported: not tensor nor numpy array"
                    )
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

            if not self.ddstore_store_per_sample:
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

        ## per-sample approach
        if self.ddstore and self.ddstore_store_per_sample:
            buf = BytesIO()
            rx = list(nsplit(list(range(self.total_ns)), self.ddstore_comm_size))[
                self.ddstore_comm_rank
            ]
            local_record_count = list()
            for idx in rx:
                data_object = dict()
                for k in self.keys:
                    count = list(self.variable_shape[k])
                    start = [
                        0,
                    ] * len(count)
                    vdim = self.variable_dim[k]
                    dtype = self.variable_dtype[k]
                    start[vdim] = (
                        self.variable_offset[k][idx] - self.variable_offset[k][rx[0]]
                    )
                    count[vdim] = self.variable_count[k][idx]
                    if vdim > 0:
                        start.insert(0, start.pop(vdim))
                        count.insert(0, count.pop(vdim))
                    assert start[0] + count[0] <= (self.data[k].shape)[0], (
                        start[0],
                        count[0],
                        (self.data[k].shape)[0],
                    )

                    slices = tuple(slice(s, s + c) for s, c in zip(start, count))
                    data = self.data[k][slices]
                    ## reset vdim
                    if vdim > 0:
                        data = np.moveaxis(data, 0, vdim)
                    data_object[k] = data

                dtype = np.dtype(
                    [(k, v.dtype, v.shape) for k, v in data_object.items()]
                )
                data_tuples = [
                    tuple([v.tolist() for v in data_object.values()]),
                ]
                record_array = np.array(data_tuples, dtype=dtype)
                assert dtype.itemsize == record_array.nbytes
                local_record_count.append(dtype.itemsize)
                buf.write(record_array.tobytes())

            record_count = self.comm.allgather(local_record_count)
            self.record_count = np.hstack(record_count)
            self.record_offset = self.record_count.cumsum()

            arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            vname = "%s/%s" % (label, "record_array")
            self.ddstore.add(vname, arr)
            nbytes += arr.nbytes

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
        if self.ddstore_store_per_sample:
            data_object = torch_geometric.data.Data()
            dtype_list = list()
            for k in self.keys:
                count = list(self.variable_shape[k])
                start = [
                    0,
                ] * len(count)
                vdim = self.variable_dim[k]
                dtype = self.variable_dtype[k]
                start[vdim] = self.variable_offset[k][idx]
                count[vdim] = self.variable_count[k][idx]

                dtype_tuple = (k, dtype, count)
                dtype_list.append(dtype_tuple)

            dtype = np.dtype(dtype_list)
            val = np.zeros(dtype.itemsize, dtype=np.uint8)
            offset = 0 if idx == 0 else self.record_offset[idx - 1]
            vname = "%s/%s" % (self.label, "record_array")
            self.ddstore.get(vname, val, offset)
            val = val.view(dtype)[0]
            for k in val.dtype.names:
                v = torch.tensor(val[k])
                exec("data_object.%s = v" % (k))

            return data_object

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
