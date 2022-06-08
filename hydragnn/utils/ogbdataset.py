from mpi4py import MPI
import time
import os

from .print_utils import print_distributed, log, iterate_tqdm

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

try:
    import gptl4py as gp
except ImportError:
    import hydragnn.utils.gptl4py_dummy as gp


class AdiosOGB:
    def __init__(self, filename, comm):
        self.filename = filename
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dataset = dict()
        self.adios = ad2.ADIOS()
        self.io = self.adios.DeclareIO(self.filename)

    def add(self, label, data: torch_geometric.data.Data):
        if label not in self.dataset:
            self.dataset[label] = list()
        if isinstance(data, list):
            self.dataset[label].extend(data)
        elif isinstance(data, torch_geometric.data.Data):
            self.dataset[label].append(data)
        else:
            raise Exception("Unsuppored data type yet.")

    def save(self):
        t0 = time.time()
        log("Adios saving:", self.filename)
        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        for label in self.dataset:
            if len(self.dataset[label]) < 1:
                continue
            ns = self.comm.allgather(len(self.dataset[label]))
            ns_offset = sum(ns[: self.rank])

            self.io.DefineAttribute("%s/ndata" % label, np.array(sum(ns)))
            if len(self.dataset[label]) > 0:
                data = self.dataset[label][0]
                self.io.DefineAttribute("%s/keys" % label, data.keys)
                keys = sorted(data.keys)

            for k in keys:
                arr_list = [data[k].cpu().numpy() for data in self.dataset[label]]
                m0 = np.min([x.shape for x in arr_list], axis=0)
                m1 = np.max([x.shape for x in arr_list], axis=0)
                wh = np.where(m0 != m1)[0]
                assert len(wh) < 2
                vdim = wh[0] if len(wh) == 1 else 1
                val = np.concatenate(arr_list, axis=vdim)
                assert val.data.contiguous
                shape_list = self.comm.allgather(list(val.shape))
                offset = [
                    0,
                ] * len(val.shape)
                for i in range(self.rank):
                    offset[vdim] += shape_list[i][vdim]
                global_shape = shape_list[0]
                for i in range(1, self.size):
                    global_shape[vdim] += shape_list[i][vdim]
                # log ("k,val shape", k, global_shape, offset, val.shape)
                var = self.io.DefineVariable(
                    "%s/%s" % (label, k),
                    val,
                    global_shape,
                    offset,
                    val.shape,
                    ad2.ConstantDims,
                )
                self.writer.Put(var, val, ad2.Mode.Sync)

                self.io.DefineAttribute(
                    "%s/%s/variable_dim" % (label, k), np.array(vdim)
                )

                vcount = np.array([x.shape[vdim] for x in arr_list])
                assert len(vcount) == len(self.dataset[label])

                offset_arr = np.zeros_like(vcount)
                offset_arr[1:] = np.cumsum(vcount)[:-1]
                offset_arr += offset[vdim]

                var = self.io.DefineVariable(
                    "%s/%s/variable_count" % (label, k),
                    vcount,
                    [
                        sum(ns),
                    ],
                    [
                        ns_offset,
                    ],
                    [
                        len(vcount),
                    ],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, vcount, ad2.Mode.Sync)

                var = self.io.DefineVariable(
                    "%s/%s/variable_offset" % (label, k),
                    offset_arr,
                    [
                        sum(ns),
                    ],
                    [
                        ns_offset,
                    ],
                    [
                        len(vcount),
                    ],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, offset_arr, ad2.Mode.Sync)

        self.writer.Close()
        t1 = time.time()
        log("Adios saving time (sec): ", (t1 - t0))


class OGBDataset(torch.utils.data.Dataset):
    def __init__(
        self, filename, label, comm, preload=True, shmem=False, enable_cache=True
    ):
        t0 = time.time()
        self.filename = filename
        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()

        self.nrank_per_node = self.comm.Get_size()
        if os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"):
            ## Summit
            self.nrank_per_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        if os.getenv("SLURM_NTASKS_PER_NODE"):
            ## Perlmutter
            self.nrank_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])

        self.local_rank = self.rank % self.nrank_per_node
        log("local_rank", self.local_rank)

        self.data = dict()
        self.preload = preload
        self.preflight = False
        self.preflight_list = list()
        self.shmem = shmem
        self.smm = None
        if self.shmem:
            self.shm = dict()
            self.local_comm = self.comm.Split(
                self.rank // self.nrank_per_node, self.rank
            )

        self.enable_cache = enable_cache
        self.cache = dict()
        log("Adios reading:", self.filename)

        with ad2.open(self.filename, "r", MPI.COMM_SELF) as f:
            self.vars = f.available_variables()
            self.keys = f.read_attribute_string("%s/keys" % label)
            self.ndata = f.read_attribute("%s/ndata" % label).item()

            self.variable_count = dict()
            self.variable_offset = dict()
            self.variable_dim = dict()

            for k in self.keys:
                self.variable_count[k] = f.read("%s/%s/variable_count" % (label, k))
                self.variable_offset[k] = f.read("%s/%s/variable_offset" % (label, k))
                self.variable_dim[k] = f.read_attribute(
                    "%s/%s/variable_dim" % (label, k)
                ).item()
                if self.preload:
                    ## load full data first
                    self.data[k] = f.read("%s/%s" % (label, k))
                elif self.shmem:
                    if self.local_rank == 0:
                        adios = ad2.ADIOS()
                        io = adios.DeclareIO("ogb_read")
                        reader = io.Open(self.filename, ad2.Mode.Read, MPI.COMM_SELF)
                        var = io.InquireVariable("%s/%s" % (label, k))

                        if var.Type() == "double":
                            dtype = np.float64
                        elif var.Type() == "float":
                            dtype = np.float32
                        elif var.Type() == "int32_t":
                            dtype = np.int32
                        elif var.Type() == "int64_t":
                            dtype = np.int64
                        else:
                            raise ValueError(var.Type())

                        nbytes = np.prod(var.Shape()) * np.dtype(dtype).itemsize
                        self.shm[k] = SharedMemory(create=True, size=nbytes)
                        arr = np.ndarray(
                            var.Shape(), dtype=dtype, buffer=self.shm[k].buf
                        )
                        reader.Get(var, arr, ad2.Mode.Sync)
                        reader.Close()
                        self.data[k] = arr
                        self.local_comm.bcast(self.shm[k].name, root=0)
                    else:
                        name = None
                        name = self.local_comm.bcast(name, root=0)
                        self.shm[k] = SharedMemory(name=name)

                        shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                        ishape = [int(x.strip(",")) for x in shape.strip().split()]

                        vartype = self.vars["%s/%s" % (self.label, k)]["Type"]
                        if vartype == "double":
                            dtype = np.float64
                        elif vartype == "float":
                            dtype = np.float32
                        elif vartype == "int32_t":
                            dtype = np.int32
                        elif vartype == "int64_t":
                            dtype = np.int64
                        else:
                            raise ValueError(vartype)

                        arr = np.ndarray(ishape, dtype=dtype, buffer=self.shm[k].buf)
                        self.data[k] = arr

            t2 = time.time()
            log("Adios reading time (sec): ", (t2 - t0))
        t1 = time.time()
        log("Data loading time (sec): ", (t1 - t0))

        if not self.preload and not self.shmem:
            self.f = ad2.open(self.filename, "r", MPI.COMM_SELF)

    def __len__(self):
        return self.ndata

    @gp.profile
    def __getitem__(self, idx):
        if self.preflight:
            self.preflight_list.append(idx)
            return torch_geometric.data.Data()
        if idx in self.cache:
            data_object = self.cache[idx]
        else:
            data_object = torch_geometric.data.Data()
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = [int(x.strip(",")) for x in shape.strip().split()]
                start = [
                    0,
                ] * len(ishape)
                count = ishape
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][idx]
                count[vdim] = self.variable_count[k][idx]
                if self.preload or self.shmem:
                    slice_list = list()
                    for n0, n1 in zip(start, count):
                        slice_list.append(slice(n0, n0 + n1))
                    val = self.data[k][tuple(slice_list)]
                else:
                    # log("getitem out-of-memory:", self.label, k, idx)
                    val = self.f.read("%s/%s" % (self.label, k), start, count)

                v = torch.tensor(val)
                exec("data_object.%s = v" % (k))
            if self.enable_cache:
                self.cache[idx] = data_object
        return data_object

    def unlink(self):
        if self.shmem:
            for k in self.keys:
                self.shm[k].close()
                if self.local_rank == 0:
                    self.shm[k].unlink()

    def __del__(self):
        if not self.preload and not self.shmem:
            self.f.close()
        try:
            self.unlink(self)
        except:
            pass

    @gp.profile
    def populate(self):
        dn = 2_000_000
        self._data = dict()
        for i in range(0, self.ndata, dn):
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = [int(x.strip(",")) for x in shape.strip().split()]
                start = [
                    0,
                ] * len(ishape)
                count = list(ishape)
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][i]
                count[vdim] = self.variable_count[k][i : i + dn].sum()

                with ad2.open(self.filename, "r", MPI.COMM_SELF) as f:
                    self._data[k] = f.read("%s/%s" % (self.label, k), start, count)

            filtered = filter(lambda x: x >= i and x < i + dn, self.preflight_list)
            for idx in iterate_tqdm(sorted(filtered), 2, desc="AdiosOGB populate"):
                data_object = torch_geometric.data.Data()
                for k in self.keys:
                    shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                    ishape = [int(x.strip(",")) for x in shape.strip().split()]
                    start = [
                        0,
                    ] * len(ishape)
                    count = list(ishape)
                    vdim = self.variable_dim[k]
                    start[vdim] = (
                        self.variable_offset[k][idx] - self.variable_offset[k][i]
                    )
                    count[vdim] = self.variable_count[k][idx]
                    slice_list = list()
                    for n0, n1 in zip(start, count):
                        slice_list.append(slice(n0, n0 + n1))
                    val = self._data[k][tuple(slice_list)]

                    v = torch.tensor(val)
                    exec("data_object.%s = v" % (k))
                self.cache[idx] = data_object

        for k in self.keys:
            del self._data[k]

        self.preflight = False
