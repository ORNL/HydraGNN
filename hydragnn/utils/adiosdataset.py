from mpi4py import MPI
import time
import os
import glob

from .print_utils import print_distributed, log, iterate_tqdm

import numpy as np

try:
    import adios2 as ad2
except ImportError:
    pass

import torch_geometric.data
import torch

from multiprocessing.shared_memory import SharedMemory

try:
    import pyddstore as dds
except ImportError:
    pass


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


class AdiosWriter:
    """Adios class to write Torch Geometric graph data"""

    def __init__(self, filename, comm):
        """
        Parameters
        ----------
        filename: str
            adios filename
        comm: MPI_comm
            MPI communicator to use for Adios parallel writing
        """
        self.filename = filename
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dataset = dict()
        self.attributes = dict()
        self.adios = ad2.ADIOS()
        self.io = self.adios.DeclareIO(self.filename)

    def add_global(self, vname, arr):
        """
        Add attribute to be written in adios file

        Parameters
        ----------
        vname: str
            attribute name
        arr: numpy array
            attribute value to be written
        """
        self.attributes[vname] = arr

    def add(self, label, data):
        """
        Add labeled data to be written in adios file

        Parameters
        ----------
        label: str
            label name
        data: PyG data or list of PyG data
            PyG data to be saved
        """
        if label not in self.dataset:
            self.dataset[label] = list()

        if isinstance(data, list):
            self.dataset[label].extend(data)
        elif isinstance(data, torch_geometric.data.Data):
            self.dataset[label].append(data)
        else:
            raise Exception("Unsuppored data type yet.")

    def save(self):
        """
        Save data into an Adios file
        """
        t0 = time.time()
        log("Adios saving:", self.filename)
        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        total_ns = 0
        for label in self.dataset:
            if len(self.dataset[label]) < 1:
                continue
            ns = self.comm.allgather(len(self.dataset[label]))
            ns_offset = sum(ns[: self.rank])

            self.io.DefineAttribute("%s/ndata" % label, np.array(sum(ns)))
            total_ns += sum(ns)

            if len(self.dataset[label]) > 0:
                data = self.dataset[label][0]
                self.io.DefineAttribute("%s/keys" % label, data.keys)
                keys = sorted(data.keys)

            for k in keys:
                arr_list = [data[k].cpu().numpy() for data in self.dataset[label]]
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

        self.io.DefineAttribute("total_ndata", np.array(total_ns))
        for vname in self.attributes:
            self.io.DefineAttribute(vname, self.attributes[vname])

        self.writer.Close()
        t1 = time.time()
        log("Adios saving time (sec): ", (t1 - t0))


class AdiosDataset(torch.utils.data.Dataset):
    """Adios dataset class"""

    def __init__(
        self,
        filename,
        label,
        comm,
        preload=False,
        shmem=False,
        enable_cache=False,
        distds=False,
    ):
        """
        Parameters
        ----------
        filename: str
            adios filename
        label: str
            data label to load, such as trainset, testing, and valset
        comm: MPI_comm
            MPI communicator
        preload: bool, optional
            Option to preload all the dataset into a memory
        shmem: bool, optional
            Option to use shmem to share data between processes in the same node
        enable_cache: bool, optional
            Option to cache data object which was already read
        distds: bool, optional
            Option to use Distributed Data Store
        """
        t0 = time.time()
        self.filename = filename
        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        self.nrank_per_node = self.comm.Get_size()
        if os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"):
            ## Summit
            self.nrank_per_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        elif os.getenv("SLURM_NTASKS_PER_NODE"):
            ## Perlmutter
            self.nrank_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        elif os.getenv("SLURM_TASKS_PER_NODE"):
            ## Crusher
            self.nrank_per_node = int(os.environ["SLURM_TASKS_PER_NODE"].split("(")[0])

        self.data = dict()
        self.preload = preload
        ## Preflight option: This is experimental.
        ## Get the index of data to load first and call "populate" to load data all together
        self.preflight = False
        self.preflight_list = list()
        self.shmem = shmem
        self.smm = None
        if self.shmem:
            self.shm = dict()
            self.local_comm = self.comm.Split(
                self.rank // self.nrank_per_node, self.rank
            )
            self.local_rank = self.local_comm.Get_rank()
            log("local_rank", self.local_rank)

        self.enable_cache = enable_cache
        self.cache = dict()
        self.ddstore = None
        self.distds = distds
        if self.distds:
            self.ddstore = dds.PyDDStore(comm)
        log("Adios reading:", self.filename)
        with ad2.open(self.filename, "r", MPI.COMM_SELF) as f:
            self.vars = f.available_variables()
            self.attrs = f.available_attributes()
            self.keys = f.read_attribute_string("%s/keys" % label)
            self.ndata = f.read_attribute("%s/ndata" % label).item()
            if "minmax_graph_feature" in self.attrs:
                self.minmax_graph_feature = f.read_attribute(
                    "minmax_graph_feature"
                ).reshape((2, -1))
            if "minmax_node_feature" in self.attrs:
                self.minmax_node_feature = f.read_attribute(
                    "minmax_node_feature"
                ).reshape((2, -1))

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
                        self.shm[k] = SharedMemory(name=name, create=False)

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
                elif self.distds:
                    ## Calculate local portion
                    shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                    ishape = [int(x.strip(",")) for x in shape.strip().split()]
                    start = [
                        0,
                    ] * len(ishape)
                    count = ishape
                    vdim = self.variable_dim[k]

                    rx = list(nsplit(self.variable_count[k], self.comm_size))
                    start[vdim] = sum([sum(x) for x in rx[: self.rank]])
                    count[vdim] = sum(rx[self.rank])

                    # Read only local portion
                    vname = "%s/%s" % (label, k)
                    self.data[k] = f.read(vname, start, count)
                    if vdim > 0:
                        self.data[k] = np.moveaxis(self.data[k], vdim, 0)
                        self.data[k] = np.ascontiguousarray(self.data[k])
                    self.ddstore.add(vname, self.data[k])
                    log(
                        "DDStore add:",
                        (
                            vname,
                            start,
                            count,
                            vdim,
                            self.data[k].dtype,
                            self.data[k].shape,
                            self.data[k].sum(),
                        ),
                    )
            t2 = time.time()
            log("Adios reading time (sec): ", (t2 - t0))
        t1 = time.time()
        log("Data loading time (sec): ", (t1 - t0))

        if not self.preload and not self.shmem:
            self.f = ad2.open(self.filename, "r", MPI.COMM_SELF)

    def __len__(self):
        """
        Return the total size of dataset
        """
        return self.ndata

    def __getitem__(self, idx):
        """
        Get data with a given index
        """
        if self.preflight:
            ## Preflight option: This is experimental.
            ## Collect only the indeces of data to load. "populate" will load the data later.
            self.preflight_list.append(idx)
            return torch_geometric.data.Data()
        if idx in self.cache:
            ## Load data from cached buffer
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
                    ## Read from memory when preloaded or used with shmem
                    slice_list = list()
                    for n0, n1 in zip(start, count):
                        slice_list.append(slice(n0, n0 + n1))
                    val = self.data[k][tuple(slice_list)]
                elif self.distds:
                    vname = "%s/%s" % (self.label, k)
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

                    val = np.zeros(count, dtype=dtype)
                    ## vdim should be the first dim for DDStore
                    if vdim > 0:
                        val = np.moveaxis(val, vdim, 0)
                        val = np.ascontiguousarray(val)

                    offset = start[vdim]
                    self.ddstore.get(vname, val, offset)
                    if vdim > 0:
                        val = np.moveaxis(val, 0, vdim)
                        val = np.ascontiguousarray(val)
                else:
                    ## Reading data directly from disk
                    # log("getitem out-of-memory:", self.label, k, idx)
                    val = self.f.read("%s/%s" % (self.label, k), start, count)

                v = torch.tensor(val)
                exec("data_object.%s = v" % (k))
            if self.enable_cache:
                self.cache[idx] = data_object
        return data_object

    def unlink(self):
        """
        Unlink shmem link
        """
        if self.shmem:
            for k in self.keys:
                self.shm[k].close()
                if self.local_rank == 0:
                    self.shm[k].unlink()

    def __del__(self):
        if self.distds:
            self.ddstore.free()
        if not self.preload and not self.shmem:
            self.f.close()
        try:
            self.unlink(self)
        except:
            pass

    def populate(self):
        """
        Populate data when preflight is on
        """
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
            for idx in iterate_tqdm(sorted(filtered), 2, desc="AdiosWriter populate"):
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

    def __iter__(self):
        for idx in range(self.ndata):
            yield self.__getitem__(idx)
