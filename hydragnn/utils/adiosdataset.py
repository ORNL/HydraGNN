from mpi4py import MPI
import pickle
import time
import os
import glob

from .print_utils import print_distributed, log, log0, iterate_tqdm

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

import hydragnn.utils.tracer as tr

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils import nsplit
from hydragnn.preprocess import update_predicted_values, update_atom_features


# A solution for bcast val > 2GB for mpi4py < 3.1.0
# For mpi4py >= 3.1.0, please use: https://mpi4py.readthedocs.io/en/stable/mpi4py.util.pkl5.html
def bulk_bcast(comm, val, root=0):
    rank = comm.Get_rank()

    # Define the maximum chunk size (2GB)
    MAX_CHUNK_SIZE = 2 * 1000 * 1000 * 1000  # ~2GB

    if rank == root:
        # Serialize the value
        serialized_val = pickle.dumps(val)
        val_size = len(serialized_val)
        num_chunks = (val_size + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
    else:
        serialized_val = None
        val_size = None
        num_chunks = None

    # Broadcast the size and number of chunks
    val_size = comm.bcast(val_size, root=root)
    num_chunks = comm.bcast(num_chunks, root=root)

    received_data = bytearray(val_size)

    for i in range(num_chunks):
        if rank == root:
            chunk_start = i * MAX_CHUNK_SIZE
            chunk_end = min((i + 1) * MAX_CHUNK_SIZE, val_size)
            chunk = serialized_val[chunk_start:chunk_end]
        else:
            chunk = None

        chunk = comm.bcast(chunk, root=root)

        chunk_start = i * MAX_CHUNK_SIZE
        received_data[chunk_start : chunk_start + len(chunk)] = chunk

    # Deserialize the received data
    if rank != root:
        val = pickle.loads(received_data)

    return val


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
        elif isinstance(data, AbstractBaseDataset):
            self.dataset[label] = data
        else:
            raise Exception("Unsuppored data type yet.")

    def save(self):
        """
        Save data into an Adios file
        """
        t0 = time.time()
        log0("Adios saving:", self.filename)
        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        total_ns = 0
        for label in self.dataset:
            if len(self.dataset[label]) == 0:
                ## If there is no data to save, simply do empty operations as follows.
                ## This process will call multiple allgather in a sequential order
                ns = self.comm.allgather(len(self.dataset[label]))

                keys_list = self.comm.allgather([])
                for keys in keys_list:
                    if len(keys) > 0:
                        break

                for k in keys:
                    vdim = self.comm.allreduce(0, op=MPI.MAX)
                    shape_list = self.comm.allgather([])

                continue
            ns = self.comm.allgather(len(self.dataset[label]))
            ns_offset = sum(ns[: self.rank])

            self.io.DefineAttribute("%s/ndata" % label, np.array(sum(ns)))
            total_ns += sum(ns)

            if len(self.dataset[label]) > 0:
                data = self.dataset[label][0]
                keys = data.keys() if callable(data.keys) else data.keys
                self.io.DefineAttribute("%s/keys" % label, keys)
                keys = sorted(keys)
                self.comm.allgather(keys)

            for k in keys:
                arr_list = list()
                for data in self.dataset[label]:
                    if isinstance(data[k], torch.Tensor):
                        arr_list.append(data[k].cpu().numpy())
                    elif isinstance(data[k], np.ndarray):
                        arr_list.append(data[k])
                    elif isinstance(data[k], (np.floating, np.integer)):
                        arr_list.append(np.array((data[k],)))
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
                ## We can handle only one varying dimension.
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

                shape_list = self.comm.allgather(list(val.shape))
                offset = [
                    0,
                ] * len(val.shape)
                for i in range(self.rank):
                    if shape_list[i]:
                        offset[vdim] += shape_list[i][vdim]
                global_shape = shape_list[0]
                for i in range(1, self.size):
                    if shape_list[i]:
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
        log0("Adios saving time (sec): ", (t1 - t0))


class AdiosDataset(AbstractBaseDataset):
    """Adios dataset class"""

    def __init__(
        self,
        filename,
        label,
        comm,
        preload=False,
        shmem=False,
        enable_cache=False,
        ddstore=False,
        ddstore_width=None,
        var_config=None,
        subset_istart=None,
        subset_iend=None,
        keys=None,
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
        ddstore: bool, optional
            Option to use Distributed Data Store
        """
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
        self.ddstore = ddstore
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        if self.ddstore:
            self.ddstore_comm = self.comm.Split(
                self.rank // self.ddstore_width, self.rank
            )
            self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
            self.ddstore_comm_size = self.ddstore_comm.Get_size()
            log0(
                "ddstore_comm_rank,ddstore_comm_size",
                self.ddstore_comm_rank,
                self.ddstore_comm_size,
            )
            self.ddstore = dds.PyDDStore(self.ddstore_comm)

        self.subset = None
        self.subset_istart = subset_istart
        self.subset_iend = subset_iend
        if self.subset_istart is not None and self.subset_iend is not None:
            self.subset = list(range(subset_istart, subset_iend))

        log0("Adios reading:", self.filename)
        adios_read_time = 0.0
        ddstore_time = 0.0
        t0 = time.time()
        with ad2.open(self.filename, "r", self.comm) as f:
            f.__next__()
            t1 = time.time()
            self.vars = f.available_variables()
            self.attrs = f.available_attributes()
            self.keys = self.read_attribute_string0(f, "%s/keys" % label)
            if keys is not None:
                self.setkeys(keys)
            self.ndata = self.read_attribute0(f, "%s/ndata" % label).item()
            if "minmax_graph_feature" in self.attrs:
                self.minmax_graph_feature = self.read_attribute0(
                    f, "minmax_graph_feature"
                ).reshape((2, -1))
            if "minmax_node_feature" in self.attrs:
                self.minmax_node_feature = self.read_attribute0(
                    f, "minmax_node_feature"
                ).reshape((2, -1))
            if "pna_deg" in self.attrs:
                self.pna_deg = self.read_attribute0(f, "pna_deg")
            t2 = time.time()
            log0("Read attr time (sec): ", (t2 - t1))

            self.variable_count = dict()
            self.variable_offset = dict()
            self.variable_dim = dict()
            self.subset_start = dict()
            self.subset_count = dict()

            nbytes = 0
            t3 = time.time()
            for k in self.keys:
                self.variable_count[k] = self.read0(
                    f, "%s/%s/variable_count" % (label, k)
                )
                log0(
                    "read and bcast:",
                    "%s/%s/variable_count" % (label, k),
                    time.time() - t3,
                )
                self.variable_offset[k] = self.read0(
                    f, "%s/%s/variable_offset" % (label, k)
                )
                log0(
                    "read and bcast:",
                    "%s/%s/variable_offset" % (label, k),
                    time.time() - t3,
                )
                self.variable_dim[k] = self.read_attribute0(
                    f, "%s/%s/variable_dim" % (label, k)
                ).item()
                log0(
                    "read and bcast:",
                    "%s/%s/variable_dim" % (label, k),
                    time.time() - t3,
                )
                if self.preload:
                    ##  preload data
                    if self.subset is not None:
                        shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                        ishape = [int(x.strip(",")) for x in shape.strip().split()]
                        start = [
                            0,
                        ] * len(ishape)
                        count = ishape
                        vdim = self.variable_dim[k]
                        start[vdim] = self.variable_offset[k][subset_istart]
                        count[vdim] = sum(
                            self.variable_count[k][subset_istart:subset_iend]
                        )
                        vname = "%s/%s" % (label, k)
                        self.data[k] = f.read(vname, start, count)
                        self.subset_start[k] = start
                        self.subset_count[k] = count
                    else:
                        self.data[k] = self.read0(f, "%s/%s" % (label, k))
                elif self.shmem:
                    if self.local_rank == 0:
                        adios = ad2.ADIOS()
                        io = adios.DeclareIO("ogb_read")
                        reader = io.Open(self.filename, ad2.Mode.Read, self.comm)
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
                elif self.ddstore:
                    ## Calculate local portion
                    shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                    ishape = [int(x.strip(",")) for x in shape.strip().split()]
                    start = [
                        0,
                    ] * len(ishape)
                    count = ishape
                    vdim = self.variable_dim[k]

                    rx = list(nsplit(self.variable_count[k], self.ddstore_comm_size))
                    start[vdim] = sum([sum(x) for x in rx[: self.ddstore_comm_rank]])
                    count[vdim] = sum(rx[self.ddstore_comm_rank])

                    # Read only local portion
                    vname = "%s/%s" % (label, k)
                    self.data[k] = f.read(vname, start, count)
                    if vdim > 0:
                        self.data[k] = np.moveaxis(self.data[k], vdim, 0)
                        self.data[k] = np.ascontiguousarray(self.data[k])

                    t4 = time.time()
                    self.ddstore.add(vname, self.data[k])
                    t5 = time.time()
                    ddstore_time += t5 - t4

                    log0(
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
                    nbytes += self.data[k].size * self.data[k].itemsize
            t6 = time.time()
            log0("Overall time (sec): ", t6 - t1)
            log0("DDStore adding time (sec): ", ddstore_time)
            if self.ddstore:
                log0("DDStore total (GB):", nbytes / 1024 / 1024 / 1024)

        t7 = time.time()
        log0("Data loading time (sec): ", (t7 - t0))

        if not self.preload and not self.shmem:
            self.f = ad2.open(self.filename, "r", self.comm)
            self.f.__next__()

        ## FIXME: Using the same routine in SimplePickleDataset. We need to make as a common function
        self.var_config = var_config

        if self.var_config is not None:
            self.input_node_features = self.var_config["input_node_features"]
            self.variables_type = self.var_config["type"]
            self.output_index = self.var_config["output_index"]
            self.graph_feature_dim = self.var_config["graph_feature_dims"]
            self.node_feature_dim = self.var_config["node_feature_dims"]

    ## rank=0 read and bcast
    def read0(self, f, vname):
        if self.rank == 0:
            val = f.read(vname)
        else:
            val = None
        val = bulk_bcast(self.comm, val, root=0)
        return val

    def read_attribute0(self, f, vname):
        if self.rank == 0:
            val = f.read_attribute(vname)
        else:
            val = None
        val = bulk_bcast(self.comm, val, root=0)
        return val

    def read_attribute_string0(self, f, vname):
        if self.rank == 0:
            val = f.read_attribute_string(vname)
        else:
            val = None
        val = bulk_bcast(self.comm, val, root=0)
        return val

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

    def setkeys(self, keys):
        for k in keys:
            assert k in self.keys
        self.keys = keys

    def setsubset(self, subset_istart, subset_iend, preload=False):
        self.subset_istart = subset_istart
        self.subset_iend = subset_iend
        if self.subset_istart is not None and self.subset_iend is not None:
            self.subset = list(range(subset_istart, subset_iend))

        if preload and self.subset is not None:
            t0 = time.time()
            self.preload = preload
            log0("Adios preloading:", self.filename)
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = [int(x.strip(",")) for x in shape.strip().split()]
                start = [
                    0,
                ] * len(ishape)
                count = ishape
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][subset_istart]
                count[vdim] = sum(self.variable_count[k][subset_istart:subset_iend])
                vname = "%s/%s" % (self.label, k)
                self.data[k] = self.f.read(vname, start, count)
                self.subset_start[k] = start
                self.subset_count[k] = count
            self.f.close()
            t2 = time.time()
            log0("Adios reading time (sec): ", (t2 - t0))

    def len(self):
        if self.subset is not None:
            return len(self.subset)
        else:
            return self.ndata

    @tr.profile("get")
    def get(self, i):
        """
        Get data with a given index
        """
        idx = i
        if self.subset is not None:
            idx = self.subset[i]

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
                    if self.subset is not None:
                        start[vdim] = start[vdim] - self.subset_start[k][vdim]
                        assert count[vdim] <= self.subset_count[k][vdim]
                    slice_list = list()
                    for n0, n1 in zip(start, count):
                        slice_list.append(slice(n0, n0 + n1))
                    val = self.data[k][tuple(slice_list)]
                elif self.ddstore:
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
                    # log0("getitem out-of-memory:", self.label, k, idx)
                    val = self.f.read("%s/%s" % (self.label, k), start, count)

                v = torch.tensor(val)
                exec("data_object.%s = v" % (k))
            if self.enable_cache:
                self.cache[idx] = data_object

        self.update_data_object(data_object)
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
        if self.ddstore:
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

                with ad2.open(self.filename, "r", self.comm) as f:
                    f.__next__()
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
