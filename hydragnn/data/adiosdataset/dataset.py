##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import os
import time
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch_geometric.data
from mpi4py import MPI

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.distributed import nsplit
from hydragnn.utils.print.print_utils import iterate_tqdm, log, log0

from ._backend import adios2_open, adios_dtype, dds, parse_shape


class AdiosDataset(AbstractBaseDataset):
    """Adios datasets class"""

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
        ddstore_store_per_sample=True,  ## True for per-sample saving. False for feature-first saving
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
            Option to preload all the datasets into a memory
        shmem: bool, optional
            Option to use shmem to share data between processes in the same node
        enable_cache: bool, optional
            Option to cache data object which was already read
        ddstore: bool, optional
            Option to use Distributed Data Store
        """
        # Test import of adios2. Let it fail with an ImportError if ADIOS2 is not installed
        import adios2

        self.filename = filename
        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        # Cache adios variables 'variable_count' and 'variable_offset' to speed up reading
        self._adios_file_metadata_cache = dict()

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
        self.use_ddstore = ddstore
        self.ddstore = None
        self.ddstore_store_per_sample = ddstore_store_per_sample
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        if self.use_ddstore:
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
        ddstore_time = 0.0
        t0 = time.time()
        with adios2_open(self.filename, "r", MPI.COMM_SELF) as f:
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

            # all processes should get the dataset name - a global attribute
            self.dataset_name = None
            if "dataset_name" in self.attrs:
                _val = f.read_attribute_string("dataset_name")
                if type(_val) == list and len(_val) > 0:
                    self.dataset_name = _val[0]
                else:
                    self.dataset_name = _val

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

                # dataset_name should not be a key, but if it is, it has already been read as an attr
                if k == "dataset_name":
                    continue

                # if k == "smiles":
                #     self._read_smiles_strings(label, f)
                #     continue

                # If adios variable min and max are available from file, use caching to get the count aod offset
                varmin = int(self.vars[f"{label}/{k}/variable_count"]["Min"])
                varmax = int(self.vars[f"{label}/{k}/variable_count"]["Max"])
                if varmin is not None and varmax is not None:
                    for vartype in ("count", "offset"):
                        if vartype == "count":
                            var = self.variable_count
                        else:
                            var = self.variable_offset

                        cache_search_key = f"{vartype}:{varmin}_{varmax}"
                        adios_var_name = f"{label}/{k}/variable_{vartype}"  # "variable_count" or "variable_offset"

                        var[k] = self._get_from_cache_or_file(
                            cache_search_key, f, adios_var_name
                        )
                        log0(
                            f"reading {adios_var_name} from cache or file",
                            time.time() - t3,
                        )

                # no caching available. read count and offset from file
                else:
                    log0(
                        "adios internal file cache not available. reading count and offset information from file"
                    )
                    self.variable_count[k] = self.read0(
                        f, f"{label}/{k}/variable_count"
                    )
                    log0(
                        "read and bcast:",
                        f"{label}/{k}/variable_count",
                        time.time() - t3,
                    )

                    self.variable_offset[k] = self.read0(
                        f, f"{label}/{k}/variable_offset"
                    )
                    log0(
                        "read and bcast:",
                        f"{label}/{k}/variable_offset",
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
                        ishape = parse_shape(shape)
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
                    vinfo = self.vars["%s/%s" % (label, k)]
                    ishape = parse_shape(vinfo["Shape"])
                    dtype = adios_dtype(vinfo["Type"])
                    if self.local_rank == 0:
                        nbytes = int(np.prod(ishape)) * np.dtype(dtype).itemsize
                        self.shm[k] = SharedMemory(create=True, size=max(nbytes, 1))
                        arr = np.ndarray(ishape, dtype=dtype, buffer=self.shm[k].buf)
                        arr[...] = f.read("%s/%s" % (label, k))
                        self.local_comm.bcast(self.shm[k].name, root=0)
                    else:
                        name = self.local_comm.bcast(None, root=0)
                        self.shm[k] = SharedMemory(name=name, create=False)
                        arr = np.ndarray(ishape, dtype=dtype, buffer=self.shm[k].buf)
                    self.data[k] = arr
                elif self.use_ddstore:
                    ## Calculate local portion
                    shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                    ishape = parse_shape(shape)
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

                    if not self.ddstore_store_per_sample:
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

            ## per-sample approach
            if self.use_ddstore and self.ddstore_store_per_sample:
                t4 = time.time()
                buf = BytesIO()
                rx = list(nsplit(list(range(self.ndata)), self.ddstore_comm_size))[
                    self.ddstore_comm_rank
                ]
                local_record_count = list()
                for idx in rx:
                    data_object = dict()
                    for k in self.keys:
                        shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                        ishape = parse_shape(shape)
                        start = [
                            0,
                        ] * len(ishape)
                        count = ishape
                        vdim = self.variable_dim[k]
                        start[vdim] = (
                            self.variable_offset[k][idx]
                            - self.variable_offset[k][rx[0]]
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

                        vname = "%s/%s" % (self.label, k)
                        vartype = self.vars["%s/%s" % (self.label, k)]["Type"]
                        dtype = adios_dtype(vartype)

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
                self.ddstore.add("record_array", arr)
                nbytes += arr.nbytes
                t5 = time.time()
                ddstore_time += t5 - t4

            t6 = time.time()
            log0("Overall time (sec): ", t6 - t1)
            log0("DDStore adding time (sec): ", ddstore_time)
            if self.use_ddstore:
                log0("DDStore total (GB):", nbytes / 1024 / 1024 / 1024)

        t7 = time.time()
        log0("Data loading time (sec): ", (t7 - t0))

        if not self.preload and not self.shmem:
            self.f = adios2_open(self.filename, "r", self.comm)
            self.f.__next__()

        ## FIXME: Using the same routine in SimplePickleDataset. We need to make as a common function

    def _read_smiles_strings(self, label, f):
        """
        Add smiles data to self.data['smiles']
        It will read from two global arrays - 'smiles_data' and 'smiles_lengths'
        e.g., assume we have the following:
        array 1: smiles_data:    ["abcdefghij"]
        array 2: smiles_lengths: [4,2,3,1]
        If we have 2 processes, then P0 will have ["abcd", "ef"], and
        P1 will have ["ghi", "j"]
        """
        self.smiles_strings = list()

        # First we will read the string lengths
        lengths_varname = f"{label}/smiles_lengths"
        global_size = int(self.vars[lengths_varname]["Shape"].split(",")[0])

        # Calculate individual process's portion
        local_size = global_size // self.comm_size
        local_size += self.rank < (global_size % self.comm_size)
        all_local_sizes = self.comm.allgather(local_size)
        local_offset = sum(all_local_sizes[: self.rank])

        # Read the string lengths. e.g. P0 will get [4,2], and P1 will get [3,1]
        local_string_lengths = f.read(lengths_varname, [local_offset], [local_size])

        # Now we will read the smiles strings as uint8 arrays
        data_varname = f"{label}/smiles_data"
        local_size = sum(local_string_lengths)
        all_local_sizes = self.comm.allgather(local_size)
        local_offset = sum(all_local_sizes[: self.rank])

        # Read the smiles strings. e.g. P0 will get "abcdef" and P1 will get "ghij" uint8 arrays
        local_data = f.read(data_varname, [local_offset], [local_size])

        # Convert uint8 arrays to strings
        start = 0
        for l in local_string_lengths:
            np_str_arr = local_data[start : start + l]
            start += l
            string_value = np_str_arr.tobytes().decode("utf-8")
            self.smiles_strings.append(string_value)

    def read0(self, f, vname):
        ## rank=0 read and bcast
        val = None
        if self.rank == 0:
            val = f.read(vname)
        val = self.comm.bcast(val, root=0)
        return val

    def read_attribute0(self, f, vname):
        if self.rank == 0:
            val = f.read_attribute(vname)
        else:
            val = None
        val = self.comm.bcast(val, root=0)
        return val

    def read_attribute_string0(self, f, vname):
        if self.rank == 0:
            val = f.read_attribute_string(vname)
        else:
            val = None
        val = self.comm.bcast(val, root=0)
        return val

    def setkeys(self, keys):
        for k in keys:
            assert k in self.keys, f"Error reading ADIOS file. {k} not in {self.keys}"
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
                ishape = parse_shape(shape)
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
        elif self.use_ddstore and self.ddstore_store_per_sample:
            data_object = torch_geometric.data.Data()
            dtype_list = list()
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = parse_shape(shape)
                start = [
                    0,
                ] * len(ishape)
                count = ishape
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][idx]
                count[vdim] = self.variable_count[k][idx]

                vname = "%s/%s" % (self.label, k)
                vartype = self.vars["%s/%s" % (self.label, k)]["Type"]
                dtype = adios_dtype(vartype)

                dtype_tuple = (k, dtype, count)
                dtype_list.append(dtype_tuple)

            dtype = np.dtype(dtype_list)
            val = np.zeros(dtype.itemsize, dtype=np.uint8)
            offset = 0 if idx == 0 else self.record_offset[idx - 1]
            self.ddstore.get("record_array", val, offset)
            val = val.view(dtype)[0]
            for k in val.dtype.names:
                v = torch.tensor(val[k])
                setattr(data_object, k, v)
        else:
            data_object = torch_geometric.data.Data()
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = parse_shape(shape)
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
                elif self.use_ddstore and not self.ddstore_store_per_sample:
                    vname = "%s/%s" % (self.label, k)
                    vartype = self.vars["%s/%s" % (self.label, k)]["Type"]
                    dtype = adios_dtype(vartype)

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

                if val.dtype == np.uint8:
                    ## Tensors do not support strings. We use strings as they are. No converting to tensors.
                    v = val.tobytes().decode("utf-8")
                else:
                    v = torch.tensor(val)
                setattr(data_object, k, v)
            if self.enable_cache:
                self.cache[idx] = data_object

        return data_object

    def _get_from_cache_or_file(self, cache_search_key, f, vname):
        """
        Search for the adios variable "variable_count" or "variable_offset" in the cache.
        If found, return the dictionary key, otherwise read the array from file
        """
        # root searches in cache. returns cached array if found, otherwise reads from file and return key string
        val = 0
        if self.rank == 0:
            val = f.read(vname)

            for k, v in self._adios_file_metadata_cache.items():
                if k == cache_search_key:
                    if np.array_equal(val, v):
                        val = cache_search_key
                        log0(f"found matching array for {vname} in cache as {k}")
                        break

        val = self.comm.bcast(val, root=0)

        # if val is the search key
        if type(val) == str:
            return self._adios_file_metadata_cache[val]
        else:
            self._adios_file_metadata_cache[cache_search_key] = val
            return val

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
        try:
            if self.use_ddstore:
                self.ddstore.free()
            if not self.preload and not self.shmem:
                self.f.close()
            self.unlink()
        except Exception:
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
                ishape = parse_shape(shape)
                start = [
                    0,
                ] * len(ishape)
                count = list(ishape)
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][i]
                count[vdim] = self.variable_count[k][i : i + dn].sum()

                with adios2_open(self.filename, "r", self.comm) as f:
                    f.__next__()
                    self._data[k] = f.read("%s/%s" % (self.label, k), start, count)

            filtered = filter(lambda x: x >= i and x < i + dn, self.preflight_list)
            for idx in iterate_tqdm(sorted(filtered), 2, desc="AdiosWriter populate"):
                data_object = torch_geometric.data.Data()
                for k in self.keys:
                    shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                    ishape = parse_shape(shape)
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
                    setattr(data_object, k, v)
                self.cache[idx] = data_object

        for k in self.keys:
            del self._data[k]

        self.preflight = False
