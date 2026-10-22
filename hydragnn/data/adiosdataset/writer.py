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
import time

import numpy as np
import torch
import torch_geometric.data
from mpi4py import MPI

from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.print.print_utils import log0

from ._backend import _comm_args, ad2


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
        # Test import of adios2. Let it fail with an ImportError if ADIOS2 is not installed
        import adios2

        self.filename = filename
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dataset = dict()
        self.attributes = dict()
        self.adios = ad2.ADIOS()
        self.io = self.adios.DeclareIO(self.filename)
        self.io.Parameters()["StatsLevel"] = "Off"

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
        self.writer = self.io.Open(
            self.filename, ad2.Mode.Write, *_comm_args(self.comm)
        )
        total_ns = 0

        # Look for the dataset_name in any one of the Data samples and add it as an ADIOS attribute
        dataset_name = self._get_dataset_name()
        if dataset_name is not None:
            if "dataset_name" not in self.attributes:
                self.attributes["dataset_name"] = dataset_name

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

                # Don't add dataset_name to 'keys'
                if "dataset_name" in keys:
                    keys.remove("dataset_name")

                self.io.DefineAttribute("%s/keys" % label, keys)
                keys = sorted(keys)
                self.comm.allgather(keys)

            for k in keys:
                if k == "dataset_name":
                    continue

                # if k == "smiles":
                #     self._write_smiles_strings(label)
                #     continue

                arr_list = list()
                for data in self.dataset[label]:
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
            if (
                isinstance(self.attributes[vname], np.ndarray)
                and not self.attributes[vname].flags["C_CONTIGUOUS"]
            ):
                self.attributes[vname] = np.ascontiguousarray(self.attributes[vname])
            self.io.DefineAttribute(vname, self.attributes[vname])

        self.writer.Close()
        t1 = time.time()
        log0("Adios saving time (sec): ", (t1 - t0))

    def _get_dataset_name(self):
        """
        Get dataset name from the first data object

        Returns
        -------
        str
            dataset name
        """
        if len(self.dataset) == 0:
            return None
        for label in self.dataset:
            for data in self.dataset[label]:
                keys = data.keys() if callable(data.keys) else data.keys
                if "dataset_name" in keys:
                    return data.dataset_name
        return None

    def _write_smiles_strings(self, label):
        """
        Write smiles data into the adios file
        This will write two global arrays, one for the smiles data and another for the string lengths
        e.g., assume 2 processes where P0 has "abcd", "ef", and P1 has "ghi", "j", then
        array 1: smiles_data:    ["abcdefghij"]
        array 2: smiles_lengths: [4,2,3,1]
        """

        # Collect smiles strings in a list
        _smiles_data = list()
        for data in self.dataset[label]:
            _smiles_data.append(data.smiles)

        # Convert to uint8 array
        local_smiles = [
            np.frombuffer(s.encode("utf-8"), dtype=np.uint8) for s in _smiles_data
        ]
        local_smiles_arr = np.concatenate(local_smiles)

        # 1. Calculate sizes and offsets for the global smiles array
        local_size = local_smiles_arr.size
        all_sizes = self.comm.allgather(local_size)
        offsets = sum(all_sizes[: self.rank])
        global_sizes = sum(all_sizes)

        # Write smiles data
        data_var = self.io.DefineVariable(
            f"{label}/smiles_data",
            local_smiles_arr,
            [global_sizes],
            [offsets],
            [local_size],
            ad2.ConstantDims,
        )
        self.writer.Put(data_var, local_smiles_arr, ad2.Mode.Sync)

        # 2. Calculate sizes and offsets for the global lengths array
        local_lengths = np.array([arr.size for arr in local_smiles])
        if any(x == 0 for x in local_lengths):
            print("something is 0")

        all_lengths = self.comm.allgather(local_lengths.size)
        offsets = sum(all_lengths[: self.rank])
        global_sizes = sum(all_lengths)

        # Write lengths data
        offsets_var = self.io.DefineVariable(
            f"{label}/smiles_lengths",
            local_lengths,
            [global_sizes],
            [offsets],
            [local_lengths.size],
            ad2.ConstantDims,
        )
        self.writer.Put(offsets_var, local_lengths, ad2.Mode.Sync)
