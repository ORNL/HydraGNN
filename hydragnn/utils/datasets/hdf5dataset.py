"""HDF5-based storage for heterogeneous PyG data.

Each MPI rank writes its own shard file inside a directory:
    <basedir>/
        meta.h5              – total counts, rank offsets
        shard-0000.h5        – samples from rank 0
        shard-0001.h5        – samples from rank 1
        …

Each shard stores samples as variable-length byte datasets (one per label)
containing pickle-serialised HeteroData objects.  No homogeneous conversion.
"""

import os
import pickle

import h5py
import numpy as np
from mpi4py import MPI

from .abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.print import iterate_tqdm


# ──────────────────────────────────────────────────────────────────────
#  Writer
# ──────────────────────────────────────────────────────────────────────
class HDF5Writer:
    """Collect labelled sample lists, then flush to a single shard file."""

    def __init__(self, basedir, comm=MPI.COMM_WORLD):
        self.basedir = basedir
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self._labels = {}  # label -> list[data]

    def add(self, label, data):
        if label not in self._labels:
            self._labels[label] = []
        if isinstance(data, list):
            self._labels[label].extend(data)
        else:
            self._labels[label].append(data)

    def save(self):
        # Rank 0 creates the base directory
        if self.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
        self.comm.Barrier()

        # Each rank writes its own shard
        shard_path = os.path.join(self.basedir, f"shard-{self.rank:04d}.h5")
        vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
        with h5py.File(shard_path, "w") as fh:
            for label, samples in self._labels.items():
                blobs = []
                for i, sample in iterate_tqdm(
                    enumerate(samples),
                    2,
                    total=len(samples),
                    desc=f"HDF5 write {label}",
                ):
                    blobs.append(
                        np.frombuffer(pickle.dumps(sample, protocol=4), dtype=np.uint8)
                    )
                if blobs:
                    ds = fh.create_dataset(label, shape=(len(blobs),), dtype=vlen_dt)
                    for j, b in enumerate(blobs):
                        ds[j] = b
                else:
                    fh.create_dataset(label, shape=(0,), dtype=vlen_dt)

        self.comm.Barrier()

        # Rank 0 writes the metadata file
        if self.rank == 0:
            meta_path = os.path.join(self.basedir, "meta.h5")
            labels = sorted(self._labels.keys())
        else:
            labels = None
        labels = self.comm.bcast(labels, root=0)

        # Gather counts per label from all ranks
        per_label_counts = {}
        for label in labels:
            local_n = len(self._labels.get(label, []))
            counts = self.comm.gather(local_n, root=0)
            if self.rank == 0:
                per_label_counts[label] = counts

        if self.rank == 0:
            with h5py.File(meta_path, "w") as fh:
                fh.attrs["num_shards"] = self.size
                for label in labels:
                    grp = fh.create_group(label)
                    counts = np.array(per_label_counts[label], dtype=np.int64)
                    grp.create_dataset("counts", data=counts)
                    grp.attrs["total"] = int(counts.sum())

        self.comm.Barrier()


# ──────────────────────────────────────────────────────────────────────
#  Dataset (reader)
# ──────────────────────────────────────────────────────────────────────
class HDF5Dataset(AbstractBaseDataset):
    """Read back a split written by HDF5Writer."""

    def __init__(self, basedir, label, var_config=None):
        super().__init__()
        self.basedir = basedir
        self.label = label
        self.var_config = var_config

        meta_path = os.path.join(basedir, "meta.h5")
        with h5py.File(meta_path, "r") as fh:
            grp = fh[label]
            self.counts = grp["counts"][:]
            self.ntotal = int(grp.attrs["total"])

        # Build a mapping from global index -> (shard_rank, local_index)
        self._offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=self._offsets[1:])

        # Cache open file handles lazily
        self._handles = {}

    def len(self):
        return self.ntotal

    def get(self, idx):
        # Find which shard this global index belongs to
        shard = int(np.searchsorted(self._offsets[1:], idx, side="right"))
        local_idx = idx - int(self._offsets[shard])
        fh = self._open_shard(shard)
        blob = fh[self.label][local_idx]
        data = pickle.loads(blob.tobytes())
        return data

    def _open_shard(self, shard):
        if shard not in self._handles:
            path = os.path.join(self.basedir, f"shard-{shard:04d}.h5")
            self._handles[shard] = h5py.File(path, "r")
        return self._handles[shard]

    def __del__(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
